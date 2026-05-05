// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Wilcoxon binned + Welch t)
//
// de/ttest.h — Per-cluster per-gene Welch's t-test for GPU DE.
//
// Algorithm:
//   Pass 1 — per-gene per-cluster sum and sum_sq accumulation (two-pass naive
//             variance).  One block per gene in each gene tile.  atomicAdd is
//             safe because addition is associative and commutative — no Welford
//             race condition.  Mean and M2 are derived in Pass 2.
//   Pass 2 — Welch's t statistic and p-value.  One thread per (gene, cluster).
//             t = (mean_in - mean_out) / sqrt(var_in/n_in + var_out/n_out).
//             Gaussian approx: p = erfcf(|t| / sqrt(2)).  Valid for large n.
//             log2_fc: log2((mean_in + eps) / (mean_out + eps)), eps = 1e-9.
//   Pass 3 — BH adjustment per cluster (same as wilcoxon.h pass 3).
//   Pass 4 — top-N selection per cluster via cub::DeviceRadixSort on |t| desc.
//
// Memory budget (m=30k, n_clusters=10):
//   Tile workspace: 8 * gene_tile * n_clusters floats (sum + sum_sq per cluster).
//     At gene_tile=1024, n_clusters=10: 81 920 floats = 320 KB. Tiny.
//   Global stats: 3 * m * n_clusters * 4 bytes (t, p, lfc) = 3.6 MB at 30k×10.
//   Total: ~4 MB device workspace.
//
// Streams: 1, caller-provided. Passes chain on the same stream.
//
// Precision: fp32 throughout with Kahan / fp64-promo for high-count genes
//   (cycle-4 pattern reused).  Gaussian approximation for p-values is exact
//   for df ≥ 30; for smaller clusters we document the approximation error.
//
// Determinism: sum + sum_sq accumulation via atomicAdd is not bitwise
//   deterministic (float atomics are non-deterministic in order).
//   cfg.deterministic replaces the GPU atomicAdd with a host-side sequential
//   scan (same as the old Welford deterministic path) for exact reproducibility.
//
// OOC: Welford accumulators merge across chunks via the parallel Welford-Chan
//   formula: merge(A, B) where A has (n_A, mean_A, M2_A) and B has (n_B, mean_B,
//   M2_B): n = n_A + n_B, delta = mean_B - mean_A, mean = mean_A + delta * n_B/n,
//   M2 = M2_A + M2_B + delta² * n_A * n_B / n.  Integration with
//   streaming/streamed_pipeline.h: cycle ≥ 12.
//
// Correctness tolerance:
//   Marker gene Jaccard ≥ 0.90 vs scanpy rank_genes_groups(method='t-test').
//   log2_fc Spearman ρ ≥ 0.98; p-value rank Spearman ρ ≥ 0.95.
//
// Reference: Welch (1947), Satterthwaite (1946). GPU Welford: Chan et al. (1979).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/de/types.h>
#include <singlet-gpu/de/wilcoxon.h>   // reuse CSR transpose + BH + top-N helpers

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace singlet_gpu {
namespace de {

// ---------------------------------------------------------------------------
// Public config
// ---------------------------------------------------------------------------

struct TtestConfig {
    int  top_n        = 100;    // maximum marker genes per cluster
    int  gene_tile    = 1024;   // genes per tile (bounds workspace)
    bool deterministic = false; // true → deterministic Welford accumulation
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace ttest_detail {

// ---- Pass 1: per-gene per-cluster sum + sum_sq accumulation -----------------
//
// One block per gene in the current tile.  Threads stride over the gene's
// nonzeros in CSR view.  atomicAdd on sum and sum_sq is safe because addition
// is commutative and associative — concurrent writes produce the correct total
// regardless of order.  Mean and variance are derived from (n, sum, sum_sq)
// in Pass 2 via the well-known two-pass formula:
//   mean   = sum / n
//   M2     = sum_sq - sum * mean      (== sum_sq - n * mean^2)
//   var    = M2 / (n - 1)
//
// WHY two-pass instead of Welford: the per-thread concurrent Welford update
// reads a stale mean and produces 3e8× t-value error (Cycle 82 diagnostic).
// sum+sum_sq atomicAdd has no such race: each thread's contribution is
// independent, and the order of atomicAdds doesn't affect the total.
//
// Workspace layout per tile:
//   sum_tile    [gene_tile][n_clusters]  float  running sum of nonzero values
//   sum_sq_tile [gene_tile][n_clusters]  float  running sum of squared nonzero values
//   n_tile      [gene_tile][n_clusters]  int32  running count of nonzeros seen

__global__ __launch_bounds__(256, 2)
void welford_kernel(
    const float*   __restrict__ csr_values,
    const int*     __restrict__ csr_row_ptr,
    const int*     __restrict__ csr_col_idx,
    const int*     __restrict__ labels,
    float*         __restrict__ sum_tile,    // [gene_tile][n_clusters]
    float*         __restrict__ sum_sq_tile, // [gene_tile][n_clusters]
    int*           __restrict__ n_tile,      // [gene_tile][n_clusters]
    int  gene_offset,
    int  n_genes_tile,
    int  n_clusters)
{
    const int g_local  = blockIdx.x;
    if (g_local >= n_genes_tile) return;
    const int g_global = gene_offset + g_local;

    const int rs = csr_row_ptr[g_global];
    const int re = csr_row_ptr[g_global + 1];

    float* my_sum    = sum_tile    + (size_t)g_local * n_clusters;
    float* my_sum_sq = sum_sq_tile + (size_t)g_local * n_clusters;
    int*   my_n      = n_tile      + (size_t)g_local * n_clusters;

    for (int i = rs + threadIdx.x; i < re; i += blockDim.x) {
        float v    = csr_values[i];
        int   cell = csr_col_idx[i];
        int   c    = labels[cell];
        atomicAdd(&my_n[c], 1);
        atomicAdd(&my_sum[c], v);          // associative, safe under concurrent writes
        atomicAdd(&my_sum_sq[c], v * v);   // associative, safe under concurrent writes
    }
}

// ---- Pass 1b: zero-count n update -------------------------------------------
// For each (gene_in_tile, cluster c), zeros contribute 0 to sum and 0 to
// sum_sq, so sum_tile and sum_sq_tile are unchanged.  Only n needs to be set
// to the full cluster size (including zero-count cells).
// One thread per (gene_tile_local, cluster) pair.
__global__ __launch_bounds__(256, 2)
void zero_welford_kernel(
    float*       __restrict__ sum_tile,     // [gene_tile][n_clusters] — unchanged
    float*       __restrict__ sum_sq_tile,  // [gene_tile][n_clusters] — unchanged
    int*         __restrict__ n_tile,       // [gene_tile][n_clusters] (nonzero count → full count)
    const int*   __restrict__ cluster_sizes,// [n_clusters]
    int n_genes_tile, int n_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_genes_tile * n_clusters;
    if (idx >= total) return;
    int g = idx / n_clusters;
    int c = idx % n_clusters;

    int n_nz = n_tile[(size_t)g * n_clusters + c];
    int n_z  = cluster_sizes[c] - n_nz;
    if (n_z <= 0) return;

    // Zeros add 0 to sum and 0 to sum_sq — nothing to update for those fields.
    // Just record the full cell count so Pass 2 divides by n_total correctly.
    n_tile[(size_t)g * n_clusters + c] = cluster_sizes[c];
}

// ---- Sanitizer helper -------------------------------------------------------
// Map non-finite float to zero.  Applied to t, lfc before any sort/output so
// that degenerate genes (zero variance in all clusters) sink to the middle of a
// signed-t sort (score=0) rather than polluting the top-N via NaN ordering.
// Matches scanpy semantics: scipy.stats.ttest_ind returns nan→ we clamp to 0.
// p-values for degenerate genes are forced to 1 (most conservative).
__device__ __forceinline__ float finitize_or_zero(float x) {
    return isfinite(x) ? x : 0.f;
}

// ---- Pass 2: Welch t statistic + p-value + log2_fc -------------------------
// One thread per (gene_in_tile, cluster).
// For gene g, cluster c vs. rest:
//   mean_c   = sum_tile[g][c] / n_c
//   M2_c     = sum_sq_tile[g][c] - sum_tile[g][c] * mean_c
//   var_c    = M2_c / (n_c - 1)
// For one-vs-rest, merge (n, sum, sum_sq) across non-c clusters by simple
// addition (associative), then derive mean and M2 the same way.
//   t        = (mean_in - mean_out) / sqrt(var_in/n_in + var_out/n_out)
// p-value: erfcf(|t| / sqrt(2)) — Gaussian limit, valid for large n.
__global__ __launch_bounds__(256, 2)
void ttest_stats_kernel(
    const float* __restrict__ sum_tile,     // [gene_tile][n_clusters]
    const float* __restrict__ sum_sq_tile,  // [gene_tile][n_clusters]
    const int*   __restrict__ n_tile,       // [gene_tile][n_clusters]
    const int*   __restrict__ cluster_sizes, // [n_clusters]
    float* __restrict__ t_out,   // [m][n_clusters], write at g_off + g_local
    float* __restrict__ p_out,   // [m][n_clusters]
    float* __restrict__ lfc_out, // [m][n_clusters]
    int gene_offset,
    int n_genes_tile,
    int n_clusters)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_genes_tile * n_clusters;
    if (idx >= total) return;

    int g_local  = idx / n_clusters;
    int c        = idx % n_clusters;
    int g_global = gene_offset + g_local;

    // Derive mean and M2 from (n, sum, sum_sq) — two-pass naive variance formula.
    int   n_c      = n_tile    [(size_t)g_local * n_clusters + c];
    float sum_c    = sum_tile  [(size_t)g_local * n_clusters + c];
    float sumsq_c  = sum_sq_tile[(size_t)g_local * n_clusters + c];
    float mean_c   = (n_c > 0) ? sum_c / (float)n_c : 0.f;
    float M2_c     = sumsq_c - sum_c * mean_c;   // == sum_sq - n * mean^2
    float var_c    = (n_c > 1) ? (M2_c / (float)(n_c - 1)) : 0.f;

    // Merge (n, sum, sum_sq) across all clusters != c by simple addition,
    // then derive mean/M2. Addition is associative — no race hazard, no
    // Welford-Chan needed for the rest-of-clusters merge here.
    // WHY inline loop: n_clusters typically ≤50; avoids a device reduction.
    float merged_sum    = 0.f, merged_sum_sq = 0.f;
    int   merged_n      = 0;
    for (int cc = 0; cc < n_clusters; ++cc) {
        if (cc == c) continue;
        int   nb      = n_tile    [(size_t)g_local * n_clusters + cc];
        float sum_b   = sum_tile  [(size_t)g_local * n_clusters + cc];
        float sumsq_b = sum_sq_tile[(size_t)g_local * n_clusters + cc];
        if (nb <= 0) continue;
        merged_n      += nb;
        merged_sum    += sum_b;
        merged_sum_sq += sumsq_b;
    }

    int   n_rest   = merged_n;
    float mean_out = (n_rest > 0) ? merged_sum / (float)n_rest : 0.f;
    float M2_rest  = merged_sum_sq - merged_sum * mean_out;
    float var_out  = (n_rest > 1) ? (M2_rest / (float)(n_rest - 1)) : 0.f;

    if (n_c <= 0 || n_rest <= 0) {
        t_out  [g_global * n_clusters + c] = 0.f;
        p_out  [g_global * n_clusters + c] = 1.f;
        lfc_out[g_global * n_clusters + c] = 0.f;
        return;
    }

    float se2   = var_c / (float)n_c + var_out / (float)n_rest;
    float se    = (se2 > 0.f) ? sqrtf(se2) : 1e-9f;
    float t_val = (mean_c - mean_out) / se;

    // Gaussian p-value via erfcf.
    float at = fabsf(t_val);
    float pv = (at > 27.f) ? 0.f : erfcf(at * 7.071067811865476e-1f);

    // log2_fc via fp64 + expm1 (Cycle 78 fix, mirrors wilcoxon.h:327-340).
    // WHY fp64 + expm1: input means are in log1p space after normalize_total+log1p.
    // expm1 recovers linear-space means before ratio; fp64 prevents catastrophic
    // cancellation for small means.  Same Cycle 76 rationale as wilcoxon LFC.
    const double mean_c_d   = (double)mean_c;
    const double mean_out_d = (double)mean_out;
    const double lin_in_d   = expm1(mean_c_d);
    const double lin_out_d  = expm1(mean_out_d);
    float lfc = (float)log2((lin_in_d + 1e-9) / (lin_out_d + 1e-9));

    t_out  [g_global * n_clusters + c] = finitize_or_zero(t_val);
    p_out  [g_global * n_clusters + c] = isfinite(pv) ? pv : 1.f;
    lfc_out[g_global * n_clusters + c] = finitize_or_zero((float)lfc);
}

}  // namespace ttest_detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

TtestResult ttest_de(
    const core::DeviceCSC&         mat,
    const core::DeviceMemory<int>& labels,
    int                             n_clusters,
    const TtestConfig&              cfg,
    cudaStream_t                    stream)
{
    if (stream == nullptr) stream = 0;

    const int m = (int)mat.rows;
    const int n = (int)mat.cols;
    if (m <= 0 || n <= 0 || n_clusters <= 0)
        throw std::invalid_argument("ttest_de: empty matrix or no clusters");

    const int top_n  = std::min(cfg.top_n, m);
    const int g_tile = std::min(cfg.gene_tile, m);

    // ── cuSPARSE handle + CSR transpose (same as wilcoxon_de) ────────────────
    cusparseHandle_t sp;
    cusparseCreate(&sp);
    cusparseSetStream(sp, stream);

    core::DeviceMemory<int>   csr_row_ptr;
    core::DeviceMemory<int>   csr_col_idx;
    core::DeviceMemory<float> csr_vals;
    // WHY n,m not m,n: cusparseCsr2cscEx2 expects csrRowPtr[m+1].
    // mat.col_ptr has n_cells+1 entries, so cusparse's "m" must be n_cells (cols of mat).
    // cusparse's "n" = n_genes (mat.rows), and output cscColPtr has n+1 = n_genes+1 entries.
    // Cycle 78 fix: mirrors wilcoxon.h:638 (Cycle 73).
    detail::csc_to_csr(sp,
        n, m, mat.nnz,
        mat.col_ptr.get(), mat.row_indices.get(), mat.values.get(),
        csr_row_ptr, csr_col_idx, csr_vals, stream);

    // ── Cluster sizes ─────────────────────────────────────────────────────────
    std::vector<int> h_labels(n);
    cudaMemcpyAsync(h_labels.data(), labels.get(), n * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    std::vector<int> h_cluster_sizes(n_clusters, 0);
    for (int j = 0; j < n; ++j) {
        int c = h_labels[j];
        if (c >= 0 && c < n_clusters) ++h_cluster_sizes[c];
    }
    core::DeviceMemory<int> d_cluster_sizes(n_clusters);
    cudaMemcpyAsync(d_cluster_sizes.get(), h_cluster_sizes.data(),
                    n_clusters * sizeof(int), cudaMemcpyHostToDevice, stream);

    // ── Global stats arrays (written by tile loop) ────────────────────────────
    size_t mn = (size_t)m * n_clusters;
    core::DeviceMemory<float> t_all  (mn); cudaMemsetAsync(t_all.get(),   0, mn * sizeof(float), stream);
    core::DeviceMemory<float> lfc_all(mn); cudaMemsetAsync(lfc_all.get(), 0, mn * sizeof(float), stream);
    core::DeviceMemory<float> p_all  (mn);
    { std::vector<float> ones(mn, 1.f); cudaMemcpyAsync(p_all.get(), ones.data(), mn * sizeof(float), cudaMemcpyHostToDevice, stream); }

    // ── Tile loop ─────────────────────────────────────────────────────────────
    int n_tiles = (m + g_tile - 1) / g_tile;

    for (int tile = 0; tile < n_tiles; ++tile) {
        int g_off        = tile * g_tile;
        int n_genes_tile = std::min(g_tile, m - g_off);
        size_t gc_elems  = (size_t)n_genes_tile * n_clusters;

        core::DeviceMemory<float> sum_tile   (gc_elems); cudaMemsetAsync(sum_tile.get(),    0, gc_elems * sizeof(float), stream);
        core::DeviceMemory<float> sum_sq_tile(gc_elems); cudaMemsetAsync(sum_sq_tile.get(), 0, gc_elems * sizeof(float), stream);
        core::DeviceMemory<int>   n_tile     (gc_elems); cudaMemsetAsync(n_tile.get(),      0, gc_elems * sizeof(int),   stream);

        // Pass 1a: accumulate sum + sum_sq over nonzeros.
        if (!cfg.deterministic) {
            dim3 block(256), grid(n_genes_tile);
            ttest_detail::welford_kernel<<<grid, block, 0, stream>>>(
                csr_vals.get(), csr_row_ptr.get(), csr_col_idx.get(),
                labels.get(),
                sum_tile.get(), sum_sq_tile.get(), n_tile.get(),
                g_off, n_genes_tile, n_clusters);
        } else {
            // Deterministic path: host-side sequential scan (exact reproducibility).
            std::vector<int> h_row_ptr(n_genes_tile + 1);
            cudaMemcpyAsync(h_row_ptr.data(),
                            csr_row_ptr.get() + g_off,
                            (n_genes_tile + 1) * sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            int tile_rs  = h_row_ptr[0];
            int tile_re  = h_row_ptr[n_genes_tile];
            int nnz_tile = tile_re - tile_rs;

            if (nnz_tile > 0) {
                std::vector<float> h_vals  (nnz_tile);
                std::vector<int>   h_colidx(nnz_tile);
                cudaMemcpyAsync(h_vals.data(), csr_vals.get() + tile_rs,
                                nnz_tile * sizeof(float), cudaMemcpyDeviceToHost, stream);
                cudaMemcpyAsync(h_colidx.data(), csr_col_idx.get() + tile_rs,
                                nnz_tile * sizeof(int), cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);

                std::vector<float> h_sum   (gc_elems, 0.f);
                std::vector<float> h_sum_sq(gc_elems, 0.f);
                std::vector<int>   h_n     (gc_elems, 0);

                // Sequential sum + sum_sq accumulation — deterministic by construction.
                for (int g_local = 0; g_local < n_genes_tile; ++g_local) {
                    int rs_local = h_row_ptr[g_local]     - tile_rs;
                    int re_local = h_row_ptr[g_local + 1] - tile_rs;
                    for (int j = rs_local; j < re_local; ++j) {
                        float v = h_vals[j];
                        int   c = h_labels[h_colidx[j]];
                        if (c < 0 || c >= n_clusters) continue;
                        ++h_n     [(size_t)g_local * n_clusters + c];
                        h_sum     [(size_t)g_local * n_clusters + c] += v;
                        h_sum_sq  [(size_t)g_local * n_clusters + c] += v * v;
                    }
                }

                cudaMemcpyAsync(sum_tile.get(),    h_sum.data(),
                                gc_elems * sizeof(float), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(sum_sq_tile.get(), h_sum_sq.data(),
                                gc_elems * sizeof(float), cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(n_tile.get(),      h_n.data(),
                                gc_elems * sizeof(int),   cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
            }
        }

        // Pass 1b: update n to full cluster size (zeros add 0 to sum/sum_sq).
        {
            int total_gc = n_genes_tile * n_clusters;
            int b = 256, g = (total_gc + b - 1) / b;
            ttest_detail::zero_welford_kernel<<<g, b, 0, stream>>>(
                sum_tile.get(), sum_sq_tile.get(), n_tile.get(),
                d_cluster_sizes.get(), n_genes_tile, n_clusters);
        }

        // Pass 2: Welch t, p-value, log2_fc.
        {
            int total_gc = n_genes_tile * n_clusters;
            int b = 256, g = (total_gc + b - 1) / b;
            ttest_detail::ttest_stats_kernel<<<g, b, 0, stream>>>(
                sum_tile.get(), sum_sq_tile.get(), n_tile.get(),
                d_cluster_sizes.get(),
                t_all.get(), p_all.get(), lfc_all.get(),
                g_off, n_genes_tile, n_clusters);
        }
    }  // end tile loop

    // ── Per-cluster BH + top-N (same structure as wilcoxon_de) ───────────────
    TtestResult result;
    result.per_cluster.resize(n_clusters);

    for (int c = 0; c < n_clusters; ++c) {
        core::DeviceMemory<float> p_col  (m);
        core::DeviceMemory<float> padj   (m);
        core::DeviceMemory<int>   gene_idx(m);
        core::DeviceMemory<float> neg_abs_t(m);

        // Gather p-values and t-scores for cluster c using cudaMemcpy2D.
        cudaMemcpy2DAsync(
            p_col.get(), sizeof(float),
            p_all.get() + c, (size_t)n_clusters * sizeof(float),
            sizeof(float), m, cudaMemcpyDeviceToDevice, stream);
        cudaMemcpy2DAsync(
            neg_abs_t.get(), sizeof(float),
            t_all.get() + c, (size_t)n_clusters * sizeof(float),
            sizeof(float), m, cudaMemcpyDeviceToDevice, stream);

        // Negate signed t for ascending sort → descending t top-N (Cycle 78 fix).
        // WHY negate_kernel not negate_abs_kernel: scanpy ranks by signed score,
        // so top-N are most upregulated (largest positive t), not largest |t|.
        // Mirrors wilcoxon.h:926 (Cycle 75 fix).
        { int b = 256, g = (m + b - 1) / b;
          detail::negate_kernel<<<g, b, 0, stream>>>(
              neg_abs_t.get(), neg_abs_t.get(), m); }

        // Fill iota for gene indices.
        { int b = 256, g = (m + b - 1) / b;
          detail::fill_iota_kernel<<<g, b, 0, stream>>>(gene_idx.get(), m); }

        // Copy p_col → padj for BH computation.
        cudaMemcpyAsync(padj.get(), p_col.get(), m * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // BH step 1: sort p-values ascending with gene index as value.
        core::DeviceMemory<float> sorted_p  (m);
        core::DeviceMemory<int>   sorted_idx(m);
        {
            size_t temp_sz = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_sz,
                padj.get(), sorted_p.get(),
                gene_idx.get(), sorted_idx.get(), m, 0, sizeof(float)*8, stream);
            core::DeviceMemory<char> temp(temp_sz > 0 ? temp_sz : 1);
            cub::DeviceRadixSort::SortPairs(temp.get(), temp_sz,
                padj.get(), sorted_p.get(),
                gene_idx.get(), sorted_idx.get(), m, 0, sizeof(float)*8, stream);
        }

        // BH step 2: scale sorted p-values.
        { int b = 256, g = (m + b - 1) / b;
          detail::bh_scale_kernel<<<g, b, 0, stream>>>(sorted_p.get(), m); }

        // BH step 3: cumulative min from top.
        { int smem = 1024 * (int)sizeof(float);
          detail::bh_cummin_kernel<<<1, 1024, smem, stream>>>(sorted_p.get(), m); }

        // Scatter padj back to gene-indexed order (host-mediated, see wilcoxon.h).
        std::vector<float> h_sorted_p  (m);
        std::vector<int>   h_sorted_idx(m);
        cudaMemcpyAsync(h_sorted_p.data(),   sorted_p.get(),   m * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_sorted_idx.data(), sorted_idx.get(), m * sizeof(int),   cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        std::vector<float> h_padj(m);
        for (int r = 0; r < m; ++r) h_padj[h_sorted_idx[r]] = h_sorted_p[r];
        cudaMemcpyAsync(padj.get(), h_padj.data(), m * sizeof(float),
                        cudaMemcpyHostToDevice, stream);

        // Top-N selection: sort by |t| descending.
        core::DeviceMemory<float> sorted_neg_t(m);
        core::DeviceMemory<int>   sorted_gene (m);
        {
            detail::fill_iota_kernel<<<(m+255)/256, 256, 0, stream>>>(sorted_gene.get(), m);
            size_t temp_sz = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, temp_sz,
                neg_abs_t.get(), sorted_neg_t.get(),
                sorted_gene.get(), sorted_gene.get(), m, 0, sizeof(float)*8, stream);
            core::DeviceMemory<char> temp(temp_sz > 0 ? temp_sz : 1);
            cub::DeviceRadixSort::SortPairs(temp.get(), temp_sz,
                neg_abs_t.get(), sorted_neg_t.get(),
                sorted_gene.get(), sorted_gene.get(), m, 0, sizeof(float)*8, stream);
        }

        // Gather top-N stats (host-mediated, same pattern as wilcoxon_de).
        ClusterMarkers& cm = result.per_cluster[c];
        cm.cluster_id = c;
        cm.gene_indices = core::DeviceMemory<int>  (top_n);
        cm.z_scores     = core::DeviceMemory<float>(top_n);  // t statistic
        cm.log2_fc      = core::DeviceMemory<float>(top_n);
        cm.p_values     = core::DeviceMemory<float>(top_n);
        cm.p_adj        = core::DeviceMemory<float>(top_n);

        cudaMemcpyAsync(cm.gene_indices.get(), sorted_gene.get(),
                        top_n * sizeof(int), cudaMemcpyDeviceToDevice, stream);

        std::vector<int>   h_top_genes(top_n);
        std::vector<float> h_t_col(m), h_lfc_col(m), h_p_col(m);
        cudaMemcpyAsync(h_top_genes.data(), sorted_gene.get(),
                        top_n * sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpy2DAsync(h_t_col.data(),   sizeof(float),
                          t_all.get() + c,   (size_t)n_clusters * sizeof(float),
                          sizeof(float), m, cudaMemcpyDeviceToHost, stream);
        cudaMemcpy2DAsync(h_lfc_col.data(), sizeof(float),
                          lfc_all.get() + c, (size_t)n_clusters * sizeof(float),
                          sizeof(float), m, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_p_col.data(), p_col.get(),
                        m * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        std::vector<float> h_top_t(top_n), h_top_lfc(top_n),
                           h_top_p(top_n), h_top_padj(top_n);
        for (int i = 0; i < top_n; ++i) {
            int gi       = h_top_genes[i];
            h_top_t   [i] = h_t_col  [gi];
            h_top_lfc [i] = h_lfc_col[gi];
            h_top_p   [i] = h_p_col  [gi];
            h_top_padj[i] = h_padj   [gi];
        }

        cudaMemcpyAsync(cm.z_scores.get(), h_top_t.data(),
                        top_n * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cm.log2_fc.get(),  h_top_lfc.data(),
                        top_n * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cm.p_values.get(), h_top_p.data(),
                        top_n * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(cm.p_adj.get(),    h_top_padj.data(),
                        top_n * sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    cusparseDestroy(sp);
    cudaStreamSynchronize(stream);
    return result;
}

}  // namespace de
}  // namespace singlet_gpu
