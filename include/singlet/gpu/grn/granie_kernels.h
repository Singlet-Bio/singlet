// SPDX-License-Identifier: MIT
// singlet/gpu/grn/granie_kernels.h
//
// CUDA kernels and kernel-launch helpers for granie.h.
// Split out of granie.h (multi-concern header cleanup): this file holds the
// __global__/__device__ kernels and detail-namespace helpers; granie.h keeps
// the public API and host orchestration and #includes this header.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand_kernel.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <string>

namespace singlet::gpu {
namespace grn {

// ─── Internal detail kernels ─────────────────────────────────────────────────

namespace detail {

static constexpr float PEARSON_EPS = 1e-8f;

// ─── Kernel 1: Peak mean + variance (Welford, one warp per peak) ─────────────
//
// WHY one warp per peak: peaks CSC is column-major (columns = peaks). Each peak
// is a column; the warp streams through its non-zero cells accumulating Welford
// online stats. For dense peaks (most cells non-zero at scale) the warp stays
// fully coalesced on the values array.
//
// peaks_col_ptr: size n_peaks + 1; peaks_row_idx / peaks_vals: CSC of
// n_cells × n_peaks (rows=cells, cols=peaks). The CSC layout means col = peak.
__global__ void __launch_bounds__(32, 16)
peak_stats_kernel(const int*   __restrict__ col_ptr,
                  const int*   __restrict__ row_idx,
                  const float* __restrict__ vals,
                  float* __restrict__       peak_mean,
                  float* __restrict__       peak_sd,
                  int                       n_peaks,
                  int                       n_cells)
{
    int peak = blockIdx.x;
    if (peak >= n_peaks) return;
    int lane = threadIdx.x;  // warp: 32 threads

    float wm = 0.0f, wm2 = 0.0f;
    int   wn = 0;
    int   p0 = col_ptr[peak], p1 = col_ptr[peak + 1];
    // Welford over non-zero values (sparse: missing cells contribute 0).
    for (int i = p0 + lane; i < p1; i += 32) {
        float v = vals[i];
        ++wn;
        float delta = v - wm;
        wm  += delta / wn;
        wm2 += delta * (v - wm);
    }
    // Reduce across warp.
    for (int offset = 16; offset >= 1; offset >>= 1) {
        wm  += __shfl_down_sync(0xffffffff, wm,  offset);
        wm2 += __shfl_down_sync(0xffffffff, wm2, offset);
        wn  += __shfl_down_sync(0xffffffff, wn,  offset);
    }
    if (lane == 0) {
        // Adjust mean for all n_cells (zeros not in CSC contribute to mean).
        // Let mu_nz = wm / max(wn,1), nnz = p1-p0.
        // Full-sample mean = mu_nz * nnz / n_cells.
        int   nnz_peak = p1 - p0;
        float mu_nz    = (wn > 0) ? (wm) : 0.0f;
        float mu_full  = mu_nz * (float)nnz_peak / (float)n_cells;
        // Variance needs between-group correction (sparse-aware Welford).
        // Approximation for binary/low-count: use Var(X) = E[X^2] - E[X]^2.
        // For sparse: E[X^2] = sum(v^2)/n_cells, E[X] = mu_full.
        // wm2 = sum_i(v_i - wm_nz)^2 for non-zero i → E[X^2] via shift.
        float sum_sq_nz = wm2 + (float)nnz_peak * mu_nz * mu_nz;
        float ex2_full  = sum_sq_nz / (float)n_cells;
        float var_full  = ex2_full - mu_full * mu_full;
        peak_mean[peak] = mu_full;
        peak_sd[peak]   = sqrtf(fmaxf(var_full, 0.0f));
    }
}

// ─── Kernel 2: Gene mean + SD (one warp per gene) ────────────────────────────
//
// gex CSC: n_cells × n_genes (rows=cells, cols=genes). Mirrors peak_stats.
__global__ void __launch_bounds__(32, 16)
gene_stats_kernel(const int*   __restrict__ col_ptr,
                  const int*   __restrict__ row_idx,
                  const float* __restrict__ vals,
                  float* __restrict__       gene_mean,
                  float* __restrict__       gene_sd,
                  int                       n_genes,
                  int                       n_cells)
{
    int gene = blockIdx.x;
    if (gene >= n_genes) return;
    int lane = threadIdx.x;

    float wm = 0.0f, wm2 = 0.0f;
    int   wn = 0;
    int   p0 = col_ptr[gene], p1 = col_ptr[gene + 1];
    for (int i = p0 + lane; i < p1; i += 32) {
        float v = vals[i];
        ++wn;
        float delta = v - wm;
        wm  += delta / wn;
        wm2 += delta * (v - wm);
    }
    for (int offset = 16; offset >= 1; offset >>= 1) {
        wm  += __shfl_down_sync(0xffffffff, wm,  offset);
        wm2 += __shfl_down_sync(0xffffffff, wm2, offset);
        wn  += __shfl_down_sync(0xffffffff, wn,  offset);
    }
    if (lane == 0) {
        int   nnz_gene = p1 - p0;
        float mu_nz    = (wn > 0) ? wm : 0.0f;
        float mu_full  = mu_nz * (float)nnz_gene / (float)n_cells;
        float sum_sq_nz = wm2 + (float)nnz_gene * mu_nz * mu_nz;
        float ex2_full  = sum_sq_nz / (float)n_cells;
        float var_full  = ex2_full - mu_full * mu_full;
        gene_mean[gene] = mu_full;
        gene_sd[gene]   = sqrtf(fmaxf(var_full, 0.0f));
    }
}

// ─── Kernel 3: Peak–gene Pearson correlation (1 block per pair) ──────────────
//
// n_pairs blocks. Each block handles one (peak, gene) pair.
// Block size 256 threads. Threads cooperate via shared memory Welford reduction.
// Uses two-pass: first compute means (from peak_mean / gene_mean precomputed),
// then compute covariance sum from sparse CSC data.
//
// Strategy: for each pair (peak p, gene g):
//   cov  = (Σ_i peak[i]*gene[i]) / n_cells - mean_p * mean_g
//   r    = cov / (sd_p * sd_g + eps)
//
// We compute Σ_i peak[i]*gene[i] over the intersection of non-zero cells.
// Peak CSC: col p → non-zero cell rows. Gene CSC: col g → non-zero cell rows.
// Merge-scan over sorted row_indices to compute dot product in O(nnz_p + nnz_g).
//
// For large nnz (dense peaks), block threads divide the work.
// For the intersection: use a sorted-merge pass — both CSC row indices are sorted.
// Each thread walks a stride; partial sums accumulated in shared memory.
//
// WHY 1-block-per-pair: avoids atomic conflicts across pairs; register/shared
// pressure is bounded per pair regardless of matrix density.
__global__ void __launch_bounds__(256, 4)
peak_gene_pearson_kernel(
    // Peaks CSC (n_cells × n_peaks)
    const int*   __restrict__ peak_col_ptr,
    const int*   __restrict__ peak_row_idx,
    const float* __restrict__ peak_vals,
    // Genes CSC (n_cells × n_genes)
    const int*   __restrict__ gene_col_ptr,
    const int*   __restrict__ gene_row_idx,
    const float* __restrict__ gene_vals,
    // Pre-computed stats
    const float* __restrict__ peak_mean,
    const float* __restrict__ gene_mean,
    const float* __restrict__ peak_sd,
    const float* __restrict__ gene_sd,
    // Pair list
    const int2*  __restrict__ pairs,    // pairs[i].x = peak_idx, pairs[i].y = gene_idx
    // Output
    float*       __restrict__ r_out,
    int                       n_pairs,
    int                       n_cells)
{
    int pair_id = blockIdx.x;
    if (pair_id >= n_pairs) return;

    int p_idx = pairs[pair_id].x;
    int g_idx = pairs[pair_id].y;

    int p0p = peak_col_ptr[p_idx];
    int p1p = peak_col_ptr[p_idx + 1];
    int p0g = gene_col_ptr[g_idx];
    int p1g = gene_col_ptr[g_idx + 1];

    int nnz_p = p1p - p0p;
    int nnz_g = p1g - p0g;

    // Dot product via sorted merge of two CSC row-index lists.
    // Each thread takes a stride of blockDim.x across the peak's non-zero cells,
    // binary-searches for the corresponding gene row, accumulates if found.
    // This is O(nnz_p * log(nnz_g)) per block — acceptable for sparse inputs.
    // For dense peaks (nnz_p close to n_cells), use the gene's list as the
    // outer loop (swap if nnz_p > nnz_g) to keep log factor small.
    // The mean product term (mean_p * mean_g * n_cells) accounts for the zeros.

    // Shared memory reduction buffer.
    __shared__ float shm[256];
    float local_dot = 0.0f;

    if (nnz_p <= nnz_g) {
        // Iterate over non-zero cells of peak, binary search in gene.
        for (int i = p0p + (int)threadIdx.x; i < p1p; i += 256) {
            int cell = peak_row_idx[i];
            // Binary search cell in gene's row_indices[p0g..p1g).
            int lo = p0g, hi = p1g - 1;
            float gv = 0.0f;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int r   = gene_row_idx[mid];
                if (r == cell) { gv = gene_vals[mid]; break; }
                if (r  < cell)  lo = mid + 1;
                else            hi = mid - 1;
            }
            local_dot += peak_vals[i] * gv;
        }
    } else {
        // Iterate over gene's non-zeros, binary search in peak.
        for (int i = p0g + (int)threadIdx.x; i < p1g; i += 256) {
            int cell = gene_row_idx[i];
            int lo = p0p, hi = p1p - 1;
            float pv = 0.0f;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int r   = peak_row_idx[mid];
                if (r == cell) { pv = peak_vals[mid]; break; }
                if (r  < cell)  lo = mid + 1;
                else            hi = mid - 1;
            }
            local_dot += gene_vals[i] * pv;
        }
    }

    shm[threadIdx.x] = local_dot;
    __syncthreads();

    // Block reduction.
    for (int s = 128; s >= 1; s >>= 1) {
        if ((int)threadIdx.x < s) shm[threadIdx.x] += shm[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // cov = dot/n_cells - mean_p * mean_g
        float cov = shm[0] / (float)n_cells - peak_mean[p_idx] * gene_mean[g_idx];
        float denom = peak_sd[p_idx] * gene_sd[g_idx] + PEARSON_EPS;
        r_out[pair_id] = cov / denom;
    }
}

// ─── Kernel 4: TF motif count (one thread per TF) ────────────────────────────
//
// Counts the number of peaks with a motif hit per TF from the CSR row_ptr.
// tf_motif_row_ptr[t+1] - tf_motif_row_ptr[t] = nnz in row t = motif count for TF t.
__global__ void __launch_bounds__(256, 8)
tf_motif_count_kernel(const int* __restrict__ row_ptr,
                      float*     __restrict__ tf_motif_count_inv,  // 1/count or 0
                      int                     n_tfs)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tfs) return;
    int cnt = row_ptr[t + 1] - row_ptr[t];
    tf_motif_count_inv[t] = (cnt > 0) ? (1.0f / (float)cnt) : 0.0f;
}

// ─── Kernel 5: Row-normalize TF activity by motif count ──────────────────────
//
// tf_activity: [n_tfs × n_cells], row-major. Each row t divided by count_inv[t].
// This normalisation turns the sum-over-peaks into a mean-over-peaks.
__global__ void __launch_bounds__(256, 4)
tf_activity_normalize_kernel(float*       __restrict__ tf_activity,
                              const float* __restrict__ tf_count_inv,
                              int                       n_tfs,
                              int                       n_cells)
{
    int t    = blockIdx.y;
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_tfs || cell >= n_cells) return;
    tf_activity[t * n_cells + cell] *= tf_count_inv[t];
}

// ─── Kernel 6: TF mean + SD for Pearson (one warp per TF) ───────────────────
//
// tf_activity_row: pointer to row t in the [n_tfs × n_cells] dense activity matrix.
// Computes mean and SD across n_cells for each TF.
// WHY separate from normalization: computing Pearson requires centering; doing it
// in a separate pass avoids any ordering dependency with the normalize kernel.
__global__ void __launch_bounds__(32, 32)
tf_mean_sd_kernel(const float* __restrict__ tf_activity,  // n_tfs × n_cells row-major
                  float*       __restrict__ tf_mean,
                  float*       __restrict__ tf_sd,
                  int                       n_tfs,
                  int                       n_cells)
{
    int t    = blockIdx.x;
    int lane = threadIdx.x;
    if (t >= n_tfs) return;

    const float* row = tf_activity + (size_t)t * n_cells;
    float sum = 0.0f, sum2 = 0.0f;
    for (int c = lane; c < n_cells; c += 32) {
        float v = row[c];
        sum  += v;
        sum2 += v * v;
    }
    for (int offset = 16; offset >= 1; offset >>= 1) {
        sum  += __shfl_down_sync(0xffffffff, sum,  offset);
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
    }
    if (lane == 0) {
        float mu  = sum  / (float)n_cells;
        float var = sum2 / (float)n_cells - mu * mu;
        tf_mean[t] = mu;
        tf_sd[t]   = sqrtf(fmaxf(var, 0.0f));
    }
}

// ─── Kernel 7: Chunk TF–target Pearson correlation ──────────────────────────
//
// Processes chunk_size TFs and all n_genes targets in one GEMM-style block launch.
// For each (TF t, gene g):
//   r(t,g) = (Σ_c TF_act[t,c] * gex[c,g]) / n_cells - mean_tf[t]*mean_gene[g]
//            / (sd_tf[t] * sd_gene[g] + eps)
// Σ_c TF_act[t,c] * gex[c,g] = (TF_act_chunk × GEX_dense^T)[t,g]
//   — this is a GEMM: chunk × n_cells · n_cells × n_genes → chunk × n_genes.
//   We call cuBLAS cublasSgemm for this; the rest of this kernel handles the
//   normalization of the GEMM result into Pearson r.
//
// This kernel only finalises the Pearson formula AFTER cublasSgemm writes the
// raw dot-products into score_chunk.
__global__ void __launch_bounds__(256, 4)
finalize_tf_target_pearson_kernel(
    float*       __restrict__ score_chunk,  // [chunk_size × n_genes] from GEMM, in-place
    const float* __restrict__ tf_mean,      // tf_mean[t] for t in chunk
    const float* __restrict__ tf_sd,        // tf_sd[t]  for t in chunk
    const float* __restrict__ gene_mean,
    const float* __restrict__ gene_sd,
    int                       chunk_size,
    int                       n_genes,
    int                       n_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_size * n_genes) return;
    int t = idx / n_genes;
    int g = idx % n_genes;
    float raw_dot = score_chunk[idx];
    float cov   = raw_dot / (float)n_cells - tf_mean[t] * gene_mean[g];
    float denom = tf_sd[t] * gene_sd[g] + PEARSON_EPS;
    score_chunk[idx] = cov / denom;
}

// ─── Kernel 8: Compute peak–gene mean |r| per TF–gene pair ──────────────────
//
// For each (TF t_chunk, gene g):
//   mean_peak_gene_r[t_off+t, g] = mean(|r_peak_gene[pairs where peak has motif t]|)
//
// This requires iterating over tf_motif_in_peak CSR row t to get peak list,
// then for each such peak, iterating over peak_gene_pairs to find pairs involving
// that peak, and averaging |r|.
//
// Due to the cross-product nature, we launch 1 block per (t, g) chunk pair.
// For chunk_size=100 and n_genes=20k: 2M blocks — this is fine on A100.
// Each block handles one (t_in_chunk, g) pair.
//
// To efficiently find which pair indices involve a given peak, we need a
// peak → pair index lookup. This is provided as peak_start[peak] and peak_len[peak]
// (precomputed from sorted pairs array).
__global__ void __launch_bounds__(64, 8)
tf_target_combined_score_kernel(
    float*       __restrict__ score_chunk,         // [chunk × n_genes] Pearson r; overwritten with combined
    const float* __restrict__ r_peak_gene,         // [n_pairs] peak–gene r values (filtered: |r|>threshold)
    const int*   __restrict__ pair_peak_idx,       // pair_peak_idx[i] = peak index for pair i
    const int*   __restrict__ peak_to_pair_start,  // sorted by peak: start index in pairs array
    const int*   __restrict__ peak_to_pair_len,    // number of pairs for each peak
    const int*   __restrict__ tf_motif_col_idx,    // CSR col indices of tf_motif_in_peak
    const int*   __restrict__ tf_motif_row_ptr,    // CSR row ptr of tf_motif_in_peak
    const int*   __restrict__ pair_gene_idx,       // pair_gene_idx[i] = gene index for pair i
    int                       tf_offset,           // first TF index in current chunk
    int                       chunk_size,
    int                       n_genes,
    int                       n_pairs)
{
    int t_local = blockIdx.x;
    int g       = blockIdx.y * blockDim.x + threadIdx.x;
    if (t_local >= chunk_size || g >= n_genes) return;

    int t = tf_offset + t_local;
    int row_start = tf_motif_row_ptr[t];
    int row_end   = tf_motif_row_ptr[t + 1];
    int n_motif_peaks = row_end - row_start;

    float sum_abs_r = 0.0f;
    int   cnt       = 0;

    // Iterate over peaks where TF t has motif binding.
    for (int mi = row_start; mi < row_end; ++mi) {
        int peak = tf_motif_col_idx[mi];
        int ps   = peak_to_pair_start[peak];
        int pl   = peak_to_pair_len[peak];
        // Find pair (peak, g) in the pair sub-range [ps, ps+pl).
        // Pairs are sorted by (peak, gene) so binary search.
        int lo = ps, hi = ps + pl - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            int pg  = pair_gene_idx[mid];
            if (pg == g) {
                sum_abs_r += fabsf(r_peak_gene[mid]);
                ++cnt;
                break;
            }
            if (pg < g) lo = mid + 1;
            else        hi = mid - 1;
        }
    }
    (void)n_motif_peaks;

    // Combined score = Pearson(TF, gene) × mean(|r_peak_gene|) for binding peaks.
    float tf_gene_r = score_chunk[t_local * n_genes + g];
    float mean_rpg  = (cnt > 0) ? (sum_abs_r / (float)cnt) : 0.0f;
    score_chunk[t_local * n_genes + g] = tf_gene_r * mean_rpg;
}

// ─── Kernel 9: Convert combined_score to naive z-score p-value ──────────────
//
// For corr-based statistics under H0 (no association), t = r * sqrt((n-2)/(1-r^2))
// follows t(n-2). We approximate with a Gaussian z-score for large n:
//   z = r * sqrt(n_cells - 1)   [Fisher z-approx]
//   p = 2 * Phi(-|z|)
// Implemented as: p = erfcf(|z| / sqrtf(2)) — fast device erfc.
__global__ void __launch_bounds__(256, 4)
score_to_pvalue_kernel(const float* __restrict__ scores,
                       float*       __restrict__ pvalues,
                       int                       n,
                       float                     sqrt_n_minus1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = fabsf(scores[i]) * sqrt_n_minus1;
    pvalues[i] = erfcf(z * 0.70710678f);  // 1/sqrt(2)
}

// ─── Kernel 10: Build per-edge (tf_idx, gene_idx) arrays for Leiden ──────────
//
// Used only when cfg.run_leiden=true. Extracts (tf, gene) integer pairs from
// the device-side accepted edge indices to feed into Leiden's KnnResult format.
// One thread per accepted edge.
__global__ void __launch_bounds__(256, 4)
extract_edge_pairs_kernel(const int*   __restrict__ accepted_indices,
                          const int*   __restrict__ chunk_offsets,   // flat idx → (tf, gene)
                          int*         __restrict__  src,
                          int*         __restrict__  dst,
                          int                        n_accepted,
                          int                        n_genes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_accepted) return;
    int flat = accepted_indices[i];
    src[i] = flat / n_genes;  // tf index
    dst[i] = flat % n_genes;  // gene index
}

// ─── Helper: build peak→pair lookup from sorted pair array ──────────────────
//
// Runs host-side (called once at function entry after uploading pairs).
// Computes peak_to_pair_start[peak] and peak_to_pair_len[peak] from the
// sorted pairs array using std::lower_bound.
inline void build_peak_to_pair_lookup(
    const std::vector<int2>&  pairs_sorted,      // sorted by (peak, gene)
    int                       n_peaks,
    std::vector<int>&         peak_to_pair_start,
    std::vector<int>&         peak_to_pair_len)
{
    peak_to_pair_start.assign(n_peaks, 0);
    peak_to_pair_len.assign(n_peaks, 0);
    int n_pairs = (int)pairs_sorted.size();
    int cur_peak = -1;
    for (int i = 0; i < n_pairs; ++i) {
        int pk = pairs_sorted[i].x;
        if (pk != cur_peak) {
            if (cur_peak >= 0) {
                peak_to_pair_len[cur_peak] = i - peak_to_pair_start[cur_peak];
            }
            cur_peak = pk;
            peak_to_pair_start[pk] = i;
        }
    }
    if (cur_peak >= 0) {
        peak_to_pair_len[cur_peak] = n_pairs - peak_to_pair_start[cur_peak];
    }
}

// ─── Namespace-scope __global__ kernels (moved out of local struct bodies) ───
// CUDA does not allow __global__ as a member function of a local struct.
// Each kernel is defined here in namespace detail and called from the
// run_granie() function body below by name.

__global__ void split_pairs_kernel(const int2* __restrict__ pairs,
                                   int* __restrict__ pk,
                                   int* __restrict__ gn,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { pk[i] = pairs[i].x; gn[i] = pairs[i].y; }
}

__global__ void iota_kernel(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

__global__ void bh_raw_kernel(const float* __restrict__ sorted_p,
                               float*       __restrict__ bh_raw,
                               int n) {
    int rank = blockIdx.x * blockDim.x + threadIdx.x;
    if (rank >= n) return;
    bh_raw[rank] = sorted_p[rank] * (float)n / (float)(rank + 1);
}

__global__ void reverse_kernel(const float* __restrict__ in,
                                float*       __restrict__ out,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[n - 1 - i];
}

__global__ void scatter_kernel(const float* __restrict__ fdr_sorted,
                                const int*   __restrict__ orig_idx,
                                float*       __restrict__ fdr_orig,
                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fdr_orig[orig_idx[i]] = fdr_sorted[i];
}

__global__ void mask_kernel(const float* __restrict__ r_vals,
                             const float* __restrict__ fdr_vals,
                             int*         __restrict__ mask,
                             int n, float min_r, float max_fdr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mask[i] = (fabsf(r_vals[i]) > min_r && fdr_vals[i] < max_fdr) ? 1 : 0;
}

__global__ void scatter_gex_kernel(const int*   __restrict__ col_ptr,
                                    const int*   __restrict__ row_idx,
                                    const float* __restrict__ vals,
                                    float*       __restrict__ dense,
                                    int n_genes) {
    int g = blockIdx.x;
    int lane = threadIdx.x;
    int p0 = col_ptr[g], p1 = col_ptr[g + 1];
    for (int i = p0 + lane; i < p1; i += 32) {
        int cell = row_idx[i];
        dense[(size_t)cell * n_genes + g] = vals[i];
    }
}

__global__ void scatter_fdr_kernel(const float* __restrict__ fdr_sorted,
                                    const int*   __restrict__ orig_idx,
                                    float*       __restrict__ fdr_orig,
                                    int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fdr_orig[orig_idx[i]] = fdr_sorted[i];
}

__global__ void tt_mask_kernel(const float* __restrict__ scores,
                                const float* __restrict__ fdr,
                                int*         __restrict__ mask,
                                int n, float max_fdr, float min_score) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        mask[i] = (fdr[i] < max_fdr && fabsf(scores[i]) > min_score) ? 1 : 0;
}

__global__ void gather_kernel(const float* __restrict__ src,
                               const int*   __restrict__ idx,
                               float*       __restrict__ dst,
                               int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[idx[i]];
}

// Scatter peaks CSC into dense peaks^T (n_peaks × n_cells, row-major).
// peaks CSC: col_ptr[n_peaks+1], row_indices[nnz] (= cell indices), values[nnz].
// peaks^T[peak_p][cell_c] = peaks[cell_c][peak_p] → row-major: peaks_T[p * n_cells + c].
// One thread per peak column (peak p); iterates over nnz in that column.
__global__ void scatter_peaks_T_kernel(
    const int*   __restrict__ col_ptr,    // [n_peaks+1]
    const int*   __restrict__ row_indices,// [nnz] cell indices
    const float* __restrict__ values,     // [nnz]
    float*       __restrict__ peaks_T,    // [n_peaks × n_cells] row-major, zero-init
    int n_peaks, int n_cells)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n_peaks) return;
    for (int e = col_ptr[p]; e < col_ptr[p + 1]; ++e) {
        int c = row_indices[e];
        if (c >= 0 && c < n_cells)
            peaks_T[(size_t)p * n_cells + c] = values[e];
    }
}

}  // namespace detail

}  // namespace grn
}  // namespace singlet::gpu
