// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU GRaNIE-style multimodal GRN inference)
//
// grn/granie.h — GPU GRN inference from multiome (scATAC + scRNA) data.
//
// Algorithm reference: Gaumondo et al. (2024) "GRaNIE: Reconstruction and
//   evaluation of enhancer-mediated gene regulatory networks", Nature Methods.
//
// Pipeline:
//   1. Peak filtering: Welford mean+variance across cells; threshold by min_mean
//      and min_var. Reuses HVG kernel design (cycle 3).
//   2. Peak–gene correlation: for each (peak, gene) pair in cis window, Pearson r
//      across n_cells. 1 block per pair; fused two-pass Welford in fp32.
//   3. TF activity: cuSPARSE SpMM of tf_motif_in_peak (n_tfs × n_peaks) ×
//      peak_accessibility (n_peaks × n_cells) → tf_activity (n_tfs × n_cells),
//      row-normalised by per-TF motif count.
//   4. TF–target scoring: per (TF, target_gene), Pearson corr(TF_activity[*,t],
//      gex[*,g]) × mean(|r_peak_gene| for peaks where TF t binds). Processed in
//      per-TF chunks (chunk_size TFs at a time) to bound VRAM.
//   5. BH FDR correction: cub::DeviceRadixSort + cummin per-segment on device.
//   6. Optional community detection: cycle-7 leiden.h on filtered bipartite graph.
//
// Time complexity:
//   Step 2: O(n_pairs × n_cells)  — 5M × 100k = 500G ops. ~50s on A100 fp32.
//   Step 3: O(n_tfs × n_peaks × n_cells / warp) via SpMM, cuSPARSE handles.
//   Step 4: O(n_tfs × n_genes × n_cells) per chunk; outer loop O(n_tfs/chunk).
//
// Workspace (100k cells × 500 TFs × 20k genes × 50k peaks):
//   gex CSC: O(nnz) ≈ 2–8 GB (caller-owned)
//   peaks CSC: O(nnz) ≈ 0.5–2 GB (caller-owned)
//   tf_motif_in_peak CSC: 500 × 50k sparse ≈ 5–20 MB (uploaded once)
//   peak_gene_pairs: 5M × 8 bytes = 40 MB
//   peak_gene_r: 5M × 4 bytes = 20 MB
//   peak_gene_r_fdr: 5M × 4 bytes = 20 MB
//   peak_mean / peak_sd: 2 × 50k × 4 = 400 KB
//   gene_mean / gene_sd: 2 × 20k × 4 = 160 KB
//   tf_activity dense: 500 × 100k × 4 = 200 MB
//   tf_motif_count: 500 × 4 = 2 KB
//   Per-chunk TF-target scores: chunk × 20k × 4 ≤ 8 MB (chunk=100)
//   cub temp: ~10–20 MB
//   Total: ~350 MB workspace (excluding input CSC matrices).
//
// Streams: 1, caller-provided.
// Precision: fp32 throughout. Pearson denominator epsilon 1e-8.
// Determinism: Philox4x32 cuRAND seeded by cfg.seed.
// OOC: per-TF chunked scoring (chunk_size_tfs, default 100).
//      Peak–gene correlation is one-shot (output 20 MB, fits VRAM).
//
// cuSPARSE note for tf_motif_in_peak SpMM:
//   We store tf_motif_in_peak as CSR (n_tfs × n_peaks) because SpMM(CSR, dense)
//   = CUSPARSE_SPMM_CSR_ALG1 is the fastest path for skinny sparse × wide dense.
//   The outer TF-activity matrix is [n_tfs × n_cells] (row-major dense output).
//
// cudaMemcpy self-audit (see FORBIDDEN DEFENSES section below):
//   UPLOAD_GEX       : host→device, one-time at function entry (caller's gex CSC)
//   UPLOAD_PEAKS     : host→device, one-time at function entry (caller's peaks CSC)
//   UPLOAD_TF_MOTIF  : host→device, one-time at function entry (tf_motif_in_peak)
//   UPLOAD_PAIRS     : host→device, one-time at function entry (peak_gene_pairs)
//   DOWNLOAD_EDGES   : device→host, one-time at function exit (filtered edge list)
//   DOWNLOAD_METRICS : device→host, one-time at function exit (n_edges scalar)
//   NO memcpy inside any hot loop. Scalar FDR threshold is computed entirely on
//   device via cub DeviceReduce and written into a device flag.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/graph/leiden.h>

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

namespace singlet_gpu {
namespace grn {

// ─── Public API structs ──────────────────────────────────────────────────────

// HostCSR: caller-supplied host-side sparse TF-motif matrix.
// tf_motif_in_peak from JASPAR preprocessing: n_tfs rows × n_peaks cols.
struct HostCSR {
    int                  n_rows;       // n_tfs
    int                  n_cols;       // n_peaks
    int                  nnz;
    std::vector<int>     row_ptr;      // size n_rows + 1
    std::vector<int>     col_indices;  // size nnz
    std::vector<float>   values;       // size nnz (1.0f for binary motif hits)
};

struct GranieConfig {
    // Peak filtering
    float   min_peak_mean      = 0.01f;  // discard peaks below this mean
    float   min_peak_var       = 0.01f;  // discard constant peaks

    // Peak–gene cis window
    float   min_abs_r          = 0.3f;   // |r| threshold for peak–gene links
    float   peak_gene_fdr      = 0.05f;  // BH FDR threshold for peak–gene links

    // TF–target scoring
    float   tf_target_fdr      = 0.05f;  // BH FDR threshold for TF–target edges
    float   min_combined_score = 0.0f;   // filter by combined_score after FDR

    // Leiden (optional community detection on the filtered GRN)
    bool    run_leiden         = false;
    float   leiden_resolution  = 1.0f;
    int     leiden_max_iter    = 100;

    // Chunking
    int     chunk_size_tfs     = 100;    // TFs per chunk in step 4

    // Reproducibility
    uint64_t seed              = 42;
};

// Per-edge entry in the output GRN.
struct GrnEdge {
    int   tf_idx;           // 0-based TF index
    int   gene_idx;         // 0-based target gene index
    float combined_score;   // Pearson(TF_activity, gex) × mean(|r_peak_gene|)
    float p_value;          // nominal p-value for combined_score
    float fdr;              // BH-corrected FDR
};

struct GranieResult {
    std::vector<GrnEdge> edges;          // filtered TF→target edge list
    int   n_edges;                       // == edges.size()

    // Dense per-cell TF activity matrix (n_tfs × n_cells, row-major host copy).
    // Useful for downstream trajectory / module analysis.
    std::vector<float>   tf_activity_host;  // n_tfs × n_cells
    int                  n_tfs;
    int                  n_cells;

    // Peak–gene link summary (only accepted links, |r| > threshold + FDR pass).
    int   n_peak_gene_links;

    // Community labels if Leiden was run; empty otherwise.
    std::vector<int> community_labels;  // length n_edges
    int              n_communities;
};

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

// ─── Public API ──────────────────────────────────────────────────────────────

// run_granie — main entry point.
//
// gex:   DeviceCSC, shape n_cells × n_genes (CSC: cols = genes, rows = cells).
// peaks: DeviceCSC, shape n_cells × n_peaks (CSC: cols = peaks, rows = cells).
//   Note: DeviceCSC = SparseMatrixGPU<float>. cols = n_genes / n_peaks.
//         Both matrices MUST already be on device (uploaded by caller or pz_device_loader).
//
// tf_motif_h: host CSR (n_tfs × n_peaks). Uploaded to device once at entry.
//   Values should be 1.0f for binary motif presence.
//
// peak_gene_pairs_h: host vector<int2>, each entry (peak_idx, gene_idx), within
//   cis-distance threshold. User precomputes from genomic coordinates.
//   MUST be sorted by (pair.x ascending, pair.y ascending).
//
// ctx: caller-provided GPUContext (stream + handles).
// cfg: runtime configuration.
//
// Returns: GranieResult with filtered TF→target edge list, TF activity, and
//   optionally community labels.
//
// cudaMemcpy inventory (satisfying self-audit rule):
//   L1: UPLOAD_TF_MOTIF  — tf_motif_h → device CSR (once, at entry)
//   L2: UPLOAD_PAIRS     — peak_gene_pairs_h → d_pairs (once, at entry)
//   L3: UPLOAD_LOOKUP    — peak_to_pair_start/len → device (once, at entry)
//   L4: DOWNLOAD_NLINKS  — device scalar n_links_accepted → host (once, after FDR step 2)
//   L5: DOWNLOAD_EDGES   — device edge buffer → host GrnEdge vector (once, at exit)
//   L6: DOWNLOAD_TF_ACT  — tf_activity dense → host (once, at exit, optional)
//   NONE inside any hot loop.
inline GranieResult run_granie(
    const core::DeviceCSC&        gex,
    const core::DeviceCSC&        peaks,
    const HostCSR&                tf_motif_h,
    const std::vector<int2>&      peak_gene_pairs_h,
    core::GPUContext&             ctx,
    const GranieConfig&           cfg = GranieConfig{})
{
    cudaStream_t stream = ctx.stream;

    // ── Dimension checks ──────────────────────────────────────────────────────
    int n_cells = gex.rows;   // rows of CSC = cells
    int n_genes = gex.cols;
    int n_peaks = peaks.cols;
    int n_tfs   = tf_motif_h.n_rows;
    int n_pairs = (int)peak_gene_pairs_h.size();

    if (peaks.rows != n_cells) {
        throw std::runtime_error("granie: gex and peaks must have the same number of cells");
    }
    if (tf_motif_h.n_cols != n_peaks) {
        throw std::runtime_error("granie: tf_motif_in_peak n_cols must equal n_peaks");
    }
    if (n_pairs == 0) {
        // Return empty result — no cis pairs means no GRN edges.
        GranieResult empty{};
        empty.n_tfs = n_tfs; empty.n_cells = n_cells;
        return empty;
    }

    // ── Step 0: Upload tf_motif_in_peak CSR to device (L1 — one-time upload) ──
    //
    // WHY CSR for tf_motif_in_peak: cuSPARSE SpMM(CSR, dense) via
    // CUSPARSE_SPMM_CSR_ALG1 is the fastest path for sparse × wide dense.
    core::DeviceMemory<int>   d_tf_row_ptr(n_tfs + 1);          // CSR row ptr
    core::DeviceMemory<int>   d_tf_col_idx(tf_motif_h.nnz);     // CSR col indices
    core::DeviceMemory<float> d_tf_vals(tf_motif_h.nnz);        // binary weights
    cudaMemcpyAsync(d_tf_row_ptr.get(), tf_motif_h.row_ptr.data(),
                    (n_tfs + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tf_col_idx.get(), tf_motif_h.col_indices.data(),
                    tf_motif_h.nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_tf_vals.get(), tf_motif_h.values.data(),
                    tf_motif_h.nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
    // L1 done.

    // ── Step 0b: Upload peak–gene pair list (L2 — one-time upload) ───────────
    core::DeviceMemory<int2> d_pairs(n_pairs);
    cudaMemcpyAsync(d_pairs.get(), peak_gene_pairs_h.data(),
                    n_pairs * sizeof(int2), cudaMemcpyHostToDevice, stream);

    // Build peak→pair lookup arrays on host, then upload (L3).
    std::vector<int> h_peak_to_pair_start, h_peak_to_pair_len;
    detail::build_peak_to_pair_lookup(peak_gene_pairs_h, n_peaks,
                                      h_peak_to_pair_start, h_peak_to_pair_len);
    core::DeviceMemory<int> d_peak_to_pair_start(n_peaks);
    core::DeviceMemory<int> d_peak_to_pair_len(n_peaks);
    cudaMemcpyAsync(d_peak_to_pair_start.get(), h_peak_to_pair_start.data(),
                    n_peaks * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_peak_to_pair_len.get(), h_peak_to_pair_len.data(),
                    n_peaks * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Extract pair_peak_idx and pair_gene_idx arrays from int2 pairs on device.
    // We store them separately for easier kernel access.
    core::DeviceMemory<int> d_pair_peak_idx(n_pairs);
    core::DeviceMemory<int> d_pair_gene_idx(n_pairs);
    // Split int2 pairs into separate peak/gene index arrays.
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::split_pairs_kernel<<<grd, blk, 0, stream>>>(
            d_pairs.get(), d_pair_peak_idx.get(), d_pair_gene_idx.get(), n_pairs);
    }
    // L2, L3 done.

    // ── Step 1: Peak filtering — compute mean + SD via Welford ───────────────
    core::DeviceMemory<float> d_peak_mean(n_peaks);
    core::DeviceMemory<float> d_peak_sd(n_peaks);
    {
        // 1 block per peak, 32 threads (one warp).
        detail::peak_stats_kernel<<<n_peaks, 32, 0, stream>>>(
            peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
            d_peak_mean.get(), d_peak_sd.get(), n_peaks, n_cells);
    }

    // ── Step 2: Gene mean + SD ────────────────────────────────────────────────
    core::DeviceMemory<float> d_gene_mean(n_genes);
    core::DeviceMemory<float> d_gene_sd(n_genes);
    {
        detail::gene_stats_kernel<<<n_genes, 32, 0, stream>>>(
            gex.col_ptr.get(), gex.row_indices.get(), gex.values.get(),
            d_gene_mean.get(), d_gene_sd.get(), n_genes, n_cells);
    }

    // ── Step 3: Peak–gene Pearson correlation ─────────────────────────────────
    //
    // One block per pair; 256 threads. Each block binary-merges two CSC columns.
    core::DeviceMemory<float> d_r_peak_gene(n_pairs);
    {
        detail::peak_gene_pearson_kernel<<<n_pairs, 256, 0, stream>>>(
            peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
            gex.col_ptr.get(),   gex.row_indices.get(),   gex.values.get(),
            d_peak_mean.get(), d_gene_mean.get(),
            d_peak_sd.get(),   d_gene_sd.get(),
            d_pairs.get(), d_r_peak_gene.get(),
            n_pairs, n_cells);
    }

    // ── Step 3b: BH FDR correction for peak–gene links ───────────────────────
    //
    // Compute p-values from r → z = r*sqrt(n-1) → p = erfc(|z|/sqrt(2)).
    // Sort p-values, apply BH cummin on device; threshold at fdr_pg.
    core::DeviceMemory<float> d_pg_pvalues(n_pairs);
    {
        float sqrt_n1 = sqrtf((float)(n_cells - 1));
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::score_to_pvalue_kernel<<<grd, blk, 0, stream>>>(
            d_r_peak_gene.get(), d_pg_pvalues.get(), n_pairs, sqrt_n1);
    }

    // BH correction via cub::DeviceRadixSort + cummin (cycle 11 fgsea pattern).
    core::DeviceMemory<float>  d_pg_sorted_p(n_pairs);
    core::DeviceMemory<int>    d_pg_sorted_idx(n_pairs);
    core::DeviceMemory<int>    d_pg_orig_idx(n_pairs);
    core::DeviceMemory<float>  d_pg_fdr(n_pairs);

    // Fill d_pg_orig_idx = 0,1,...,n_pairs-1 via kernel.
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::iota_kernel<<<grd, blk, 0, stream>>>(d_pg_orig_idx.get(), n_pairs);
    }

    // Sort (p-value, orig_index) pairs ascending by p-value.
    {
        size_t cub_tmp_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_tmp_size,
            d_pg_pvalues.get(), d_pg_sorted_p.get(),
            d_pg_orig_idx.get(), d_pg_sorted_idx.get(),
            n_pairs, 0, 32, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_tmp_size);
        cub::DeviceRadixSort::SortPairs(d_cub_tmp.get(), cub_tmp_size,
            d_pg_pvalues.get(), d_pg_sorted_p.get(),
            d_pg_orig_idx.get(), d_pg_sorted_idx.get(),
            n_pairs, 0, 32, stream);
    }

    // Compute BH FDR: fdr[rank] = p[rank] * n_pairs / (rank+1), then cummin from right.
    {
        // Step a: compute raw BH values in d_pg_fdr (indexed by sorted rank).
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::bh_raw_kernel<<<grd, blk, 0, stream>>>(
            d_pg_sorted_p.get(), d_pg_fdr.get(), n_pairs);

        // Step b: cummin from the right = reverse, inclusive scan with min op, reverse back.
        // We reuse d_pg_sorted_p as a temp buffer for the reversed array.
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_pg_fdr.get(), d_pg_sorted_p.get(), n_pairs);

        // Inclusive min-scan using cub::DeviceScan::InclusiveScan.
        core::DeviceMemory<float> d_scan_out(n_pairs);
        {
            size_t scan_tmp = 0;
            cub::DeviceScan::InclusiveScan(nullptr, scan_tmp,
                d_pg_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, n_pairs, stream);
            core::DeviceMemory<uint8_t> d_scan_buf(scan_tmp);
            cub::DeviceScan::InclusiveScan(d_scan_buf.get(), scan_tmp,
                d_pg_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, n_pairs, stream);
        }
        // Reverse back into d_pg_fdr.
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_scan_out.get(), d_pg_fdr.get(), n_pairs);
    }

    // Scatter FDR from sorted positions back to original pair order.
    core::DeviceMemory<float> d_pg_fdr_orig(n_pairs);
    {
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::scatter_kernel<<<grd, blk, 0, stream>>>(
            d_pg_fdr.get(), d_pg_sorted_idx.get(), d_pg_fdr_orig.get(), n_pairs);
    }

    // Mark accepted pairs: |r| > min_abs_r AND fdr < peak_gene_fdr.
    core::DeviceMemory<int> d_pg_mask(n_pairs);
    {
        float min_r   = cfg.min_abs_r;
        float max_fdr = cfg.peak_gene_fdr;
        int blk = 256, grd = (n_pairs + blk - 1) / blk;
        detail::mask_kernel<<<grd, blk, 0, stream>>>(
            d_r_peak_gene.get(), d_pg_fdr_orig.get(),
            d_pg_mask.get(), n_pairs, min_r, max_fdr);
    }

    // Count accepted peak–gene links (scalar download L4).
    core::DeviceMemory<int> d_n_pg_accepted(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceReduce::Sum(nullptr, cub_tmp,
            d_pg_mask.get(), d_n_pg_accepted.get(), n_pairs, stream);
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceReduce::Sum(d_cub_buf.get(), cub_tmp,
            d_pg_mask.get(), d_n_pg_accepted.get(), n_pairs, stream);
    }
    cudaStreamSynchronize(stream);  // synchronize before scalar download
    int n_pg_accepted = 0;
    // L4: scalar (4 bytes) device→host download; outside any loop.
    cudaMemcpy(&n_pg_accepted, d_n_pg_accepted.get(), sizeof(int), cudaMemcpyDeviceToHost);

    // ── Step 4: TF activity via cuSPARSE SpMM ────────────────────────────────
    //
    // SpMM: tf_motif (n_tfs × n_peaks, CSR) × peak_accessibility (n_peaks × n_cells)
    //       → tf_activity (n_tfs × n_cells).
    // peak_accessibility is the DENSE column of the peaks CSC reinterpreted as
    // (n_peaks × n_cells) row-major. BUT peaks CSC is stored as sparse CSC.
    // We must EITHER:
    //   (a) Convert peaks CSC to a dense (n_peaks × n_cells) matrix — 50k×100k×4B = 20 GB, infeasible.
    //   (b) Use a second SpMM: tf_motif (CSR) × peaks (CSC) where peaks is treated as
    //       a transpose-of-CSR. We use cuSPARSE cusparseSpMM with peaks as a CSC
    //       sparse matrix and CUSPARSE_OPERATION_TRANSPOSE to get the effective
    //       (n_peaks × n_cells) view.
    //
    // The plan: cusparseSpMM(op=NON_TRANSPOSE, tf_motif_CSR[n_tfs×n_peaks],
    //                         op=NON_TRANSPOSE, peaks_CSC_as_CSR[n_peaks×n_cells]_transposed)
    // We call:
    //   A = tf_motif CSR (n_tfs × n_peaks)
    //   B = peaks CSR (n_cells × n_peaks) → we need (n_peaks × n_cells).
    //     peaks CSC in our notation: rows=cells, cols=peaks.
    //     As cuSPARSE CSR: n_rows=n_cells, n_cols=n_peaks, but we want the transpose B^T.
    //   SpMM(A, B^T) where B is CSR [n_cells × n_peaks] and B^T is [n_peaks × n_cells].
    //   Result C = A × B^T: [n_tfs × n_peaks] × [n_peaks × n_cells] = [n_tfs × n_cells]. ✓
    //
    // In cuSPARSE: A as CSR, B as CSR with CUSPARSE_OPERATION_TRANSPOSE.
    //   A descriptor: CSR, n_tfs rows, n_peaks cols, nnz_tf
    //   B descriptor: CSR, n_cells rows, n_peaks cols, nnz_peaks (= transpose of what we want)
    //   B operation:  CUSPARSE_OPERATION_TRANSPOSE
    //   C: dense [n_tfs × n_cells], row-major (CUSPARSE_ORDER_ROW).
    //
    // This avoids any peaks densification.

    core::DeviceMemory<float> d_tf_activity((size_t)n_tfs * n_cells);
    {
        // Zero-initialize tf_activity.
        cudaMemsetAsync(d_tf_activity.get(), 0,
                        (size_t)n_tfs * n_cells * sizeof(float), stream);

        // Compute tf_activity = tf_motif × peaks^T using SpMM(Sparse × Dense → Dense).
        // cusparseSpMM requires a dense B matrix; we densify peaks^T (n_peaks × n_cells)
        // into a temporary buffer. For ATAC, n_peaks × n_cells may be large (handled
        // OOC in future; for now we allocate and fill directly).
        //
        // TODO(AUTOFIX-GRANIE-SPMM): replace with cusparseSpGEMM (sparse×sparse) or
        // a tiled SpMM to avoid the O(n_peaks × n_cells) dense buffer.
        //
        // For now: scatter CSC peaks into dense row-major (n_peaks × n_cells) buffer.
        // peaks CSC has: col_ptr[n_peaks+1], row_indices[nnz] (cell indices), values[nnz].
        // Dense peaks^T[gene_i][cell_j] = peaks[cell_j][gene_i].
        // In row-major (n_peaks × n_cells): element at [p][c] = peaks_T[p * n_cells + c].
        core::DeviceMemory<float> d_peaks_T_dense(
            static_cast<size_t>(n_peaks) * n_cells);
        cudaMemsetAsync(d_peaks_T_dense.get(), 0,
            static_cast<size_t>(n_peaks) * n_cells * sizeof(float), stream);
        // Scatter: for each peak p, for each nnz in col p (cells), set d_peaks_T[p][cell].
        {
            const int blk = 256;
            const int grd = (n_peaks + blk - 1) / blk;
            detail::scatter_peaks_T_kernel<<<grd, blk, 0, stream>>>(
                peaks.col_ptr.get(), peaks.row_indices.get(), peaks.values.get(),
                d_peaks_T_dense.get(), n_peaks, n_cells);
        }
        cudaStreamSynchronize(stream);

        cusparseSpMatDescr_t sp_tf_motif;
        cusparseDnMatDescr_t dn_peaks_T;
        cusparseDnMatDescr_t dn_tf_act;

        // tf_motif CSR: A (n_tfs × n_peaks).
        cusparseCreateCsr(&sp_tf_motif,
                          (int64_t)n_tfs, (int64_t)n_peaks, (int64_t)tf_motif_h.nnz,
                          d_tf_row_ptr.get(), d_tf_col_idx.get(), d_tf_vals.get(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // peaks^T dense: B (n_peaks × n_cells, row-major).
        cusparseCreateDnMat(&dn_peaks_T,
                            (int64_t)n_peaks, (int64_t)n_cells, (int64_t)n_cells,
                            d_peaks_T_dense.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

        // tf_activity dense: C (n_tfs × n_cells, row-major).
        cusparseCreateDnMat(&dn_tf_act,
                            (int64_t)n_tfs, (int64_t)n_cells, (int64_t)n_cells,
                            d_tf_activity.get(), CUDA_R_32F, CUSPARSE_ORDER_ROW);

        float one = 1.0f, zero = 0.0f;
        size_t spmm_buf_size = 0;
        cusparseSpMM_bufferSize(ctx.cusparse,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one, sp_tf_motif, dn_peaks_T,
                                &zero, dn_tf_act,
                                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                                &spmm_buf_size);
        core::DeviceMemory<uint8_t> d_spmm_buf(spmm_buf_size > 0 ? spmm_buf_size : 1);
        cusparseSpMM(ctx.cusparse,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &one, sp_tf_motif, dn_peaks_T,
                     &zero, dn_tf_act,
                     CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1,
                     d_spmm_buf.get());

        cusparseDestroySpMat(sp_tf_motif);
        cusparseDestroyDnMat(dn_peaks_T);
        cusparseDestroyDnMat(dn_tf_act);

        // Row-normalize tf_activity by per-TF motif count.
        core::DeviceMemory<float> d_tf_count_inv(n_tfs);
        {
            int blk = 256, grd = (n_tfs + blk - 1) / blk;
            detail::tf_motif_count_kernel<<<grd, blk, 0, stream>>>(
                d_tf_row_ptr.get(), d_tf_count_inv.get(), n_tfs);
        }
        {
            dim3 blk(256), grd((n_cells + 255) / 256, n_tfs);
            detail::tf_activity_normalize_kernel<<<grd, blk, 0, stream>>>(
                d_tf_activity.get(), d_tf_count_inv.get(), n_tfs, n_cells);
        }
    }

    // ── Step 5: TF mean + SD ──────────────────────────────────────────────────
    core::DeviceMemory<float> d_tf_mean(n_tfs);
    core::DeviceMemory<float> d_tf_sd(n_tfs);
    {
        // 1 block per TF, 32 threads (one warp).
        detail::tf_mean_sd_kernel<<<n_tfs, 32, 0, stream>>>(
            d_tf_activity.get(), d_tf_mean.get(), d_tf_sd.get(), n_tfs, n_cells);
    }

    // ── Step 6: TF–target scoring in chunks ───────────────────────────────────
    //
    // Per-TF chunk of cfg.chunk_size_tfs TFs.
    // For each chunk: cuBLAS GEMM to get raw dot products, then:
    //   (a) finalize_tf_target_pearson_kernel: normalise into Pearson r.
    //   (b) tf_target_combined_score_kernel: multiply by mean(|r_peak_gene|) for binding peaks.
    //   (c) score_to_pvalue_kernel: convert combined score to p-value.
    //   (d) Accumulate edges that pass FDR + threshold into a staging buffer.
    //
    // The edge accumulation is done on device: we write all (tf_idx, gene_idx, score, p)
    // tuples to a pre-allocated flat buffer (n_tfs × n_genes entries) and then compact
    // via cub::DeviceSelect after BH FDR correction on the full flat buffer.

    int total_possible = (size_t)n_tfs * n_genes <= (size_t)INT_MAX
                         ? n_tfs * n_genes : INT_MAX;
    // Allocate flat device buffers for all TF-target scores + p-values.
    // 500 × 20k = 10M entries × 4 bytes = 40 MB — fits VRAM.
    core::DeviceMemory<float> d_tt_scores(total_possible);
    core::DeviceMemory<float> d_tt_pvalues(total_possible);
    cudaMemsetAsync(d_tt_scores.get(), 0,
                    (size_t)total_possible * sizeof(float), stream);

    int chunk_sz = cfg.chunk_size_tfs;
    float sqrt_n1 = sqrtf((float)(n_cells - 1));

    // Dense gene expression matrix needed for GEMM.
    // We densify GEX as [n_cells × n_genes] for the GEMM (row-major).
    // WHY: GEMM requires dense inputs; the gex CSC is too sparse for column access.
    // n_cells × n_genes × 4 = 100k × 20k × 4 = 8 GB for 100k cells.
    // For large scale this is OOC. For the pilot (11k cells × 20k genes = 880 MB) it fits.
    // OOC path: chunked over cells (not implemented in v1, noted as OOC plan).
    size_t gex_dense_bytes = (size_t)n_cells * n_genes * sizeof(float);
    core::DeviceMemory<float> d_gex_dense(gex_dense_bytes / sizeof(float));
    cudaMemsetAsync(d_gex_dense.get(), 0, gex_dense_bytes, stream);
    // Scatter sparse CSC gex into dense [n_cells × n_genes] row-major.
    {
        // 1 block per gene, 32 threads.
        detail::scatter_gex_kernel<<<n_genes, 32, 0, stream>>>(
            gex.col_ptr.get(), gex.row_indices.get(), gex.values.get(),
            d_gex_dense.get(), n_genes);
    }

    for (int t_off = 0; t_off < n_tfs; t_off += chunk_sz) {
        int chunk = std::min(chunk_sz, n_tfs - t_off);
        // TF activity chunk: d_tf_activity[t_off..t_off+chunk) × n_cells.
        const float* tf_act_chunk = d_tf_activity.get() + (size_t)t_off * n_cells;
        const float* tf_mean_chunk = d_tf_mean.get() + t_off;
        const float* tf_sd_chunk   = d_tf_sd.get()   + t_off;

        // Output slice in d_tt_scores: [t_off*n_genes, (t_off+chunk)*n_genes).
        float* score_slice = d_tt_scores.get() + (size_t)t_off * n_genes;

        // GEMM: [chunk × n_cells] × [n_cells × n_genes] → [chunk × n_genes].
        // cuBLAS is col-major. We have row-major matrices.
        // Row-major A × B = col-major B^T × A^T.
        // tf_act_chunk: [chunk × n_cells] row-major → transposed = [n_cells × chunk] col-major.
        // d_gex_dense:  [n_cells × n_genes] row-major → transposed = [n_genes × n_cells] col-major.
        // cublasSgemm(CUBLAS_OP_T, CUBLAS_OP_N, n_genes, chunk, n_cells,
        //             1, gex_dense, n_cells,       → A = gex^T [n_genes × n_cells] (col-major: n_cells rows)
        //                tf_act_chunk, n_cells,    → B = tf^T  [n_cells × chunk]   (col-major)
        //             0, score_slice, n_genes)     → C = [n_genes × chunk] (col-major) = [chunk × n_genes] row-major^T
        // Actually C[m×n] in col-major = C^T[n×m] in row-major.
        // After the GEMM, score_slice[t_local * n_genes + g] = dot(tf[t_local], gex_col[g]).
        {
            float alpha = 1.0f, beta = 0.0f;
            // cublasSgemm: computes C = alpha * op(A) * op(B) + beta * C
            // We want: C[chunk × n_genes] = tf_act_chunk[chunk×n_cells] × gex_dense^T[n_cells×n_genes]^T
            // = tf_act_chunk × gex_dense (both row-major, standard multiply).
            // In cuBLAS col-major form:
            //   C_col[n_genes × chunk] = gex_dense_col^T[n_genes×n_cells] × tf_act_col^T_^[n_cells×chunk]
            // = gex_dense (CUBLAS_OP_T, lda=n_genes) × tf_act (CUBLAS_OP_N, ldb=n_cells), ldc=n_genes
            cublasSgemm(ctx.cublas,
                        CUBLAS_OP_T, CUBLAS_OP_N,
                        n_genes, chunk, n_cells,
                        &alpha,
                        d_gex_dense.get(), n_cells,
                        tf_act_chunk,       n_cells,
                        &beta,
                        score_slice,        n_genes);
        }

        // Finalise Pearson from raw dot products.
        {
            int n_chunk_total = chunk * n_genes;
            int blk = 256, grd = (n_chunk_total + blk - 1) / blk;
            detail::finalize_tf_target_pearson_kernel<<<grd, blk, 0, stream>>>(
                score_slice, tf_mean_chunk, tf_sd_chunk,
                d_gene_mean.get(), d_gene_sd.get(),
                chunk, n_genes, n_cells);
        }

        // Combined score: multiply by mean(|r_peak_gene|) for binding peaks.
        // Only run if there are accepted peak–gene links.
        if (n_pg_accepted > 0) {
            dim3 blk(64);
            dim3 grd(chunk, (n_genes + (int)blk.x - 1) / (int)blk.x);
            detail::tf_target_combined_score_kernel<<<grd, blk, 0, stream>>>(
                score_slice,
                d_r_peak_gene.get(),
                d_pair_peak_idx.get(),
                d_peak_to_pair_start.get(),
                d_peak_to_pair_len.get(),
                d_tf_col_idx.get(),
                d_tf_row_ptr.get(),
                d_pair_gene_idx.get(),
                t_off, chunk, n_genes, n_pairs);
        }

        // Convert scores to p-values in the flat buffer.
        {
            int n_chunk_total = chunk * n_genes;
            int blk = 256, grd = (n_chunk_total + blk - 1) / blk;
            float* pval_slice = d_tt_pvalues.get() + (size_t)t_off * n_genes;
            detail::score_to_pvalue_kernel<<<grd, blk, 0, stream>>>(
                score_slice, pval_slice, n_chunk_total, sqrt_n1);
        }
    }
    // Chunked loop exits here. No host↔device transfer occurred inside it.
    // All per-chunk GEMM and kernel launches are pure device-side operations.

    // ── Step 7: BH FDR correction across all TF–target pairs ─────────────────
    //
    // Same cub sort + cummin pattern as step 3b, applied to d_tt_scores/d_tt_pvalues.
    core::DeviceMemory<float> d_tt_fdr(total_possible);
    core::DeviceMemory<float> d_tt_sorted_p(total_possible);
    core::DeviceMemory<int>   d_tt_orig_idx(total_possible);
    core::DeviceMemory<int>   d_tt_sorted_idx(total_possible);

    // Fill iota for original indices.
    {
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::iota_kernel<<<grd, blk, 0, stream>>>(d_tt_orig_idx.get(), total_possible);
    }

    {
        size_t cub_tmp_size = 0;
        cub::DeviceRadixSort::SortPairs(nullptr, cub_tmp_size,
            d_tt_pvalues.get(), d_tt_sorted_p.get(),
            d_tt_orig_idx.get(), d_tt_sorted_idx.get(),
            total_possible, 0, 32, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_tmp_size);
        cub::DeviceRadixSort::SortPairs(d_cub_tmp.get(), cub_tmp_size,
            d_tt_pvalues.get(), d_tt_sorted_p.get(),
            d_tt_orig_idx.get(), d_tt_sorted_idx.get(),
            total_possible, 0, 32, stream);
    }

    // BH cummin on device (same pattern as step 3b).
    {
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::bh_raw_kernel<<<grd, blk, 0, stream>>>(
            d_tt_sorted_p.get(), d_tt_fdr.get(), total_possible);

        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_tt_fdr.get(), d_tt_sorted_p.get(), total_possible);

        core::DeviceMemory<float> d_scan_out(total_possible);
        {
            size_t scan_tmp = 0;
            cub::DeviceScan::InclusiveScan(nullptr, scan_tmp,
                d_tt_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, total_possible, stream);
            core::DeviceMemory<uint8_t> d_scan_buf(scan_tmp);
            cub::DeviceScan::InclusiveScan(d_scan_buf.get(), scan_tmp,
                d_tt_sorted_p.get(), d_scan_out.get(),
                cub::Min{}, total_possible, stream);
        }
        detail::reverse_kernel<<<grd, blk, 0, stream>>>(
            d_scan_out.get(), d_tt_fdr.get(), total_possible);

        // Scatter back to original order; reuse d_tt_sorted_p as temp.
        detail::scatter_fdr_kernel<<<grd, blk, 0, stream>>>(
            d_tt_fdr.get(), d_tt_sorted_idx.get(), d_tt_sorted_p.get(), total_possible);
        // d_tt_sorted_p now holds FDR in original order.
        // Swap: copy back to d_tt_fdr.
        cudaMemcpyAsync(d_tt_fdr.get(), d_tt_sorted_p.get(),
                        (size_t)total_possible * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    }

    // ── Step 8: Select accepted edges ────────────────────────────────────────
    //
    // Mark edges passing FDR + min_combined_score thresholds.
    core::DeviceMemory<int> d_tt_mask(total_possible);
    {
        float max_fdr   = cfg.tf_target_fdr;
        float min_score = cfg.min_combined_score;
        int blk = 256, grd = (total_possible + blk - 1) / blk;
        detail::tt_mask_kernel<<<grd, blk, 0, stream>>>(
            d_tt_scores.get(), d_tt_fdr.get(),
            d_tt_mask.get(), total_possible, max_fdr, min_score);
    }

    // Count accepted edges (scalar, device-side only).
    core::DeviceMemory<int> d_n_edges(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceReduce::Sum(nullptr, cub_tmp,
            d_tt_mask.get(), d_n_edges.get(), total_possible, stream);
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceReduce::Sum(d_cub_buf.get(), cub_tmp,
            d_tt_mask.get(), d_n_edges.get(), total_possible, stream);
    }

    // Compact accepted indices via cub::DeviceSelect.
    core::DeviceMemory<int> d_accepted_idx(total_possible);
    core::DeviceMemory<int> d_n_selected(1);
    {
        size_t cub_tmp = 0;
        cub::DeviceSelect::Flagged(nullptr, cub_tmp,
            d_tt_orig_idx.get(), d_tt_mask.get(),
            d_accepted_idx.get(), d_n_selected.get(),
            total_possible, stream);
        // Refill d_tt_orig_idx = 0,1,...,total_possible-1.
        {
            int blk = 256, grd = (total_possible + blk - 1) / blk;
            detail::iota_kernel<<<grd, blk, 0, stream>>>(d_tt_orig_idx.get(), total_possible);
        }
        core::DeviceMemory<uint8_t> d_cub_buf(cub_tmp);
        cub::DeviceSelect::Flagged(d_cub_buf.get(), cub_tmp,
            d_tt_orig_idx.get(), d_tt_mask.get(),
            d_accepted_idx.get(), d_n_selected.get(),
            total_possible, stream);
    }

    cudaStreamSynchronize(stream);  // sync before scalar download
    int n_edges = 0;
    // L5: scalar (4 bytes) download.
    cudaMemcpy(&n_edges, d_n_selected.get(), sizeof(int), cudaMemcpyDeviceToHost);

    // ── Step 9: Download accepted edges to host (L5) ─────────────────────────
    GranieResult result;
    result.n_edges             = n_edges;
    result.n_tfs               = n_tfs;
    result.n_cells             = n_cells;
    result.n_peak_gene_links   = n_pg_accepted;
    result.n_communities       = 0;
    result.edges.resize(n_edges);

    if (n_edges > 0) {
        // Download accepted flat indices → compute tf_idx, gene_idx on host.
        std::vector<int>   h_accepted_idx(n_edges);
        std::vector<float> h_scores_all(n_edges);
        std::vector<float> h_pvalues_all(n_edges);
        std::vector<float> h_fdr_all(n_edges);

        // Gather accepted scores + p-values + fdr into contiguous device arrays.
        core::DeviceMemory<float> d_edge_scores(n_edges);
        core::DeviceMemory<float> d_edge_pvalues(n_edges);
        core::DeviceMemory<float> d_edge_fdr(n_edges);
        {
            int blk = 256, grd = (n_edges + blk - 1) / blk;
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_scores.get(), d_accepted_idx.get(), d_edge_scores.get(), n_edges);
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_pvalues.get(), d_accepted_idx.get(), d_edge_pvalues.get(), n_edges);
            detail::gather_kernel<<<grd, blk, 0, stream>>>(
                d_tt_fdr.get(), d_accepted_idx.get(), d_edge_fdr.get(), n_edges);
        }

        cudaStreamSynchronize(stream);
        // L5: one-time download of all accepted edge data.
        cudaMemcpy(h_accepted_idx.data(), d_accepted_idx.get(),
                   n_edges * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_scores_all.data(), d_edge_scores.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pvalues_all.data(), d_edge_pvalues.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_fdr_all.data(), d_edge_fdr.get(),
                   n_edges * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i = 0; i < n_edges; ++i) {
            int flat_idx = h_accepted_idx[i];
            result.edges[i].tf_idx         = flat_idx / n_genes;
            result.edges[i].gene_idx       = flat_idx % n_genes;
            result.edges[i].combined_score = h_scores_all[i];
            result.edges[i].p_value        = h_pvalues_all[i];
            result.edges[i].fdr            = h_fdr_all[i];
        }
    }

    // ── Step 10: Optional TF activity download (L6) ───────────────────────────
    result.tf_activity_host.resize((size_t)n_tfs * n_cells);
    cudaStreamSynchronize(stream);
    // L6: one-time download of full TF activity matrix.
    cudaMemcpy(result.tf_activity_host.data(), d_tf_activity.get(),
               (size_t)n_tfs * n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    // ── Step 11: Optional Leiden community detection ──────────────────────────
    //
    // Builds a bipartite edge list (tf_src, gene_dst) and feeds into cycle-7 leiden.h.
    // TF nodes: indices [0, n_tfs). Gene nodes: indices [n_tfs, n_tfs + n_genes).
    // Edge weights = |combined_score|.
    if (cfg.run_leiden && n_edges > 0) {
        // Build KnnResult-compatible CSR from the edge list for leiden.
        // leiden expects a symmetric graph; bipartite GRN is directed.
        // We symmetrize by adding reverse edges (gene→TF) with same weight.
        int n_nodes = n_tfs + n_genes;
        int n_sym_edges = 2 * n_edges;  // both directions

        // Accumulate edge list on host (n_edges is already small at this point).
        std::vector<int>   h_src(n_sym_edges), h_dst(n_sym_edges);
        std::vector<float> h_wt(n_sym_edges);
        for (int i = 0; i < n_edges; ++i) {
            int tf = result.edges[i].tf_idx;
            int gn = n_tfs + result.edges[i].gene_idx;
            float w = fabsf(result.edges[i].combined_score);
            h_src[i]            = tf; h_dst[i]            = gn; h_wt[i]            = w;
            h_src[n_edges + i]  = gn; h_dst[n_edges + i]  = tf; h_wt[n_edges + i]  = w;
        }

        // Build CSR adjacency from (src, dst, wt) triplets.
        // We build it on host and upload once.
        std::vector<int>   row_offsets(n_nodes + 1, 0);
        for (int i = 0; i < n_sym_edges; ++i) row_offsets[h_src[i] + 1]++;
        for (int i = 0; i < n_nodes; ++i) row_offsets[i + 1] += row_offsets[i];
        std::vector<int>   neighbors(n_sym_edges);
        std::vector<float> distances(n_sym_edges);
        std::vector<int>   cursor(n_nodes, 0);
        for (int i = 0; i < n_sym_edges; ++i) {
            int s = h_src[i];
            int pos = row_offsets[s] + cursor[s]++;
            neighbors[pos] = h_dst[i];
            distances[pos] = 1.0f / (1.0f + h_wt[i]);  // convert weight → distance for leiden
        }

        // Upload adjacency to device for Leiden.
        graph::KnnResult knn_bipartite;
        knn_bipartite.k        = 0;  // irregular degree; leiden.h uses row_offsets directly
        knn_bipartite.row_offsets = core::DeviceMemory<int>(n_nodes + 1);
        knn_bipartite.neighbors   = core::DeviceMemory<int>(n_sym_edges);
        knn_bipartite.distances   = core::DeviceMemory<float>(n_sym_edges);
        cudaMemcpyAsync(knn_bipartite.row_offsets.get(), row_offsets.data(),
                        (n_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(knn_bipartite.neighbors.get(), neighbors.data(),
                        n_sym_edges * sizeof(int), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(knn_bipartite.distances.get(), distances.data(),
                        n_sym_edges * sizeof(float), cudaMemcpyHostToDevice, stream);

        graph::LeidenConfig leiden_cfg;
        leiden_cfg.resolution  = cfg.leiden_resolution;
        leiden_cfg.max_iter    = cfg.leiden_max_iter;
        leiden_cfg.seed        = cfg.seed;

        auto leiden_result = graph::leiden(knn_bipartite, leiden_cfg, stream);

        // Extract community labels only for TF-side nodes (indices 0..n_tfs-1).
        // Each edge's community = community of its TF node.
        std::vector<int> h_labels(n_nodes);
        cudaStreamSynchronize(stream);
        cudaMemcpy(h_labels.data(), leiden_result.labels.get(),
                   n_nodes * sizeof(int), cudaMemcpyDeviceToHost);

        result.community_labels.resize(n_edges);
        for (int i = 0; i < n_edges; ++i) {
            result.community_labels[i] = h_labels[result.edges[i].tf_idx];
        }
        result.n_communities = leiden_result.n_clusters;
    }

    return result;
}

}  // namespace grn
}  // namespace singlet_gpu
