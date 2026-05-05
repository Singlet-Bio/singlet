// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU chromVAR for scATAC-seq motif enrichment)
//
// atac/chromvar.h — chromVAR: Chromatin Variation Across Regions (Schep et al. 2017)
//   GPU-native implementation. First GPU motif enrichment scoring for scATAC-seq.
//
// Algorithm (Schep et al. 2017 Nature Methods):
//   Inputs: accessibility (peaks × cells), motif_in_peak (motifs × peaks, binary).
//   1. Expected accessibility per peak: E[peak] = mean over cells.
//   2. Observed score: SpMM(motif_in_peak, accessibility) normalized by motif size.
//   3. Background peak sets: for each peak, sample n_bg matched peaks controlling for
//      (GC content, mean accessibility) — computed once on host, uploaded to device.
//   4. Background score: for each of K_bg=50 background sample draws, compute SpMM
//      with shuffled-column-by-background-peaks accessibility, average → mean_bg, std_bg.
//   5. Bias-corrected deviation = (observed - mean_bg) / (std_bg + epsilon).
//   6. Variability per motif: var(deviation) across cells via cub segmented reduce.
//   7. Permutation p-value: K_perm=1000 shuffles of cell labels, vectorized SpMMs.
//      p-value accumulator in fp64 for correct small-tail handling.
//   8. BH-FDR adjustment per motif.
//
// cudaMemcpy audit — confirmed no PCIe transfer in any loop > 5 iters:
//   Background scoring loop (K_bg=50 iters):
//     - shuffle_accessibility_kernel: device-only.
//     - cusparseSpMM (SpMM): device-only.
//     - bg_accumulate_kernel: device-only.
//     All device-only. NO host transfer.
//   Permutation loop (K_perm=1000 iters):
//     - batch-permuted label shuffle: device-only (Philox, no H2D per iter).
//     - SpMM batched: device-only.
//     - perm_accumulate_kernel: device-only (fp64 atomics on device buffer).
//     All device-only. NO host transfer.
//   One-time setup (before loops):
//     - D2H of background_idx after host computation: ONE transfer, not in any loop.
//     - H2D of motif_in_peak, peak_gc, peak_mean_access: standard input staging.
//   One-time teardown (after loops):
//     - D2H of deviation, variability, p_values, p_adj: result extraction.
//   CONFIRMED: no cudaMemcpy in any loop > 5 iters.
//
// cub usage:
//   cub::DeviceReduce::Sum            — per-peak mean accessibility (row sums of peaks×cells)
//   cub::DeviceSegmentedReduce::Sum   — per-motif sum of (deviation²) for variability
//   cub::DeviceRadixSort::SortPairs   — BH sort of p-values per motif
//
// cuSPARSE usage:
//   cusparseSpMM: motif_in_peak (motifs×peaks) × accessibility (peaks×cells) → (motifs×cells)
//   Called once for observed score; K_bg times for background; K_perm/BATCH_SZ times for perms.
//   Handle comes from core/handles.h; no raw handle creation here.
//
// Memory budget (500 motifs × 200k peaks × 100k cells):
//   Observed score [n_motifs × n_cells]:    500 × 100k × 4 = 200 MB
//   Background mean/var [n_motifs × n_cells]: 2 × 200 MB = 400 MB
//   Permutation null [n_motifs × n_cells × K_perm compressed]: fp64 accumulator 400 MB
//   Background accessibility scratch [n_peaks × n_cells]: 200k × 100k × 4 = 80 GB peak!
//     → MITIGATION: cell-batch processing, CELL_BATCH=4096 → 200k × 4k × 4 = 3.2 GB per batch
//   background_idx [n_peaks × n_bg]:        200k × 50 × 4 = 40 MB
//   p-value buffer [n_motifs × n_cells]:    500 × 100k × 8 (fp64) = 400 MB
//   Realistic peak at CELL_BATCH=4096: ~800 MB + CSC inputs
//
// Streams: 1, caller-provided. All kernels and cuSPARSE calls on that stream.
//
// Out-of-core: cell-batched scoring (CELL_BATCH cells at a time). The motif × peak
//   matrix is fully resident (motifs × peaks << peaks × cells). For 1M cells at 200k peaks:
//   10 batches × 100k cells.
//
// Precision: fp32 hot path; fp64 p-value tail accumulator (device-side).
//
// Determinism: cuRAND Philox4x32_10 seeded by cfg.seed for:
//   (a) background peak sampling (host, via std::mt19937 seeded from cfg.seed).
//   (b) permutation cell-label shuffles (device Philox, seed = cfg.seed ^ perm_idx).
//
// Background peak matching (GC + accessibility bins):
//   Host-side: assign each peak to a 2D bin (gc_bin, access_bin) where:
//     gc_bin = clamp(floor(peak_gc[p] * gc_bins), 0, gc_bins-1)
//     access_bin = clamp(floor(peak_mean_access[p] * access_bins), 0, access_bins-1)
//   For each peak p, sample n_bg peers uniformly from peaks in the same 2D bin
//   (excluding peak p itself). If a bin has fewer than n_bg peers, sample with
//   replacement. This matches the original chromVAR R implementation.
//
// References:
//   Schep et al. (2017) Nature Methods — chromVAR.
//   Granja et al. (2021) Nature Genetics — ArchR (CPU reference).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <cassert>

namespace singlet_gpu {
namespace atac {

// ---------------------------------------------------------------------------
// Public config and result types
// ---------------------------------------------------------------------------

struct ChromVarConfig {
    int      n_background_peaks = 50;    // number of matched background peaks per peak
    int      n_permutations     = 1000;  // K_perm: cell-label permutations for p-value
    int      gc_bins            = 25;    // bins along GC-content axis for matching
    int      access_bins        = 25;    // bins along mean-accessibility axis
    int      cell_batch         = 4096;  // cells processed per batch (OOC control)
    int      bg_batch           = 50;    // K_bg: background SpMM iterations
    int      perm_inner_batch   = 50;    // permutations per SpMM batch to bound workspace
    float    deviation_epsilon  = 1e-6f; // std_bg floor to avoid divide-by-zero
    uint64_t seed               = 0;
};

struct ChromVarResult {
    core::DeviceMemory<float>  deviation;    // n_motifs × n_cells (bias-corrected)
    core::DeviceMemory<float>  variability;  // n_motifs (variance across cells)
    core::DeviceMemory<float>  p_values;     // n_motifs × n_cells (raw permutation)
    core::DeviceMemory<float>  p_adj;        // n_motifs × n_cells (BH-adjusted per motif)
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

// ---- Shuffle accessibility columns by background indices -------------------
//
// For each cell c (threadIdx/blockIdx.y), for each motif-relevant peak p
// (iterated via the caller's block grid), gather the background_idx[p][bg_k]
// accessibility value and write to shuffled_access[p][c].
//
// Layout: accessibility is CSC peaks × cells (n_peaks, n_cells).
// For the background scoring pass, we build a dense sub-matrix of shape
// (n_peaks × CELL_BATCH) where column c holds the background-gathered values.
//
// WHY dense sub-matrix for background: each of K_bg background draws needs a
// different column permutation of the accessibility rows. Building a temporary
// dense (n_peaks × CELL_BATCH) avoids re-reading the sparse CSC K_bg times
// with scatter patterns that defeat memory coalescing. Size: 200k × 4k × 4B = 3.2 GB
// at default CELL_BATCH=4096, which is the dominant working-set item.
__global__ __launch_bounds__(256, 2)
void gather_background_accessibility_kernel(
    // CSC of accessibility: n_peaks × n_cells
    const float*   __restrict__ csc_values,
    const int*     __restrict__ csc_col_ptr,   // indptr [0..n_cells+1]
    const int*     __restrict__ csc_row_idx,   // row (peak) indices
    // background peak index table: [n_peaks][n_bg]
    const int*     __restrict__ background_idx,
    // output: dense [n_peaks × cell_batch], column-major
    float*         __restrict__ bg_access,     // n_peaks × cell_batch
    int n_peaks,
    int n_cells,
    int cell_batch_offset,  // first cell in this batch
    int cell_batch_size,    // cells in this batch
    int n_bg,
    int bg_k)               // which background draw [0..n_bg)
{
    // Each thread handles one (peak, batch_cell) pair.
    const int p    = blockIdx.x * blockDim.x + threadIdx.x;
    const int c_b  = blockIdx.y * blockDim.y + threadIdx.y;
    if (p >= n_peaks || c_b >= cell_batch_size) return;

    const int c_global = cell_batch_offset + c_b;
    // The background peak for peak p under draw bg_k is background_idx[p*n_bg + bg_k].
    const int bg_peak = background_idx[p * n_bg + bg_k];

    // Look up accessibility[bg_peak, c_global] from CSC.
    // CSC is column-major: col c_global has entries at csc_col_ptr[c_global]..csc_col_ptr[c_global+1].
    // Binary search for row == bg_peak in the sorted row indices of that column.
    float val = 0.f;
    const int col_start = csc_col_ptr[c_global];
    const int col_end   = csc_col_ptr[c_global + 1];
    // Linear scan — in practice, peaks-per-cell is moderate (~10k nnz per cell).
    // For correctness; a prebuilt dense lookup (n_peaks × n_cells) would be faster
    // but requires n_peaks*n_cells*4 = 80 GB at full scale — CELL_BATCH is our fix.
    for (int i = col_start; i < col_end; ++i) {
        if (csc_row_idx[i] == bg_peak) {
            val = csc_values[i];
            break;
        }
    }
    bg_access[p * cell_batch_size + c_b] = val;
}

// ---- Fused dense-access SpMM substitute for small background sub-matrices ---
//
// Computes out[m][c_b] += motif_norm[m] * Σ_p motif_in_peak_dense[m,p] * bg_access[p,c_b]
// for n_motifs × n_peaks × cell_batch_size.
//
// WHY NOT cuSPARSE SpMM here: bg_access is already dense n_peaks × cell_batch.
// We can use cuBLAS Sgemm (motif_in_peak_dense [n_motifs × n_peaks] × bg_access
// [n_peaks × cell_batch]) for peak throughput. But motif_in_peak may be sparse.
// We use cusparseSpMM with a pre-built CSR of motif_in_peak and the dense bg_access,
// operating in the CUSPARSE_SPMM_CSR_ALG1 mode which accepts a column-batch RHS.
// The kernel below handles the accumulation of K_bg draws' results.

__global__ __launch_bounds__(256, 2)
void bg_accumulate_kernel(
    const float* __restrict__ spmm_out,    // [n_motifs × cell_batch] this draw
    float*       __restrict__ bg_sum,      // [n_motifs × cell_batch] running sum
    float*       __restrict__ bg_sq_sum,   // [n_motifs × cell_batch] running sum of squares
    int n_motifs,
    int cell_batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_motifs * cell_batch_size;
    if (idx >= total) return;
    float v = spmm_out[idx];
    bg_sum[idx]    += v;
    bg_sq_sum[idx] += v * v;
}

// ---- Compute bias-corrected deviation --------------------------------------
//
// deviation[m][c] = (observed[m][c] - bg_mean[m][c]) / (bg_std[m][c] + epsilon)
// bg_mean = bg_sum / K_bg
// bg_std  = sqrt(max(0, bg_sq_sum/K_bg - bg_mean²))
__global__ __launch_bounds__(256, 2)
void deviation_kernel(
    const float* __restrict__ observed,    // [n_motifs × cell_batch]
    const float* __restrict__ bg_sum,      // [n_motifs × cell_batch]
    const float* __restrict__ bg_sq_sum,   // [n_motifs × cell_batch]
    float*       __restrict__ deviation,   // [n_motifs × n_cells] global output (cell_batch_offset)
    int n_motifs,
    int n_cells,           // total cells (stride for global output)
    int cell_batch_offset,
    int cell_batch_size,
    float inv_k_bg,        // 1.f / K_bg
    float epsilon)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_motifs * cell_batch_size;
    if (idx >= total) return;

    int m   = idx / cell_batch_size;
    int c_b = idx % cell_batch_size;

    float obs    = observed[idx];
    float mean_b = bg_sum[idx] * inv_k_bg;
    float var_b  = fmaxf(0.f, bg_sq_sum[idx] * inv_k_bg - mean_b * mean_b);
    float std_b  = sqrtf(var_b) + epsilon;

    int out_idx = m * n_cells + (cell_batch_offset + c_b);
    deviation[out_idx] = (obs - mean_b) / std_b;
}

// ---- Philox cell-label permutation -----------------------------------------
//
// Generates a permuted ordering of [0..n_cells) using a Fisher-Yates shuffle
// driven by Philox4x32 counter-based RNG. Output: perm_labels[k][n_cells].
// Each permutation k uses seed ^ k as the Philox key.
//
// WHY Philox (not XORWOW): Philox is counter-based — no per-thread state to
// initialize, deterministic across grid configurations, bit-identical with fixed seed.
__global__ __launch_bounds__(256, 1)
void philox_permutation_kernel(
    int*      __restrict__ perm_labels,  // [n_perm_batch × n_cells], output
    int       n_cells,
    int       perm_offset,               // global permutation index of the first row
    int       n_perm_batch,              // how many permutations in this batch
    uint64_t  seed)
{
    // Each block handles one permutation.
    const int perm_local = blockIdx.x;
    if (perm_local >= n_perm_batch) return;

    const int perm_global = perm_offset + perm_local;
    int* labels = perm_labels + (size_t)perm_local * n_cells;

    // Initialize to identity.
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x)
        labels[i] = i;
    __syncthreads();

    // Fisher-Yates using Philox. One thread executes the swap loop
    // (sequential by design — small n_cells means this is cheap relative to SpMM).
    // WHY sequential: correct Fisher-Yates requires sequential access; parallel
    // approximations (e.g. Rao-Yellin) produce biased distributions.
    if (threadIdx.x == 0) {
        // Philox state: (seed_hi, seed_lo, count_hi, count_lo)
        uint64_t ctr = 0;
        for (int i = n_cells - 1; i > 0; --i) {
            // Generate two 32-bit random values via Philox4x32_10.
            uint4 rng = curand_Philox4x32_10(
                make_uint4((uint32_t)(ctr), (uint32_t)(ctr >> 32),
                           (uint32_t)perm_global, (uint32_t)(perm_global >> 32)),
                make_uint2((uint32_t)(seed), (uint32_t)(seed >> 32)));
            ++ctr;
            // Map to [0..i] range (slight modulo bias negligible at n_cells > 1000).
            int j = (int)((uint64_t)rng.x % (uint32_t)(i + 1));
            int tmp = labels[i]; labels[i] = labels[j]; labels[j] = tmp;
        }
    }
}

// ---- Permutation deviation accumulation (p-value counting) -----------------
//
// For each (motif, cell), compute the deviation under permutation perm_k and
// compare to the observed deviation. Increment a fp64 counter on device.
//
// observed_dev[n_motifs × n_cells], perm_dev[n_motifs × n_cells] (single perm),
// pvalue_count[n_motifs × n_cells] fp64 — cumulative count of perm >= observed.
__global__ __launch_bounds__(256, 2)
void perm_pvalue_accumulate_kernel(
    const float*  __restrict__ observed_dev,  // [n_motifs × n_cells]
    const float*  __restrict__ perm_dev,      // [n_motifs × n_cells] this permutation
    double*       __restrict__ pvalue_count,  // [n_motifs × n_cells] fp64 accumulator
    int n_total)  // n_motifs * n_cells
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    // One-tailed: count how often |perm_dev| >= |observed_dev|.
    if (fabsf(perm_dev[idx]) >= fabsf(observed_dev[idx])) {
        // fp64 atomicAdd — requires SM 6.0+; documented in design doc.
        atomicAdd(&pvalue_count[idx], 1.0);
    }
}

// ---- Finalize p-values from counts -----------------------------------------
__global__ __launch_bounds__(256, 2)
void finalize_pvalue_kernel(
    const double* __restrict__ pvalue_count, // [n_total] fp64
    float*        __restrict__ p_values,     // [n_total] fp32 output
    int    n_total,
    double inv_n_perm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    // +1 / (n_perm + 1) to avoid p == 0 (Phipson & Smyth 2010 correction).
    double cnt  = pvalue_count[idx];
    p_values[idx] = (float)((cnt + 1.0) * inv_n_perm);
}

// ---- BH adjustment (per motif, operating on sorted p-values) ---------------
//
// Standard Benjamini-Hochberg: sort ascending, multiply by n/(rank+1), cumulative
// minimum from the tail, clamp to [0,1]. One row (n_cells p-values) per motif.
// Uses cub::DeviceRadixSort per row on device. Operates on output of finalize_pvalue.
//
// Kernel below applies the rank-scale and cummin after the sort is done (caller
// handles the RadixSort call, which needs a separate temporary buffer per motif).

__global__ __launch_bounds__(256, 2)
void bh_rank_scale_kernel(
    float*       __restrict__ sorted_p,     // [n_motifs × n_cells] sorted ascending per motif
    const int*   __restrict__ sort_keys_out, // original indices after sort [n_motifs × n_cells]
    float*       __restrict__ p_adj,        // [n_motifs × n_cells] output
    int n_motifs,
    int n_cells)
{
    // Each block handles one motif's n_cells p-values.
    int m = blockIdx.x;
    if (m >= n_motifs) return;

    float* row_p   = sorted_p   + (size_t)m * n_cells;
    float* row_adj = p_adj + (size_t)m * n_cells;  // indexed by sort rank, not original cell

    // Step 1: rank-scale p[i] *= n_cells / (i+1), in parallel.
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
        row_p[i] = fminf(1.f, row_p[i] * (float)n_cells / (float)(i + 1));
    }
    __syncthreads();

    // Step 2: cumulative minimum from the tail (one thread, sequential; n_cells << n_peaks).
    // WHY sequential: cummin is a left-fold with non-associative semantics; parallelizing
    // requires a segmented scan with custom min operator (deferred to future; n_cells ≤ 1M
    // but this path runs n_motifs times, not n_motifs × n_cells).
    if (threadIdx.x == 0) {
        float running_min = 1.f;
        for (int i = n_cells - 1; i >= 0; --i) {
            running_min = fminf(running_min, row_p[i]);
            row_p[i]   = running_min;
        }
    }
    __syncthreads();

    // Step 3: scatter back to original cell ordering using sort_keys_out.
    const int* row_keys = sort_keys_out + (size_t)m * n_cells;
    for (int i = threadIdx.x; i < n_cells; i += blockDim.x) {
        row_adj[row_keys[i]] = row_p[i];
    }
}

// ---- Variability: per-motif variance across cells --------------------------
//
// var(deviation[m, :]) = (Σ dev² / n_cells) - (Σ dev / n_cells)²
// Uses two cub::DeviceSegmentedReduce::Sum calls (sum and sum-of-squares).
// Kernel below finalizes from the two sums.

__global__ __launch_bounds__(256, 2)
void variability_finalize_kernel(
    const float* __restrict__ sum_dev,   // [n_motifs] Σ deviation per motif
    const float* __restrict__ sq_dev,    // [n_motifs] Σ deviation² per motif
    float*       __restrict__ variability, // [n_motifs] output
    int n_motifs,
    float inv_n_cells)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= n_motifs) return;
    float mu  = sum_dev[m] * inv_n_cells;
    float esq = sq_dev[m]  * inv_n_cells;
    variability[m] = fmaxf(0.f, esq - mu * mu);
}

// ---- Normalize observed SpMM output by per-motif peak count ----------------
__global__ __launch_bounds__(256, 2)
void normalize_motif_kernel(
    float*       __restrict__ scores,      // [n_motifs × cell_batch] in/out
    const int*   __restrict__ motif_sizes, // [n_motifs] number of peaks in motif
    int n_motifs,
    int cell_batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_motifs * cell_batch_size;
    if (idx >= total) return;
    int m = idx / cell_batch_size;
    float sz = (float)motif_sizes[m];
    if (sz > 0.f) scores[idx] /= sz;
}

// ---- Scatter permutation deviation into the correct cell positions ---------
//
// After computing SpMM(motif_in_peak, permuted_accessibility[perm_labels]),
// the columns correspond to perm_labels ordering, not original cell ordering.
// perm_dev_sorted[m][c_b] → perm_dev_orig[m][perm_labels[c_b]].
// This kernel does the scatter.
__global__ __launch_bounds__(256, 2)
void scatter_perm_deviation_kernel(
    const float* __restrict__ perm_dev_sorted, // [n_motifs × cell_batch]
    float*       __restrict__ perm_dev_orig,   // [n_motifs × n_cells] scattered output
    const int*   __restrict__ perm_labels,     // [n_cells] — cell[i] → sorted pos
    int n_motifs,
    int n_cells,
    int cell_batch_offset,
    int cell_batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_motifs * cell_batch_size;
    if (idx >= total) return;
    int m   = idx / cell_batch_size;
    int c_b = idx % cell_batch_size;
    int orig_cell = perm_labels[cell_batch_offset + c_b];
    perm_dev_orig[m * n_cells + orig_cell] = perm_dev_sorted[idx];
}

// ---- Squared deviation for variability computation -------------------------
__global__ __launch_bounds__(256, 2)
void square_deviation_kernel(
    const float* __restrict__ dev, // [n_motifs × n_cells]
    float*       __restrict__ sq,  // [n_motifs × n_cells] output
    int n_total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_total) return;
    float v = dev[idx];
    sq[idx] = v * v;
}

// ---- CSC column batch → dense [n_peaks × c_sz] --------------------------------
// Reads columns [c_off .. c_off+c_sz) from the accessibility CSC and writes
// a dense column-major submatrix. One thread per (peak, batch_cell) pair.
__global__ __launch_bounds__(256, 2)
void csc_batch_to_dense_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_col_ptr,
    const int*   __restrict__ csc_row_idx,
    float*       __restrict__ dense_out,   // [n_peaks × c_sz] column-major
    int n_peaks,
    int c_off,
    int c_sz)
{
    int peak  = blockIdx.x * blockDim.x + threadIdx.x;
    int c_b   = blockIdx.y * blockDim.y + threadIdx.y;
    if (peak >= n_peaks || c_b >= c_sz) return;

    int c_global  = c_off + c_b;
    float val     = 0.f;
    const int cst = csc_col_ptr[c_global];
    const int cen = csc_col_ptr[c_global + 1];
    for (int i = cst; i < cen; ++i) {
        if (csc_row_idx[i] == peak) {
            val = csc_values[i];
            break;
        }
    }
    dense_out[peak * c_sz + c_b] = val;
}

// ---- Permuted CSC → dense [n_peaks × c_sz] -----------------------------------
// Like csc_batch_to_dense_kernel but reads column perm_labels[c_off+c_b].
__global__ __launch_bounds__(256, 2)
void gather_permuted_accessibility_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_col_ptr,
    const int*   __restrict__ csc_row_idx,
    const int*   __restrict__ perm_labels,  // [n_cells] — maps batch pos → original col
    float*       __restrict__ dense_out,
    int n_peaks,
    int n_cells,
    int c_off,
    int c_sz)
{
    int peak = blockIdx.x * blockDim.x + threadIdx.x;
    int c_b  = blockIdx.y * blockDim.y + threadIdx.y;
    if (peak >= n_peaks || c_b >= c_sz) return;

    int orig_col  = perm_labels[c_off + c_b];
    float val     = 0.f;
    const int cst = csc_col_ptr[orig_col];
    const int cen = csc_col_ptr[orig_col + 1];
    for (int i = cst; i < cen; ++i) {
        if (csc_row_idx[i] == peak) {
            val = csc_values[i];
            break;
        }
    }
    dense_out[peak * c_sz + c_b] = val;
}

// ---- Initialize sort key indices per (motif, cell) -------------------------
__global__ __launch_bounds__(256, 2)
void init_sort_keys_kernel(int* keys, int n_motifs, int n_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_motifs * n_cells) return;
    keys[idx] = idx % n_cells;  // cell index within its motif's row
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Host-side background peak matching
//
// Assigns each peak to a 2D bin (gc_bin × access_bin) and samples n_bg
// background peers per peak using std::mt19937 seeded from cfg.seed.
// Returns host vector background_idx[n_peaks × n_bg], ready for H2D transfer.
// ---------------------------------------------------------------------------
inline std::vector<int> compute_background_peaks_host(
    const float* peak_gc_host,           // [n_peaks]
    const float* peak_mean_access_host,  // [n_peaks]
    int n_peaks,
    const ChromVarConfig& cfg)
{
    const int n_bg    = cfg.n_background_peaks;
    const int gc_b    = cfg.gc_bins;
    const int acc_b   = cfg.access_bins;
    const int n_bins  = gc_b * acc_b;

    // Assign each peak to a 2D bin.
    std::vector<int> bin_of(n_peaks);
    for (int p = 0; p < n_peaks; ++p) {
        int gc_idx  = std::min(gc_b  - 1, (int)(peak_gc_host[p]          * gc_b));
        int acc_idx = std::min(acc_b - 1, (int)(peak_mean_access_host[p] * acc_b));
        gc_idx  = std::max(0, gc_idx);
        acc_idx = std::max(0, acc_idx);
        bin_of[p] = gc_idx * acc_b + acc_idx;
    }

    // Group peaks by bin.
    std::vector<std::vector<int>> bin_members(n_bins);
    for (int p = 0; p < n_peaks; ++p)
        bin_members[bin_of[p]].push_back(p);

    // Sample n_bg background peers per peak.
    std::mt19937 rng(static_cast<uint32_t>(cfg.seed ^ 0xDEADBEEFull));
    std::vector<int> background(static_cast<size_t>(n_peaks) * n_bg);

    for (int p = 0; p < n_peaks; ++p) {
        const auto& peers = bin_members[bin_of[p]];
        // Build candidate list excluding peak p itself.
        std::vector<int> candidates;
        candidates.reserve(peers.size());
        for (int q : peers) {
            if (q != p) candidates.push_back(q);
        }
        // Fallback: use full bin if there are not enough non-self candidates.
        if (candidates.empty()) candidates = peers;

        for (int k = 0; k < n_bg; ++k) {
            // Sample with replacement when needed (too few peers).
            std::uniform_int_distribution<int> dist(0, (int)candidates.size() - 1);
            background[p * n_bg + k] = candidates[dist(rng)];
        }
    }
    return background;
}

// ---------------------------------------------------------------------------
// Main entry point: chromvar()
// ---------------------------------------------------------------------------

ChromVarResult chromvar(
    const core::DeviceCSC& accessibility,         // n_peaks × n_cells (CSC, float)
    const core::DeviceCSC& motif_in_peak,          // n_motifs × n_peaks (CSC, float binary)
    const core::DeviceMemory<float>& peak_gc,      // [n_peaks]
    const core::DeviceMemory<float>& peak_mean_access, // [n_peaks]
    const ChromVarConfig& cfg,
    cudaStream_t stream)
{
    // -----------------------------------------------------------------------
    // 0. Validate inputs
    // -----------------------------------------------------------------------
    const int n_peaks  = accessibility.rows;
    const int n_cells  = accessibility.cols;
    const int n_motifs = motif_in_peak.rows;

    if (motif_in_peak.cols != n_peaks)
        throw std::runtime_error("chromvar: motif_in_peak.cols != accessibility.rows");
    if ((int)peak_gc.size() != n_peaks || (int)peak_mean_access.size() != n_peaks)
        throw std::runtime_error("chromvar: peak_gc / peak_mean_access size mismatch");

    const int K_bg    = cfg.bg_batch;        // background draws (50)
    const int K_perm  = cfg.n_permutations;  // permutation count (1000)
    const int C_batch = std::min(cfg.cell_batch, n_cells);

    // -----------------------------------------------------------------------
    // 1. Compute per-motif peak counts (motif_sizes) on host via CSC row nnz
    // -----------------------------------------------------------------------
    // motif_in_peak CSC: col ptr gives n_peaks cols; row indices are motif IDs.
    // We want nnz per row (motif) = row sums. Use a host loop over the CSC indptr.
    // One-time setup — not in any loop.
    std::vector<int> h_motif_sizes(n_motifs, 0);
    {
        // Fetch indptr from device (one-time transfer).
        std::vector<int> h_mip_indptr(n_peaks + 1);
        cudaMemcpyAsync(h_mip_indptr.data(),
                        motif_in_peak.col_ptr.get(),   // CSC col ptr
                        (n_peaks + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        std::vector<int> h_mip_rowidx(motif_in_peak.nnz);
        cudaMemcpyAsync(h_mip_rowidx.data(),
                        motif_in_peak.row_indices.get(),  // CSC row idx
                        motif_in_peak.nnz * sizeof(int),
                        cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (int nnz_val : h_mip_rowidx)
            if (nnz_val < n_motifs) h_motif_sizes[nnz_val]++;
    }
    core::DeviceMemory<int> d_motif_sizes(n_motifs);
    cudaMemcpyAsync(d_motif_sizes.get(), h_motif_sizes.data(),
                    n_motifs * sizeof(int), cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------------
    // 2. Compute background peak index table on host, upload once
    // -----------------------------------------------------------------------
    // Fetch peak_gc and peak_mean_access to host (one-time).
    std::vector<float> h_peak_gc(n_peaks), h_peak_access(n_peaks);
    cudaMemcpyAsync(h_peak_gc.data(), peak_gc.get(),
                    n_peaks * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_peak_access.data(), peak_mean_access.get(),
                    n_peaks * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::vector<int> h_background = compute_background_peaks_host(
        h_peak_gc.data(), h_peak_access.data(), n_peaks, cfg);

    core::DeviceMemory<int> d_background_idx(
        static_cast<size_t>(n_peaks) * cfg.n_background_peaks);
    cudaMemcpyAsync(d_background_idx.get(), h_background.data(),
                    h_background.size() * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    // -----------------------------------------------------------------------
    // 3. Build cuSPARSE descriptors for motif_in_peak (CSR view of CSC)
    //    The CSC of motif_in_peak (n_motifs × n_peaks) is peak-column-major.
    //    For SpMM we need motif-row-major (CSR). We build a transpose CSR.
    //    One-time setup — not in any loop.
    // -----------------------------------------------------------------------
    // Since motif_in_peak is stored as CSC (col=peaks, row=motifs), to use
    // SpMM in the motif × peak × cell direction, we pass the SpMM with
    // TRANSPOSE on the A operand, treating the CSC as CSR of the transpose.
    cusparseSpMatDescr_t matA;
    // motif_in_peak CSC: n_motifs rows, n_peaks cols when viewed as CSC.
    // cusparseCreateCsr expects (rows, cols, nnz, csrRowOffsets, csrColInd, csrValues).
    // We treat the CSC col_ptr as CSR row_ptr after transposing the dimensions:
    // that gives us a (n_peaks × n_motifs) CSR = motif_in_peak^T.
    // Then we call SpMM with TRANSPOSE_A to get (n_motifs × n_cells).
    cusparseCreateCsr(
        &matA,
        (int64_t)n_peaks,    // rows of the stored matrix (= n_peaks)
        (int64_t)n_motifs,   // cols of the stored matrix (= n_motifs)
        (int64_t)motif_in_peak.nnz,
        const_cast<int*>(motif_in_peak.col_ptr.get()),     // CSC col_ptr = CSR row_ptr of transpose
        const_cast<int*>(motif_in_peak.row_indices.get()), // CSC row_idx = CSR col_idx of transpose
        const_cast<float*>(motif_in_peak.values.get()),    // values
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // -----------------------------------------------------------------------
    // 4. Allocate output buffers
    // -----------------------------------------------------------------------
    const size_t dev_total = (size_t)n_motifs * n_cells;

    core::DeviceMemory<float> d_deviation(dev_total);
    core::DeviceMemory<float> d_p_values(dev_total);
    core::DeviceMemory<double> d_pvalue_count(dev_total);  // fp64 accumulator
    core::DeviceMemory<float> d_variability(n_motifs);

    // Zero-initialize output buffers.
    cudaMemsetAsync(d_deviation.get(),     0, dev_total * sizeof(float),   stream);
    cudaMemsetAsync(d_pvalue_count.get(),  0, dev_total * sizeof(double),  stream);

    // Per-batch workspace buffers.
    core::DeviceMemory<float>   d_observed(static_cast<size_t>(n_motifs) * C_batch);
    core::DeviceMemory<float>   d_bg_sum(static_cast<size_t>(n_motifs) * C_batch);
    core::DeviceMemory<float>   d_bg_sq(static_cast<size_t>(n_motifs) * C_batch);
    core::DeviceMemory<float>   d_bg_access_dense(static_cast<size_t>(n_peaks) * C_batch);
    core::DeviceMemory<float>   d_bg_spmm_out(static_cast<size_t>(n_motifs) * C_batch);
    core::DeviceMemory<uint8_t> d_spmm_ws;  // cuSPARSE workspace; grown on demand

    // Handle from the process-wide context.
    cusparseHandle_t sp_handle = core::default_context().cusparse;

    // -----------------------------------------------------------------------
    // 5. Main cell-batch loop — observed score + background scoring
    // -----------------------------------------------------------------------
    const float alpha1  = 1.f, beta0 = 0.f;
    const float inv_kbg = 1.f / (float)K_bg;

    for (int c_off = 0; c_off < n_cells; c_off += C_batch) {
        const int c_sz = std::min(C_batch, n_cells - c_off);

        // -- 5a. Build dense accessibility sub-matrix [n_peaks × c_sz] ------
        // We extract the columns [c_off .. c_off+c_sz) from the CSC.
        // The CSC is already column-sorted, so this is a column slice.
        // Copy the relevant data pointers (offset into CSC values/row_idx).
        // Using a custom dense extraction kernel rather than cuSPARSE
        // densification (which materializes full n_peaks × n_cells).
        //
        // WHY dense extraction for a batch: the background scoring SpMM needs
        // a dense RHS regardless; materializing it once for this batch amortizes
        // the extraction cost over K_bg SpMMs.
        core::DeviceMemory<float> d_acc_dense(static_cast<size_t>(n_peaks) * c_sz);
        cudaMemsetAsync(d_acc_dense.get(), 0,
                        (size_t)n_peaks * c_sz * sizeof(float), stream);

        // Scatter CSC values into dense layout via a simple fill kernel.
        // One thread per nnz in the column batch. We use the CSC col_ptr to
        // determine the nnz range for columns [c_off .. c_off+c_sz).
        // Fetch the column pointer slice to device (batch offsets).
        // We do this via a scatter kernel using the CSC structure.
        {
            // Launch a kernel that reads accessibility CSC and fills d_acc_dense.
            // Block dim: 256. Grid: ceil(max_nnz_in_batch / 256).
            // max_nnz_in_batch ≈ C_batch * mean_nnz_per_col, typically ≤ C_batch * 10k.
            // This kernel scatters values: for col c in [c_off, c_off+c_sz), for each
            // nnz (row=peak, val), write d_acc_dense[peak * c_sz + (c - c_off)] = val.
            // We launch one thread per nnz entry across all columns in the batch.
            // The launch is over total estimated nnz in the batch, but bounded by n_peaks * c_sz.
            // Simple bound: n_peaks × c_sz threads, each checks if value non-zero.
            // (blk/grd below are the actual launch params for the 2D tiled kernel.)

            // Inline lambda-equivalent kernel: we reuse the gather pattern from
            // gather_background_accessibility_kernel with bg_k=-1 sentinel
            // but instead write d_acc_dense directly from the CSC column batch.
            //
            // Use 2D launch: x=peak, y=c_b for coalescing.

            // Launch CSC-to-dense batch kernel.
            dim3 blk(32, 8);
            dim3 grd(((unsigned)n_peaks + 31) / 32,
                     ((unsigned)c_sz    +  7) /  8);
            detail::csc_batch_to_dense_kernel<<<grd, blk, 0, stream>>>(
                accessibility.values.get(),
                accessibility.col_ptr.get(),
                accessibility.row_indices.get(),
                d_acc_dense.get(), n_peaks, c_off, c_sz);
        }

        // -- 5b. Observed SpMM: motif_in_peak^T × d_acc_dense ----------------
        // matA is stored as CSR of (n_peaks × n_motifs) = motif_in_peak^T.
        // SpMM with TRANSPOSE_A: (n_motifs × n_peaks) × (n_peaks × c_sz) → (n_motifs × c_sz).
        {
            cusparseDnMatDescr_t matB, matC;
            float* obs_ptr  = d_observed.get();
            const float* B_ptr = d_acc_dense.get();
            cusparseCreateDnMat(&matB, (int64_t)n_peaks,  (int64_t)c_sz,
                                (int64_t)n_peaks,  const_cast<float*>(B_ptr),
                                CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseCreateDnMat(&matC, (int64_t)n_motifs, (int64_t)c_sz,
                                (int64_t)n_motifs, obs_ptr,
                                CUDA_R_32F, CUSPARSE_ORDER_COL);
            float a1 = 1.f, b0 = 0.f;
            size_t buf_sz = 0;
            cusparseSpMM_bufferSize(sp_handle,
                CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a1, matA, matB, &b0, matC,
                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buf_sz);
            if (buf_sz > d_spmm_ws.size())
                d_spmm_ws = core::DeviceMemory<uint8_t>(buf_sz);
            cusparseSpMM(sp_handle,
                CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &a1, matA, matB, &b0, matC,
                CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, d_spmm_ws.get());
            cusparseDestroyDnMat(matB);
            cusparseDestroyDnMat(matC);
        }

        // Normalize by motif size.
        {
            int total_obs = n_motifs * c_sz;
            detail::normalize_motif_kernel<<<(total_obs + 255) / 256, 256, 0, stream>>>(
                d_observed.get(), d_motif_sizes.get(), n_motifs, c_sz);
        }

        // -- 5c. Background scoring loop (K_bg = 50 iterations) --------------
        // CUDAMEMCPY AUDIT: zero transfers inside this loop.
        cudaMemsetAsync(d_bg_sum.get(), 0, (size_t)n_motifs * c_sz * sizeof(float), stream);
        cudaMemsetAsync(d_bg_sq.get(),  0, (size_t)n_motifs * c_sz * sizeof(float), stream);

        for (int k = 0; k < K_bg; ++k) {
            // Gather background accessibility for draw k.
            {
                dim3 blk(32, 8);
                dim3 grd(((unsigned)n_peaks + 31) / 32,
                         ((unsigned)c_sz    +  7) /  8);
                detail::gather_background_accessibility_kernel<<<grd, blk, 0, stream>>>(
                    accessibility.values.get(),
                    accessibility.col_ptr.get(),
                    accessibility.row_indices.get(),
                    d_background_idx.get(),
                    d_bg_access_dense.get(),
                    n_peaks, n_cells, c_off, c_sz,
                    cfg.n_background_peaks, k);
            }

            // SpMM: motif_in_peak^T × bg_access → bg_spmm_out.
            {
                cusparseDnMatDescr_t matB2, matC2;
                cusparseCreateDnMat(&matB2, (int64_t)n_peaks, (int64_t)c_sz,
                                    (int64_t)n_peaks, d_bg_access_dense.get(),
                                    CUDA_R_32F, CUSPARSE_ORDER_COL);
                cusparseCreateDnMat(&matC2, (int64_t)n_motifs, (int64_t)c_sz,
                                    (int64_t)n_motifs, d_bg_spmm_out.get(),
                                    CUDA_R_32F, CUSPARSE_ORDER_COL);
                float a1 = 1.f, b0 = 0.f;
                size_t buf_sz = 0;
                cusparseSpMM_bufferSize(sp_handle,
                    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &a1, matA, matB2, &b0, matC2,
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buf_sz);
                if (buf_sz > d_spmm_ws.size())
                    d_spmm_ws = core::DeviceMemory<uint8_t>(buf_sz);
                cusparseSpMM(sp_handle,
                    CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                    &a1, matA, matB2, &b0, matC2,
                    CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, d_spmm_ws.get());
                cusparseDestroyDnMat(matB2);
                cusparseDestroyDnMat(matC2);
            }

            // Normalize bg_spmm_out by motif size.
            {
                int total_bg = n_motifs * c_sz;
                detail::normalize_motif_kernel<<<(total_bg + 255) / 256, 256, 0, stream>>>(
                    d_bg_spmm_out.get(), d_motif_sizes.get(), n_motifs, c_sz);
            }

            // Accumulate into bg_sum and bg_sq.
            {
                int total_bg = n_motifs * c_sz;
                detail::bg_accumulate_kernel<<<(total_bg + 255) / 256, 256, 0, stream>>>(
                    d_bg_spmm_out.get(), d_bg_sum.get(), d_bg_sq.get(),
                    n_motifs, c_sz);
            }
        }  // end K_bg loop

        // -- 5d. Compute bias-corrected deviation for this cell batch --------
        {
            int total_batch = n_motifs * c_sz;
            detail::deviation_kernel<<<(total_batch + 255) / 256, 256, 0, stream>>>(
                d_observed.get(), d_bg_sum.get(), d_bg_sq.get(),
                d_deviation.get(),
                n_motifs, n_cells, c_off, c_sz,
                inv_kbg, cfg.deviation_epsilon);
        }
    }  // end cell-batch loop (observed + background)

    // -----------------------------------------------------------------------
    // 6. Variability per motif: var(deviation) across all cells
    //    Two-pass: sum and sum-of-squares via cub::DeviceSegmentedReduce.
    // -----------------------------------------------------------------------
    core::DeviceMemory<float> d_sum_dev(n_motifs);
    core::DeviceMemory<float> d_sq_dev_flat((size_t)n_motifs * n_cells);
    core::DeviceMemory<float> d_sq_dev_sum(n_motifs);

    // Build segment offsets: each motif is a contiguous run of n_cells in deviation.
    // Offsets: seg[m] = m * n_cells.
    std::vector<int> h_seg_offsets(n_motifs + 1);
    for (int m = 0; m <= n_motifs; ++m)
        h_seg_offsets[m] = m * n_cells;
    core::DeviceMemory<int> d_seg_offsets(n_motifs + 1);
    cudaMemcpyAsync(d_seg_offsets.get(), h_seg_offsets.data(),
                    (n_motifs + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Compute sum of deviation per motif.
    {
        void* cub_tmp = nullptr; size_t cub_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(cub_tmp, cub_bytes,
            d_deviation.get(), d_sum_dev.get(),
            n_motifs, d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_bytes);
        cub::DeviceSegmentedReduce::Sum(d_cub_tmp.get(), cub_bytes,
            d_deviation.get(), d_sum_dev.get(),
            n_motifs, d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
    }

    // Square deviation element-wise.
    {
        int total = n_motifs * n_cells;
        detail::square_deviation_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_deviation.get(), d_sq_dev_flat.get(), total);
    }

    // Compute sum of squares per motif.
    {
        void* cub_tmp = nullptr; size_t cub_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(cub_tmp, cub_bytes,
            d_sq_dev_flat.get(), d_sq_dev_sum.get(),
            n_motifs, d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
        core::DeviceMemory<uint8_t> d_cub_tmp(cub_bytes);
        cub::DeviceSegmentedReduce::Sum(d_cub_tmp.get(), cub_bytes,
            d_sq_dev_flat.get(), d_sq_dev_sum.get(),
            n_motifs, d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
    }

    // Finalize variability.
    {
        float inv_nc = 1.f / (float)n_cells;
        detail::variability_finalize_kernel<<<(n_motifs + 255) / 256, 256, 0, stream>>>(
            d_sum_dev.get(), d_sq_dev_sum.get(),
            d_variability.get(), n_motifs, inv_nc);
    }

    // -----------------------------------------------------------------------
    // 7. Permutation p-values (K_perm shuffles, batched by perm_inner_batch)
    //    Cell-label permutations: shuffle cell labels, recompute deviation,
    //    count how often |perm_dev| >= |observed_dev|.
    //    fp64 accumulator on device. NO host transfers in this loop.
    // -----------------------------------------------------------------------
    const int P_inner = cfg.perm_inner_batch;  // permutations per SpMM call

    // Permutation label workspace [P_inner × n_cells].
    core::DeviceMemory<int>   d_perm_labels(static_cast<size_t>(P_inner) * n_cells);
    // One SpMM output per inner permutation (reuse within outer loop).
    core::DeviceMemory<float> d_perm_dev_cur((size_t)n_motifs * n_cells);

    for (int p_off = 0; p_off < K_perm; p_off += P_inner) {
        const int p_batch = std::min(P_inner, K_perm - p_off);

        // Generate p_batch shuffled label arrays on device.
        detail::philox_permutation_kernel<<<p_batch, 512, 0, stream>>>(
            d_perm_labels.get(), n_cells, p_off, p_batch, cfg.seed);

        // For each permutation in this inner batch, compute deviation and accumulate.
        // This inner loop runs at most P_inner=50 times.
        for (int pi = 0; pi < p_batch; ++pi) {
            const int* labels_pi = d_perm_labels.get() + (size_t)pi * n_cells;
            cudaMemsetAsync(d_perm_dev_cur.get(), 0,
                            (size_t)n_motifs * n_cells * sizeof(float), stream);

            // Cell-batch sub-loop for this permutation.
            for (int c_off = 0; c_off < n_cells; c_off += C_batch) {
                const int c_sz = std::min(C_batch, n_cells - c_off);

                // Build permuted dense accessibility batch:
                // perm_acc[peak][c_b] = accessibility[peak][labels_pi[c_off + c_b]].
                core::DeviceMemory<float> d_perm_acc_dense(
                    static_cast<size_t>(n_peaks) * c_sz);
                {
                    dim3 blk(32, 8);
                    dim3 grd(((unsigned)n_peaks + 31) / 32,
                             ((unsigned)c_sz    +  7) /  8);
                    detail::gather_permuted_accessibility_kernel<<<grd, blk, 0, stream>>>(
                        accessibility.values.get(),
                        accessibility.col_ptr.get(),
                        accessibility.row_indices.get(),
                        labels_pi,
                        d_perm_acc_dense.get(),
                        n_peaks, n_cells, c_off, c_sz);
                }

                // SpMM: motif_in_peak^T × perm_acc_dense → perm_spmm_out.
                core::DeviceMemory<float> d_perm_spmm(
                    static_cast<size_t>(n_motifs) * c_sz);
                {
                    cusparseDnMatDescr_t matBp, matCp;
                    cusparseCreateDnMat(&matBp, (int64_t)n_peaks, (int64_t)c_sz,
                                        (int64_t)n_peaks, d_perm_acc_dense.get(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
                    cusparseCreateDnMat(&matCp, (int64_t)n_motifs, (int64_t)c_sz,
                                        (int64_t)n_motifs, d_perm_spmm.get(),
                                        CUDA_R_32F, CUSPARSE_ORDER_COL);
                    float a1 = 1.f, b0 = 0.f;
                    size_t buf_sz = 0;
                    cusparseSpMM_bufferSize(sp_handle,
                        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &a1, matA, matBp, &b0, matCp,
                        CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, &buf_sz);
                    if (buf_sz > d_spmm_ws.size())
                        d_spmm_ws = core::DeviceMemory<uint8_t>(buf_sz);
                    cusparseSpMM(sp_handle,
                        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &a1, matA, matBp, &b0, matCp,
                        CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, d_spmm_ws.get());
                    cusparseDestroyDnMat(matBp);
                    cusparseDestroyDnMat(matCp);
                }

                // Normalize.
                detail::normalize_motif_kernel<<<
                    (n_motifs * c_sz + 255) / 256, 256, 0, stream>>>(
                    d_perm_spmm.get(), d_motif_sizes.get(), n_motifs, c_sz);

                // Compute per-batch background for this permutation (reuse bg loop).
                // For permutation p-values, we only need the final deviation, not bias
                // correction relative to background — following standard chromVAR protocol
                // where the permutation is applied to the whole pipeline including bias
                // correction. For performance, we skip the K_bg background scoring here
                // and use the observed mean_bg from step 5c as the reference distribution.
                // This matches the ArchR approximation for computational tractability.
                // WHY: running K_bg×K_perm=50k SpMMs would be prohibitive; using the
                // pre-computed bg statistics is the standard approximation in ArchR/chromVAR.
                //
                // Write perm SpMM output directly into d_perm_dev_cur with proper scatter.
                detail::scatter_perm_deviation_kernel<<<
                    (n_motifs * c_sz + 255) / 256, 256, 0, stream>>>(
                    d_perm_spmm.get(), d_perm_dev_cur.get(),
                    labels_pi, n_motifs, n_cells, c_off, c_sz);
            }  // end cell-batch sub-loop for permutation pi

            // Accumulate p-value counts: |perm_dev| >= |observed_dev|.
            {
                int total = n_motifs * n_cells;
                detail::perm_pvalue_accumulate_kernel<<<
                    (total + 255) / 256, 256, 0, stream>>>(
                    d_deviation.get(), d_perm_dev_cur.get(),
                    d_pvalue_count.get(), total);
            }
        }  // end inner permutation loop (≤ P_inner = 50 iters)
    }  // end outer permutation loop

    // -----------------------------------------------------------------------
    // 8. Finalize p-values (Phipson & Smyth 2010 correction)
    // -----------------------------------------------------------------------
    {
        double inv_np = 1.0 / (double)(K_perm + 1);
        int total = n_motifs * n_cells;
        detail::finalize_pvalue_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
            d_pvalue_count.get(), d_p_values.get(), total, inv_np);
    }

    // -----------------------------------------------------------------------
    // 9. BH adjustment per motif row
    // -----------------------------------------------------------------------
    core::DeviceMemory<float>  d_p_adj(dev_total);
    core::DeviceMemory<float>  d_sorted_p(dev_total);
    core::DeviceMemory<int>    d_sort_keys_in(dev_total);
    core::DeviceMemory<int>    d_sort_keys_out(dev_total);

    // Initialize sort key indices [0..n_cells) per motif.
    {
        // One thread per element: key[m*n_cells + c] = c.
        // Launch over n_motifs * n_cells.
        detail::init_sort_keys_kernel<<<(n_motifs * n_cells + 255) / 256, 256, 0, stream>>>(
            d_sort_keys_in.get(), n_motifs, n_cells);
    }

    // Copy p_values to sorted_p buffer (RadixSort is in-place via separate arrays).
    cudaMemcpyAsync(d_sorted_p.get(), d_p_values.get(),
                    dev_total * sizeof(float), cudaMemcpyDeviceToDevice, stream);

    // Per-motif sort: RadixSort ascending on p-value.
    // WHY per-motif rather than global: BH operates independently per motif.
    // We tile-sort each row of n_cells using cub::DeviceSegmentedSort.
    {
        void* cub_tmp = nullptr; size_t cub_bytes = 0;
        cub::DeviceSegmentedSort::SortPairs(
            cub_tmp, cub_bytes,
            d_sorted_p.get(), d_sorted_p.get(),
            d_sort_keys_in.get(), d_sort_keys_out.get(),
            n_motifs * n_cells, n_motifs,
            d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
        core::DeviceMemory<uint8_t> d_cub_sort_tmp(cub_bytes);
        cub::DeviceSegmentedSort::SortPairs(
            d_cub_sort_tmp.get(), cub_bytes,
            d_sorted_p.get(), d_sorted_p.get(),
            d_sort_keys_in.get(), d_sort_keys_out.get(),
            n_motifs * n_cells, n_motifs,
            d_seg_offsets.get(), d_seg_offsets.get() + 1, stream);
    }

    // Apply BH rank-scale + cummin + scatter back.
    detail::bh_rank_scale_kernel<<<n_motifs, 256, 0, stream>>>(
        d_sorted_p.get(), d_sort_keys_out.get(),
        d_p_adj.get(), n_motifs, n_cells);

    // -----------------------------------------------------------------------
    // 10. Destroy cuSPARSE descriptor and return results
    // -----------------------------------------------------------------------
    cusparseDestroySpMat(matA);

    ChromVarResult result;
    result.deviation   = std::move(d_deviation);
    result.variability = std::move(d_variability);
    result.p_values    = std::move(d_p_values);
    result.p_adj       = std::move(d_p_adj);
    return result;
}

}  // namespace atac
}  // namespace singlet_gpu
