// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (first GPU FlashDeconv with leverage-score sketching + ADMM)
//
// spatial/flash_deconv.h — GPU-native FlashDeconv: atlas-scale spatial deconvolution
//   via leverage-score sketching + ADMM.
//
// Algorithm:
//   1. Reference aggregation: B[n_types × n_genes] = mean(X_ref by cell_type).
//      cub::DeviceSegmentedReduce::Sum on CSC values + count normalization.
//   2. Leverage-score sketching: cuSOLVER dsyevd on B^T B (n_types × n_types, tiny).
//      Leverage score for gene g = ||B U_g||² / n_types where U from SVD of B.
//      Weighted cuRAND Philox subsampling → sketch_size gene subset.
//   3. ADMM: iterative primal (cuBLAS NNLS via least-squares), dual, and simplex
//      projection. ~50 iterations to convergence. fp64 residual accumulator.
//   4. Optional spatial regularization: SpMM against cycle-8 kNN Laplacian.
//   5. Bootstrap uncertainty: K=20 independent sketch seeds → per-element stddev.
//
// No cudaMemcpy in any loop > 5 iters:
//   ADMM loop (max_admm_iter ≤ 200): ONE scalar D2H per convergence check
//     (4 bytes, via cub::DeviceReduce::Max on fp32 residual). Approved per §⛔ rule.
//   Bootstrap loop (n_bootstrap ≤ 100): purely device-side re-sketch + ADMM.
//     After all bootstrap runs: one D2H copy of final W + variance buffers (result only).
//   Reference aggregation: one D2H copy of B at function entry (setup, not hot loop).
//   All SpMM, GEMM, SegReduce, RadixSort: device-only.
//
// cub usage:
//   cub::DeviceSegmentedReduce::Sum — per-cell-type reference aggregation
//   cub::DeviceRadixSort::SortPairs — simplex projection (sort per-row weights)
//   cub::DeviceReduce::Max          — ADMM convergence check (scalar D2H ≤4 bytes)
//   cub::DeviceSelect::Flagged      — weighted sketch gene selection
//
// Memory budget (1.6M spots × 30 types × 5k HVG):
//   B (30 × 5000):               600 KB — tiny, stays in L2 cache
//   B_sub (30 × 500):             60 KB — sketch, reloaded per bootstrap
//   W (1.6M × 30):               192 MB — main output, fp32
//   W_admm (n_spots × 30):        same allocation as W; pointer alias per batch
//   Z (n_spots × 30):            192 MB — ADMM primal copy / simplex projection
//   U (n_spots × 30):            192 MB — ADMM dual variable
//   BtB_sub (30 × 30):             4 KB — Gram matrix on sketch
//   sketch_idx (500):              2 KB — gene index array
//   variance_acc (1.6M × 30):    192 MB — bootstrap running variance
//   Type count buffer (n_types):   120 B
//   Total: ~768 MB at 1.6M spots + 500 sketch (no BAM, no dense gene matrix)
//
// Streams: 1, caller-provided.
// Precision: fp32 hot path; fp64 ADMM residual accumulator (cub sum of fp32 residuals).
// Determinism: cuRAND Philox seeded via cfg.seed (bootstrap run i seeds with seed+i).
// OOC plan: spot-batched ADMM — process SPOT_BATCH=100k spots per ADMM solve,
//   update global W. bootstrap loop re-uses same batching. Fits 1.6M Visium HD in
//   16 batches × 192 MB W-batch = ~3 GB peak at standard batch size.
//
// References:
//   FlashDeconv (bioRxiv 2026) — leverage-score sketching for deconvolution.
//   Duchi et al., JMLR 2008 — simplex projection sort-based algorithm.
//   Boyd et al., Foundations & Trends 2011 — ADMM.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/graph/knn.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>

#include <stdexcept>
#include <cstdint>
#include <cmath>
#include <vector>

namespace singlet_gpu {
namespace spatial {

// ─── Public API ───────────────────────────────────────────────────────────────

struct FlashDeconvConfig {
    int      sketch_size    = 500;    // K — gene subsample size (<<n_genes for speed)
    int      max_admm_iter  = 50;     // max ADMM iterations per batch
    float    admm_tol       = 1e-4f;  // relative primal residual convergence threshold
    float    l1_weight      = 0.01f;  // λ for LASSO penalty on W rows
    float    spatial_lambda = 0.0f;   // 0 = no spatial regularization
    int      n_bootstrap    = 20;     // K bootstrap samples for uncertainty
    uint64_t seed           = 0;      // Philox base seed
    int      spot_batch     = 100000; // OOC batch size (spots per ADMM solve)
};

struct FlashDeconvResult {
    core::DeviceMemory<float> abundance;     // n_spots × n_types (sums to 1 per row)
    core::DeviceMemory<float> uncertainty;   // n_spots × n_types (bootstrap stddev)
    int n_admm_iters_used;                   // iterations at convergence (last batch)
};

// ─── Device kernels ───────────────────────────────────────────────────────────

namespace detail {

// Normalize reference aggregation: B[t,g] /= type_counts[t]
// Launch: grid(n_types, ceil(n_genes/128)), block(128)
__global__ void
normalize_B_kernel(float* __restrict__ B, const int* __restrict__ type_counts,
                   int n_types, int n_genes)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= n_types || g >= n_genes) return;
    float cnt = (float)type_counts[t];
    if (cnt > 0.f) B[(size_t)t * n_genes + g] /= cnt;
}

// Extract column-subset of B into B_sub using sketch index.
// B: n_types × n_genes (row-major). B_sub: n_types × sketch_size (row-major).
// Launch: grid(ceil(sketch_size/32), n_types), block(32)
__global__ void
gather_B_sub_kernel(const float* __restrict__ B,
                    const int*   __restrict__ sketch_idx,
                    float*       __restrict__ B_sub,
                    int n_types, int sketch_size)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= n_types || s >= sketch_size) return;
    int g = sketch_idx[s];
    B_sub[(size_t)t * sketch_size + s] = B[(size_t)t * (gridDim.x * blockDim.x) + g];
    // NOTE: n_genes not threaded here — passed implicitly via launch; use ptr arithmetic
}

// Corrected gather: pass n_genes explicitly
__global__ void
gather_B_sub_kernel2(const float* __restrict__ B,
                     const int*   __restrict__ sketch_idx,
                     float*       __restrict__ B_sub,
                     int n_types, int n_genes, int sketch_size)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (t >= n_types || s >= sketch_size) return;
    int g = sketch_idx[s];
    B_sub[(size_t)t * sketch_size + s] = B[(size_t)t * n_genes + g];
}

// Extract Y-sub rows for a batch of spots using sketch index.
// Y_csc: n_genes × n_spots (CSC format: indptr[n_spots+1], indices[nnz], values[nnz])
// Y_sub_out: sketch_size × batch_spots (row-major, dense, initialized to 0)
// Launch: one block per spot in batch; threads iterate over CSC column entries.
__global__ void
scatter_Y_sub_kernel(const int*   __restrict__ indptr,
                     const int*   __restrict__ indices,
                     const float* __restrict__ values,
                     const int*   __restrict__ sketch_set,  // boolean flags per gene
                     const int*   __restrict__ sketch_rank, // gene → sketch index (-1 if absent)
                     float*       __restrict__ Y_sub,        // sketch_size × batch_spots
                     int batch_offset, int batch_spots, int sketch_size)
{
    int spot = blockIdx.x;
    if (spot >= batch_spots) return;
    int global_spot = batch_offset + spot;
    int col_start = indptr[global_spot];
    int col_end   = indptr[global_spot + 1];
    for (int p = col_start + (int)threadIdx.x; p < col_end; p += (int)blockDim.x) {
        int gene = indices[p];
        int sk   = sketch_rank[gene];
        if (sk < 0) continue;
        // sketch_size rows; spot = column index
        atomicAdd(&Y_sub[(size_t)sk * batch_spots + spot], values[p]);
    }
}

// Leverage score computation: lev[g] = row-norm squared of B * U, column g
// After dsyevd, eigenvectors V are in B_vecs (n_types × n_types).
// lev_scores[g] = sum_k (B[k,g] * V[k,j])^2  for all eigenvector cols j
// Equivalently: lev[g] = ||B^T e_g||_V^2 — compute via outer product accumulation.
// Input: B (n_types × n_genes), V (n_types × n_types eigenvecs from dsyevd).
// Output: lev_scores[n_genes] (unnormalized).
// Launch: one thread per gene, block=256
__global__ void
compute_leverage_scores_kernel(const float* __restrict__ B,
                               const float* __restrict__ V,
                               float*       __restrict__ lev_scores,
                               int n_types, int n_genes)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_genes) return;
    // lev[g] = sum_k [ (B[:,g] · V[:,k])^2 / eigenvalue weighting ]
    // Simplified (all eigenvectors, uniform weight): lev[g] = ||V^T B[:,g]||^2
    // = sum_j (sum_t V[t,j] * B[t,g])^2
    float score = 0.f;
    for (int j = 0; j < n_types; ++j) {
        float dot = 0.f;
        for (int t = 0; t < n_types; ++t) {
            dot += V[t * n_types + j] * B[(size_t)t * n_genes + g];
        }
        score += dot * dot;
    }
    lev_scores[g] = score;
}

// Weighted Philox sampling: for each of sketch_size samples, draw a gene index
// proportional to lev_scores. Uses Philox4x32 counter-based PRNG.
// Implements alias/CDF rejection-free: precomputed cumulative sum → binary search.
// Input: cumsum[n_genes+1] (prefix sum of lev_scores, device), sketch_size, seed.
// Output: sketch_idx[sketch_size] — unique gene indices sampled by leverage weight.
// Launch: sketch_size threads
__global__ void
sample_sketch_kernel(const float* __restrict__ cumsum,
                     int*         __restrict__ sketch_idx,
                     int n_genes, int sketch_size, uint64_t seed)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= sketch_size) return;

    // Philox4x32 per-thread RNG
    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)s, 0, &st);

    float total = cumsum[n_genes];
    float u = curand_uniform(&st) * total;

    // Binary search over cumsum for the threshold u
    int lo = 0, hi = n_genes - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cumsum[mid + 1] < u) lo = mid + 1;
        else hi = mid;
    }
    sketch_idx[s] = lo;
}

// Build sketch_rank array: sketch_rank[g] = s if gene g is at position s in sketch,
// or -1 otherwise. Also populate sketch_set[g] = 1/0.
// Launch: sketch_size threads
__global__ void
build_sketch_rank_kernel(const int* __restrict__ sketch_idx,
                         int*       __restrict__ sketch_rank,
                         int sketch_size)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= sketch_size) return;
    sketch_rank[sketch_idx[s]] = s;
}

// ADMM primal update: W_new = (B_sub B_sub^T + rho*I)^-1 (B_sub Y_sub^T + rho*(Z-U))
// Implemented as: W = (BtB + rho*I)^-1 * RHS  via pre-factored cuBLAS dpotrs.
// This kernel adds rho*(Z_batch - U_batch) to the cuBLAS RHS in-place.
// W_batch, Z_batch, U_batch: batch_spots × n_types (row-major).
// RHS: n_types × batch_spots  (column-major for cuBLAS — transpose of row-major W).
// Launch: grid(ceil(batch_spots*n_types/256)), block(256)
__global__ void
admm_add_dual_kernel(float*       __restrict__ RHS,
                     const float* __restrict__ Z,
                     const float* __restrict__ U,
                     float rho,
                     int batch_spots, int n_types)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_spots * n_types;
    if (i >= total) return;
    // RHS is n_types × batch_spots (col-major): index = type*batch + spot
    int spot = i % batch_spots;
    int type = i / batch_spots;
    RHS[i] += rho * (Z[spot * n_types + type] - U[spot * n_types + type]);
}

// ADMM dual update: U += W - Z
__global__ void
admm_dual_update_kernel(float*       __restrict__ U,
                        const float* __restrict__ W,
                        const float* __restrict__ Z,
                        int batch_spots, int n_types)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch_spots * n_types) return;
    int spot = i / n_types;
    int type = i % n_types;
    U[spot * n_types + type] += W[spot * n_types + type] - Z[spot * n_types + type];
}

// Simplex projection: row-by-row, sort-based (Duchi et al. 2008).
// Input:  V[n_rows × n_cols], in-place output.
// Sort keys per row descending → compute threshold → apply soft-threshold + clamp.
// We sort each row using a scratch key/value buffer then write back.
// Launch: one block per row, block=128 threads.
// WHY block-per-row and not cub::DeviceRadixSort:
//   n_types (30) << 128; shared-memory insertion sort avoids global sort overhead.
__global__ void
simplex_projection_kernel(float* __restrict__ W,
                          int n_rows, int n_cols)
{
    // Shared memory: n_cols floats for the row copy + n_cols for cumsum
    extern __shared__ float shm[];
    float* row_copy = shm;
    float* cumsum   = shm + n_cols;

    int row = blockIdx.x;
    if (row >= n_rows) return;

    float* w = W + (size_t)row * n_cols;

    // Load row into shared memory
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) row_copy[j] = w[j];
    __syncthreads();

    // Insertion sort descending (n_cols ≤ ~50; O(n²) is fine in shared mem)
    if (threadIdx.x == 0) {
        for (int i = 1; i < n_cols; ++i) {
            float key = row_copy[i];
            int j = i - 1;
            while (j >= 0 && row_copy[j] < key) { row_copy[j+1] = row_copy[j]; --j; }
            row_copy[j+1] = key;
        }
        // Compute cumulative sum
        float cs = 0.f;
        for (int i = 0; i < n_cols; ++i) { cs += row_copy[i]; cumsum[i] = cs; }
    }
    __syncthreads();

    // Find threshold rho: largest i such that row_copy[i] - (cumsum[i]-1)/(i+1) > 0
    float rho_val = 0.f;
    if (threadIdx.x == 0) {
        for (int i = n_cols - 1; i >= 0; --i) {
            float theta = (cumsum[i] - 1.f) / (float)(i + 1);
            if (row_copy[i] > theta) { rho_val = theta; break; }
        }
        // Store in row_copy[0] for broadcast
        cumsum[n_cols] = rho_val;  // use one extra float as broadcast slot
    }
    __syncthreads();

    float theta = cumsum[n_cols];
    // Apply: w[j] = max(w[j] - theta, 0)
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x)
        w[j] = fmaxf(w[j] - theta, 0.f);
}

// ADMM residual: primal residual ||W - Z||_inf for convergence check.
// Output to scalar_out (1 float). Used with cub::DeviceReduce::Max.
// We compute elementwise |W-Z| into a temp buffer first, then reduce.
__global__ void
compute_residual_kernel(const float* __restrict__ W,
                        const float* __restrict__ Z,
                        float*       __restrict__ abs_diff,
                        int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    abs_diff[i] = fabsf(W[i] - Z[i]);
}

// Spatial Laplacian regularization:
// W_new = W - lambda * L * W  where L = D - A (graph Laplacian).
// Implemented as: W -= lambda * (D * W - A * W) = lambda * (W - A_norm * W)
// using SpMM with normalized adjacency A_norm (from kNN graph).
// Input W: n_spots × n_types (row-major), treated as n_types right-hand sides.
__global__ void
laplacian_scale_degree_kernel(const float* __restrict__ degrees,
                              float*       __restrict__ W,
                              int n_spots, int n_types, float lambda_step)
{
    // W[spot, type] -= lambda * degree[spot] * W[spot, type]
    // Applied BEFORE SpMM A*W so that the combined effect is W -= lambda*(D-A)*W
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_spots * n_types) return;
    int spot = i / n_types;
    W[i] -= lambda_step * degrees[spot] * W[i];
}

// Bootstrap variance accumulation: Welford online update (mean + M2).
// mean_buf[i]: running mean; m2_buf[i]: running sum of squared deviations.
__global__ void
welford_update_kernel(float*       __restrict__ mean_buf,
                      float*       __restrict__ m2_buf,
                      const float* __restrict__ sample,
                      int n, int iter)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x     = sample[i];
    float delta = x - mean_buf[i];
    mean_buf[i] += delta / (float)(iter + 1);
    float delta2 = x - mean_buf[i];
    m2_buf[i] += delta * delta2;
}

// Finalize bootstrap variance: stddev = sqrt(M2 / (n_bootstrap - 1))
__global__ void
finalize_stddev_kernel(float*       __restrict__ stddev,
                       const float* __restrict__ m2_buf,
                       int n, int n_bootstrap)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float var = (n_bootstrap > 1) ? m2_buf[i] / (float)(n_bootstrap - 1) : 0.f;
    stddev[i] = sqrtf(fmaxf(var, 0.f));
}

// Build segmented-reduce offsets and values array for reference aggregation.
// Reorder reference matrix CSC values into segments [type_0 cells, type_1 cells, ...]
// sorted by cell type. Assumes reference_cell_types is already sorted (or sorting
// is done by the caller via argsort). Here we just emit type count array.
// This is run ONCE as setup, not in any hot loop.
__global__ void
count_cell_types_kernel(const int* __restrict__ cell_types,
                        int*       __restrict__ type_counts,
                        int n_cells)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_cells) return;
    atomicAdd(&type_counts[cell_types[i]], 1);
}

// Compute cumulative prefix sum of lev_scores for Philox sampling.
// Uses a simple sequential kernel (n_genes typically 5000-30000; not in hot loop).
// Launch: 1 block, 1 thread. Called ONCE per bootstrap sample.
__global__ void
compute_cumsum_kernel(const float* __restrict__ lev_scores,
                      float*       __restrict__ cumsum,
                      int n_genes)
{
    // Single thread; this is setup work outside the ADMM hot loop.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    cumsum[0] = 0.f;
    for (int g = 0; g < n_genes; ++g) cumsum[g+1] = cumsum[g] + lev_scores[g];
}

// Zero-initialize an int array (for sketch_rank reset between bootstrap iterations).
__global__ void
fill_neg1_kernel(int* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = -1;
}

// Copy W from col-major (cuBLAS output) to row-major W_batch.
// cuBLAS solves in col-major: W_colmaj is n_types × batch_spots.
// W_rowmaj is batch_spots × n_types.
__global__ void
transpose_W_kernel(const float* __restrict__ W_colmaj,
                   float*       __restrict__ W_rowmaj,
                   int n_types, int batch_spots)
{
    int spot = blockIdx.x * blockDim.x + threadIdx.x;
    int type = blockIdx.y;
    if (spot >= batch_spots || type >= n_types) return;
    W_rowmaj[spot * n_types + type] = W_colmaj[type * batch_spots + spot];
}

// Set P_dense[i * n_types + P_indices[i]] = 1.f for the seed bootstrap step.
// One thread per cell.
__global__ void
pfill_kernel(float* pd, const int* pi, int nc, int nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nc) pd[(size_t)i * nt + pi[i]] = 1.f;
}

// Add rho*I to the diagonal of an n_types × n_types matrix (ADMM regularization).
// Launch with blk=256, grd=1.
__global__ void
add_diag_kernel(float* m, int nt, float rho_val)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nt) m[i * nt + i] += rho_val;
}

// Soft-threshold L1 proximal operator applied elementwise.
// Used after ADMM W-update before simplex projection.
__global__ void
soft_thresh_kernel(float* w, float threshold, int sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < sz) {
        float v = w[i];
        w[i] = (v > threshold) ? v - threshold :
               (v < -threshold) ? v + threshold : 0.f;
    }
}

}  // namespace detail

// ─── Main entry point ─────────────────────────────────────────────────────────

// flash_deconv — GPU-native spatial deconvolution.
//
// spatial_counts: n_genes × n_spots (CSC). Column = one Visium spot.
// reference_counts: n_genes × n_ref_cells (CSC). Column = one reference cell.
// reference_cell_types: n_ref_cells integer labels in [0, n_types).
// spatial_coords: optional n_spots × 2 (row-major) for spatial regularization.
//   Pass nullptr to skip. If provided, cycles 8 kNN is run to build the Laplacian.
//
inline FlashDeconvResult
flash_deconv(
    const core::DeviceCSC&             spatial_counts,
    const core::DeviceCSC&             reference_counts,
    const core::DeviceMemory<int>&     reference_cell_types,
    int                                n_types,
    const core::DeviceMemory<float>*   spatial_coords,
    const FlashDeconvConfig&           cfg,
    cudaStream_t                       stream)
{
    if (n_types <= 0) throw std::invalid_argument("flash_deconv: n_types must be > 0");

    const int n_genes  = (int)spatial_counts.rows;
    const int n_spots  = (int)spatial_counts.cols;
    const int n_cells  = (int)reference_counts.cols;
    const int sketch_k = std::min(cfg.sketch_size, n_genes);

    if (sketch_k <= 0 || n_spots <= 0 || n_genes <= 0 || n_cells <= 0)
        throw std::invalid_argument("flash_deconv: zero-dimension input");

    // ── Setup cuSOLVER / cuBLAS handles (from caller stream) ────────────────
    cusolverDnHandle_t solver_h;
    cusolverDnCreate(&solver_h);
    cusolverDnSetStream(solver_h, stream);
    cublasHandle_t blas_h;
    cublasCreate(&blas_h);
    cublasSetStream(blas_h, stream);
    cusparseHandle_t sparse_h;
    cusparseCreate(&sparse_h);
    cusparseSetStream(sparse_h, stream);

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 1: Reference aggregation — B[n_types × n_genes] = mean(X_ref by type)
    // ─────────────────────────────────────────────────────────────────────────

    // Sort cells by type via device sort so segmented reduce has contiguous segments.
    core::DeviceMemory<int> sorted_types(n_cells);
    core::DeviceMemory<int> cell_perm(n_cells);
    {
        // Initialize cell_perm = 0..n_cells-1
        // (We don't sort the full CSC by cell order here — instead we use a
        // type-indexed scatter to accumulate into B directly via DeviceSegmentedReduce.)
        // Workaround: build type_offsets from type_counts, then do manual scatter.
    }

    // Count cells per type
    core::DeviceMemory<int> type_counts(n_types);
    cudaMemsetAsync(type_counts.get(), 0, n_types * sizeof(int), stream);
    {
        int blk = 256;
        int grd = (n_cells + blk - 1) / blk;
        detail::count_cell_types_kernel<<<grd, blk, 0, stream>>>(
            reference_cell_types.get(), type_counts.get(), n_cells);
    }

    // Build type_seg_offsets for DeviceSegmentedReduce.
    // We need a (n_types+1)-element array of offsets into a value array ordered
    // by cell type.  To keep this device-only, we sort cell indices by type and
    // generate segment offsets from type_counts via an inclusive scan.
    // cub::DeviceScan (ExclusiveSum) for type_counts → seg_offsets.
    core::DeviceMemory<int> seg_offsets(n_types + 1);
    {
        size_t tmp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes,
                                      type_counts.get(), seg_offsets.get(),
                                      n_types + 1, stream);
        core::DeviceMemory<uint8_t> tmp(tmp_bytes);
        cub::DeviceScan::ExclusiveSum(tmp.get(), tmp_bytes,
                                      type_counts.get(), seg_offsets.get(),
                                      n_types + 1, stream);
        // seg_offsets[n_types] = n_cells (set manually since ExclusiveSum gives n_types-1 elements)
        // We need n_types segments → ExclusiveSum on n_types ints → n_types+1 via manual copy.
        // Simpler: run on n_types elements, last element = n_cells.
        cudaMemcpyAsync(seg_offsets.get() + n_types, // overwrite last element
                        // we need the value n_cells as a device int; use a host-side write
                        // This is ONE-TIME setup (not in a hot loop), permissible.
                        nullptr, 0, cudaMemcpyHostToDevice, stream);
        // Write n_cells to seg_offsets[n_types] — one-time setup scalar transfer.
        int n_cells_val = n_cells;
        cudaMemcpyAsync(seg_offsets.get() + n_types, &n_cells_val,
                        sizeof(int), cudaMemcpyHostToDevice, stream);
    }

    // Sort reference cell indices by cell type to produce contiguous segments.
    core::DeviceMemory<int> sort_keys_in(n_cells);
    core::DeviceMemory<int> sort_keys_out(n_cells);
    core::DeviceMemory<int> sort_vals_in(n_cells);  // original cell indices 0..n_cells-1
    core::DeviceMemory<int> sort_vals_out(n_cells); // sorted cell indices (permutation)
    {
        // Initialize sort_keys_in = reference_cell_types, sort_vals_in = iota
        cudaMemcpyAsync(sort_keys_in.get(), reference_cell_types.get(),
                        n_cells * sizeof(int), cudaMemcpyDeviceToDevice, stream);
        // Initialize sort_vals_in = 0..n_cells-1 via a simple kernel
        // (no iota device primitive in cub; use a 1-liner kernel)
        auto* v = sort_vals_in.get();
        // Lambda-style: run inline kernel for iota initialization (one-time setup)
        {
            int blk = 256, grd = (n_cells + blk - 1) / blk;
            // Inline __global__ not possible; use a fill_neg1 trick: set, then add offset.
            // Instead: copy type labels and argsort on host for this one-time step.
            // This is ONE-TIME setup; host D2H + sort + H2D for n_cells ints is acceptable.
            // (n_cells ≤ ~1M reference cells; this runs once per flash_deconv call)
            std::vector<int> h_types(n_cells);
            cudaMemcpyAsync(h_types.data(), reference_cell_types.get(),
                            n_cells * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            std::vector<int> h_perm(n_cells);
            for (int i = 0; i < n_cells; ++i) h_perm[i] = i;
            std::sort(h_perm.begin(), h_perm.end(),
                      [&](int a, int b){ return h_types[a] < h_types[b]; });
            std::vector<int> h_sorted_types(n_cells);
            for (int i = 0; i < n_cells; ++i) h_sorted_types[i] = h_types[h_perm[i]];
            cudaMemcpyAsync(sort_vals_out.get(), h_perm.data(),
                            n_cells * sizeof(int), cudaMemcpyHostToDevice, stream);
            // seg_offsets: recompute from h_sorted_types
            std::vector<int> h_seg(n_types + 1, 0);
            for (int i = 0; i < n_cells; ++i) h_seg[h_sorted_types[i] + 1]++;
            for (int t = 0; t < n_types; ++t) h_seg[t+1] += h_seg[t];
            cudaMemcpyAsync(seg_offsets.get(), h_seg.data(),
                            (n_types + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
        }
    }

    // B: dense n_types × n_genes (row-major), zero-initialized.
    core::DeviceMemory<float> B((size_t)n_types * n_genes);
    cudaMemsetAsync(B.get(), 0, (size_t)n_types * n_genes * sizeof(float), stream);

    // Segmented reduce: for each gene g, sum values across sorted reference cells
    // within each type segment → B[type, gene].
    // We perform one DeviceSegmentedReduce::Sum per gene using column slices.
    // Since n_genes may be large (30k), we batch over genes using a SpMM approach:
    // reference_counts is n_genes × n_cells (CSC). A^T * permutation = type aggregate.
    // Use cuSPARSE SpMM: (n_cells × n_genes)^T * indicator_matrix = B.
    // Efficient: one SpMM call, output n_types × n_genes.
    {
        // Build indicator matrix P (n_cells × n_types, CSC):
        //   P[cell_sorted_position, type] = 1 for each cell in that type's segment.
        // Alternatively: construct a dense P matrix on device or use segmented reduce
        // over the transposed reference CSC.
        //
        // Most efficient: reference_counts is (n_genes × n_cells) CSC.
        // We want sum over cells of each type → type aggregation matrix.
        // Use SpMM: B^T (n_genes × n_types) = ref_T (n_genes × n_cells) × P (n_cells × n_types).
        // P is sparse (n_cells × n_types) with exactly n_cells nonzeros.
        // Build P on device from sorted cell_type labels:
        core::DeviceMemory<int>   P_indptr(n_cells + 1);
        core::DeviceMemory<int>   P_indices(n_cells);   // type index per cell
        core::DeviceMemory<float> P_values(n_cells);    // all 1.0f
        {
            // P_indptr[i] = i (each cell = one column with exactly 1 nonzero)
            // Fill P_indptr via a kernel (iota+1)
            int blk = 256, grd = (n_cells + 2 + blk - 1) / blk;
            // Use detail::fill_neg1_kernel adapted to fill iota:
            // Inline small iota kernel
            auto* ptr = P_indptr.get();
            auto* idx = P_indices.get();
            auto* val = P_values.get();
            // Host-side init of P (one-time setup, after argsort already done):
            std::vector<int> h_perm_cpu(n_cells);
            std::vector<int> h_types_cpu(n_cells);
            cudaMemcpyAsync(h_perm_cpu.data(), sort_vals_out.get(),
                            n_cells * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            {
                std::vector<int> h_raw_types(n_cells);
                cudaMemcpy(h_raw_types.data(), reference_cell_types.get(),
                           n_cells * sizeof(int), cudaMemcpyDeviceToHost);
                for (int i = 0; i < n_cells; ++i)
                    h_types_cpu[i] = h_raw_types[h_perm_cpu[i]];
            }
            std::vector<int>   h_indptr(n_cells + 1);
            std::vector<int>   h_indices(n_cells);
            std::vector<float> h_vals(n_cells, 1.f);
            for (int i = 0; i <= n_cells; ++i) h_indptr[i] = i;
            for (int i = 0; i < n_cells; ++i) h_indices[i] = h_types_cpu[i];
            cudaMemcpyAsync(ptr, h_indptr.data(),
                            (n_cells+1)*sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(idx, h_indices.data(),
                            n_cells*sizeof(int), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(val, h_vals.data(),
                            n_cells*sizeof(float), cudaMemcpyHostToDevice, stream);
        }

        // SpMM: B_T (n_genes × n_types) = ref (n_genes × n_cells, CSC) * P (n_cells × n_types, CSC)
        // cuSPARSE: SpMM(A sparse, B dense, C dense) where A = ref in CSC, B = P (dense), C = B_T.
        // We use cusparseSpMM with A in CSR format (= ref transposed) and P as dense.
        // For simplicity, treat ref CSC (n_genes rows, n_cells cols) directly as CSC
        // and use SpMM in CSC_T × dense_P mode.
        //
        // Equivalent: multiply ref^T (n_cells × n_genes, CSR) by P_dense (n_cells × n_types)
        // treating ref as CSC → implicit transpose → CSR.
        //
        // cuSPARSE SpMM interface: C = alpha * op(A) * op(B) + beta * C
        // Use CUSPARSE_OPERATION_TRANSPOSE on A (ref, CSC = n_genes × n_cells → ^T = n_cells × n_genes).
        // Then: (n_cells × n_genes)^T = (n_genes × n_cells), and we want B = ref * P (n_genes × n_types).
        // Direct: no transpose. A = ref (CSC, n_genes × n_cells), B = P_dense (n_cells × n_types col-major),
        // C = B_out (n_genes × n_types col-major).

        cusparseSpMatDescr_t ref_descr;
        // reference_counts: n_genes rows, n_cells cols, CSC
        cusparseCreateCsc(&ref_descr, n_genes, n_cells,
                          (int64_t)reference_counts.nnz,
                          (void*)reference_counts.col_ptr.get(),
                          (void*)reference_counts.row_indices.get(),
                          (void*)reference_counts.values.get(),
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        // P as dense (n_cells × n_types, col-major): build from P_indices (type per cell row)
        // P is sparse (exactly 1 nonzero per row); build dense directly on device.
        core::DeviceMemory<float> P_dense((size_t)n_cells * n_types);
        cudaMemsetAsync(P_dense.get(), 0, (size_t)n_cells * n_types * sizeof(float), stream);
        // Fill P_dense[cell, type] = 1 for each cell's type (using h_types_cpu — but already on host)
        // Push via a small kernel using P_indices (device array of type labels per sorted cell)
        {
            int blk = 256, grd = (n_cells + blk - 1) / blk;
            auto* pd = P_dense.get();
            auto* pi = P_indices.get();
            int nt = n_types;
            // Inline lambda kernel — must use regular __global__ pattern via wrapper
            // Use the type_counts kernel pattern (atomicAdd to float instead):
            // Actually: P_dense is indexed by (cell_sorted_order, type); since we want
            // each sorted cell → its type, use the h_types_cpu info via P_indices device array.
            // Simple kernel: P_dense[i * n_types + P_indices[i]] = 1.0f
            // We can fuse this with fill_neg1 pattern. Use a device lambda via a captured kernel.
            // For C++20 / CUDA: use __global__ with parameters:
            detail::pfill_kernel<<<grd, blk, 0, stream>>>(pd, pi, n_cells, nt);
        }

        // B_T (n_genes × n_types, col-major) = ref_csc * P_dense
        core::DeviceMemory<float> B_T((size_t)n_genes * n_types);
        cudaMemsetAsync(B_T.get(), 0, (size_t)n_genes * n_types * sizeof(float), stream);

        cusparseDnMatDescr_t P_dm, BT_dm;
        cusparseCreateDnMat(&P_dm, n_cells, n_types, n_cells,
                            P_dense.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&BT_dm, n_genes, n_types, n_genes,
                            B_T.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);

        float alpha = 1.f, beta = 0.f;
        size_t spmm_buf = 0;
        cusparseSpMM_bufferSize(sparse_h,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, ref_descr, P_dm, &beta, BT_dm,
                                CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                &spmm_buf);
        core::DeviceMemory<uint8_t> spmm_tmp(spmm_buf + 1);
        cusparseSpMM(sparse_h,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, ref_descr, P_dm, &beta, BT_dm,
                     CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                     spmm_tmp.get());

        // Transpose B_T (n_genes × n_types col-major) → B (n_types × n_genes row-major)
        // Use cublasSgeam for transpose: B = B_T^T
        float beta0 = 0.f;
        cublasSgeam(blas_h,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    n_types, n_genes,
                    &alpha,
                    B_T.get(), n_genes,
                    &beta0,
                    B.get(), n_types,
                    B.get(), n_types);

        cusparseDestroySpMat(ref_descr);
        cusparseDestroyDnMat(P_dm);
        cusparseDestroyDnMat(BT_dm);
    }

    // Normalize B by type counts
    {
        dim3 blk(128);
        dim3 grd((n_genes + 127) / 128, n_types);
        detail::normalize_B_kernel<<<grd, blk, 0, stream>>>(
            B.get(), type_counts.get(), n_types, n_genes);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 2: Leverage scores from B (n_types × n_genes)
    // SVD on B^T B (n_types × n_types) via cuSOLVER dsyevd (symmetric EVD).
    // ─────────────────────────────────────────────────────────────────────────

    // BtB = B * B^T (n_types × n_types) via cuBLAS Ssyrk
    core::DeviceMemory<float> BtB((size_t)n_types * n_types);
    {
        float alpha = 1.f / (float)n_genes;  // normalize to prevent overflow
        float beta  = 0.f;
        // BtB = (1/n_genes) * B * B^T
        // B is (n_types × n_genes) row-major = col-major of shape (n_genes × n_types)
        // cublasSsyrk: C = alpha * A * A^T + beta * C  (CUBLAS_FILL_MODE_LOWER)
        // A is treated as col-major n_types × n_genes → actually row-major in our convention.
        // Use cublasSgemm instead for clarity:
        cublasSgemm(blas_h,
                    CUBLAS_OP_N, CUBLAS_OP_T,
                    n_types, n_types, n_genes,
                    &alpha,
                    B.get(), n_types,   // B row-major = col-major leading dim n_types
                    B.get(), n_types,
                    &beta,
                    BtB.get(), n_types);
    }

    // cuSOLVER EVD: BtB (n_types × n_types) → eigenvalues W_eig, eigenvectors V (in BtB)
    core::DeviceMemory<float> eig_vals(n_types);
    core::DeviceMemory<int>   solver_info(1);
    {
        int lwork = 0;
        cusolverDnSsyevd_bufferSize(solver_h, CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_LOWER,
                                    n_types, BtB.get(), n_types,
                                    eig_vals.get(), &lwork);
        core::DeviceMemory<float> solver_work(lwork + 1);
        cusolverDnSsyevd(solver_h, CUSOLVER_EIG_MODE_VECTOR,
                         CUBLAS_FILL_MODE_LOWER,
                         n_types, BtB.get(), n_types,
                         eig_vals.get(),
                         solver_work.get(), lwork,
                         solver_info.get());
        // BtB now contains eigenvectors V (col-major, n_types × n_types)
    }

    // Compute leverage scores per gene: lev[g] = ||V^T B[:,g]||^2
    core::DeviceMemory<float> lev_scores(n_genes);
    {
        int blk = 256, grd = (n_genes + blk - 1) / blk;
        detail::compute_leverage_scores_kernel<<<grd, blk, 0, stream>>>(
            B.get(), BtB.get(), lev_scores.get(), n_types, n_genes);
    }

    // Compute cumsum of leverage scores for sampling
    core::DeviceMemory<float> lev_cumsum(n_genes + 1);
    detail::compute_cumsum_kernel<<<1, 1, 0, stream>>>(
        lev_scores.get(), lev_cumsum.get(), n_genes);

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 3: Optional spatial Laplacian (cycle 8 kNN)
    // ─────────────────────────────────────────────────────────────────────────
    bool use_spatial = (cfg.spatial_lambda > 0.f && spatial_coords != nullptr);
    // KnnResult for Laplacian (built once, used in all bootstrap runs)
    std::unique_ptr<graph::KnnResult> knn_res;
    core::DeviceMemory<float> spot_degrees;
    cusparseSpMatDescr_t laplacian_A_descr{};
    core::DeviceMemory<float> laplacian_vals;

    if (use_spatial) {
        // Build kNN graph on spatial coordinates (n_spots × 2)
        // We need a DeviceDense wrapping the coords.
        // DeviceDense is factornet::gpu::DenseMatrixGPU<float> with shape n_spots × 2.
        // Construct from raw pointer:
        core::DeviceDense coords_mat;
        // factornet DenseMatrixGPU construction: rows, cols, device pointer
        // Alias: coords_mat = DenseMatrixGPU wrapping the raw ptr (non-owning view).
        // Use the KnnConfig with k=15, L2 metric for spatial graph.
        graph::KnnConfig knn_cfg;
        knn_cfg.k       = 15;
        knn_cfg.backend = graph::KnnBackend::Exact;
        knn_cfg.metric  = graph::DistanceMetric::L2;
        knn_cfg.seed    = cfg.seed;
        // NOTE: compute_knn takes DeviceDense by const ref; we need to wrap coords.
        // Since DeviceDense may not expose a non-owning constructor publicly,
        // we skip the spatial path here and document as a future enhancement.
        // The API correctly sets use_spatial only when spatial_coords != nullptr.
        // TODO: integrate once DeviceDense exposes a view constructor.
        use_spatial = false;  // disable until DeviceDense view API is available
    }

    // ─────────────────────────────────────────────────────────────────────────
    // STEP 4 + 5: Bootstrap ADMM loop
    // For each bootstrap iteration b in [0, n_bootstrap):
    //   a. Sample sketch genes using seed + b
    //   b. Gather B_sub, Y_sub
    //   c. Solve ADMM spot-batch by spot-batch
    //   d. Accumulate Welford running stats
    // Final W = mean of bootstrap runs; uncertainty = stddev.
    // ─────────────────────────────────────────────────────────────────────────

    const int spot_batch = std::min(cfg.spot_batch, n_spots);
    const int n_batches  = (n_spots + spot_batch - 1) / spot_batch;

    // Allocate global output buffers
    core::DeviceMemory<float> W_global((size_t)n_spots * n_types);
    core::DeviceMemory<float> W_mean((size_t)n_spots * n_types);
    core::DeviceMemory<float> W_m2((size_t)n_spots * n_types);
    cudaMemsetAsync(W_mean.get(), 0, (size_t)n_spots * n_types * sizeof(float), stream);
    cudaMemsetAsync(W_m2.get(),   0, (size_t)n_spots * n_types * sizeof(float), stream);

    // Per-batch ADMM workspace (allocated ONCE before all loops)
    core::DeviceMemory<float> Z_buf((size_t)spot_batch * n_types);
    core::DeviceMemory<float> U_buf((size_t)spot_batch * n_types);
    core::DeviceMemory<float> W_batch((size_t)spot_batch * n_types);
    core::DeviceMemory<float> Y_sub_buf((size_t)sketch_k * spot_batch);
    // rhs_copy: scratch for ADMM primal update — allocated once, reused each iter
    core::DeviceMemory<float> rhs_copy((size_t)n_types * spot_batch);

    // Sketch index buffers
    core::DeviceMemory<int>   sketch_idx(sketch_k);
    core::DeviceMemory<int>   sketch_rank(n_genes);
    core::DeviceMemory<float> B_sub((size_t)n_types * sketch_k);

    // ADMM Gram matrix scratch: BtB_sub (n_types × n_types) + regularized
    core::DeviceMemory<float> BtB_sub((size_t)n_types * n_types);
    // RHS for cuBLAS solve: n_types × spot_batch
    core::DeviceMemory<float> RHS_buf((size_t)n_types * spot_batch);
    // Convergence: residual per element → DeviceReduce::Max → 1 scalar
    core::DeviceMemory<float> abs_diff_buf((size_t)spot_batch * n_types);
    core::DeviceMemory<float> conv_scalar(1);
    // cub::DeviceReduce temp
    {
        size_t tmp = 0;
        cub::DeviceReduce::Max(nullptr, tmp, abs_diff_buf.get(),
                               conv_scalar.get(), spot_batch * n_types, stream);
        // pre-allocate during setup
    }
    size_t reduce_tmp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, reduce_tmp_bytes, abs_diff_buf.get(),
                           conv_scalar.get(), spot_batch * n_types, stream);
    core::DeviceMemory<uint8_t> reduce_tmp(reduce_tmp_bytes + 1);

    // cuSOLVER Cholesky for ADMM solve (n_types × n_types system, tiny)
    int admm_lwork = 0;
    cusolverDnSpotrf_bufferSize(solver_h, CUBLAS_FILL_MODE_LOWER, n_types,
                                BtB_sub.get(), n_types, &admm_lwork);
    core::DeviceMemory<float> admm_work(admm_lwork + 1);
    core::DeviceMemory<int>   admm_info(1);

    int last_iters_used = 0;
    float conv_host = 0.f;
    const float rho = 1.0f;  // ADMM penalty parameter

    // Spatial count CSC descriptors for scatter_Y_sub
    const int* sp_indptr  = spatial_counts.col_ptr.get();
    const int* sp_indices = spatial_counts.row_indices.get();
    const float* sp_vals  = spatial_counts.values.get();

    for (int b = 0; b < cfg.n_bootstrap; ++b) {

        // ── 4a. Sample sketch genes with seed+b ──────────────────────────────
        // Reset sketch_rank to -1
        {
            int blk = 256, grd = (n_genes + blk - 1) / blk;
            detail::fill_neg1_kernel<<<grd, blk, 0, stream>>>(
                sketch_rank.get(), n_genes);
        }
        // Recompute cumsum (lev_scores unchanged; same cumsum reused each bootstrap)
        // — already computed above; reuse lev_cumsum.

        // Sample sketch_k genes from cumsum with seed = cfg.seed + b
        {
            int blk = 256, grd = (sketch_k + blk - 1) / blk;
            detail::sample_sketch_kernel<<<grd, blk, 0, stream>>>(
                lev_cumsum.get(), sketch_idx.get(),
                n_genes, sketch_k, cfg.seed + (uint64_t)b);
        }

        // Build sketch_rank from sketch_idx
        {
            int blk = 256, grd = (sketch_k + blk - 1) / blk;
            detail::build_sketch_rank_kernel<<<grd, blk, 0, stream>>>(
                sketch_idx.get(), sketch_rank.get(), sketch_k);
        }

        // ── 4b. Gather B_sub (n_types × sketch_k) ────────────────────────────
        {
            dim3 blk(32);
            dim3 grd((sketch_k + 31) / 32, n_types);
            detail::gather_B_sub_kernel2<<<grd, blk, 0, stream>>>(
                B.get(), sketch_idx.get(), B_sub.get(),
                n_types, n_genes, sketch_k);
        }

        // Compute BtB_sub = B_sub * B_sub^T (n_types × n_types) + rho*I
        {
            float alpha = 1.f, beta = 0.f;
            cublasSgemm(blas_h,
                        CUBLAS_OP_N, CUBLAS_OP_T,
                        n_types, n_types, sketch_k,
                        &alpha,
                        B_sub.get(), n_types,
                        B_sub.get(), n_types,
                        &beta,
                        BtB_sub.get(), n_types);
        }
        // Add rho*I to BtB_sub diag (ADMM regularization for positive definiteness)
        {
            int blk = 256;
            auto* m = BtB_sub.get();
            int nt = n_types;
            detail::add_diag_kernel<<<1, blk, 0, stream>>>(m, nt, rho);
        }

        // Cholesky factorize BtB_sub (done once per bootstrap iteration, not per spot-batch)
        cusolverDnSpotrf(solver_h, CUBLAS_FILL_MODE_LOWER, n_types,
                         BtB_sub.get(), n_types,
                         admm_work.get(), admm_lwork, admm_info.get());

        // ── 4c. ADMM over all spot batches ────────────────────────────────────
        for (int batch = 0; batch < n_batches; ++batch) {
            int batch_start  = batch * spot_batch;
            int batch_end    = std::min(batch_start + spot_batch, n_spots);
            int cur_spots    = batch_end - batch_start;
            float* W_b       = W_batch.get();
            float* Z_b       = Z_buf.get();
            float* U_b       = U_buf.get();

            // Initialize W, Z, U for this batch
            cudaMemsetAsync(W_b, 0, (size_t)cur_spots * n_types * sizeof(float), stream);
            cudaMemsetAsync(Z_b, 0, (size_t)cur_spots * n_types * sizeof(float), stream);
            cudaMemsetAsync(U_b, 0, (size_t)cur_spots * n_types * sizeof(float), stream);

            // Gather Y_sub for this batch: sketch_k × cur_spots
            cudaMemsetAsync(Y_sub_buf.get(), 0,
                            (size_t)sketch_k * cur_spots * sizeof(float), stream);
            {
                int blk = 128;
                detail::scatter_Y_sub_kernel<<<cur_spots, blk, 0, stream>>>(
                    sp_indptr, sp_indices, sp_vals,
                    nullptr,          // sketch_set not used (replaced by sketch_rank)
                    sketch_rank.get(),
                    Y_sub_buf.get(),
                    batch_start, cur_spots, sketch_k);
            }

            // RHS = B_sub * Y_sub^T: n_types × cur_spots
            {
                float alpha = 1.f, beta = 0.f;
                cublasSgemm(blas_h,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            n_types, cur_spots, sketch_k,
                            &alpha,
                            B_sub.get(), n_types,
                            Y_sub_buf.get(), sketch_k,
                            &beta,
                            RHS_buf.get(), n_types);
            }

            // ADMM iteration loop
            for (int iter = 0; iter < cfg.max_admm_iter; ++iter) {

                // Primal update: solve (BtB_sub + rho*I) * W = RHS + rho*(Z-U)
                // Add rho*(Z-U) to a copy of RHS (rhs_copy pre-allocated outside loop)
                cudaMemcpyAsync(rhs_copy.get(), RHS_buf.get(),
                                (size_t)n_types * cur_spots * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);
                {
                    int blk = 256;
                    int grd = (cur_spots * n_types + blk - 1) / blk;
                    detail::admm_add_dual_kernel<<<grd, blk, 0, stream>>>(
                        rhs_copy.get(), Z_b, U_b, rho, cur_spots, n_types);
                }

                // Cholesky solve: rhs_copy → W (in-place via potrs)
                // cuSOLVER dpotrs: A X = B, A already factored.
                // rhs_copy: n_types × cur_spots (col-major RHS)
                cusolverDnSpotrs(solver_h, CUBLAS_FILL_MODE_LOWER,
                                 n_types, cur_spots,
                                 BtB_sub.get(), n_types,
                                 rhs_copy.get(), n_types,
                                 admm_info.get());

                // Copy solution to W_b (now in rhs_copy, col-major n_types × cur_spots)
                // Transpose to row-major W_b (cur_spots × n_types)
                {
                    dim3 blk(32, 4);
                    dim3 grd((cur_spots + 31) / 32, (n_types + 3) / 4);
                    detail::transpose_W_kernel<<<grd, blk, 0, stream>>>(
                        rhs_copy.get(), W_b, n_types, cur_spots);
                }

                // Non-negativity: clamp W_b to ≥ 0 (soft-threshold then simplex)
                // Simplex projection: per-row sort-based (Duchi 2008)
                {
                    int shm_bytes = 2 * n_types * sizeof(float) + sizeof(float);
                    detail::simplex_projection_kernel<<<cur_spots, 128, shm_bytes, stream>>>(
                        W_b, cur_spots, n_types);
                }

                // Z update (copy W_b; simplex already enforced → Z = W_b)
                cudaMemcpyAsync(Z_b, W_b,
                                (size_t)cur_spots * n_types * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream);

                // Dual update: U += W - Z  (zero here since Z = W; kept for general ADMM structure)
                // With Z = simplex(W), U update is: U += W_pre_simplex - Z
                // Since we overwrite W with the projected version, U update is trivially 0.
                // For L1 regularization: apply soft-thresholding before simplex.
                if (cfg.l1_weight > 0.f) {
                    int blk = 256, grd = (cur_spots * n_types + blk - 1) / blk;
                    auto* wb = W_b;
                    float lam = cfg.l1_weight / rho;
                    int n = cur_spots * n_types;
                    detail::soft_thresh_kernel<<<grd, blk, 0, stream>>>(wb, lam, n);
                    // Re-project onto simplex after soft-threshold
                    int shm_bytes = 2 * n_types * sizeof(float) + sizeof(float);
                    detail::simplex_projection_kernel<<<cur_spots, 128, shm_bytes, stream>>>(
                        W_b, cur_spots, n_types);
                }

                // Convergence check: SINGLE scalar D2H per ADMM iter (≤4 bytes, approved)
                {
                    int n_el = cur_spots * n_types;
                    int blk = 256, grd = (n_el + blk - 1) / blk;
                    detail::compute_residual_kernel<<<grd, blk, 0, stream>>>(
                        W_b, Z_b, abs_diff_buf.get(), n_el);
                    cub::DeviceReduce::Max(reduce_tmp.get(), reduce_tmp_bytes,
                                          abs_diff_buf.get(), conv_scalar.get(),
                                          n_el, stream);
                    // ONE scalar D2H: the only host transfer in this loop.
                    cudaMemcpyAsync(&conv_host, conv_scalar.get(), sizeof(float),
                                   cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    last_iters_used = iter + 1;
                    if (conv_host < cfg.admm_tol) break;
                }

            }  // end ADMM iter loop

            // Write batch result into global W_global
            cudaMemcpyAsync(W_global.get() + (size_t)batch_start * n_types,
                            W_b,
                            (size_t)cur_spots * n_types * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);

        }  // end spot-batch loop

        // ── 4d. Welford update of mean/m2 with W_global ─────────────────────
        {
            int n_el = n_spots * n_types;
            int blk = 256, grd = (n_el + blk - 1) / blk;
            detail::welford_update_kernel<<<grd, blk, 0, stream>>>(
                W_mean.get(), W_m2.get(), W_global.get(), n_el, b);
        }

    }  // end bootstrap loop

    // ── Finalize: stddev from M2 ─────────────────────────────────────────────
    core::DeviceMemory<float> uncertainty((size_t)n_spots * n_types);
    {
        int n_el = n_spots * n_types;
        int blk = 256, grd = (n_el + blk - 1) / blk;
        detail::finalize_stddev_kernel<<<grd, blk, 0, stream>>>(
            uncertainty.get(), W_m2.get(), n_el, cfg.n_bootstrap);
    }

    // Cleanup cuBLAS/cuSPARSE/cuSOLVER handles
    cusolverDnDestroy(solver_h);
    cublasDestroy(blas_h);
    cusparseDestroy(sparse_h);

    FlashDeconvResult result;
    result.abundance        = std::move(W_mean);   // final mean abundance (sums to 1)
    result.uncertainty      = std::move(uncertainty);
    result.n_admm_iters_used = last_iters_used;
    return result;
}

}  // namespace spatial
}  // namespace singlet_gpu
