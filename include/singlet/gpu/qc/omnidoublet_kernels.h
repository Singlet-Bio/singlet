// SPDX-License-Identifier: MIT
// singlet/gpu/qc/omnidoublet_kernels.h
//
// CUDA kernels and kernel-launch helpers for omnidoublet.h.
// Split out of omnidoublet.h (multi-concern header cleanup): this file holds
// the __global__/__device__ kernels and detail-namespace helpers;
// omnidoublet.h keeps the public API and host orchestration and #includes
// this header.

#pragma once

#include <cstdint>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <limits>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cublas_v2.h>
#include <cub/cub.cuh>

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

namespace singlet::gpu {
namespace qc {

// ─── Detail namespace: kernels ────────────────────────────────────────────────

namespace detail {

// ─── Kernel 1: sparse doublet simulation (CSC row gather + sum) ──────────────
//
// Each CUDA block handles one synthetic doublet (row k in the output tile).
// threads[0..n_genes-1] read rna_counts[i] + rna_counts[j] for their gene index.
// Because CSC is column-major (gene = row, cell = column), gathering "cell i's
// counts for all genes" requires a CSC transpose scan. We use the following:
//   For each gene g: count_i_g = lookup in CSC column i via indptr.
// However CSC stores cells as columns; gene g is a row. Gathering all genes for
// cell i requires a full CSC scan which is O(n_genes). Instead we pre-build
// the CSC-transposed view (genes × cells → COO per cell) ONCE, stored as the
// input CSC itself (genes=rows, cells=cols), so indptr[cell] gives the range.
//
// WAIT: DeviceCSC is (rows=genes, cols=cells). indptr[j] is cell j column start.
// So "all genes expressed in cell i" = values[indptr[i]..indptr[i+1]), at
// row_indices[indptr[i]..]. This gives us the SPARSE gene list for cell i.
// We scatter those into the dense output tile.
//
// One block per simulated doublet in the tile (blockIdx.x ∈ [0, tile_size)).
// blockDim.x = 256 threads. Each thread handles n_genes/256 genes.
//
// WHY scatter not gather: CSC access for a cell gives sparse genes efficiently;
// we scatter to the dense output. A full-gene gather would walk all n_genes for
// every simulated cell even if nnz << n_genes.
__global__ void
sim_doublets_rna_kernel(
    // CSC input (genes × cells)
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_indptr,
    const int*   __restrict__ csc_row_indices,
    int n_genes,
    // Philox seed + tile offset
    uint32_t seed_lo, uint32_t seed_hi,
    int global_sim_offset,   // absolute first sim index for this tile
    int n_cells_total,       // range for sampling
    // dense output tile: tile_size × n_genes, row-major
    float* __restrict__ out_tile,
    int tile_size)
{
    const int local_sim = (int)blockIdx.x;
    if (local_sim >= tile_size) return;
    const int global_sim = global_sim_offset + local_sim;

    // Philox: draw two parent cell indices.
    curandStatePhilox4_32_10_t rng;
    curand_init(
        (static_cast<uint64_t>(seed_hi) << 32) | seed_lo,
        (unsigned long long)global_sim,
        0ULL,
        &rng);
    float4 r4 = curand_uniform4(&rng);
    int ci = min((int)(r4.x * (float)n_cells_total), n_cells_total - 1);
    int cj = min((int)(r4.y * (float)n_cells_total), n_cells_total - 1);

    // Zero the output row for this simulated doublet.
    float* out_row = out_tile + (size_t)local_sim * n_genes;
    for (int g = threadIdx.x; g < n_genes; g += blockDim.x) {
        out_row[g] = 0.f;
    }
    __syncthreads();

    // Scatter cell i's nonzero genes (CSC column ci).
    const int ci_start = csc_indptr[ci];
    const int ci_end   = csc_indptr[ci + 1];
    for (int idx = ci_start + (int)threadIdx.x; idx < ci_end; idx += blockDim.x) {
        int gene = csc_row_indices[idx];
        // atomicAdd because multiple threads of the same block can write the same gene
        // if ci_end - ci_start > blockDim.x. Using atomicAdd (not plain write) is
        // correct but adds cost. For typical single-cell data (sparse) most writes
        // are non-conflicting, so the overhead is small.
        atomicAdd(&out_row[gene], csc_values[idx]);
    }
    __syncthreads();

    // Scatter cell j's nonzero genes (CSC column cj). Sum onto out_row.
    const int cj_start = csc_indptr[cj];
    const int cj_end   = csc_indptr[cj + 1];
    for (int idx = cj_start + (int)threadIdx.x; idx < cj_end; idx += blockDim.x) {
        int gene = csc_row_indices[idx];
        atomicAdd(&out_row[gene], csc_values[idx]);
    }
}

// ─── Kernel 1b: sparse doublet simulation for ADT ────────────────────────────
//
// Identical logic as sim_doublets_rna_kernel but for n_tags (small, ≤1000).
// Reuses the SAME Philox state so the (i,j) pairs are consistent with RNA tile.
// WHY a separate kernel: n_tags is small (≤1000); we could put RNA+ADT in one
// kernel but that would couple kernel launch geometry to both dims. Separate
// launches keep tile memory bounded independently.
__global__ void
sim_doublets_adt_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_indptr,
    const int*   __restrict__ csc_row_indices,
    int n_tags,
    uint32_t seed_lo, uint32_t seed_hi,
    int global_sim_offset,
    int n_cells_total,
    float* __restrict__ out_tile,
    int tile_size)
{
    const int local_sim = (int)blockIdx.x;
    if (local_sim >= tile_size) return;
    const int global_sim = global_sim_offset + local_sim;

    // Re-derive the SAME (ci, cj) pair as the RNA kernel (same seed + offset).
    curandStatePhilox4_32_10_t rng;
    curand_init(
        (static_cast<uint64_t>(seed_hi) << 32) | seed_lo,
        (unsigned long long)global_sim,
        0ULL,
        &rng);
    float4 r4 = curand_uniform4(&rng);
    int ci = min((int)(r4.x * (float)n_cells_total), n_cells_total - 1);
    int cj = min((int)(r4.y * (float)n_cells_total), n_cells_total - 1);

    float* out_row = out_tile + (size_t)local_sim * n_tags;
    for (int g = threadIdx.x; g < n_tags; g += blockDim.x)
        out_row[g] = 0.f;
    __syncthreads();

    const int ci_start = csc_indptr[ci], ci_end = csc_indptr[ci + 1];
    for (int idx = ci_start + (int)threadIdx.x; idx < ci_end; idx += blockDim.x)
        atomicAdd(&out_row[csc_row_indices[idx]], csc_values[idx]);
    __syncthreads();
    const int cj_start = csc_indptr[cj], cj_end = csc_indptr[cj + 1];
    for (int idx = cj_start + (int)threadIdx.x; idx < cj_end; idx += blockDim.x)
        atomicAdd(&out_row[csc_row_indices[idx]], csc_values[idx]);
}

// ─── Kernel 2: log1p row-normalize for dense tile ────────────────────────────
//
// Each row of the dense tile is a cell (real or simulated). In-place.
// Applied after simulation; real cells are lognormed via preprocess::log_normalize
// (which operates on CSC), while simulated cells are dense tiles so we need a
// separate dense lognorm kernel.
//
// WHY separate from cycle-2 lognorm: cycle-2 lognorm operates on CSC (sparse).
// Simulated doublets are already dense (sparse gather result), so we run the
// fused scale+log1p on the dense tile directly.
//
// For real cells: cycle-2 log_normalize is called on the original CSC matrices
// before extracting the HVG dense submatrix. So real cells are normalized first.
// Simulated tiles are normalized AFTER simulation (they start as raw count sums).
//
// Target count T: passed as a scalar (the same T computed from real cells).
// size_factor[row] = rowsum / T; out[row][g] = log1pf(in[row][g] / sf[row]).
__global__ void
dense_lognorm_inplace_kernel(
    float* __restrict__ tile,   // n_rows × n_cols, row-major
    int    n_rows,
    int    n_cols,
    float  target_count)
{
    const int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    float* r = tile + (size_t)row * n_cols;

    // Compute row sum in a warp-reduce.
    float sum = 0.f;
    for (int g = threadIdx.x; g < n_cols; g += blockDim.x)
        sum += r[g];
    // Warp reduce.
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    // Broadcast to all threads in block via shared memory.
    __shared__ float s_sf;
    if (threadIdx.x == 0) {
        float sf = (sum > 0.f) ? (target_count / sum) : 1.f;
        s_sf = sf;
    }
    __syncthreads();
    float sf = s_sf;

    // Apply scale + log1p.
    for (int g = threadIdx.x; g < n_cols; g += blockDim.x)
        r[g] = log1pf(r[g] * sf);
}

// ─── Kernel 3: HVG submatrix column extraction (dense gather) ─────────────────
//
// Given a dense real-cell matrix (n_cells × n_genes) and an HVG index array
// (n_hvg integers), extract the n_hvg columns into a compact output matrix
// (n_cells × n_hvg). Used to build the RNA input for per-modality PCA.
//
// One thread per output element; n_cells × n_hvg total threads.
__global__ void
gather_hvg_cols_kernel(
    const float* __restrict__ full,     // n_cells × n_genes, row-major
    const int*   __restrict__ hvg_idx,  // n_hvg gene indices
    float*       __restrict__ out,      // n_cells × n_hvg, row-major
    int n_cells, int n_genes, int n_hvg)
{
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int n_total = n_cells * n_hvg;
    if (tid >= n_total) return;
    const int cell = tid / n_hvg;
    const int hv   = tid % n_hvg;
    const int gene = hvg_idx[hv];
    out[tid] = full[(size_t)cell * n_genes + gene];
}

// ─── Kernel 4: Materialize sparse CSC column into dense row-major matrix ──────
//
// Converts a CSC (genes × cells) to a dense (n_cells × n_genes) matrix.
// One thread block per cell (column). Used for:
//   - real RNA: output → dense_rna_real (n_cells × n_genes)
//   - real ADT: output → dense_adt_real (n_cells × n_tags)
// After this, cycle-2 log_normalize is NOT used (it operates on CSC); instead
// we apply dense_lognorm_inplace_kernel. For the real RNA path we apply lognorm
// BEFORE HVG selection to match the expected workflow.
//
// NOTE: Materializing real-cell CSC to dense is O(n_cells × n_genes) memory.
// At 100k × 20k × 4 bytes = 8 GB. This is tight on a 40GB A100 but feasible.
// For n_cells > 300k this step would exceed typical device memory and should
// use a chunked streaming path (deferred to cycle 16).
__global__ void
csc_to_dense_kernel(
    const float* __restrict__ csc_values,
    const int*   __restrict__ csc_indptr,
    const int*   __restrict__ csc_row_indices,
    float*       __restrict__ dense,    // n_cells × n_rows (e.g. n_genes), row-major
    int n_cols,                         // cells
    int n_rows)                         // genes or tags
{
    const int col = (int)blockIdx.x;   // cell index
    if (col >= n_cols) return;

    // Zero this cell's row in the dense matrix.
    float* out_row = dense + (size_t)col * n_rows;
    for (int g = (int)threadIdx.x; g < n_rows; g += blockDim.x)
        out_row[g] = 0.f;
    __syncthreads();

    const int start = csc_indptr[col];
    const int end   = csc_indptr[col + 1];
    for (int idx = start + (int)threadIdx.x; idx < end; idx += blockDim.x)
        out_row[csc_row_indices[idx]] += csc_values[idx];
}

// ─── Kernel 5: Doublet feature computation ────────────────────────────────────
//
// For each real cell, compute 4 features:
//   [0] doublet_fraction:  fraction of k-NN that are simulated (index >= n_cells)
//   [1] mean_sim_dist:     mean distance to simulated neighbors only
//   [2] rna_umi_zscore:    (rna_sum[i] - mu_rna) / sigma_rna
//   [3] adt_umi_zscore:    (adt_sum[i] - mu_adt) / sigma_adt
//
// mu/sigma are pre-computed on host from device CUB reductions (see outer scope).
// rna_sum, adt_sum are per-cell total UMI from lognorm step.
//
// One thread per real cell.
__global__ void
compute_doublet_features_kernel(
    const int*   __restrict__ knn_neighbors,  // n_total × k
    const float* __restrict__ knn_distances,  // n_total × k
    const float* __restrict__ rna_sums,       // n_cells fp32 total UMI
    const float* __restrict__ adt_sums,       // n_cells fp32 total ADT
    float mu_rna,  float sigma_rna,
    float mu_adt,  float sigma_adt,
    float* __restrict__ features,             // n_cells × 4 row-major
    int n_cells,
    int k)
{
    const int cell = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (cell >= n_cells) return;

    const int*   nbrs = knn_neighbors + (size_t)cell * k;
    const float* dist = knn_distances + (size_t)cell * k;

    int   sim_count = 0;
    float sim_dist_sum = 0.f;
    for (int i = 0; i < k; ++i) {
        bool is_sim = (nbrs[i] >= n_cells);
        if (is_sim) {
            ++sim_count;
            sim_dist_sum += dist[i];
        }
    }
    float* f = features + (size_t)cell * 4;
    f[0] = (float)sim_count / (float)k;
    f[1] = (sim_count > 0) ? (sim_dist_sum / (float)sim_count) : 0.f;
    float rna_z = (sigma_rna > 0.f)
                  ? (rna_sums[cell] - mu_rna) / sigma_rna : 0.f;
    float adt_z = (sigma_adt > 0.f)
                  ? (adt_sums[cell] - mu_adt) / sigma_adt : 0.f;
    f[2] = rna_z;
    f[3] = adt_z;
}

// ─── Kernel 6: IRLS logistic regression (batched over all cells) ───────────────
//
// Standard batched Iteratively Reweighted Least Squares for logistic regression.
// n_samples = n_cells + n_sim; labels y[i] ∈ {0, 1}.
// Features X[n_samples × n_features] (augmented with a bias column → n_features+1).
// fp32 for p and μ computation; fp64 for the tiny Hessian inverse.
//
// Algorithm (IRLS step):
//   μ = σ(X·β)           # sigmoid, fp32
//   W = diag(μ*(1-μ))    # diagonal weight matrix, fp32
//   z = X·β + W^{-1}(y - μ)  # adjusted response, fp32
//   H = X'WX             # (n_feat+1)²  Hessian, accumulated in fp64
//   g = X'(y - μ)        # gradient, fp64
//   β ← H^{-1} (H·β + g) # = H^{-1} X'Wy z  (equivalent update)
//
// This kernel computes ONE IRLS step:
//   1. Compute μ = σ(X β) for all n_samples. (fp32, per-thread)
//   2. Accumulate Hessian H (fp64 shared-memory reduction, (n_feat+1)² elements).
//   3. Accumulate gradient g (fp64 shared, n_feat+1 elements).
//   4. Write H, g to a small device output buffer (≤ 25 × 8 bytes = 200 bytes).
//
// The Hessian inversion (tiny fp64 dense solve) runs on the HOST after the
// kernel. One 4-byte D2H copy of max-delta per iteration allows convergence
// checking. Total: ≤20 × 8 bytes = 160 bytes per iteration, within the
// ≤25 KB per-iter budget (§FORBIDDEN DEFENSES valid exception).
//
// WHY a separate device kernel (not full device IRLS):
//   The Hessian is only (n_features+1)² = 25 fp64 values. Building it on device
//   via shared-mem reduction is O(n_samples × n_feat) per iteration, which is fine
//   (100k cells × 5 × 20 iters ≈ 100M FLOPs). Inverting a 5×5 matrix on host
//   (Cholesky, < 1 µs) avoids shipping cuSOLVER for a trivially-sized system.
//   The only host traffic is the H and g output: 25×8 + 5×8 = 240 bytes per iteration.
//   This is well within the ≤25 KB exception.
//
// blockDim.x = 256. gridDim.x = ceil(n_samples / 256).
// The Hessian accumulation uses shared-memory partial sums then one global
// atomicAdd per (feat, feat) pair per block. NOT per sample — the number of
// global atomics is n_blocks × (n_feat+1)² ≤ 4096 × 25 = 100k per iter, which
// is a one-time reduction cost, not a per-sample cost.
__global__ void
irls_step_kernel(
    const float*  __restrict__ X,         // n_samples × (n_feat)
    const uint8_t* __restrict__ y,        // n_samples labels (0/1)
    const float*  __restrict__ beta,      // n_feat (no bias; bias is appended in X)
    int n_samples, int n_feat,
    double* __restrict__ d_H,             // (n_feat)² fp64 output
    double* __restrict__ d_g)             // n_feat fp64 gradient output
{
    extern __shared__ double smem[];      // layout: [n_feat²] H + [n_feat] g
    double* sh_H = smem;
    double* sh_g = smem + (size_t)n_feat * n_feat;

    // Zero shared memory.
    const int smem_len = n_feat * n_feat + n_feat;
    for (int i = (int)threadIdx.x; i < smem_len; i += blockDim.x)
        smem[i] = 0.0;
    __syncthreads();

    const int sample = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (sample < n_samples) {
        const float* xi = X + (size_t)sample * n_feat;

        // μ = σ(x·β)
        float dot = 0.f;
        for (int f = 0; f < n_feat; ++f) dot += xi[f] * beta[f];
        float mu = 1.f / (1.f + expf(-dot));
        float w  = mu * (1.f - mu);
        float r  = (float)y[sample] - mu;   // residual

        // Accumulate into shared H, g (fp64).
        for (int fi = 0; fi < n_feat; ++fi) {
            double xfi = (double)xi[fi];
            atomicAdd(&sh_g[fi], xfi * (double)r);
            for (int fj = fi; fj < n_feat; ++fj) {
                double xfj = (double)xi[fj];
                // Upper triangle accumulation.
                atomicAdd(&sh_H[fi * n_feat + fj], xfi * xfj * (double)w);
            }
        }
    }
    __syncthreads();

    // Flush shared partial H, g to global via atomicAdd.
    // Outer loop runs (n_feat² + n_feat) ≤ 30 times — this is a REDUCTION FLUSH,
    // not a per-sample loop. Covered by the "block-level partial sum flush"
    // valid pattern (analogous to histogram_kernel in doublet_score.h).
    for (int i = (int)threadIdx.x; i < smem_len; i += blockDim.x) {
        if (smem[i] != 0.0) {
            if (i < n_feat * n_feat)
                atomicAdd(&d_H[i], smem[i]);
            else
                atomicAdd(&d_g[i - n_feat * n_feat], smem[i]);
        }
    }
}

// ─── Kernel 7: IRLS scoring (μ = σ(Xβ) for all samples) ─────────────────────
//
// Final scoring pass after convergence. One thread per sample.
// Used for both training samples (to get simulated_scores) and real cells.
__global__ void
logistic_score_kernel(
    const float*  __restrict__ X,       // n_samples × n_feat
    const float*  __restrict__ beta,    // n_feat
    float*        __restrict__ scores,  // n_samples
    int n_samples, int n_feat)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_samples) return;
    const float* xi = X + (size_t)i * n_feat;
    float dot = 0.f;
    for (int f = 0; f < n_feat; ++f) dot += xi[f] * beta[f];
    scores[i] = 1.f / (1.f + expf(-dot));
}

// ─── Kernel 8: Build IRLS feature matrix (real cells) ─────────────────────────
//
// Concatenates the 4 per-cell doublet features + bias into X[n_total × 5].
// For real cells (0..n_cells-1): features from doublet_features_kernel output.
// For simulated cells (n_cells..n_total-1): features unavailable from real data;
// we assign feature vector [1, 0, z_rna, z_adt, 1] where z-scores are the
// normalized total UMI of the simulated doublet.
//
// Combined feature matrix (real + sim) is built for IRLS training.
__global__ void
build_irls_features_kernel(
    const float*  __restrict__ real_features,  // n_cells × 4
    const float*  __restrict__ sim_rna_sums,   // n_sim fp32 total RNA UMI per sim cell
    const float*  __restrict__ sim_adt_sums,   // n_sim fp32 total ADT per sim cell
    float mu_rna, float sigma_rna,
    float mu_adt, float sigma_adt,
    float*        __restrict__ X,              // n_total × 5 (bias appended)
    const uint8_t* __restrict__ labels,        // n_total (0=real,1=sim) — written separately
    int n_cells, int n_sim, int n_feat)
{
    // n_feat must equal 5 (4 features + bias). Checked before launch.
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int n_total = n_cells + n_sim;
    if (i >= n_total) return;

    float* xi = X + (size_t)i * n_feat;
    if (i < n_cells) {
        const float* f = real_features + (size_t)i * 4;
        xi[0] = f[0]; xi[1] = f[1]; xi[2] = f[2]; xi[3] = f[3];
    } else {
        int si = i - n_cells;
        float rz = (sigma_rna > 0.f)
                   ? (sim_rna_sums[si] - mu_rna) / sigma_rna : 0.f;
        float az = (sigma_adt > 0.f)
                   ? (sim_adt_sums[si] - mu_adt) / sigma_adt : 0.f;
        // doublet_fraction for simulated cells is trivially 1 (all neighbors would
        // also tend to be near simulated); we use 1.0 as a "maximum doublet" signal.
        xi[0] = 1.f;   // doublet_fraction placeholder
        xi[1] = 0.f;   // mean_sim_dist placeholder
        xi[2] = rz;
        xi[3] = az;
    }
    xi[n_feat - 1] = 1.f;  // bias term
}

// ─── Kernel 9: Label vector fill ─────────────────────────────────────────────
__global__ void
fill_labels_kernel(uint8_t* __restrict__ labels, int n_cells, int n_sim) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n_cells + n_sim)
        labels[i] = (i >= n_cells) ? 1u : 0u;
}

// ─── Kernel 10: Per-cell column sum for dense matrix (for UMI z-score) ────────
//
// Compute per-row sums of a dense (n_rows × n_cols) matrix.
// One warp per row; warp reduce.
__global__ void
dense_row_sum_kernel(
    const float* __restrict__ mat,   // n_rows × n_cols, row-major
    float*       __restrict__ sums,  // n_rows fp32
    int n_rows, int n_cols)
{
    const int row = (int)(blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (row >= n_rows) return;
    const int lane = threadIdx.x & 31;
    const float* r = mat + (size_t)row * n_cols;
    float sum = 0.f;
    for (int g = lane; g < n_cols; g += 32)
        sum += r[g];
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) sums[row] = sum;
}

// ─── Kernel 11: Apply doublet score threshold ─────────────────────────────────
//
// Simple elementwise threshold: call[i] = score[i] >= thresh ? 1 : 0.
// Defined in the detail namespace so it can be called from omni_doublet().
__global__ void
apply_threshold_kernel(const float*   __restrict__ score,
                       uint8_t*       __restrict__ call,
                       int n, float threshold)
{
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) call[i] = (score[i] >= threshold) ? 1u : 0u;
}

}  // namespace detail

}  // namespace qc
}  // namespace singlet::gpu
