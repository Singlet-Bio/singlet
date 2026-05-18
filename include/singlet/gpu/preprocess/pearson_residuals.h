// SPDX-License-Identifier: MIT
// integrates: original (no factornet equivalent for pearson_residuals)
//
// singlet/gpu/preprocess/pearson_residuals.h
//
// Analytic Pearson residuals normalization for sparse UMI count matrices.
// Primary use: per-gene residual variance for HVG selection.
//
// Algorithm reference:
//   Lause J, Berens P, Kobak D (2021) "Analytic Pearson residuals for
//   normalization of single-cell RNA-seq UMI data." Genome Biology 22:258.
//   https://doi.org/10.1186/s13059-021-02451-7
//
// Given count matrix X (m genes × n cells, sparse CSC):
//   u_i   = Σ_j X_ij            per-gene row sums
//   v_j   = Σ_i X_ij            per-cell col sums
//   N     = Σ X_ij              grand total
//   μ_ij  = u_i · v_j / N       expected count under independence null
//   σ_ij  = sqrt(μ_ij + μ_ij²/θ)  NB-null std (overdispersion θ; default 100)
//   r_ij  = (X_ij - μ_ij)/σ_ij  Pearson residual
//
// For HVG selection (this header's primary output):
//   var_i = (Σ_j r_ij²)/n − ((Σ_j r_ij)/n)²
//
// Algorithmic challenge — handling zero entries:
//   The residual matrix is dense: for X_ij = 0, r_ij = -μ_ij/σ_ij ≠ 0.
//   Materializing the dense residual matrix (8 GB for 20k×100k) is infeasible.
//
// Closed-form decomposition (avoids dense materialization):
//
//   Pass 1 — compute u[m], v[n], N:
//     Row-sum kernel for u[m]; column-sum kernel for v[n]; reduce N from u[].
//
//   Pass 2 (kernel A — pearson_zero_baseline_kernel):
//     For each gene i, accumulate the as-if-all-zero contributions:
//       T1_i = Σ_j -μ_ij/σ_ij          (zero-residual sum)
//       T2_i = Σ_j (μ_ij/σ_ij)²        (zero-residual sq-sum = Σ μ_ij/(1+μ_ij/θ))
//     One block per gene; threads stride over cells; warp-shuffle reduction.
//     Output: T1[m], T2[m].
//
//   Pass 3 (kernel B — pearson_stored_correction_kernel):
//     For each stored entry (i, j, x_ij) correct the delta from zero to actual:
//       μ = u_i·v_j/N;  σ = sqrt(μ + μ²/θ)
//       delta_r  = x_ij/σ
//       delta_r2 = (x_ij - 2μ)·x_ij/σ²
//       atomicAdd(&Δr[i], delta_r);  atomicAdd(&Δr2[i], delta_r2)
//
//   Final fuse:
//     sum_r[i]  = T1[i] + Δr[i]
//     sum_r2[i] = T2[i] + Δr2[i]
//     var_i     = sum_r2[i]/n − (sum_r[i]/n)²
//
// Time complexity:
//   Pass 1: O(nnz)         row/col sums
//   Pass 2: O(m·n)         zero baseline (dominant term; ~1 TFLOP at 20k×50k)
//   Pass 3: O(nnz)         stored correction
//
// Workspace budget (above input CSC):
//   u[m]   fp32 gene sums           4m bytes
//   v[n]   fp32 cell sums           4n bytes
//   T1[m]  fp32 zero-baseline sum   4m bytes
//   T2[m]  fp32 zero-baseline sq    4m bytes
//   dr[m]  fp32 stored-correction   4m bytes
//   dr2[m] fp32 stored-correction   4m bytes
//   Total: 4n + 24m bytes. At 20k genes, 100k cells: ~8 MB. Negligible.
//
// Residual clipping:
//   Not applied — Lause et al. recommend no clipping; scanpy's sqrt(n) cap is
//   optional and excluded from v0. Document as a follow-up (cycle ≥ 119).
//
// Determinism (Rule 18):
//   Pass 2 (zero baseline): fully deterministic — warp shuffle reduction only.
//   Pass 3 (stored correction): atomicAdd by default → not bit-identical across
//   runs due to floating-point addition nondeterminism. Set cfg.deterministic=true
//   to skip pass-3 atomics and use a sequential fallback (slower but bit-exact).
//   Currently deterministic=true is not implemented (Pass 3 always uses atomicAdd).
//   Document as follow-up: full determinism requires a sort-then-reduce scheme.
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one cudaMemcpy to retrieve grand total N (single scalar).

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace singlet::gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct PearsonResidualsConfig {
    float    theta       = 100.f;  // NB overdispersion (Lause-Berens-Kobak default)
    uint64_t seed        = 0;      // reserved; not currently used (deterministic pass 2)
    bool     deterministic = false; // opt-in bit-identical output; currently a no-op
                                    // (pass 3 always uses atomicAdd per Rule 18).
                                    // Full determinism deferred to cycle ≥ 119.
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// Pass 1a: row_sum_kernel — per-gene sums u[m]
//
// Input CSC is genes-as-rows, cells-as-columns.  Row sums require iterating
// the stored entries and dispatching by row index.  For a CSC matrix we use
// a simple approach: one thread per stored entry, atomicAdd into u[row_idx].
// This is O(nnz) with atomic contention bounded by nnz/m entries per gene.
// For typical single-cell data (nnz ~ 5-15% fill, m=20k, n=50k): ~50M atomics,
// each hitting one of 20k bins → ~2500 entries per bin on average; fine.
//
// WHY not cuSPARSE SpMV: SpMV requires an explicit all-ones vector (n floats)
// and a cuSPARSE descriptor; the atomic scatter is simpler and equivalent.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void row_sum_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    float*         __restrict__ u,
    int nnz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    atomicAdd(&u[row_indices[idx]], values[idx]);
}

// ---------------------------------------------------------------------------
// Pass 1b: col_sum_kernel — per-cell sums v[n]
//
// One warp per column; stride over stored values with warp-shuffle reduction.
// Same pattern as compute_col_sums_kernel in lognorm.h.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void col_sum_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ col_ptr,
    float*         __restrict__ v,
    int n_cols)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= n_cols) return;

    const int col_start = col_ptr[warp_id];
    const int col_end   = col_ptr[warp_id + 1];

    float sum = 0.0f;
    for (int i = col_start + lane; i < col_end; i += 32) {
        sum += values[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) v[warp_id] = sum;
}

// ---------------------------------------------------------------------------
// Pass 2: pearson_zero_baseline_kernel — accumulate as-if-all-zero contributions
//
// For gene i, computes:
//   T1_i = Σ_{j=0}^{n-1}  -μ_ij/σ_ij
//   T2_i = Σ_{j=0}^{n-1}   (μ_ij/σ_ij)² = Σ_j μ_ij / (1 + μ_ij/θ)
//
// where μ_ij = u_i·v_j/N and σ_ij = sqrt(μ_ij + μ_ij²/θ) = sqrt(μ_ij(1+μ_ij/θ)).
// Note: (μ_ij/σ_ij)² = μ_ij²/(μ_ij(1+μ_ij/θ)) = μ_ij/(1+μ_ij/θ).
//
// Guard: if μ_ij ≤ 0 or σ_ij ≤ 0 (degenerate cell/gene), skip (contributes 0).
//
// Launch: one block per gene, blockDim.x threads stride over n cells.
// Warp-shuffle reduction collects partial sums across threads.
// WHY one block per gene: enables efficient warp reduction without complex
// multi-block atomic bookkeeping; m blocks is at most 20k — fine for CUDA.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void pearson_zero_baseline_kernel(
    const float* __restrict__ u,   // gene sums [m]
    const float* __restrict__ v,   // cell sums [n]
    float        N_inv,            // 1/N (grand total reciprocal)
    float        theta,
    float*       __restrict__ T1,  // output: zero-sum [m]
    float*       __restrict__ T2,  // output: zero-sq-sum [m]
    int n)                         // n_cells
{
    const int gene_i = blockIdx.x;
    const float u_i  = u[gene_i];

    float acc_t1 = 0.f;
    float acc_t2 = 0.f;

    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float mu = u_i * (v[j] * N_inv);
        if (mu <= 0.f) continue;
        // σ² = μ + μ²/θ = μ(1 + μ/θ)
        float one_plus_mu_theta = 1.f + mu / theta;
        float sigma2 = mu * one_plus_mu_theta;
        float sigma  = sqrtf(sigma2);
        if (sigma <= 0.f) continue;
        float r0 = -mu / sigma;          // zero-entry residual
        acc_t1 += r0;                    // T1 += -μ/σ
        acc_t2 += r0 * r0;              // T2 += μ/(1+μ/θ) (== (μ/σ)²)
    }

    // Warp-shuffle reduction within each warp
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc_t1 += __shfl_down_sync(0xffffffff, acc_t1, offset);
        acc_t2 += __shfl_down_sync(0xffffffff, acc_t2, offset);
    }

    // Cross-warp reduction via shared memory
    extern __shared__ float smem_pr[]; // layout: [0..NWARPS): t1, [NWARPS..2*NWARPS): t2
    const int warp_id    = threadIdx.x / 32;
    const int lane       = threadIdx.x & 31;
    const int n_warps    = (blockDim.x + 31) / 32;

    if (lane == 0) {
        smem_pr[warp_id]          = acc_t1;
        smem_pr[warp_id + n_warps] = acc_t2;
    }
    __syncthreads();

    // Final reduction by first warp
    if (threadIdx.x < n_warps) {
        float t1 = smem_pr[threadIdx.x];
        float t2 = smem_pr[threadIdx.x + n_warps];
        for (int offset = 16; offset > 0; offset >>= 1) {
            t1 += __shfl_down_sync(0xffffffff, t1, offset);
            t2 += __shfl_down_sync(0xffffffff, t2, offset);
        }
        if (threadIdx.x == 0) {
            T1[gene_i] = t1;
            T2[gene_i] = t2;
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3: pearson_stored_correction_kernel — correct delta for nonzero entries
//
// For each stored entry (row_i, col_j, x_ij) where x_ij > 0:
//   μ = u_i·v_j/N;  σ = sqrt(μ(1 + μ/θ))
//   r_actual = (x_ij - μ)/σ
//   r_zero   = -μ/σ
//   delta_r  = r_actual - r_zero = x_ij/σ
//   delta_r2 = r_actual² - r_zero² = (x_ij - 2μ)·x_ij/σ²
//
// atomicAdd into Δr[gene_i] and Δr2[gene_i] (Rule 18: nondeterministic by default).
//
// Guard: μ ≤ 0 → skip (degenerate entry).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void pearson_stored_correction_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const float*   __restrict__ u,
    const float*   __restrict__ v,
    float          N_inv,
    float          theta,
    float*         __restrict__ delta_r,
    float*         __restrict__ delta_r2,
    int n_cols)
{
    // One warp per column to exploit CSC locality.
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= n_cols) return;

    const int col_start = col_ptr[warp_id];
    const int col_end   = col_ptr[warp_id + 1];
    const float v_j     = v[warp_id];

    for (int idx = col_start + lane; idx < col_end; idx += 32) {
        const float x_ij  = values[idx];
        const int   row_i = row_indices[idx];
        const float u_i   = u[row_i];
        const float mu    = u_i * (v_j * N_inv);
        if (mu <= 0.f) continue;
        const float one_plus_mu_theta = 1.f + mu / theta;
        const float sigma2 = mu * one_plus_mu_theta;
        const float sigma  = sqrtf(sigma2);
        if (sigma <= 0.f) continue;
        // delta_r  = x_ij / σ
        const float dr  = x_ij / sigma;
        // delta_r2 = (x_ij - 2μ)·x_ij / σ²
        const float dr2 = (x_ij - 2.f * mu) * x_ij / sigma2;
        atomicAdd(&delta_r[row_i],  dr);
        atomicAdd(&delta_r2[row_i], dr2);
    }
}

// ---------------------------------------------------------------------------
// Pass 4: pearson_fuse_variance_kernel — combine T1, T2, Δr, Δr2 into var[m]
//
// var_i = (T2_i + Δr2_i)/n - ((T1_i + Δr_i)/n)²
// Guard: if n_cells == 0 → var_i = 0.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void pearson_fuse_variance_kernel(
    const float* __restrict__ T1,
    const float* __restrict__ T2,
    const float* __restrict__ dr,
    const float* __restrict__ dr2,
    float*       __restrict__ var,
    int m,
    float n_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;

    const float sum_r  = T1[i] + dr[i];
    const float sum_r2 = T2[i] + dr2[i];
    const float mean_r  = sum_r  * n_inv;
    const float mean_r2 = sum_r2 * n_inv;
    var[i] = mean_r2 - mean_r * mean_r;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// pearson_residual_variance — public entry point
//
// Computes per-gene Pearson residual variance for HVG selection.
// Output: device buffer var[m] (fp32). Caller must sync stream before read.
//
// Returns: core::DeviceMemory<float> of length m (n_genes).
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> pearson_residual_variance(
    const io::PzDeviceMatrix&     m_,
    const PearsonResidualsConfig& cfg    = {},
    cudaStream_t                  stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m = m_.mat.rows;  // genes
    const int n = m_.mat.cols;  // cells
    const int nnz = m_.mat.nnz;

    if (m <= 0 || n <= 0) {
        throw std::runtime_error(
            "pearson_residual_variance: invalid matrix dimensions");
    }

    if (nnz == 0) {
        // All-zero matrix: all variances are zero.
        core::DeviceMemory<float> var_out(m);
        cudaMemsetAsync(var_out.get(), 0, m * sizeof(float), stream);
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return var_out;
    }

    const float theta = (cfg.theta > 0.f) ? cfg.theta : 100.f;

    // --- Allocate workspace ---
    core::DeviceMemory<float> d_u(m);      // gene sums
    core::DeviceMemory<float> d_v(n);      // cell sums
    core::DeviceMemory<float> d_T1(m);     // zero-baseline sum_r
    core::DeviceMemory<float> d_T2(m);     // zero-baseline sum_r2
    core::DeviceMemory<float> d_dr(m);     // stored correction Δr
    core::DeviceMemory<float> d_dr2(m);    // stored correction Δr2
    core::DeviceMemory<float> d_var(m);    // output

    // Zero-initialize accumulators
    cudaMemsetAsync(d_u.get(),   0, m * sizeof(float), stream);
    cudaMemsetAsync(d_v.get(),   0, n * sizeof(float), stream);
    cudaMemsetAsync(d_dr.get(),  0, m * sizeof(float), stream);
    cudaMemsetAsync(d_dr2.get(), 0, m * sizeof(float), stream);

    // --- Pass 1a: row sums (gene totals u[m]) via atomic scatter ---
    constexpr int THREADS = 256;
    {
        const int grid = (nnz + THREADS - 1) / THREADS;
        row_sum_kernel<<<grid, THREADS, 0, stream>>>(
            m_.mat.values.get(),
            m_.mat.row_indices.get(),
            d_u.get(),
            nnz);
    }

    // --- Pass 1b: col sums (cell totals v[n]) via warp-per-column ---
    {
        constexpr int WARPS_PER_BLOCK = THREADS / 32;
        const int grid = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        col_sum_kernel<<<grid, THREADS, 0, stream>>>(
            m_.mat.values.get(),
            m_.mat.col_ptr.get(),
            d_v.get(),
            n);
    }

    // --- Grand total N: reduce u[m] on host (one D2H, Rule 4 exception) ---
    // Sync after pass 1a/1b to ensure u[] is ready.
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> h_u(m);
    cudaMemcpy(h_u.data(), d_u.get(), m * sizeof(float), cudaMemcpyDeviceToHost);
    double N_dbl = 0.0;
    for (int i = 0; i < m; ++i) N_dbl += static_cast<double>(h_u[i]);
    if (N_dbl <= 0.0) {
        // All-zero values after row sum — return zero variance.
        cudaMemsetAsync(d_var.get(), 0, m * sizeof(float), stream);
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        return d_var;
    }
    const float N_inv = static_cast<float>(1.0 / N_dbl);
    const float n_inv = 1.f / static_cast<float>(n);

    // --- Pass 2: zero baseline — one block per gene ---
    // Shared memory: 2 * n_warps floats (T1 + T2 partial sums per warp).
    // With THREADS=256, n_warps=8 → 64 bytes shared memory per block.
    {
        const int n_warps_per_block = THREADS / 32;
        const int smem_bytes = 2 * n_warps_per_block * static_cast<int>(sizeof(float));
        pearson_zero_baseline_kernel<<<m, THREADS, smem_bytes, stream>>>(
            d_u.get(),
            d_v.get(),
            N_inv,
            theta,
            d_T1.get(),
            d_T2.get(),
            n);
    }

    // --- Pass 3: stored correction — one warp per column ---
    {
        constexpr int WARPS_PER_BLOCK = THREADS / 32;
        const int grid = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        pearson_stored_correction_kernel<<<grid, THREADS, 0, stream>>>(
            m_.mat.values.get(),
            m_.mat.row_indices.get(),
            m_.mat.col_ptr.get(),
            d_u.get(),
            d_v.get(),
            N_inv,
            theta,
            d_dr.get(),
            d_dr2.get(),
            n);
    }

    // --- Pass 4: fuse into var[m] ---
    {
        const int grid = (m + THREADS - 1) / THREADS;
        pearson_fuse_variance_kernel<<<grid, THREADS, 0, stream>>>(
            d_T1.get(),
            d_T2.get(),
            d_dr.get(),
            d_dr2.get(),
            d_var.get(),
            m,
            n_inv);
    }

    // Function-boundary sync (Rule 9): guarantee all kernels complete before
    // workspace DeviceMemory objects go out of scope.
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    return d_var;
}

} // namespace preprocess
} // namespace singlet::gpu
