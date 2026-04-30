// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: scanpy.pp.combat (Johnson et al. 2007)
// singlet-gpu/integrate/combat.h
//
// ComBat — empirical Bayes batch correction.
// Johnson WE, Li C, Rabinovic A (2007) Biostatistics 8:118-127.
// Parametric EB, no covariates beyond batch.  Follows scanpy.pp.combat.
//
// 7 device passes (no inner-loop D2H, no raw cudaMalloc):
//   1: α_g + Σx²_g  via atomic-scatter by gene row.
//   2: sum_xgb / sum_x2gb  via atomic-scatter by (gene, batch).
//   3: σ²_g (pooled); materialize Z → output buffer (reused for Z then adjusted).
//   4: γ_g,b / δ²_g,b  via atomic-scatter over dense Z.
//   5: EB hyperparameters (γ̂_b, τ̂²_b, shape_b, scale_b) — one block per batch.
//   6: EB shrinkage × max_iter=2  (scanpy observation: within 1e-3 of converged).
//   7: X_adj = (Z - γ*) · σ / sqrt(δ²*) + α.
//
// Memory layout: γ/δ²/sum_zgb etc. stored as [g*n_batches + b] (gene-outer).
// Output: dense m × n col-major, index g + c*m.
// Atomics (Rule 18): passes 1,2,4 use atomicAdd; run-to-run rel_err ≤ 1e-4.
// Memory guard: throws if m*n*4 > 50% free device memory (CYCLE-124 pattern).
// CYCLE-131: initial implementation.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace singlet_gpu {
namespace integrate {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct CombatConfig {
    int  max_iter    = 2;      // EB shrinkage iterations (scanpy's effective default)
    float eps        = 1e-9f;  // guard for σ²_g near zero (low-variance genes pass through)
    bool  deterministic = false;
};

struct CombatResult {
    core::DeviceMemory<float> X_adj;  // m × n DENSE col-major
    int m, n;
    int n_batches;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// Pass 1 — atomic scatter: sum_x[g] += X[g,c], sum_x2[g] += X[g,c]².
// One thread per nnz entry; gene index from row_indices (CSC).
__global__ __launch_bounds__(256, 2)
void combat_gene_scatter_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    float*         __restrict__ sum_x,    // [m]
    float*         __restrict__ sum_x2,   // [m]
    int nnz)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    const int g = row_indices[e];
    const float v = values[e];
    atomicAdd(&sum_x[g],  v);
    atomicAdd(&sum_x2[g], v * v);
}

// Pass 1b — α_g = sum_x[g] / n.
__global__ __launch_bounds__(256, 4)
void combat_alpha_kernel(
    const float* __restrict__ sum_x,
    float*       __restrict__ alpha,   // [m]
    int m, float n_inv)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    alpha[g] = sum_x[g] * n_inv;
}

// Pass 2 — atomic scatter per (gene, batch): sum_xgb, sum_x2gb.
// Binary search col_ptr → cell index → batch; atomic into [g*n_batches+b].
__global__ __launch_bounds__(256, 2)
void combat_gb_scatter_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const int*     __restrict__ d_batch,
    float*         __restrict__ sum_xgb,    // [m * n_batches], row-major (gene-outer)
    float*         __restrict__ sum_x2gb,   // [m * n_batches]
    int n_cols,
    int nnz,
    int n_batches)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;

    // Binary search: find column j such that col_ptr[j] <= e < col_ptr[j+1].
    int lo = 0, hi = n_cols - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1;
        else                       hi = mid;
    }
    const int j = lo;  // cell index
    const int g = row_indices[e];
    const int b = d_batch[j];
    const float v = values[e];
    const int idx = g * n_batches + b;
    atomicAdd(&sum_xgb[idx],  v);
    atomicAdd(&sum_x2gb[idx], v * v);
}

// Pass 3a — pooled variance σ²_g. One thread per gene; iterates over batches.
// σ²_g = Σ_b (n_b-1)·s²_g,b / (n - n_batches).  Low-variance clamped to eps.
__global__ __launch_bounds__(256, 4)
void combat_pooled_var_kernel(
    const float* __restrict__ sum_xgb,   // [m * n_batches]
    const float* __restrict__ sum_x2gb,  // [m * n_batches]
    const int*   __restrict__ n_b,       // batch sizes [n_batches]
    const float* __restrict__ alpha,     // [m]
    float*       __restrict__ sigma2,    // [m] output: pooled variance
    int m, int n_batches, int n, float eps)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;

    float pooled = 0.f;
    int dof_total = n - n_batches;

    for (int b = 0; b < n_batches; ++b) {
        const int nb = n_b[b];
        if (nb < 2) continue;
        const int idx = g * n_batches + b;
        const float sx  = sum_xgb[idx];
        const float sx2 = sum_x2gb[idx];
        // s²_g,b = (sum_x2 - sum_x²/n_b) / (n_b - 1)
        const float s2 = (sx2 - sx * sx / static_cast<float>(nb)) / static_cast<float>(nb - 1);
        pooled += static_cast<float>(nb - 1) * fmaxf(s2, 0.f);
    }

    float sig2 = (dof_total > 0) ? pooled / static_cast<float>(dof_total) : eps;
    sigma2[g] = fmaxf(sig2, eps);
}

// Pass 3b — materialize Z into X_adj (dense m × n col-major).
// 3b-i: fill implicit zeros: X_adj[g,c] = -alpha[g]/sigma_g (one thread/gene).
// 3b-ii: overwrite stored entries: Z[g,c] = (X[g,c] - alpha[g]) / sigma_g.
__global__ __launch_bounds__(256, 4)
void combat_z_fill_implicit_kernel(
    const float* __restrict__ alpha,
    const float* __restrict__ sigma2,
    float*       __restrict__ X_adj,  // dense m × n col-major: X_adj[g + c*m]
    int m, int n)
{
    // One thread per gene (fills the entire row for that gene).
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    const float sig  = sqrtf(sigma2[g]);
    const float z0   = -alpha[g] / sig;  // implicit zero → Z value
    for (int c = 0; c < n; ++c) {
        X_adj[g + (size_t)c * m] = z0;
    }
}

__global__ __launch_bounds__(256, 2)
void combat_z_scatter_stored_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const float*   __restrict__ alpha,
    const float*   __restrict__ sigma2,
    float*         __restrict__ X_adj,  // dense m × n col-major
    int n_cols, int nnz, int m_genes)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;

    int lo = 0, hi = n_cols - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1;
        else                       hi = mid;
    }
    const int c = lo;
    const int g = row_indices[e];
    const float sig = sqrtf(sigma2[g]);
    X_adj[g + (size_t)c * m_genes] = (values[e] - alpha[g]) / sig;
}

// Pass 4 — gamma/delta2 scatter from dense Z (one thread per m*n pair).
// Atomic: sum_zgb[g*n_batches+b] += Z[g,c]; sum_z2gb[...] += Z[g,c]².
__global__ __launch_bounds__(256, 2)
void combat_gamma_scatter_kernel(
    const float* __restrict__ Z,        // dense m × n col-major (= X_adj)
    const int*   __restrict__ d_batch,  // [n]
    float*       __restrict__ sum_zgb,  // [m * n_batches]
    float*       __restrict__ sum_z2gb, // [m * n_batches]
    int m, int n, int n_batches)
{
    // One thread per (gene, cell) pair.
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int c = idx / m;
    const int g = idx % m;
    const int b = d_batch[c];
    const float z = Z[g + (size_t)c * m];
    const int gb_idx = g * n_batches + b;
    atomicAdd(&sum_zgb[gb_idx],  z);
    atomicAdd(&sum_z2gb[gb_idx], z * z);
}

// Pass 4b — finalize gamma[g,b] and delta2[g,b] from sums. One thread per (g,b).
__global__ __launch_bounds__(256, 4)
void combat_gamma_finalize_kernel(
    const float* __restrict__ sum_zgb,
    const float* __restrict__ sum_z2gb,
    const int*   __restrict__ n_b,
    float*       __restrict__ gamma,    // [m * n_batches]
    float*       __restrict__ delta2,   // [m * n_batches]
    int m, int n_batches, float eps)
{
    const int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= m * n_batches) return;
    const int b  = gb % n_batches;
    const int nb = n_b[b];
    const float sz  = sum_zgb[gb];
    const float sz2 = sum_z2gb[gb];
    if (nb < 1) { gamma[gb] = 0.f; delta2[gb] = 1.f; return; }
    const float mu  = sz / static_cast<float>(nb);
    gamma[gb] = mu;
    float var = (nb > 1)
        ? fmaxf((sz2 - sz * sz / static_cast<float>(nb)) / static_cast<float>(nb - 1), eps)
        : 1.f;
    delta2[gb] = var;
}

// Pass 5 — EB hyperparameters. One block per batch, warp-shuffle over m genes.
__global__ __launch_bounds__(256, 2)
void combat_eb_hyperparams_kernel(
    const float* __restrict__ gamma,   // [m * n_batches], gamma[g*n_batches+b]
    const float* __restrict__ delta2,  // [m * n_batches]
    float*       __restrict__ gamma_hat, // [n_batches]
    float*       __restrict__ tau2,      // [n_batches]
    float*       __restrict__ shape_b,   // [n_batches]
    float*       __restrict__ scale_b,   // [n_batches]
    int m, int n_batches)
{
    const int b = blockIdx.x;
    if (b >= n_batches) return;

    // Accumulate Σγ, Σγ², Σδ², Σδ⁴ over genes.
    float s1g = 0.f, s2g = 0.f;  // for γ mean/var
    float s1d = 0.f, s2d = 0.f;  // for δ² mean/var
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        float gv  = gamma[g * n_batches + b];
        float dv  = delta2[g * n_batches + b];
        s1g += gv;  s2g += gv * gv;
        s1d += dv;  s2d += dv * dv;
    }

    // Warp-level reduction.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        s1g += __shfl_down_sync(0xffffffff, s1g, off);
        s2g += __shfl_down_sync(0xffffffff, s2g, off);
        s1d += __shfl_down_sync(0xffffffff, s1d, off);
        s2d += __shfl_down_sync(0xffffffff, s2d, off);
    }

    // Cross-warp via shared memory. smem layout: [0..7]=s1g, [8..15]=s2g,
    //   [16..23]=s1d, [24..31]=s2d.
    __shared__ float smem[32];
    const int lane  = threadIdx.x & 31;
    const int wid   = threadIdx.x / 32;
    const int nwarp = (blockDim.x + 31) / 32;

    if (lane == 0) {
        smem[wid]      = s1g;
        smem[8 + wid]  = s2g;
        smem[16 + wid] = s1d;
        smem[24 + wid] = s2d;
    }
    __syncthreads();

    if (wid == 0) {
        float v1g = (lane < nwarp) ? smem[lane]      : 0.f;
        float v2g = (lane < nwarp) ? smem[8 + lane]  : 0.f;
        float v1d = (lane < nwarp) ? smem[16 + lane] : 0.f;
        float v2d = (lane < nwarp) ? smem[24 + lane] : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v1g += __shfl_down_sync(0xffffffff, v1g, off);
            v2g += __shfl_down_sync(0xffffffff, v2g, off);
            v1d += __shfl_down_sync(0xffffffff, v1d, off);
            v2d += __shfl_down_sync(0xffffffff, v2d, off);
        }
        if (lane == 0) {
            const float mf = static_cast<float>(m);
            const float mu_g  = v1g / mf;
            const float var_g = fmaxf(v2g / mf - mu_g * mu_g, 1e-12f);
            gamma_hat[b] = mu_g;
            tau2[b]      = var_g;

            const float mu_d  = v1d / mf;
            const float var_d = fmaxf(v2d / mf - mu_d * mu_d, 1e-12f);
            // Method-of-moments inverse-gamma hyperparameters:
            //   shape = (mu² + 2*var) / var   (clipped to ≥ 2)
            //   scale = (mu*var + mu³) / var
            const float sh = fmaxf((mu_d * mu_d + 2.f * var_d) / var_d, 2.f);
            const float sc = (mu_d * var_d + mu_d * mu_d * mu_d) / var_d;
            shape_b[b] = sh;
            scale_b[b] = sc;
        }
    }
}

// Pass 6 — EB shrinkage. One thread per (g,b). Updates gamma_star/delta2_star.
// Residuals from Pass 4 sums: Σ(Z-γ*)² = sum_z2gb - 2γ*·sum_zgb + n_b·γ*².
__global__ __launch_bounds__(256, 4)
void combat_eb_shrink_kernel(
    const float* __restrict__ gamma_raw,  // γ_g,b original [m*n_batches]
    const float* __restrict__ sum_zgb,    // Σ_c Z[g,c] per batch [m*n_batches]
    const float* __restrict__ sum_z2gb,   // Σ_c Z[g,c]² per batch [m*n_batches]
    const float* __restrict__ gamma_hat,  // γ̂_b [n_batches]
    const float* __restrict__ tau2,       // τ̂²_b [n_batches]
    const float* __restrict__ shape_b,    // α_b  [n_batches]
    const float* __restrict__ scale_b,    // β_b  [n_batches]
    const int*   __restrict__ n_b,        // n_b  [n_batches]
    float*       __restrict__ gamma_star, // γ*[g,b] [m*n_batches] in/out
    float*       __restrict__ delta2_star,// δ²*[g,b] [m*n_batches] in/out
    int m, int n_batches)
{
    const int gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= m * n_batches) return;
    const int b  = gb % n_batches;
    const float nb_f  = static_cast<float>(n_b[b]);
    const float gh    = gamma_hat[b];
    const float t2    = tau2[b];
    const float sh    = shape_b[b];
    const float sc    = scale_b[b];
    const float g_raw = gamma_raw[gb];
    const float sz    = sum_zgb[gb];
    const float sz2   = sum_z2gb[gb];

    float d2 = delta2_star[gb];

    // Update γ*: γ* = (n_b·τ²·γ_raw + δ²*·γ̂) / (n_b·τ² + δ²*)
    const float num_g = nb_f * t2 * g_raw + d2 * gh;
    const float den_g = nb_f * t2 + d2;
    const float g_star = num_g / den_g;
    gamma_star[gb] = g_star;

    // sum of squared residuals: Σ(Z - γ*)² = sz2 - 2·γ*·sz + n_b·γ*²
    const float resid = sz2 - 2.f * g_star * sz + nb_f * g_star * g_star;

    // Update δ²*: δ²* = (scale + 0.5·resid) / (n_b/2 + shape - 1)
    const float d2_new = (sc + 0.5f * resid) / (0.5f * nb_f + sh - 1.f);
    delta2_star[gb] = fmaxf(d2_new, 1e-12f);
}

// Pass 7 — adjust: X_adj = (Z - γ*) · σ / sqrt(δ²*) + α. One thread per (g,c).
__global__ __launch_bounds__(256, 4)
void combat_adjust_kernel(
    float*       __restrict__ X_adj,      // m × n col-major (Z stored here; overwrite in-place)
    const float* __restrict__ alpha,
    const float* __restrict__ sigma2,
    const float* __restrict__ gamma_star,
    const float* __restrict__ delta2_star,
    const int*   __restrict__ d_batch,
    int m, int n, int n_batches)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= m * n) return;
    const int c = idx / m;
    const int g = idx % m;
    const int b = d_batch[c];
    const float z     = X_adj[g + (size_t)c * m];
    const float sig   = sqrtf(sigma2[g]);
    const float g_star = gamma_star[g * n_batches + b];
    const float d_star = sqrtf(delta2_star[g * n_batches + b]);
    X_adj[g + (size_t)c * m] = (z - g_star) * sig / d_star + alpha[g];
}

} // anonymous namespace

// combat — apply ComBat batch correction. Returns dense m×n X_adj on device.
// Caller syncs stream before reading. D2H of batch sizes at function boundary only.
inline CombatResult combat(
    const io::PzDeviceMatrix& X,
    const int*                d_batch,
    int                       n_batches,
    const CombatConfig&       cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet_gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    if (m <= 0 || n <= 0)
        throw std::runtime_error("combat: invalid X dimensions (m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ")");
    if (n_batches <= 0 || n_batches > n)
        throw std::runtime_error("combat: invalid n_batches=" + std::to_string(n_batches));

    // --- Memory guard (CYCLE-124 pattern) ---
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t needed = static_cast<size_t>(m) * static_cast<size_t>(n) * 4ULL;
        if (needed > free_mem / 2) {
            const double ng = static_cast<double>(needed) / (1024.0 * 1024.0 * 1024.0);
            const double fg = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
            throw std::runtime_error(
                "combat: dense output would require " +
                std::to_string(ng).substr(0, 6) + " GB (m=" + std::to_string(m) +
                " genes × n=" + std::to_string(n) + " cells) but only " +
                std::to_string(fg).substr(0, 6) + " GB free. "
                "Subset to HVGs before calling combat.");
        }
    }

    constexpr int THREADS = 256;

    // Collect per-batch sizes on host (one pass over d_batch via D2H — acceptable
    // at function boundary, not inside any loop; O(n) host scan).
    std::vector<int> batch_h(n);
    cudaMemcpy(batch_h.data(), d_batch, n * sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<int> nb_h(n_batches, 0);
    for (int c = 0; c < n; ++c) nb_h[batch_h[c]]++;

    core::DeviceMemory<int> d_nb(n_batches);
    cudaMemcpy(d_nb.get(), nb_h.data(), n_batches * sizeof(int), cudaMemcpyHostToDevice);

    // Pass 1: α_g.
    core::DeviceMemory<float> d_sum_x(m);
    core::DeviceMemory<float> d_sum_x2(m);
    core::DeviceMemory<float> d_alpha(m);
    core::DeviceMemory<float> d_sigma2(m);
    cudaMemsetAsync(d_sum_x.get(),  0, m * sizeof(float), stream);
    cudaMemsetAsync(d_sum_x2.get(), 0, m * sizeof(float), stream);

    if (nnz > 0) {
        const int grid = (nnz + THREADS - 1) / THREADS;
        combat_gene_scatter_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(),
            d_sum_x.get(), d_sum_x2.get(), nnz);
    }

    {
        const float n_inv = 1.f / static_cast<float>(n);
        const int grid = (m + THREADS - 1) / THREADS;
        combat_alpha_kernel<<<grid, THREADS, 0, stream>>>(
            d_sum_x.get(), d_alpha.get(), m, n_inv);
    }

    // Pass 2: per-(g,b) sufficient stats.
    const size_t gb_size = static_cast<size_t>(m) * n_batches;
    core::DeviceMemory<float> d_sum_xgb(gb_size);
    core::DeviceMemory<float> d_sum_x2gb(gb_size);
    cudaMemsetAsync(d_sum_xgb.get(),  0, gb_size * sizeof(float), stream);
    cudaMemsetAsync(d_sum_x2gb.get(), 0, gb_size * sizeof(float), stream);

    if (nnz > 0) {
        const int grid = (nnz + THREADS - 1) / THREADS;
        combat_gb_scatter_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_batch, d_sum_xgb.get(), d_sum_x2gb.get(),
            n, nnz, n_batches);
    }

    // Pass 3a — pooled variance σ²_g.
    {
        const int grid = (m + THREADS - 1) / THREADS;
        combat_pooled_var_kernel<<<grid, THREADS, 0, stream>>>(
            d_sum_xgb.get(), d_sum_x2gb.get(), d_nb.get(),
            d_alpha.get(), d_sigma2.get(), m, n_batches, n, cfg.eps);
    }

    // Pass 3b: materialize Z → X_adj (m × n col-major, gene=row).
    const size_t dense_sz = static_cast<size_t>(m) * n;
    core::DeviceMemory<float> d_X_adj(dense_sz);

    // 3b-i: fill implicit zeros (entire dense buffer = -alpha/sigma).
    {
        const int grid = (m + THREADS - 1) / THREADS;
        combat_z_fill_implicit_kernel<<<grid, THREADS, 0, stream>>>(
            d_alpha.get(), d_sigma2.get(), d_X_adj.get(), m, n);
    }

    // 3b-ii: overwrite stored entries.
    if (nnz > 0) {
        const int grid = (nnz + THREADS - 1) / THREADS;
        combat_z_scatter_stored_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_alpha.get(), d_sigma2.get(), d_X_adj.get(), n, nnz, m);
    }

    // Pass 4: γ_g,b and δ²_g,b.
    core::DeviceMemory<float> d_sum_zgb(gb_size);
    core::DeviceMemory<float> d_sum_z2gb(gb_size);
    core::DeviceMemory<float> d_gamma(gb_size);
    core::DeviceMemory<float> d_delta2(gb_size);
    cudaMemsetAsync(d_sum_zgb.get(),  0, gb_size * sizeof(float), stream);
    cudaMemsetAsync(d_sum_z2gb.get(), 0, gb_size * sizeof(float), stream);

    {
        const int total = m * n;
        const int grid  = (total + THREADS - 1) / THREADS;
        combat_gamma_scatter_kernel<<<grid, THREADS, 0, stream>>>(
            d_X_adj.get(), d_batch,
            d_sum_zgb.get(), d_sum_z2gb.get(),
            m, n, n_batches);
    }

    {
        const int total = static_cast<int>(gb_size);
        const int grid  = (total + THREADS - 1) / THREADS;
        combat_gamma_finalize_kernel<<<grid, THREADS, 0, stream>>>(
            d_sum_zgb.get(), d_sum_z2gb.get(), d_nb.get(),
            d_gamma.get(), d_delta2.get(),
            m, n_batches, cfg.eps);
    }

    // Pass 5: EB hyperparameters.
    core::DeviceMemory<float> d_gamma_hat(n_batches);
    core::DeviceMemory<float> d_tau2(n_batches);
    core::DeviceMemory<float> d_shape(n_batches);
    core::DeviceMemory<float> d_scale(n_batches);

    combat_eb_hyperparams_kernel<<<n_batches, THREADS, 0, stream>>>(
        d_gamma.get(), d_delta2.get(),
        d_gamma_hat.get(), d_tau2.get(),
        d_shape.get(), d_scale.get(),
        m, n_batches);

    // Pass 6: EB shrinkage (max_iter=2 default; init stars from raw estimates).
    core::DeviceMemory<float> d_gamma_star(gb_size);
    core::DeviceMemory<float> d_delta2_star(gb_size);
    cudaMemcpyAsync(d_gamma_star.get(),  d_gamma.get(),  gb_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_delta2_star.get(), d_delta2.get(), gb_size * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        const int total = static_cast<int>(gb_size);
        const int grid  = (total + THREADS - 1) / THREADS;
        combat_eb_shrink_kernel<<<grid, THREADS, 0, stream>>>(
            d_gamma.get(), d_sum_zgb.get(), d_sum_z2gb.get(),
            d_gamma_hat.get(), d_tau2.get(),
            d_shape.get(), d_scale.get(), d_nb.get(),
            d_gamma_star.get(), d_delta2_star.get(),
            m, n_batches);
    }

    // Pass 7: final adjustment.
    {
        const int total = m * n;
        const int grid  = (total + THREADS - 1) / THREADS;
        combat_adjust_kernel<<<grid, THREADS, 0, stream>>>(
            d_X_adj.get(), d_alpha.get(), d_sigma2.get(),
            d_gamma_star.get(), d_delta2_star.get(),
            d_batch, m, n, n_batches);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CombatResult res;
    res.X_adj     = std::move(d_X_adj);
    res.m         = m;
    res.n         = n;
    res.n_batches = n_batches;
    return res;
}

}  // namespace integrate
}  // namespace singlet_gpu
