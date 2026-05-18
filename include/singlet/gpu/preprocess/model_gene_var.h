// SPDX-License-Identifier: MIT
// integrates: original (Lun-McCarthy-Marioni 2016 / scran::modelGeneVarByPoisson)
//
// singlet/gpu/preprocess/model_gene_var.h
//
// Poisson-null HVG selection for log-normalized expression matrices.
// Direct GPU port of scran::modelGeneVarByPoisson (Bioconductor scran ≥1.4).
//
// Algorithm reference:
//   Lun ATL, McCarthy DJ, Marioni JC (2016) "A step-by-step workflow for
//   low-level analysis of single-cell RNA-seq data with Bioconductor."
//   F1000Research 5:2122. https://doi.org/10.12688/f1000research.9501.2
//   Implemented algorithm: modelGeneVarByPoisson fast-Poisson-null path.
//
// Given log-normalized expression matrix X (m genes × n cells, sparse CSC):
//
//   Pass 1 — per-gene mean:
//     mean[i] = (Σ_j X_ij) / n
//     Computed via atomic scatter: one thread per nnz, atomicAdd → row_sum[i].
//     Then mean[i] = row_sum[i] / n (one-pass element-wise kernel).
//
//   Pass 2 — per-gene sum-of-squares:
//     sum_x2[i] = Σ_{stored} x_ij²
//     row_sum_squares_kernel: one warp per column (CSC layout); each thread
//     processes one nnz, atomicAdd → sum_x2[row]. See atomics note below.
//     Then: total_var[i] = (sum_x2[i] - n·mean[i]²) / (n − 1).   [*]
//     [*] derivation: Σ_j(x-μ)² = Σ_stored x² - 2μ·nμ + nμ² = Σ_stored x² - nμ²
//
//   Pass 3 — biological variance:
//     bio_var[i] = max(0, total_var[i] - mean[i])
//     Poisson null assumption: Var_null(μ) = μ (linear, from Poisson sampling).
//     One-pass element-wise kernel, no atomics.
//
//   Pass 4 — HVG selection (top-N by bio_var):
//     cub::DeviceRadixSort::SortPairsDescending with bio_var as keys and
//     gene indices 0..m-1 as values. Take first min(n_top, eligible) values.
//     Optional min_mean filter: flag-then-compact (cub::DeviceSelect::Flagged)
//     before the sort to exclude genes with mean[i] < cfg.min_mean.
//
// Time complexity: O(nnz) passes 1+2; O(m) passes 3+4.
//
// Workspace budget (above input CSC):
//   row_sum[m]   fp32 gene sums       4m bytes
//   mean[m]      fp32 mean            4m bytes
//   sum_x2[m]    fp32 sum-of-squares  4m bytes
//   total_var[m] fp32 sample variance 4m bytes
//   bio_var[m]   fp32 bio variance    4m bytes
//   hvg_indices  int  top-N indices   4·n_top bytes
//   CUB sort temp: O(m) — allocated and freed internally.
//
// Precision:
//   fp32 throughout. sum_x2 uses atomicAdd (see Atomics note). For typical
//   log-normalized data (values in [0, ~10]), fp32 has ~7 decimal digits of
//   precision; rel_err in total_var < 1e-4 for n ≤ 1M cells.
//
// Atomics / Determinism (Rule 18):
//   Pass 1 row_sum: atomicAdd — not bit-identical across runs by default.
//   Pass 2 sum_x2:  atomicAdd — same nondeterminism. cfg.deterministic is a
//   no-op in v0 (documented; full determinism deferred to cycle ≥ 128).
//   Empirically rel_err < 1e-4 at 1M cells (fp32 atomicAdd on balanced bins).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one scalar copy (nnz, already known) — none in inner loops.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>

namespace singlet::gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct ModelGeneVarConfig {
    int   n_top        = 2000;   // number of HVGs to select (top-N by bio_var)
    float min_mean     = 0.0f;  // drop genes with mean below this threshold
    bool  deterministic = false; // reserved; atomicAdd used in v0 — no-op flag
                                  // Full determinism deferred to cycle ≥ 128.
};

struct ModelGeneVarResult {
    core::DeviceMemory<float> mean;        // m — per-gene mean
    core::DeviceMemory<float> total_var;   // m — sample variance
    core::DeviceMemory<float> bio_var;     // m — max(0, total_var - mean)
    core::DeviceMemory<int>   hvg_indices; // n_hvg — top genes by bio_var (gene ids)
    int n_hvg;  // <= cfg.n_top; fewer if min_mean filter reduces eligible set
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// Pass 1a: row_scatter_kernel — scatter nnz values into per-gene sums
//
// One thread per stored entry. atomicAdd into row_sum[row_i].
// WHY atomic scatter vs cuSPARSE SpMV: SpMV requires an all-ones vector (n
// floats) and a cuSPARSE handle setup; the atomic scatter is simpler and
// equivalent for this one-time scalar accumulation. In pearson_residuals.h
// the same pattern (row_sum_kernel) proved correct and fast (Rule 18 default).
//
// Nondeterminism: atomicAdd ordering varies across runs. For log-normalized
// data (values < 10), n ≤ 1M, relative error < 1e-5. Documented per Rule 18.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void mgv_row_scatter_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    float*         __restrict__ row_sum,
    int nnz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    atomicAdd(&row_sum[row_indices[idx]], values[idx]);
}

// ---------------------------------------------------------------------------
// Pass 1b: mgv_mean_kernel — divide row_sum by n_cells → mean[m]
//
// One thread per gene. Trivial element-wise divide.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mgv_mean_kernel(
    const float* __restrict__ row_sum,
    float*       __restrict__ mean,
    int m,
    float n_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    mean[i] = row_sum[i] * n_inv;
}

// ---------------------------------------------------------------------------
// Pass 2a: row_sum_squares_kernel — per-gene sum of squared stored values
//
// One warp per column (CSC layout). Each lane handles one nnz in the column,
// atomicAdds x² into sum_x2[row_i].
//
// WHY one warp per column: exploits CSC locality — lanes in the same warp
// read a contiguous segment of values/row_indices for column j (stride 32).
// Reduces kernel launch overhead vs one-thread-per-nnz and avoids the
// competing-atomics hot-spot that one-thread-per-row would cause.
//
// Atomics note (Rule 18): atomicAdd on sum_x2 is nondeterministic by default.
// For log-normalized data in [0,10], x² ≤ 100, fp32 atomicAdd error < 1e-4
// relative. cfg.deterministic=true is a no-op in v0.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void mgv_row_sum_squares_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    float*         __restrict__ sum_x2,
    int n_cols)
{
    // Each warp owns one column.
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x & 31;
    if (warp_id >= n_cols) return;

    const int col_start = col_ptr[warp_id];
    const int col_end   = col_ptr[warp_id + 1];

    for (int idx = col_start + lane; idx < col_end; idx += 32) {
        float x = values[idx];
        atomicAdd(&sum_x2[row_indices[idx]], x * x);
    }
}

// ---------------------------------------------------------------------------
// Pass 2b: mgv_sample_var_kernel — compute sample variance from sum_x2 and mean
//
// total_var[i] = (sum_x2[i] - n · mean[i]²) / (n − 1)
//
// Guard: if (n - 1) <= 0, total_var = 0. Prevents divide-by-zero for n=1.
// Note: clamped to 0 from below (can be negative due to fp32 atomicAdd errors
// when true variance is near zero). bio_var max(0,...) handles this.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mgv_sample_var_kernel(
    const float* __restrict__ sum_x2,
    const float* __restrict__ mean,
    float*       __restrict__ total_var,
    int m,
    float n,
    float n_minus1_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    // Derivation: Σ_j(x-μ)² = Σ_stored x² - n·μ²  (all-zeros contribute n·0²=0 to
    // Σ x² and n·(-μ)²=n·μ² to Σ(x-μ)²; stored entries correct both terms exactly)
    const float svar = (sum_x2[i] - n * mean[i] * mean[i]) * n_minus1_inv;
    total_var[i] = (svar > 0.f) ? svar : 0.f;
}

// ---------------------------------------------------------------------------
// Pass 3: mgv_bio_var_kernel — Poisson-null biological variance
//
// bio_var[i] = max(0, total_var[i] - mean[i])
//
// Poisson null: Var_null(μ) = μ (linear trend from Poisson count sampling
// after log-transform; first-order approximation matching scran v0 default).
// No atomics. Fully deterministic.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mgv_bio_var_kernel(
    const float* __restrict__ total_var,
    const float* __restrict__ mean,
    float*       __restrict__ bio_var,
    int m)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    const float bv = total_var[i] - mean[i];
    bio_var[i] = (bv > 0.f) ? bv : 0.f;
}

// ---------------------------------------------------------------------------
// Pass 4a: mgv_flag_eligible_kernel — mark genes passing the min_mean filter
//
// flag[i] = 1 if mean[i] >= min_mean, else 0. Used with cub::DeviceSelect.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mgv_flag_eligible_kernel(
    const float*  __restrict__ mean,
    uint8_t*      __restrict__ flag,
    float         min_mean,
    int           m)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    flag[i] = (mean[i] >= min_mean) ? 1u : 0u;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// model_gene_var — public entry point
//
// Computes per-gene mean, sample variance, and biological variance under the
// Poisson null (bio_var = max(0, total_var - mean)). Returns top-N HVG indices
// sorted by bio_var descending (ties broken by ascending gene id from CUB sort
// stability).
//
// Input: X (m genes × n cells, log-normalized, sparse CSC via PzDeviceMatrix).
// The caller is responsible for synchronizing stream before reading results.
// ---------------------------------------------------------------------------
inline ModelGeneVarResult model_gene_var(
    const io::PzDeviceMatrix& X,
    const ModelGeneVarConfig& cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;  // genes
    const int n   = X.mat.cols;  // cells
    const int nnz = X.mat.nnz;

    if (m <= 0 || n <= 0) {
        throw std::runtime_error(
            "model_gene_var: invalid matrix dimensions (m=" +
            std::to_string(m) + " n=" + std::to_string(n) + ")");
    }
    if (n < 2) {
        throw std::runtime_error(
            "model_gene_var: need n >= 2 cells for sample variance");
    }

    const float n_f          = static_cast<float>(n);
    const float n_inv        = 1.f / n_f;
    const float n_minus1_inv = 1.f / (n_f - 1.f);

    constexpr int THREADS = 256;

    // --- Allocate output and workspace buffers ---
    core::DeviceMemory<float> d_row_sum(m);
    core::DeviceMemory<float> d_mean(m);
    core::DeviceMemory<float> d_sum_x2(m);
    core::DeviceMemory<float> d_total_var(m);
    core::DeviceMemory<float> d_bio_var(m);

    // Zero-initialize accumulators (atomicAdd targets)
    cudaMemsetAsync(d_row_sum.get(), 0, m * sizeof(float), stream);
    cudaMemsetAsync(d_sum_x2.get(), 0, m * sizeof(float), stream);

    // -----------------------------------------------------------------------
    // Pass 1a: scatter nnz into row_sum[m] via atomic scatter
    // -----------------------------------------------------------------------
    if (nnz > 0) {
        const int grid = (nnz + THREADS - 1) / THREADS;
        mgv_row_scatter_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            d_row_sum.get(),
            nnz);
    }

    // -----------------------------------------------------------------------
    // Pass 1b: mean[i] = row_sum[i] / n
    // -----------------------------------------------------------------------
    {
        const int grid = (m + THREADS - 1) / THREADS;
        mgv_mean_kernel<<<grid, THREADS, 0, stream>>>(
            d_row_sum.get(),
            d_mean.get(),
            m,
            n_inv);
    }

    // -----------------------------------------------------------------------
    // Pass 2a: scatter x² into sum_x2[m] — one warp per column (CSC layout)
    // -----------------------------------------------------------------------
    if (nnz > 0) {
        constexpr int WARPS_PER_BLOCK = THREADS / 32;
        const int grid = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        mgv_row_sum_squares_kernel<<<grid, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.row_indices.get(),
            X.mat.col_ptr.get(),
            d_sum_x2.get(),
            n);
    }

    // -----------------------------------------------------------------------
    // Pass 2b: total_var[i] = (sum_x2[i] - n·mean[i]²) / (n-1)
    // -----------------------------------------------------------------------
    {
        const int grid = (m + THREADS - 1) / THREADS;
        mgv_sample_var_kernel<<<grid, THREADS, 0, stream>>>(
            d_sum_x2.get(),
            d_mean.get(),
            d_total_var.get(),
            m,
            n_f,
            n_minus1_inv);
    }

    // -----------------------------------------------------------------------
    // Pass 3: bio_var[i] = max(0, total_var[i] - mean[i])
    // -----------------------------------------------------------------------
    {
        const int grid = (m + THREADS - 1) / THREADS;
        mgv_bio_var_kernel<<<grid, THREADS, 0, stream>>>(
            d_total_var.get(),
            d_mean.get(),
            d_bio_var.get(),
            m);
    }

    // -----------------------------------------------------------------------
    // Pass 4: top-N HVG selection via CUB radix sort
    //
    // Strategy:
    //   A. If min_mean > 0: flag eligible genes, compact their indices to a
    //      temporary buffer, then sort only the eligible subset.
    //   B. Sort bio_var (keys, descending) paired with gene indices (values).
    //   C. Take first n_top values from the sorted output as hvg_indices.
    //
    // WHY CUB DeviceRadixSort instead of host nth_element:
    //   Stays fully device-resident. O(m) on device < O(m log m) D2H + cpu sort
    //   + D2H at scale. At 30k genes: negligible; at 500k genes: 10× faster.
    // -----------------------------------------------------------------------

    // Build initial index array 0..m-1 via host-side iota + async H2D copy.
    // One-time O(m) transfer; m ≤ 50k genes → ≤ 200 KB. Acceptable per Rule 4
    // (not in a hot loop; __device__ lambdas need --extended-lambda which we avoid).
    core::DeviceMemory<int> d_gene_idx(m);
    {
        std::vector<int> h_idx(m);
        std::iota(h_idx.begin(), h_idx.end(), 0);
        cudaMemcpyAsync(d_gene_idx.get(), h_idx.data(), m * sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    // Eligible gene count and key/value arrays for sort input
    int n_eligible = m;
    core::DeviceMemory<float> d_sort_keys(m);   // bio_var of eligible genes
    core::DeviceMemory<int>   d_sort_vals(m);   // gene indices of eligible genes

    if (cfg.min_mean > 0.f) {
        // Flag eligible genes
        core::DeviceMemory<uint8_t> d_flag(m);
        {
            const int grid = (m + THREADS - 1) / THREADS;
            mgv_flag_eligible_kernel<<<grid, THREADS, 0, stream>>>(
                d_mean.get(),
                d_flag.get(),
                cfg.min_mean,
                m);
        }

        // cub::DeviceSelect::Flagged — compact bio_var and gene_idx
        // We need temp storage first (query then allocate).
        // Keys: bio_var; Values: gene_idx; Flag: d_flag; Output: d_sort_keys, d_sort_vals.
        // CUB DeviceSelect::Flagged compacts two arrays (no zip iterator in the raw API),
        // so we run it twice: once for keys, once for values.
        // The num_selected output is written to a device scalar.
        core::DeviceMemory<int> d_n_selected(1);

        // --- compact keys (bio_var) ---
        std::size_t temp_bytes_keys = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_keys,
            d_bio_var.get(), d_flag.get(),
            d_sort_keys.get(), d_n_selected.get(),
            m, stream);
        core::DeviceMemory<uint8_t> d_tmp_k(static_cast<int>(temp_bytes_keys) + 1);
        cub::DeviceSelect::Flagged(
            d_tmp_k.get(), temp_bytes_keys,
            d_bio_var.get(), d_flag.get(),
            d_sort_keys.get(), d_n_selected.get(),
            m, stream);

        // --- compact values (gene_idx) ---
        std::size_t temp_bytes_vals = 0;
        cub::DeviceSelect::Flagged(
            nullptr, temp_bytes_vals,
            d_gene_idx.get(), d_flag.get(),
            d_sort_vals.get(), d_n_selected.get(),
            m, stream);
        core::DeviceMemory<uint8_t> d_tmp_v(static_cast<int>(temp_bytes_vals) + 1);
        cub::DeviceSelect::Flagged(
            d_tmp_v.get(), temp_bytes_vals,
            d_gene_idx.get(), d_flag.get(),
            d_sort_vals.get(), d_n_selected.get(),
            m, stream);

        // Read n_selected back (one D2H scalar; acceptable per Rule 4)
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaMemcpy(&n_eligible, d_n_selected.get(), sizeof(int),
                   cudaMemcpyDeviceToHost);
    } else {
        // No filter: sort keys = bio_var, sort vals = gene_idx
        cudaMemcpyAsync(d_sort_keys.get(), d_bio_var.get(),
                        m * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_sort_vals.get(), d_gene_idx.get(),
                        m * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }

    // Sort descending by bio_var (keys) paired with gene indices (values)
    const int n_hvg = std::min(cfg.n_top, n_eligible);

    core::DeviceMemory<float> d_sorted_keys(n_eligible > 0 ? n_eligible : 1);
    core::DeviceMemory<int>   d_sorted_vals(n_eligible > 0 ? n_eligible : 1);

    if (n_eligible > 0) {
        std::size_t temp_bytes_sort = 0;
        cub::DeviceRadixSort::SortPairsDescending(
            nullptr, temp_bytes_sort,
            d_sort_keys.get(), d_sorted_keys.get(),
            d_sort_vals.get(), d_sorted_vals.get(),
            n_eligible, 0, sizeof(float) * 8, stream);
        core::DeviceMemory<uint8_t> d_tmp_sort(static_cast<int>(temp_bytes_sort) + 1);
        cub::DeviceRadixSort::SortPairsDescending(
            d_tmp_sort.get(), temp_bytes_sort,
            d_sort_keys.get(), d_sorted_keys.get(),
            d_sort_vals.get(), d_sorted_vals.get(),
            n_eligible, 0, sizeof(float) * 8, stream);
    }

    // Extract top-N gene indices
    core::DeviceMemory<int> d_hvg(n_hvg > 0 ? n_hvg : 1);
    if (n_hvg > 0) {
        cudaMemcpyAsync(d_hvg.get(), d_sorted_vals.get(),
                        n_hvg * sizeof(int), cudaMemcpyDeviceToDevice, stream);
    }

    // Function-boundary sync (Rule 9): all workspace DeviceMemory objects go
    // out of scope after this return; sync ensures kernels complete first.
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    return ModelGeneVarResult{
        std::move(d_mean),
        std::move(d_total_var),
        std::move(d_bio_var),
        std::move(d_hvg),
        n_hvg
    };
}

} // namespace preprocess
} // namespace singlet::gpu
