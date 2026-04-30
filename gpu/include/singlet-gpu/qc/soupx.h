// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: SoupX ambient RNA correction (Young & Behjati 2020)
// qc/soupx.h — GPU-native ambient-RNA decontamination for 10X droplet data.
// Young MD, Behjati S (2020) GigaScience 9:giaa151.
//
// 5 GPU passes:
//   (1) per-droplet UMI total t[j]: warp-per-column sum from CSC X;
//   (2) ambient profile a[g] from empty droplets (t[j] <= lower), normalize → π[g];
//   (3) top-ambient-gene mask: one D2H of π, host sort + threshold, H2D mask;
//   (4) per-cell ρ_c: warp-per-column, using top_amb_mask;
//   (5) correct counts: dense output (cudaMemset + nnz-overwrite);
//       stored entries: corrected[g,c] = max(0, x - ρ_c * t_c * π[g]);
//       unstored zeros: max(0, 0 - ...) = 0 → stay zero from cudaMemset.
// Memory guard: throws if m*n*4 > 0.5 * free GPU memory.
// Rules: 5 (DeviceMemory), 4 (one-shot D2H only), 18 (no atomics in correction),
//        31 (<= 350 LOC).
// CYCLE-141: initial implementation.

#pragma once
#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace qc {

struct SoupxConfig {
    int   lower            = 100;    // droplet UMI threshold for "obviously empty"
    float top_ambient_frac = 0.10f;  // fraction of genes treated as ambient markers
    float min_rho          = 0.0f;   // floor for per-cell contamination fraction
    float max_rho          = 0.9f;   // cap  for per-cell contamination fraction
    bool  deterministic    = true;
};

struct SoupxResult {
    core::DeviceMemory<float> corrected;    // m × n DENSE col-major fp32 corrected counts
    core::DeviceMemory<float> rho_c;         // n_cells: per-cell contamination fraction
    core::DeviceMemory<float> ambient_pi;    // m: estimated ambient profile (sums to 1)
    int m, n;
};

namespace detail {

static constexpr int SX_THREADS = 256;

// Pass 1: warp-per-column UMI sum → t[j].
__global__ __launch_bounds__(256, 2)
void sx_col_sum_kernel(const float* __restrict__ values,
                       const int32_t* __restrict__ col_ptr,
                       int32_t* __restrict__ t, int n_cols)
{
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_cols) return;
    float acc = 0.f;
    for (int i = col_ptr[wid] + lane; i < col_ptr[wid + 1]; i += 32)
        acc += values[i];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) t[wid] = static_cast<int32_t>(acc + 0.5f);
}

// Pass 2: atomic-scatter values into ambient[g] for empty columns.
__global__ __launch_bounds__(256, 2)
void sx_ambient_scatter_kernel(
    const float* __restrict__ values, const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr, const int32_t* __restrict__ t,
    float* __restrict__ ambient, int n_cols, int nnz, int lower)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    int lo = 0, hi = n_cols - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1; else hi = mid;
    }
    if (t[lo] > lower) return;
    atomicAdd(&ambient[row_indices[e]], values[e]);
}

// Pass 2c: normalize ambient → π[g] in-place.
__global__ __launch_bounds__(256, 4)
void sx_normalize_ambient_kernel(float* __restrict__ ambient, int m, float total_inv)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    ambient[g] *= total_inv;
}

// Pass 4: per-cell ρ_c — warp-per-column over cells (t[j] > lower).
// top_mask[g] = 1 if π[g] is in the top-ambient fraction; 0 otherwise.
// denom_pi = Σ_g [top_mask[g] * π[g]]  (precomputed host-side, passed as scalar).
__global__ __launch_bounds__(256, 2)
void sx_rho_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const int32_t* __restrict__ t,
    const uint8_t* __restrict__ top_mask,
    const float*   __restrict__ pi,
    float*         __restrict__ rho_c,
    int n_cols, float denom_pi, float min_rho, float max_rho, int lower)
{
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_cols) return;
    if (t[wid] <= lower) { if (lane == 0) rho_c[wid] = 0.f; return; }

    const float tc = static_cast<float>(t[wid]);
    float num = 0.f;
    for (int i = col_ptr[wid] + lane; i < col_ptr[wid + 1]; i += 32) {
        if (top_mask[row_indices[i]])
            num += values[i];
    }
    for (int off = 16; off > 0; off >>= 1)
        num += __shfl_down_sync(0xffffffffu, num, off);

    if (lane == 0) {
        float rho = 0.f;
        if (denom_pi > 0.f && tc > 0.f)
            rho = num / (tc * denom_pi);
        rho = fmaxf(min_rho, fminf(max_rho, rho));
        rho_c[wid] = rho;
    }
}

// Pass 5a: per-element correction over stored nnz entries.
// corrected[g, c] = max(0, x[g,c] - rho_c[c] * t[c] * pi[g]).
// Implicit zeros remain 0 (set via cudaMemset before this kernel).
__global__ __launch_bounds__(256, 2)
void sx_correct_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr,
    const float*   __restrict__ pi,
    const float*   __restrict__ rho_c,
    const int32_t* __restrict__ t,
    float*         __restrict__ corrected,   // dense m × n col-major
    int n_cols, int nnz, int m_genes)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    int lo = 0, hi = n_cols - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1; else hi = mid;
    }
    const int c = lo;
    const int g = row_indices[e];
    const float tc   = static_cast<float>(t[c]);
    const float rho  = rho_c[c];
    const float sub  = rho * tc * pi[g];
    corrected[g + (size_t)c * m_genes] = fmaxf(0.f, values[e] - sub);
}

} // namespace detail

// ---------------------------------------------------------------------------
// soupx — public entry point
// ---------------------------------------------------------------------------
inline SoupxResult
soupx(const io::PzDeviceMatrix& X, const SoupxConfig& cfg = {},
      cudaStream_t stream = nullptr)
{
    if (stream == nullptr) stream = singlet_gpu::core::default_context().stream();

    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    if (m <= 0 || n <= 0)
        throw std::runtime_error("soupx: invalid matrix (m=" +
            std::to_string(m) + " n=" + std::to_string(n) + ")");

    // Memory guard: dense corrected is m*n*4 bytes.
    {
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);
        const size_t needed = static_cast<size_t>(m) * static_cast<size_t>(n) * 4ULL;
        if (needed > free_mem / 2) {
            const double ng = static_cast<double>(needed) / (1024.0 * 1024.0 * 1024.0);
            const double fg = static_cast<double>(free_mem) / (1024.0 * 1024.0 * 1024.0);
            throw std::runtime_error(
                "soupx: dense corrected output would require " +
                std::to_string(ng).substr(0, 6) + " GB (m=" + std::to_string(m) +
                " × n=" + std::to_string(n) + ") but only " +
                std::to_string(fg).substr(0, 6) + " GB free. Subset to HVGs first.");
        }
    }

    constexpr int T = detail::SX_THREADS;

    // Pass 1: UMI totals t[j].
    core::DeviceMemory<int32_t> d_t(n);
    cudaMemsetAsync(d_t.get(), 0, n * sizeof(int32_t), stream);
    if (nnz > 0) {
        constexpr int WPB = T / 32;
        detail::sx_col_sum_kernel<<<(n + WPB - 1) / WPB, T, 0, stream>>>(
            X.mat.values.get(), X.mat.col_ptr.get(), d_t.get(), n);
    }

    // Pass 2: ambient scatter from empty droplets.
    core::DeviceMemory<float> d_ambient(m);
    cudaMemsetAsync(d_ambient.get(), 0, m * sizeof(float), stream);
    if (nnz > 0) {
        detail::sx_ambient_scatter_kernel<<<(nnz + T - 1) / T, T, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_t.get(), d_ambient.get(), n, nnz, cfg.lower);
    }

    // Reduce ambient total (one D2H scalar at boundary).
    core::DeviceMemory<float> d_total(1);
    {
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes,
                               d_ambient.get(), d_total.get(), m, stream);
        core::DeviceMemory<uint8_t> d_tmp(tmp_bytes + 1);
        cub::DeviceReduce::Sum(d_tmp.get(), tmp_bytes,
                               d_ambient.get(), d_total.get(), m, stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float h_total = 0.f;
    cudaMemcpy(&h_total, d_total.get(), sizeof(float), cudaMemcpyDeviceToHost);
    if (h_total <= 0.f)
        throw std::runtime_error("soupx: no empty droplets found (lower=" +
            std::to_string(cfg.lower) + "). Increase lower threshold.");

    // Normalize → π[g].
    detail::sx_normalize_ambient_kernel<<<(m + T - 1) / T, T, 0, stream>>>(
        d_ambient.get(), m, 1.f / h_total);

    // Pass 3: host-side top-ambient-gene mask construction (one D2H of π).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> h_pi(m);
    cudaMemcpy(h_pi.data(), d_ambient.get(), m * sizeof(float), cudaMemcpyDeviceToHost);

    // Sort gene indices descending by π[g], take top top_ambient_frac fraction.
    const int n_top = std::max(1, static_cast<int>(static_cast<float>(m) * cfg.top_ambient_frac));
    std::vector<int> idx(m);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + n_top, idx.end(),
                      [&](int a, int b){ return h_pi[a] > h_pi[b]; });

    std::vector<uint8_t> h_mask(m, 0);
    float h_denom_pi = 0.f;
    for (int k = 0; k < n_top; ++k) {
        h_mask[idx[k]] = 1;
        h_denom_pi += h_pi[idx[k]];
    }

    core::DeviceMemory<uint8_t> d_top_mask(m);
    cudaMemcpyAsync(d_top_mask.get(), h_mask.data(), m * sizeof(uint8_t),
                    cudaMemcpyHostToDevice, stream);

    // Pass 4: per-cell ρ_c.
    core::DeviceMemory<float> d_rho_c(n);
    cudaMemsetAsync(d_rho_c.get(), 0, n * sizeof(float), stream);
    if (n > 0) {
        constexpr int WPB = T / 32;
        detail::sx_rho_kernel<<<(n + WPB - 1) / WPB, T, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_t.get(), d_top_mask.get(), d_ambient.get(),
            d_rho_c.get(), n, h_denom_pi,
            cfg.min_rho, cfg.max_rho, cfg.lower);
    }

    // Pass 5: dense corrected output.
    // cudaMemset → 0 (handles implicit zeros efficiently: max(0, 0 - sub) = 0).
    const size_t dense_sz = static_cast<size_t>(m) * static_cast<size_t>(n);
    core::DeviceMemory<float> d_corrected(dense_sz);
    cudaMemsetAsync(d_corrected.get(), 0, dense_sz * sizeof(float), stream);

    if (nnz > 0) {
        detail::sx_correct_kernel<<<(nnz + T - 1) / T, T, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_ambient.get(), d_rho_c.get(), d_t.get(),
            d_corrected.get(), n, nnz, m);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    SoupxResult res;
    res.corrected   = std::move(d_corrected);
    res.rho_c       = std::move(d_rho_c);
    res.ambient_pi  = std::move(d_ambient);
    res.m           = m;
    res.n           = n;
    return res;
}

}  // namespace qc
}  // namespace singlet_gpu
