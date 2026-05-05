// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: DropletUtils::emptyDrops (Lun et al. 2019)
// qc/empty_drops.h — GPU-native cell/empty-droplet calling for raw 10X data.
// Lun ATL et al. (2019) Genome Biology 20:63.
//
// 6 GPU passes: (1) per-droplet UMI total (one warp/col);
//   (2a) mark empty (t[j] <= lower); (2b) ambient atomic-scatter;
//   (2c) normalize π + log_π; (3) observed LL per candidate (one warp/cand);
//   (4) MC p-values — one block per candidate, 256 threads, niters distributed;
//       categorical via shared-mem cumulative CDF + binary search (O(log m)/draw);
//   (5) BH FDR (host-side sort+scan on candidate array, one D2H+H2D at boundary);
//   (6) is_cell scatter.
// Limits (v0): m_genes <= 32768 (smem CDF); t_j > 100000 → p=1, skipped.
// Rules: 5 (DeviceMemory), 4 (no inner-loop D2H), 18 (Philox, deterministic seed),
//        31 (<= 500 LOC).

#pragma once
#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/device/device_reduce.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace qc {

struct EmptyDropsConfig {
    int      lower          = 100;
    int      niters         = 10000;
    float    fdr_thresh     = 0.001f;
    uint64_t seed           = 0;
    bool     deterministic  = false;  // no-op; MC is inherently stochastic
    int      max_candidates = 50000;
};

struct EmptyDropsResult {
    core::DeviceMemory<float>   pvalue;   // n_droplets; 1.0 for empty/skipped
    core::DeviceMemory<float>   fdr;      // n_droplets; BH q-values
    core::DeviceMemory<uint8_t> is_cell;  // n_droplets; 1 = called as cell
    int n_droplets;
    int n_candidates;
};

namespace detail {

static constexpr int   ED_THREADS      = 256;
static constexpr float ED_LOG_PI_EPS   = 1e-30f;
static constexpr int   ED_MAX_UMI      = 100000;
static constexpr int   ED_MAX_GENES    = 32768;

// Pass 1: one warp per column — sum UMI into t[j].
__global__ __launch_bounds__(256, 2)
void ed_col_sum_kernel(const float* __restrict__ values,
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

// Pass 2a: mark empty droplets.
__global__ __launch_bounds__(256, 4)
void ed_mark_empty_kernel(const int32_t* __restrict__ t,
                          uint8_t* __restrict__ is_empty, int n, int lower)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    is_empty[j] = (t[j] <= lower) ? 1u : 0u;
}

// Pass 2b: atomic-scatter X[i,j] into ambient[i] for empty columns.
__global__ __launch_bounds__(256, 2)
void ed_ambient_scatter_kernel(
    const float* __restrict__ values, const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr, const uint8_t* __restrict__ is_empty,
    float* __restrict__ ambient, int n_cols, int nnz)
{
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= nnz) return;
    int lo = 0, hi = n_cols - 1;
    while (lo < hi) { int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= e) lo = mid + 1; else hi = mid; }
    if (!is_empty[lo]) return;
    atomicAdd(&ambient[row_indices[e]], values[e]);
}

// Pass 2c: normalize ambient → π[g] in-place; fill log_π[g].
__global__ __launch_bounds__(256, 4)
void ed_normalize_ambient_kernel(float* __restrict__ ambient,
                                 float* __restrict__ log_pi,
                                 int m, float total_inv)
{
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= m) return;
    float pi   = ambient[g] * total_inv;
    ambient[g] = pi;
    log_pi[g]  = logf(fmaxf(pi, ED_LOG_PI_EPS));
}

// Pass 3: observed LL — one warp per candidate column.
__global__ __launch_bounds__(256, 2)
void ed_obs_ll_kernel(
    const float* __restrict__ values, const int32_t* __restrict__ row_indices,
    const int32_t* __restrict__ col_ptr, const float* __restrict__ log_pi,
    const int32_t* __restrict__ cand_cols, float* __restrict__ ll_obs, int n_cand)
{
    const int wid  = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (wid >= n_cand) return;
    const int j  = cand_cols[wid];
    float acc = 0.f;
    for (int i = col_ptr[j] + lane; i < col_ptr[j + 1]; i += 32)
        acc += values[i] * log_pi[row_indices[i]];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) ll_obs[wid] = acc;
}

// Pass 4: MC p-values — one block per candidate, THREADS threads.
// Shared memory: m floats for cumulative CDF of π.
// Categorical sampling: cuRAND uniform u → binary search on smem_cdf → gene.
// Each thread handles ceil(niters/THREADS) synthetics; results summed via atomicAdd.
__global__ __launch_bounds__(256, 1)
void ed_mc_pvalue_kernel(
    const float* __restrict__ pi, const float* __restrict__ log_pi,
    const int32_t* __restrict__ cand_t, const float* __restrict__ ll_obs,
    int32_t* __restrict__ hit_count,
    int n_cand, int m, int niters, uint64_t seed)
{
    extern __shared__ float smem_cdf[];
    const int cand_idx = blockIdx.x;
    if (cand_idx >= n_cand) return;
    const int tid  = threadIdx.x;
    const int ntid = blockDim.x;

    // Build cumulative CDF collaboratively.
    for (int g = tid; g < m; g += ntid) smem_cdf[g] = pi[g];
    __syncthreads();
    if (tid == 0) {
        float cum = 0.f;
        for (int g = 0; g < m; ++g) { cum += smem_cdf[g]; smem_cdf[g] = cum; }
        smem_cdf[m - 1] = 1.f;
    }
    __syncthreads();

    curandStatePhilox4_32_10_t rng;
    curand_init(seed, static_cast<uint64_t>(cand_idx) * 263ULL + tid, 0ULL, &rng);

    const int   t_j     = cand_t[cand_idx];
    const float obs_ll  = ll_obs[cand_idx];
    const int   iters   = (niters + ntid - 1) / ntid;
    int local_hits = 0;

    for (int it = 0; it < iters; ++it) {
        float synth_ll = 0.f;
        for (int draw = 0; draw < t_j; ++draw) {
            float u = curand_uniform(&rng);
            if (u >= 1.f) u = 0.9999999f;
            int lo_g = 0, hi_g = m - 1;
            while (lo_g < hi_g) {
                int mid_g = (lo_g + hi_g) >> 1;
                if (smem_cdf[mid_g] < u) lo_g = mid_g + 1; else hi_g = mid_g;
            }
            synth_ll += log_pi[lo_g];
        }
        if (synth_ll <= obs_ll) ++local_hits;
    }
    atomicAdd(&hit_count[cand_idx], local_hits);
}

// Scatter p-values from hit counts.
__global__ __launch_bounds__(256, 4)
void ed_pvalue_kernel(const int32_t* __restrict__ hit_count,
                      const int32_t* __restrict__ cand_cols,
                      float* __restrict__ pvalue, int n_cand, int niters)
{
    const int ci = blockIdx.x * blockDim.x + threadIdx.x;
    if (ci >= n_cand) return;
    pvalue[cand_cols[ci]] = static_cast<float>(1 + hit_count[ci]) /
                            static_cast<float>(niters + 1);
}

// Scatter FDR + is_cell from candidate-indexed arrays.
__global__ __launch_bounds__(256, 4)
void ed_scatter_fdr_kernel(const float* __restrict__ fdr_cand,
                           const int* __restrict__ sort_order,
                           const int32_t* __restrict__ cand_cols,
                           float* __restrict__ fdr, uint8_t* __restrict__ is_cell,
                           int n_cand, float fdr_thresh)
{
    const int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= n_cand) return;
    const int ci = sort_order[si];
    const int j  = cand_cols[ci];
    fdr[j]     = fdr_cand[ci];
    is_cell[j] = (fdr_cand[ci] < fdr_thresh) ? 1u : 0u;
}

// Initialize per-droplet outputs.
__global__ __launch_bounds__(256, 4)
void ed_init_outputs_kernel(float* __restrict__ pvalue, float* __restrict__ fdr,
                            uint8_t* __restrict__ is_cell, int n)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    pvalue[j] = 1.f; fdr[j] = 1.f; is_cell[j] = 0u;
}

} // namespace detail

// ---------------------------------------------------------------------------
// empty_drops — public entry point
// ---------------------------------------------------------------------------
inline EmptyDropsResult
empty_drops(const io::PzDeviceMatrix& X, const EmptyDropsConfig& cfg = {},
            cudaStream_t stream = nullptr)
{
    if (stream == nullptr) stream = singlet_gpu::core::default_context().stream();

    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    if (m <= 0 || n <= 0)
        throw std::runtime_error("empty_drops: invalid matrix (m=" +
            std::to_string(m) + " n=" + std::to_string(n) + ")");
    if (m > detail::ED_MAX_GENES)
        throw std::runtime_error(
            "empty_drops: m_genes=" + std::to_string(m) +
            " > smem limit " + std::to_string(detail::ED_MAX_GENES) +
            ". Subset to HVGs (v0 limit).");

    constexpr int T = detail::ED_THREADS;

    // Initialize outputs.
    core::DeviceMemory<float>   d_pvalue(n);
    core::DeviceMemory<float>   d_fdr(n);
    core::DeviceMemory<uint8_t> d_is_cell(n);
    detail::ed_init_outputs_kernel<<<(n + T - 1) / T, T, 0, stream>>>(
        d_pvalue.get(), d_fdr.get(), d_is_cell.get(), n);

    // Pass 1: UMI totals.
    core::DeviceMemory<int32_t> d_t(n);
    cudaMemsetAsync(d_t.get(), 0, n * sizeof(int32_t), stream);
    if (nnz > 0) {
        constexpr int WPB = T / 32;
        detail::ed_col_sum_kernel<<<(n + WPB - 1) / WPB, T, 0, stream>>>(
            X.mat.values.get(), X.mat.col_ptr.get(), d_t.get(), n);
    }

    // Pass 2a: mark empty.
    core::DeviceMemory<uint8_t> d_is_empty(n);
    detail::ed_mark_empty_kernel<<<(n + T - 1) / T, T, 0, stream>>>(
        d_t.get(), d_is_empty.get(), n, cfg.lower);

    // Pass 2b: ambient scatter.
    core::DeviceMemory<float> d_ambient(m);
    cudaMemsetAsync(d_ambient.get(), 0, m * sizeof(float), stream);
    if (nnz > 0) {
        detail::ed_ambient_scatter_kernel<<<(nnz + T - 1) / T, T, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_is_empty.get(), d_ambient.get(), n, nnz);
    }

    // Reduce total ambient (one scalar D2H at function boundary).
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
        throw std::runtime_error("empty_drops: no empty droplets (lower=" +
            std::to_string(cfg.lower) + ").");

    // Pass 2c: normalize.
    core::DeviceMemory<float> d_log_pi(m);
    detail::ed_normalize_ambient_kernel<<<(m + T - 1) / T, T, 0, stream>>>(
        d_ambient.get(), d_log_pi.get(), m, 1.f / h_total);

    // Build candidate list on host (one D2H at boundary).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> h_t(n);
    cudaMemcpy(h_t.data(), d_t.get(), n * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::vector<int32_t> h_cand_cols, h_cand_t;
    for (int j = 0; j < n; ++j) {
        if (h_t[j] > cfg.lower && h_t[j] <= detail::ED_MAX_UMI) {
            h_cand_cols.push_back(j);
            h_cand_t.push_back(h_t[j]);
        }
    }
    const int n_cand = static_cast<int>(h_cand_cols.size());
    const int nc     = std::min(n_cand, cfg.max_candidates);

    EmptyDropsResult result;
    result.n_droplets   = n;
    result.n_candidates = n_cand;
    result.pvalue       = std::move(d_pvalue);
    result.fdr          = std::move(d_fdr);
    result.is_cell      = std::move(d_is_cell);

    if (nc == 0) { CUDA_CHECK(cudaStreamSynchronize(stream)); return result; }

    core::DeviceMemory<int32_t> d_cand_cols(nc), d_cand_t(nc);
    cudaMemcpyAsync(d_cand_cols.get(), h_cand_cols.data(),
                    nc * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_cand_t.get(), h_cand_t.data(),
                    nc * sizeof(int32_t), cudaMemcpyHostToDevice, stream);

    // Pass 3: observed LL.
    core::DeviceMemory<float> d_ll_obs(nc);
    if (nnz > 0) {
        constexpr int WPB = T / 32;
        detail::ed_obs_ll_kernel<<<(nc + WPB - 1) / WPB, T, 0, stream>>>(
            X.mat.values.get(), X.mat.row_indices.get(), X.mat.col_ptr.get(),
            d_log_pi.get(), d_cand_cols.get(), d_ll_obs.get(), nc);
    } else {
        cudaMemsetAsync(d_ll_obs.get(), 0, nc * sizeof(float), stream);
    }

    // Pass 4: MC p-values.
    core::DeviceMemory<int32_t> d_hit(nc);
    cudaMemsetAsync(d_hit.get(), 0, nc * sizeof(int32_t), stream);
    const size_t smem = static_cast<size_t>(m) * sizeof(float);
    detail::ed_mc_pvalue_kernel<<<nc, T, smem, stream>>>(
        d_ambient.get(), d_log_pi.get(), d_cand_t.get(), d_ll_obs.get(),
        d_hit.get(), nc, m, cfg.niters, cfg.seed);

    detail::ed_pvalue_kernel<<<(nc + T - 1) / T, T, 0, stream>>>(
        d_hit.get(), d_cand_cols.get(), result.pvalue.get(), nc, cfg.niters);

    // Pass 5: BH FDR (host-side; one D2H + one H2D).
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<int32_t> h_hit(nc);
    cudaMemcpy(h_hit.data(), d_hit.get(), nc * sizeof(int32_t), cudaMemcpyDeviceToHost);

    std::vector<float> h_pval(nc);
    for (int ci = 0; ci < nc; ++ci)
        h_pval[ci] = static_cast<float>(1 + h_hit[ci]) /
                     static_cast<float>(cfg.niters + 1);

    std::vector<int> h_ord(nc);
    for (int i = 0; i < nc; ++i) h_ord[i] = i;
    std::sort(h_ord.begin(), h_ord.end(),
              [&](int a, int b){ return h_pval[a] < h_pval[b]; });

    std::vector<float>   h_fdr(nc);
    std::vector<uint8_t> h_iscell(nc);
    float rmin = 1.f;
    for (int rank = nc - 1; rank >= 0; --rank) {
        const int ci = h_ord[rank];
        float q = h_pval[ci] * static_cast<float>(nc) /
                  static_cast<float>(rank + 1);
        q = (q < 1.f) ? q : 1.f;
        rmin = (q < rmin) ? q : rmin;
        h_fdr[ci]    = rmin;
        h_iscell[ci] = (rmin < cfg.fdr_thresh) ? 1u : 0u;
    }

    // Scatter FDR+is_cell back (identity sort_order, one kernel).
    {
        core::DeviceMemory<float>   d_fdr_c(nc);
        core::DeviceMemory<uint8_t> d_isc(nc);
        std::vector<int> h_id(nc); for (int i = 0; i < nc; ++i) h_id[i] = i;
        core::DeviceMemory<int> d_id(nc);
        cudaMemcpyAsync(d_fdr_c.get(), h_fdr.data(),
                        nc * sizeof(float),   cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_isc.get(),   h_iscell.data(),
                        nc * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_id.get(),    h_id.data(),
                        nc * sizeof(int),     cudaMemcpyHostToDevice, stream);
        detail::ed_scatter_fdr_kernel<<<(nc + T - 1) / T, T, 0, stream>>>(
            d_fdr_c.get(), d_id.get(), d_cand_cols.get(),
            result.fdr.get(), result.is_cell.get(), nc, cfg.fdr_thresh);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

} // namespace qc
} // namespace singlet_gpu
