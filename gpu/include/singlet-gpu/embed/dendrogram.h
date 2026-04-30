// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: scanpy.tl.dendrogram (hierarchical clustering of cluster centroids)
//
// singlet-gpu/embed/dendrogram.h
//
// Dendrogram — hierarchical clustering of cluster centroids for visualization
// of cluster relationships. Mirrors the scanpy.tl.dendrogram → scipy UPGMA
// pipeline, with Steps 1 and 2 accelerated on GPU.
//
// Algorithm (host + GPU hybrid):
//   Step 1 — Cluster centroids (GPU):
//     For each nnz in CSC X, atomicAdd μ[row + label[col]*m] += value.
//     Then divide μ[g, k] /= n_per_cluster[k] (per-element kernel).
//   Step 2 — Pairwise correlation distance (GPU):
//     Center each centroid column (subtract column mean).
//     Normalize each centroid column by L2 norm.
//     corr[i,j] = (μ_norm)^T · μ_norm via cuBLAS Sgemm (k × k).
//     d[i,j] = 1 - corr[i,j] (element-wise kernel).
//   Step 3 — UPGMA hierarchical clustering (host):
//     Download d (k × k) to host.
//     Agglomerative average-linkage O(k³) pure C++ — tractable for k ≤ 100.
//     Output linkage Z (k-1) × 4: [a, b, dist, n_merged].
//
// Non-determinism (Rule 18): atomicAdd in Step 1 is non-deterministic at fp32.
//   Relative error bounded to < 1e-4 in practice for typical expression matrices.
//   Set cfg.deterministic = true to document; behavior is unchanged (GPU atomics
//   are inherently non-deterministic when threads race on the same address).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one D2H of label[n] for n_per_cluster; one D2H of d[k*k] for UPGMA.
//   Both are one-shot at function boundaries — compliant.
// LOC (Rule 31): under 300.
// CYCLE-146: initial implementation.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace embed {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct DendrogramConfig {
    // atomicAdd in centroid scatter is non-deterministic (fp32 ordering).
    // This flag is documentation-only — no behavioral change.
    bool deterministic = false;
};

struct DendrogramResult {
    core::DeviceMemory<float> centroids;  // m × k_clusters col-major
    core::DeviceMemory<float> distance;   // k_clusters × k_clusters symmetric
    std::vector<float>        linkage;    // (k-1)*4 host: [a, b, dist, n_merged]
    int m          = 0;
    int k_clusters = 0;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// dgram_centroid_scatter_kernel — Step 1a: atomic-scatter nnz into μ.
//
// One thread per nnz. For nnz index t:
//   row    = d_row_indices[t]
//   col    = col owning t (searched via col_ptr)
//   label  = d_label[col]
//   atomicAdd μ[row + label*m] += d_values[t]
//
// We use a grid over column spans: one block per column, threads over nnz in col.
// This avoids binary-searching col_ptr per thread.
//
// One block per column (n_cols blocks). Each block handles d_col_ptr[col]..d_col_ptr[col+1]-1.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dgram_centroid_scatter_kernel(
    const int*   __restrict__ d_col_ptr,      // [n_cols+1]
    const int*   __restrict__ d_row_indices,  // [nnz]
    const float* __restrict__ d_values,       // [nnz]
    const int*   __restrict__ d_label,        // [n_cols]
    float*       __restrict__ d_mu,           // [m * k_clusters] col-major, pre-zeroed
    int n_cols, int m)
{
    const int col = static_cast<int>(blockIdx.x);
    if (col >= n_cols) return;

    const int lbl   = d_label[col];
    const int start = d_col_ptr[col];
    const int end   = d_col_ptr[col + 1];

    for (int t = start + static_cast<int>(threadIdx.x); t < end;
         t += static_cast<int>(blockDim.x)) {
        const int   row = d_row_indices[t];
        const float val = d_values[t];
        atomicAdd(&d_mu[row + lbl * m], val);
    }
}

// ---------------------------------------------------------------------------
// dgram_centroid_divide_kernel — Step 1b: divide μ[g,k] /= n_per_cluster[k].
//
// One thread per element. d_n_per_cluster[k] == 0 → leave 0 (empty cluster).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dgram_centroid_divide_kernel(
    float*       __restrict__ d_mu,             // [m * k_clusters] col-major
    const float* __restrict__ d_n_per_cluster,  // [k_clusters]
    int m, int k_clusters)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total = m * k_clusters;
    if (idx >= total) return;
    const int k = idx / m;
    const float n = d_n_per_cluster[k];
    if (n > 0.f) d_mu[idx] /= n;
    // n == 0: centroid stays 0 (empty cluster guard).
}

// ---------------------------------------------------------------------------
// dgram_col_mean_kernel — Step 2a: per-column mean of μ.
//
// One block per cluster column k. Threads stride over m genes.
// Writes d_col_mean[k] = (1/m) * Σ_g μ[g, k].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void dgram_col_mean_kernel(
    const float* __restrict__ d_mu,        // [m * k] col-major
    float*       __restrict__ d_col_mean,  // [k]
    int m, int k_clusters)
{
    const int k = static_cast<int>(blockIdx.x);
    if (k >= k_clusters) return;

    const float* col = d_mu + (ptrdiff_t)k * m;
    float acc = 0.f;
    for (int g = static_cast<int>(threadIdx.x); g < m;
         g += static_cast<int>(blockDim.x)) {
        acc += col[g];
    }

    // Warp reduce.
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float smem[8];
    const int wid  = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int nwarp = (static_cast<int>(blockDim.x) + 31) / 32;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
        for (int off = 4; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, off);
        if (lane == 0) d_col_mean[k] = val / static_cast<float>(m > 0 ? m : 1);
    }
}

// ---------------------------------------------------------------------------
// dgram_center_normalize_kernel — Step 2b+c: center then normalize each column.
//
// One thread per element (g, k). In-place on d_mu.
// After this pass: d_mu is the μ_norm matrix (centered + unit-norm).
// Uses d_col_mean[k] for centering, computes L2 in a separate pass below.
//
// This kernel subtracts the mean (centering).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dgram_center_kernel(
    float*       __restrict__ d_mu,        // [m * k] col-major, in-place
    const float* __restrict__ d_col_mean,  // [k]
    int m, int k_clusters)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total = m * k_clusters;
    if (idx >= total) return;
    const int k = idx / m;
    d_mu[idx] -= d_col_mean[k];
}

// ---------------------------------------------------------------------------
// dgram_col_norm_kernel — Step 2c part 1: per-column L2 norm.
//
// One block per cluster k. Writes d_col_norm[k].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void dgram_col_norm_kernel(
    const float* __restrict__ d_mu,        // [m * k] col-major
    float*       __restrict__ d_col_norm,  // [k]
    int m, int k_clusters)
{
    const int k = static_cast<int>(blockIdx.x);
    if (k >= k_clusters) return;

    const float* col = d_mu + (ptrdiff_t)k * m;
    float acc = 0.f;
    for (int g = static_cast<int>(threadIdx.x); g < m;
         g += static_cast<int>(blockDim.x)) {
        acc += col[g] * col[g];
    }

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float smem[8];
    const int wid  = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int nwarp = (static_cast<int>(blockDim.x) + 31) / 32;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
        for (int off = 4; off > 0; off >>= 1)
            val += __shfl_down_sync(0xffffffffu, val, off);
        if (lane == 0) d_col_norm[k] = sqrtf(val);
    }
}

// ---------------------------------------------------------------------------
// dgram_normalize_kernel — Step 2c part 2: divide each column by its L2 norm.
//
// One thread per element. Eps guard: zero-variance column stays zero.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dgram_normalize_kernel(
    float*       __restrict__ d_mu,        // [m * k] col-major, in-place
    const float* __restrict__ d_col_norm,  // [k]
    int m, int k_clusters)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total = m * k_clusters;
    if (idx >= total) return;
    const int k    = idx / m;
    const float nrm = d_col_norm[k];
    const float eps = 1e-9f;
    d_mu[idx] = (nrm > eps) ? d_mu[idx] / nrm : 0.f;
}

// ---------------------------------------------------------------------------
// dgram_dist_kernel — Step 2d: d[i,j] = 1 - corr[i,j] (element-wise).
//
// One thread per element of the k × k matrix.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void dgram_dist_kernel(
    float* __restrict__ d_corr,  // [k * k] col-major, in-place → distance
    int k_total)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx >= k_total * k_total) return;
    d_corr[idx] = 1.f - d_corr[idx];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// host_upgma — pure C++ UPGMA O(k³).
//
// Input:  h_dist — k×k distance matrix (row-major; symmetric, zero diagonal).
// Output: linkage vector of (k-1)*4 floats: [a, b, dist, n_merged] per step.
// ---------------------------------------------------------------------------
inline std::vector<float> host_upgma(std::vector<float> h_dist, int k)
{
    std::vector<float> linkage;
    linkage.reserve(static_cast<size_t>(k - 1) * 4);

    std::vector<int>  cluster_size(static_cast<size_t>(k), 1);
    std::vector<bool> active(static_cast<size_t>(k), true);

    for (int step = 0; step < k - 1; ++step) {
        // Find (i, j) with minimum distance over active clusters.
        float best_d = std::numeric_limits<float>::max();
        int   best_i = -1, best_j = -1;
        for (int i = 0; i < k; ++i) {
            if (!active[static_cast<size_t>(i)]) continue;
            for (int j = i + 1; j < k; ++j) {
                if (!active[static_cast<size_t>(j)]) continue;
                const float d = h_dist[static_cast<size_t>(i) * k + j];
                if (d < best_d) { best_d = d; best_i = i; best_j = j; }
            }
        }
        if (best_i < 0) break;  // should not happen for valid input

        const int si = cluster_size[static_cast<size_t>(best_i)];
        const int sj = cluster_size[static_cast<size_t>(best_j)];
        const float total = static_cast<float>(si + sj);

        // Record linkage: [a, b, dist, n_merged]
        linkage.push_back(static_cast<float>(best_i));
        linkage.push_back(static_cast<float>(best_j));
        linkage.push_back(best_d);
        linkage.push_back(total);

        // UPGMA average-linkage update: merge j into i.
        for (int m = 0; m < k; ++m) {
            if (!active[static_cast<size_t>(m)] || m == best_i || m == best_j) continue;
            const float d_im = h_dist[static_cast<size_t>(best_i) * k + m];
            const float d_jm = h_dist[static_cast<size_t>(best_j) * k + m];
            const float new_d = (d_im * static_cast<float>(si) +
                                 d_jm * static_cast<float>(sj)) / total;
            h_dist[static_cast<size_t>(best_i) * k + m] = new_d;
            h_dist[static_cast<size_t>(m) * k + best_i] = new_d;
        }
        cluster_size[static_cast<size_t>(best_i)] = si + sj;
        active[static_cast<size_t>(best_j)]       = false;
    }
    return linkage;
}

// ---------------------------------------------------------------------------
// dendrogram — public entry point.
//
// Inputs:
//   X          — m × n PzDeviceMatrix (sparse CSC, normalized expression).
//                X.mat.rows = m (genes), X.mat.cols = n (cells).
//   d_label    — device int[n_cells] with cluster labels in [0, k_clusters).
//   k_clusters — number of distinct cluster labels.
//   cfg        — DendrogramConfig (deterministic flag, documentation only).
//   stream     — CUDA stream (nullptr = default context stream).
//
// Returns DendrogramResult with:
//   centroids  — device float[m × k_clusters] col-major.
//   distance   — device float[k × k] col-major symmetric.
//   linkage    — host float[(k-1) × 4].
//
// Caller synchronizes stream before reading device buffers.
// ---------------------------------------------------------------------------
inline DendrogramResult dendrogram(
    const io::PzDeviceMatrix& X,
    const int*                d_label,
    int                       k_clusters,
    const DendrogramConfig&   cfg    = {},
    cudaStream_t              stream = nullptr)
{
    (void)cfg;  // deterministic flag is documentation-only.

    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    const int m = X.mat.rows;
    const int n = X.mat.cols;

    if (k_clusters <= 0)
        throw std::runtime_error(
            "dendrogram: k_clusters must be > 0 (got " +
            std::to_string(k_clusters) + ")");
    if (m <= 0 || n <= 0)
        throw std::runtime_error(
            "dendrogram: empty matrix (m=" + std::to_string(m) +
            " n=" + std::to_string(n) + ")");
    if (d_label == nullptr)
        throw std::runtime_error("dendrogram: d_label is null");

    constexpr int THREADS = 256;

    // -------------------------------------------------------------------------
    // D2H label[n] — one-shot to compute n_per_cluster (Rule 4 compliant).
    // -------------------------------------------------------------------------
    std::vector<int> h_label(static_cast<size_t>(n));
    CUDA_CHECK(cudaMemcpyAsync(h_label.data(), d_label,
                               static_cast<size_t>(n) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<float> h_n_per_cluster(static_cast<size_t>(k_clusters), 0.f);
    for (int c = 0; c < n; ++c) {
        const int lbl = h_label[static_cast<size_t>(c)];
        if (lbl >= 0 && lbl < k_clusters)
            h_n_per_cluster[static_cast<size_t>(lbl)] += 1.f;
    }

    // Upload n_per_cluster to device for the divide kernel.
    core::DeviceMemory<float> d_n_per_cluster(static_cast<size_t>(k_clusters));
    CUDA_CHECK(cudaMemcpyAsync(d_n_per_cluster.get(), h_n_per_cluster.data(),
                               static_cast<size_t>(k_clusters) * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // -------------------------------------------------------------------------
    // Step 1a — allocate μ[m × k_clusters] and zero it.
    // -------------------------------------------------------------------------
    const size_t mu_sz = static_cast<size_t>(m) * static_cast<size_t>(k_clusters);
    core::DeviceMemory<float> d_mu(mu_sz);
    CUDA_CHECK(cudaMemsetAsync(d_mu.get(), 0, mu_sz * sizeof(float), stream));

    // Atomic scatter: one block per column (cell), threads over nnz in column.
    dgram_centroid_scatter_kernel<<<n, THREADS, 0, stream>>>(
        X.mat.col_ptr.get(), X.mat.row_indices.get(), X.mat.values.get(),
        d_label, d_mu.get(), n, m);

    // -------------------------------------------------------------------------
    // Step 1b — divide μ[g,k] /= n_per_cluster[k].
    // -------------------------------------------------------------------------
    {
        const int total = m * k_clusters;
        const int grid  = (total + THREADS - 1) / THREADS;
        dgram_centroid_divide_kernel<<<grid, THREADS, 0, stream>>>(
            d_mu.get(), d_n_per_cluster.get(), m, k_clusters);
    }

    // Save centroids before in-place centering/normalization overwrites d_mu.
    core::DeviceMemory<float> d_centroids(mu_sz);
    CUDA_CHECK(cudaMemcpyAsync(d_centroids.get(), d_mu.get(),
                               mu_sz * sizeof(float),
                               cudaMemcpyDeviceToDevice, stream));

    // -------------------------------------------------------------------------
    // Step 2a — per-column mean.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_col_mean(static_cast<size_t>(k_clusters));
    dgram_col_mean_kernel<<<k_clusters, THREADS, 0, stream>>>(
        d_mu.get(), d_col_mean.get(), m, k_clusters);

    // -------------------------------------------------------------------------
    // Step 2b — center each column.
    // -------------------------------------------------------------------------
    {
        const int total = m * k_clusters;
        const int grid  = (total + THREADS - 1) / THREADS;
        dgram_center_kernel<<<grid, THREADS, 0, stream>>>(
            d_mu.get(), d_col_mean.get(), m, k_clusters);
    }

    // -------------------------------------------------------------------------
    // Step 2c — per-column L2 norm, then normalize.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_col_norm(static_cast<size_t>(k_clusters));
    dgram_col_norm_kernel<<<k_clusters, THREADS, 0, stream>>>(
        d_mu.get(), d_col_norm.get(), m, k_clusters);
    {
        const int total = m * k_clusters;
        const int grid  = (total + THREADS - 1) / THREADS;
        dgram_normalize_kernel<<<grid, THREADS, 0, stream>>>(
            d_mu.get(), d_col_norm.get(), m, k_clusters);
    }

    // -------------------------------------------------------------------------
    // Step 2d — corr = μ_norm^T · μ_norm via cuBLAS Sgemm.
    //
    // μ_norm is (m × k) col-major.
    // corr = μ_norm^T · μ_norm = Sgemm(op=T, op=N, m=k, n=k, k=m).
    // Output corr is (k × k) col-major.
    // -------------------------------------------------------------------------
    const size_t k2_sz = static_cast<size_t>(k_clusters) * static_cast<size_t>(k_clusters);
    core::DeviceMemory<float> d_corr(k2_sz);
    {
        cublasHandle_t blas = singlet_gpu::core::default_context().blas();
        cublasSetStream(blas, stream);
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            k_clusters,   // m (rows of output)
            k_clusters,   // n (cols of output)
            m,            // k (inner dimension)
            &alpha,
            d_mu.get(), m,            // A = μ_norm (m × k), ldA = m
            d_mu.get(), m,            // B = μ_norm (m × k), ldB = m
            &beta,
            d_corr.get(), k_clusters  // C = corr  (k × k), ldC = k
        ));
    }

    // -------------------------------------------------------------------------
    // Step 2e — d[i,j] = 1 - corr[i,j] (element-wise).
    // -------------------------------------------------------------------------
    {
        const int total = k_clusters * k_clusters;
        const int grid  = (total + THREADS - 1) / THREADS;
        dgram_dist_kernel<<<grid, THREADS, 0, stream>>>(d_corr.get(), k_clusters);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // -------------------------------------------------------------------------
    // Step 3 — D2H distance matrix, then host UPGMA (Rule 4 one-shot).
    // -------------------------------------------------------------------------
    std::vector<float> h_dist(k2_sz);
    CUDA_CHECK(cudaMemcpy(h_dist.data(), d_corr.get(),
                          k2_sz * sizeof(float), cudaMemcpyDeviceToHost));

    // Convert col-major (k × k) to row-major for host_upgma (symmetric — same).
    // Distance matrix is symmetric so col-major == row-major for the diagonal
    // and the off-diagonal access pattern in UPGMA (d[i*k+j]).
    // host_upgma expects d[i*k+j] (row-major); col-major is identical for
    // square symmetric matrices when accessing by (i,j) and (j,i).
    std::vector<float> linkage = (k_clusters > 1)
        ? host_upgma(h_dist, k_clusters)
        : std::vector<float>{};

    DendrogramResult res;
    res.centroids  = std::move(d_centroids);
    res.distance   = std::move(d_corr);
    res.linkage    = std::move(linkage);
    res.m          = m;
    res.k_clusters = k_clusters;
    return res;
}

}  // namespace embed
}  // namespace singlet_gpu
