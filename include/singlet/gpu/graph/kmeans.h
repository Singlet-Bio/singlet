// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: Lloyd (1982) "Least squares quantization in PCM." IEEE TIT 28:129-137.
//
// graph/kmeans.h — GPU k-means clustering (Lloyd's algorithm)
//
// Algorithm:
//   Init  — Forgy: random k cell columns copied to centroid matrix (host mt19937).
//   Loop  — Per iteration:
//     (a) ||C[:,k]||² per column (one block per k, warp-shuffle reduction).
//     (b) Sgemm:   D[n×k] = X^T · C  (cuBLAS Sgemm).
//     (c) Dist+argmin: dist[c,k] = ||X||²[c] + ||C||²[k] - 2·D[c,k];
//                       label[c] = argmin_k dist[c,k] (one block per cell).
//     (d) Change count: compare label_old vs label_new; atomicAdd scalar.
//     (e) Centroid update: atomic-scatter X[:,c] into C_new[:,label[c]]; divide.
//   Inertia (final): sum_c ||X[:,c] - C[:,label[c]]||².
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>.
// D2H (Rule 4): one scalar (changes) per Lloyd iter — APPROVED exception (convergence
//   check; identical pattern to NMF).
// Determinism (Rule 18): atomicAdd in centroid update is non-deterministic;
//   cfg.deterministic is documentation-only in v0.
// LOC (Rule 31): < 350.
// CYCLE-149: initial implementation.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace graph {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct KmeansConfig {
    int      max_iter    = 100;
    int      tol_changes = 0;    // converge when fewer than this many cells changed label
    uint64_t seed        = 0;
    bool     deterministic = false;  // documentation-only in v0 (atomicAdd is non-det.)
};

struct KmeansResult {
    core::DeviceMemory<int>   labels;      // n_cells
    core::DeviceMemory<float> centroids;   // d_pcs × k_clusters col-major
    int   n_cells    = 0;
    int   d_pcs      = 0;
    int   k_clusters = 0;
    int   iterations = 0;   // Lloyd iterations executed
    float inertia    = 0.f; // sum of squared distances at convergence
    bool  converged  = false;
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace {

constexpr int KM_THREADS = 256;

// ---------------------------------------------------------------------------
// km_col_norm_kernel — squared L2 norm of each column of M (d × n col-major).
// One block per column; warp-shuffle reduction into smem; thread 0 writes.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void km_col_norm_kernel(
    const float* __restrict__ M,     // [d × n] col-major
    float*       __restrict__ norms, // [n]
    int d, int n)
{
    const int col = static_cast<int>(blockIdx.x);
    if (col >= n) return;
    const float* c = M + static_cast<ptrdiff_t>(col) * d;
    float acc = 0.f;
    for (int g = static_cast<int>(threadIdx.x); g < d; g += static_cast<int>(blockDim.x))
        acc += c[g] * c[g];
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    __shared__ float smem[8];
    const int wid  = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int nwarp = (static_cast<int>(blockDim.x) + 31) / 32;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nwarp) ? smem[lane] : 0.f;
        for (int off = 4; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) norms[col] = v;
    }
}

// ---------------------------------------------------------------------------
// km_assign_kernel — distance matrix + per-cell argmin.
//
// dist[c,k] = x_norm[c] + c_norm[k] - 2 * D[c,k]   (D = X^T·C, n×k col-major).
// label_out[c] = argmin_k dist[c,k].
// One block per cell; threads stride over k_clusters.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void km_assign_kernel(
    const float* __restrict__ D,        // [n_cells × k_clusters] col-major (Sgemm output)
    const float* __restrict__ x_norm,   // [n_cells]
    const float* __restrict__ c_norm,   // [k_clusters]
    int*         __restrict__ label_out,// [n_cells]
    int n_cells, int k_clusters)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    const int tid  = static_cast<int>(threadIdx.x);
    const int ntid = static_cast<int>(blockDim.x);
    const float xn  = x_norm[cell];

    float best_d   = 3.402823e+38f;  // FLT_MAX without <cfloat>
    int   best_k   = 0;

    // D is n × k col-major: element (cell, k) = D[cell + k*n_cells]
    for (int k = tid; k < k_clusters; k += ntid) {
        float d = xn + c_norm[k] - 2.f * D[cell + (ptrdiff_t)k * n_cells];
        if (d < best_d) { best_d = d; best_k = k; }
    }

    // Warp-level min over (distance, index) pairs.
    for (int off = 16; off > 0; off >>= 1) {
        float vd = __shfl_down_sync(0xffffffffu, best_d, off);
        int   vi = __shfl_down_sync(0xffffffffu, best_k, off);
        if (vd < best_d) { best_d = vd; best_k = vi; }
    }

    __shared__ float smem_d[8];
    __shared__ int   smem_k[8];
    const int wid  = tid / 32;
    const int lane = tid & 31;
    const int nwarp = (ntid + 31) / 32;
    if (lane == 0) { smem_d[wid] = best_d; smem_k[wid] = best_k; }
    __syncthreads();

    if (wid == 0) {
        float v = (lane < nwarp) ? smem_d[lane] : 3.402823e+38f;
        int   w = (lane < nwarp) ? smem_k[lane] : 0;
        for (int off = 4; off > 0; off >>= 1) {
            float vd = __shfl_down_sync(0xffffffffu, v, off);
            int   vi = __shfl_down_sync(0xffffffffu, w, off);
            if (vd < v) { v = vd; w = vi; }
        }
        if (lane == 0) label_out[cell] = w;
    }
}

// ---------------------------------------------------------------------------
// km_count_changes_kernel — count cells whose label changed.
// Atomic increment into d_changes[0].
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void km_count_changes_kernel(
    const int* __restrict__ label_old,
    const int* __restrict__ label_new,
    int*       __restrict__ d_changes,
    int n_cells)
{
    const int i = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (i >= n_cells) return;
    if (label_old[i] != label_new[i])
        atomicAdd(d_changes, 1);
}

// ---------------------------------------------------------------------------
// km_scatter_kernel — atomic-scatter X[:,c] into C_new[:,label[c]].
// One thread per cell c; strides over d_pcs dimensions.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void km_scatter_kernel(
    const float* __restrict__ d_X,       // [d_pcs × n_cells] col-major
    const int*   __restrict__ d_label,   // [n_cells]
    float*       __restrict__ d_C_new,   // [d_pcs × k_clusters] col-major, pre-zeroed
    int*         __restrict__ d_count,   // [k_clusters], pre-zeroed
    int d_pcs, int n_cells)
{
    const int cell = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                   + static_cast<int>(threadIdx.x);
    if (cell >= n_cells) return;

    const int lbl = d_label[cell];
    const float* Xc     = d_X    + static_cast<ptrdiff_t>(cell) * d_pcs;
    float*       C_col  = d_C_new + static_cast<ptrdiff_t>(lbl)  * d_pcs;

    for (int g = 0; g < d_pcs; ++g)
        atomicAdd(&C_col[g], Xc[g]);
    atomicAdd(&d_count[lbl], 1);
}

// ---------------------------------------------------------------------------
// km_centroid_divide_kernel — C[:,k] /= count[k]; guards empty clusters (count=0).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void km_centroid_divide_kernel(
    float*     __restrict__ d_C,      // [d_pcs × k_clusters] col-major
    const int* __restrict__ d_count,  // [k_clusters]
    int d_pcs, int k_clusters)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total = d_pcs * k_clusters;
    if (idx >= total) return;
    const int k = idx / d_pcs;
    const int cnt = d_count[k];
    if (cnt > 0) d_C[idx] /= static_cast<float>(cnt);
}

// ---------------------------------------------------------------------------
// km_inertia_kernel — accumulate sum_c ||X[:,c] - C[:,label[c]]||² into d_inertia.
// One block per cell; warp-shuffle reduction; thread 0 atomicAdds.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void km_inertia_kernel(
    const float* __restrict__ d_X,       // [d_pcs × n_cells] col-major
    const float* __restrict__ d_C,       // [d_pcs × k_clusters] col-major
    const int*   __restrict__ d_label,   // [n_cells]
    float*       __restrict__ d_inertia, // [1], pre-zeroed
    int d_pcs, int n_cells)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    const int lbl = d_label[cell];
    const float* Xc = d_X + static_cast<ptrdiff_t>(cell) * d_pcs;
    const float* Ck = d_C + static_cast<ptrdiff_t>(lbl)  * d_pcs;

    float acc = 0.f;
    for (int g = static_cast<int>(threadIdx.x); g < d_pcs; g += static_cast<int>(blockDim.x)) {
        float diff = Xc[g] - Ck[g];
        acc += diff * diff;
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
        float v = (lane < nwarp) ? smem[lane] : 0.f;
        for (int off = 4; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, off);
        if (lane == 0) atomicAdd(d_inertia, v);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// kmeans — public entry point.
//
// Inputs:
//   d_X        — d_pcs × n_cells dense col-major fp32 (PCA-reduced embedding).
//   d_pcs      — embedding dimension (rows of d_X).
//   n_cells    — number of cells (cols of d_X).
//   k_clusters — target cluster count.
//   cfg        — KmeansConfig.
//   stream     — CUDA stream (nullptr = default context stream).
//
// Returns KmeansResult with device-resident labels[n_cells], centroids[d×k],
//   plus scalar fields iterations, inertia, converged.
// Caller synchronizes stream before reading device buffers.
// ---------------------------------------------------------------------------
inline KmeansResult kmeans(
    const float*     d_X,
    int              d_pcs,
    int              n_cells,
    int              k_clusters,
    const KmeansConfig& cfg    = {},
    cudaStream_t        stream = nullptr)
{
    if (stream == nullptr)
        stream = singlet_gpu::core::default_context().stream();

    if (k_clusters <= 0)
        throw std::invalid_argument(
            "kmeans: k_clusters must be > 0 (got " + std::to_string(k_clusters) + ")");
    if (k_clusters > n_cells)
        throw std::invalid_argument(
            "kmeans: k_clusters (" + std::to_string(k_clusters) +
            ") > n_cells (" + std::to_string(n_cells) + ")");
    if (d_pcs <= 0 || n_cells <= 0)
        throw std::invalid_argument("kmeans: d_pcs and n_cells must be > 0");
    if (!d_X)
        throw std::invalid_argument("kmeans: d_X is null");

    // -------------------------------------------------------------------------
    // Allocate device buffers
    // -------------------------------------------------------------------------
    const size_t dk_sz  = static_cast<size_t>(d_pcs)    * static_cast<size_t>(k_clusters);
    const size_t nk_sz  = static_cast<size_t>(n_cells)  * static_cast<size_t>(k_clusters);

    core::DeviceMemory<float> d_C(dk_sz);       // current centroids [d × k]
    core::DeviceMemory<float> d_C_new(dk_sz);   // updated centroids each iter
    core::DeviceMemory<int>   d_labels(static_cast<size_t>(n_cells));
    core::DeviceMemory<int>   d_labels_new(static_cast<size_t>(n_cells));
    core::DeviceMemory<float> d_x_norm(static_cast<size_t>(n_cells));
    core::DeviceMemory<float> d_c_norm(static_cast<size_t>(k_clusters));
    core::DeviceMemory<float> d_D(nk_sz);       // Sgemm output [n × k]
    core::DeviceMemory<int>   d_count(static_cast<size_t>(k_clusters));
    core::DeviceMemory<int>   d_changes(1);
    core::DeviceMemory<float> d_inertia(1);

    // -------------------------------------------------------------------------
    // Forgy init — pick k random cell indices (no replacement), host-side mt19937
    // -------------------------------------------------------------------------
    {
        std::mt19937 rng(static_cast<uint32_t>(cfg.seed & 0xffffffffu));
        std::vector<int> indices(static_cast<size_t>(n_cells));
        for (int i = 0; i < n_cells; ++i) indices[static_cast<size_t>(i)] = i;
        // partial Fisher-Yates: shuffle first k_clusters slots
        for (int i = 0; i < k_clusters; ++i) {
            std::uniform_int_distribution<int> dist(i, n_cells - 1);
            int j = dist(rng);
            std::swap(indices[static_cast<size_t>(i)], indices[static_cast<size_t>(j)]);
        }
        // Copy selected columns of X into C, one per centroid
        for (int ki = 0; ki < k_clusters; ++ki) {
            const int src_col = indices[static_cast<size_t>(ki)];
            CUDA_CHECK(cudaMemcpyAsync(
                d_C.get() + static_cast<ptrdiff_t>(ki) * d_pcs,
                d_X       + static_cast<ptrdiff_t>(src_col) * d_pcs,
                static_cast<size_t>(d_pcs) * sizeof(float),
                cudaMemcpyDeviceToDevice, stream));
        }
    }

    // -------------------------------------------------------------------------
    // One-shot: ||X[:,c]||² for all cells
    // -------------------------------------------------------------------------
    km_col_norm_kernel<<<n_cells, KM_THREADS, 0, stream>>>(
        d_X, d_x_norm.get(), d_pcs, n_cells);

    // Seed labels to -1 so first iteration always counts all cells as changed
    CUDA_CHECK(cudaMemsetAsync(d_labels.get(), 0xff,
        static_cast<size_t>(n_cells) * sizeof(int), stream));

    cublasHandle_t blas = singlet_gpu::core::default_context().blas();
    cublasSetStream(blas, stream);

    const float sgemm_alpha = 1.f, sgemm_beta = 0.f;

    // -------------------------------------------------------------------------
    // Lloyd loop
    // -------------------------------------------------------------------------
    KmeansResult res;
    res.n_cells    = n_cells;
    res.d_pcs      = d_pcs;
    res.k_clusters = k_clusters;
    res.converged  = false;
    res.iterations = 0;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {

        // (a) ||C[:,k]||²
        km_col_norm_kernel<<<k_clusters, KM_THREADS, 0, stream>>>(
            d_C.get(), d_c_norm.get(), d_pcs, k_clusters);

        // (b) D = X^T · C   (n × k col-major)
        // X: d × n col-major; C: d × k col-major
        // cublasSgemm(op=T, op=N, m=n, n=k, k=d) gives (n × k) col-major
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_cells,    // m (rows of D)
            k_clusters, // n (cols of D)
            d_pcs,      // k (inner dim)
            &sgemm_alpha,
            d_X,   d_pcs,      // A = X  (d × n), ldA = d
            d_C.get(), d_pcs,  // B = C  (d × k), ldB = d
            &sgemm_beta,
            d_D.get(), n_cells // C = D  (n × k), ldC = n
        ));

        // (c) Dist + argmin → d_labels_new
        km_assign_kernel<<<n_cells, KM_THREADS, 0, stream>>>(
            d_D.get(), d_x_norm.get(), d_c_norm.get(),
            d_labels_new.get(), n_cells, k_clusters);

        // (d) Count changes (D2H scalar — Rule 4 approved exception)
        CUDA_CHECK(cudaMemsetAsync(d_changes.get(), 0, sizeof(int), stream));
        {
            const int g = (n_cells + KM_THREADS - 1) / KM_THREADS;
            km_count_changes_kernel<<<g, KM_THREADS, 0, stream>>>(
                d_labels.get(), d_labels_new.get(), d_changes.get(), n_cells);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int h_changes = 0;
        CUDA_CHECK(cudaMemcpy(&h_changes, d_changes.get(), sizeof(int), cudaMemcpyDeviceToHost));

        // Swap labels
        {
            int* tmp = d_labels.get();
            // Manual pointer swap via underlying raw get — use cudaMemcpy swap:
            CUDA_CHECK(cudaMemcpyAsync(d_labels.get(), d_labels_new.get(),
                static_cast<size_t>(n_cells) * sizeof(int),
                cudaMemcpyDeviceToDevice, stream));
        }

        ++res.iterations;

        // (e) Check convergence
        if (h_changes <= cfg.tol_changes) {
            res.converged = true;
            break;
        }

        // (f) Update centroids: zero C_new, atomic scatter, divide
        CUDA_CHECK(cudaMemsetAsync(d_C_new.get(), 0, dk_sz * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_count.get(), 0,
            static_cast<size_t>(k_clusters) * sizeof(int), stream));
        {
            const int g = (n_cells + KM_THREADS - 1) / KM_THREADS;
            km_scatter_kernel<<<g, KM_THREADS, 0, stream>>>(
                d_X, d_labels.get(), d_C_new.get(), d_count.get(), d_pcs, n_cells);
        }
        {
            const int total = d_pcs * k_clusters;
            const int g = (total + KM_THREADS - 1) / KM_THREADS;
            km_centroid_divide_kernel<<<g, KM_THREADS, 0, stream>>>(
                d_C_new.get(), d_count.get(), d_pcs, k_clusters);
        }
        // Swap C and C_new (device-to-device copy)
        CUDA_CHECK(cudaMemcpyAsync(d_C.get(), d_C_new.get(),
            dk_sz * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    // -------------------------------------------------------------------------
    // Final inertia
    // -------------------------------------------------------------------------
    CUDA_CHECK(cudaMemsetAsync(d_inertia.get(), 0, sizeof(float), stream));
    km_inertia_kernel<<<n_cells, KM_THREADS, 0, stream>>>(
        d_X, d_C.get(), d_labels.get(), d_inertia.get(), d_pcs, n_cells);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(&res.inertia, d_inertia.get(),
        sizeof(float), cudaMemcpyDeviceToHost));

    res.labels    = std::move(d_labels);
    res.centroids = std::move(d_C);
    return res;
}

}  // namespace graph
}  // namespace singlet_gpu
