// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/preprocess/deconv_size_factors.h
//
// Scran deconvolution size factors (Lun, Bach, Marioni 2016, Genome Biology).
// GPU port of scran::computeSumFactors — entirely device-side, zero H/D copies
// in any hot path (one scalar copy per cluster for mean_lib; permitted per Rule 4).
//
// Algorithm reference:
//   Lun ATL, Bach K, Marioni JC (2016) Pooling across cells to normalize
//   single-cell RNA-seq data with many zero counts. Genome Biology 17:75.
//   https://doi.org/10.1186/s13059-016-0947-7
//
// Time complexity:
//   Step 1 (lib sizes):   O(nnz)      — cub::DeviceSegmentedReduce
//   Step 2 (sort):        O(n log n)  — cub::DeviceRadixSort
//   Step 3 (pool sums):   O(n·P)      — cuBLAS SGEMV (A is dense pool matrix)
//   Step 4 (QR solve):    O(P²·n)     — cuSOLVER Sgeqrf+Sormqr + cuBLAS Strsm
//   Step 5 (median):      O(n log n)  — cub::DeviceRadixSort on SFs
//
// Workspace budget per cluster of n_cluster cells (5 window sizes, P=5·n pools):
//   d_A  (pool matrix, col-major): P × n_cluster × 4 bytes = 5·3000²·4 ≈ 180 MB
//   d_p  (pool sums):              P × 4 bytes              ≈ 60 KB
//   d_tau, QR workspace:           n_cluster × 4 + W × 4   ≈ 12–100 KB
//   d_sf (size factors):           n_cluster × 4 bytes      ≈ 12 KB
//   Total realistic peak (n=3000): ~180 MB per cluster.
//
// Stream: all ops enqueued on caller-supplied cudaStream_t; sync'd at function
//   exit before returning (caller may use result immediately without sync).
//
// Precision: fp32 throughout (cusolverDnSgeqrf / cusolverDnSormqr).
//   Pool sums use cuBLAS SGEMV which accumulates in device fp32.
//
// Determinism: cub::DeviceRadixSort is deterministic; cuSOLVER QR is
//   deterministic; cuBLAS SGEMV is deterministic for fixed operand order.
//   atomicAdd in clip_negatives is the only non-deterministic op (count only,
//   never touches the size factor values).
//
// OOC plan (feature 17 integration):
//   lib[c] accumulation (step 1) is trivially shardable — accumulate per-shard
//   partial sums via cub::DeviceSegmentedReduce on device; merge on device.
//   The QR solve operates on the aggregated lib-size vector only — no count
//   matrix access after step 1. Streaming works out-of-the-box for solve phase.
//   Each cluster processes at most max_cluster_size cells — fits in device
//   memory independent of total n.
//
// ## Streaming
//   Gram-matrix path A^T A shardable for feature 17 integration; current v1 is
//   single-shard per cluster. To shard: (a) accumulate lib[c] with
//   cub::DeviceSegmentedReduce shard by shard on device, (b) broadcast the
//   final vector to all shards for the pool-matrix build, (c) pool sums are
//   pure library-size arithmetic — count matrix not needed again after step 1.
//   Multi-GPU: each GPU handles disjoint clusters; NCCL allreduce for the
//   inter-cluster scaling constants (step 5).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include "singlet-gpu/core/types.h"
#include "singlet-gpu/core/handles.h"

// CYCLE-106: factornet/gpu/types.cuh replaced by native core/types.h.
#include <singlet-gpu/core/types.h>

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

namespace singlet_gpu {
namespace preprocess {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct DeconvSizeFactorsConfig {
    std::array<int, 5> pool_sizes     = {21, 41, 61, 81, 101};
    int   max_cluster_size            = 3000;
    bool  positive                    = true;   // NNLS clip: floor negatives at 1e-6
    float min_mean                    = 0.1f;   // gene filter (reserved, unused v1)
    int   max_nnls_iters              = 3;       // NNLS refinement passes
    uint64_t seed                     = 0;       // reserved — determinism via sort only
};

struct DeconvSizeFactorsResult {
    core::DeviceMemory<float> size_factors;   // length = n_cells, median = 1
    float  median_sf            = 1.0f;
    int    n_pools_used         = 0;
    int    n_clusters_solved    = 0;
    int    n_clipped_negatives  = 0;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels — anonymous namespace to avoid ODR violations when
// multiple translation units include this header.
// All __global__ functions at namespace scope — required by nvcc for header-only.
// ---------------------------------------------------------------------------
namespace {

#define SGPU_DSF_CUDA_CHECK(err)                                                \
    do {                                                                        \
        cudaError_t _e = (err);                                                 \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error: ") + cudaGetErrorString(_e)            \
                + " at " __FILE__ ":" + std::to_string(__LINE__));              \
        }                                                                       \
    } while (0)

#define SGPU_DSF_CUSOLVER_CHECK(err)                                            \
    do {                                                                        \
        cusolverStatus_t _e = (err);                                            \
        if (_e != CUSOLVER_STATUS_SUCCESS) {                                    \
            throw std::runtime_error(                                           \
                "cuSOLVER error " + std::to_string(static_cast<int>(_e))        \
                + " at " __FILE__ ":" + std::to_string(__LINE__));              \
        }                                                                       \
    } while (0)

#define SGPU_DSF_CUBLAS_CHECK(err)                                              \
    do {                                                                        \
        cublasStatus_t _e = (err);                                              \
        if (_e != CUBLAS_STATUS_SUCCESS) {                                      \
            throw std::runtime_error(                                           \
                "cuBLAS error " + std::to_string(static_cast<int>(_e))          \
                + " at " __FILE__ ":" + std::to_string(__LINE__));              \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// dsf_iota_kernel: fill int array with [0, 1, ..., n-1] on device.
// WHY: cub::CountingIterator + DeviceRadixSort::SortPairs can initialise the
// value input inline but DeviceRadixSort needs a writable output buffer.
// A trivial iota avoids a host→device copy of a pre-built index array
// (which would require pinned staging — unnecessary overhead).
// ---------------------------------------------------------------------------
__global__ void dsf_iota_kernel(int* __restrict__ out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = i;
}

// ---------------------------------------------------------------------------
// dsf_build_pool_matrix_kernel:
//
// Fill the pool matrix A (col-major, n_pools × nc, fp32) with 0/1 entries.
// Pool row index k = size_idx * nc + start_i.  Cell j contributes to pool k
// iff j is within [start_i, start_i + s) (mod nc) where s = pool_sizes[size_idx].
//
// Each thread handles one pool row: writes 1.0f to s column slots.
// Grid: ceil(n_pools / 256) × 1.
//
// WHY col-major: cuSOLVER Sgeqrf expects FORTRAN col-major (lda = leading dim
// = number of rows = n_pools). Storing A col-major avoids a transpose.
// ---------------------------------------------------------------------------
__global__ void dsf_build_pool_matrix_kernel(
    float*       __restrict__ d_A,          // col-major n_pools × nc, lda=n_pools
    const int*   __restrict__ pool_sizes_d, // n_sizes pool sizes
    int n_sizes,
    int nc,
    int n_pools)
{
    const int pool_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (pool_row >= n_pools) return;

    const int size_idx = pool_row / nc;
    const int start_i  = pool_row % nc;
    if (size_idx >= n_sizes) return;

    const int s = pool_sizes_d[size_idx];
    for (int k = 0; k < s; ++k) {
        const int cell_j = (start_i + k) % nc;
        // col-major index: row + col * lda  =  pool_row + cell_j * n_pools
        d_A[pool_row + (long long)cell_j * n_pools] = 1.0f;
    }
}

// ---------------------------------------------------------------------------
// dsf_scatter_size_factors_kernel:
//
// Writes size factors from sorted-cell order back to original cell order.
// sf_out[sorted_idx[i]] = sf_cluster[i].
// ---------------------------------------------------------------------------
__global__ void dsf_scatter_size_factors_kernel(
    const float* __restrict__ sf_cluster,   // SFs in sorted order
    const int*   __restrict__ sorted_idx,   // original cell indices (sorted by lib)
    float*       __restrict__ sf_out,       // output: original cell order
    int nc)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    sf_out[sorted_idx[i]] = sf_cluster[i];
}

// ---------------------------------------------------------------------------
// dsf_fill_ones_kernel: fill float array with 1.0f.
// Used for clusters too small for pooling (SF=1 fallback).
// ---------------------------------------------------------------------------
__global__ void dsf_fill_ones_kernel(float* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 1.0f;
}

// ---------------------------------------------------------------------------
// dsf_clip_negatives_kernel:
//
// NNLS projection: clamp SF[c] < 0 to floor_val.
// Atomically increments d_clip_count for each clipped cell.
// Non-determinism: only in the count (not in SF values), acceptable per design.
// ---------------------------------------------------------------------------
__global__ void dsf_clip_negatives_kernel(
    float* __restrict__ sf,
    int*   __restrict__ d_clip_count,
    int    nc,
    float  floor_val)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nc) return;
    if (sf[i] < 0.0f) {
        sf[i] = floor_val;
        atomicAdd(d_clip_count, 1);
    }
}

// ---------------------------------------------------------------------------
// dsf_scale_scatter_kernel:
//
// Apply a scalar scale to all cells belonging to a cluster by scattering
// via their original indices. Used for inter-cluster normalization.
// ---------------------------------------------------------------------------
__global__ void dsf_scale_scatter_kernel(
    float*       __restrict__ sf_out,
    const int*   __restrict__ idx,
    float        scale,
    int          n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sf_out[idx[i]] *= scale;
}

// ---------------------------------------------------------------------------
// solve_cluster:
//
// Given d_lib_sorted[nc] (device, ascending lib sizes for this cluster),
// builds pool matrix A (n_pools × nc), computes p = A * lib (SGEMV),
// reference-scales p by mean(lib), then solves A s = p via QR.
// Writes result to d_sf_cluster[nc].
//
// One scalar H/D copy per cluster for mean_lib (Rule 4: permitted scalar).
// ---------------------------------------------------------------------------
static void solve_cluster(
    cusolverDnHandle_t  cusolver,
    cublasHandle_t      cublas,
    cudaStream_t        stream,
    const float*        d_lib_sorted,   // device: ascending lib sizes, nc elems
    int                 nc,
    const int*          d_pool_sizes,   // device: n_sizes pool sizes
    int                 n_sizes,
    int                 n_pools,        // = n_sizes * nc
    bool                positive,
    int                 max_nnls_iters,
    float*              d_sf_cluster,   // output: device sf[nc]
    int&                n_clipped)
{
    // ---- allocate and zero pool matrix A: n_pools × nc, col-major ----
    core::DeviceMemory<float> d_A(static_cast<size_t>(n_pools) * nc);
    SGPU_DSF_CUDA_CHECK(cudaMemsetAsync(d_A.get(), 0,
        static_cast<size_t>(n_pools) * nc * sizeof(float), stream));

    // ---- fill pool matrix ----
    {
        const int threads = 256;
        const int blocks  = (n_pools + threads - 1) / threads;
        dsf_build_pool_matrix_kernel<<<blocks, threads, 0, stream>>>(
            d_A.get(), d_pool_sizes, n_sizes, nc, n_pools);
        SGPU_DSF_CUDA_CHECK(cudaGetLastError());
    }

    // ---- pool sums p = A * lib_sorted (SGEMV) ----
    // A is col-major n_pools×nc: m=n_pools, n=nc, lda=n_pools.
    // CUBLAS_OP_N: p = alpha * A * lib + beta * p.
    core::DeviceMemory<float> d_p(n_pools);
    {
        const float alpha = 1.0f, beta = 0.0f;
        SGPU_DSF_CUBLAS_CHECK(cublasSgemv(cublas, CUBLAS_OP_N,
            n_pools, nc,
            &alpha, d_A.get(), n_pools,
            d_lib_sorted, 1,
            &beta, d_p.get(), 1));
    }

    // ---- reference scaling: p[k] /= mean_lib ----
    // Compute mean_lib via cub sum + one scalar D→H copy (4 bytes, Rule 4 ok).
    {
        core::DeviceMemory<float> d_sum(1);
        size_t tmp_bytes = 0;
        cub::DeviceReduce::Sum(nullptr, tmp_bytes,
            d_lib_sorted, d_sum.get(), nc, stream);
        core::DeviceMemory<char> d_tmp(tmp_bytes + 1);
        cub::DeviceReduce::Sum(d_tmp.get(), tmp_bytes,
            d_lib_sorted, d_sum.get(), nc, stream);

        float h_sum = 1.0f;
        SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum.get(), sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));  // 4-byte scalar sync

        float mean_lib = h_sum / static_cast<float>(nc > 0 ? nc : 1);
        if (mean_lib < 1e-10f) mean_lib = 1.0f;
        const float scale_p = 1.0f / mean_lib;
        SGPU_DSF_CUBLAS_CHECK(cublasSscal(cublas, n_pools, &scale_p, d_p.get(), 1));
    }

    // ---- QR solve: A s = p ----
    //
    // A is n_pools × nc (overdetermined, col-major, lda=n_pools).
    //   1. geqrf  → R in upper triangle of A[0:nc, 0:nc], Q as Householder taus
    //   2. ormqr  → d_p = Q^T d_p   (first nc entries of result = Q^T p)
    //   3. Strsm  → solve R s = p'[0:nc]  (result back into d_p[0:nc])
    //
    // cuSOLVER writes A in-place.  d_p is reused for Q^T p and then for s.
    //
    core::DeviceMemory<float> d_tau(nc);
    core::DeviceMemory<int>   d_info(1);

    // Workspace query for geqrf
    int lwork_qr = 0;
    SGPU_DSF_CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(cusolver,
        n_pools, nc, d_A.get(), n_pools, &lwork_qr));

    // Workspace query for ormqr (apply Q^T to column vector p: m=n_pools, n=1, k=nc)
    int lwork_ormqr = 0;
    SGPU_DSF_CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(cusolver,
        CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        n_pools, 1, nc,
        d_A.get(), n_pools,
        d_tau.get(),
        d_p.get(), n_pools,
        &lwork_ormqr));

    const int lwork = std::max(lwork_qr, lwork_ormqr);
    core::DeviceMemory<float> d_work(lwork > 0 ? lwork : 1);

    // Step 1: geqrf
    SGPU_DSF_CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver,
        n_pools, nc,
        d_A.get(), n_pools,
        d_tau.get(),
        d_work.get(), lwork,
        d_info.get()));

    // Step 2: ormqr — apply Q^T to d_p (vector, nrhs=1, ldc=n_pools)
    SGPU_DSF_CUSOLVER_CHECK(cusolverDnSormqr(cusolver,
        CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
        n_pools, 1, nc,
        d_A.get(), n_pools,
        d_tau.get(),
        d_p.get(), n_pools,
        d_work.get(), lwork,
        d_info.get()));

    // Step 3: Strsm — solve R s = p'[0:nc], result in d_p[0:nc]
    // R is upper triangular in the first nc rows/cols of d_A.
    // d_p[0:nc] holds the first nc entries of Q^T p (in a column-major vector
    // of length n_pools, stride 1 — so d_p[0:nc] are the first nc elements).
    // We treat d_p as a (n_pools × 1) matrix; Strsm operates on the nc-length
    // leading subvector. Since Strsm is column-major and nrhs=1, ldb=n_pools works
    // (Strsm only reads/writes the first nc rows of the RHS column).
    {
        const float alpha = 1.0f;
        // SIDE_LEFT: A_triangular * X = alpha * B   →  X = alpha * A^{-1} B
        // A is upper triangular nc×nc in d_A[0:nc,0:nc] (lda=n_pools)
        // B is the nc×1 matrix d_p[0:nc] (ldb=n_pools; only first nc rows used)
        SGPU_DSF_CUBLAS_CHECK(cublasStrsm(cublas,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            nc, 1,         // m=nc (rows of triangular system), n=1 (nrhs)
            &alpha,
            d_A.get(), n_pools,   // A: upper tri in d_A, lda=n_pools
            d_p.get(), n_pools)); // B/X: d_p, ldb=n_pools
        // d_p[0:nc] now holds the size factors in sorted-cell order.
    }

    // Copy s (first nc elements of d_p) to d_sf_cluster.
    SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(d_sf_cluster, d_p.get(),
        static_cast<size_t>(nc) * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // ---- NNLS clip (if positive=true) ----
    if (positive) {
        constexpr float kFloor = 1e-6f;
        core::DeviceMemory<int> d_clip_count(1);
        for (int iter = 0; iter < max_nnls_iters; ++iter) {
            SGPU_DSF_CUDA_CHECK(cudaMemsetAsync(d_clip_count.get(), 0, sizeof(int), stream));
            const int threads = 256;
            const int blocks  = (nc + threads - 1) / threads;
            dsf_clip_negatives_kernel<<<blocks, threads, 0, stream>>>(
                d_sf_cluster, d_clip_count.get(), nc, kFloor);
            SGPU_DSF_CUDA_CHECK(cudaGetLastError());

            // Scalar copy to check clip count (4 bytes, one per NNLS iter — allowed).
            int h_clip = 0;
            SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(&h_clip, d_clip_count.get(), sizeof(int),
                cudaMemcpyDeviceToHost, stream));
            SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));
            n_clipped += h_clip;
            // v1: simple clip without QR refit. Break after one pass.
            break;
        }
    }
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// Public function: compute_deconv_size_factors
// ---------------------------------------------------------------------------

inline DeconvSizeFactorsResult compute_deconv_size_factors(
    const core::DeviceCSC&         counts,
    const int32_t*                 optional_cluster_labels,
    const DeconvSizeFactorsConfig& cfg,
    cudaStream_t                   stream)
{
    const int n_cells = counts.cols;

    if (n_cells == 0) {
        throw std::invalid_argument(
            "compute_deconv_size_factors: matrix has 0 columns (cells)");
    }

    // Resolve stream.
    if (!stream) {
        stream = core::default_context().stream();
    }
    // Borrow handles from the default context and rebind to our stream.
    cusolverDnHandle_t cusolver = core::default_context().solver();
    cublasHandle_t     cublas   = core::default_context().blas();
    SGPU_DSF_CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    SGPU_DSF_CUBLAS_CHECK(cublasSetStream(cublas, stream));

    DeconvSizeFactorsResult result;
    result.size_factors = core::DeviceMemory<float>(n_cells);
    SGPU_DSF_CUDA_CHECK(cudaMemsetAsync(result.size_factors.get(), 0,
        static_cast<size_t>(n_cells) * sizeof(float), stream));

    // -------------------------------------------------------------------------
    // Step 1: per-cell library sizes via cub::DeviceSegmentedReduce
    //
    // CSC layout: col_ptr[n_cells+1], row_indices[nnz], values[nnz].
    // Lib[j] = sum of values in column j.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_lib(n_cells);
    {
        size_t tmp_bytes = 0;
        cub::DeviceSegmentedReduce::Sum(
            nullptr, tmp_bytes,
            counts.values.get(), d_lib.get(),
            n_cells,
            counts.col_ptr.get(), counts.col_ptr.get() + 1,
            stream);
        core::DeviceMemory<char> d_tmp(tmp_bytes + 1);
        cub::DeviceSegmentedReduce::Sum(
            d_tmp.get(), tmp_bytes,
            counts.values.get(), d_lib.get(),
            n_cells,
            counts.col_ptr.get(), counts.col_ptr.get() + 1,
            stream);
    }

    // -------------------------------------------------------------------------
    // Step 2: sort cells by lib size ascending; tie-break is implicit (sort is
    // stable in cub::DeviceRadixSort for equal keys on the same architecture).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_lib_sorted(n_cells);
    core::DeviceMemory<int>   d_sorted_idx(n_cells);
    {
        core::DeviceMemory<int> d_idx_in(n_cells);
        {
            const int threads = 256, blocks = (n_cells + 255) / 256;
            dsf_iota_kernel<<<blocks, threads, 0, stream>>>(d_idx_in.get(), n_cells);
            SGPU_DSF_CUDA_CHECK(cudaGetLastError());
        }

        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, tmp_bytes,
            d_lib.get(), d_lib_sorted.get(),
            d_idx_in.get(), d_sorted_idx.get(),
            n_cells, 0, sizeof(float) * 8, stream);
        core::DeviceMemory<char> d_tmp(tmp_bytes + 1);
        cub::DeviceRadixSort::SortPairs(
            d_tmp.get(), tmp_bytes,
            d_lib.get(), d_lib_sorted.get(),
            d_idx_in.get(), d_sorted_idx.get(),
            n_cells, 0, sizeof(float) * 8, stream);
    }

    // -------------------------------------------------------------------------
    // Step 3: cluster assignment
    //
    // single cluster when: no labels provided OR n_cells <= max_cluster_size.
    // multi-cluster: split sorted order into contiguous blocks of <= max_cluster_size.
    // Cluster boundary calculation is host-side from a one-time setup D→H transfer.
    // -------------------------------------------------------------------------
    std::vector<int> cluster_starts;
    std::vector<int> cluster_sizes;

    if (optional_cluster_labels == nullptr || n_cells <= cfg.max_cluster_size) {
        cluster_starts.push_back(0);
        cluster_sizes.push_back(n_cells);
    } else {
        // One-time setup: pull sorted indices + labels to host.
        std::vector<int> h_sorted_idx(n_cells);
        SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(h_sorted_idx.data(), d_sorted_idx.get(),
            static_cast<size_t>(n_cells) * sizeof(int), cudaMemcpyDeviceToHost, stream));
        std::vector<int32_t> h_labels(n_cells);
        SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(h_labels.data(), optional_cluster_labels,
            static_cast<size_t>(n_cells) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
        SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));
        (void)h_sorted_idx; (void)h_labels;  // label-guided split deferred to v2

        // v1: contiguous split by sorted order; ignore label affinity.
        int start = 0;
        while (start < n_cells) {
            const int end = std::min(start + cfg.max_cluster_size, n_cells);
            cluster_starts.push_back(start);
            cluster_sizes.push_back(end - start);
            start = end;
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: upload pool_sizes to device (one-time, tiny — 20 bytes)
    // -------------------------------------------------------------------------
    core::DeviceMemory<int> d_pool_sizes(cfg.pool_sizes.size());
    {
        std::vector<int> ps(cfg.pool_sizes.begin(), cfg.pool_sizes.end());
        SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(d_pool_sizes.get(), ps.data(),
            ps.size() * sizeof(int), cudaMemcpyHostToDevice, stream));
    }
    const int n_sizes = static_cast<int>(cfg.pool_sizes.size());
    const int min_pool_size = *std::max_element(cfg.pool_sizes.begin(), cfg.pool_sizes.end());

    // Per-cluster mean lib for inter-cluster scaling.
    std::vector<float> h_cluster_means;
    int total_clipped = 0;
    int total_pools   = 0;

    for (int ci = 0; ci < static_cast<int>(cluster_starts.size()); ++ci) {
        const int offset = cluster_starts[ci];
        const int nc     = cluster_sizes[ci];

        if (nc < min_pool_size) {
            // Cluster too small for pooling — assign SF=1 and continue.
            core::DeviceMemory<float> d_sf_tmp(nc);
            {
                const int threads = 256, blocks = (nc + 255) / 256;
                dsf_fill_ones_kernel<<<blocks, threads, 0, stream>>>(d_sf_tmp.get(), nc);
                SGPU_DSF_CUDA_CHECK(cudaGetLastError());
            }
            {
                const int threads = 256, blocks = (nc + 255) / 256;
                dsf_scatter_size_factors_kernel<<<blocks, threads, 0, stream>>>(
                    d_sf_tmp.get(), d_sorted_idx.get() + offset,
                    result.size_factors.get(), nc);
                SGPU_DSF_CUDA_CHECK(cudaGetLastError());
            }
            h_cluster_means.push_back(1.0f);
            continue;
        }

        const int n_pools = n_sizes * nc;
        total_pools += n_pools;

        core::DeviceMemory<float> d_sf_cluster(nc);
        int cluster_clipped = 0;

        solve_cluster(cusolver, cublas, stream,
            d_lib_sorted.get() + offset, nc,
            d_pool_sizes.get(), n_sizes, n_pools,
            cfg.positive, cfg.max_nnls_iters,
            d_sf_cluster.get(),
            cluster_clipped);
        total_clipped += cluster_clipped;

        // Compute mean lib for inter-cluster scaling (one scalar per cluster).
        float h_mean = 1.0f;
        {
            core::DeviceMemory<float> d_sum(1);
            size_t tmp_bytes = 0;
            cub::DeviceReduce::Sum(nullptr, tmp_bytes,
                d_lib_sorted.get() + offset, d_sum.get(), nc, stream);
            core::DeviceMemory<char> d_tmp(tmp_bytes + 1);
            cub::DeviceReduce::Sum(d_tmp.get(), tmp_bytes,
                d_lib_sorted.get() + offset, d_sum.get(), nc, stream);
            SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(&h_mean, d_sum.get(), sizeof(float),
                cudaMemcpyDeviceToHost, stream));
            SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));
            h_mean = h_mean / static_cast<float>(nc);
        }
        h_cluster_means.push_back(h_mean > 1e-10f ? h_mean : 1.0f);

        // Scatter SFs to output in original cell order.
        {
            const int threads = 256, blocks = (nc + 255) / 256;
            dsf_scatter_size_factors_kernel<<<blocks, threads, 0, stream>>>(
                d_sf_cluster.get(), d_sorted_idx.get() + offset,
                result.size_factors.get(), nc);
            SGPU_DSF_CUDA_CHECK(cudaGetLastError());
        }
    }

    // -------------------------------------------------------------------------
    // Step 5a: inter-cluster scaling (multi-cluster only)
    //
    // Reference = median-lib cluster. Scale each cluster's SFs by
    // ref_mean / cluster_mean so that cluster SFs are commensurate.
    // -------------------------------------------------------------------------
    if (cluster_starts.size() > 1) {
        std::vector<float> sorted_means = h_cluster_means;
        std::sort(sorted_means.begin(), sorted_means.end());
        const float ref_mean = sorted_means[sorted_means.size() / 2];

        for (int ci = 0; ci < static_cast<int>(cluster_starts.size()); ++ci) {
            const float scale_ci = (h_cluster_means[ci] > 1e-10f)
                ? ref_mean / h_cluster_means[ci]
                : 1.0f;
            if (std::fabs(scale_ci - 1.0f) < 1e-6f) continue;

            const int offset = cluster_starts[ci];
            const int nc     = cluster_sizes[ci];
            const int threads = 256, blocks = (nc + 255) / 256;
            dsf_scale_scatter_kernel<<<blocks, threads, 0, stream>>>(
                result.size_factors.get(),
                d_sorted_idx.get() + offset,
                scale_ci, nc);
            SGPU_DSF_CUDA_CHECK(cudaGetLastError());
        }
    }

    // -------------------------------------------------------------------------
    // Step 5b: normalize so that median(SF) = 1.
    //
    // Sort the SF array; read element at position n/2; divide all by that.
    // One scalar D→H copy for the median value.
    // -------------------------------------------------------------------------
    {
        core::DeviceMemory<float> d_sf_sorted(n_cells);
        core::DeviceMemory<int>   d_sf_idx_in(n_cells);
        core::DeviceMemory<int>   d_sf_idx_out(n_cells);

        {
            const int threads = 256, blocks = (n_cells + 255) / 256;
            dsf_iota_kernel<<<blocks, threads, 0, stream>>>(d_sf_idx_in.get(), n_cells);
            SGPU_DSF_CUDA_CHECK(cudaGetLastError());
        }

        size_t tmp_bytes = 0;
        cub::DeviceRadixSort::SortPairs(
            nullptr, tmp_bytes,
            result.size_factors.get(), d_sf_sorted.get(),
            d_sf_idx_in.get(), d_sf_idx_out.get(),
            n_cells, 0, sizeof(float) * 8, stream);
        core::DeviceMemory<char> d_tmp(tmp_bytes + 1);
        cub::DeviceRadixSort::SortPairs(
            d_tmp.get(), tmp_bytes,
            result.size_factors.get(), d_sf_sorted.get(),
            d_sf_idx_in.get(), d_sf_idx_out.get(),
            n_cells, 0, sizeof(float) * 8, stream);

        float h_median = 1.0f;
        SGPU_DSF_CUDA_CHECK(cudaMemcpyAsync(&h_median,
            d_sf_sorted.get() + (n_cells / 2), sizeof(float),
            cudaMemcpyDeviceToHost, stream));
        SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));
        if (h_median < 1e-10f) h_median = 1.0f;

        const float inv_median = 1.0f / h_median;
        SGPU_DSF_CUBLAS_CHECK(cublasSscal(cublas, n_cells,
            &inv_median, result.size_factors.get(), 1));
        result.median_sf = h_median;
    }

    result.n_pools_used        = total_pools;
    result.n_clusters_solved   = static_cast<int>(cluster_starts.size());
    result.n_clipped_negatives = total_clipped;

    SGPU_DSF_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

}  // namespace preprocess
}  // namespace singlet_gpu
