// SPDX-License-Identifier: MIT
// integrates: original (Harmony adapted) + cycle 8 KNN (BBKNN)
//
// integrate/harmony.h — GPU Harmony batch-integration kernel.
//
// Algorithm: Korsunsky et al. 2019 (Nature Methods 16:1289–1296).
//   Iterative soft k-means correction:
//     1. Init: cuBLAS GEMM distances to K centroids → softmax → R[n × K].
//     2. Loop:
//        a. Centroid accumulation: cub::DeviceSegmentedReduce::Sum keyed by
//           (cluster_id * n_batches + batch_id) segments — all on device.
//        b. Correction kernel: Z_j -= Σ_k R[j][k] * (centroid[k][b_j] - global[k]).
//        c. Soft-assignment update: GEMM to K centroids + softmax → R_new.
//        d. Convergence: cub::DeviceReduce::Max on |R_new - R_old| → 4-byte host copy.
//     3. Return corrected Z (copy of input + corrections), n_iters_used, final_delta.
//
// Centroid accumulation — CYCLE-14 FIX (zero host transfers inside the iteration loop):
//   Segment offsets = scaled_offsets_dev[(B+1) elements] uploaded ONCE at entry.
//   scaled_offsets_dev[b] = batch_offsets[b] * d  (scales the flat n×d layout).
//   Per cluster k, per iteration:
//     build_weighted_z_sorted_kernel → weighted_sorted[n×d] + denom_sorted[n]
//     cub::DeviceSegmentedReduce::Sum → seg_numer[B×d]  (flat reduce of n*d floats)
//     cub::DeviceSegmentedReduce::Sum → seg_denom[B]    (flat reduce of n floats)
//     scatter_cent_kb_kernel / scatter_denom_kb_kernel → centroids_kb / denom_kb
//     normalize_cent_kb_kernel → centroids_kb / (denom_kb + lambda)
//   Global centroid: cuBLAS GEMM + normalize_cent_global_kernel — no host transfer.
//   The ONLY per-iter host transfer: 4-byte final_delta scalar for convergence check.
//
//   PCIe budget:
//     Before (cycle-14 bug): ~280 MB/iter × 10 iters ≈ 2.8 GB per Harmony run.
//     After  (this fix): (B+1)*4 bytes one-time upload + 4 bytes × max_iter ≈ <1 KB.
//
// One-time setup uploads (function entry, NOT per-iter):
//   h_Z + h_batch → host K-means++ init (download of Z once, at entry only).
//   sorted_indices_dev[n]         — stable sort of cell indices by batch_id.
//   batch_offsets_dev[B+1]        — per-batch start/end in the sorted list.
//   scaled_offsets_dev[B+1]       — batch_offsets * d (for flat n×d segmented reduce).
//   initial centroids_global      — K×d from k-means++ seed rows.
//
// Memory budget (n cells, d=50 PCs, K clusters, B batches):
//   Z:              4*n*d bytes (embedding copy, corrected in-place)
//   R:              4*n*K bytes (soft assignments)
//   centroids_kb:   4*K*B*d bytes
//   weighted_sorted:4*n*d bytes (per-cluster scratch, reused each k iter)
//   denom_sorted:   4*n bytes
//   seg_numer:      4*B*d bytes
//   Example 1M × K=20 × B=10 × d=50: 200 MB Z + 80 MB R + 200 MB weighted ≈ 480 MB.
//
// Streams: 1, caller-provided.
// Precision: fp32 throughout; stable softmax (max-subtracted per row).
// Determinism: always deterministic (centroid accum is segmented reduce, not atomicAdd).
//   cfg.deterministic is preserved as a no-op flag for API compatibility.
// OOC: Z and R must fit on device. For n>2M, reduce d first via streaming PCA.

#pragma once

#include <singlet/gpu/integrate/types.h>
#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cooperative_groups.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>

namespace singlet::gpu {
namespace integrate {

// ─── Device kernels ───────────────────────────────────────────────────────────

namespace detail {

// Row-wise stable softmax on D[n × K] in-place.
// Divides by the regularization denominator (lambda + Σ exp) per design doc.
__launch_bounds__(256, 4)
__global__ void
softmax_rows_kernel(float* __restrict__ D, int n, int K, float lambda)
{
    int row = blockIdx.x;
    if (row >= n) return;
    float* r = D + (size_t)row * K;

    float mx = -1e30f;
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        mx = fmaxf(mx, r[j]);
    for (int off = 16; off > 0; off >>= 1)
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, off));

    extern __shared__ float smem[];
    float* s_exp = smem;
    float* s_sum = smem + K;
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        float e = __expf(r[j] - mx);
        s_exp[j] = e;
        local_sum += e;
    }
    s_sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) s_sum[threadIdx.x] += s_sum[threadIdx.x + st];
        __syncthreads();
    }
    float inv = 1.f / (s_sum[0] + lambda);
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        r[j] = s_exp[j] * inv;
}

// D[i,k] += ||z_i||^2 + ||c_k||^2 — added to the -2*Z*C^T GEMM result to form ||z-c||^2.
__launch_bounds__(256, 4)
__global__ void
add_centroid_norms_kernel(float* __restrict__ D,
                          const float* __restrict__ z_norms,
                          const float* __restrict__ c_norms,
                          int n, int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= n || k >= K) return;
    D[(size_t)i * K + k] += z_norms[i] + c_norms[k];
}

// Row L2 norms squared of an m×d matrix.
__launch_bounds__(256, 4)
__global__ void
row_norms_kernel(const float* __restrict__ Z, float* __restrict__ norms,
                 int n, int d)
{
    int row = blockIdx.x;
    if (row >= n) return;
    const float* r = Z + (size_t)row * d;
    extern __shared__ float sh[];
    float acc = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) acc += r[j] * r[j];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1) {
        if (threadIdx.x < st) sh[threadIdx.x] += sh[threadIdx.x + st];
        __syncthreads();
    }
    if (threadIdx.x == 0) norms[row] = sh[0];
}

// R^T: Wt[k*n + j] = R[j*K + k].  Used for cuBLAS GEMM → global centroids.
__launch_bounds__(256, 4)
__global__ void
transpose_r_kernel(const float* __restrict__ R,
                   float* __restrict__       Wt,
                   int n, int K)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    if (j >= n || k >= K) return;
    Wt[(size_t)k * n + j] = R[(size_t)j * K + k];
}

// Z[j] -= Σ_k R[j,k] * (cent_kb[k,b_j,:] - cent_global[k,:]).  One block per cell.
__launch_bounds__(256, 4)
__global__ void
apply_correction_kernel(float* __restrict__        Z,
                        const float* __restrict__  R,
                        const float* __restrict__  cent_kb,
                        const float* __restrict__  cent_global,
                        const int* __restrict__    batch_ids,
                        int n, int K, int n_batches, int d)
{
    int j = blockIdx.x;
    if (j >= n) return;
    int bj = batch_ids[j];
    float* zj = Z + (size_t)j * d;

    for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
        float corr = 0.f;
        for (int k = 0; k < K; ++k) {
            float r_jk = R[(size_t)j * K + k];
            float c_kb = cent_kb[((size_t)k * n_batches + bj) * d + dim];
            float c_gk = cent_global[(size_t)k * d + dim];
            corr += r_jk * (c_kb - c_gk);
        }
        zj[dim] -= corr;
    }
}

// |R_new[j] - R_old[j]| max across K, into delta[j].
__launch_bounds__(256, 4)
__global__ void
r_delta_kernel(const float* __restrict__ R_new,
               const float* __restrict__ R_old,
               float* __restrict__       delta,
               int n, int K)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    float mx = 0.f;
    for (int k = 0; k < K; ++k)
        mx = fmaxf(mx, fabsf(R_new[(size_t)j * K + k] - R_old[(size_t)j * K + k]));
    delta[j] = mx;
}

// ─── Centroid accumulation helpers (CYCLE-14 FIX) ────────────────────────────
//
// These kernels keep ALL centroid math on device via cub::DeviceSegmentedReduce.
// weighted_sorted[si * d + dim] = R[sorted_indices[si], k_idx] * Z[sorted_indices[si], dim]
// denom_sorted[si]              = R[sorted_indices[si], k_idx]
// where si ranges over [0, n) in batch-sorted cell order.
// DeviceSegmentedReduce::Sum then sums flat n*d floats into B*d floats in one call.

// Fill weighted_sorted[n×d] and denom_sorted[n] in batch-sorted cell order for cluster k.
// sorted_indices maps sorted position → original cell index.
__launch_bounds__(256, 4)
__global__ void
build_weighted_z_sorted_kernel(
        float* __restrict__       weighted_sorted,  // n × d, sorted-cell row order
        float* __restrict__       denom_sorted,     // n
        const float* __restrict__ R,                // n × K
        const float* __restrict__ Z,                // n × d
        const int*   __restrict__ sorted_indices,   // n  (batch-stable sort)
        int n, int K, int d, int k_idx)
{
    int si = blockIdx.x * blockDim.x + threadIdx.x;
    if (si >= n) return;
    int j   = sorted_indices[si];
    float w = R[(size_t)j * K + k_idx];
    denom_sorted[si] = w;
    const float* zj = Z + (size_t)j * d;
    float* wj = weighted_sorted + (size_t)si * d;
    for (int dim = 0; dim < d; ++dim)
        wj[dim] = w * zj[dim];
}

// Scatter seg_numer[B×d] into cent_kb[k_idx, 0..B-1, 0..d-1].
__launch_bounds__(256, 4)
__global__ void
scatter_cent_kb_kernel(float* __restrict__       cent_kb,
                       const float* __restrict__ seg_numer,
                       int k_idx, int n_batches, int d)
{
    int bd = blockIdx.x * blockDim.x + threadIdx.x;
    if (bd >= n_batches * d) return;
    cent_kb[(size_t)k_idx * n_batches * d + bd] = seg_numer[bd];
}

// Scatter seg_denom[B] into denom_kb[k_idx * n_batches + 0..B-1].
__launch_bounds__(256, 4)
__global__ void
scatter_denom_kb_kernel(float* __restrict__       denom_kb,
                        const float* __restrict__ seg_denom,
                        int k_idx, int n_batches)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_batches) return;
    denom_kb[(size_t)k_idx * n_batches + b] = seg_denom[b];
}

// cent_kb[k,b,:] /= (denom_kb[k*B+b] + lambda) in-place.  One thread per (k,b) pair.
__launch_bounds__(256, 4)
__global__ void
normalize_cent_kb_kernel(float* __restrict__       cent_kb,
                         const float* __restrict__ denom_kb,
                         int K, int n_batches, int d, float lambda)
{
    int kb = blockIdx.x * blockDim.x + threadIdx.x;
    if (kb >= K * n_batches) return;
    float den = denom_kb[kb] + lambda;
    float inv = (den > 0.f) ? (1.f / den) : 0.f;
    float* c = cent_kb + (size_t)kb * d;
    for (int dim = 0; dim < d; ++dim)
        c[dim] *= inv;
}

// cent_global[k,:] /= (denom_global[k] + lambda) in-place.  One block per k.
__launch_bounds__(256, 4)
__global__ void
normalize_cent_global_kernel(float* __restrict__       cent_global,
                              const float* __restrict__ denom_global,
                              int K, int d, float lambda)
{
    int k = blockIdx.x;
    if (k >= K) return;
    float den = denom_global[k] + lambda;
    float inv = (den > 0.f) ? (1.f / den) : 0.f;
    float* c = cent_global + (size_t)k * d;
    for (int dim = threadIdx.x; dim < d; dim += blockDim.x)
        c[dim] *= inv;
}

// Extract R column k_idx contiguously into col_out[n].
// Reused both for global-denom reduce and for per-cluster denom reduce fallback.
__launch_bounds__(256, 4)
__global__ void
extract_r_col_kernel(const float* __restrict__ R,
                     float* __restrict__       col_out,
                     int n, int K, int k_idx)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    col_out[j] = R[(size_t)j * K + k_idx];
}

}  // namespace detail


// ─── Harmony public function ──────────────────────────────────────────────────

// harmony() — GPU-native Harmony batch correction.
//
// embedding:    row-major n × d float on device (n cells, d PCA components).
// batch_labels: integer batch id per cell on device (values in [0, n_batches)).
// n_batches:    number of distinct batch labels.
// cfg:          HarmonyConfig (see types.h).
// stream:       caller-provided CUDA stream.
//
// Returns HarmonyResult::corrected (n×d device memory, copy of embedding + corrections).
inline HarmonyResult
harmony(const core::DeviceDense& embedding,
        const core::DeviceMemory<int>& batch_labels,
        int n_batches,
        const HarmonyConfig& cfg,
        cudaStream_t stream)
{
    const int n = (int)embedding.rows;
    const int d = (int)embedding.cols;
    const int K = cfg.n_clusters;

    if (n <= 0 || d <= 0) throw std::invalid_argument("harmony: empty embedding");
    if (n_batches <= 0)   throw std::invalid_argument("harmony: n_batches must be > 0");
    if (K <= 0)           throw std::invalid_argument("harmony: n_clusters must be > 0");

    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSetStream(blas, stream);

    // ── Device buffers ────────────────────────────────────────────────────────
    core::DeviceMemory<float> Z(static_cast<size_t>(n) * d);
    core::DeviceMemory<float> R(static_cast<size_t>(n) * K);
    core::DeviceMemory<float> R_old(static_cast<size_t>(n) * K);
    core::DeviceMemory<float> centroids_kb(static_cast<size_t>(K) * n_batches * d);
    core::DeviceMemory<float> centroids_global(static_cast<size_t>(K) * d);
    core::DeviceMemory<float> denom_kb(static_cast<size_t>(K) * n_batches);
    core::DeviceMemory<float> denom_global_dev(K);
    core::DeviceMemory<float> z_norms(n);
    core::DeviceMemory<float> c_norms(K);
    core::DeviceMemory<float> D(static_cast<size_t>(n) * K);
    core::DeviceMemory<float> Wt(static_cast<size_t>(K) * n);
    core::DeviceMemory<float> delta_per_cell(n);
    core::DeviceMemory<float> delta_max(1);
    // Segmented-reduce centroid buffers (CYCLE-14 fix).
    core::DeviceMemory<float> weighted_sorted(static_cast<size_t>(n) * d);
    core::DeviceMemory<float> denom_sorted(n);
    core::DeviceMemory<float> seg_numer(static_cast<size_t>(n_batches) * d);
    core::DeviceMemory<float> seg_denom(n_batches);
    core::DeviceMemory<float> r_col_scratch(n);
    core::DeviceMemory<int>   sorted_indices_dev(n);
    core::DeviceMemory<int>   batch_offsets_dev(n_batches + 1);
    core::DeviceMemory<int>   scaled_offsets_dev(n_batches + 1);

    // Copy embedding → Z (caller's buffer unchanged).
    cudaMemcpyAsync(Z.get(), embedding.data.get(),
                    static_cast<size_t>(n) * d * sizeof(float),
                    cudaMemcpyDeviceToDevice, stream);

    // ── ONE-TIME SETUP: download Z + batch_labels for k-means++ init ──────────
    // This is the ONLY allowed host download — one-time at function entry.
    cudaStreamSynchronize(stream);
    std::vector<float> h_Z(static_cast<size_t>(n) * d);
    std::vector<int>   h_batch(n);
    cudaMemcpy(h_Z.data(), Z.get(),
               static_cast<size_t>(n) * d * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_batch.data(), batch_labels.get(),
               static_cast<size_t>(n) * sizeof(int), cudaMemcpyDeviceToHost);

    // Build batch-stable sorted cell indices (fixed across iterations).
    std::vector<int> h_sorted(n);
    std::iota(h_sorted.begin(), h_sorted.end(), 0);
    std::stable_sort(h_sorted.begin(), h_sorted.end(),
                     [&](int a, int b){ return h_batch[a] < h_batch[b]; });

    // Compute batch_offsets[B+1]: batch_offsets[b] = first sorted position of batch b.
    std::vector<int> h_batch_offsets(n_batches + 1, 0);
    for (int i = 0; i < n; ++i)
        ++h_batch_offsets[h_batch[h_sorted[i]] + 1];
    for (int b = 0; b < n_batches; ++b)
        h_batch_offsets[b + 1] += h_batch_offsets[b];

    // Scaled offsets: batch_offsets * d (for flat n×d segmented reduce).
    std::vector<int> h_scaled_offsets(n_batches + 1);
    for (int b = 0; b <= n_batches; ++b)
        h_scaled_offsets[b] = h_batch_offsets[b] * d;

    // Upload sorted indices + both offset arrays once.
    cudaMemcpyAsync(sorted_indices_dev.get(), h_sorted.data(),
                    static_cast<size_t>(n) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(batch_offsets_dev.get(), h_batch_offsets.data(),
                    static_cast<size_t>(n_batches + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(scaled_offsets_dev.get(), h_scaled_offsets.data(),
                    static_cast<size_t>(n_batches + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Pre-query cub workspace sizes (done ONCE; buffers allocated ONCE).
    size_t seg_reduce_tmp_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(nullptr, seg_reduce_tmp_bytes,
                                    (const float*)nullptr, (float*)nullptr,
                                    n_batches,
                                    batch_offsets_dev.get(), batch_offsets_dev.get() + 1,
                                    stream);
    core::DeviceMemory<uint8_t> seg_reduce_tmp(seg_reduce_tmp_bytes + 1);

    size_t scalar_reduce_tmp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, scalar_reduce_tmp_bytes,
                           (const float*)nullptr, (float*)nullptr, n, stream);
    core::DeviceMemory<uint8_t> scalar_reduce_tmp(scalar_reduce_tmp_bytes + 1);

    size_t max_reduce_tmp_bytes = 0;
    cub::DeviceReduce::Max(nullptr, max_reduce_tmp_bytes,
                           (const float*)nullptr, (float*)nullptr, n, stream);
    core::DeviceMemory<uint8_t> max_reduce_tmp(max_reduce_tmp_bytes + 1);

    // ── K-means++ initialization (seeded, uses h_Z downloaded above) ─────────
    std::mt19937_64 rng(cfg.seed);
    std::vector<int> center_idx(K);
    std::uniform_int_distribution<int> pick(0, n - 1);
    center_idx[0] = pick(rng);
    std::vector<float> min_sq_dist(n, std::numeric_limits<float>::max());
    for (int ci = 1; ci < K; ++ci) {
        int prev = center_idx[ci - 1];
        for (int j = 0; j < n; ++j) {
            float dist = 0.f;
            for (int dim = 0; dim < d; ++dim) {
                float diff = h_Z[(size_t)j * d + dim] - h_Z[(size_t)prev * d + dim];
                dist += diff * diff;
            }
            min_sq_dist[j] = std::min(min_sq_dist[j], dist);
        }
        std::vector<float> cumsum(n);
        cumsum[0] = min_sq_dist[0];
        for (int j = 1; j < n; ++j) cumsum[j] = cumsum[j-1] + min_sq_dist[j];
        float total = cumsum[n-1];
        if (total <= 0.f) { center_idx[ci] = pick(rng); continue; }
        std::uniform_real_distribution<float> udist(0.f, total);
        float samp = udist(rng);
        int idx = (int)(std::lower_bound(cumsum.begin(), cumsum.end(), samp) - cumsum.begin());
        center_idx[ci] = std::min(idx, n - 1);
    }

    // Upload initial centroids_global from h_Z seed rows.
    std::vector<float> h_cent_global(static_cast<size_t>(K) * d);
    for (int k = 0; k < K; ++k)
        for (int dim = 0; dim < d; ++dim)
            h_cent_global[(size_t)k * d + dim] = h_Z[(size_t)center_idx[k] * d + dim];
    cudaMemcpyAsync(centroids_global.get(), h_cent_global.data(),
                    static_cast<size_t>(K) * d * sizeof(float), cudaMemcpyHostToDevice, stream);

    // h_Z no longer needed after this point.
    h_Z.clear(); h_Z.shrink_to_fit();

    // ── Soft-assignment update helper ─────────────────────────────────────────
    // D = -2*Z*C^T + norms → stable softmax rows → R.
    // All on device; no host transfer.
    auto update_soft_assignments = [&](const float* C_ptr) {
        detail::row_norms_kernel<<<n, 256, 256*sizeof(float), stream>>>(
            Z.get(), z_norms.get(), n, d);
        detail::row_norms_kernel<<<K, 256, 256*sizeof(float), stream>>>(
            C_ptr, c_norms.get(), K, d);
        const float alpha = -2.f, beta = 0.f;
        cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N,
                    n, K, d, &alpha,
                    Z.get(), d, C_ptr, d, &beta,
                    D.get(), n);
        { dim3 b(256), g((K + 255)/256, n);
          detail::add_centroid_norms_kernel<<<g, b, 0, stream>>>(
              D.get(), z_norms.get(), c_norms.get(), n, K); }
        size_t shmem = (static_cast<size_t>(K) + 256) * sizeof(float);
        detail::softmax_rows_kernel<<<n, 256, shmem, stream>>>(
            D.get(), n, K, cfg.lambda);
        cudaMemcpyAsync(R.get(), D.get(),
                        static_cast<size_t>(n) * K * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);
    };

    update_soft_assignments(centroids_global.get());

    // ── Iteration loop — ZERO host transfers inside (CYCLE-14 guarantee) ─────
    // Self-check: only cudaMemcpy allowed inside this loop is the 4-byte
    // convergence scalar at the bottom.  All centroid math is via
    // cub::DeviceSegmentedReduce::Sum on pre-built sorted index + offset arrays.

    int n_iters = 0;
    float final_delta = 0.f;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {

        // Save R_old (D2D — no PCIe traffic).
        cudaMemcpyAsync(R_old.get(), R.get(),
                        static_cast<size_t>(n) * K * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream);

        // ── Per-cluster centroid accumulation via DeviceSegmentedReduce ──────
        // For each cluster k:
        //   1. build_weighted_z_sorted_kernel → weighted_sorted[n×d], denom_sorted[n]
        //      (both in batch-stable sorted cell order)
        //   2. DeviceSegmentedReduce::Sum on weighted_sorted[n*d] flat, B segments
        //      each of size batch_size * d (using scaled_offsets_dev) → seg_numer[B×d]
        //   3. DeviceSegmentedReduce::Sum on denom_sorted[n], B segments
        //      (using batch_offsets_dev) → seg_denom[B]
        //   4. scatter → centroids_kb[k, :, :] and denom_kb[k, :]
        //   5. DeviceReduce::Sum on R column k → denom_global_dev[k]
        // All on device; no host round-trip.
        for (int k = 0; k < K; ++k) {
            // Step 1.
            { int threads = 256, blocks = (n + threads - 1) / threads;
              detail::build_weighted_z_sorted_kernel<<<blocks, threads, 0, stream>>>(
                  weighted_sorted.get(), denom_sorted.get(),
                  R.get(), Z.get(), sorted_indices_dev.get(),
                  n, K, d, k); }

            // Step 2: flat n*d → B*d numerator (segments scale by d).
            cub::DeviceSegmentedReduce::Sum(
                seg_reduce_tmp.get(), seg_reduce_tmp_bytes,
                weighted_sorted.get(),          // flat n*d input
                seg_numer.get(),                // B*d output
                n_batches,
                scaled_offsets_dev.get(),       // segment starts (= batch_offsets * d)
                scaled_offsets_dev.get() + 1,   // segment ends
                stream);

            // Step 3: n → B denominator.
            cub::DeviceSegmentedReduce::Sum(
                seg_reduce_tmp.get(), seg_reduce_tmp_bytes,
                denom_sorted.get(),
                seg_denom.get(),
                n_batches,
                batch_offsets_dev.get(),
                batch_offsets_dev.get() + 1,
                stream);

            // Step 4: scatter into centroids_kb and denom_kb.
            { int threads = 256, blocks = (n_batches * d + threads - 1) / threads;
              detail::scatter_cent_kb_kernel<<<blocks, threads, 0, stream>>>(
                  centroids_kb.get(), seg_numer.get(), k, n_batches, d); }
            { int threads = 256, blocks = (n_batches + threads - 1) / threads;
              detail::scatter_denom_kb_kernel<<<blocks, threads, 0, stream>>>(
                  denom_kb.get(), seg_denom.get(), k, n_batches); }

            // Step 5: global weight sum for cluster k (writes to denom_global_dev[k]).
            { int threads = 256, blocks = (n + threads - 1) / threads;
              detail::extract_r_col_kernel<<<blocks, threads, 0, stream>>>(
                  R.get(), r_col_scratch.get(), n, K, k); }
            cub::DeviceReduce::Sum(
                scalar_reduce_tmp.get(), scalar_reduce_tmp_bytes,
                r_col_scratch.get(),
                denom_global_dev.get() + k,  // direct write to denom_global_dev[k]
                n, stream);
        }

        // Normalize centroids_kb in-place: cent_kb[k,b,:] /= denom_kb[k,b] + lambda.
        { int threads = 256, blocks = (K * n_batches + threads - 1) / threads;
          detail::normalize_cent_kb_kernel<<<blocks, threads, 0, stream>>>(
              centroids_kb.get(), denom_kb.get(), K, n_batches, d, cfg.lambda); }

        // Global centroids: centroids_global = R^T * Z, then normalize.
        // Wt = R^T (K×n row-major for cuBLAS col-major convention).
        { dim3 b(256), g((n + 255)/256, K);
          detail::transpose_r_kernel<<<g, b, 0, stream>>>(
              R.get(), Wt.get(), n, K); }
        { const float a2 = 1.f, b2 = 0.f;
          cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_T,
                      d, K, n, &a2,
                      Z.get(), d, Wt.get(), K, &b2,
                      centroids_global.get(), d); }
        detail::normalize_cent_global_kernel<<<K, 256, 0, stream>>>(
            centroids_global.get(), denom_global_dev.get(), K, d, cfg.lambda);

        // Apply correction.
        detail::apply_correction_kernel<<<n, 256, 0, stream>>>(
            Z.get(), R.get(), centroids_kb.get(), centroids_global.get(),
            batch_labels.get(), n, K, n_batches, d);

        // Update soft assignments.
        update_soft_assignments(centroids_global.get());

        // Convergence check: max |R_new - R_old|.
        { int b = 256, g = (n + b - 1) / b;
          detail::r_delta_kernel<<<g, b, 0, stream>>>(
              R.get(), R_old.get(), delta_per_cell.get(), n, K); }
        cub::DeviceReduce::Max(max_reduce_tmp.get(), max_reduce_tmp_bytes,
                               delta_per_cell.get(), delta_max.get(), n, stream);

        // ↓ ONLY allowed per-iteration host transfer: 4-byte scalar.
        cudaStreamSynchronize(stream);
        cudaMemcpy(&final_delta, delta_max.get(), sizeof(float), cudaMemcpyDeviceToHost);

        ++n_iters;
        if (final_delta < cfg.tol) break;
    }

    cublasDestroy(blas);

    HarmonyResult result;
    result.corrected    = std::move(Z);
    result.n_iters_used = n_iters;
    result.final_delta  = final_delta;
    return result;
}

}  // namespace integrate
}  // namespace singlet::gpu
