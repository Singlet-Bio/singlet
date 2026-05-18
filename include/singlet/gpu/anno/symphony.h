// SPDX-License-Identifier: MIT
// integrates: Symphony (Kang et al. 2021)
// anno/symphony.h — GPU-native reference-based cell-type mapping via
// PCA projection + soft cluster assignment + label transfer.
//
// Kang JB, Nathan A, Weinand K, et al. (2021)
// "Efficient and precise single-cell reference atlas mapping with Symphony."
// Nat Commun 12:5890.
// Reference R package: https://github.com/immunogenomics/symphony
//
// Algorithm (5 sequential GPU passes, no D2H in hot path):
//   Pass 1 — Standardize:  Z_std[g,c] = (Z[g,c] - mu[g]) / max(sigma[g], eps_sigma)
//             One element-wise kernel (n_features × n_query_cells threads).
//   Pass 2 — PCA project:  X_pca = W_pca^T · Z_std  (cuBLAS Sgemm)
//             Output X_pca is (k_pcs × n_query_cells, col-major).
//   Pass 3 — Distance:
//             3a. Per-column ||X_pca||²[c] kernel (1 thread per cell, warp reduce).
//             3b. Per-row ||C_ref||²[k] (same kernel, precomputed once per call).
//             3c. cuBLAS Sgemm: cross = X_pca^T · C_ref^T  → (n_query × k_clusters).
//             3d. Combine: d[k,c] = ||C||²[k] + ||X||²[c] - 2·cross[c,k].
//   Pass 4 — Soft assign: s[k,c] = 1/max(d[k,c], eps_dist); row-normalize per cell
//             via warp-shuffle reduction (one block per cell).
//   Pass 5 — Label transfer: label_prob = P_label^T · s_soft  (cuBLAS Sgemm)
//             then argmax + confidence kernel (one block per cell, warp-shuffle).
//
// Numerical stability: eps_sigma guards σ≈0; eps_dist guards d≈0 in 1/d.
//
// Rules: 5 (DeviceMemory), 4 (no inner-loop D2H), 18 (deterministic — cuBLAS
//        Sgemm + warp-shuffle, no atomics), 31 (< 400 LOC).
// CYCLE-138: initial implementation.

#pragma once
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace anno {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct SymphonyConfig {
    float eps_sigma  = 1e-3f;  // guard for sigma_ref close to 0
    float eps_dist   = 1e-6f;  // guard for d[k,c] close to 0 in 1/d weighting
    bool  deterministic = true; // cuBLAS Sgemm + warp-shuffle reductions are deterministic
};

struct SymphonyResult {
    core::DeviceMemory<int>   pred_class;   // n_query_cells; int in [0, n_classes)
    core::DeviceMemory<float> confidence;   // n_query_cells; max label probability
    int n_query_cells;
    int n_classes;
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

static constexpr int SY_THREADS = 256;

// Pass 1: standardize Z.
// Z_std[g + c*n_features] = (Z[g + c*n_features] - mu[g]) / max(sigma[g], eps_sigma)
// One thread per element.
__global__ __launch_bounds__(256, 4)
void sy_standardize_kernel(
    const float* __restrict__ Z,       // n_features x n_cells col-major
    float*       __restrict__ Z_std,   // n_features x n_cells col-major (output)
    const float* __restrict__ mu,      // n_features
    const float* __restrict__ sigma,   // n_features
    float eps_sigma,
    int n_features, int n_cells)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_features * n_cells;
    if (idx >= total) return;
    const int g = idx % n_features;
    const float s = sigma[g] < eps_sigma ? eps_sigma : sigma[g];
    Z_std[idx] = (Z[idx] - mu[g]) / s;
}

// Pass 3a: per-column sum of squares for X (k_pcs x n_cells col-major).
// sq[c] = sum_p X[p + c*k_pcs]^2.  One block per column; warp-shuffle reduction.
__global__ __launch_bounds__(256, 2)
void sy_col_sumsq_kernel(
    const float* __restrict__ X,   // k_dim x n_cols col-major
    float*       __restrict__ sq,  // n_cols output
    int k_dim, int n_cols)
{
    const int col = static_cast<int>(blockIdx.x);
    if (col >= n_cols) return;
    const int tid   = static_cast<int>(threadIdx.x);
    const int ntid  = static_cast<int>(blockDim.x);
    const float* Xc = X + static_cast<ptrdiff_t>(col) * k_dim;

    float acc = 0.f;
    for (int p = tid; p < k_dim; p += ntid)
        acc += Xc[p] * Xc[p];

    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);

    __shared__ float smem[8];
    const int wid  = tid / 32;
    const int lane = tid & 31;
    const int nwarp = (ntid + 31) / 32;
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    float total = 0.f;
    if (tid < nwarp) total = smem[tid];
    if (wid == 0) {
        for (int off = 4; off > 0; off >>= 1)
            total += __shfl_down_sync(0xffffffffu, total, off);
    }
    if (tid == 0) sq[col] = total;
}

// Pass 3b: per-row sum of squares for C_ref (k_clusters x k_pcs col-major).
// C_sq[k] = sum_p C_ref[k + p*k_clusters]^2.
// One thread per row (cluster); loops over PCs.  Deterministic — no atomics.
__global__ __launch_bounds__(256, 4)
void sy_row_sumsq_kernel(
    const float* __restrict__ C,    // k_clusters x k_pcs col-major: C[k + p*k_clusters]
    float*       __restrict__ sq,   // k_clusters output
    int k_clusters, int k_pcs)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= k_clusters) return;
    float acc = 0.f;
    for (int p = 0; p < k_pcs; ++p) {
        float v = C[k + static_cast<ptrdiff_t>(p) * k_clusters];
        acc += v * v;
    }
    sq[k] = acc;
}

// Pass 3d: combine distances.
// d[k + c*k_clusters] = C_sq[k] + X_sq[c] - 2 * cross[c + k*n_query] (cross is col-major).
// cross from Sgemm (X_pca^T · C_ref^T): shape (n_query x k_clusters) col-major
//   element (c, k) stored at cross[c + k*n_query].
__global__ __launch_bounds__(256, 4)
void sy_dist_combine_kernel(
    const float* __restrict__ cross,    // n_query x k_clusters col-major: cross[c + k*n_query]
    const float* __restrict__ C_sq,     // k_clusters
    const float* __restrict__ X_sq,     // n_query
    float*       __restrict__ dist,     // k_clusters x n_query col-major: dist[k + c*k_clusters]
    int k_clusters, int n_query)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = k_clusters * n_query;
    if (idx >= total) return;
    const int k = idx % k_clusters;
    const int c = idx / k_clusters;
    // cross[c, k] in (n_query x k_clusters) col-major = cross[c + k*n_query]
    const float d = C_sq[k] + X_sq[c] - 2.f * cross[c + static_cast<ptrdiff_t>(k) * n_query];
    dist[idx] = d > 0.f ? d : 0.f;  // clamp to [0, ∞) for numerical safety
}

// Pass 4: soft assignment + per-cell L1 normalize.
// s[k + c*k_clusters] = 1 / max(dist[k + c*k_clusters], eps_dist), then divide by sum.
// One block per cell; warp-shuffle reduction for sum.
__global__ __launch_bounds__(256, 2)
void sy_soft_assign_kernel(
    const float* __restrict__ dist,    // k_clusters x n_query col-major
    float*       __restrict__ s,       // k_clusters x n_query col-major (output)
    float eps_dist,
    int k_clusters, int n_query)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_query) return;
    const int tid  = static_cast<int>(threadIdx.x);
    const int ntid = static_cast<int>(blockDim.x);

    const float* dc = dist + static_cast<ptrdiff_t>(cell) * k_clusters;
    float*       sc = s    + static_cast<ptrdiff_t>(cell) * k_clusters;

    // Compute inverse distances and partial sum.
    float thread_sum = 0.f;
    for (int k = tid; k < k_clusters; k += ntid) {
        float inv = 1.f / (dc[k] > eps_dist ? dc[k] : eps_dist);
        sc[k] = inv;
        thread_sum += inv;
    }

    // Reduce sum across block via warp shuffles + shared memory.
    for (int off = 16; off > 0; off >>= 1)
        thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, off);

    __shared__ float smem[8];
    const int wid  = tid / 32;
    const int lane = tid & 31;
    const int nwarp = (ntid + 31) / 32;
    if (lane == 0) smem[wid] = thread_sum;
    __syncthreads();

    float sum_inv = 0.f;
    if (tid < nwarp) sum_inv = smem[tid];
    if (wid == 0) {
        for (int off = 4; off > 0; off >>= 1)
            sum_inv += __shfl_down_sync(0xffffffffu, sum_inv, off);
    }
    __syncthreads();
    if (tid == 0) smem[0] = sum_inv;
    __syncthreads();
    sum_inv = smem[0];

    // Normalize; guard against degenerate all-zero case.
    const float norm = (sum_inv > 0.f) ? sum_inv : 1.f;
    for (int k = tid; k < k_clusters; k += ntid)
        sc[k] /= norm;
}

// Pass 5b: argmax + confidence on label_prob (n_classes x n_query col-major).
// Mirror ct_softmax_argmax_kernel pattern.
__global__ __launch_bounds__(256, 2)
void sy_argmax_conf_kernel(
    const float* __restrict__ label_prob,  // n_classes x n_query col-major
    int*         __restrict__ pred_class,
    float*       __restrict__ confidence,
    int n_classes, int n_query)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_query) return;

    const int tid   = static_cast<int>(threadIdx.x);
    const int ntid  = static_cast<int>(blockDim.x);
    const float* Pc = label_prob + static_cast<ptrdiff_t>(cell) * n_classes;

    float thread_max = -3.402823e+38f;
    int   thread_arg = tid;

    for (int k = tid; k < n_classes; k += ntid) {
        float v = Pc[k];
        if (v > thread_max) { thread_max = v; thread_arg = k; }
    }

    for (int off = 16; off > 0; off >>= 1) {
        float v = __shfl_down_sync(0xffffffffu, thread_max, off);
        int   a = __shfl_down_sync(0xffffffffu, thread_arg, off);
        if (v > thread_max) { thread_max = v; thread_arg = a; }
    }

    __shared__ float smem_f[8];
    __shared__ int   smem_i[8];
    const int wid  = tid / 32;
    const int lane = tid & 31;
    const int nwarp = (ntid + 31) / 32;

    if (lane == 0) { smem_f[wid] = thread_max; smem_i[wid] = thread_arg; }
    __syncthreads();

    float best_val = -3.402823e+38f;
    int   best_cls = 0;
    if (tid < nwarp) { best_val = smem_f[tid]; best_cls = smem_i[tid]; }
    if (wid == 0) {
        for (int off = 4; off > 0; off >>= 1) {
            float v = __shfl_down_sync(0xffffffffu, best_val, off);
            int   a = __shfl_down_sync(0xffffffffu, best_cls, off);
            if (v > best_val) { best_val = v; best_cls = a; }
        }
    }
    __syncthreads();
    if (tid == 0) { smem_f[0] = best_val; smem_i[0] = best_cls; }
    __syncthreads();
    best_val = smem_f[0];
    best_cls = smem_i[0];

    if (tid == 0) {
        pred_class[cell] = best_cls;
        confidence[cell] = (best_val > 0.f) ? best_val : 0.f;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// symphony_predict — public entry point
//
// Inputs (all device pointers):
//   d_Z_query   — n_features x n_query_cells dense col-major fp32 (log1p)
//   d_W_pca     — n_features x k_pcs dense col-major fp32 (PCA loadings)
//   d_C_ref     — k_clusters x k_pcs dense col-major fp32 (reference centroids)
//   d_P_label   — k_clusters x n_classes dense col-major fp32 (label dist; rows sum to 1)
//   d_mu_ref    — n_features fp32 (reference per-gene mean)
//   d_sigma_ref — n_features fp32 (reference per-gene std)
//   cfg         — SymphonyConfig (eps guards, deterministic flag)
//   stream      — CUDA stream (nullptr = default context stream)
//
// Returns SymphonyResult with pred_class and confidence on device.
// Caller synchronizes stream before reading.
// ---------------------------------------------------------------------------
inline SymphonyResult symphony_predict(
    const float* d_Z_query, int n_features, int n_query_cells,
    const float* d_W_pca, int k_pcs,
    const float* d_C_ref, int k_clusters,
    const float* d_P_label, int n_classes,
    const float* d_mu_ref, const float* d_sigma_ref,
    const SymphonyConfig& cfg    = {},
    cudaStream_t          stream = nullptr)
{
    if (stream == nullptr)
        stream = singlet::gpu::core::default_context().stream();

    if (n_features <= 0 || n_query_cells <= 0 || k_pcs <= 0 ||
        k_clusters <= 0 || n_classes <= 0)
        throw std::runtime_error(
            "symphony_predict: invalid dimensions (n_features=" +
            std::to_string(n_features) + " n_query=" + std::to_string(n_query_cells) +
            " k_pcs=" + std::to_string(k_pcs) +
            " k_clusters=" + std::to_string(k_clusters) +
            " n_classes=" + std::to_string(n_classes) + ")");
    if (!d_Z_query || !d_W_pca || !d_C_ref || !d_P_label || !d_mu_ref || !d_sigma_ref)
        throw std::runtime_error("symphony_predict: null input pointer");

    cublasHandle_t blas = singlet::gpu::core::default_context().blas();
    cublasSetStream(blas, stream);

    // --- Allocate intermediate and output buffers ---
    const size_t FQ  = static_cast<size_t>(n_features)     * n_query_cells;
    const size_t PQ  = static_cast<size_t>(k_pcs)          * n_query_cells;
    const size_t KQ  = static_cast<size_t>(k_clusters)     * n_query_cells;
    const size_t CQ  = static_cast<size_t>(n_classes)      * n_query_cells;
    const size_t QK  = static_cast<size_t>(n_query_cells)  * k_clusters;  // cross term

    core::DeviceMemory<float> d_Z_std(FQ);
    core::DeviceMemory<float> d_X_pca(PQ);
    core::DeviceMemory<float> d_X_sq(static_cast<size_t>(n_query_cells));
    core::DeviceMemory<float> d_C_sq(static_cast<size_t>(k_clusters));
    core::DeviceMemory<float> d_cross(QK);   // (n_query x k_clusters) col-major
    core::DeviceMemory<float> d_dist(KQ);    // (k_clusters x n_query) col-major
    core::DeviceMemory<float> d_soft(KQ);    // (k_clusters x n_query) col-major
    core::DeviceMemory<float> d_label_prob(CQ);  // (n_classes x n_query) col-major

    SymphonyResult result;
    result.n_query_cells = n_query_cells;
    result.n_classes     = n_classes;
    result.pred_class    = core::DeviceMemory<int>(static_cast<size_t>(n_query_cells));
    result.confidence    = core::DeviceMemory<float>(static_cast<size_t>(n_query_cells));

    // --- Pass 1: standardize query ---
    {
        const int total = n_features * n_query_cells;
        const int grid  = (total + detail::SY_THREADS - 1) / detail::SY_THREADS;
        detail::sy_standardize_kernel<<<grid, detail::SY_THREADS, 0, stream>>>(
            d_Z_query, d_Z_std.get(), d_mu_ref, d_sigma_ref,
            cfg.eps_sigma, n_features, n_query_cells);
    }

    // --- Pass 2: PCA projection  X_pca = W_pca^T · Z_std ---
    // W_pca (n_features x k_pcs) col-major; W_pca^T is (k_pcs x n_features).
    // Z_std (n_features x n_query) col-major.
    // Output X_pca (k_pcs x n_query) col-major.
    // cublasSgemm(opA=T, opB=N, m=k_pcs, n=n_query, k=n_features)
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            k_pcs,        // m
            n_query_cells, // n
            n_features,   // k
            &alpha,
            d_W_pca, n_features,     // A = W_pca (n_features x k_pcs), ldA = n_features
            d_Z_std.get(), n_features, // B = Z_std (n_features x n_query), ldB = n_features
            &beta,
            d_X_pca.get(), k_pcs));  // C = X_pca (k_pcs x n_query), ldC = k_pcs
    }

    // --- Pass 3: distances ---

    // 3a. Per-column ||X_pca||²[c]  (k_pcs-length columns, n_query_cells cols).
    {
        const int threads = (k_pcs <= detail::SY_THREADS)
            ? (((k_pcs + 31) / 32) * 32) : detail::SY_THREADS;
        detail::sy_col_sumsq_kernel<<<n_query_cells, threads, 0, stream>>>(
            d_X_pca.get(), d_X_sq.get(), k_pcs, n_query_cells);
    }

    // 3b. Per-row ||C_ref||²[k].
    // C_ref (k_clusters x k_pcs) col-major: element (k,p) = C_ref[k + p*k_clusters].
    // Rows are strided; sy_row_sumsq_kernel assigns one thread per row.
    {
        const int grid = (k_clusters + detail::SY_THREADS - 1) / detail::SY_THREADS;
        detail::sy_row_sumsq_kernel<<<grid, detail::SY_THREADS, 0, stream>>>(
            d_C_ref, d_C_sq.get(), k_clusters, k_pcs);
    }

    // 3c. cross = X_pca^T · C_ref^T  → (n_query x k_clusters) col-major.
    // X_pca (k_pcs x n_query) col-major; X_pca^T = (n_query x k_pcs).
    // C_ref (k_clusters x k_pcs) col-major; C_ref^T = (k_pcs x k_clusters).
    // cublasSgemm(opA=T on X_pca, opB=T on C_ref):
    //   m = n_query, n = k_clusters, k = k_pcs
    //   A = X_pca (k_pcs x n_query), ldA = k_pcs
    //   B = C_ref (k_clusters x k_pcs), ldB = k_clusters
    //   C = cross (n_query x k_clusters), ldC = n_query
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_T,
            n_query_cells, // m
            k_clusters,    // n
            k_pcs,         // k
            &alpha,
            d_X_pca.get(), k_pcs,    // A = X_pca (k_pcs x n_query), ldA = k_pcs
            d_C_ref, k_clusters,     // B = C_ref (k_clusters x k_pcs), ldB = k_clusters
            &beta,
            d_cross.get(), n_query_cells)); // C = cross (n_query x k_clusters), ldC = n_query
    }

    // 3d. Combine: d[k, c] = C_sq[k] + X_sq[c] - 2 * cross[c, k].
    {
        const int total = k_clusters * n_query_cells;
        const int grid  = (total + detail::SY_THREADS - 1) / detail::SY_THREADS;
        detail::sy_dist_combine_kernel<<<grid, detail::SY_THREADS, 0, stream>>>(
            d_cross.get(), d_C_sq.get(), d_X_sq.get(),
            d_dist.get(), k_clusters, n_query_cells);
    }

    // --- Pass 4: soft assignment + per-cell normalize ---
    {
        const int threads = (k_clusters <= detail::SY_THREADS)
            ? (((k_clusters + 31) / 32) * 32) : detail::SY_THREADS;
        detail::sy_soft_assign_kernel<<<n_query_cells, threads, 0, stream>>>(
            d_dist.get(), d_soft.get(), cfg.eps_dist, k_clusters, n_query_cells);
    }

    // --- Pass 5a: label transfer  label_prob = P_label^T · s_soft ---
    // P_label (k_clusters x n_classes) col-major; P_label^T = (n_classes x k_clusters).
    // s_soft (k_clusters x n_query) col-major.
    // Output label_prob (n_classes x n_query) col-major.
    // cublasSgemm(opA=T on P_label, opB=N on s_soft):
    //   m = n_classes, n = n_query, k = k_clusters
    //   A = P_label (k_clusters x n_classes), ldA = k_clusters
    //   B = s_soft  (k_clusters x n_query),   ldB = k_clusters
    //   C = label_prob (n_classes x n_query),  ldC = n_classes
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_classes,    // m
            n_query_cells, // n
            k_clusters,   // k
            &alpha,
            d_P_label, k_clusters,      // A = P_label (k_clusters x n_classes), ldA = k_clusters
            d_soft.get(), k_clusters,   // B = s_soft  (k_clusters x n_query),   ldB = k_clusters
            &beta,
            d_label_prob.get(), n_classes)); // C = label_prob (n_classes x n_query), ldC = n_classes
    }

    // --- Pass 5b: argmax + confidence ---
    {
        const int threads = (n_classes <= detail::SY_THREADS)
            ? (((n_classes + 31) / 32) * 32) : detail::SY_THREADS;
        detail::sy_argmax_conf_kernel<<<n_query_cells, threads, 0, stream>>>(
            d_label_prob.get(), result.pred_class.get(), result.confidence.get(),
            n_classes, n_query_cells);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

}  // namespace anno
}  // namespace singlet::gpu
