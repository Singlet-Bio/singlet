// SPDX-License-Identifier: MIT
// integrates: CellTypist (Dominguez Conde et al. 2022)
// anno/celltypist.h — GPU-native cell-type annotation via pre-trained logistic
// regression models (Human Cell Atlas reference).
//
// Dominguez Conde C, Xu C, Jarvis LB, et al. (2022)
// "Cross-tissue immune cell analysis reveals tissue-specific features in humans."
// Science 376:eabl5197.
//
// Algorithm (3 sequential GPU passes, no D2H in hot path):
//   Pass 1 — cuBLAS Sgemm:  L = W^T . Z  (n_classes x n_cells, col-major)
//   Pass 2 — bias-add:       L[k, c] += b[k]  for all (k, c)
//   Pass 3 — softmax+argmax: one block per cell; warp-shuffle reductions.
//             No full P matrix stored — only pred_class[c] and confidence[c].
//             Strided iteration for n_classes > 256.
//
// Numerical stability: log-sum-exp trick — subtract L_max[c] before exp.
//
// Rules: 5 (DeviceMemory), 4 (no inner-loop D2H), 18 (deterministic — cuBLAS
//        Sgemm + warp-shuffle, no atomics), 31 (< 350 LOC).
// CYCLE-135: initial implementation.

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

struct CelltypistConfig {
    bool use_log_input  = true;   // assert input is log1p-transformed (CellTypist convention)
    bool deterministic  = true;   // cuBLAS Sgemm + warp-shuffle reductions are deterministic
};

struct CelltypistResult {
    core::DeviceMemory<int>   pred_class;  // n_cells; int in [0, n_classes)
    core::DeviceMemory<float> confidence;  // n_cells; max softmax prob in [0, 1]
    int n_cells;
    int n_classes;
};

// ---------------------------------------------------------------------------
// Internal kernels
// ---------------------------------------------------------------------------
namespace detail {

static constexpr int CT_THREADS = 256;

// Pass 2: bias-add kernel.
// L has layout (n_classes x n_cells) col-major; element (k, c) = L[k + c*n_classes].
// One thread per element; b is broadcast across all cells.
__global__ __launch_bounds__(256, 4)
void ct_bias_add_kernel(float* __restrict__ L,
                        const float* __restrict__ b,
                        int n_classes, int n_cells)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_classes * n_cells;
    if (idx >= total) return;
    const int k = idx % n_classes;   // class index
    L[idx] += b[k];
}

// Pass 3: softmax + argmax + confidence — one block per cell.
//
// Each block handles cell c = blockIdx.x.
// Column c of L is L[c * n_classes .. c * n_classes + n_classes - 1].
//
// For n_classes <= CT_THREADS: one-to-one thread-to-class assignment.
// For n_classes > CT_THREADS:  each thread strides over multiple classes.
//
// Reductions use warp shuffles + shared memory (standard pattern).
// Output: pred_class[c] = argmax_k, confidence[c] = softmax(L[argmax,c]).
//
// Every kernel parameter explicitly named (no globals).
__global__ __launch_bounds__(256, 2)
void ct_softmax_argmax_kernel(
    const float* __restrict__ L,        // n_classes x n_cells col-major
    int*         __restrict__ pred_class,
    float*       __restrict__ confidence,
    int n_classes, int n_cells)
{
    const int cell = static_cast<int>(blockIdx.x);
    if (cell >= n_cells) return;

    const int tid   = static_cast<int>(threadIdx.x);
    const int ntid  = static_cast<int>(blockDim.x);
    const float* Lc = L + static_cast<ptrdiff_t>(cell) * n_classes;

    // --- Reduction 1: find L_max[c] and argmax ---
    float thread_max = -3.402823e+38f;  // -FLT_MAX without <cfloat>
    int   thread_argmax = tid;

    for (int k = tid; k < n_classes; k += ntid) {
        float v = Lc[k];
        if (v > thread_max) { thread_max = v; thread_argmax = k; }
    }

    // Warp-level max (value only — argmax tracked separately).
    for (int off = 16; off > 0; off >>= 1) {
        float v = __shfl_down_sync(0xffffffffu, thread_max, off);
        int   a = __shfl_down_sync(0xffffffffu, thread_argmax, off);
        if (v > thread_max) { thread_max = v; thread_argmax = a; }
    }

    // Cross-warp reduce: shared memory for (max, argmax) per warp.
    // Layout: smem_f[0..nwarps-1] = per-warp max; smem_i[0..nwarps-1] = per-warp argmax.
    __shared__ float smem_f[8];   // max 256/32 = 8 warps
    __shared__ int   smem_i[8];
    const int wid  = tid / 32;
    const int lane = tid & 31;
    const int nwarp = (ntid + 31) / 32;

    if (lane == 0) { smem_f[wid] = thread_max; smem_i[wid] = thread_argmax; }
    __syncthreads();

    float L_max   = -3.402823e+38f;
    int   argmax_k = 0;
    if (tid < nwarp) {
        L_max    = smem_f[tid];
        argmax_k = smem_i[tid];
    }
    if (wid == 0) {
        for (int off = 4; off > 0; off >>= 1) {
            float v = __shfl_down_sync(0xffffffffu, L_max,    off);
            int   a = __shfl_down_sync(0xffffffffu, argmax_k, off);
            if (v > L_max) { L_max = v; argmax_k = a; }
        }
    }
    __syncthreads();
    // Broadcast L_max and argmax_k from thread 0 of warp 0.
    if (tid == 0) { smem_f[0] = L_max; smem_i[0] = argmax_k; }
    __syncthreads();
    L_max    = smem_f[0];
    argmax_k = smem_i[0];

    // --- Reduction 2: sum of exp(L - L_max) for denominator ---
    float thread_sum = 0.f;
    for (int k = tid; k < n_classes; k += ntid) {
        thread_sum += expf(Lc[k] - L_max);
    }

    for (int off = 16; off > 0; off >>= 1)
        thread_sum += __shfl_down_sync(0xffffffffu, thread_sum, off);

    __shared__ float smem_sum[8];
    if (lane == 0) smem_sum[wid] = thread_sum;
    __syncthreads();

    float sum_exp = 0.f;
    if (tid < nwarp) sum_exp = smem_sum[tid];
    if (wid == 0) {
        for (int off = 4; off > 0; off >>= 1)
            sum_exp += __shfl_down_sync(0xffffffffu, sum_exp, off);
    }
    __syncthreads();
    if (tid == 0) smem_sum[0] = sum_exp;
    __syncthreads();
    sum_exp = smem_sum[0];

    // --- Write outputs (thread 0 only) ---
    if (tid == 0) {
        pred_class[cell]  = argmax_k;
        const float guard = (sum_exp > 0.f) ? sum_exp : 1.f;
        confidence[cell]  = expf(Lc[argmax_k] - L_max) / guard;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// celltypist_predict — public entry point
//
// Inputs (all device pointers):
//   d_Z        — n_features x n_cells dense col-major fp32
//                (log1p-normalized + scaled expression; same features as model)
//   n_features — number of features (rows of Z and rows of W)
//   n_cells    — number of query cells (columns of Z)
//   d_W        — n_features x n_classes dense col-major fp32 (model weights)
//   d_b        — n_classes fp32 (bias vector)
//   n_classes  — number of cell-type classes
//   cfg        — CelltypistConfig (use_log_input, deterministic flags)
//   stream     — CUDA stream (nullptr = default context stream)
//
// Returns CelltypistResult with pred_class and confidence on device.
// Caller synchronizes stream before reading.
// ---------------------------------------------------------------------------
inline CelltypistResult celltypist_predict(
    const float* d_Z, int n_features, int n_cells,
    const float* d_W, const float* d_b, int n_classes,
    const CelltypistConfig& cfg    = {},
    cudaStream_t            stream = nullptr)
{
    if (stream == nullptr)
        stream = singlet::gpu::core::default_context().stream();

    if (n_features <= 0 || n_cells <= 0 || n_classes <= 0)
        throw std::runtime_error(
            "celltypist_predict: invalid dimensions (n_features=" +
            std::to_string(n_features) + " n_cells=" + std::to_string(n_cells) +
            " n_classes=" + std::to_string(n_classes) + ")");
    if (!d_Z || !d_W || !d_b)
        throw std::runtime_error("celltypist_predict: null input pointer");

    // --- Allocate outputs and logit buffer ---
    CelltypistResult result;
    result.n_cells   = n_cells;
    result.n_classes = n_classes;
    result.pred_class  = core::DeviceMemory<int>(static_cast<size_t>(n_cells));
    result.confidence  = core::DeviceMemory<float>(static_cast<size_t>(n_cells));

    const size_t L_sz = static_cast<size_t>(n_classes) * static_cast<size_t>(n_cells);
    core::DeviceMemory<float> d_L(L_sz);

    // --- Pass 1: cuBLAS Sgemm  L = W^T . Z ---
    // W is (n_features x n_classes) col-major; W^T is (n_classes x n_features).
    // Z is (n_features x n_cells) col-major.
    // Output L is (n_classes x n_cells) col-major.
    // cublasSgemm(opA=T, opB=N, m=n_classes, n=n_cells, k=n_features)
    {
        cublasHandle_t blas = singlet::gpu::core::default_context().blas();
        cublasSetStream(blas, stream);
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n_classes,   // m
            n_cells,     // n
            n_features,  // k
            &alpha,
            d_W, n_features,  // A = W (n_features x n_classes), ldA = n_features
            d_Z, n_features,  // B = Z (n_features x n_cells),   ldB = n_features
            &beta,
            d_L.get(), n_classes));  // C = L (n_classes x n_cells), ldC = n_classes
    }

    // --- Pass 2: bias-add  L[k, c] += b[k] ---
    {
        const int total = n_classes * n_cells;
        const int grid  = (total + detail::CT_THREADS - 1) / detail::CT_THREADS;
        detail::ct_bias_add_kernel<<<grid, detail::CT_THREADS, 0, stream>>>(
            d_L.get(), d_b, n_classes, n_cells);
    }

    // --- Pass 3: softmax + argmax + confidence ---
    // One block per cell; threads = min(CT_THREADS, round_up_32(n_classes)).
    {
        const int threads = (n_classes <= detail::CT_THREADS)
            ? (((n_classes + 31) / 32) * 32)   // round up to next warp
            : detail::CT_THREADS;
        detail::ct_softmax_argmax_kernel<<<n_cells, threads, 0, stream>>>(
            d_L.get(), result.pred_class.get(), result.confidence.get(),
            n_classes, n_cells);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
}

}  // namespace anno
}  // namespace singlet::gpu
