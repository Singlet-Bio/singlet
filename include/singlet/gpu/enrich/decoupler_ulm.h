// SPDX-License-Identifier: MIT
// integrates: decoupleR (Badia-i-Mompel et al. 2022)
//
// singlet/gpu/enrich/decoupler_ulm.h
//
// ULM (Univariate Linear Model) pathway-scoring method from decoupleR.
//
// Algorithm reference:
//   Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022)
//   "decoupleR: ensemble of computational methods to infer biological
//   activities from omics data." Bioinformatics Advances 2:vbac016.
//   https://doi.org/10.1093/bioadv/vbac016
//
// For each cell c and pathway p, fit a univariate OLS regression:
//
//   X[g, c] = β_0 + β_1 · W[g, p] + ε   (over all genes g)
//
// The OLS slope is:
//
//   β_1[c, p] = cov(X_c, W_p) / var(W_p)
//
// Expanding in terms of matrix products:
//
//   cov(X_c, W_p) = (1/m)·(X^T·W)[c,p] − mean_X[c]·mean_W[p]
//   var(W_p)      = (1/m)·sum_W2[p]     − mean_W[p]²
//
// So:
//
//   score[c, p] = ((XW[c,p]/m) − mean_X[c]·mean_W[p])
//                 / max(var_W[p], ε)
//
// v0 returns β_1 (regression slope).  A future v1 will return the t-statistic
// for null testing: t = β_1 / se(β_1), where se² = MSE / (m·var_W[p]).
//
// Implementation (4 sequential device passes, no inner loop):
//   Pass 1 — mean_X[c]: cuSPARSE SpMV  X · 1_m → col_sum[n], divide by m.
//             (Custom column-sum kernel is equivalent; SpMV reuses the cuSPARSE
//              handle already created for the SpMM in Pass 3, and handles the
//              CSC-transpose transpose correctly via CUSPARSE_OPERATION_TRANSPOSE.
//              However, for implementation simplicity we use a custom two-kernel
//              approach: scatter + divide, identical to score_genes.h Pass 1.)
//   Pass 2 — mean_W[p] and sum_W2[p]: single-pass fused kernel.  One block per
//             pathway; warp-shuffle reduction over m genes accumulates both
//             Σ W[g] and Σ W[g]² simultaneously.  No Welford — two accumulators
//             per thread, one final cross-warp reduce each.  Avoids a second
//             pass over W.
//   Pass 3 — var_W[p] = (sum_W2[p]/m) − mean_W[p]²: one thread per pathway.
//   Pass 4 — XW = X^T · W: cuSPARSE SpMM (same as WSUM/WMEAN).
//   Pass 5 — score kernel: one thread per output element.
//             score[c,p] = ((XW[c,p]/m) − mean_X[c]·mean_W[p]) / max(var_W[p], ε)
//
// Determinism (Rule 18):
//   mean_X Pass 1: atomic scatter (same as score_genes.h; fp32 noise < 1e-5 for
//   log-norm data). mean_W / sum_W2 kernel: warp-shuffle only, no atomics → exact.
//   SpMM: CUSPARSE_SPMM_ALG_DEFAULT on fixed arch is deterministic.
//   Scoring kernel: one thread per element → bit-exact.
//   cfg.deterministic is a no-op (API symmetry with WSUM/WMEAN).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): none in the hot path. Only the nnz==0 early-exit reads nnz on host.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct UlmConfig {
    float epsilon      = 1e-9f;  // guard against division by near-zero var_W
    bool  deterministic = false; // no-op: ULM is already deterministic by construction
};

struct UlmResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
    int n_cells;
    int n_pathways;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// ulm_col_scatter_kernel — accumulate per-cell (column) sums of X.
//
// One thread per stored CSC entry. atomicAdd into col_sum[col_j].
// Identical pattern to sg_row_scatter_kernel in score_genes.h, except we
// scatter by column index (cell) rather than row index (gene), exploiting
// the CSC structure: each stored entry in column j belongs to cell j.
//
// Strategy: iterate over all col_ptr boundaries to identify the cell index
// for each entry.  This is O(nnz) but requires binary search or a precomputed
// cell-index array.  For simplicity and to avoid an extra allocation, we use
// the col_ptr array directly: each entry at position k belongs to column j
// where col_ptr[j] <= k < col_ptr[j+1].  A binary search over n columns is
// O(log n) per entry.  For n ≤ 1M and nnz ≤ 1G this is negligible.
//
// Alternative: precompute a coo_col array (O(nnz) extra memory).  Not worth
// the allocation overhead for this pass.  The binary-search approach is used.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void ulm_col_scatter_kernel(
    const float*   __restrict__ values,
    const int32_t* __restrict__ col_ptr,
    float*         __restrict__ col_sum,
    int n_cols,
    int nnz)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;

    // Binary search: find column j such that col_ptr[j] <= idx < col_ptr[j+1].
    int lo = 0, hi = n_cols - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (col_ptr[mid + 1] <= idx)
            lo = mid + 1;
        else
            hi = mid;
    }
    atomicAdd(&col_sum[lo], values[idx]);
}

// ---------------------------------------------------------------------------
// ulm_col_mean_kernel — divide col_sum by m → mean_X[n].
//
// One thread per cell. Element-wise divide.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void ulm_col_mean_kernel(
    const float* __restrict__ col_sum,
    float*       __restrict__ mean_X,
    int n,
    float m_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    mean_X[i] = col_sum[i] * m_inv;
}

// ---------------------------------------------------------------------------
// ulm_w_stats_kernel — fused per-pathway mean and sum-of-squares over W.
//
// For pathway p (blockIdx.x), computes:
//   mean_W[p]  = (1/m) Σ_g W[g, p]
//   sum_W2[p]  = Σ_g W[g, p]²   (raw sum; division by m happens in Pass 3)
//
// Two accumulators per thread: s1 (sum) and s2 (sum of squares).
// Warp-shuffle reduction for each accumulator independently.
// Cross-warp reduction via shared memory (two rows: smem[0..nwarp-1] = s1,
// smem[8..8+nwarp-1] = s2).
//
// No atomics → deterministic (Rule 18).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void ulm_w_stats_kernel(
    const float* __restrict__ W,         // m × p col-major
    float*       __restrict__ mean_W,    // output: per-pathway mean [p]
    float*       __restrict__ sum_W2,    // output: per-pathway sum-of-squares [p]
    int m)
{
    const int pathway = blockIdx.x;
    const float* W_col = W + (size_t)pathway * m;

    float s1 = 0.f, s2 = 0.f;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        float w = W_col[g];
        s1 += w;
        s2 += w * w;
    }

    // Warp-level reduction for both accumulators.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, offset);
        s2 += __shfl_down_sync(0xffffffff, s2, offset);
    }

    // Cross-warp reduction via shared memory.
    // smem layout: [0..7] = warp s1 partials, [8..15] = warp s2 partials.
    __shared__ float smem[16];
    const int wid   = threadIdx.x / 32;
    const int lane  = threadIdx.x & 31;
    const int nwarp = (blockDim.x + 31) / 32;

    if (lane == 0) {
        smem[wid]     = s1;
        smem[8 + wid] = s2;
    }
    __syncthreads();

    if (wid == 0) {
        float v1 = (lane < nwarp) ? smem[lane]     : 0.f;
        float v2 = (lane < nwarp) ? smem[8 + lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
        }
        if (lane == 0) {
            mean_W[pathway]  = v1;   // raw sum; divide by m in Pass 3
            sum_W2[pathway]  = v2;
        }
    }
}

// ---------------------------------------------------------------------------
// ulm_var_kernel — compute var_W[p] = (sum_W2[p]/m) - mean_W[p]² .
//
// Also converts mean_W from raw sum → true mean (divides by m).
// One thread per pathway.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void ulm_var_kernel(
    float* __restrict__ mean_W,   // in: raw sum; out: true mean
    float* __restrict__ sum_W2,   // in: raw sum of squares; out: var_W
    int p,
    float m_inv)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    const float mu   = mean_W[i] * m_inv;
    mean_W[i]        = mu;
    sum_W2[i]        = sum_W2[i] * m_inv - mu * mu;  // var_W
}

// ---------------------------------------------------------------------------
// ulm_score_kernel — element-wise final scoring.
//
// score[c, p] = ((XW[c,p] / m) - mean_X[c] * mean_W[p]) / max(var_W[p], ε)
//
// scores and XW are n × p col-major (stride n per pathway).
// One thread per output element.
// var_W ≤ 0 after numerical noise → guard with max(var_W, ε); if var_W == 0
// exactly (constant column), result is 0 (numerator / eps ≈ 0/eps = 0 when
// X and W are both constant, but eps guard prevents NaN). We force 0 when
// var_W < ε to satisfy the mission spec (constant pathway → score = 0).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void ulm_score_kernel(
    float*       __restrict__ scores,    // n × p col-major [n*p], in: XW; out: β_1
    const float* __restrict__ mean_X,   // per-cell mean [n]
    const float* __restrict__ mean_W,   // per-pathway mean [p]
    const float* __restrict__ var_W,    // per-pathway variance [p]
    int n,
    int p,
    float m_inv,
    float epsilon)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * p) return;

    const int pathway = idx / n;
    const int cell    = idx % n;

    const float vw = var_W[pathway];
    if (vw <= epsilon) {
        scores[idx] = 0.f;  // undefined regression (constant W column)
        return;
    }

    const float cov = scores[idx] * m_inv - mean_X[cell] * mean_W[pathway];
    scores[idx] = cov / vw;
}

// ---------------------------------------------------------------------------
// run_spmm_ulm — internal: compute XW = X^T · W via cuSPARSE SpMM.
//
// Identical to run_spmm in decoupler_wsum.h. Inlined here to avoid
// cross-header anonymous-namespace linkage issues.
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> run_spmm_ulm(
    const io::PzDeviceMatrix& X,
    const float* d_W,
    int n_pathways,
    cudaStream_t stream)
{
    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    const size_t out_sz = static_cast<size_t>(n) * static_cast<size_t>(n_pathways);
    core::DeviceMemory<float> d_scores(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_scores.get(), 0, out_sz * sizeof(float), stream);

    if (nnz == 0 || out_sz == 0) {
        return d_scores;
    }

    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    cusparseSetStream(sp_handle, stream);

    cusparseSpMatDescr_t spX;
    CUSPARSE_CHECK(cusparseCreateCsc(
        &spX,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n),
        static_cast<int64_t>(nnz),
        X.mat.col_ptr.get(),
        X.mat.row_indices.get(),
        X.mat.values.get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    cusparseDnMatDescr_t dnW;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnW,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(m),
        const_cast<float*>(d_W),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    cusparseDnMatDescr_t dnOut;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnOut,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(n),
        d_scores.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    size_t ws_bytes = 0;
    const float alpha = 1.f, beta = 0.f;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spX, dnW, &beta, dnOut,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &ws_bytes));

    core::DeviceMemory<uint8_t> d_workspace(ws_bytes > 0 ? ws_bytes : 1);

    CUSPARSE_CHECK(cusparseSpMM(
        sp_handle,
        CUSPARSE_OPERATION_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spX, dnW, &beta, dnOut,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        d_workspace.get()));

    cusparseDestroyDnMat(dnOut);
    cusparseDestroyDnMat(dnW);
    cusparseDestroySpMat(spX);
    cusparseDestroy(sp_handle);

    return d_scores;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// ulm — public entry point
//
// Computes ULM (Univariate Linear Model) pathway scores (Badia-i-Mompel et al. 2022).
// Returns the OLS regression slope β_1[c, p] for each (cell, pathway) pair.
//
// Inputs:
//   X          — m_genes × n_cells sparse CSC expression matrix (PzDeviceMatrix)
//   d_W        — device pointer to dense weight matrix (m × p, col-major fp32)
//                Gene order in W must match X (caller's responsibility).
//   n_pathways — number of pathways p (W's column count)
//   cfg        — UlmConfig (epsilon guard, deterministic flag)
//   stream     — optional cudaStream_t (nullptr → default context)
//
// Returns UlmResult with scores (n × p, col-major) on device.
// Caller must sync stream before reading scores.
// ---------------------------------------------------------------------------
inline UlmResult ulm(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const UlmConfig&          cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;
    const int n   = X.mat.cols;
    const int nnz = X.mat.nnz;

    if (n_pathways <= 0) {
        throw std::runtime_error("ulm: n_pathways must be > 0");
    }
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("ulm: X has invalid dimensions (m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ")");
    }

    constexpr int THREADS = 256;
    const float m_inv = 1.f / static_cast<float>(m);

    // -------------------------------------------------------------------------
    // Pass 1 — mean_X[c]: per-cell (column) mean of X.
    //
    // Step 1a: scatter nnz values into col_sum[n] via atomic scatter.
    //          One thread per nnz entry; binary search over col_ptr to find j.
    // Step 1b: divide col_sum by m element-wise → mean_X[n].
    //
    // WHY custom kernel instead of cuSPARSE SpMV (X · 1_m):
    //   SpMV would require creating a dense all-1 vector of length m and a
    //   second cuSPARSE handle/descriptor, adding ~10 LOC of boilerplate.
    //   The atomic scatter + binary search approach costs O(nnz·log n) but is
    //   functionally equivalent and reuses the col_ptr already on device.
    //   For typical nnz/n ratios in scRNA (nnz/n ≈ m * fill ≈ 20k * 0.1 = 2k
    //   entries per cell), log n < 20, so the overhead is negligible vs SpMM.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_col_sum(n);
    core::DeviceMemory<float> d_mean_X(n);
    cudaMemsetAsync(d_col_sum.get(), 0, n * sizeof(float), stream);

    if (nnz > 0) {
        const int grid_nnz = (nnz + THREADS - 1) / THREADS;
        ulm_col_scatter_kernel<<<grid_nnz, THREADS, 0, stream>>>(
            X.mat.values.get(),
            X.mat.col_ptr.get(),
            d_col_sum.get(),
            n, nnz);
    }

    {
        const int grid_n = (n + THREADS - 1) / THREADS;
        ulm_col_mean_kernel<<<grid_n, THREADS, 0, stream>>>(
            d_col_sum.get(), d_mean_X.get(), n, m_inv);
    }

    // -------------------------------------------------------------------------
    // Pass 2 — mean_W[p] and sum_W2[p]: fused single-pass kernel.
    //
    // One block per pathway; warp-shuffle reduction over m genes.
    // Accumulates Σ W[g] and Σ W[g]² simultaneously — no Welford, just two
    // independent accumulators.  Avoids a second pass over W (saves m*p reads).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_mean_W(n_pathways);
    core::DeviceMemory<float> d_sum_W2(n_pathways);

    ulm_w_stats_kernel<<<n_pathways, THREADS, 0, stream>>>(
        d_W, d_mean_W.get(), d_sum_W2.get(), m);

    // -------------------------------------------------------------------------
    // Pass 3 — var_W[p] and finalize mean_W[p].
    //
    // mean_W[p] = raw_sum / m;  var_W[p] = sum_W2/m - mean_W[p]²
    // Overwrites mean_W and sum_W2 in-place (sum_W2 → var_W after this pass).
    // -------------------------------------------------------------------------
    {
        const int grid_p = (n_pathways + THREADS - 1) / THREADS;
        ulm_var_kernel<<<grid_p, THREADS, 0, stream>>>(
            d_mean_W.get(), d_sum_W2.get(), n_pathways, m_inv);
    }

    // -------------------------------------------------------------------------
    // Pass 4 — SpMM: XW = X^T · W (n × p, col-major).
    //
    // Same cuSPARSE strategy as WSUM/WMEAN: CSC descriptor with opA=TRANSPOSE
    // computes X^T (n×m) · W (m×p) = XW (n×p).
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_scores = run_spmm_ulm(X, d_W, n_pathways, stream);

    // -------------------------------------------------------------------------
    // Pass 5 — score kernel: compute β_1[c, p] element-wise.
    //
    // score[c,p] = ((XW[c,p]/m) - mean_X[c]*mean_W[p]) / max(var_W[p], ε)
    // d_sum_W2 now holds var_W after Pass 3.
    // -------------------------------------------------------------------------
    {
        const int total = n * n_pathways;
        const int grid  = (total + THREADS - 1) / THREADS;
        ulm_score_kernel<<<grid, THREADS, 0, stream>>>(
            d_scores.get(),
            d_mean_X.get(),
            d_mean_W.get(),
            d_sum_W2.get(),   // var_W
            n, n_pathways,
            m_inv,
            cfg.epsilon);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    UlmResult res;
    res.scores     = std::move(d_scores);
    res.n_cells    = n;
    res.n_pathways = n_pathways;
    return res;
}

}  // namespace enrich
}  // namespace singlet::gpu
