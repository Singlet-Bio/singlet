// SPDX-License-Identifier: MIT
// integrates: decoupleR (Badia-i-Mompel et al. 2022)
//
// singlet/gpu/enrich/decoupler_wsum.h
//
// WSUM and WMEAN pathway-scoring methods from decoupleR.
//
// Algorithm reference:
//   Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022)
//   "decoupleR: ensemble of computational methods to infer biological
//   activities from omics data." Bioinformatics Advances 2:vbac016.
//   https://doi.org/10.1093/bioadv/vbac016
//
// Given expression matrix X (m_genes × n_cells, sparse CSC) and a dense
// weight matrix W (m_genes × n_pathways, col-major), both methods compute:
//
//   wsum[c, p] = Σ_g  X[g, c] · W[g, p]        (sparse × dense matmul)
//
// WSUM then normalises by L1 norm of each pathway's weights:
//   norm[p]     = Σ_g |W[g, p]|
//   score[c, p] = wsum[c, p] / max(norm[p], ε)
//
// WMEAN normalises by the count of nonzero weights:
//   n_nz[p]     = #{g : W[g, p] ≠ 0}
//   score[c, p] = wsum[c, p] / max(n_nz[p], 1)
//
// Implementation:
//   Step 1 — cuSPARSE SpMM: X^T · W → dense (n × p) col-major.
//             X is CSC (m × n); with CUSPARSE_OPERATION_TRANSPOSE for the
//             sparse operand, cuSPARSE treats it as X^T (n × m) and multiplies
//             by W (m × p), yielding output (n × p).
//   Step 2 — Per-pathway normaliser kernel (wsum_l1_norm_kernel or
//             wmean_nz_count_kernel): one block per pathway, warp-shuffle
//             reduction, no atomics → deterministic.
//   Step 3 — Column-scale kernel: divides each column p of scores by norm[p]
//             or n_nz[p].  One thread per element; fused into a single pass.
//
// Time complexity:
//   SpMM: O(nnz × p) — dominant term.
//   Norm: O(m × p)   — one pass over W, embarrassingly parallel.
//   Scale: O(n × p)  — trivial.
//
// Workspace budget (above input CSC and W):
//   scores: n × p × 4 bytes (output, col-major)
//   norm:   p × 4 bytes     (per-pathway normaliser, temporary)
//   cuSPARSE workspace: queried at runtime (typically 0 for CSC transpose SpMM)
//
// Determinism (Rule 18):
//   SpMM: CUSPARSE_SPMM_ALG_DEFAULT on fixed architecture is deterministic for
//         a fixed sparsity pattern and operand values.
//   Norm kernel: warp-shuffle reduction only — no atomics → bit-exact.
//   Scale kernel: one thread per element → bit-exact.
//   Overall: bit-exact deterministic for fixed inputs. cfg.deterministic is
//   accepted for API symmetry but is a no-op (already deterministic).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): no per-inner-loop D2H. One D2H only in the nnz=0 early-exit path.

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

struct WsumConfig {
    float epsilon      = 1e-9f;  // guard against division by near-zero L1 norm
    bool  deterministic = false; // SpMM + warp-shuffle norm: already deterministic;
                                 // flag accepted for API symmetry, no behavioural change.
};

struct WmeanConfig {
    bool  deterministic = false; // same — no-op; see header notes.
};

struct WsumResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
    int n_cells;
    int n_pathways;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// wsum_l1_norm_kernel — compute per-pathway L1 absolute weight sum.
//
// norm[p] = Σ_{g=0}^{m-1} |W[g + p*m]|   (W is col-major: stride m per pathway)
//
// Launch: one block per pathway; threads stride over m genes.
// Reduction: intra-warp with __shfl_down_sync; cross-warp via shared memory.
// No atomics → fully deterministic (Rule 18).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void wsum_l1_norm_kernel(
    const float* __restrict__ W,       // m × p col-major weight matrix
    float*       __restrict__ norm,    // output: L1 norm per pathway [p]
    int m)                             // n_genes
{
    const int pathway = blockIdx.x;
    const float* W_col = W + (size_t)pathway * m;  // pointer to pathway column

    float acc = 0.f;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        acc += fabsf(W_col[g]);
    }

    // Warp-level reduction.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Cross-warp reduction via shared memory.
    __shared__ float smem[8];  // up to 256/32 = 8 warps
    const int wid  = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int nwarp = (blockDim.x + 31) / 32;

    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) norm[pathway] = val;
    }
}

// ---------------------------------------------------------------------------
// wmean_nz_count_kernel — count nonzero weights per pathway.
//
// n_nz[p] = #{g : W[g + p*m] != 0}  stored as float for the division step.
//
// Same launch and reduction pattern as wsum_l1_norm_kernel (deterministic).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 2)
void wmean_nz_count_kernel(
    const float* __restrict__ W,       // m × p col-major weight matrix
    float*       __restrict__ n_nz,    // output: nonzero count per pathway [p], as float
    int m)
{
    const int pathway = blockIdx.x;
    const float* W_col = W + (size_t)pathway * m;

    float acc = 0.f;
    for (int g = threadIdx.x; g < m; g += blockDim.x) {
        acc += (W_col[g] != 0.f) ? 1.f : 0.f;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    __shared__ float smem[8];
    const int wid   = threadIdx.x / 32;
    const int lane  = threadIdx.x & 31;
    const int nwarp = (blockDim.x + 31) / 32;

    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (wid == 0) {
        float val = (lane < nwarp) ? smem[lane] : 0.f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) n_nz[pathway] = val;
    }
}

// ---------------------------------------------------------------------------
// column_scale_kernel — divide each column p of scores by normaliser[p].
//
// scores is n × p col-major: column p occupies scores[p*n .. (p+1)*n - 1].
// One thread per element; guard against denominator < eps (WSUM) or < 1 (WMEAN).
//
// The two division modes (L1 / count) are unified by passing the pre-computed
// normaliser vector — this kernel is shared by both WSUM and WMEAN paths.
//
// denom_mode:
//   0 → WSUM: guard = max(norm[p], epsilon)
//   1 → WMEAN: guard = max(n_nz[p], 1.f)
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void column_scale_kernel(
    float*       __restrict__ scores,      // n × p col-major [n*p]
    const float* __restrict__ normaliser,  // per-pathway denominator [p]
    int n,
    int p,
    float epsilon,
    int denom_mode)  // 0=WSUM, 1=WMEAN
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * p) return;

    const int pathway = idx / n;  // col-major: pathway = element / n_cells
    float denom = normaliser[pathway];
    if (denom_mode == 0) {
        denom = (denom < epsilon) ? epsilon : denom;  // WSUM eps guard
    } else {
        denom = (denom < 1.f) ? 1.f : denom;          // WMEAN count guard
    }
    scores[idx] /= denom;
}

// ---------------------------------------------------------------------------
// run_spmm — internal helper: compute wsum = X^T · W via cuSPARSE SpMM.
//
// X is CSC (m × n) stored as (col_ptr, row_indices, values).
// W is dense m × p col-major (on device).
// Output: scores allocated as n × p col-major DeviceMemory<float>.
//
// cuSPARSE strategy:
//   Create a CSC sparse descriptor for X. With opA = TRANSPOSE, cuSPARSE
//   treats X^T (n × m) as the left operand and W (m × p) as the right dense
//   matrix, producing an n × p dense output — exactly wsum.
//
// cuSPARSE CSC descriptor: use cusparseCreateCsc with the CSC arrays.
//   opA = CUSPARSE_OPERATION_TRANSPOSE → cuSPARSE will compute X^T · W.
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> run_spmm(
    const io::PzDeviceMatrix& X,
    const float* d_W,
    int n_pathways,
    cudaStream_t stream)
{
    const int m   = X.mat.rows;   // genes
    const int n   = X.mat.cols;   // cells
    const int nnz = X.mat.nnz;

    // Output: n × p col-major.
    const size_t out_sz = static_cast<size_t>(n) * static_cast<size_t>(n_pathways);
    core::DeviceMemory<float> d_scores(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_scores.get(), 0, out_sz * sizeof(float), stream);

    if (nnz == 0 || out_sz == 0) {
        // All-zero X → all-zero scores; no cuSPARSE call needed.
        return d_scores;
    }

    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    cusparseSetStream(sp_handle, stream);

    // Sparse descriptor for X as CSC (m × n).
    // cusparseCreateCsc expects: (rows=m, cols=n, nnz, col_offsets, row_indices, values).
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

    // Dense descriptor for W (m × p, col-major; leading dim = m).
    cusparseDnMatDescr_t dnW;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnW,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(m),  // leading dimension (col-major)
        const_cast<float*>(d_W),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    // Dense descriptor for output scores (n × p, col-major; leading dim = n).
    cusparseDnMatDescr_t dnOut;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnOut,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(n),  // leading dimension
        d_scores.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    // Workspace query.
    // opA = TRANSPOSE → computes X^T (n×m) · W (m×p) = scores (n×p).
    size_t ws_bytes = 0;
    const float alpha = 1.f, beta = 0.f;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        sp_handle,
        CUSPARSE_OPERATION_TRANSPOSE,      // opA: X^T
        CUSPARSE_OPERATION_NON_TRANSPOSE,  // opB: W unchanged
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
// wsum — public entry point
//
// Computes WSUM pathway scores (Badia-i-Mompel et al. 2022).
// Inputs:
//   X          — m_genes × n_cells sparse CSC expression matrix (PzDeviceMatrix)
//   d_W        — device pointer to dense weight matrix (m × p, col-major fp32)
//                Gene order in W must match X (caller's responsibility).
//   n_pathways — number of pathways p (W's column count)
//   cfg        — WsumConfig (epsilon guard, deterministic flag)
//   stream     — optional cudaStream_t (nullptr → default context)
//
// Returns WsumResult with scores (n × p, col-major) on device.
// Caller must sync stream before reading scores.
// ---------------------------------------------------------------------------
inline WsumResult wsum(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const WsumConfig&         cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m = X.mat.rows;
    const int n = X.mat.cols;

    if (n_pathways <= 0) {
        throw std::runtime_error("wsum: n_pathways must be > 0");
    }
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("wsum: X has invalid dimensions (m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ")");
    }

    // --- Step 1: SpMM — scores = X^T · W (n × p, col-major) ---
    core::DeviceMemory<float> d_scores = run_spmm(X, d_W, n_pathways, stream);

    // --- Step 2: Compute per-pathway L1 norm of W ---
    core::DeviceMemory<float> d_norm(n_pathways);
    {
        constexpr int THREADS = 256;
        wsum_l1_norm_kernel<<<n_pathways, THREADS, 0, stream>>>(
            d_W, d_norm.get(), m);
    }

    // --- Step 3: Divide each column of scores by its L1 norm ---
    {
        constexpr int THREADS = 256;
        const int total = n * n_pathways;
        const int grid  = (total + THREADS - 1) / THREADS;
        column_scale_kernel<<<grid, THREADS, 0, stream>>>(
            d_scores.get(),
            d_norm.get(),
            n, n_pathways,
            cfg.epsilon,
            0 /* WSUM mode */);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    WsumResult res;
    res.scores     = std::move(d_scores);
    res.n_cells    = n;
    res.n_pathways = n_pathways;
    return res;
}

// ---------------------------------------------------------------------------
// wmean — public entry point
//
// Computes WMEAN pathway scores (Badia-i-Mompel et al. 2022).
// Identical to WSUM except the per-pathway normaliser is the count of nonzero
// weights (n_nz[p]) rather than the L1 weight sum.
//
// Inputs: same as wsum (WmeanConfig mirrors WsumConfig; epsilon not used —
//         the guard is max(n_nz[p], 1) to prevent division by zero when all
//         weights in a pathway are zero).
// ---------------------------------------------------------------------------
inline WsumResult wmean(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const WmeanConfig&        cfg    = {},
    cudaStream_t              stream = nullptr)
{
    (void)cfg;  // deterministic flag is a no-op

    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m = X.mat.rows;
    const int n = X.mat.cols;

    if (n_pathways <= 0) {
        throw std::runtime_error("wmean: n_pathways must be > 0");
    }
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("wmean: X has invalid dimensions (m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ")");
    }

    // --- Step 1: SpMM — scores = X^T · W (n × p, col-major) ---
    core::DeviceMemory<float> d_scores = run_spmm(X, d_W, n_pathways, stream);

    // --- Step 2: Count nonzero weights per pathway ---
    core::DeviceMemory<float> d_nnz(n_pathways);
    {
        constexpr int THREADS = 256;
        wmean_nz_count_kernel<<<n_pathways, THREADS, 0, stream>>>(
            d_W, d_nnz.get(), m);
    }

    // --- Step 3: Divide each column of scores by its nonzero count ---
    {
        constexpr int THREADS = 256;
        const int total = n * n_pathways;
        const int grid  = (total + THREADS - 1) / THREADS;
        column_scale_kernel<<<grid, THREADS, 0, stream>>>(
            d_scores.get(),
            d_nnz.get(),
            n, n_pathways,
            1e-9f,
            1 /* WMEAN mode */);
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    WsumResult res;
    res.scores     = std::move(d_scores);
    res.n_cells    = n;
    res.n_pathways = n_pathways;
    return res;
}

}  // namespace enrich
}  // namespace singlet::gpu
