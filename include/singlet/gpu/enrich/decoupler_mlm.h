// SPDX-License-Identifier: MIT
// integrates: decoupleR (Badia-i-Mompel et al. 2022)
//
// singlet/gpu/enrich/decoupler_mlm.h
//
// MLM (Multivariate Linear Model) pathway-scoring method from decoupleR.
//
// Algorithm reference:
//   Badia-i-Mompel P, Vélez Santiago J, Braunger J, et al. (2022)
//   "decoupleR: ensemble of computational methods to infer biological
//   activities from omics data." Bioinformatics Advances 2:vbac016.
//   https://doi.org/10.1093/bioadv/vbac016
//
// For each cell c, fit a joint multivariate OLS across all p pathways:
//
//   X[:, c] = W · β_c + ε    (over all m genes)
//
// Closed-form solution (OLS):
//
//   β_c = (W^T W)^{-1} W^T X[:, c]
//
// Vectorized over all cells:
//
//   B = (W^T W)^{-1} (W^T X)
//
// where B is p × n_cells (each column = β for one cell).
//
// GPU implementation (4 sequential device passes, no inner loop):
//   Pass 1 — A = W^T W  (p × p, dense symmetric positive definite).
//             cuBLAS Sgemm with opA=T, opB=N:  m=p, n=p, k=m_genes.
//   Pass 2 — Ridge: add cfg.ridge to diagonal of A — protects against
//             rank deficiency (e.g. correlated or duplicate pathways).
//   Pass 3 — Y = W^T X  (p × n_cells, dense col-major).
//             cuSPARSE SpMM with opA=T on CSC X.  Output Y is p × n_cells.
//   Pass 4a — Cholesky factor A = L L^T via cusolverDnSpotrf (LOWER mode).
//              Checks info == 0 (positive definite after ridge).
//   Pass 4b — Solve A B = Y via cusolverDnSpotrs (LOWER mode, in-place).
//              Y is overwritten with B (p × n_cells, col-major).
//   Pass 5  — Transpose B from (p × n_cells) col-major to (n_cells × p)
//              col-major via cuBLAS Sgeam with opA=T. Matches WSUM/WMEAN/ULM
//              output convention.
//
// Determinism (Rule 18):
//   cuBLAS Sgemm (Pass 1) + cuSPARSE SpMM (Pass 3) + cuSOLVER Potrf/Potrs
//   (Pass 4) + cuBLAS Sgeam (Pass 5): all deterministic at fp32 for a fixed
//   sparsity pattern and operand values on a fixed GPU architecture.
//   cfg.deterministic is a no-op (API symmetry with WSUM/WMEAN/ULM).
//
// Memory (Rule 5): all device buffers via core::DeviceMemory<T>. No raw cudaMalloc.
// D2H (Rule 4): one D2H scalar (info after Spotrf) is acceptable (one-shot check).
//               No D2H in any inner loop.
// CYCLE-136: initial implementation.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>

// Local CUSOLVER_CHECK — short unprefixed alias used by this header's cuSOLVER
// call sites. The canonical macro is SINGLET_GPU_CUSOLVER_CHECK in
// core/types.h; this #ifndef-guarded alias keeps the local call sites terse.
#ifndef CUSOLVER_CHECK
#  define CUSOLVER_CHECK(call)                                              \
    do {                                                                    \
        cusolverStatus_t _s = (call);                                       \
        if (_s != CUSOLVER_STATUS_SUCCESS) {                                \
            throw std::runtime_error(                                       \
                std::string("cuSOLVER error at ") + __FILE__ + ":" +        \
                std::to_string(__LINE__) + " status=" + std::to_string(_s));\
        }                                                                   \
    } while (0)
#endif

namespace singlet::gpu {
namespace enrich {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

struct MlmConfig {
    float ridge       = 1e-6f;  // tiny ridge added to diag(W^T W) for stability
    bool  deterministic = true; // no-op: MLM is already deterministic by construction
};

struct MlmResult {
    core::DeviceMemory<float> scores;  // n_cells × n_pathways, col-major
                                        // = transpose of (p × n_cells) internal B
    int n_cells;
    int n_pathways;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of the public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// mlm_ridge_kernel — add ridge to diagonal of A (p × p, col-major).
//
// A[i, i] += ridge for i in [0, p).
// A is col-major with leading dimension p: A[i + i*p] = A[i*(p+1)].
// One thread per diagonal element.  No atomics → deterministic.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mlm_ridge_kernel(
    float* __restrict__ A,
    int    p,
    float  ridge)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    A[i * p + i] += ridge;   // col-major: element (i,i) at index i*p + i
}

// ---------------------------------------------------------------------------
// mlm_transpose_kernel — transpose B (p × n_cells) col-major → out (n × p)
//                         col-major.
//
// out[c + p_idx * n] = B[p_idx + c * p]
//
// Equivalently: out is n_cells × n_pathways col-major, matching WSUM/WMEAN/ULM
// score convention.  One thread per output element.
//
// NOTE: for large p × n we use cuBLAS Sgeam (Pass 5 in the entry point) which
// uses tiled shared-memory transpose and is faster.  This kernel is kept as a
// fallback reference — it is NOT launched in mlm(); Sgeam is always used.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void mlm_transpose_kernel(
    const float* __restrict__ B,    // p × n col-major: B[p_idx + c * p]
    float*       __restrict__ out,  // n × p col-major: out[c + p_idx * n]
    int p,
    int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= p * n) return;
    const int p_idx = idx % p;
    const int c     = idx / p;
    out[c + (ptrdiff_t)p_idx * n] = B[p_idx + (ptrdiff_t)c * p];
}

// ---------------------------------------------------------------------------
// run_spmm_mlm — internal: compute Y = W^T X via cuSPARSE SpMM.
//
// X is CSC (m × n_cells).  W is dense m × p col-major.
// With opA = TRANSPOSE on X, cuSPARSE computes:
//   (n_cells × m) · (m × p) = (n_cells × p)
// BUT we want Y = W^T X of shape (p × n_cells).
//
// Strategy: compute X^T · W (n_cells × p col-major) into d_Y first via SpMM,
// then the caller transposes via Sgeam.  Wait — the mission spec says:
//   "Compute Y = W^T X (p × n_cells, dense, via SpMM with op_A=T)".
// This requires setting X as the sparse factor and using it transposed:
//   cuSPARSE SpMM (opX=T, opW=N): X^T (n×m) · W (m×p) → n×p output.
// That gives us n_cells × p, NOT p × n_cells.
//
// We therefore allocate Y as (n_cells × p) col-major from SpMM, then
// in Pass 5 we transpose to (n_cells × p) from the solver's (p × n_cells).
//
// Actually the cleanest path (matching the spec): compute Y as p × n_cells
// col-major directly.  cuSPARSE SpMM supports:
//   opA=NON_TRANSPOSE on W (m×p dense) with opB=TRANSPOSE on X (m×n_cells CSC).
// But cusparseSpMM requires the *sparse* matrix as the first operand.
//
// Compromise: call SpMM as X^T · W → (n_cells × p) col-major, store in d_Y
// of shape [n_cells, p].  Then cuSOLVER needs a p × n_cells RHS.  So we call
// Sgeam to transpose d_Y into a p × n_cells buffer before Potrs.
//
// Alternative: just call SpMM to get n_cells × p, then treat that as the RHS
// after transposing once to p × n_cells before Spotrs.
//
// Implementation here: SpMM produces (n_cells × p) col-major (same as ULM/WSUM
// SpMM output), then the caller does one Sgeam transpose to (p × n_cells) for
// Spotrs, and another Sgeam transpose to (n_cells × p) for the final output.
// ---------------------------------------------------------------------------
inline core::DeviceMemory<float> run_spmm_mlm(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    cudaStream_t              stream)
{
    const int m   = X.mat.rows;   // genes
    const int n   = X.mat.cols;   // cells
    const int nnz = X.mat.nnz;

    // Output: n_cells × p col-major (same layout as WSUM/WMEAN/ULM SpMM).
    const size_t out_sz = static_cast<size_t>(n) * static_cast<size_t>(n_pathways);
    core::DeviceMemory<float> d_out(out_sz > 0 ? out_sz : 1);
    cudaMemsetAsync(d_out.get(), 0, out_sz * sizeof(float), stream);

    if (nnz == 0 || out_sz == 0) {
        return d_out;
    }

    cusparseHandle_t sp_handle;
    CUSPARSE_CHECK(cusparseCreate(&sp_handle));
    cusparseSetStream(sp_handle, stream);

    // Sparse descriptor for X as CSC (m × n_cells).
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

    // Dense descriptor for W (m × p, col-major; ld = m).
    cusparseDnMatDescr_t dnW;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnW,
        static_cast<int64_t>(m),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(m),
        const_cast<float*>(d_W),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

    // Dense descriptor for output (n_cells × p, col-major; ld = n_cells).
    // opA = TRANSPOSE on X → X^T (n×m) · W (m×p) = out (n×p).
    cusparseDnMatDescr_t dnOut;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &dnOut,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n_pathways),
        static_cast<int64_t>(n),
        d_out.get(),
        CUDA_R_32F,
        CUSPARSE_ORDER_COL));

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

    return d_out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// mlm — public entry point
//
// Computes MLM (Multivariate Linear Model) pathway scores (Badia-i-Mompel
// et al. 2022).  Jointly regresses all pathways on each cell simultaneously
// using closed-form OLS: B = (W^T W)^{-1} W^T X.
//
// Inputs:
//   X          — m_genes × n_cells sparse CSC expression matrix (PzDeviceMatrix)
//   d_W        — device pointer to dense weight matrix (m × p, col-major fp32)
//                Gene order in W must match X (caller's responsibility).
//   n_pathways — number of pathways p (W's column count)
//   cfg        — MlmConfig (ridge, deterministic flag)
//   stream     — optional cudaStream_t (nullptr → default context)
//
// Returns MlmResult with scores (n × p, col-major) on device.
// Caller must sync stream before reading scores.
// ---------------------------------------------------------------------------
inline MlmResult mlm(
    const io::PzDeviceMatrix& X,
    const float*              d_W,
    int                       n_pathways,
    const MlmConfig&          cfg    = {},
    cudaStream_t              stream = nullptr)
{
    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int m   = X.mat.rows;   // genes
    const int n   = X.mat.cols;   // cells
    const int p   = n_pathways;

    // Dimension validation.
    if (p <= 0) {
        throw std::runtime_error("mlm: n_pathways must be > 0");
    }
    if (m <= 0 || n <= 0) {
        throw std::runtime_error("mlm: X has invalid dimensions (m=" +
            std::to_string(m) + ", n=" + std::to_string(n) + ")");
    }
    if (p > m) {
        throw std::runtime_error("mlm: n_pathways (" + std::to_string(p) +
            ") > n_genes (" + std::to_string(m) + "); W^T W is rank-deficient");
    }

    constexpr int THREADS = 256;

    // Acquire handles from default context (single set of handles per process).
    cublasHandle_t     blas_h   = singlet::gpu::core::default_context().blas();
    cusolverDnHandle_t solver_h = singlet::gpu::core::default_context().solver();
    cublasSetStream(blas_h, stream);
    cusolverDnSetStream(solver_h, stream);

    // -------------------------------------------------------------------------
    // Pass 1 — A = W^T W  (p × p, col-major symmetric positive definite).
    //
    // cuBLAS Sgemm: opA=T, opB=N
    //   A (p×p) = W^T (p×m) · W (m×p)
    //   m=p, n=p, k=m_genes; lda=m_genes, ldb=m_genes, ldc=p
    // -------------------------------------------------------------------------
    const size_t A_sz = static_cast<size_t>(p) * static_cast<size_t>(p);
    core::DeviceMemory<float> d_A(A_sz);
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgemm(
            blas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            p, p, m,                    // m=p, n=p, k=m_genes
            &alpha,
            d_W, m,                     // A = W (m×p), ldA = m
            d_W, m,                     // B = W (m×p), ldB = m
            &beta,
            d_A.get(), p));             // C = A (p×p), ldC = p
    }

    // -------------------------------------------------------------------------
    // Pass 2 — Ridge: add cfg.ridge to diag(A).
    //
    // One thread per diagonal element; A[i*(p+1)] += ridge.
    // -------------------------------------------------------------------------
    if (cfg.ridge != 0.f) {
        const int grid_p = (p + THREADS - 1) / THREADS;
        mlm_ridge_kernel<<<grid_p, THREADS, 0, stream>>>(
            d_A.get(), p, cfg.ridge);
    }

    // -------------------------------------------------------------------------
    // Pass 3 — Y = W^T X  (n_cells × p, col-major) via cuSPARSE SpMM.
    //
    // SpMM computes X^T · W → (n_cells × p).  We then transpose to (p × n_cells)
    // for cuSOLVER Spotrs, which needs RHS in (p × n_cells) col-major.
    // -------------------------------------------------------------------------
    core::DeviceMemory<float> d_YnxP = run_spmm_mlm(X, d_W, p, stream);
    // d_YnxP is (n_cells × p) col-major.

    // Transpose d_YnxP (n × p) → d_Y (p × n) col-major for Spotrs.
    const size_t Y_sz = static_cast<size_t>(p) * static_cast<size_t>(n);
    core::DeviceMemory<float> d_Y(Y_sz > 0 ? Y_sz : 1);
    {
        const float alpha = 1.f, beta = 0.f;
        // Sgeam: C = alpha * op(A) + beta * op(B)
        //   opA=T: transposes d_YnxP (n × p) → (p × n)
        //   lda = n (leading dim of d_YnxP, col-major n×p)
        //   ldc = p (leading dim of d_Y, col-major p×n)
        CUBLAS_CHECK(cublasSgeam(
            blas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            p, n,                     // rows=p, cols=n of output
            &alpha,
            d_YnxP.get(), n,          // A: (n×p) col-major, ldA=n
            &beta,
            d_Y.get(), p,             // B: ignored (beta=0), ldB=p
            d_Y.get(), p));           // C: (p×n) col-major, ldC=p
    }
    // d_YnxP no longer needed; free by going out of scope after this block.
    // (DeviceMemory RAII: freed when d_YnxP goes out of scope at function end.)

    // -------------------------------------------------------------------------
    // Pass 4a — Cholesky factor A = L L^T (in-place, LOWER mode).
    //
    // cusolverDnSpotrf_bufferSize → query workspace size.
    // cusolverDnSpotrf → compute factorization.
    // d_info is a device integer; one D2H read after Spotrf (acceptable: one-shot).
    // -------------------------------------------------------------------------
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(
        solver_h,
        CUBLAS_FILL_MODE_LOWER,
        p,
        d_A.get(),
        p,
        &lwork));

    core::DeviceMemory<float> d_work(lwork > 0 ? static_cast<size_t>(lwork) : 1);
    core::DeviceMemory<int>   d_info(1);

    CUSOLVER_CHECK(cusolverDnSpotrf(
        solver_h,
        CUBLAS_FILL_MODE_LOWER,
        p,
        d_A.get(),
        p,
        d_work.get(),
        lwork,
        d_info.get()));

    // Synchronize and check Cholesky info (one D2H scalar — Rule 4 allows one-shot).
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    int h_info = 0;
    cudaMemcpy(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        throw std::runtime_error(
            "mlm: cusolverDnSpotrf failed (info=" + std::to_string(h_info) +
            "). W^T W is not positive definite — increase cfg.ridge.");
    }

    // -------------------------------------------------------------------------
    // Pass 4b — Solve A B = Y via cusolverDnSpotrs (in-place, LOWER mode).
    //
    // d_Y (p × n_cells, col-major) is overwritten with B (p × n_cells).
    // A is already factored (L L^T stored in LOWER triangle of d_A).
    // -------------------------------------------------------------------------
    CUSOLVER_CHECK(cusolverDnSpotrs(
        solver_h,
        CUBLAS_FILL_MODE_LOWER,
        p,                 // n (matrix order = p × p)
        n,                 // nrhs (number of right-hand sides = n_cells)
        d_A.get(),         // factored A (L L^T in lower triangle)
        p,                 // lda = p
        d_Y.get(),         // RHS on entry, B on exit (p × n_cells, col-major)
        p,                 // ldb = p
        d_info.get()));    // device info output

    // Check Spotrs info (synchronize, one D2H scalar).
    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaMemcpy(&h_info, d_info.get(), sizeof(int), cudaMemcpyDeviceToHost);
    if (h_info != 0) {
        throw std::runtime_error(
            "mlm: cusolverDnSpotrs failed (info=" + std::to_string(h_info) + ").");
    }

    // -------------------------------------------------------------------------
    // Pass 5 — Transpose B from (p × n_cells) col-major to (n_cells × p)
    //          col-major for output.
    //
    // cuBLAS Sgeam: opA=T, C = alpha * A^T + beta * B
    //   A = d_Y (p × n), ldA = p  →  A^T is (n × p)
    //   C = d_scores (n × p), ldC = n
    // -------------------------------------------------------------------------
    const size_t out_sz = static_cast<size_t>(n) * static_cast<size_t>(p);
    core::DeviceMemory<float> d_scores(out_sz > 0 ? out_sz : 1);
    {
        const float alpha = 1.f, beta = 0.f;
        CUBLAS_CHECK(cublasSgeam(
            blas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            n, p,                     // rows=n, cols=p of output
            &alpha,
            d_Y.get(), p,             // A: (p×n) col-major, ldA=p
            &beta,
            d_scores.get(), n,        // B: ignored, ldB=n
            d_scores.get(), n));      // C: (n×p) col-major, ldC=n
    }

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    MlmResult res;
    res.scores     = std::move(d_scores);
    res.n_cells    = n;
    res.n_pathways = p;
    return res;
}

}  // namespace enrich
}  // namespace singlet::gpu
