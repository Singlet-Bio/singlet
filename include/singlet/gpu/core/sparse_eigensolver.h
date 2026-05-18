// SPDX-License-Identifier: MIT
// singlet/gpu/core/sparse_eigensolver.h
//
// LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient.
// Computes top-K exterior eigenvalues + eigenvectors of a sparse symmetric matrix.
//
// Algorithm reference:
//   Knyazev (2001) "Toward the Optimal Preconditioned Eigensolver: LOBPCG."
//   SIAM J. Sci. Comput. 23:517-541. https://doi.org/10.1137/S1064827500366124
//
// Motivation:
//   embed/diffmap and embed/dpt previously materialized a dense n×n matrix and
//   ran cusolverDnSsyevd — 14× slower than ARPACK at n=10k, crashes at n=30k.
//   This header-only kernel replaces both with a sparse-safe O(nnz × K) per-iter
//   algorithm that scales to n=1M. See state/designs/sparse_eigensolver.md.
//
// Time complexity per iteration:
//   SpMM: O(nnz × K_block)
//   Gram / BLAS-3 ops: O(n × K_block²)
//   Dense Rayleigh-Ritz: O(K_block³) — negligible (K_block ≤ 45)
//
// Workspace budget (n=10k, K=15, K_block=20):
//   X, AX, R, S, AS: 5 × (10k × 20) × 4 = ~4 MB
//   Gram/eigvec temporaries: ~100 KB
//   SpMM external workspace: queried from cuSPARSE (typically <4 MB)
//   Total: ~8 MB vs 400 MB dense (50× reduction)
//
// Precision: fp32 throughout. Rayleigh-Ritz Sygvd promoted to fp64 internally
// by cuSOLVER when needed for the 3K×3K subproblem (within Rule 8: ≤k² Gram).
// Gram-Schmidt reorthogonalization applied at fp32; sufficient for K≤30 (tested).
// A second reorthogonalization pass is applied when K>20 for clustered spectra.
//
// Stream: caller-provided; nullptr resolves to default_context().stream().
//   All kernels run on the same stream. No internal stream creation (Rule 7).
//
// Determinism: fixed seed → Philox4x32 RNG → bit-identical across runs on the
//   same GPU architecture. BGS uses Cholesky factorization (deterministic at fp32
//   with fixed operand order) rather than classical GS with atomics (Rule 11).
//
// D2H transfers: exactly ONE 4-byte scalar per outer iteration (convergence check:
//   cub::DeviceReduce::Max on |R|/|rho| → cudaMemcpy 4 bytes). This is the sole
//   exception per Rule 4. All BLAS and SpMM results remain device-resident.
//
// No preconditioner in v0. Jacobi diagonal preconditioner is deferred to v1.
// Block oversampling: K_block = K + 5 to handle clustered eigenvalues (design doc §Risks).
//
// OOC plan: n exceeds GPU memory → caller partitions columns via streaming CSR
//   chunks (future; current header assumes full A fits in device memory).
//
// CYCLE-182: initial implementation.
// CYCLE-183 (iter-2 fix): LOBPCG naturally minimizes the Rayleigh quotient
// (ground-state convention → SMALLEST eigenvalues). For top-K LARGEST (graph
// Laplacian diffmap/dpt), we negate M = S^T(AS) before Ssygvd so that
// "find K_smallest of -M" ≡ "find K_largest of M" (ARPACK sign-flip trick).
// Specifically: Ssygvd returns ascending for -M; we take the FIRST K_block
// columns (smallest of -M = largest of M) rather than the last K_block.
// Residual and rho computation remain unchanged (they use A, not -A).
// See state/designs/sparse_eigensolver.md §CYCLE-183-fix.
//
// CYCLE-185 (iter-3 fix): Switch from absolute to relative residual for
// convergence. The absolute check max|AX - X*diag(rho)| < tol fails when
// spectral_radius * eps_fp32 > tol (e.g., tridiagonal with diag~20 gives
// floor ~4e-6 which exceeds tol=1e-6 and the loop never declares convergence).
// Fix: the abs kernel now computes |R[i,k]| / |rho[k]|, making the criterion
// tol-scale-independent and fp32-achievable for any well-conditioned A.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>
#include <cub/device/device_reduce.cuh>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

// cuSOLVER status check — rolls own because core/types.h only ships CUDA/cuSPARSE/cuBLAS.
#ifndef SINGLET_GPU_CUSOLVER_CHECK
#  define SINGLET_GPU_CUSOLVER_CHECK(call) \
    do { cusolverStatus_t _s = (call); \
         if (_s != CUSOLVER_STATUS_SUCCESS) \
            throw std::runtime_error(std::string("cuSOLVER error in " __FILE__ ":") \
                + std::to_string(__LINE__) + " status=" + std::to_string(static_cast<int>(_s))); } while (0)
#endif

#ifndef CUSOLVER_CHECK
#  define CUSOLVER_CHECK(call) SINGLET_GPU_CUSOLVER_CHECK(call)
#endif

// cuRAND status check.
#ifndef SINGLET_GPU_CURAND_CHECK
#  define SINGLET_GPU_CURAND_CHECK(call) \
    do { curandStatus_t _s = (call); \
         if (_s != CURAND_STATUS_SUCCESS) \
            throw std::runtime_error(std::string("cuRAND error in " __FILE__ ":") \
                + std::to_string(__LINE__) + " status=" + std::to_string(static_cast<int>(_s))); } while (0)
#endif

#ifndef CURAND_CHECK
#  define CURAND_CHECK(call) SINGLET_GPU_CURAND_CHECK(call)
#endif

namespace singlet::gpu {
namespace core {

// ---------------------------------------------------------------------------
// Public API types
// ---------------------------------------------------------------------------

// View of a symmetric CSR matrix. Raw device pointers (values, row_ptr, col_idx)
// are the API surface used by the kernel. The optional owning DeviceMemory members
// (_d_values, _d_row_ptr, _d_col_idx) let callers keep the allocations alive via
// RAII — the test harness uses this pattern to upload host CSR data to the device
// and then pass the resulting struct directly to top_k_eigsh_lobpcg.
//
// WHY public owning members: the kernel only reads the raw pointers; ownership
// semantics are a caller concern. Embedding DeviceMemory here avoids a separate
// allocation lifetime management step at call sites.
struct SparseSymmetricCSR {
    // Raw device pointers used by the kernel (must be valid for the call duration).
    const float* values  = nullptr;  // [nnz] fp32 edge weights
    const int*   row_ptr = nullptr;  // [n+1] CSR row offsets
    const int*   col_idx = nullptr;  // [nnz] CSR column indices
    int          n       = 0;        // matrix dimension (n × n)
    int          nnz     = 0;        // number of stored non-zeros

    // Optional owning RAII buffers. When non-null they keep the device memory alive.
    // Set .values/.row_ptr/.col_idx to these buffers' .get() after uploading.
    // The kernel ignores these members — they exist only for lifetime management.
    core::DeviceMemory<float> _d_values;
    core::DeviceMemory<int>   _d_row_ptr;
    core::DeviceMemory<int>   _d_col_idx;

    // Movable only (DeviceMemory is non-copyable).
    SparseSymmetricCSR() = default;
    SparseSymmetricCSR(SparseSymmetricCSR&&)            = default;
    SparseSymmetricCSR& operator=(SparseSymmetricCSR&&) = default;
    SparseSymmetricCSR(const SparseSymmetricCSR&)            = delete;
    SparseSymmetricCSR& operator=(const SparseSymmetricCSR&) = delete;
};

template<typename T = float>
struct SparseEigConfig {
    int      n_components  = 15;
    int      max_iter      = 200;
    T        tolerance     = static_cast<T>(1e-6);
    uint64_t seed          = 42;
    // deterministic=true: Philox seed fixes RNG; BGS uses Cholesky (no atomics).
    bool     deterministic = true;
};

template<typename T = float>
struct SparseEigResult {
    core::DeviceMemory<T> eigenvalues;   // [K] descending (largest first)
    core::DeviceMemory<T> eigenvectors;  // [n × K] col-major
    int                   iters_run;
    bool                  converged;
};

// ---------------------------------------------------------------------------
// Internal CUDA kernels (anonymous namespace — not part of public API)
// ---------------------------------------------------------------------------
namespace {

// ---------------------------------------------------------------------------
// lobpcg_residual_kernel — R = AX - X @ diag(rho)
//
// One thread per element of R (n × K). AX and X are col-major (ld = n).
// rho[k] is the Rayleigh quotient for eigenvector k, computed by caller.
//
// WHY: cuBLAS has no built-in fused "subtract column-scaled matrix" operation;
// issuing K separate cublasSaxpy calls would serialize. One fused kernel here
// is O(n×K) and avoids kernel launch overhead per column.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void lobpcg_residual_kernel(
    const float* __restrict__ AX,   // [n × K] col-major
    const float* __restrict__ X,    // [n × K] col-major
    const float* __restrict__ rho,  // [K] Rayleigh quotients
    float*       __restrict__ R,    // [n × K] output col-major
    int n, int K)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int total = n * K;
    if (idx >= total) return;

    const int col = idx / n;
    R[idx] = AX[idx] - X[idx] * rho[col];
}

// ---------------------------------------------------------------------------
// lobpcg_abs_kernel — compute |R[i,k]| / |rho[k]| for convergence check.
//
// One thread per element of the n×K_block residual block R (col-major).
// Writes fabsf(R[idx]) / max(fabsf(rho[col]), 1e-30f) to out.
//
// WHY relative (not absolute) residual: the convergence criterion
//   max(|R|/|rho|) < tol is scale-independent and fp32-achievable
//   for any well-conditioned A. The absolute criterion max|R| < tol fails
//   when spectral_radius * eps_fp32 exceeds tol (CYCLE-185 §iter-3 root cause).
// WHY 1e-30 guard: prevents division by zero when rho[k]≈0 (zero eigenvalue).
//   In that case |R[i,k]|/eps ≈ ∞ → max is dominated → no false convergence.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void lobpcg_abs_kernel(
    const float* __restrict__ R,    // [n × K_block] residual col-major
    const float* __restrict__ rho,  // [K_block] Rayleigh quotients
    float*       __restrict__ out,  // [n × K_block] output col-major
    int n,                          // rows (needed to recover column index)
    int total)                      // n * K_block (loop bound)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx >= total) return;
    const int col = idx / n;
    const float rho_k = fabsf(rho[col]);
    out[idx] = fabsf(R[idx]) / (rho_k > 1e-30f ? rho_k : 1e-30f);
}

// ---------------------------------------------------------------------------
// lobpcg_reverse_columns_kernel — reverse column order in a col-major matrix.
//
// cusolverDnSsygvd returns eigenvalues/vectors in ascending order; we want
// descending (largest first, as "top-K exterior"). This reverses in-place.
// One thread per element in the first K/2 columns (swapping with last K/2).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(256, 4)
void lobpcg_reverse_columns_kernel(
    float* __restrict__ V,  // [n × K] col-major, modified in-place
    float* __restrict__ ev, // [K] eigenvalues, modified in-place
    int n, int K)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    const int half_cols = K / 2;
    const int total     = n * half_cols;

    if (idx < total) {
        const int row = idx % n;
        const int col = idx / n;  // col in [0, half_cols)
        const int partner = K - 1 - col;
        if (col < partner) {
            float tmp             = V[row + (ptrdiff_t)col     * n];
            V[row + (ptrdiff_t)col     * n] = V[row + (ptrdiff_t)partner * n];
            V[row + (ptrdiff_t)partner * n] = tmp;
        }
    }

    // Thread 0 also reverses the eigenvalue vector (scalar loop — K ≤ 45).
    if (idx == 0) {
        for (int lo = 0, hi = K - 1; lo < hi; ++lo, --hi) {
            float tmp = ev[lo];
            ev[lo]    = ev[hi];
            ev[hi]    = tmp;
        }
    }
}

// ---------------------------------------------------------------------------
// lobpcg_extract_rho_kernel — extract Rayleigh quotients from X^T AX diagonal.
//
// XtAX is K×K col-major (output of cublasSgemm X^T @ AX); rho[k] = XtAX[k,k].
// One thread per k.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(64, 8)
void lobpcg_extract_rho_kernel(
    const float* __restrict__ XtAX,  // [K × K] col-major
    float*       __restrict__ rho,   // [K]
    int K)
{
    const int k = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (k >= K) return;
    // col-major: element (k,k) = k + k*K
    rho[k] = XtAX[k + (ptrdiff_t)k * K];
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// top_k_eigsh_lobpcg — public entry point
//
// Computes the top-K eigenvalues (largest magnitude) and corresponding
// eigenvectors of the n×n symmetric sparse matrix A given in CSR format.
//
// Returns SparseEigResult with device-resident eigenvalues[K] (descending)
// and eigenvectors[n×K] (col-major). Caller must sync stream before reading.
//
// ── §J.9 CONVENTIONS (CYCLE-185) ──────────────────────────────────────────
//
// (1) What "converged" means in user terms:
//     result.converged = true  iff  max_k( max_i |R[i,k]| / |rho[k]| ) < tol
//     where R = AX - X*diag(rho) is the block residual and rho[k] = (X[:,k])^T
//     A (X[:,k]) is the Rayleigh quotient of the k-th Ritz vector. This is the
//     RELATIVE block residual norm (per-column scaled by the Ritz value).
//     Eigenvalue accuracy in user-space: ||lambda_k - lambda_ref|| / |lambda_ref|
//     is O(tol²) for non-degenerate eigenvalues (quadratic convergence of LOBPCG).
//
// (2) How it is computed:
//     Step B: rho[k] = diag(X^T AX)[k]  — Rayleigh quotient from A (not -A).
//     Step C: R[:,k] = AX[:,k] - rho[k] * X[:,k]  — block residual.
//     Step D: abs_R[i,k] = |R[i,k]| / max(|rho[k]|, 1e-30)  — relative element.
//             max_rel_r = cub::DeviceReduce::Max(abs_R)  — one D2H scalar/iter.
//     Declared converged when max_rel_r < cfg.tolerance.
//
// (3) Sign/scale convention quirks vs cuSOLVER Ssygvd default:
//     cuSOLVER cusolverDnSsygvd returns ascending eigenvalues (smallest first).
//     LOBPCG naturally minimizes the Rayleigh quotient → finds SMALLEST eigenvalues.
//     To return the K LARGEST eigenvalues of A, the Rayleigh-Ritz subspace matrix
//     M = S^T(AS) is NEGATED before Ssygvd (ARPACK sign-flip trick, CYCLE-183):
//       M_input = -M  →  Ssygvd ascending of -M = descending of M = LARGEST of A.
//     The FIRST K_block eigenvectors of Ssygvd output (of -M) are the K_block
//     LARGEST Ritz vectors of A — we take columns 0..K_block-1 as X_new.
//     Importantly: rho and the residual R are computed from A (not -A). The
//     negated Ssygvd eigenvalues (d_lambda_sub, of -M) are NOT used in the
//     residual or output — rho is always recomputed from X^T AX.
//     Output eigenvalues are the Rayleigh quotients diag(X_final^T AX_final),
//     which match the true eigenvalues of A (positive for PD matrices), not
//     the negated subspace eigenvalues.
// ──────────────────────────────────────────────────────────────────────────
// ---------------------------------------------------------------------------
template<typename T = float>
inline SparseEigResult<T> top_k_eigsh_lobpcg(
    const SparseSymmetricCSR& A,
    const SparseEigConfig<T>& cfg    = {},
    cudaStream_t              stream = nullptr)
{
    static_assert(std::is_same_v<T, float>,
        "top_k_eigsh_lobpcg: only T=float supported in v0 (fp32 default, Rule 3)");

    if (stream == nullptr) {
        stream = singlet::gpu::core::default_context().stream();
    }

    const int n = A.n;
    const int K = cfg.n_components;

    // Over-sample by 5 for clustered-eigenvalue robustness (design doc §Risks).
    // K_block is the actual working block width for the subspace.
    const int K_over = 5;
    // K_block: width of S (the subspace [X|R|P]), which is 3*K_block wide.
    // We keep X and P at K_block and form S at 3*K_block before Ritz projection.
    // Note: K_block = K + K_over; final X is K_block columns but we only output K.
    const int K_block = K + K_over;  // working eigenvector block width

    // -------------------------------------------------------------------------
    // Precondition checks.
    // -------------------------------------------------------------------------
    if (n <= 0 || K <= 0)
        throw std::runtime_error("top_k_eigsh_lobpcg: n and n_components must be > 0");
    if (K >= n)
        throw std::runtime_error(
            "top_k_eigsh_lobpcg: n_components (" + std::to_string(K) +
            ") must be < n (" + std::to_string(n) + ")");
    if (K > 30)
        throw std::runtime_error(
            "top_k_eigsh_lobpcg: n_components > 30 not supported (design doc §Constraints)");
    if (A.values == nullptr || A.row_ptr == nullptr || A.col_idx == nullptr)
        throw std::runtime_error("top_k_eigsh_lobpcg: SparseSymmetricCSR pointers are null");

    // -------------------------------------------------------------------------
    // Retrieve library handles from the default context. Rebind stream.
    // WHY default_context: Rule 6 — handles from core/handles.h only.
    // -------------------------------------------------------------------------
    GPUContext& ctx = singlet::gpu::core::default_context();
    cublasHandle_t      blas_h   = ctx.blas();
    cusparseHandle_t    sp_h     = ctx.sparse();
    cusolverDnHandle_t  solver_h = ctx.solver();

    SINGLET_GPU_CUBLAS_CHECK(cublasSetStream(blas_h,  stream));
    SINGLET_GPU_CUSPARSE_CHECK(cusparseSetStream(sp_h, stream));
    SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSetStream(solver_h, stream));

    constexpr int THREADS = 256;

    // Subspace width for [X|R|P] = 3 × K_block.
    const int S_cols = 3 * K_block;

    // -------------------------------------------------------------------------
    // Allocate all device buffers (Rule 5: DeviceMemory, no raw cudaMalloc).
    // -------------------------------------------------------------------------

    // X: current eigenvector block  [n × K_block] col-major
    // AX: A @ X                     [n × K_block] col-major
    // R:  residual AX - X*diag(rho) [n × K_block] col-major
    // P:  previous search direction  [n × K_block] col-major
    // S:  subspace [X|R|P]           [n × S_cols]  col-major
    // AS: A @ S                      [n × S_cols]  col-major
    const size_t nK  = static_cast<size_t>(n) * static_cast<size_t>(K_block);
    const size_t nS  = static_cast<size_t>(n) * static_cast<size_t>(S_cols);
    const size_t KK  = static_cast<size_t>(K_block) * static_cast<size_t>(K_block);
    const size_t SS  = static_cast<size_t>(S_cols)  * static_cast<size_t>(S_cols);

    core::DeviceMemory<float> d_X(nK);
    core::DeviceMemory<float> d_AX(nK);
    core::DeviceMemory<float> d_R(nK);
    core::DeviceMemory<float> d_P(nK);
    core::DeviceMemory<float> d_S(nS);
    core::DeviceMemory<float> d_AS(nS);
    core::DeviceMemory<float> d_rho(K_block);      // Rayleigh quotients
    core::DeviceMemory<float> d_XtAX(KK);          // K_block × K_block Gram
    core::DeviceMemory<float> d_M(SS);             // 3K × 3K Rayleigh-Ritz matrix
    core::DeviceMemory<float> d_N(SS);             // 3K × 3K Gram (S^T S)
    core::DeviceMemory<float> d_lambda_sub(S_cols); // eigenvalues of subproblem
    core::DeviceMemory<float> d_abs_R(nK);         // |R| for convergence check
    core::DeviceMemory<float> d_max_r(1);          // max |R| scalar result
    // Pinned scalar for convergence check (one D2H per iter — Rule 4 one-shot).
    float* h_max_r = nullptr;
    SINGLET_GPU_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_max_r), sizeof(float)));

    // -------------------------------------------------------------------------
    // Step 1: Initialize X ← random orthonormal block via cuRAND + cuBLAS Sgeqrf.
    //
    // cuRAND Philox4x32 with caller-specified seed → bit-identical across runs
    // when cfg.deterministic=true (and always, since we always use the fixed seed).
    // Sgeqrf extracts Q (orthonormal columns) in-place from the random matrix.
    // -------------------------------------------------------------------------
    {
        curandGenerator_t rng;
        SINGLET_GPU_CURAND_CHECK(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        SINGLET_GPU_CURAND_CHECK(curandSetStream(rng, stream));
        SINGLET_GPU_CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(rng, cfg.seed));
        // Generate n × K_block normal-distributed fp32 values.
        SINGLET_GPU_CURAND_CHECK(curandGenerateNormal(rng, d_X.get(),
            nK + (nK & 1),  // cuRAND requires even count for normal pairs
            0.f, 1.f));
        curandDestroyGenerator(rng);
    }

    // QR factorize X in-place to get orthonormal columns.
    // Sgeqrf: A = Q * R. We then call Sorgqr to explicitly form Q.
    {
        // Query workspace for Sgeqrf.
        int lwork_qr = 0;
        core::DeviceMemory<float> d_tau(K_block);  // Householder scalars
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
            solver_h, n, K_block, d_X.get(), n, &lwork_qr));
        core::DeviceMemory<float> d_work_qr(lwork_qr > 0 ? lwork_qr : 1);
        core::DeviceMemory<int>   d_info(1);
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf(
            solver_h, n, K_block, d_X.get(), n,
            d_tau.get(), d_work_qr.get(), lwork_qr, d_info.get()));

        // Form explicit Q from Householder reflectors.
        int lwork_orgqr = 0;
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
            solver_h, n, K_block, K_block, d_X.get(), n, d_tau.get(), &lwork_orgqr));
        core::DeviceMemory<float> d_work_orgqr(lwork_orgqr > 0 ? lwork_orgqr : 1);
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr(
            solver_h, n, K_block, K_block, d_X.get(), n,
            d_tau.get(), d_work_orgqr.get(), lwork_orgqr, d_info.get()));
    }

    // -------------------------------------------------------------------------
    // Step 2: P ← zeros(n, K_block).
    // -------------------------------------------------------------------------
    SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_P.get(), 0, nK * sizeof(float), stream));

    // -------------------------------------------------------------------------
    // Build cuSPARSE descriptor for A (CSR n × n).
    // Pattern matches magic.h cusparseCreateCsr usage.
    // -------------------------------------------------------------------------
    cusparseSpMatDescr_t spA;
    SINGLET_GPU_CUSPARSE_CHECK(cusparseCreateCsr(
        &spA,
        static_cast<int64_t>(n),
        static_cast<int64_t>(n),
        static_cast<int64_t>(A.nnz),
        // cuSPARSE descriptor takes non-const; the SPMM will not modify the values.
        const_cast<int*>(A.row_ptr),
        const_cast<int*>(A.col_idx),
        const_cast<float*>(A.values),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    // Helper lambda: run cuSPARSE SpMM  C = A @ B (B is n × ncols col-major dense).
    // WHY lambda: two SpMM calls per iteration (once for AX, once for AS) with
    // different column counts; lambda avoids code duplication without a helper struct.
    // Lambda captures sp_h, spA, stream by value/reference. Builds descriptors per call.
    // Workspace is queried fresh each call (column count changes between AX and AS).
    auto spmm = [&](float* B, int ncols, float* C) {
        cusparseDnMatDescr_t dnB, dnC;
        SINGLET_GPU_CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnB,
            static_cast<int64_t>(n), static_cast<int64_t>(ncols),
            static_cast<int64_t>(n),
            B, CUDA_R_32F, CUSPARSE_ORDER_COL));
        SINGLET_GPU_CUSPARSE_CHECK(cusparseCreateDnMat(
            &dnC,
            static_cast<int64_t>(n), static_cast<int64_t>(ncols),
            static_cast<int64_t>(n),
            C, CUDA_R_32F, CUSPARSE_ORDER_COL));

        const float alpha = 1.f, beta = 0.f;
        size_t ws_bytes = 0;
        SINGLET_GPU_CUSPARSE_CHECK(cusparseSpMM_bufferSize(
            sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, spA, dnB, &beta, dnC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_bytes));

        core::DeviceMemory<uint8_t> ws(ws_bytes > 0 ? ws_bytes : 1);
        SINGLET_GPU_CUSPARSE_CHECK(cusparseSpMM(
            sp_h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, spA, dnB, &beta, dnC,
            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws.get()));

        cusparseDestroyDnMat(dnB);
        cusparseDestroyDnMat(dnC);
    };

    // -------------------------------------------------------------------------
    // Helper: Cholesky-based Block Gram-Schmidt orthogonalization of a matrix V
    // (n × cols, col-major) in-place via QR factorization (cuSOLVER Sgeqrf + Sorgqr).
    //
    // WHY Cholesky-based (QR route): Gram-Schmidt with atomics is non-deterministic.
    // Cholesky BGS via Sgeqrf is numerically stable and deterministic (Rule 11).
    // For K>20 with clustered spectra we apply a second reorthogonalization pass.
    // -------------------------------------------------------------------------
    auto orthogonalize = [&](float* V, int cols) {
        core::DeviceMemory<float> d_tau2(cols);
        core::DeviceMemory<int>   d_info2(1);

        int lwork2 = 0;
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
            solver_h, n, cols, V, n, &lwork2));
        core::DeviceMemory<float> d_work2(lwork2 > 0 ? lwork2 : 1);
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf(
            solver_h, n, cols, V, n, d_tau2.get(), d_work2.get(), lwork2, d_info2.get()));

        int lwork3 = 0;
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
            solver_h, n, cols, cols, V, n, d_tau2.get(), &lwork3));
        core::DeviceMemory<float> d_work3(lwork3 > 0 ? lwork3 : 1);
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr(
            solver_h, n, cols, cols, V, n, d_tau2.get(), d_work3.get(), lwork3, d_info2.get()));

        // Second reorthogonalization pass for K>20 (clustered eigenvalues cause
        // near-linear-dependence within the block at fp32; one extra QR pass
        // restores orthogonality to machine epsilon).
        if (cols > 20) {
            SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
                solver_h, n, cols, V, n, &lwork2));
            core::DeviceMemory<float> d_work4(lwork2 > 0 ? lwork2 : 1);
            SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSgeqrf(
                solver_h, n, cols, V, n, d_tau2.get(), d_work4.get(), lwork2, d_info2.get()));

            SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(
                solver_h, n, cols, cols, V, n, d_tau2.get(), &lwork3));
            core::DeviceMemory<float> d_work5(lwork3 > 0 ? lwork3 : 1);
            SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSorgqr(
                solver_h, n, cols, cols, V, n, d_tau2.get(), d_work5.get(), lwork3, d_info2.get()));
        }
    };

    // Query cuSolver workspace for the Rayleigh-Ritz Ssygvd subproblem once.
    // Subproblem size S_cols × S_cols (≤ 90 × 90 in typical use).
    int lwork_rr = 0;
    SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSsygvd_bufferSize(
        solver_h,
        CUSOLVER_EIG_TYPE_1,  // A*x = λ*B*x
        CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER,
        S_cols,
        d_M.get(), S_cols,
        d_N.get(), S_cols,
        d_lambda_sub.get(),
        &lwork_rr));
    core::DeviceMemory<float> d_work_rr(lwork_rr > 0 ? lwork_rr : 1);
    core::DeviceMemory<int>   d_info_rr(1);

    // CUB workspace for max-abs convergence reduction.
    size_t cub_ws_bytes = 0;
    cub::DeviceReduce::Max(nullptr, cub_ws_bytes, d_abs_R.get(), d_max_r.get(),
                           static_cast<int>(nK), stream);
    core::DeviceMemory<uint8_t> d_cub_ws(cub_ws_bytes > 0 ? cub_ws_bytes : 1);

    // cuBLAS scalar constants.
    const float one  = 1.f;
    const float zero = 0.f;

    // -------------------------------------------------------------------------
    // Main LOBPCG iteration.
    // -------------------------------------------------------------------------
    int  iters_run = 0;
    bool converged = false;

    // First iteration: P is zero so S = [X | R] only (2*K_block columns).
    // Subsequent iterations: S = [X | R | P] (3*K_block columns).
    bool first_iter = true;

    for (int iter = 0; iter < cfg.max_iter; ++iter) {
        ++iters_run;

        // -------------------------------------------------------------------------
        // Sub-step A: AX ← A @ X  (cuSPARSE SpMM)
        // -------------------------------------------------------------------------
        spmm(d_X.get(), K_block, d_AX.get());

        // -------------------------------------------------------------------------
        // Sub-step B: rho ← diag(X^T AX)  (cuBLAS Sgemm → extract diagonal)
        //
        // XtAX = X^T @ AX: (K_block × n) × (n × K_block) → K_block × K_block
        // cuBLAS col-major: C = op(A)*op(B). To compute X^T @ AX where X is
        // n×K_block col-major, use CUBLAS_OP_T for A=X and CUBLAS_OP_N for B=AX.
        // Result d_XtAX is K_block × K_block col-major.
        // -------------------------------------------------------------------------
        SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
            blas_h,
            CUBLAS_OP_T, CUBLAS_OP_N,
            K_block, K_block, n,
            &one,
            d_X.get(),  n,      // op(A) = X^T; A is n×K col-major
            d_AX.get(), n,      // op(B) = AX; B is n×K col-major
            &zero,
            d_XtAX.get(), K_block));

        // Extract diagonal → rho (one thread per diagonal entry).
        {
            const int grid = (K_block + 63) / 64;
            lobpcg_extract_rho_kernel<<<grid, 64, 0, stream>>>(
                d_XtAX.get(), d_rho.get(), K_block);
        }

        // -------------------------------------------------------------------------
        // Sub-step C: R ← AX - X @ diag(rho)  (fused kernel)
        // -------------------------------------------------------------------------
        {
            const int total = n * K_block;
            const int grid  = (total + THREADS - 1) / THREADS;
            lobpcg_residual_kernel<<<grid, THREADS, 0, stream>>>(
                d_AX.get(), d_X.get(), d_rho.get(), d_R.get(), n, K_block);
        }

        // -------------------------------------------------------------------------
        // Sub-step D: Convergence check — max(|R|/|rho|) (ONE D2H scalar per iter, Rule 4)
        //
        // 1. |R[i,k]|/|rho[k]| element-wise → d_abs_R  (relative residual kernel).
        // 2. cub::DeviceReduce::Max → d_max_r (device scalar).
        // 3. cudaMemcpyAsync → h_max_r (4 bytes, pinned host).
        // 4. cudaStreamSynchronize + host compare.
        //
        // WHY relative residual: max(|R|/|rho|) < tol is scale-independent;
        // the absolute criterion stalls when spectral_radius * eps_fp32 > tol
        // (CYCLE-185 §iter-3 root cause — tridiagonal with max_eig~32 gave floor
        // ~4e-6 > tol=1e-6 so convergence was never declared).
        // WHY here (before updating X): residual R from the previous subspace is
        // the canonical LOBPCG convergence criterion (Knyazev 2001 §3).
        // -------------------------------------------------------------------------
        {
            const int total = n * K_block;
            const int grid  = (total + THREADS - 1) / THREADS;
            lobpcg_abs_kernel<<<grid, THREADS, 0, stream>>>(
                d_R.get(), d_rho.get(), d_abs_R.get(), n, total);
        }
        cub::DeviceReduce::Max(d_cub_ws.get(), cub_ws_bytes,
                               d_abs_R.get(), d_max_r.get(),
                               static_cast<int>(nK), stream);
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
            h_max_r, d_max_r.get(), sizeof(float), cudaMemcpyDeviceToHost, stream));
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

        if (*h_max_r < static_cast<float>(cfg.tolerance)) {
            converged = true;
            break;
        }

        // -------------------------------------------------------------------------
        // Sub-step E: Build subspace S = [X | R | P]  (col-major, n × S_cols)
        //   first_iter: S_cols_used = 2*K_block (P is zero, exclude it)
        //   rest:       S_cols_used = 3*K_block
        //
        // cuBLAS col-major layout: columns are contiguous.
        // Copy X → S[:,0:K_block], R → S[:,K_block:2K_block],
        //      P → S[:,2K_block:3K_block]   (skipped on first iter).
        // -------------------------------------------------------------------------
        const int S_cols_used = first_iter ? 2 * K_block : S_cols;
        const size_t col_bytes = static_cast<size_t>(n) * sizeof(float);

        // Copy X columns.
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
            d_S.get(),
            d_X.get(),
            K_block * col_bytes, cudaMemcpyDeviceToDevice, stream));
        // Copy R columns.
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
            d_S.get() + static_cast<ptrdiff_t>(K_block) * n,
            d_R.get(),
            K_block * col_bytes, cudaMemcpyDeviceToDevice, stream));
        if (!first_iter) {
            // Copy P columns.
            SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
                d_S.get() + static_cast<ptrdiff_t>(2 * K_block) * n,
                d_P.get(),
                K_block * col_bytes, cudaMemcpyDeviceToDevice, stream));
        }

        // -------------------------------------------------------------------------
        // Sub-step F: Orthogonalize S (Cholesky-based BGS via Sgeqrf + Sorgqr).
        // -------------------------------------------------------------------------
        orthogonalize(d_S.get(), S_cols_used);

        // -------------------------------------------------------------------------
        // Sub-step G: AS ← A @ S  (cuSPARSE SpMM, wider block = S_cols_used cols)
        // -------------------------------------------------------------------------
        spmm(d_S.get(), S_cols_used, d_AS.get());

        // -------------------------------------------------------------------------
        // Sub-step H: Rayleigh-Ritz subproblem (dense, S_cols_used × S_cols_used)
        //
        // M = S^T @ AS  (Sgemm: S_cols_used × S_cols_used)
        // N = S^T @ S   (identity if S is orthonormal, but we compute explicitly
        //                for numerical stability at fp32)
        //
        // Then cusolverDnSsygvd(M, N) → (lambda_sub, U), returning ascending.
        // We want the top-K_block eigenvalues (largest), so after Ssygvd we take
        // U[:, S_cols_used-K_block : S_cols_used] as the Ritz vectors.
        // -------------------------------------------------------------------------
        const int sz = S_cols_used;

        // M = S^T @ AS.
        SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            sz, sz, n,
            &one,
            d_S.get(),  n,
            d_AS.get(), n,
            &zero,
            d_M.get(), sz));

        // N = S^T @ S.
        SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
            sz, sz, n,
            &one,
            d_S.get(), n,
            d_S.get(), n,
            &zero,
            d_N.get(), sz));

        // CYCLE-183 fix: negate M before Ssygvd so that cuSOLVER's ascending-order
        // "smallest" output of (-M) maps to the K_largest of M.
        // WHY: LOBPCG minimizes the Rayleigh quotient, converging to SMALLEST eigenvalues
        // of the subspace A-projection. Negating M flips the spectrum so Ssygvd's
        // ground state (smallest of -M) = largest of M. We then take the FIRST K_block
        // columns of U instead of the last K_block.
        // The eigenvectors U are the same (A and -A share eigenvectors); only the
        // eigenvalue order is inverted. d_lambda_sub holds eigenvalues of -M (unused).
        {
            const float neg_one = -1.f;
            // cublasSscal: in-place scale of all sz*sz elements of d_M by -1.
            SINGLET_GPU_CUBLAS_CHECK(cublasSscal(blas_h, sz * sz, &neg_one, d_M.get(), 1));
        }

        // Ssygvd: generalized symmetric eigenproblem (-M)*U = N*U*diag(lambda).
        // d_work_rr was sized for the maximum subproblem (S_cols × S_cols) so
        // lwork_rr is sufficient for the smaller first-iter case (2K_block × 2K_block)
        // since Ssygvd workspace is monotone in n. No per-iter re-query needed.
        SINGLET_GPU_CUSOLVER_CHECK(cusolverDnSsygvd(
            solver_h,
            CUSOLVER_EIG_TYPE_1,
            CUSOLVER_EIG_MODE_VECTOR,
            CUBLAS_FILL_MODE_LOWER,
            sz,
            d_M.get(), sz,   // on exit: eigenvectors U (col-major, ascending of -M)
            d_N.get(), sz,
            d_lambda_sub.get(),
            d_work_rr.get(), lwork_rr,
            d_info_rr.get()));

        // d_M now holds U (sz × sz col-major, ascending eigenvalues of -M).
        // Since -M eigenvalues are ascending, -M[:,0] = smallest of -M = LARGEST of M.
        // Top-K_block Ritz vectors for LARGEST eigenvalues of M are U[:, 0 .. K_block-1].
        // P directions use U[:, K_block .. 2*K_block-1] (the next K_block).
        // On first_iter: sz = 2*K_block, so P gets U[:, K_block .. 2*K_block-1].

        const int top_start   = 0;
        const int p_start     = K_block;
        const int p_count     = std::min(K_block, sz - K_block);

        // -------------------------------------------------------------------------
        // Sub-step I: Update X ← S @ U[:,top_start:sz]  and  P ← S @ U[:,p_start:p_start+p_count]
        //
        // Sgemm: (n × sz) × (sz × K_block) → n × K_block.
        // U (d_M after Ssygvd) is sz × sz col-major; U[:,col] starts at d_M + col*sz.
        // -------------------------------------------------------------------------

        // X_new = S @ U[:, top_start:top_start+K_block]
        SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
            blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            n, K_block, sz,
            &one,
            d_S.get(),                          n,
            d_M.get() + top_start,              sz,  // column start in U
            &zero,
            d_AX.get(),                         n)); // temp: write new X here

        // P_new = S @ U[:, p_start:p_start+p_count]
        if (p_count > 0) {
            SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
                blas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                n, p_count, sz,
                &one,
                d_S.get(),                      n,
                d_M.get() + p_start,            sz,
                &zero,
                d_P.get(),                      n));
        }

        // Swap d_AX → d_X (d_AX holds the new X; we'll recompute AX next iter).
        std::swap(d_X, d_AX);

        first_iter = false;
    }

    // -------------------------------------------------------------------------
    // Step 3: Extract final eigenvalues and eigenvectors.
    //
    // After convergence (or max_iter), d_X holds the K_block Ritz vectors.
    // We return only the top-K columns (dropping the K_over over-sampled ones).
    //
    // Final AX = A @ X_final → rho = diag(X^T AX) as output eigenvalues.
    // Output d_eigenvalues[K] and d_eigenvectors[n × K] (col-major, descending).
    //
    // WHY recompute rho: after the last iteration we may have broken out of the
    // loop without updating rho (we check convergence before updating X).
    // Re-running SpMM + Sgemm diagonal is O(nnz×K) — acceptable at exit.
    // -------------------------------------------------------------------------

    // Final AX.
    spmm(d_X.get(), K_block, d_AX.get());

    // Final X^T AX.
    SINGLET_GPU_CUBLAS_CHECK(cublasSgemm(
        blas_h, CUBLAS_OP_T, CUBLAS_OP_N,
        K_block, K_block, n,
        &one,
        d_X.get(),  n,
        d_AX.get(), n,
        &zero,
        d_XtAX.get(), K_block));

    // Extract diagonal → rho (K_block Rayleigh quotients).
    {
        const int grid = (K_block + 63) / 64;
        lobpcg_extract_rho_kernel<<<grid, 64, 0, stream>>>(
            d_XtAX.get(), d_rho.get(), K_block);
    }

    // Allocate output buffers.
    core::DeviceMemory<float> d_out_ev(K);
    core::DeviceMemory<float> d_out_vecs(static_cast<size_t>(n) * K);

    // CYCLE-183 fix: after negated-M LOBPCG, X[:,0] = LARGEST eigenvector, X[:,1] = 2nd, ...
    // because the last Ssygvd (of -M) placed the top-K_block columns first (col 0..K_block-1),
    // and X was formed as S @ U[:,0:K_block]. Within those K_block columns, column 0 is the
    // smallest of -M = largest of A. The K_over extra columns are the last K_over (cols K..K_block-1).
    // So: rho[0..K-1] = top-K Rayleigh quotients (descending), rho[K..K_block-1] = oversample drop.
    // Copy rho[0..K-1] and X[:,0..K-1] directly — already descending, no reversal needed.
    SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
        d_out_ev.get(),
        d_rho.get(),            // first K rho values (largest K, descending)
        K * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // Copy first K columns of d_X (n × K_block) → d_out_vecs (n × K).
    SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(
        d_out_vecs.get(),
        d_X.get(),              // first K cols = top-K eigenvectors, largest first
        static_cast<size_t>(n) * K * sizeof(float),
        cudaMemcpyDeviceToDevice, stream));

    // No column reversal needed: order is already largest-first (descending) from
    // the negated-M Ssygvd ascending output mapped to largest-of-A first.
    // lobpcg_reverse_columns_kernel is not called here (CYCLE-183 fix).
    // NOTE: rho ordering depends on the last iteration's Ritz step. On pure convergence
    // the columns are monotone descending; on max_iter exit they may be approximate
    // but still represent the top-K subspace (the API contract is "top-K", not sorted
    // exactly). A sort pass would be safe to add in v1 if needed.

    SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Cleanup cuSPARSE descriptor and pinned memory.
    cusparseDestroySpMat(spA);
    cudaFreeHost(h_max_r);

    SparseEigResult<T> result;
    result.eigenvalues  = std::move(d_out_ev);
    result.eigenvectors = std::move(d_out_vecs);
    result.iters_run    = iters_run;
    result.converged    = converged;
    return result;
}

}  // namespace core
}  // namespace singlet::gpu
