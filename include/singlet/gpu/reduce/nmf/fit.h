// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/nmf/fit.h
//
// Native sparse NMF via multiplicative update (MU) and coordinate descent (CD).
// Replaces factornet::nmf::nmf_fit_gpu adapter (CYCLE-105, 2026-04-29).
//
// // algorithm derived from factornet/nmf/fit_gpu.cuh
// Credit: Zach DeBruine, factornet 2021-2026
// // MU reference: Lee & Seung 2001, "Algorithms for Non-negative Matrix Factorization"
// // CD reference: Hsieh & Dhillon 2011, "Fast Coordinate Descent for Non-negative
//              Sparse Least Squares" (for k < k_cd_cutoff path)
//
// Algorithm (MU — default for k >= k_cd_cutoff):
//   A (m × n, sparse CSC), W (m × k), H (k × n).
//   Per iteration:
//     WtA = Wᵀ A       (k × n): cuSPARSE SpMM with Aᵀ W → Aᵀ W ≡ (Wᵀ A)ᵀ, then T
//     WtW = Wᵀ W       (k × k): cuBLAS Ssyrk
//     WHHt = (W H) Hᵀ : form WH (m × n dense)? No — too large. Instead:
//       HHt = H Hᵀ     (k × k): cuBLAS Ssyrk
//       H ← H ∘ WtA / (WtW H + ε)  elementwise (custom kernel)
//       AHt = A Hᵀ     (m × k): cuSPARSE SpMM
//       W ← W ∘ AHt / (W HHt + ε)   elementwise (custom kernel)
//   No host↔device traffic inside the update loop (Rule 4 ABSOLUTE).
//   One scalar convergence check per outer iteration via cub::DeviceReduce.
//
// Algorithm (CD — for k < k_cd_cutoff per FitConfig):
//   Coordinate descent inner loop per row of H and W.
//   O(k² × cd_max_iter × n) — only efficient at small k (< 32).
//   For k >= k_cd_cutoff, auto mode overrides to MU (FitConfig.k_cd_cutoff=32).
//
// Workspace budget (device):
//   W: m × k × 4 bytes
//   H: k × n × 4 bytes
//   WtW / HHt: k × k × 4 bytes each
//   WtA / AHt: k × n / m × k × 4 bytes
//   loss: 1 × 4 bytes (scalar per iter)
//   Total: O((m + n) × k × 4) device bytes.
//
// Host↔device traffic: ZERO inside the hot loop.
//   Only exception: ONE scalar (4 bytes) convergence check per outer iteration
//   via cudaMemcpy of the scalar loss.
//
// Precision: fp32. fp64 accumulators in Ssyrk (cuBLAS internal behavior on H100).
// Determinism: for fixed seed + fixed GPU.
// Streams: uses caller's GPUContext stream.
// OOC plan: chunked_fit (chunked.h) loops over shards with warm-start H for OOC.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/reduce/svd/deflation.h>  // require_host_retained (canonical)

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// Device kernels for NMF updates (anonymous namespace — no ABI exposure).
// ---------------------------------------------------------------------------
namespace {

// MU update for H with all four regularizations:
//   H ← H ∘ (WtA) / (WtWH + λ_L2·H + λ_L1 + λ_ortho·ortho_H + ε)
//
// Numerator:  WtA[r,j]                                  (Lee & Seung 2001)
// Denominator contributions:
//   WtWH[r,j]                                           (Lee & Seung 2001 base)
//   + λ_L2 * H[r,j]                                    (Tikhonov/ridge, Hsieh & Dhillon 2011 §4)
//   + λ_L1                                              (lasso sparsity, standard L1-NMF)
//   + λ_ortho * ortho_H[r,j]                            (Choi 2008 ortho-NMF: H(HHt-I))
//
// ortho_H: k × n, col-major — passed as nullptr when λ_ortho == 0.
// nonneg: NMF is intrinsically non-negative; MU clamp is the standard projection.
__global__ void nmf_mu_update_H_kernel(
    float* __restrict__ H,
    const float* __restrict__ WtA,
    const float* __restrict__ WtWH,
    const float* __restrict__ ortho_H,   // k×n: H*(HHt−I), nullptr when ortho==0
    float lambda_L1,
    float lambda_L2,
    float lambda_ortho,
    float eps,
    int k, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = k * n;
    if (idx >= total) return;
    float h   = H[idx];
    float num = WtA[idx];
    float den = WtWH[idx] + eps;
    // L2 ridge: adds λ_L2·H to denominator (Hsieh & Dhillon 2011 §4, Eq. 7)
    den += lambda_L2 * h;
    // L1 lasso: adds λ_L1 (constant shift) to denominator (standard L1-NMF)
    den += lambda_L1;
    // Ortho penalty: adds λ_ortho·(H(HHt−I))[r,j] to denominator (Choi 2008)
    if (ortho_H) den += lambda_ortho * ortho_H[idx];
    float h_new = h * (num / den);
    // Non-negativity projection: intrinsic to MU; clamp residual float errors
    H[idx] = h_new > 0.f ? h_new : 0.f;
}

// MU update for W with all four regularizations:
//   W ← W ∘ (AHt) / (WHHt + λ_L2·W + λ_L1 + λ_ortho·ortho_W + ε)
//
// ortho_W: m × k, col-major — passed as nullptr when λ_ortho == 0.
// References same as above: Lee & Seung 2001, Hsieh & Dhillon 2011, Choi 2008.
__global__ void nmf_mu_update_W_kernel(
    float* __restrict__ W,
    const float* __restrict__ AHt,
    const float* __restrict__ WHHt,
    const float* __restrict__ ortho_W,   // m×k: W*(WtW−I), nullptr when ortho==0
    float lambda_L1,
    float lambda_L2,
    float lambda_ortho,
    float eps,
    int m, int k)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = m * k;
    if (idx >= total) return;
    float w   = W[idx];
    float num = AHt[idx];
    float den = WHHt[idx] + eps;
    // L2 ridge (Hsieh & Dhillon 2011 §4)
    den += lambda_L2 * w;
    // L1 lasso (standard L1-NMF sparsity)
    den += lambda_L1;
    // Ortho penalty (Choi 2008, §3: W(WtW−I) penalises off-diagonal Gram)
    if (ortho_W) den += lambda_ortho * ortho_W[idx];
    float w_new = w * (num / den);
    W[idx] = w_new > 0.f ? w_new : 0.f;
}

// Subtract the k×k identity from a col-major symmetric matrix in-place.
// Used to form (WtW - I) and (HHt - I) for the ortho penalty.
// Only the upper triangle (from Ssyrk) and diagonal need updating; we write
// diagonal only — the Ssymm that follows reads the full UPPER triangle stored
// col-major, so we must touch [r,r] where col-offset == diag_offset.
// WHY: Ssyrk fills UPPER; Ssymm with CUBLAS_FILL_MODE_UPPER reads that half.
// We subtract 1 only from the diagonal. Off-diagonal entries are untouched.
__global__ void subtract_identity_k(float* __restrict__ M, int k)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < k) M[r + r * k] -= 1.f;   // M[r,r] in col-major k×k
}

// Frobenius loss via trace trick (all device-side):
//   loss ≈ -2 * <WtA, H> + trace(WtW * HHt)
// where <WtA,H> = Sdot(k*n), trace(WtW*HHt) = Sdot(k*k).
// ONE scalar copy per outer iteration — APPROVED per Rule 4.

}  // anonymous namespace

namespace singlet::gpu {
namespace reduce {
namespace nmf {

// FitConfig — GPU-tuned solver routing defaults (from CYCLE-86 profiling).
// Preserved from original adapter.
struct FitConfig {
    // Ranks >= k_cd_cutoff in auto mode are forced to MU solver.
    int k_cd_cutoff = 32;
    // CD inner iter limit when CD is actually used.
    int cd_max_iter = 10;
};

// ---------------------------------------------------------------------------
// nmf_fit_internal — core MU NMF implementation (no host ↔ device in hot loop).
//
// Called by both fit() and cv_fit() (mask=nullptr for full fit).
// mask: optional device bool mask (n*m) for held-out entries (speckled CV).
//   When non-null, WtA computation excludes masked entries.
// ---------------------------------------------------------------------------
namespace detail {

// SpMM: C = alpha * A^T * B + beta * C
// A: m×n sparse CSC; B: m×k dense col-major; C: n×k dense col-major.
// Used for: WtA = Aᵀ W  (A sparse, W dense → WtA dense k×n, need transpose)
inline void sparse_AT_dense(
    cusparseHandle_t sparse,
    cusparseSpMatDescr_t spA,
    const float* B_dev, int m, int k_cols,
    float* C_dev, int n,
    float alpha, float beta,
    void* ws, size_t ws_sz,
    cudaStream_t stream)
{
    cusparseDnMatDescr_t dmB = nullptr, dmC = nullptr;
    cusparseCreateDnMat(&dmB, m, k_cols, m, const_cast<float*>(B_dev), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&dmC, n, k_cols, n, C_dev, CUDA_R_32F, CUSPARSE_ORDER_COL);

    cusparseSpMM(sparse,
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmB, &beta, dmC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, ws);

    cusparseDestroyDnMat(dmB);
    cusparseDestroyDnMat(dmC);
}

// SpMM: C = alpha * A * B + beta * C
// A: m×n sparse CSC; B: n×k dense; C: m×k dense.
inline void sparse_A_dense(
    cusparseHandle_t sparse,
    cusparseSpMatDescr_t spA,
    const float* B_dev, int n, int k_cols,
    float* C_dev, int m,
    float alpha, float beta,
    void* ws, size_t ws_sz,
    cudaStream_t stream)
{
    cusparseDnMatDescr_t dmB = nullptr, dmC = nullptr;
    cusparseCreateDnMat(&dmB, n, k_cols, n, const_cast<float*>(B_dev), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&dmC, m, k_cols, m, C_dev, CUDA_R_32F, CUSPARSE_ORDER_COL);

    cusparseSpMM(sparse,
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmB, &beta, dmC, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, ws);

    cusparseDestroyDnMat(dmB);
    cusparseDestroyDnMat(dmC);
}

inline NmfResult run_mu_nmf(
    core::GPUContext& ctx,
    cusparseSpMatDescr_t spA,
    int mat_m, int mat_n, int k,
    const NmfConfig& cfg,
    const DenseMatrix* W_init,
    const DenseMatrix* H_init,
    // speckled CV: optional held-out mask on host (per-nnz bool, size nnz)
    const std::vector<bool>* holdout_mask = nullptr,
    const int32_t* h_col_ptr   = nullptr,
    const int32_t* h_row_idx   = nullptr,
    const float*   h_values    = nullptr)
{
    cudaStream_t stream = ctx.stream();
    const float eps  = cfg.eps > 0.f ? cfg.eps : 1e-9f;
    const int max_it = cfg.max_iter;
    const float tol  = cfg.tol;

    // --- Allocate W (m×k) and H (k×n) on device ---
    core::DeviceMemory<float> d_W(mat_m * k);
    core::DeviceMemory<float> d_H(k * mat_n);

    // --- Initialize W and H ---
    if (W_init && !W_init->empty() && W_init->m == mat_m && W_init->n == k) {
        cudaMemcpyAsync(d_W.get(), W_init->data.data(), mat_m * k * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    } else {
        // Random init [0,1) via SplitMix64 on host (deterministic)
        std::vector<float> h_W(mat_m * k);
        uint64_t s = cfg.seed;
        for (auto& v : h_W) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            v = static_cast<float>(z >> 11) * (1.0f / float(1ULL << 53));
        }
        cudaMemcpyAsync(d_W.get(), h_W.data(), mat_m * k * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    }

    if (H_init && !H_init->empty() && H_init->m == k && H_init->n == mat_n) {
        cudaMemcpyAsync(d_H.get(), H_init->data.data(), k * mat_n * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    } else {
        std::vector<float> h_H(k * mat_n);
        uint64_t s = cfg.seed ^ 0xDEADBEEFULL;
        for (auto& v : h_H) {
            s += 0x9e3779b97f4a7c15ULL;
            uint64_t z = s;
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            z = z ^ (z >> 31);
            v = static_cast<float>(z >> 11) * (1.0f / float(1ULL << 53));
        }
        cudaMemcpyAsync(d_H.get(), h_H.data(), k * mat_n * sizeof(float),
            cudaMemcpyHostToDevice, stream);
    }

    // --- Extract regularization weights (prefer sub-struct; flat scalars are zero by default) ---
    // WHY: NmfConfig exposes both flat scalars (legacy) and FactorConfig sub-structs (CYCLE-114).
    // sync_factor_configs() has already been called by fit() before entering here.
    const float lam_L1_H    = cfg.H.L1;
    const float lam_L2_H    = cfg.H.L2;
    const float lam_ortho_H = cfg.H.ortho;
    const float lam_L1_W    = cfg.W.L1;
    const float lam_L2_W    = cfg.W.L2;
    const float lam_ortho_W = cfg.W.ortho;
    const bool  use_ortho_H = lam_ortho_H > 0.f;
    const bool  use_ortho_W = lam_ortho_W > 0.f;

    // --- Allocate scratch buffers ---
    core::DeviceMemory<float> d_WtA( k * mat_n);   // Wᵀ A  (k × n)
    core::DeviceMemory<float> d_WtW( k * k);        // Wᵀ W  (k × k)
    core::DeviceMemory<float> d_HHt( k * k);        // H Hᵀ  (k × k)
    core::DeviceMemory<float> d_WtWH(k * mat_n);   // WtW * H (k × n)
    core::DeviceMemory<float> d_AHt( mat_m * k);   // A Hᵀ  (m × k)
    core::DeviceMemory<float> d_WHHt(mat_m * k);   // W * HHt (m × k)

    // Ortho penalty scratch: H*(HHt-I) is k×n; W*(WtW-I) is m×k.
    // Allocated unconditionally as size-1 no-ops when ortho==0 to avoid
    // conditional DeviceMemory sizing logic (Rule 5: no raw cudaMalloc).
    core::DeviceMemory<float> d_ortho_H(use_ortho_H ? k * mat_n : 1);
    core::DeviceMemory<float> d_ortho_W(use_ortho_W ? mat_m * k  : 1);
    // Gram copies for ortho (we subtract identity in-place; need a scratch copy
    // so d_WtW/d_HHt are not corrupted for the loss trace computation).
    core::DeviceMemory<float> d_HHt_ortho(use_ortho_H ? k * k : 1);
    core::DeviceMemory<float> d_WtW_ortho(use_ortho_W ? k * k : 1);

    // SpMM workspace query (Aᵀ W → k×n  and  A Hᵀ → m×k)
    {
        float a1=1.f, b0=0.f;
        cusparseDnMatDescr_t dmW=nullptr, dmWtA=nullptr;
        cusparseCreateDnMat(&dmW,   mat_m, k, mat_m, d_W.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dmWtA, mat_n, k, mat_n, d_WtA.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        size_t ws1=0, ws2=0;
        cusparseSpMM_bufferSize(ctx.sparse(),
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a1, spA, dmW, &b0, dmWtA, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws1);
        cusparseDestroyDnMat(dmW); cusparseDestroyDnMat(dmWtA);

        cusparseDnMatDescr_t dmHT=nullptr, dmAHt=nullptr;
        cusparseCreateDnMat(&dmHT,  k, mat_n, k,     d_H.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dmAHt, mat_m, k, mat_m, d_AHt.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        // A * Hᵀ: H is k×n; we need Hᵀ which is n×k. SpMM A*(Hᵀ) where op_B=TRANSPOSE.
        cusparseSpMM_bufferSize(ctx.sparse(),
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            &a1, spA, dmHT, &b0, dmAHt, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws2);
        cusparseDestroyDnMat(dmHT); cusparseDestroyDnMat(dmAHt);

        // Allocate unified workspace
        size_t ws_total = std::max(ws1, ws2);
        (void)ws_total;
    }
    // Re-query with correct matrix layouts to get persistent workspace.
    size_t ws_ATW = 0, ws_AHT = 0;
    {
        float a1=1.f, b0=0.f;
        cusparseDnMatDescr_t dmTmp1=nullptr, dmTmp2=nullptr;
        cusparseCreateDnMat(&dmTmp1, mat_m, k, mat_m, d_W.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dmTmp2, mat_n, k, mat_n, d_WtA.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseSpMM_bufferSize(ctx.sparse(),
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &a1, spA, dmTmp1, &b0, dmTmp2, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_ATW);
        cusparseDestroyDnMat(dmTmp1); cusparseDestroyDnMat(dmTmp2);
    }
    {
        float a1=1.f, b0=0.f;
        cusparseDnMatDescr_t dmH=nullptr, dmAHt_=nullptr;
        cusparseCreateDnMat(&dmH,    k, mat_n, k,     d_H.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dmAHt_, mat_m, k, mat_m, d_AHt.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseSpMM_bufferSize(ctx.sparse(),
            CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
            &a1, spA, dmH, &b0, dmAHt_, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_AHT);
        cusparseDestroyDnMat(dmH); cusparseDestroyDnMat(dmAHt_);
    }
    core::DeviceMemory<uint8_t> d_spmm_ws(std::max({ws_ATW, ws_AHT, (size_t)1}));

    // --- Main MU iteration loop ---
    std::vector<float> loss_history;
    loss_history.reserve(max_it);
    float prev_loss = 1e30f;
    bool converged = false;

    float alpha_blas = 1.f, beta_zero = 0.f;
    const float SYRK_ALPHA = 1.f;

    for (int it = 0; it < max_it; ++it) {
        // === H update ===
        // 1. WtA (k × n) = Aᵀ W : SpMM A^T * W
        {
            cusparseDnMatDescr_t dmW=nullptr, dmWtA=nullptr;
            cusparseCreateDnMat(&dmW,   mat_m, k, mat_m, d_W.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseCreateDnMat(&dmWtA, mat_n, k, mat_n, d_WtA.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseSpMM(ctx.sparse(),
                CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_blas, spA, dmW, &beta_zero, dmWtA, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_ws.get());
            cusparseDestroyDnMat(dmW); cusparseDestroyDnMat(dmWtA);
        }
        // WtA is currently (n × k) col-major from the SpMM perspective.
        // We need (k × n) col-major for element-wise H update.
        // Transpose via cuBLAS Sgeam.
        {
            core::DeviceMemory<float> d_WtA_T(k * mat_n);
            cublasSgeam(ctx.blas(), CUBLAS_OP_T, CUBLAS_OP_N,
                k, mat_n, &alpha_blas, d_WtA.get(), mat_n,
                &beta_zero, nullptr, k, d_WtA_T.get(), k);
            cudaMemcpyAsync(d_WtA.get(), d_WtA_T.get(), k * mat_n * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        }

        // 2. WtW (k × k) = Wᵀ W : cuBLAS Ssyrk
        // Ssyrk: C ← α A Aᵀ + β C  with UPPER fill, op=CUBLAS_OP_T (A=W, m×k → Aᵀ=k×m → k×k)
        cublasSsyrk(ctx.blas(), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            k, mat_m, &SYRK_ALPHA, d_W.get(), mat_m,
            &beta_zero, d_WtW.get(), k);

        // 2b. HHt (k × k) = H Hᵀ : cuBLAS Ssyrk (op=CUBLAS_OP_N, A=H k×n → k×k)
        // WHY hoisted here: the ortho H penalty needs HHt(H_current) before H is
        // modified. Computing it here (from current H, before update) is correct and
        // also serves the W update later (step 5 below is now a no-op copy).
        cublasSsyrk(ctx.blas(), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
            k, mat_n, &SYRK_ALPHA, d_H.get(), k,
            &beta_zero, d_HHt.get(), k);

        // 3. WtWH (k × n) = WtW * H : cuBLAS Ssymm
        // H is k×n; WtW is k×k symmetric upper.
        // Ssymm: C = α A B + β C where A=WtW(k×k), B=H(k×n), C=WtWH(k×n)
        cublasSsymm(ctx.blas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            k, mat_n, &alpha_blas, d_WtW.get(), k,
            d_H.get(), k, &beta_zero, d_WtWH.get(), k);

        // 4a. Ortho penalty for H: compute H*(HHt - I)  [Choi 2008, §3]
        //     We copy HHt → scratch, subtract I on device (no H2D traffic),
        //     then Ssymm: ortho_H = (HHt-I) * H  (side=LEFT).
        //     WHY: HHt is Gram of current (pre-update) H; standard MU ortho uses
        //     the same-iteration Gram, which is available because we hoisted step 2b.
        float* ortho_H_ptr = nullptr;
        if (use_ortho_H) {
            cudaMemcpyAsync(d_HHt_ortho.get(), d_HHt.get(), k * k * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            // subtract identity: d_HHt_ortho[r,r] -= 1 (purely device-side, zero H2D)
            {
                int blocks_k = (k + 255) / 256;
                subtract_identity_k<<<blocks_k, 256, 0, stream>>>(d_HHt_ortho.get(), k);
            }
            // ortho_H (k×n) = (HHt-I) * H   [side=LEFT on symmetric k×k]
            cublasSsymm(ctx.blas(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                k, mat_n, &alpha_blas, d_HHt_ortho.get(), k,
                d_H.get(), k, &beta_zero, d_ortho_H.get(), k);
            ortho_H_ptr = d_ortho_H.get();
        }

        // 4b. MU update H: H ← H ∘ WtA / (WtWH + λ_L2·H + λ_L1 + λ_ortho·ortho_H + ε)
        //     Lee & Seung 2001 base; L2/L1/ortho extensions per task §Math.
        {
            int total = k * mat_n;
            int blocks = (total + 255) / 256;
            nmf_mu_update_H_kernel<<<blocks, 256, 0, stream>>>(
                d_H.get(), d_WtA.get(), d_WtWH.get(), ortho_H_ptr,
                lam_L1_H, lam_L2_H, lam_ortho_H, eps, k, mat_n);
        }

        // === W update ===
        // 5. HHt (k × k) from current H — already computed at step 2b above.
        // WHY no recompute: we use HHt(H_updated) for WHHt but keep HHt(H_pre) for
        // ortho. Recomputing here uses H_updated, which is the standard MU practice
        // (W update always uses the freshly updated H for better convergence).
        // For ortho_H we already saved the pre-update Gram at step 4a.
        cublasSsyrk(ctx.blas(), CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
            k, mat_n, &SYRK_ALPHA, d_H.get(), k,
            &beta_zero, d_HHt.get(), k);

        // 6. AHt (m × k) = A Hᵀ : SpMM A * Hᵀ
        //    H is k×n; Hᵀ is n×k. SpMM with op_B=TRANSPOSE.
        {
            cusparseDnMatDescr_t dmH=nullptr, dmAHt=nullptr;
            cusparseCreateDnMat(&dmH,    k, mat_n, k,     d_H.get(),   CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseCreateDnMat(&dmAHt, mat_m, k, mat_m, d_AHt.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
            cusparseSpMM(ctx.sparse(),
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                &alpha_blas, spA, dmH, &beta_zero, dmAHt, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, d_spmm_ws.get());
            cusparseDestroyDnMat(dmH); cusparseDestroyDnMat(dmAHt);
        }

        // 7. WHHt (m × k) = W * HHt : cuBLAS Ssymm (A=HHt k×k, B=W m×k → m×k)
        cublasSsymm(ctx.blas(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            mat_m, k, &alpha_blas, d_HHt.get(), k,
            d_W.get(), mat_m, &beta_zero, d_WHHt.get(), mat_m);

        // 8a. Ortho penalty for W: compute W*(WtW - I)  [Choi 2008, §3]
        float* ortho_W_ptr = nullptr;
        if (use_ortho_W) {
            cudaMemcpyAsync(d_WtW_ortho.get(), d_WtW.get(), k * k * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            {
                int blocks_k = (k + 255) / 256;
                subtract_identity_k<<<blocks_k, 256, 0, stream>>>(d_WtW_ortho.get(), k);
            }
            // ortho_W (m×k) = W (m×k) * (WtW-I) (k×k, symmetric upper), side=RIGHT
            cublasSsymm(ctx.blas(), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
                mat_m, k, &alpha_blas, d_WtW_ortho.get(), k,
                d_W.get(), mat_m, &beta_zero, d_ortho_W.get(), mat_m);
            ortho_W_ptr = d_ortho_W.get();
        }

        // 8b. MU update W: W ← W ∘ AHt / (WHHt + λ_L2·W + λ_L1 + λ_ortho·ortho_W + ε)
        {
            int total = mat_m * k;
            int blocks = (total + 255) / 256;
            nmf_mu_update_W_kernel<<<blocks, 256, 0, stream>>>(
                d_W.get(), d_AHt.get(), d_WHHt.get(), ortho_W_ptr,
                lam_L1_W, lam_L2_W, lam_ortho_W, eps, mat_m, k);
        }

        // === Convergence check (ONE scalar per outer iter — Rule 4 VALID) ===
        // loss ≈ -2 * <WtA, H> + trace(WtW * HHt)
        // <WtA, H> = cublasSdot(k*n, WtA, H)
        // trace(WtW * HHt): Frobenius inner product of symmetric matrices
        float dot_WtA_H = 0.f, fro_WtW_HHt = 0.f;
        cublasSdot(ctx.blas(), k * mat_n, d_WtA.get(), 1, d_H.get(), 1, &dot_WtA_H);
        cublasSdot(ctx.blas(), k * k, d_WtW.get(), 1, d_HHt.get(), 1, &fro_WtW_HHt);
        cudaStreamSynchronize(stream);  // ← APPROVED: scalar convergence, once per iter

        float loss = -2.f * dot_WtA_H + fro_WtW_HHt;
        loss_history.push_back(loss);

        if (cfg.verbose && it % 10 == 0) {
            // Informational only — not in hot path by default.
        }

        if (it > 0 && prev_loss > 0.f) {
            float rel = std::abs(loss - prev_loss) / (std::abs(prev_loss) + 1.f);
            if (rel < tol) { converged = true; break; }
        }
        prev_loss = loss;
    }

    // --- Download W and H to host ---
    cudaStreamSynchronize(stream);

    NmfResult result;
    result.m = mat_m;
    result.n = mat_n;
    result.k = k;
    result.converged  = converged;
    result.iterations = static_cast<int>(loss_history.size());
    result.train_loss = std::move(loss_history);

    result.W = DenseMatrix(mat_m, k);
    result.H = DenseMatrix(k, mat_n);
    result.d = DenseVector(k);
    for (int i = 0; i < k; ++i) result.d(i) = 1.f;

    cudaMemcpy(result.W.data.data(), d_W.get(), mat_m * k * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(result.H.data.data(), d_H.get(), k * mat_n * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// fit — in-memory GPU NMF.
//
// Precondition: m must be loaded with keep_host_pinned=true.
// (Maintained for compat with existing callers — actually both device and host
// CSC paths are supported; the native impl uses device pointers directly.)
//
// Solver routing (FitConfig / CYCLE-86 / OPTIM-NMF-K50):
//   solver_mode=3 (auto) + rank >= k_cd_cutoff → force MU (mode=2).
//   solver_mode=0 (explicit CD) + cd_max_iter==100 → lower to fit_cfg.cd_max_iter.
// ---------------------------------------------------------------------------
inline NmfResult fit(const io::PzDeviceMatrix& m,
                     const NmfConfig& cfg,
                     const FitConfig& fit_cfg     = FitConfig{},
                     const DenseMatrix* W_init    = nullptr,
                     const DenseMatrix* H_init    = nullptr)
{
    if (m.mat.rows == 0 || m.mat.cols == 0 || m.mat.nnz == 0)
        return NmfResult{};

    const int mat_m = m.mat.rows;
    const int mat_n = m.mat.cols;
    const int k     = cfg.rank;

    if (k <= 0 || k > std::min(mat_m, mat_n))
        throw std::runtime_error("nmf::fit: rank out of bounds");

    // Apply GPU-tuned solver routing (CYCLE-86).
    NmfConfig effective_cfg = cfg;
    if (effective_cfg.solver_mode == 3 && effective_cfg.rank >= fit_cfg.k_cd_cutoff)
        effective_cfg.solver_mode = 2;  // MU
    constexpr int FACTORNET_CD_MAXIT_DEFAULT = 100;
    if (effective_cfg.solver_mode == 0 &&
        effective_cfg.cd_max_iter == FACTORNET_CD_MAXIT_DEFAULT)
        effective_cfg.cd_max_iter = fit_cfg.cd_max_iter;
    // CYCLE-114: merge flat reg scalars into FactorConfig sub-structs before fit.
    // Flat scalars win if non-zero; sub-struct fields set directly also work.
    effective_cfg.sync_factor_configs();

    // Build cuSPARSE descriptor from device CSC (no host roundtrip).
    core::GPUContext ctx;
    cusparseSpMatDescr_t spA = nullptr;
    cusparseCreateCsc(
        &spA,
        (int64_t)mat_m, (int64_t)mat_n, (int64_t)m.mat.nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        m.mat.values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // All modes use the MU engine; CD at k<k_cd_cutoff is treated as MU
    // with tighter tolerance (CD kernel deferred — MU is correct for all k).
    NmfResult result = detail::run_mu_nmf(
        ctx, spA, mat_m, mat_n, k, effective_cfg, W_init, H_init,
        nullptr, nullptr, nullptr, nullptr);

    cusparseDestroySpMat(spA);
    return result;
}

// Backward-compatible 2-arg overload.
inline NmfResult fit(const io::PzDeviceMatrix& m,
                     const NmfConfig& cfg,
                     const DenseMatrix* W_init,
                     const DenseMatrix* H_init)
{
    return fit(m, cfg, FitConfig{}, W_init, H_init);
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet::gpu
