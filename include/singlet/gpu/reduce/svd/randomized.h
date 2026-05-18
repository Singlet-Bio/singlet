// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/svd/randomized.h
//
// Randomized SVD via Halko-Martinsson-Tropp (2011), Algorithm 4.4.
// Replaces factornet::svd::randomized_svd_gpu adapter (CYCLE-105, 2026-04-29).
//
// // algorithm derived from factornet/svd/randomized_gpu.cuh
// Credit: Zach DeBruine, factornet 2021-2026
// // Reference: Halko, Martinsson & Tropp 2011, "Finding Structure with Randomness"
//              §4, Algorithm 4.4 (Randomized SVD with power iteration).
//
// Algorithm:
//   1. Draw Ω (n × l) Gaussian random matrix via cuRAND Philox (l = k + oversample).
//   2. Y = A · Ω  via cuSPARSE cusparseSpMM   (m × l dense).
//   3. Power iteration:  Y = (A Aᵀ)^q Y  (q = n_power_iter, default 3).
//      Each pass:  Y ← A (Aᵀ Y).
//   4. QR(Y) → orthonormal Q (m × l) via cuSOLVER cusolverDnSgeqrf + Sorgqr.
//   5. B = Qᵀ A (l × n):  B = (A^T Q)^T via cuSPARSE SpMM (A^T * Q → n × l, then T).
//   6. SVD(B) (l × n small dense) via cuSOLVER cusolverDnSgesvdj.
//   7. U_A ≈ Q U_B  (cuBLAS Sgemm).
//   8. Truncate to rank k.
//
// Host↔device traffic: ZERO inside the inner loop (Step 3 is device-only).
//   Final D/U/V download to host happens once at step 8.
//
// Workspace budget (device):
//   Ω:  n × l × 4 bytes
//   Y:  m × l × 4 bytes
//   Q:  m × l × 4 bytes   (overwrites Y after QR)
//   B:  l × n × 4 bytes   (l ≤ k + oversample ≤ k+10, typically small)
//   SVD of B: cuSOLVER Sgesvdj workspace (auto-queried)
//   Total: O((m + n) × l × 4) device bytes, l ≪ min(m,n).
//
// Streams: uses the stream from the caller's GPUContext.
// Precision: fp32 for sketching + matvecs; fp64 reduction root in QR (cuSOLVER internal).
// Determinism: deterministic for fixed cfg.seed.
// Optimal range: 32 ≤ k_max < 64 on GPU; lowest memory footprint.
// OOC plan: same as deflation — full CSC must fit on device.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/reduce/svd/types.h>
#include <singlet/gpu/reduce/svd/deflation.h>  // require_host_retained

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace reduce {
namespace svd {

// ---------------------------------------------------------------------------
// randomized — HMT randomized SVD.
//
// cfg.oversample:     extra sketch columns (l = k + oversample, default 10)
// cfg.n_power_iter:   power iteration passes q (default 3)
// cfg.seed:           cuRAND seed
//
// Returns SvdResult with k_selected = min(k, l) singular triplets on host.
// ---------------------------------------------------------------------------
inline SvdResult randomized(const io::PzDeviceMatrix& m, const SvdConfig& cfg)
{
    if (m.mat.rows == 0 || m.mat.cols == 0 || m.mat.nnz == 0)
        return SvdResult{};

    const int mat_m = m.mat.rows;
    const int mat_n = m.mat.cols;
    const int k     = std::min(cfg.k_max, std::min(mat_m, mat_n));
    if (k <= 0) return SvdResult{};

    const int oversample    = (cfg.oversample > 0) ? cfg.oversample : 10;
    const int n_power_iter  = (cfg.n_power_iter >= 0) ? cfg.n_power_iter : 3;
    const int l = std::min(k + oversample, std::min(mat_m, mat_n));  // sketch dimension

    core::GPUContext ctx;
    cudaStream_t stream = ctx.stream();

    // --- Seed cuRAND ---
    curandSetPseudoRandomGeneratorSeed(ctx.curand(), cfg.seed);
    curandGenerateSeeds(ctx.curand());

    // --- Build cuSPARSE CSC descriptor ---
    cusparseSpMatDescr_t spA = nullptr;
    cusparseCreateCsc(
        &spA,
        (int64_t)mat_m, (int64_t)mat_n, (int64_t)m.mat.nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        m.mat.values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // --- Step 1: Ω (n × l) Gaussian random --- via cuRAND
    core::DeviceMemory<float> d_Omega(mat_n * l);
    curandGenerateNormal(ctx.curand(), d_Omega.get(), (size_t)mat_n * l, 0.f, 1.f);

    // --- Step 2: Y = A * Ω  (m × l) via cuSPARSE SpMM ---
    // A is CSC (m × n); Ω is dense (n × l, col-major).
    // SpMM computes C = A * B where B is n×l dense.
    core::DeviceMemory<float> d_Y(mat_m * l);    // m × l, col-major

    cusparseDnMatDescr_t dmOmega = nullptr, dmY = nullptr;
    cusparseCreateDnMat(&dmOmega, mat_n, l, mat_n, d_Omega.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&dmY,     mat_m, l, mat_m, d_Y.get(),     CUDA_R_32F, CUSPARSE_ORDER_COL);

    float alpha = 1.f, beta = 0.f;
    size_t ws_sz = 0;
    cusparseSpMM_bufferSize(ctx.sparse(),
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmOmega, &beta, dmY, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &ws_sz);
    core::DeviceMemory<uint8_t> d_ws_spmm(ws_sz > 0 ? ws_sz : 1);

    cusparseSpMM(ctx.sparse(),
        CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmOmega, &beta, dmY, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, d_ws_spmm.get());

    // --- Step 3: Power iteration: Y = (A Aᵀ)^q Y ---
    if (n_power_iter > 0) {
        // Each pass: Z = Aᵀ Y (n × l), then Y = A Z (m × l).
        core::DeviceMemory<float> d_Z(mat_n * l);
        cusparseDnMatDescr_t dmZ = nullptr;
        cusparseCreateDnMat(&dmZ, mat_n, l, mat_n, d_Z.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);

        size_t ws_at = 0;
        cusparseSpMM_bufferSize(ctx.sparse(),
            CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, spA, dmY, &beta, dmZ, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &ws_at);
        core::DeviceMemory<uint8_t> d_ws_at(ws_at > 0 ? ws_at : 1);

        for (int p = 0; p < n_power_iter; ++p) {
            // Z = Aᵀ * Y
            cusparseSpMM(ctx.sparse(),
                CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, spA, dmY, &beta, dmZ, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, d_ws_at.get());
            // Y = A * Z
            cusparseSpMM(ctx.sparse(),
                CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, spA, dmZ, &beta, dmY, CUDA_R_32F,
                CUSPARSE_SPMM_ALG_DEFAULT, d_ws_spmm.get());
        }

        cusparseDestroyDnMat(dmZ);
    }

    // --- Step 4: QR(Y) → Q (m × l) via cuSOLVER ---
    // cusolverDnSgeqrf computes the QR factorization of Y in-place.
    // cusolverDnSorgqr extracts Q.

    // cuSOLVER works on column-major matrices: Y is m × l already.
    int lwork_qr = 0;
    cusolverDnSgeqrf_bufferSize(ctx.solver(), mat_m, l, d_Y.get(), mat_m, &lwork_qr);
    core::DeviceMemory<float>  d_work_qr(lwork_qr > 0 ? lwork_qr : 1);
    core::DeviceMemory<float>  d_tau(l);     // Householder reflector scalars
    core::DeviceMemory<int>    d_info(1);

    cusolverDnSgeqrf(ctx.solver(), mat_m, l,
        d_Y.get(), mat_m, d_tau.get(),
        d_work_qr.get(), lwork_qr,
        d_info.get());

    // Sorgqr: form Q from the Householder reflectors stored in Y.
    int lwork_orgqr = 0;
    cusolverDnSorgqr_bufferSize(ctx.solver(), mat_m, l, l,
        d_Y.get(), mat_m, d_tau.get(), &lwork_orgqr);
    core::DeviceMemory<float> d_work_orgqr(lwork_orgqr > 0 ? lwork_orgqr : 1);

    cusolverDnSorgqr(ctx.solver(), mat_m, l, l,
        d_Y.get(), mat_m, d_tau.get(),
        d_work_orgqr.get(), lwork_orgqr,
        d_info.get());
    // d_Y now holds Q (m × l, orthonormal columns).
    float* d_Q = d_Y.get();  // alias

    // --- Step 5: B = Qᵀ A  (l × n dense) ---
    // = (Aᵀ Q)ᵀ. Compute Aᵀ Q via SpMM (n × l), then transpose to l × n.
    core::DeviceMemory<float> d_AtQ(mat_n * l);   // n × l
    core::DeviceMemory<float> d_B(l * mat_n);     // l × n

    cusparseDnMatDescr_t dmQ = nullptr, dmAtQ = nullptr;
    cusparseCreateDnMat(&dmQ,   mat_m, l, mat_m, d_Q,      CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&dmAtQ, mat_n, l, mat_n, d_AtQ.get(), CUDA_R_32F, CUSPARSE_ORDER_COL);

    size_t ws_atq = 0;
    cusparseSpMM_bufferSize(ctx.sparse(),
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmQ, &beta, dmAtQ, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &ws_atq);
    core::DeviceMemory<uint8_t> d_ws_atq(ws_atq > 0 ? ws_atq : 1);

    cusparseSpMM(ctx.sparse(),
        CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dmQ, &beta, dmAtQ, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, d_ws_atq.get());

    // Transpose: B (l × n) = (Aᵀ Q)ᵀ (n × l → l × n)
    // cuBLAS Sgeam: B = αA^T + βB  where A = d_AtQ (n × l), α=1, β=0.
    // Sgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, l, n, &alpha, AtQ(n×l,lda=n), n,
    //                                                &beta,  nullptr, l, B(l×n,ldb=l), l)
    float b0 = 0.f;
    cublasSgeam(ctx.blas(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        l, mat_n,
        &alpha, d_AtQ.get(), mat_n,
        &b0,    nullptr, l,
        d_B.get(), l);

    // --- Step 6: SVD(B) (l × n) via cuSOLVER Sgesvdj ---
    // We need U_B (l × l), S (l), V_B (n × l).
    core::DeviceMemory<float> d_UB(l * l);
    core::DeviceMemory<float> d_S(l);
    core::DeviceMemory<float> d_VBT(l * mat_n);  // cuSOLVER returns V^T

    gesvdjInfo_t gesvdj_params = nullptr;
    cusolverDnCreateGesvdjInfo(&gesvdj_params);
    cusolverDnXgesvdjSetTolerance(gesvdj_params, 1e-7f);
    cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, 100);

    int lwork_svd = 0;
    cusolverDnSgesvdj_bufferSize(ctx.solver(),
        CUSOLVER_EIG_MODE_VECTOR, 1,   // compute U and V
        l, mat_n,
        d_B.get(), l,
        d_S.get(), d_UB.get(), l, d_VBT.get(), l,
        &lwork_svd,
        gesvdj_params);
    core::DeviceMemory<float> d_work_svd(lwork_svd > 0 ? lwork_svd : 1);

    cusolverDnSgesvdj(ctx.solver(),
        CUSOLVER_EIG_MODE_VECTOR, 1,
        l, mat_n,
        d_B.get(), l,
        d_S.get(), d_UB.get(), l, d_VBT.get(), l,
        d_work_svd.get(), lwork_svd,
        d_info.get(),
        gesvdj_params);
    cusolverDnDestroyGesvdjInfo(gesvdj_params);

    // --- Step 7: U_A = Q * U_B  (m × l) via cuBLAS Sgemm ---
    // Q: m × l; U_B: l × l; U_A: m × l.
    core::DeviceMemory<float> d_UA(mat_m * l);
    cublasSgemm(ctx.blas(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        mat_m, l, l,
        &alpha,
        d_Q,   mat_m,   // Q  (m × l)
        d_UB.get(), l,  // UB (l × l)
        &b0,
        d_UA.get(), mat_m);

    // --- Step 8: Truncate to rank k and download to host ---
    cudaStreamSynchronize(stream);  // ← APPROVED: one-time result download

    int k_actual = std::min(k, l);

    std::vector<float> U_out(mat_m * k_actual);
    std::vector<float> S_out(k_actual);
    std::vector<float> V_out(mat_n * k_actual);

    // Download U_A columns 0..k_actual-1 (m × k_actual)
    cudaMemcpy(U_out.data(), d_UA.get(), mat_m * k_actual * sizeof(float), cudaMemcpyDeviceToHost);

    // Download S (first k_actual singular values)
    cudaMemcpy(S_out.data(), d_S.get(), k_actual * sizeof(float), cudaMemcpyDeviceToHost);

    // V: d_VBT is l × n (V^T). Row j of V^T = column j of V.
    // We want V (n × k_actual). V[:,j] = d_VBT[j, :] = row j.
    // col-major layout: V(i, j) = V_out[i + j*n] = d_VBT[j + i*l] (row-major index).
    {
        std::vector<float> VBT_host((size_t)l * mat_n);
        cudaMemcpy(VBT_host.data(), d_VBT.get(), (size_t)l * mat_n * sizeof(float), cudaMemcpyDeviceToHost);
        for (int j = 0; j < k_actual; ++j)
            for (int i = 0; i < mat_n; ++i)
                V_out[i + j * mat_n] = VBT_host[j + i * l];
    }

    // --- Cleanup ---
    cusparseDestroyDnMat(dmOmega);
    cusparseDestroyDnMat(dmY);
    cusparseDestroyDnMat(dmQ);
    cusparseDestroyDnMat(dmAtQ);
    cusparseDestroySpMat(spA);

    // --- Build result ---
    SvdResult result;
    result.U_data     = std::move(U_out);
    result.d.data_    = std::move(S_out);
    result.V_data     = std::move(V_out);
    result.finalize(mat_m, mat_n, k_actual);

    return result;
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet::gpu
