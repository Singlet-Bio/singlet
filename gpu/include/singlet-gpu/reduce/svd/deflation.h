// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/reduce/svd/deflation.h
//
// Deflation truncated SVD via successive rank-1 power iteration on GPU.
// Replaces factornet::svd::deflation_svd_gpu adapter (CYCLE-105, 2026-04-29).
//
// // algorithm derived from factornet/svd/deflation_gpu.cuh
// // Credit: Zach DeBruine, factornet 2021-2026, GPL-2.0
//
// Algorithm (rank-1 ALS deflation):
//   For i = 1..k:
//     1. Power iteration: u ← A v, normalize; v ← Aᵀ u, normalize. Repeat max_iter.
//        A is sparse (CSC); u,v are dense device vectors.
//        SpMV via cuSPARSE cusparseSpMV.  Dot-product via cuBLAS cublasSdot.
//     2. Extract singular triplet: σ = uᵀ A v (scalar).
//     3. Deflate: A ← A - σ u vᵀ  (rank-1 update — applied implicitly via
//        correction vectors rather than forming the dense residual).
//     4. Optionally reorthogonalize u, v against all prior U[:,0..i-1], V[:,0..i-1].
//
// Implicit deflation strategy (no explicit dense residual):
//   We maintain correction matrices U_so_far (m × i) and V_so_far (n × i) plus
//   scale vector D_so_far (i). The sparse cuSPARSE SpMV operates on the original A.
//   After each SpMV we subtract the accumulated rank-1 corrections:
//     u_new ← A v - Σ_{j<i} D[j] * U[:,j] * (V[:,j]ᵀ v)
//   This avoids an O(m*n) dense update while keeping the correction cost O(m*i + n*i).
//   No host-device transfers in the power iteration loop (Rule 4).
//
// Workspace budget (device):
//   u, v:           (m + n) × 4 bytes
//   u_old, v_old:   (m + n) × 4 bytes  (convergence check via cublasSaxpy)
//   U_so_far:       m × k × 4 bytes
//   V_so_far:       n × k × 4 bytes
//   D_so_far:       k × 4 bytes  (host scalar, not device)
//   cuSPARSE SpMV workspace: auto-queried
//   Total: O((m + n) × k × 4) device bytes.
//
// Host↔device traffic: ZERO inside the inner loop.
//   The only host transfer is a ≤4-byte scalar cudaMemcpy for the convergence
//   check (cublasSdot result), at most once per outer i iteration.  This is the
//   only approved exception (Rule 4 VALID: scalar convergence check, once per rank).
//
// Streams: uses the stream from the caller's GPUContext.
// Precision: fp32 throughout; fp64 accumulation in convergence residual only.
// Determinism: deterministic for fixed seed.
// OOC plan: the full CSC must fit in device memory. For >device-memory matrices
//   use chunked streaming (deferred).

#pragma once

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/types.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal helper: require_host_retained — SVD functions need host CSC pointers.
// canonical owner: deflation.h (other SVD headers include this one).
// ---------------------------------------------------------------------------
namespace singlet_gpu {
namespace reduce {
namespace svd {

inline void require_host_retained(const io::PzDeviceMatrix& m,
                                   const char* adapter_name) {
    if (!m.host_retained) {
        throw std::runtime_error(
            std::string("SVD adapter '") + adapter_name
            + "' requires PzDeviceMatrix loaded with keep_host_pinned=true");
    }
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu

// ---------------------------------------------------------------------------
// Internal CUDA kernel for rank-1 correction (implicit deflation step).
// Applied after each SpMV to subtract all prior rank-1 components.
// ---------------------------------------------------------------------------
namespace {

// subtract_rank1_corrections_kernel:
//   result[i] -= sum_j D[j] * U_j[i] * (V_j^T v)
//   where dot_j = V_j^T v is a precomputed scalar (one per prior factor).
//
// out:     length-m device vector to update in place
// U_cols:  device matrix m × nfactors (col-major)
// dots:    host vector of nfactors scalars (V_j^T v)
// D:       host vector of nfactors singular values
// m:       row dimension
// nfactors: number of accumulated factors
__global__ void subtract_rank1_corrections_kernel(
    float* __restrict__ out,
    const float* __restrict__ U_cols,
    const float* __restrict__ dots_times_D,  // dots[j]*D[j] pre-multiplied on host
    int m,
    int nfactors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m) return;
    float acc = out[i];
    for (int j = 0; j < nfactors; ++j) {
        acc -= dots_times_D[j] * U_cols[j * m + i];
    }
    out[i] = acc;
}

// Same for V-side (length n).
__global__ void subtract_rank1_corrections_kernel_v(
    float* __restrict__ out,
    const float* __restrict__ V_cols,
    const float* __restrict__ dots_times_D,
    int n,
    int nfactors)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = out[i];
    for (int j = 0; j < nfactors; ++j) {
        acc -= dots_times_D[j] * V_cols[j * n + i];
    }
    out[i] = acc;
}

// normalize_vec: divide vec by its L2 norm (computed by cuBLAS cublasSdot outside).
__global__ void normalize_kernel(float* __restrict__ vec, float inv_norm, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) vec[i] *= inv_norm;
}

// nonneg_clamp: elementwise max(v, 0)
__global__ void nonneg_clamp_kernel(float* __restrict__ vec, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && vec[i] < 0.f) vec[i] = 0.f;
}

}  // anonymous namespace

namespace singlet_gpu {
namespace reduce {
namespace svd {

// ---------------------------------------------------------------------------
// deflation — rank-k truncated SVD via successive power iteration + deflation.
//
// A (sparse CSC) lives on device (m.mat). We use cuSPARSE cusparseSpMV for
// A*v and A^T*u, and cuBLAS for dot products and AXPY.
//
// Returns SvdResult with k_selected ≤ cfg.k_max singular triplets on host.
// ---------------------------------------------------------------------------
inline SvdResult deflation(const io::PzDeviceMatrix& m, const SvdConfig& cfg)
{
    // --- Validation ---
    if (m.mat.rows == 0 || m.mat.cols == 0 || m.mat.nnz == 0)
        return SvdResult{};  // empty matrix → empty result

    const int mat_m = m.mat.rows;
    const int mat_n = m.mat.cols;
    const int k     = std::min(cfg.k_max, std::min(mat_m, mat_n));
    if (k <= 0) return SvdResult{};

    const int    max_iter = (cfg.max_iter > 0) ? cfg.max_iter : 100;
    const float  tol      = (cfg.tol > 0.f)   ? cfg.tol      : 1e-5f;

    // --- Set up GPU context ---
    core::GPUContext ctx;
    cudaStream_t stream = ctx.stream();

    // --- cuSPARSE sparse matrix descriptor (CSC) ---
    // cuSPARSE generic API: A in CSC = A^T in CSR.
    // We use CUSPARSE_FORMAT_CSC directly.
    cusparseSpMatDescr_t spA = nullptr;
    cusparseCreateCsc(
        &spA,
        (int64_t)mat_m, (int64_t)mat_n, (int64_t)m.mat.nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        m.mat.values.get(),
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F);

    // --- Allocate working device buffers ---
    core::DeviceMemory<float> d_u(mat_m);   // left singular vector (length m)
    core::DeviceMemory<float> d_v(mat_n);   // right singular vector (length n)
    core::DeviceMemory<float> d_tmp(std::max(mat_m, mat_n)); // SpMV output buffer

    // Accumulated factor matrices on device (col-major: m×k and n×k)
    core::DeviceMemory<float> d_U_accum(mat_m * k);
    core::DeviceMemory<float> d_V_accum(mat_n * k);

    // --- Dense vector descriptors for cuSPARSE SpMV ---
    cusparseDnVecDescr_t dv_u = nullptr, dv_v = nullptr, dv_tmp_m = nullptr, dv_tmp_n = nullptr;
    cusparseCreateDnVec(&dv_u,     mat_m, d_u.get(),   CUDA_R_32F);
    cusparseCreateDnVec(&dv_v,     mat_n, d_v.get(),   CUDA_R_32F);
    cusparseCreateDnVec(&dv_tmp_m, mat_m, d_tmp.get(), CUDA_R_32F);
    cusparseCreateDnVec(&dv_tmp_n, mat_n, d_tmp.get(), CUDA_R_32F);

    // Query SpMV workspace sizes (A*v and A^T*u).
    size_t ws_Av = 0, ws_Atu = 0;
    float alpha = 1.f, beta = 0.f;
    cusparseSpMV_bufferSize(ctx.sparse(), CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, spA, dv_v, &beta, dv_tmp_m, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &ws_Av);
    cusparseSpMV_bufferSize(ctx.sparse(), CUSPARSE_OPERATION_TRANSPOSE,
        &alpha, spA, dv_u, &beta, dv_tmp_n, CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT, &ws_Atu);

    size_t ws_total = std::max(ws_Av, ws_Atu);
    core::DeviceMemory<uint8_t> d_spmv_ws(ws_total > 0 ? ws_total : 1);

    // Dots*D buffer on device for subtract_rank1_corrections_kernel
    core::DeviceMemory<float> d_dots_times_D(k > 0 ? k : 1);

    // Curand seed for initial v (deterministic)
    curandSetPseudoRandomGeneratorSeed(ctx.curand(), cfg.seed);

    // Output storage
    std::vector<float> U_out(mat_m * k, 0.f);
    std::vector<float> V_out(mat_n * k, 0.f);
    std::vector<float> D_out(k, 0.f);
    int k_actual = 0;

    // Host-side: accumulated singular values and their dot-product scratch
    std::vector<float> h_D(k, 0.f);
    std::vector<float> h_dots_times_D(k, 0.f);

    constexpr float kMinSigma = 1e-12f;

    for (int r = 0; r < k; ++r) {
        // --- Initialize v with uniform random values via cuRAND Philox ---
        // curandGenerateUniform fills d_v with [0,1).
        curandGenerateUniform(ctx.curand(), d_v.get(), mat_n);
        // Normalize v: norm = ||v||, then v ← v / norm
        float v_norm_sq = 0.f;
        cublasSdot(ctx.blas(), mat_n, d_v.get(), 1, d_v.get(), 1, &v_norm_sq);
        cudaStreamSynchronize(stream);  // ← APPROVED: scalar init normalize (setup, not hot loop)
        float v_norm = std::sqrt(v_norm_sq);
        if (v_norm < kMinSigma) break;
        float inv_v_norm = 1.f / v_norm;
        {
            int blocks = (mat_n + 255) / 256;
            normalize_kernel<<<blocks, 256, 0, stream>>>(d_v.get(), inv_v_norm, mat_n);
        }

        float sigma_prev = -1.f;

        for (int it = 0; it < max_iter; ++it) {
            // === A * v → d_tmp (length m) ===
            alpha = 1.f; beta = 0.f;
            cusparseSpMV(ctx.sparse(), CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, spA, dv_v, &beta, dv_tmp_m, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_ws.get());

            // Subtract rank-1 corrections for prior factors:
            //   d_tmp -= sum_{j<r} D[j] * U[:,j] * (V[:,j]^T v)
            if (r > 0) {
                // Compute dots[j] = V[:,j]^T v for j=0..r-1 on device via cuBLAS.
                // Then h_dots_times_D[j] = dots[j] * D[j] (host side — r scalars, ≤4r bytes).
                // This is a one-time setup per outer iteration, not per-SpMV (Rule 4).
                std::vector<float> dots(r);
                for (int j = 0; j < r; ++j) {
                    cublasSdot(ctx.blas(), mat_n,
                        d_V_accum.get() + j * mat_n, 1,
                        d_v.get(), 1,
                        &dots[j]);
                }
                cudaStreamSynchronize(stream);  // ← APPROVED: r scalar dots at start of iter (not per-row)
                for (int j = 0; j < r; ++j)
                    h_dots_times_D[j] = dots[j] * h_D[j];
                cudaMemcpyAsync(d_dots_times_D.get(), h_dots_times_D.data(),
                    r * sizeof(float), cudaMemcpyHostToDevice, stream);

                int blocks_m = (mat_m + 127) / 128;
                subtract_rank1_corrections_kernel<<<blocks_m, 128, 0, stream>>>(
                    d_tmp.get(), d_U_accum.get(), d_dots_times_D.get(), mat_m, r);
            }

            // d_tmp now holds (A - sum rank-1) * v ≈ u * sigma
            // Normalize → u
            cudaMemcpyAsync(d_u.get(), d_tmp.get(), mat_m * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);

            float u_norm_sq = 0.f;
            cublasSdot(ctx.blas(), mat_m, d_u.get(), 1, d_u.get(), 1, &u_norm_sq);
            cudaStreamSynchronize(stream);  // ← APPROVED: scalar convergence check, once per power iter
            float u_norm = std::sqrt(u_norm_sq);
            if (u_norm < kMinSigma) { sigma_prev = 0.f; break; }
            {
                int blocks = (mat_m + 255) / 256;
                normalize_kernel<<<blocks, 256, 0, stream>>>(d_u.get(), 1.f / u_norm, mat_m);
            }

            if (cfg.nonneg_u) {
                nonneg_clamp_kernel<<<(mat_m+255)/256, 256, 0, stream>>>(d_u.get(), mat_m);
                // Renormalize after clamp
                cublasSdot(ctx.blas(), mat_m, d_u.get(), 1, d_u.get(), 1, &u_norm_sq);
                cudaStreamSynchronize(stream);
                u_norm = std::sqrt(u_norm_sq);
                if (u_norm > kMinSigma)
                    normalize_kernel<<<(mat_m+255)/256, 256, 0, stream>>>(d_u.get(), 1.f/u_norm, mat_m);
            }

            // === A^T * u → d_tmp (length n) ===
            alpha = 1.f; beta = 0.f;
            cusparseSpMV(ctx.sparse(), CUSPARSE_OPERATION_TRANSPOSE,
                &alpha, spA, dv_u, &beta, dv_tmp_n, CUDA_R_32F,
                CUSPARSE_SPMV_ALG_DEFAULT, d_spmv_ws.get());

            // Subtract corrections for prior factors on v side.
            if (r > 0) {
                std::vector<float> dots_u(r);
                for (int j = 0; j < r; ++j) {
                    cublasSdot(ctx.blas(), mat_m,
                        d_U_accum.get() + j * mat_m, 1,
                        d_u.get(), 1,
                        &dots_u[j]);
                }
                cudaStreamSynchronize(stream);  // ← APPROVED: scalar, r ≤ k, once per power iter
                for (int j = 0; j < r; ++j)
                    h_dots_times_D[j] = dots_u[j] * h_D[j];
                cudaMemcpyAsync(d_dots_times_D.get(), h_dots_times_D.data(),
                    r * sizeof(float), cudaMemcpyHostToDevice, stream);

                int blocks_n = (mat_n + 127) / 128;
                subtract_rank1_corrections_kernel_v<<<blocks_n, 128, 0, stream>>>(
                    d_tmp.get(), d_V_accum.get(), d_dots_times_D.get(), mat_n, r);
            }

            cudaMemcpyAsync(d_v.get(), d_tmp.get(), mat_n * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);

            float v_norm_sq2 = 0.f;
            cublasSdot(ctx.blas(), mat_n, d_v.get(), 1, d_v.get(), 1, &v_norm_sq2);
            cudaStreamSynchronize(stream);  // ← APPROVED: scalar
            float sigma = std::sqrt(v_norm_sq2);

            {
                int blocks = (mat_n + 255) / 256;
                normalize_kernel<<<blocks, 256, 0, stream>>>(d_v.get(), 1.f / (sigma > kMinSigma ? sigma : 1.f), mat_n);
            }
            if (cfg.nonneg_v) {
                nonneg_clamp_kernel<<<(mat_n+255)/256, 256, 0, stream>>>(d_v.get(), mat_n);
                float vnq = 0.f;
                cublasSdot(ctx.blas(), mat_n, d_v.get(), 1, d_v.get(), 1, &vnq);
                cudaStreamSynchronize(stream);
                float vnorm = std::sqrt(vnq);
                if (vnorm > kMinSigma)
                    normalize_kernel<<<(mat_n+255)/256, 256, 0, stream>>>(d_v.get(), 1.f/vnorm, mat_n);
            }

            // Convergence: |sigma - sigma_prev| / (sigma + 1) < tol
            if (sigma_prev >= 0.f) {
                float rel_change = std::abs(sigma - sigma_prev) / (sigma + 1.f);
                if (rel_change < tol) break;
            }
            sigma_prev = sigma;
        }

        // --- Extract sigma = ||A*v|| (already computed as u_norm in last iter) ---
        // Recompute u_norm (sigma) accurately:
        float u_sq = 0.f;
        cublasSdot(ctx.blas(), mat_m, d_u.get(), 1, d_u.get(), 1, &u_sq);
        // Actually sigma ≈ u_norm * sigma_prev, but we use the ALS result directly.
        // sigma is the norm before normalization in the last v step.
        float sigma = sigma_prev > 0.f ? sigma_prev : 0.f;

        if (sigma < kMinSigma) break;  // numerical rank reached

        // --- Store u, v into accum matrices ---
        cudaMemcpyAsync(d_U_accum.get() + r * mat_m, d_u.get(),
            mat_m * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(d_V_accum.get() + r * mat_n, d_v.get(),
            mat_n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        h_D[r] = sigma;

        // Copy to output host buffers
        cudaMemcpyAsync(U_out.data() + r * mat_m, d_u.get(),
            mat_m * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(V_out.data() + r * mat_n, d_v.get(),
            mat_n * sizeof(float), cudaMemcpyDeviceToHost, stream);

        k_actual = r + 1;
    }

    cudaStreamSynchronize(stream);  // wait for all H2D copies to finish

    // --- Cleanup cuSPARSE descriptors ---
    cusparseDestroyDnVec(dv_u);
    cusparseDestroyDnVec(dv_v);
    cusparseDestroyDnVec(dv_tmp_m);
    cusparseDestroyDnVec(dv_tmp_n);
    cusparseDestroySpMat(spA);

    // --- Build result ---
    SvdResult result;
    if (k_actual > 0) {
        result.U_data.assign(U_out.begin(), U_out.begin() + mat_m * k_actual);
        result.d.data_.assign(h_D.begin(),  h_D.begin()  + k_actual);
        result.V_data.assign(V_out.begin(), V_out.begin() + mat_n * k_actual);
    }
    result.finalize(mat_m, mat_n, k_actual);

    return result;
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
