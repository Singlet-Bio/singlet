// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/reduce/svd/cv.h
//
// Speckled cross-validation for truncated SVD — random-mask hold-out, sweep k,
// find optimal rank (Wold 1978 "drop-out" / "Gabriel cross-validation" style).
// CYCLE-115, 2026-04-29.
//
// Mirror of reduce/nmf/cv.h — same hash, same seed convention, reproducible
// against the same synthetic fixture.
//
// Algorithm:
//   1. Randomly mask holdout_fraction of nonzero entries via deterministic hash:
//        hash32(i, j, seed) = ((i*0x85ebca6bu)^(j*0xc2b2ae35u)^seed)*0x27d4eb2du
//        Masked iff hash32(i,j,seed) < holdout_fraction * 2^32.
//      Identical seed → identical mask (reproducible, NMF/SVD use same fixture).
//   2. Zero out masked entries in a host copy → A_train.
//   3. Fit truncated SVD via svd::auto_select on A_train.
//   4. Reconstruct held-out entries: A_hat[i,j] = U[i,:] · diag(d) · V[j,:]^T.
//      Computed on device via a custom CUDA kernel over masked nnz entries.
//   5. test_mse = mean((A_orig[i,j] - A_hat[i,j])^2) over held-out only.
//   6. cv_fit returns SvdResult with .test_loss and .train_loss populated.
//   7. speckled_cv sweeps k_values, returns CVSVDResult with chosen_k=argmin(test_mse).
//
// Wold drop-out rationale: zeroing entries is the standard approach for SVD/PCA
// cross-validation (Wold 1978; Bro 2008). The punctured matrix forces the SVD to
// "predict" held-out entries via the low-rank approximation; too-high rank
// overfits (memorises noise) yielding U-shaped test_mse in k.
//
// Deflation backend (CYCLE-61 winner): zeroing entries works in-place via the
// cusparseSpMV path without any kernel rewrite — the zeroed entries simply
// contribute nothing to the matrix-vector products.
//
// Host↔device traffic:
//   Mask applied on HOST at load time (once, before fit — not hot).
//   Test MSE kernel runs on device; two scalar cudaMemcpy AFTER fit (approved).
//   Train MSE kernel likewise — one additional scalar pair post-fit.
//   Zero host traffic during SVD power iteration.
//
// Precision: fp32.
// Determinism: seeded via cfg.seed.
// OOC plan: full CSC must fit on device. For very large matrices, run CV on a
//   representative chunk (e.g., first 50 k cells).

#pragma once

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/types.h>
#include <singlet-gpu/reduce/svd/auto_select.h>   // calls deflation / randomized
#include <singlet-gpu/reduce/svd/deflation.h>     // require_host_retained

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Device kernels — anonymous namespace, not part of the public API.
// ---------------------------------------------------------------------------
namespace {

// svd_test_mse_kernel: for each masked nnz entry, accumulate
//   (true_val - U[i,:] · diag(d) · V[j,:]^T)^2.
//
// mask_flags: uint8 device array, 1=masked, length nnz.
// col_ptr:    int32 device array, length n+1 (CSC col pointers).
// row_idx:    int32 device array, length nnz.
// values:     float device array, length nnz (original — NOT zeroed).
// U_dev:      m × k col-major (host → device already copied by caller).
// d_dev:      k  singular values.
// V_dev:      n × k col-major.
// d_loss:     float device scalar (atomicAdd accumulator).
// d_count:    int   device scalar (count of masked entries).
__global__ void svd_test_mse_kernel(
    const uint8_t* __restrict__ mask_flags,
    const int32_t* __restrict__ col_ptr,
    const int32_t* __restrict__ row_idx,
    const float*   __restrict__ values,
    const float*   __restrict__ U_dev,
    const float*   __restrict__ d_sv,
    const float*   __restrict__ V_dev,
    float* __restrict__ d_loss,
    int*   __restrict__ d_count,
    int m, int n, int k, int nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    if (!mask_flags[idx]) return;

    // Find column j for this nnz entry by binary search on col_ptr.
    // WHY binary search: CSC format lacks a direct nnz→col map.
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (col_ptr[mid] <= idx) lo = mid; else hi = mid;
    }
    int j = lo;
    int i = row_idx[idx];
    float true_val = values[idx];

    // Predict A_hat[i,j] = sum_r U[i,r] * d[r] * V[j,r]
    float pred = 0.f;
    for (int r = 0; r < k; ++r) {
        pred += U_dev[i + (std::size_t)r * m] * d_sv[r] * V_dev[j + (std::size_t)r * n];
    }

    float diff = true_val - pred;
    atomicAdd(d_loss, diff * diff);
    atomicAdd(d_count, 1);
}

// svd_train_mse_kernel: MSE on UNMASKED entries (mask_flags[idx] == 0).
// Identical structure to svd_test_mse_kernel — separate kernel avoids a branch
// divergence hot path inside the masked kernel.
__global__ void svd_train_mse_kernel(
    const uint8_t* __restrict__ mask_flags,
    const int32_t* __restrict__ col_ptr,
    const int32_t* __restrict__ row_idx,
    const float*   __restrict__ values,
    const float*   __restrict__ U_dev,
    const float*   __restrict__ d_sv,
    const float*   __restrict__ V_dev,
    float* __restrict__ d_loss,
    int*   __restrict__ d_count,
    int m, int n, int k, int nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    if (mask_flags[idx]) return;  // skip held-out entries

    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (col_ptr[mid] <= idx) lo = mid; else hi = mid;
    }
    int j = lo;
    int i = row_idx[idx];
    float true_val = values[idx];

    float pred = 0.f;
    for (int r = 0; r < k; ++r) {
        pred += U_dev[i + (std::size_t)r * m] * d_sv[r] * V_dev[j + (std::size_t)r * n];
    }

    float diff = true_val - pred;
    atomicAdd(d_loss, diff * diff);
    atomicAdd(d_count, 1);
}

}  // anonymous namespace

namespace singlet_gpu {
namespace reduce {
namespace svd {

// ---------------------------------------------------------------------------
// LCG/mixing hash — same function as nmf::lcg_hash_f32.
// Copied here (not factored to core/mask.h) per task instructions.
// Same seed → identical mask across NMF and SVD CV runs.
// ---------------------------------------------------------------------------
static inline float svd_cv_hash_f32(uint32_t i, uint32_t j, uint64_t seed) {
    uint32_t seed32 = static_cast<uint32_t>(seed ^ (seed >> 32));
    uint32_t h = ((i * 0x85ebca6bu) ^ (j * 0xc2b2ae35u) ^ seed32) * 0x27d4eb2du;
    // Map to [0, 1)
    return static_cast<float>(h) * (1.0f / 4294967296.0f);
}

// ---------------------------------------------------------------------------
// cv_fit — single-rank speckled-mask SVD cross-validation fit.
//
// 1. Build A_train by zeroing masked entries in a host copy of values.
// 2. Upload A_train to device (same indptr / row_idx structure).
// 3. Fit truncated SVD via auto_select on A_train.
// 4. Evaluate test MSE and train MSE on device over original values.
// 5. Return SvdResult with test_loss and train_loss populated.
//
// Precondition: m must have host_retained=true.
// ---------------------------------------------------------------------------
inline SvdResult cv_fit(const io::PzDeviceMatrix& m,
                        int                       k,
                        const SvdConfig&          cfg)
{
    require_host_retained(m, "svd::cv_fit");

    const int mat_m = m.mat.rows;
    const int mat_n = m.mat.cols;
    const int nnz   = m.mat.nnz;

    const float holdout_frac =
        (cfg.holdout_fraction > 0.f && cfg.holdout_fraction < 1.f)
        ? cfg.holdout_fraction : 0.05f;

    // host_indptr / host_indices: shared_ptr<int>; int == int32_t on LP64 (mirrors nmf/cv.h).
    const int32_t* h_col_ptr = reinterpret_cast<const int32_t*>(m.host_indptr.get());
    const int32_t* h_row_idx = reinterpret_cast<const int32_t*>(m.host_indices.get());
    const float*   h_values  = m.host_values.get();

    // --- Build A_train (host): zero out masked entries ---
    std::vector<float>   h_train_values(nnz);
    std::vector<uint8_t> h_mask(nnz, 0);
    int n_masked = 0;

    for (int j = 0; j < mat_n; ++j) {
        for (int p = h_col_ptr[j]; p < h_col_ptr[j + 1]; ++p) {
            int i = h_row_idx[p];
            float rv = svd_cv_hash_f32(
                static_cast<uint32_t>(i),
                static_cast<uint32_t>(j),
                cfg.seed);
            if (rv < holdout_frac) {
                h_mask[p]         = 1;
                h_train_values[p] = 0.f;
                ++n_masked;
            } else {
                h_train_values[p] = h_values[p];
            }
        }
    }

    // --- Upload A_train values to device (structure reused from m.mat) ---
    core::DeviceMemory<float> d_train_values(nnz);
    cudaMemcpy(d_train_values.get(), h_train_values.data(),
        nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Build a cuSPARSE descriptor over A_train (same col_ptr / row_idx, zeroed values).
    cusparseSpMatDescr_t spA_train = nullptr;
    cusparseCreateCsc(
        &spA_train,
        (int64_t)mat_m, (int64_t)mat_n, (int64_t)nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        d_train_values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Wrap in a PzDeviceMatrix view so auto_select can accept it.
    // WHY from_device_ptrs: we share the existing col_ptr/row_idx device buffers —
    // they are non-owning pointers (false = do not free), keeping A_train structurally
    // identical to m.mat while substituting the value buffer.
    io::PzDeviceMatrix m_train;
    m_train.mat = core::DeviceCSC::from_device_ptrs(
        mat_m, mat_n, nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        d_train_values.get());
    // host_retained not needed for auto_select (deflation does not require it).
    m_train.host_retained = false;

    // Clamp k to valid range.
    int k_clamped = std::min(k, std::min(mat_m, mat_n));
    if (k_clamped <= 0) {
        cusparseDestroySpMat(spA_train);
        return SvdResult{};
    }

    SvdConfig cfg_k = cfg;
    cfg_k.k_max = k_clamped;

    // --- Fit truncated SVD on A_train ---
    SvdResult result = auto_select(m_train, k_clamped, cfg_k);

    cusparseDestroySpMat(spA_train);

    if (result.k_selected == 0) {
        result.test_loss  = 1e30f;
        result.train_loss = 1e30f;
        return result;
    }

    const int k_act = result.k_selected;

    // --- Upload U, d, V to device for MSE kernel ---
    core::DeviceMemory<float>   d_U(mat_m * k_act);
    core::DeviceMemory<float>   d_sv(k_act);
    core::DeviceMemory<float>   d_V(mat_n * k_act);
    core::DeviceMemory<uint8_t> d_mask(nnz);
    core::DeviceMemory<int32_t> d_col_ptr(mat_n + 1);
    core::DeviceMemory<int32_t> d_row_idx(nnz);
    core::DeviceMemory<float>   d_orig_values(nnz);

    cudaMemcpy(d_U.get(),           result.U_data.data(), mat_m * k_act * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sv.get(),          result.d.data_.data(), k_act * sizeof(float),          cudaMemcpyHostToDevice);
    cudaMemcpy(d_V.get(),           result.V_data.data(), mat_n * k_act * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask.get(),        h_mask.data(),         nnz * sizeof(uint8_t),           cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_ptr.get(),     h_col_ptr,             (mat_n + 1) * sizeof(int32_t),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_idx.get(),     h_row_idx,             nnz * sizeof(int32_t),            cudaMemcpyHostToDevice);
    cudaMemcpy(d_orig_values.get(), h_values,              nnz * sizeof(float),              cudaMemcpyHostToDevice);

    // --- Test MSE (held-out entries) ---
    result.test_loss = 1e30f;
    if (n_masked > 0) {
        core::DeviceMemory<float> d_loss_test(1);
        core::DeviceMemory<int>   d_count_test(1);
        float z0 = 0.f; int i0 = 0;
        cudaMemcpy(d_loss_test.get(),  &z0, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count_test.get(), &i0, sizeof(int),   cudaMemcpyHostToDevice);

        int blocks = (nnz + 255) / 256;
        svd_test_mse_kernel<<<blocks, 256>>>(
            d_mask.get(), d_col_ptr.get(), d_row_idx.get(), d_orig_values.get(),
            d_U.get(), d_sv.get(), d_V.get(),
            d_loss_test.get(), d_count_test.get(),
            mat_m, mat_n, k_act, nnz);

        cudaDeviceSynchronize();  // APPROVED: scalar result download, post-fit (one-time)

        float sq_err = 0.f; int cnt = 0;
        cudaMemcpy(&sq_err, d_loss_test.get(),  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cnt,    d_count_test.get(), sizeof(int),   cudaMemcpyDeviceToHost);
        result.test_loss = (cnt > 0) ? sq_err / cnt : 1e30f;
    }

    // --- Train MSE (unmasked entries) ---
    {
        core::DeviceMemory<float> d_loss_train(1);
        core::DeviceMemory<int>   d_count_train(1);
        float z0 = 0.f; int i0 = 0;
        cudaMemcpy(d_loss_train.get(),  &z0, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count_train.get(), &i0, sizeof(int),   cudaMemcpyHostToDevice);

        int blocks = (nnz + 255) / 256;
        svd_train_mse_kernel<<<blocks, 256>>>(
            d_mask.get(), d_col_ptr.get(), d_row_idx.get(), d_orig_values.get(),
            d_U.get(), d_sv.get(), d_V.get(),
            d_loss_train.get(), d_count_train.get(),
            mat_m, mat_n, k_act, nnz);

        cudaDeviceSynchronize();  // APPROVED: scalar result download, post-fit (one-time)

        float sq_err = 0.f; int cnt = 0;
        cudaMemcpy(&sq_err, d_loss_train.get(),  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&cnt,    d_count_train.get(), sizeof(int),   cudaMemcpyDeviceToHost);
        result.train_loss = (cnt > 0) ? sq_err / cnt : 1e30f;
    }

    return result;
}

// ---------------------------------------------------------------------------
// speckled_cv — sweep k_values, return CVSVDResult with chosen_k at minimum
// test MSE.
//
// Default k_values (empty → auto): {5, 10, 15, 20, 30, 50, 100}.
// Values exceeding min(m, n) are silently filtered.
// Same mask fixture reused across all k (fair comparison — same seed).
// ---------------------------------------------------------------------------
inline CVSVDResult speckled_cv(
    const io::PzDeviceMatrix& m,
    const SvdConfig&          cfg,
    const std::vector<int>&   k_values_in = {})
{
    require_host_retained(m, "svd::speckled_cv");

    std::vector<int> k_values = k_values_in;
    if (k_values.empty())
        k_values = {5, 10, 15, 20, 30, 50, 100};

    // Filter k > min(m, n) — SVD rank cannot exceed matrix dimension.
    const int max_rank = std::min(m.mat.rows, m.mat.cols);
    {
        std::vector<int> filtered;
        filtered.reserve(k_values.size());
        for (int k : k_values)
            if (k <= max_rank && k > 0)
                filtered.push_back(k);
        k_values = std::move(filtered);
    }

    CVSVDResult cv_result;
    cv_result.k_values = k_values;
    cv_result.test_mse.reserve(k_values.size());
    cv_result.train_mse.reserve(k_values.size());

    float best_test_mse = std::numeric_limits<float>::max();
    int   best_k        = k_values.empty() ? 0 : k_values.front();

    for (int k : k_values) {
        SvdResult r = cv_fit(m, k, cfg);

        cv_result.test_mse.push_back(r.test_loss);
        cv_result.train_mse.push_back(r.train_loss);

        if (r.test_loss < best_test_mse) {
            best_test_mse = r.test_loss;
            best_k = k;
        }
    }

    cv_result.chosen_k = best_k;
    return cv_result;
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
