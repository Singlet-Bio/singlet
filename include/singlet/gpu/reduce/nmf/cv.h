// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/reduce/nmf/cv.h
//
// Speckled cross-validation NMF — random-mask hold-out, vary k, find elbow.
// Replaces factornet::nmf::nmf_cv_fit_gpu adapter (CYCLE-105, 2026-04-29).
//
// // algorithm derived from factornet/nmf/fit_cv_gpu.cuh
// // Credit: Zach DeBruine, factornet 2021-2026, GPL-2.0
//
// Algorithm:
//   1. Randomly mask holdout_fraction of nonzero entries (deterministic per seed).
//   2. Replace masked values with 0 in A_train (NMF sees a sparser matrix).
//   3. Fit NMF on A_train at the given rank.
//   4. Predict held-out values as (W H)[i,j] for masked (i,j).
//   5. Compute test MSE = mean((A[i,j] - (WH)[i,j])^2) over held-out only.
//      Prediction step: (WH)_{ij} = sum_r W_{ir} H_{rj} — computed on device.
//   6. cv_fit returns a single NmfResult with test_loss populated.
//
//   speckled_cv sweeps k_values, calls cv_fit for each k, returns CVNMFResult.
//   chosen_k is the k with minimum test MSE (elbow in MSE curve).
//
// Speckled mask: O(1) query per entry via LCG hash — no materialization.
//   entry (i,j) is masked iff hash(i,j,seed) < holdout_fraction * 2^32.
//   This is equivalent to a uniform random sample without the memory overhead.
//
// Host↔device traffic:
//   Mask determination is on HOST at load time (once — not hot).
//   Test MSE evaluated on device via custom kernel after fit.
//   Result scalars downloaded once per k (approved: scalar result, not hot loop).
//
// Precision: fp32.
// Determinism: seeded via cfg.seed.
// OOC plan: full CSC must fit on device. For large matrices, run cv on a
//   representative chunk (e.g., first 50k cells).

#pragma once

#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/nmf/fit.h>
#include <singlet-gpu/reduce/svd/deflation.h>  // require_host_retained

#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Device kernel: compute test MSE over masked entries.
// ---------------------------------------------------------------------------
namespace {

// test_mse_kernel: for each masked nnz entry, add (true_val - W[i,:] . H[:,j])^2.
// mask_flags: uint8 device array, 1=masked, length nnz.
// col_ptr:    int32 device array, length n+1 (CSC col pointers).
// row_idx:    int32 device array, length nnz.
// values:     float device array, length nnz (original values incl. masked).
// W_dev:      m × k col-major.
// H_dev:      k × n col-major.
// d_loss:     float device scalar (accumulated with atomicAdd).
// d_count:    int   device scalar (count of masked entries).
__global__ void test_mse_kernel(
    const uint8_t* __restrict__ mask_flags,
    const int32_t* __restrict__ col_ptr,
    const int32_t* __restrict__ row_idx,
    const float*   __restrict__ values,
    const float*   __restrict__ W_dev,
    const float*   __restrict__ H_dev,
    float* __restrict__ d_loss,
    int*   __restrict__ d_count,
    int m, int n, int k, int nnz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nnz) return;
    if (!mask_flags[idx]) return;

    // Find column j for this nnz entry by binary search on col_ptr.
    // WHY: CSC format — col_ptr[j] ≤ idx < col_ptr[j+1].
    int lo = 0, hi = n;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (col_ptr[mid] <= idx) lo = mid; else hi = mid;
    }
    int j = lo;
    int i = row_idx[idx];
    float true_val = values[idx];

    // Predict (W H)_{ij} = sum_r W[i,r] * H[r,j]
    float pred = 0.f;
    const float* W_row = W_dev + i;      // W(i, r) = W_dev[i + r*m]
    const float* H_col = H_dev + j * k;  // H(r, j) = H_dev[r + j*k]
    for (int r = 0; r < k; ++r) {
        pred += W_row[r * m] * H_col[r];
    }

    float diff = true_val - pred;
    atomicAdd(d_loss, diff * diff);
    atomicAdd(d_count, 1);
}

}  // anonymous namespace

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// ---------------------------------------------------------------------------
// LCG hash: deterministic [0,1) pseudo-random per (i, j, seed).
// Equivalent to factornet's LazySpeckledMask.
// ---------------------------------------------------------------------------
static inline float lcg_hash_f32(uint32_t i, uint32_t j, uint64_t seed) {
    uint64_t h = seed;
    h ^= (uint64_t)i * 0x9e3779b97f4a7c15ULL;
    h ^= (uint64_t)j * 0x6c62272e07bb0142ULL;
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return static_cast<float>(h >> 11) * (1.0f / static_cast<float>(1ULL << 53));
}

// ---------------------------------------------------------------------------
// cv_fit — single-rank speckled CV fit.
//
// 1. Builds a modified A_train: zeroes out masked entries in a copy of values.
// 2. Constructs a device CSC from A_train (same indptr/indices, masked values).
// 3. Fits NMF on A_train.
// 4. Evaluates test MSE on masked entries using the original values.
// 5. Returns NmfResult with test_loss populated.
//
// Precondition: m must have host_retained=true (so we have h_col_ptr etc.).
// ---------------------------------------------------------------------------
inline NmfResult cv_fit(const io::PzDeviceMatrix& m,
                        const NmfConfig& cfg,
                        const DenseMatrix* W_init = nullptr,
                        const DenseMatrix* H_init = nullptr)
{
    require_host_retained(m, "nmf::cv_fit");

    const int mat_m = m.mat.rows;
    const int mat_n = m.mat.cols;
    const int nnz   = m.mat.nnz;

    const float holdout_frac = (cfg.holdout_fraction > 0.f && cfg.holdout_fraction < 1.f)
                               ? cfg.holdout_fraction : 0.05f;

    // --- Build A_train (host): zero out masked entries ---
    // mask_flags[p] = 1 if entry p (col_ptr, row_idx) is held out.
    const int32_t* h_col_ptr = m.host_indptr.get();
    const int32_t* h_row_idx = m.host_indices.get();
    const float*   h_values  = m.host_values.get();

    std::vector<float>   h_train_values(nnz);
    std::vector<uint8_t> h_mask(nnz, 0);

    int n_masked = 0;
    for (int j = 0; j < mat_n; ++j) {
        for (int p = h_col_ptr[j]; p < h_col_ptr[j + 1]; ++p) {
            int i = h_row_idx[p];
            float rand_val = lcg_hash_f32(
                static_cast<uint32_t>(i),
                static_cast<uint32_t>(j),
                cfg.seed);
            if (rand_val < holdout_frac) {
                h_mask[p]          = 1;
                h_train_values[p]  = 0.f;  // zero out held-out entry
                ++n_masked;
            } else {
                h_train_values[p] = h_values[p];
            }
        }
    }

    // --- Upload A_train to device (same structure, zeroed held-out entries) ---
    core::DeviceMemory<float> d_train_values(nnz);
    cudaMemcpy(d_train_values.get(), h_train_values.data(), nnz * sizeof(float),
        cudaMemcpyHostToDevice);

    // Build cuSPARSE descriptor for A_train (device col_ptr + row_idx from m.mat).
    cusparseSpMatDescr_t spA_train = nullptr;
    cusparseCreateCsc(
        &spA_train,
        (int64_t)mat_m, (int64_t)mat_n, (int64_t)nnz,
        m.mat.col_ptr.get(),
        m.mat.row_indices.get(),
        d_train_values.get(),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // --- Fit NMF on A_train ---
    // Build a PzDeviceMatrix view pointing at A_train.
    io::PzDeviceMatrix m_train;
    m_train.mat.col_ptr     = core::DeviceMemory<int32_t>(0);  // non-owning view
    m_train.mat.row_indices = core::DeviceMemory<int32_t>(0);
    m_train.mat.values      = core::DeviceMemory<float>(0);
    // Assign device pointers directly via the internal layout.
    // We need a PzDeviceMatrix to pass to fit(), but fit() now uses device
    // pointers from mat.col_ptr.get() etc. We'll call detail::run_mu_nmf directly.
    core::GPUContext ctx;
    NmfResult result = detail::run_mu_nmf(
        ctx, spA_train, mat_m, mat_n, cfg.rank, cfg,
        W_init, H_init,
        nullptr, nullptr, nullptr, nullptr);

    cusparseDestroySpMat(spA_train);

    // --- Evaluate test MSE on held-out entries ---
    if (n_masked > 0) {
        core::DeviceMemory<uint8_t> d_mask(nnz);
        core::DeviceMemory<int32_t> d_col_ptr(mat_n + 1);
        core::DeviceMemory<int32_t> d_row_idx(nnz);
        core::DeviceMemory<float>   d_orig_values(nnz);
        core::DeviceMemory<float>   d_W(mat_m * cfg.rank);
        core::DeviceMemory<float>   d_H(cfg.rank * mat_n);
        core::DeviceMemory<float>   d_loss(1);
        core::DeviceMemory<int>     d_count(1);

        cudaMemcpy(d_mask.get(),        h_mask.data(),     nnz * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_ptr.get(),     h_col_ptr,         (mat_n+1) * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_idx.get(),     h_row_idx,         nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_orig_values.get(), h_values,          nnz * sizeof(float),   cudaMemcpyHostToDevice);
        cudaMemcpy(d_W.get(), result.W.data.data(), mat_m * cfg.rank * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_H.get(), result.H.data.data(), cfg.rank * mat_n * sizeof(float), cudaMemcpyHostToDevice);

        float init_loss = 0.f; int init_count = 0;
        cudaMemcpy(d_loss.get(),  &init_loss,  sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_count.get(), &init_count, sizeof(int),   cudaMemcpyHostToDevice);

        int blocks = (nnz + 255) / 256;
        test_mse_kernel<<<blocks, 256>>>(
            d_mask.get(), d_col_ptr.get(), d_row_idx.get(), d_orig_values.get(),
            d_W.get(), d_H.get(),
            d_loss.get(), d_count.get(),
            mat_m, mat_n, cfg.rank, nnz);

        cudaDeviceSynchronize();  // ← APPROVED: scalar result download (one-time, post-fit)

        float total_sq_err = 0.f; int count = 0;
        cudaMemcpy(&total_sq_err, d_loss.get(),  sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&count,        d_count.get(), sizeof(int),   cudaMemcpyDeviceToHost);

        result.test_loss      = (count > 0) ? total_sq_err / count : 0.f;
        result.best_test_loss = result.test_loss;
        result.best_iter      = result.iterations;
    }

    return result;
}

// ---------------------------------------------------------------------------
// speckled_cv — sweep k_values, return CVNMFResult with chosen_k at elbow.
//
// k_values: candidate ranks to try. Default empty → auto-select {5,10,15,20,30}.
// cfg.holdout_fraction: fraction of nnz to hold out (default 0.05).
// ---------------------------------------------------------------------------
inline CVNMFResult speckled_cv(
    const io::PzDeviceMatrix& m,
    const NmfConfig& base_cfg,
    const std::vector<int>& k_values_in = {},
    const FitConfig& fit_cfg = FitConfig{})
{
    require_host_retained(m, "nmf::speckled_cv");

    std::vector<int> k_values = k_values_in;
    if (k_values.empty())
        k_values = {5, 10, 15, 20, 30};

    CVNMFResult cv_result;
    cv_result.k_values  = k_values;
    cv_result.test_mse.reserve(k_values.size());
    cv_result.train_mse.reserve(k_values.size());

    float best_test_mse = 1e30f;
    int   best_k        = k_values.front();

    for (int k : k_values) {
        NmfConfig cfg_k = base_cfg;
        cfg_k.rank = k;
        // Apply GPU-tuned solver routing (mirrors fit()).
        if (cfg_k.solver_mode == 3 && k >= fit_cfg.k_cd_cutoff)
            cfg_k.solver_mode = 2;

        NmfResult r = cv_fit(m, cfg_k, nullptr, nullptr);

        float train_mse = r.train_loss.empty() ? 0.f : r.train_loss.back();
        cv_result.test_mse.push_back(r.test_loss);
        cv_result.train_mse.push_back(train_mse);

        if (r.test_loss < best_test_mse) {
            best_test_mse = r.test_loss;
            best_k = k;
        }
    }

    cv_result.chosen_k = best_k;
    return cv_result;
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
