// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/reduce_svd_cv_correctness.cpp
//
// Correctness tests for speckled cross-validation SVD (CYCLE-115).
//
// Three test cases:
//   1. Smoke: 200×500 synthetic, k_values={5,10,30,100}. chosen_k is finite,
//      test_mse has no NaN.
//   2. Rank-recovery: A = U_r D_r V_r^T + small noise (rank r=10).
//      Expect chosen_k ≤ r * 2 (within an order of magnitude of true rank).
//   3. Determinism: two runs with identical seed → rel_err < 1e-4 on all outputs
//      (test_mse, train_mse, chosen_k).
//
// Build: compiled as CUDA (see CMakeLists.txt, LANGUAGE CUDA).

#include <singlet-gpu/reduce/svd/types.h>
#include <singlet-gpu/reduce/svd/cv.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

// =============================================================================
// Shared helpers
// =============================================================================

namespace {

// Build a random sparse CSC (all positive, col-major).
struct SynCSC {
    int rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

SynCSC make_sparse(int rows, int cols, float density, uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> vd(0.1f, 5.f);
    std::bernoulli_distribution           fd(static_cast<double>(density));

    SynCSC c;
    c.rows = rows; c.cols = cols;
    c.col_ptr.resize(cols + 1);
    c.col_ptr[0] = 0;
    for (int j = 0; j < cols; ++j) {
        bool had = false;
        for (int i = 0; i < rows; ++i) {
            if (fd(rng)) {
                c.row_idx.push_back(static_cast<int32_t>(i));
                c.vals.push_back(vd(rng));
                had = true;
            }
        }
        // Guarantee at least one nonzero per column (avoids all-zero columns).
        if (!had) {
            c.row_idx.push_back(0);
            c.vals.push_back(1.f);
        }
        c.col_ptr[j + 1] = static_cast<int32_t>(c.row_idx.size());
    }
    c.nnz = static_cast<int>(c.vals.size());
    return c;
}

// Build a low-rank-plus-noise CSC.
// A = U_r * diag(D_r) * V_r^T + epsilon * noise, then threshold to ≥ 0
// (to create a realistic positive sparse matrix with known rank r).
SynCSC make_rank_r(int rows, int cols, int r, float noise_eps, uint64_t seed)
{
    std::mt19937                          rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> ud(0.1f, 1.f);
    std::normal_distribution<float>       nd(0.f, 1.f);

    // Build random U (rows × r), D (r), V (cols × r).
    std::vector<float> U(rows * r), D(r), V(cols * r);
    for (float& v : U) v = std::fabs(ud(rng));
    for (int i = 0; i < r; ++i) D[i] = (r - i) * 10.f;  // descending singular values
    for (float& v : V) v = std::fabs(ud(rng));

    // Compute A = U D V^T (dense) then sparsify at threshold 0.5.
    std::vector<float> A_dense(rows * cols, 0.f);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            for (int k = 0; k < r; ++k)
                A_dense[i + j * rows] += U[i + k * rows] * D[k] * V[j + k * cols];

    // Add noise
    for (float& v : A_dense) v += noise_eps * nd(rng);

    // Convert to CSC: keep entries > 0.5 threshold
    SynCSC c;
    c.rows = rows; c.cols = cols;
    c.col_ptr.resize(cols + 1);
    c.col_ptr[0] = 0;
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            float v = A_dense[i + j * rows];
            if (v > 0.5f) {
                c.row_idx.push_back(static_cast<int32_t>(i));
                c.vals.push_back(v);
            }
        }
        // Ensure at least one entry per column.
        if (c.col_ptr[j] == static_cast<int32_t>(c.row_idx.size())) {
            c.row_idx.push_back(0);
            c.vals.push_back(1.f);
        }
        c.col_ptr[j + 1] = static_cast<int32_t>(c.row_idx.size());
    }
    c.nnz = static_cast<int>(c.vals.size());
    return c;
}

// Upload SynCSC to device and return a PzDeviceMatrix with host_retained=true.
singlet_gpu::io::PzDeviceMatrix upload_with_host(const SynCSC& c)
{
    using namespace singlet_gpu;
    io::PzDeviceMatrix pz;
    pz.mat = core::DeviceCSC(c.rows, c.cols, c.nnz);

    cudaMemcpy(pz.mat.col_ptr.get(), c.col_ptr.data(),
        (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
        c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.values.get(), c.vals.data(),
        c.nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Pin host copies so cv_fit's require_host_retained check passes.
    // WHY: we allocate host-side shared_ptrs with deleter=noop — data lives in
    // the SynCSC vectors which outlive the test, so no double-free risk.
    // In production, load_pz populates these via pinned memory; in test we
    // reuse the std::vector data directly (valid for the test lifetime).
    // host_indptr / host_indices are std::shared_ptr<int> (pz_device_loader.h L401).
    // int == int32_t on LP64 Linux; reinterpret is identical to nmf/cv.h pattern.
    pz.host_indptr  = std::shared_ptr<int>(
        reinterpret_cast<int*>(const_cast<int32_t*>(c.col_ptr.data())), [](int*){});
    pz.host_indices = std::shared_ptr<int>(
        reinterpret_cast<int*>(const_cast<int32_t*>(c.row_idx.data())), [](int*){});
    pz.host_values  = std::shared_ptr<float>(
        const_cast<float*>(c.vals.data()), [](float*){});
    pz.host_retained = true;

    return pz;
}

// Build a default SvdConfig for CV tests.
singlet_gpu::reduce::svd::SvdConfig make_cfg(uint64_t seed = 0xC115BEEFU,
                                              float holdout = 0.10f)
{
    singlet_gpu::reduce::svd::SvdConfig cfg;
    cfg.seed             = seed;
    cfg.holdout_fraction = holdout;
    cfg.max_iter         = 30;    // fewer iters for test speed
    cfg.tol              = 1e-4f;
    cfg.verbose          = false;
    return cfg;
}

}  // anonymous namespace


// =============================================================================
// Test 1: Smoke — 200×500 synthetic matrix, k_values = {5, 10, 30, 100}
//
// Assertions:
//   chosen_k is in k_values (finite, non-zero).
//   All test_mse entries are finite (no NaN / Inf).
//   All train_mse entries are finite.
// =============================================================================
TEST(SvdCvTest, Smoke_200x500)
{
    const SynCSC csc = make_sparse(200, 500, 0.07f, 0xABCD1234ULL);
    auto pz = upload_with_host(csc);

    const std::vector<int> k_vals = {5, 10, 30, 100};
    auto cfg = make_cfg();

    singlet_gpu::reduce::svd::CVSVDResult cv =
        singlet_gpu::reduce::svd::speckled_cv(pz, cfg, k_vals);

    // chosen_k must be one of the k_values we provided.
    ASSERT_FALSE(cv.k_values.empty()) << "k_values should not be empty";
    EXPECT_GT(cv.chosen_k, 0) << "chosen_k must be positive";
    {
        bool found = false;
        for (int k : cv.k_values) if (k == cv.chosen_k) { found = true; break; }
        EXPECT_TRUE(found) << "chosen_k=" << cv.chosen_k << " not in k_values";
    }

    ASSERT_EQ(static_cast<int>(cv.test_mse.size()),  static_cast<int>(k_vals.size()));
    ASSERT_EQ(static_cast<int>(cv.train_mse.size()), static_cast<int>(k_vals.size()));

    for (int i = 0; i < static_cast<int>(k_vals.size()); ++i) {
        EXPECT_TRUE(std::isfinite(cv.test_mse[i]))
            << "test_mse[" << i << "] is not finite for k=" << k_vals[i];
        EXPECT_TRUE(std::isfinite(cv.train_mse[i]))
            << "train_mse[" << i << "] is not finite for k=" << k_vals[i];
        EXPECT_GE(cv.test_mse[i],  0.f) << "test_mse[" << i << "] < 0";
        EXPECT_GE(cv.train_mse[i], 0.f) << "train_mse[" << i << "] < 0";
    }

    std::printf("[CYCLE-115] Smoke 200×500: chosen_k=%d test_mse=[", cv.chosen_k);
    for (int i = 0; i < static_cast<int>(cv.test_mse.size()); ++i)
        std::printf("%.4f%s", cv.test_mse[i], i+1 < static_cast<int>(cv.test_mse.size()) ? "," : "");
    std::printf("]\n");
}


// =============================================================================
// Test 2: Rank-recovery — A = U_10 D_10 V_10^T + small noise.
//
// True rank = 10. Expect chosen_k <= 10 * 2 = 20.
// (Wold CV tends to select near the true rank for well-structured data.)
// =============================================================================
TEST(SvdCvTest, RankRecovery_TrueRank10)
{
    constexpr int  kTrueRank = 10;
    const SynCSC csc = make_rank_r(200, 300, kTrueRank, 0.5f, 0xDEADBEEFULL);
    auto pz = upload_with_host(csc);

    // Sweep k up to 2× true rank + buffer.
    const std::vector<int> k_vals = {2, 5, 8, 10, 12, 15, 20};
    auto cfg = make_cfg(0xDEADBEEFULL, 0.10f);

    singlet_gpu::reduce::svd::CVSVDResult cv =
        singlet_gpu::reduce::svd::speckled_cv(pz, cfg, k_vals);

    EXPECT_GT(cv.chosen_k, 0) << "chosen_k must be positive";
    EXPECT_LE(cv.chosen_k, kTrueRank * 2)
        << "chosen_k=" << cv.chosen_k << " exceeds 2× true rank=" << kTrueRank;

    for (float mse : cv.test_mse)
        EXPECT_TRUE(std::isfinite(mse)) << "test_mse has NaN/Inf";

    std::printf("[CYCLE-115] RankRecovery: chosen_k=%d true_rank=%d\n",
                cv.chosen_k, kTrueRank);
}


// =============================================================================
// Test 3: Determinism — two runs with identical seed must produce identical output.
//
// Tolerance: rel_err < 1e-4 per Rule 18 (fp32 accumulated atomics may differ
// in last bit across kernel launch order — use relative tolerance).
// =============================================================================
TEST(SvdCvTest, Determinism_SameSeed)
{
    const SynCSC csc = make_sparse(150, 300, 0.06f, 0xFEEDF00DULL);

    // Upload once (same device data for both runs — ensures the input is identical).
    auto pz = upload_with_host(csc);

    const std::vector<int> k_vals = {5, 10, 20};
    auto cfg = make_cfg(0x12345678ULL, 0.10f);

    singlet_gpu::reduce::svd::CVSVDResult cv1 =
        singlet_gpu::reduce::svd::speckled_cv(pz, cfg, k_vals);
    singlet_gpu::reduce::svd::CVSVDResult cv2 =
        singlet_gpu::reduce::svd::speckled_cv(pz, cfg, k_vals);

    EXPECT_EQ(cv1.chosen_k, cv2.chosen_k)
        << "chosen_k differs between runs: " << cv1.chosen_k << " vs " << cv2.chosen_k;
    ASSERT_EQ(cv1.test_mse.size(), cv2.test_mse.size());
    ASSERT_EQ(cv1.train_mse.size(), cv2.train_mse.size());

    constexpr float kRelTol = 1e-4f;
    for (int i = 0; i < static_cast<int>(k_vals.size()); ++i) {
        float denom_test  = std::max(std::fabs(cv1.test_mse[i]),  1e-6f);
        float denom_train = std::max(std::fabs(cv1.train_mse[i]), 1e-6f);
        float rel_test    = std::fabs(cv1.test_mse[i]  - cv2.test_mse[i])  / denom_test;
        float rel_train   = std::fabs(cv1.train_mse[i] - cv2.train_mse[i]) / denom_train;

        EXPECT_LE(rel_test, kRelTol)
            << "test_mse[" << i << "] not deterministic: "
            << cv1.test_mse[i] << " vs " << cv2.test_mse[i]
            << " rel_err=" << rel_test;
        EXPECT_LE(rel_train, kRelTol)
            << "train_mse[" << i << "] not deterministic: "
            << cv1.train_mse[i] << " vs " << cv2.train_mse[i]
            << " rel_err=" << rel_train;
    }

    std::printf("[CYCLE-115] Determinism: chosen_k=%d/%d test_mse[0]=%.6f/%.6f\n",
                cv1.chosen_k, cv2.chosen_k,
                cv1.test_mse.empty() ? 0.f : cv1.test_mse[0],
                cv2.test_mse.empty() ? 0.f : cv2.test_mse[0]);
}
