// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/reduce_nmf_regularization_correctness.cpp
//
// Structural correctness tests for the four regularization modes wired into
// the Frobenius MU-NMF kernel by CYCLE-114.
//
// Tolerance philosophy (from task spec):
//   No bit-equivalence to any reference — reg-NMF has no unique solution.
//   Structural assertions only:
//     L1=0.1  → sparsity ↑  (more |W| < 1e-3 than L1=0 baseline)
//     L2=0.1  → ‖W‖_F ↓    (smaller than L2=0 at same iteration count)
//     ortho=0.1 → off-diagonal of W^T W (col-normalized) ↓ vs ortho=0
//     All three combined → converges (loss decreases), no NaN
//
// Build: compiled as CUDA (see CMakeLists.txt).
// Run:  --gtest_filter=NmfTest.*Regularization*

#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/nmf/fit.h>
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
// Shared test fixtures
// =============================================================================

namespace {

constexpr uint64_t kRegSeed  = 0xABCDEF01ULL;
constexpr int      kRegRows  = 300;   // m
constexpr int      kRegCols  = 150;   // n
constexpr int      kRegK     = 8;     // rank
constexpr float    kDensity  = 0.08f; // ~8% fill: sparse but not degenerate
constexpr int      kRegIter  = 80;    // enough iters for structural effects to emerge

// ---------------------------------------------------------------------------
// Build a tiny synthetic sparse CSC on-device via PzDeviceMatrix.
// We use the host-side CSC helper and upload manually since there is no
// in-memory builder API — follow the pattern from reduce_nmf_correctness.cpp.
// ---------------------------------------------------------------------------
struct SynCSC {
    int   rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

SynCSC make_csc(int rows, int cols, float density, uint64_t seed)
{
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> vd(0.5f, 10.f);
    std::bernoulli_distribution           fd(static_cast<double>(density));

    SynCSC c;
    c.rows = rows; c.cols = cols;
    c.col_ptr.resize(cols + 1);
    c.col_ptr[0] = 0;
    for (int j = 0; j < cols; ++j) {
        bool had = false;
        for (int i = 0; i < rows; ++i) {
            if (fd(rng)) {
                c.row_idx.push_back(i);
                c.vals.push_back(vd(rng));
                had = true;
            }
        }
        if (!had) { c.row_idx.push_back(0); c.vals.push_back(1.f); }
        c.col_ptr[j+1] = static_cast<int32_t>(c.row_idx.size());
    }
    c.nnz = static_cast<int>(c.vals.size());
    return c;
}

// Upload a SynCSC to the GPU and wrap it in PzDeviceMatrix.
// Uses the 3-arg DeviceCSC constructor (allocates, then copies).
singlet_gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    using namespace singlet_gpu::io;

    PzDeviceMatrix pz;
    // DeviceCSC(m, n, nz) allocates all three device buffers via the RAII pool.
    pz.mat = singlet_gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);

    cudaMemcpy(pz.mat.col_ptr.get(), c.col_ptr.data(),
        (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
        c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.values.get(), c.vals.data(),
        c.nnz * sizeof(float), cudaMemcpyHostToDevice);

    return pz;
}

// Build a baseline NmfConfig with no regularization.
singlet_gpu::reduce::nmf::NmfConfig base_cfg(int iters = kRegIter)
{
    singlet_gpu::reduce::nmf::NmfConfig cfg;
    cfg.rank        = kRegK;
    cfg.max_iter    = iters;
    cfg.tol         = 0.f;    // disable early stopping: run all iters for comparisons
    cfg.seed        = static_cast<uint64_t>(kRegSeed);
    cfg.solver_mode = 2;      // MU explicit
    cfg.verbose     = false;
    return cfg;
}

// Frobenius norm of a host DenseMatrix (uses the raw data vector).
float frob(const singlet_gpu::reduce::nmf::DenseMatrix& M)
{
    float acc = 0.f;
    for (float v : M.data) acc += v * v;
    return std::sqrt(acc);
}

// Sparsity: fraction of |W| entries < threshold.
float sparsity(const singlet_gpu::reduce::nmf::DenseMatrix& W, float thr = 1e-3f)
{
    int cnt = 0;
    for (float v : W.data) if (std::fabs(v) < thr) ++cnt;
    return static_cast<float>(cnt) / static_cast<float>(W.data.size());
}

// Max absolute off-diagonal element of the col-normalised Gram WtW.
// Col-normalise each factor: W_norm[:,r] = W[:,r] / ||W[:,r]||_2.
// Then Gram = W_norm^T W_norm and we return max |G[i,j]| for i≠j.
float max_offdiag_gram(const singlet_gpu::reduce::nmf::DenseMatrix& W)
{
    int m = W.m, k = W.n;
    // Column norms
    std::vector<float> col_norms(k, 0.f);
    for (int r = 0; r < k; ++r) {
        float s = 0.f;
        for (int i = 0; i < m; ++i) { float v = W(i,r); s += v*v; }
        col_norms[r] = std::sqrt(s + 1e-15f);
    }
    // Gram (k×k): G[a,b] = dot(W_norm[:,a], W_norm[:,b])
    std::vector<float> G(k * k, 0.f);
    for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b) {
            float d = 0.f;
            for (int i = 0; i < m; ++i)
                d += W(i,a)/col_norms[a] * W(i,b)/col_norms[b];
            G[a + b*k] = d;
        }
    float mx = 0.f;
    for (int a = 0; a < k; ++a)
        for (int b = 0; b < k; ++b)
            if (a != b) mx = std::max(mx, std::fabs(G[a + b*k]));
    return mx;
}

// Check all values in DenseMatrix are finite and non-negative (NMF constraint).
bool is_valid_nmf_factor(const singlet_gpu::reduce::nmf::DenseMatrix& M)
{
    for (float v : M.data)
        if (!std::isfinite(v) || v < -1e-6f) return false;
    return true;
}

}  // anonymous namespace


// =============================================================================
// NmfTest.Regularization_L1_IncreasesSparsity
//
// L1=0.1 on W → fraction of |W| < 1e-3 must be strictly higher than baseline.
// =============================================================================
TEST(NmfTest, Regularization_L1_IncreasesSparsity)
{
    const auto csc = make_csc(kRegRows, kRegCols, kDensity, kRegSeed);
    auto pz = upload_csc(csc);

    // Baseline: no regularization
    auto cfg0 = base_cfg();
    const auto r0 = singlet_gpu::reduce::nmf::fit(pz, cfg0);
    ASSERT_TRUE(is_valid_nmf_factor(r0.W)) << "Baseline W has NaN/Inf or negative values";
    float sp0 = sparsity(r0.W);

    // L1 on W
    auto cfg1 = base_cfg();
    cfg1.W.L1 = 0.1f;
    const auto r1 = singlet_gpu::reduce::nmf::fit(pz, cfg1);
    ASSERT_TRUE(is_valid_nmf_factor(r1.W))
        << "L1-reg W has NaN/Inf or negative values";
    float sp1 = sparsity(r1.W);

    EXPECT_GT(sp1, sp0)
        << "L1=0.1 should increase sparsity: sp_L1=" << sp1
        << " sp_baseline=" << sp0;

    std::printf("[REGISTRY] %s | reduce/nmf/reg_L1 | tiny | sparsity_ratio"
                " | %.3f | 0 | none | CYCLE-114 | %s"
                " (sp_baseline=%.3f sp_L1=%.3f)\n",
                __DATE__, sp1 - sp0,
                sp1 > sp0 ? "PASS" : "FAIL", sp0, sp1);
}


// =============================================================================
// NmfTest.Regularization_L2_ShrinksNorm
//
// L2=0.1 on W → ‖W‖_F must be strictly smaller than baseline.
// =============================================================================
TEST(NmfTest, Regularization_L2_ShrinksNorm)
{
    const auto csc = make_csc(kRegRows, kRegCols, kDensity, kRegSeed);
    auto pz = upload_csc(csc);

    auto cfg0 = base_cfg();
    const auto r0 = singlet_gpu::reduce::nmf::fit(pz, cfg0);
    ASSERT_TRUE(is_valid_nmf_factor(r0.W)) << "Baseline W has NaN/Inf";
    float norm0 = frob(r0.W);

    auto cfg2 = base_cfg();
    cfg2.W.L2 = 0.1f;
    const auto r2 = singlet_gpu::reduce::nmf::fit(pz, cfg2);
    ASSERT_TRUE(is_valid_nmf_factor(r2.W)) << "L2-reg W has NaN/Inf";
    float norm2 = frob(r2.W);

    EXPECT_LT(norm2, norm0)
        << "L2=0.1 should shrink ||W||_F: norm_L2=" << norm2
        << " norm_baseline=" << norm0;

    std::printf("[REGISTRY] %s | reduce/nmf/reg_L2 | tiny | norm_ratio"
                " | %.3f | 0 | none | CYCLE-114 | %s"
                " (norm_baseline=%.3f norm_L2=%.3f)\n",
                __DATE__, norm0 - norm2,
                norm2 < norm0 ? "PASS" : "FAIL", norm0, norm2);
}


// =============================================================================
// NmfTest.Regularization_Ortho_ReducesOffDiagGram
//
// ortho=0.1 on W → max off-diagonal of col-normalised W^T W must shrink.
// =============================================================================
TEST(NmfTest, Regularization_Ortho_ReducesOffDiagGram)
{
    const auto csc = make_csc(kRegRows, kRegCols, kDensity, kRegSeed);
    auto pz = upload_csc(csc);

    auto cfg0 = base_cfg();
    const auto r0 = singlet_gpu::reduce::nmf::fit(pz, cfg0);
    ASSERT_TRUE(is_valid_nmf_factor(r0.W)) << "Baseline W has NaN/Inf";
    float od0 = max_offdiag_gram(r0.W);

    auto cfg3 = base_cfg();
    cfg3.W.ortho = 0.1f;
    const auto r3 = singlet_gpu::reduce::nmf::fit(pz, cfg3);
    ASSERT_TRUE(is_valid_nmf_factor(r3.W)) << "Ortho-reg W has NaN/Inf";
    float od3 = max_offdiag_gram(r3.W);

    EXPECT_LT(od3, od0)
        << "ortho=0.1 should reduce max off-diagonal of W^TW: ortho=" << od3
        << " baseline=" << od0;

    std::printf("[REGISTRY] %s | reduce/nmf/reg_ortho | tiny | offdiag_gram"
                " | %.4f | 0 | none | CYCLE-114 | %s"
                " (baseline=%.4f ortho=%.4f)\n",
                __DATE__, od0 - od3,
                od3 < od0 ? "PASS" : "FAIL", od0, od3);
}


// =============================================================================
// NmfTest.Regularization_CombinedConvergesNoNaN
//
// L1=0.1, L2=0.1, ortho=0.1 on both W and H simultaneously:
//   (a) final W and H are finite and non-negative.
//   (b) loss decreases overall (final < initial, after warm-up).
//   (c) no NaN in loss history.
// =============================================================================
TEST(NmfTest, Regularization_CombinedConvergesNoNaN)
{
    const auto csc = make_csc(kRegRows, kRegCols, kDensity, kRegSeed);
    auto pz = upload_csc(csc);

    auto cfg = base_cfg(/*iters=*/100);
    cfg.tol   = 0.f;    // disable early stopping to get full history
    cfg.W.L1  = 0.1f;
    cfg.W.L2  = 0.1f;
    cfg.W.ortho = 0.1f;
    cfg.H.L1  = 0.1f;
    cfg.H.L2  = 0.1f;
    cfg.H.ortho = 0.1f;

    const auto r = singlet_gpu::reduce::nmf::fit(pz, cfg);

    // (a) finite and non-negative
    ASSERT_TRUE(is_valid_nmf_factor(r.W))
        << "Combined-reg: W has NaN/Inf or negative values";
    ASSERT_TRUE(is_valid_nmf_factor(r.H))
        << "Combined-reg: H has NaN/Inf or negative values";

    // (b) loss_history non-empty and no NaN entries
    ASSERT_FALSE(r.train_loss.empty())
        << "Combined-reg: train_loss is empty (no iterations ran)";
    for (int i = 0; i < static_cast<int>(r.train_loss.size()); ++i) {
        EXPECT_TRUE(std::isfinite(r.train_loss[i]))
            << "Combined-reg: NaN/Inf in train_loss at iter " << i;
    }

    // (c) overall loss decrease: final < initial (allow warm-up, check last half)
    if (r.train_loss.size() >= 10u) {
        float first_half_min = *std::min_element(
            r.train_loss.begin(),
            r.train_loss.begin() + static_cast<int>(r.train_loss.size() / 2));
        float second_half_min = *std::min_element(
            r.train_loss.begin() + static_cast<int>(r.train_loss.size() / 2),
            r.train_loss.end());
        EXPECT_LT(second_half_min, first_half_min)
            << "Combined-reg: loss did not decrease from first half to second half";
    }

    std::printf("[REGISTRY] %s | reduce/nmf/reg_combined | tiny | convergence"
                " | 1 | 0 | none | CYCLE-114 | %s"
                " (iters=%d W_finite=%s H_finite=%s)\n",
                __DATE__,
                (is_valid_nmf_factor(r.W) && is_valid_nmf_factor(r.H)) ? "PASS" : "FAIL",
                r.iterations,
                is_valid_nmf_factor(r.W) ? "yes" : "NO",
                is_valid_nmf_factor(r.H) ? "yes" : "NO");
}
