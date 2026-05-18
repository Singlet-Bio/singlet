// SPDX-License-Identifier: MIT
// singlet/gpu/tests/qc_soupx_correctness.cpp
//
// integrates: SoupX ambient RNA correction (Young & Behjati 2020 GigaScience 9:giaa151)
//
// Correctness harness for singlet::gpu::qc::soupx.
//
// Tests:
//   1. Soupx_TinyClosedForm
//      5 genes × 8 droplets; 4 empty (t<100), 4 real cells.
//      Hand-constructed π, ρ_c, corrected; verified to abs_err < 1e-3.
//
//   2. Soupx_NoContamination_RhoZero
//      Cells have NO top-ambient-gene expression.
//      ρ_c ≈ 0; corrected ≈ original stored counts.
//
//   3. Soupx_HighContamination_RhoLarge
//      Cells are a 50% mix of pure cell signal + ambient.
//      ρ_c should approach 0.5; corrected[g,c] ≈ pure cell signal.
//
//   4. Soupx_Determinism_BitIdentical
//      Two identical runs; corrected output is bit-for-bit identical.
//
//   5. Soupx_MemoryGuard_RejectsTooLarge
//      100000 × 100000 synthetic dims → throws runtime_error.

#include <singlet/gpu/qc/soupx.h>
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/handles.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// =============================================================================
// Utility helpers
// =============================================================================

namespace {

bool gpu_available() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    return (e == cudaSuccess && count > 0);
}

struct HostCSC {
    int m, n;
    std::vector<float>   values;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_indices;
};

HostCSC dense_to_csc(const std::vector<float>& dense, int m, int n) {
    HostCSC h;
    h.m = m; h.n = n;
    h.col_ptr.resize(static_cast<size_t>(n) + 1, 0);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float v = dense[static_cast<size_t>(i) + static_cast<size_t>(j) * m];
            if (v > 0.f) { h.values.push_back(v); h.row_indices.push_back(i); }
        }
        h.col_ptr[j + 1] = static_cast<int32_t>(h.values.size());
    }
    return h;
}

singlet::gpu::io::PzDeviceMatrix upload_csc(const HostCSC& h, cudaStream_t stream) {
    using namespace singlet::gpu;
    const int64_t nnz = static_cast<int64_t>(h.values.size());
    io::PzDeviceMatrix pz;
    pz.mat.rows = h.m;
    pz.mat.cols = h.n;
    pz.mat.nnz  = static_cast<int>(nnz);
    pz.mat.values      = core::DeviceMemory<float>(nnz > 0 ? nnz : 1);
    pz.mat.row_indices = core::DeviceMemory<int32_t>(nnz > 0 ? nnz : 1);
    pz.mat.col_ptr     = core::DeviceMemory<int32_t>(static_cast<size_t>(h.n) + 1);
    if (nnz > 0) {
        cudaMemcpyAsync(pz.mat.values.get(), h.values.data(),
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(pz.mat.row_indices.get(), h.row_indices.data(),
                        nnz * sizeof(int32_t), cudaMemcpyHostToDevice, stream);
    }
    cudaMemcpyAsync(pz.mat.col_ptr.get(), h.col_ptr.data(),
                    (static_cast<size_t>(h.n) + 1) * sizeof(int32_t),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    return pz;
}

std::vector<float> download_f32(const singlet::gpu::core::DeviceMemory<float>& d,
                                 size_t count, cudaStream_t stream) {
    std::vector<float> h(count);
    cudaMemcpyAsync(h.data(), d.get(), count * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

} // anonymous namespace

// =============================================================================
// Test fixture
// =============================================================================
class SoupxTest : public ::testing::Test {
protected:
    cudaStream_t stream_{nullptr};
    void SetUp() override {
        if (!gpu_available()) return;
        cudaStreamCreate(&stream_);
    }
    void TearDown() override {
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    }
};

// =============================================================================
// Test 1: Soupx_TinyClosedForm
// 5 genes × 8 droplets. Columns 0-3 are empty (t=10), columns 4-7 are cells.
//
// Hand-constructed data (col-major, gene × droplet):
//   Empty droplets (cols 0-3), each: gene0=6, gene1=2, gene2=1, gene3=1, gene4=0
//   → total = 10 per empty droplet
//   → ambient a[g] = sum over 4 empties: [24,8,4,4,0]  total=40
//   → π = [0.6, 0.2, 0.1, 0.1, 0.0]
//
//   top 10% of 5 genes = top 1 gene = gene0 (π[0]=0.6)
//   denom_pi = 0.6
//
//   Cell col 4: gene0=60, gene1=20, gene2=10, gene3=10, gene4=100  t=200
//   → num = X[gene0,c4] = 60  (only gene0 in top mask)
//   → ρ_c4 = 60 / (200 * 0.6) = 0.5
//   → corrected[gene4, c4] = max(0, 100 - 0.5*200*0.0) = 100
//   → corrected[gene0, c4] = max(0, 60 - 0.5*200*0.6) = max(0, 60-60) = 0
//   → corrected[gene1, c4] = max(0, 20 - 0.5*200*0.2) = max(0, 20-20) = 0
//
// Verifies: π abs_err < 1e-3, ρ_c abs_err < 1e-3, corrected abs_err < 1.0
// =============================================================================
TEST_F(SoupxTest, Soupx_TinyClosedForm) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M = 5, N = 8;
    // col-major dense, gene × droplet
    std::vector<float> dense(M * N, 0.f);
    // Empties (cols 0-3): gene0=6, gene1=2, gene2=1, gene3=1, gene4=0
    for (int j = 0; j < 4; ++j) {
        dense[0 + j * M] = 6.f;
        dense[1 + j * M] = 2.f;
        dense[2 + j * M] = 1.f;
        dense[3 + j * M] = 1.f;
    }
    // Cell col 4: gene0=60, gene1=20, gene2=10, gene3=10, gene4=100
    dense[0 + 4 * M] = 60.f;
    dense[1 + 4 * M] = 20.f;
    dense[2 + 4 * M] = 10.f;
    dense[3 + 4 * M] = 10.f;
    dense[4 + 4 * M] = 100.f;
    // Cells 5-7: uniform signal on gene4 only, no ambient genes
    for (int j = 5; j < 8; ++j)
        dense[4 + j * M] = 150.f;

    HostCSC h = dense_to_csc(dense, M, N);
    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::SoupxConfig cfg;
    cfg.lower            = 50;    // empties have t=10; cells have t=200+
    cfg.top_ambient_frac = 0.20f; // top 20% of 5 genes = top 1 gene (gene0)
    cfg.min_rho          = 0.0f;
    cfg.max_rho          = 0.9f;

    auto res = singlet::gpu::qc::soupx(pz, cfg, stream_);

    ASSERT_EQ(res.m, M);
    ASSERT_EQ(res.n, N);

    auto h_pi   = download_f32(res.ambient_pi, M, stream_);
    auto h_rho  = download_f32(res.rho_c,      N, stream_);
    auto h_corr = download_f32(res.corrected,  static_cast<size_t>(M) * N, stream_);

    // Verify π
    const float expected_pi[5] = {0.6f, 0.2f, 0.1f, 0.1f, 0.0f};
    for (int g = 0; g < M; ++g)
        EXPECT_NEAR(h_pi[g], expected_pi[g], 1e-3f) << "pi[" << g << "]";

    // Verify ρ_c for col 4 ≈ 0.5
    EXPECT_NEAR(h_rho[4], 0.5f, 1e-3f) << "rho_c[4]";

    // Verify corrected[gene4, col4] ≈ 100 (no ambient on gene4)
    EXPECT_NEAR(h_corr[4 + 4 * M], 100.f, 1.0f) << "corrected[gene4, col4]";

    // Verify corrected[gene0, col4] ≈ 0 (fully subtracted)
    EXPECT_NEAR(h_corr[0 + 4 * M], 0.f, 1.0f) << "corrected[gene0, col4]";

    // Cells 5-7: ρ_c should be near 0 (no top-ambient-gene expression)
    for (int j = 5; j < 8; ++j)
        EXPECT_NEAR(h_rho[j], 0.f, 0.01f) << "rho_c[" << j << "]";

    std::printf("[TinyClosedForm] pi=[%.3f,%.3f,%.3f,%.3f,%.3f] rho_c4=%.3f "
                "corr[gene4,c4]=%.1f corr[gene0,c4]=%.1f\n",
                h_pi[0],h_pi[1],h_pi[2],h_pi[3],h_pi[4],
                h_rho[4], h_corr[4+4*M], h_corr[0+4*M]);
}

// =============================================================================
// Test 2: Soupx_NoContamination_RhoZero
// Cells express ONLY genes with low π (not in top mask).
// ρ_c should be ≈ 0; corrected ≈ original.
// =============================================================================
TEST_F(SoupxTest, Soupx_NoContamination_RhoZero) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M = 20, N = 30;
    // 10 empty droplets: express only genes 0-1 heavily
    // 20 cells: express only genes 10-19 (no overlap with top-ambient)
    std::vector<float> dense(M * N, 0.f);
    for (int j = 0; j < 10; ++j) {
        dense[0 + j * M] = 30.f;
        dense[1 + j * M] = 10.f;
    }
    for (int j = 10; j < 30; ++j) {
        for (int g = 10; g < 20; ++g)
            dense[g + j * M] = 50.f;
    }

    HostCSC h = dense_to_csc(dense, M, N);
    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::SoupxConfig cfg;
    cfg.lower            = 50;
    cfg.top_ambient_frac = 0.10f;  // top 10% of 20 = top 2 genes (gene0, gene1)
    cfg.min_rho          = 0.0f;
    cfg.max_rho          = 0.9f;

    auto res = singlet::gpu::qc::soupx(pz, cfg, stream_);

    auto h_rho  = download_f32(res.rho_c,     N, stream_);
    auto h_corr = download_f32(res.corrected, static_cast<size_t>(M) * N, stream_);

    // All cells (cols 10-29) should have rho ≈ 0
    for (int j = 10; j < 30; ++j)
        EXPECT_NEAR(h_rho[j], 0.f, 0.01f) << "rho_c[" << j << "]";

    // Corrected values for cell genes should remain ≈ 50
    double max_err = 0.0;
    for (int j = 10; j < 30; ++j)
        for (int g = 10; g < 20; ++g) {
            float c = h_corr[g + j * M];
            max_err = std::max(max_err, std::abs(static_cast<double>(c) - 50.0));
        }
    std::printf("[NoContamination] max corrected err=%.4f (expect≈0)\n", max_err);
    EXPECT_LT(max_err, 1.0) << "corrected values should stay near original";
}

// =============================================================================
// Test 3: Soupx_HighContamination_RhoLarge
// Cells are a 50% mixture: true_signal + 0.5 * ambient_contribution.
// ρ_c should approach 0.5; corrected should recover the true signal.
// =============================================================================
TEST_F(SoupxTest, Soupx_HighContamination_RhoLarge) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M = 10, N = 20;
    // Empty droplets: cols 0-9, t=10, all expression on gene0
    // → π ≈ [1.0, 0, 0, ...0]
    std::vector<float> dense(M * N, 0.f);
    for (int j = 0; j < 10; ++j)
        dense[0 + j * M] = 10.f;

    // Cells: cols 10-19.
    // True signal = 100 on gene5. Contamination = 50% ambient on gene0.
    // observed: gene0=100, gene5=100 (so t=200, rho=100/(200*1.0)=0.5)
    for (int j = 10; j < 20; ++j) {
        dense[0 + j * M] = 100.f;   // ambient component
        dense[5 + j * M] = 100.f;   // true signal
    }

    HostCSC h = dense_to_csc(dense, M, N);
    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::SoupxConfig cfg;
    cfg.lower            = 15;   // t=10 for empty < 15; cells have t=200 > 15
    cfg.top_ambient_frac = 0.10f; // top 10% of 10 genes = top 1 gene (gene0)
    cfg.min_rho          = 0.0f;
    cfg.max_rho          = 0.9f;

    auto res = singlet::gpu::qc::soupx(pz, cfg, stream_);

    auto h_rho  = download_f32(res.rho_c,     N, stream_);
    auto h_corr = download_f32(res.corrected, static_cast<size_t>(M) * N, stream_);

    // Check cells have ρ ≈ 0.5
    double avg_rho = 0.0;
    for (int j = 10; j < 20; ++j) avg_rho += h_rho[j];
    avg_rho /= 10.0;
    std::printf("[HighContamination] avg_rho=%.3f (expect≈0.5)\n", avg_rho);
    EXPECT_NEAR(avg_rho, 0.5, 0.05) << "avg rho should be near 0.5";

    // corrected[gene0, cell] ≈ 0 (ambient subtracted)
    // corrected[gene5, cell] ≈ 100 (true signal, π[gene5]=0 → no subtraction)
    for (int j = 10; j < 20; ++j) {
        EXPECT_NEAR(h_corr[0 + j * M], 0.f, 5.f)   << "corrected[gene0,c" << j << "]";
        EXPECT_NEAR(h_corr[5 + j * M], 100.f, 5.f) << "corrected[gene5,c" << j << "]";
    }
}

// =============================================================================
// Test 4: Soupx_Determinism_BitIdentical
// Two identical runs on the same matrix. rel_err = 0.
// =============================================================================
TEST_F(SoupxTest, Soupx_Determinism_BitIdentical) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    constexpr int M = 50, N = 80;
    std::vector<float> dense(M * N, 0.f);
    // Cells 0..19: empty droplets — only 5 genes with small values, total UMI ≈ 5
    for (int j = 0; j < 20; ++j)
        for (int g = 0; g < 5; ++g)
            dense[g + j * M] = 1.f;  // t_j = 5 < lower=20
    // Cells 20..79: real cells — 50 genes with larger values, total UMI ≈ 200
    for (int j = 20; j < 80; ++j)
        for (int g = 0; g < M; ++g)
            dense[g + j * M] = static_cast<float>((g + j) % 7 + 1);  // t_j ≈ 200

    HostCSC h = dense_to_csc(dense, M, N);
    auto pz = upload_csc(h, stream_);

    singlet::gpu::qc::SoupxConfig cfg;
    cfg.lower = 20;

    auto res_a = singlet::gpu::qc::soupx(pz, cfg, stream_);
    auto res_b = singlet::gpu::qc::soupx(pz, cfg, stream_);

    const size_t sz = static_cast<size_t>(M) * N;
    auto h_ca = download_f32(res_a.corrected, sz, stream_);
    auto h_cb = download_f32(res_b.corrected, sz, stream_);

    int n_differ = 0;
    for (size_t i = 0; i < sz; ++i)
        if (h_ca[i] != h_cb[i]) ++n_differ;

    std::printf("[Determinism] n_differ=%d / %zu (expect 0)\n", n_differ, sz);
    EXPECT_EQ(n_differ, 0) << "soupx is deterministic; corrected should be bit-identical";
}

// =============================================================================
// Test 5: Soupx_MemoryGuard_RejectsTooLarge
// 100000 × 100000 synthetic dims would require 40 GB dense output → throws.
// =============================================================================
TEST_F(SoupxTest, Soupx_MemoryGuard_RejectsTooLarge) {
    if (!gpu_available()) GTEST_SKIP() << "no GPU";

    // Build a tiny real CSC but set rows/cols to huge values manually.
    // We can't actually allocate 100k×100k, so we forge the dimensions.
    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat.rows = 100000;
    pz.mat.cols = 100000;
    pz.mat.nnz  = 0;
    pz.mat.values      = singlet::gpu::core::DeviceMemory<float>(1);
    pz.mat.row_indices = singlet::gpu::core::DeviceMemory<int32_t>(1);
    pz.mat.col_ptr     = singlet::gpu::core::DeviceMemory<int32_t>(100001);
    cudaMemset(pz.mat.col_ptr.get(), 0, 100001 * sizeof(int32_t));

    singlet::gpu::qc::SoupxConfig cfg;
    cfg.lower = 100;

    bool threw = false;
    try {
        auto res = singlet::gpu::qc::soupx(pz, cfg, stream_);
    } catch (const std::runtime_error& e) {
        threw = true;
        std::printf("[MemoryGuard] caught: %s\n", e.what());
    }
    EXPECT_TRUE(threw) << "Expected runtime_error for 100k×100k dense output";
}
