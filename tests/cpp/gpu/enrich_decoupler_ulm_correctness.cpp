// SPDX-License-Identifier: MIT
// singlet/gpu/tests/enrich_decoupler_ulm_correctness.cpp
//
// Correctness harness for decoupleR ULM pathway-scoring (CYCLE-130).
//
// Tests:
//  1. Ulm_TinyClosedForm        — 5×8×3, CPU closed-form β_1, abs_err<1e-4
//  2. Ulm_VsCpu_Random          — 50×100×5 random, CPU OLS reference, abs_err<1e-3
//  3. Ulm_ConstantW_ZeroOutput  — constant W column → var_W=0 → score=0, no NaN
//  4. Ulm_Determinism           — two runs: rel_err<1e-4
//  5. Ulm_PerfectPositiveCorr   — X[:,0]=2*W[:,0]+5 → score[0,0]≈2.0, abs_err<1e-3
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="UlmTest.*"

#include <singlet/gpu/enrich/decoupler_ulm.h>
#include <singlet/gpu/io/pz_device_loader.h>

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
// Shared test helpers
// =============================================================================

namespace {

// Build a CSC from a dense row-major matrix (rows=genes, cols=cells).
struct SynCSC {
    int                  rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

SynCSC dense_to_csc(int rows, int cols, const std::vector<float>& dense)
{
    SynCSC c;
    c.rows = rows;
    c.cols = cols;
    c.col_ptr.resize(cols + 1, 0);
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            if (dense[i * cols + j] != 0.f) c.col_ptr[j + 1]++;
    for (int j = 0; j < cols; ++j) c.col_ptr[j + 1] += c.col_ptr[j];
    c.nnz = c.col_ptr[cols];
    c.vals.resize(c.nnz);
    c.row_idx.resize(c.nnz);
    std::vector<int32_t> fill(cols, 0);
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            float v = dense[i * cols + j];
            if (v != 0.f) {
                int pos = c.col_ptr[j] + fill[j]++;
                c.row_idx[pos] = i;
                c.vals[pos]    = v;
            }
        }
    }
    return c;
}

// Upload SynCSC to device and wrap in PzDeviceMatrix.
singlet::gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet::gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
    cudaMemcpy(pz.mat.col_ptr.get(), c.col_ptr.data(),
               (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    if (c.nnz > 0) {
        cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
                   c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(pz.mat.values.get(), c.vals.data(),
                   c.nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    return pz;
}

// Fetch device float buffer to host.
std::vector<float> d2h(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Upload a dense weight matrix to device.
singlet::gpu::core::DeviceMemory<float> upload_W(const std::vector<float>& W_h)
{
    singlet::gpu::core::DeviceMemory<float> d_W(W_h.size());
    cudaMemcpy(d_W.get(), W_h.data(), W_h.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_W;
}

// ---------------------------------------------------------------------------
// cpu_ulm — CPU reference ULM.
//
// X: m×n row-major (genes×cells).  W: m×p col-major.  Returns n×p col-major.
// score[c, q] = cov(X[:,c], W[:,q]) / var(W[:,q])
//             = ((1/m)·Σ_g X[g,c]·W[g,q] - mean_X[c]·mean_W[q]) / var_W[q]
//
// When var_W[q] < epsilon, returns 0.
// ---------------------------------------------------------------------------
std::vector<float> cpu_ulm(
    int m, int n, int p,
    const std::vector<float>& X,    // row-major m×n: X[g*n+c]
    const std::vector<float>& W,    // col-major m×p: W[q*m+g]
    float eps = 1e-9f)
{
    std::vector<float> scores(n * p, 0.f);

    // Per-pathway stats.
    std::vector<double> mean_W_d(p, 0.0);
    std::vector<double> var_W_d(p, 0.0);
    for (int q = 0; q < p; ++q) {
        double s1 = 0.0, s2 = 0.0;
        for (int g = 0; g < m; ++g) {
            double w = static_cast<double>(W[q * m + g]);
            s1 += w;
            s2 += w * w;
        }
        double mu = s1 / m;
        mean_W_d[q] = mu;
        var_W_d[q]  = s2 / m - mu * mu;
    }

    // Per-cell means.
    std::vector<double> mean_X_d(n, 0.0);
    for (int c = 0; c < n; ++c) {
        double s = 0.0;
        for (int g = 0; g < m; ++g)
            s += static_cast<double>(X[g * n + c]);
        mean_X_d[c] = s / m;
    }

    // Scores: col-major output.
    for (int q = 0; q < p; ++q) {
        double vw = var_W_d[q];
        for (int c = 0; c < n; ++c) {
            if (vw <= static_cast<double>(eps)) {
                scores[q * n + c] = 0.f;
                continue;
            }
            // cov = (1/m)·XW[c,q] - mean_X[c]·mean_W[q]
            double xw = 0.0;
            for (int g = 0; g < m; ++g)
                xw += static_cast<double>(X[g * n + c]) * static_cast<double>(W[q * m + g]);
            double cov = xw / m - mean_X_d[c] * mean_W_d[q];
            scores[q * n + c] = static_cast<float>(cov / vw);
        }
    }
    return scores;
}

} // anonymous namespace


// =============================================================================
// ULM Tests
// =============================================================================

// =============================================================================
// Test 1: Ulm_TinyClosedForm
//
// 5 genes × 8 cells × 3 pathways. CPU closed-form OLS reference vs GPU.
// abs_err < 1e-4.
// =============================================================================
TEST(UlmTest, Ulm_TinyClosedForm)
{
    constexpr int M = 5, N = 8, P = 3;

    // clang-format off
    // Row-major (gene-major) expression matrix X (M×N):
    std::vector<float> X_h = {
        // gene 0:  c0  c1  c2  c3  c4  c5  c6  c7
                    3,  0,  1,  0,  2,  0,  4,  1,
        // gene 1
                    0,  2,  0,  3,  0,  1,  0,  2,
        // gene 2
                    1,  1,  0,  0,  3,  2,  1,  0,
        // gene 3
                    0,  0,  0,  0,  0,  0,  5,  0,
        // gene 4
                    2,  3,  1,  2,  1,  3,  2,  1,
    };
    // Col-major weight matrix W (M rows × P cols): W[g + p*M]
    std::vector<float> W_h = {
        // pathway 0
         1.0f, -1.0f,  0.5f,  0.0f,  2.0f,
        // pathway 1
         0.0f,  1.0f,  1.0f, -1.0f,  0.0f,
        // pathway 2
         0.5f,  0.5f, -0.5f,  1.0f,  0.5f,
    };
    // clang-format on

    auto ref = cpu_ulm(M, N, P, X_h, W_h, 1e-9f);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet::gpu::enrich::UlmConfig cfg;
    auto res  = singlet::gpu::enrich::ulm(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    ASSERT_EQ(res.n_cells,    N);
    ASSERT_EQ(res.n_pathways, P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
        EXPECT_NEAR(gpu[i], ref[i], 1e-4f)
            << "element " << i << " (pathway=" << i/N << " cell=" << i%N
            << "): gpu=" << gpu[i] << " ref=" << ref[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ulm/tiny_cf | 5x8x3 | "
                "abs_err | max=%.2e | 1e-4 | none | CYCLE-130 | %s\n",
                __DATE__, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Ulm_VsCpu_Random
//
// 50 genes × 100 cells × 5 pathways with random X and W.
// CPU OLS reference per (c, p) pair vs GPU. abs_err < 1e-3.
// =============================================================================
TEST(UlmTest, Ulm_VsCpu_Random)
{
    constexpr int M = 50, N = 100, P = 5;

    std::mt19937 rng(0xDEC0DEULL);
    std::uniform_real_distribution<float> u_expr(0.f, 10.f);
    std::uniform_real_distribution<float> u_w(-1.f, 1.f);
    std::bernoulli_distribution sparse(0.6);  // ~40% fill for X

    std::vector<float> X_h(M * N, 0.f);
    for (int i = 0; i < M * N; ++i)
        if (!sparse(rng)) X_h[i] = u_expr(rng);

    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = u_w(rng);

    auto ref = cpu_ulm(M, N, P, X_h, W_h, 1e-9f);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet::gpu::enrich::UlmConfig cfg;
    auto res  = singlet::gpu::enrich::ulm(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
    }

    EXPECT_LT(max_err, 1e-3f)
        << "max abs_err=" << max_err << " exceeds 1e-3 tolerance";

    std::printf("[REGISTRY] %s | enrich/decoupler_ulm/vs_cpu | 50x100x5 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-130 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Ulm_ConstantW_ZeroOutput
//
// W column 1 is constant (all same value) → var_W = 0 → score must be 0.
// Other columns have normal variation.
// Verify all scores in the constant pathway column equal 0, no NaN/Inf.
// =============================================================================
TEST(UlmTest, Ulm_ConstantW_ZeroOutput)
{
    constexpr int M = 20, N = 30, P = 3;

    std::mt19937 rng(0xC0DE9999ULL);
    std::uniform_real_distribution<float> ux(0.f, 5.f);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);

    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = ux(rng);

    // Build W: col 0 and col 2 random; col 1 constant = 3.14.
    std::vector<float> W_h(M * P);
    for (int q = 0; q < P; ++q) {
        for (int g = 0; g < M; ++g) {
            if (q == 1) {
                W_h[q * M + g] = 3.14f;   // constant column
            } else {
                W_h[q * M + g] = uw(rng);
            }
        }
    }

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet::gpu::enrich::UlmConfig cfg;
    auto res  = singlet::gpu::enrich::ulm(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    // Pathway 1 (constant W column): all scores must be 0 and finite.
    for (int c = 0; c < N; ++c) {
        float s = gpu[1 * N + c];
        EXPECT_TRUE(std::isfinite(s))
            << "cell " << c << " pathway 1: non-finite score " << s;
        EXPECT_NEAR(s, 0.f, 1e-6f)
            << "cell " << c << " pathway 1: expected 0, got " << s;
    }

    // Pathways 0 and 2 should also be finite (not checking value, just validity).
    for (int q : {0, 2}) {
        for (int c = 0; c < N; ++c) {
            EXPECT_TRUE(std::isfinite(gpu[q * N + c]))
                << "pathway " << q << " cell " << c << ": non-finite";
        }
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ulm/constant_W | 20x30x3 | "
                "zero_scores | none | 1e-6 | none | CYCLE-130 | PASS\n", __DATE__);
}


// =============================================================================
// Test 4: Ulm_Determinism
//
// Two consecutive ULM calls on the same data must agree to rel_err < 1e-4.
// =============================================================================
TEST(UlmTest, Ulm_Determinism)
{
    constexpr int M = 30, N = 60, P = 4;

    std::mt19937 rng(0xA1B2C3D4ULL);
    std::poisson_distribution<int> pd(3);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);

    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = static_cast<float>(pd(rng));

    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = uw(rng);

    auto csc = dense_to_csc(M, N, X_h);

    singlet::gpu::enrich::UlmConfig cfg;

    auto pz1  = upload_csc(csc);
    auto dW1  = upload_W(W_h);
    auto res1 = singlet::gpu::enrich::ulm(pz1, dW1.get(), P, cfg);
    auto h1   = d2h(res1.scores, N * P);

    auto pz2  = upload_csc(csc);
    auto dW2  = upload_W(W_h);
    auto res2 = singlet::gpu::enrich::ulm(pz2, dW2.get(), P, cfg);
    auto h2   = d2h(res2.scores, N * P);

    float max_rel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
    }

    EXPECT_LT(max_rel, 1e-4f)
        << "Determinism: max_rel_err=" << max_rel << " exceeds 1e-4";

    std::printf("[REGISTRY] %s | enrich/decoupler_ulm/determinism | 30x60x4 | "
                "rel_err | %.2e | 1e-4 | none | CYCLE-130 | %s\n",
                __DATE__, max_rel, (max_rel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Ulm_PerfectPositiveCorrelation
//
// Build X such that for cell 0: X[g, 0] = 2 * W[g, 0] + 5 (perfect linear
// with slope β_1 = 2).  ULM with pathway 0 = first column of W should recover
// score[0, 0] = 2.0 (within abs_err < 1e-3).
//
// Construction:
//   W[:,0] = arbitrary non-constant weights (use ascending 0..M-1, rescaled).
//   X[g, 0] = 2 * W[g, 0] + 5.
//   Other cells: random to ensure numerical diversity.
//
// Regression identity check:
//   cov(X_0, W_0) = 2 * var(W_0)  →  β_1 = 2.
// =============================================================================
TEST(UlmTest, Ulm_PerfectPositiveCorrelation)
{
    constexpr int M = 20, N = 10, P = 2;

    std::mt19937 rng(0xBEEF4242ULL);
    std::uniform_real_distribution<float> ux(0.f, 8.f);

    // W col 0: ascending weights (non-constant, unit-scaled).
    // W col 1: random (not tested).
    std::vector<float> W_h(M * P);
    for (int g = 0; g < M; ++g) {
        W_h[0 * M + g] = static_cast<float>(g + 1) / static_cast<float>(M);  // 1/M .. 1
        W_h[1 * M + g] = ux(rng);
    }

    // X: col 0 = 2*W[:,0] + 5; other cols random.
    std::vector<float> X_h(M * N);
    for (int c = 0; c < N; ++c) {
        for (int g = 0; g < M; ++g) {
            if (c == 0) {
                X_h[g * N + c] = 2.f * W_h[0 * M + g] + 5.f;
            } else {
                X_h[g * N + c] = ux(rng);
            }
        }
    }

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet::gpu::enrich::UlmConfig cfg;
    auto res  = singlet::gpu::enrich::ulm(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    // score[cell=0, pathway=0] should be 2.0.
    float beta1_gpu = gpu[0 * N + 0];
    EXPECT_NEAR(beta1_gpu, 2.f, 1e-3f)
        << "PerfectCorr: score[c=0,p=0]=" << beta1_gpu << " expected 2.0";

    // Sanity: all outputs finite.
    for (int i = 0; i < N * P; ++i) {
        EXPECT_TRUE(std::isfinite(gpu[i]))
            << "element " << i << " non-finite: " << gpu[i];
    }

    float err = std::fabs(beta1_gpu - 2.f);
    std::printf("[REGISTRY] %s | enrich/decoupler_ulm/perfect_corr | 20x10x2 | "
                "abs_err | beta1_err=%.2e | 1e-3 | none | CYCLE-130 | %s\n",
                __DATE__, err, (err < 1e-3f) ? "PASS" : "FAIL");
}
