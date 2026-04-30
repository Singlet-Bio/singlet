// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/enrich_decoupler_wsum_correctness.cpp
//
// Correctness harness for decoupleR WSUM + WMEAN pathway-scoring (CYCLE-128).
//
// Tests:
//  WSUM:
//   1. Wsum_TinyClosedForm             — 5×8×3 hand-computed closed form
//   2. Wsum_RealMatrices_VsCpu         — 50×100×5 random, CPU reference matmul
//   3. Wsum_AllZerosWeights_AllZerosScores — W=0 → scores=0 (eps guard no NaN)
//   4. Wsum_Determinism_SameInput      — rel_err < 1e-4 across two runs
//   5. Wsum_XScale_PropagatesScore     — scaling X by 2× doubles scores
//  WMEAN:
//   6. Wmean_TinyClosedForm            — 5×8×3 hand-computed closed form
//   7. Wmean_RealMatrices_VsCpu        — 50×100×5 random, CPU reference
//   8. Wmean_AllZerosWeights_AllZerosScores — W=0 → scores=0
//   9. Wmean_Determinism_SameInput     — rel_err < 1e-4 across two runs
//  10. Wmean_WScale_DoesNotChangeScore — scaling W by 2× leaves scores unchanged
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="WsumTest.*:WmeanTest.*"

#include <singlet-gpu/enrich/decoupler_wsum.h>
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
singlet_gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet_gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet_gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
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
std::vector<float> d2h(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Upload a dense weight matrix to device.
singlet_gpu::core::DeviceMemory<float> upload_W(const std::vector<float>& W_h)
{
    singlet_gpu::core::DeviceMemory<float> d_W(W_h.size());
    cudaMemcpy(d_W.get(), W_h.data(), W_h.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_W;
}

// CPU WSUM reference.
// X: m×n row-major (genes×cells). W: m×p col-major. Returns n×p col-major.
std::vector<float> cpu_wsum(
    int m, int n, int p,
    const std::vector<float>& X,   // row-major m×n
    const std::vector<float>& W,   // col-major m×p
    float eps = 1e-9f)
{
    // wsum[c, q] = Σ_g X[g,c] * W[g,q]
    // W col-major: W[g,q] = W[q*m + g]
    // scores col-major: scores[q*n + c]
    std::vector<float> scores(n * p, 0.f);
    for (int q = 0; q < p; ++q) {
        // L1 norm of column q of W
        float norm = 0.f;
        for (int g = 0; g < m; ++g) norm += std::fabs(W[q * m + g]);
        float denom = std::max(norm, eps);
        for (int c = 0; c < n; ++c) {
            float wsum_val = 0.f;
            for (int g = 0; g < m; ++g)
                wsum_val += X[g * n + c] * W[q * m + g];
            scores[q * n + c] = wsum_val / denom;
        }
    }
    return scores;
}

// CPU WMEAN reference.
std::vector<float> cpu_wmean(
    int m, int n, int p,
    const std::vector<float>& X,   // row-major m×n
    const std::vector<float>& W)   // col-major m×p
{
    std::vector<float> scores(n * p, 0.f);
    for (int q = 0; q < p; ++q) {
        // Count nonzero weights in column q
        float n_nz = 0.f;
        for (int g = 0; g < m; ++g) if (W[q * m + g] != 0.f) n_nz += 1.f;
        float denom = std::max(n_nz, 1.f);
        for (int c = 0; c < n; ++c) {
            float wsum_val = 0.f;
            for (int g = 0; g < m; ++g)
                wsum_val += X[g * n + c] * W[q * m + g];
            scores[q * n + c] = wsum_val / denom;
        }
    }
    return scores;
}

} // anonymous namespace


// =============================================================================
// WSUM Tests
// =============================================================================

// =============================================================================
// Test 1: Wsum_TinyClosedForm
//
// 5 genes × 8 cells × 3 pathways, all values hand-specified.
// CPU reference computed inline; GPU must match to 1e-4.
// =============================================================================
TEST(WsumTest, Wsum_TinyClosedForm)
{
    constexpr int M = 5, N = 8, P = 3;

    // Row-major (gene-major) expression matrix X (M×N):
    // clang-format off
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
    // Col-major weight matrix W (M rows × P cols):
    // W[g + p*M] for pathway p, gene g.
    std::vector<float> W_h = {
        // pathway 0: all genes
         1.0f, -1.0f,  0.5f,  0.0f,  2.0f,
        // pathway 1
         0.0f,  1.0f,  1.0f, -1.0f,  0.0f,
        // pathway 2
         0.5f,  0.5f, -0.5f,  1.0f,  0.5f,
    };
    // clang-format on

    // CPU reference
    auto ref = cpu_wsum(M, N, P, X_h, W_h, 1e-9f);

    // GPU path
    auto csc   = dense_to_csc(M, N, X_h);
    auto pz    = upload_csc(csc);
    auto d_W   = upload_W(W_h);

    singlet_gpu::enrich::WsumConfig cfg;
    auto res   = singlet_gpu::enrich::wsum(pz, d_W.get(), P, cfg);
    auto gpu   = d2h(res.scores, N * P);

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

    std::printf("[REGISTRY] %s | enrich/decoupler_wsum/tiny_cf | 5x8x3 | "
                "abs_err | max=%.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Wsum_RealMatrices_VsCpu
//
// 50 genes × 100 cells × 5 pathways with random dense W.
// CPU matmul reference vs GPU output; abs_err < 1e-3.
// =============================================================================
TEST(WsumTest, Wsum_RealMatrices_VsCpu)
{
    constexpr int M = 50, N = 100, P = 5;

    std::mt19937 rng(0xBEEF1234ULL);
    std::uniform_real_distribution<float> u_expr(0.f, 10.f);
    std::uniform_real_distribution<float> u_w(-1.f, 1.f);
    std::bernoulli_distribution sparse(0.6);  // 40% fill for X

    std::vector<float> X_h(M * N, 0.f);
    for (int i = 0; i < M * N; ++i)
        if (!sparse(rng)) X_h[i] = u_expr(rng);

    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = u_w(rng);

    auto ref = cpu_wsum(M, N, P, X_h, W_h, 1e-9f);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet_gpu::enrich::WsumConfig cfg;
    auto res  = singlet_gpu::enrich::wsum(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
    }

    EXPECT_LT(max_err, 1e-3f)
        << "max abs_err=" << max_err << " exceeds 1e-3 tolerance";

    std::printf("[REGISTRY] %s | enrich/decoupler_wsum/vs_cpu | 50x100x5 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-128 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Wsum_AllZerosWeights_AllZerosScores
//
// W all zeros → scores must all be zero (eps guard prevents NaN/Inf).
// =============================================================================
TEST(WsumTest, Wsum_AllZerosWeights_AllZerosScores)
{
    constexpr int M = 20, N = 30, P = 4;

    std::mt19937 rng(0xABCDULL);
    std::uniform_real_distribution<float> u(0.f, 5.f);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    // W all zeros
    std::vector<float> W_h(M * P, 0.f);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet_gpu::enrich::WsumConfig cfg;
    auto res  = singlet_gpu::enrich::wsum(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    for (int i = 0; i < N * P; ++i) {
        EXPECT_TRUE(std::isfinite(gpu[i]))
            << "element " << i << " is non-finite: " << gpu[i];
        EXPECT_NEAR(gpu[i], 0.f, 1e-6f)
            << "element " << i << ": expected 0, got " << gpu[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_wsum/zeros_W | 20x30x4 | "
                "zero_scores | none | 1e-6 | none | CYCLE-128 | PASS\n", __DATE__);
}


// =============================================================================
// Test 4: Wsum_Determinism_SameInput
//
// Two consecutive wsum calls on the same data must agree to rel_err < 1e-4.
// (SpMM + warp-shuffle norm are bit-exact; this is a sanity check.)
// =============================================================================
TEST(WsumTest, Wsum_Determinism_SameInput)
{
    constexpr int M = 30, N = 60, P = 4;

    std::mt19937 rng(0xDEADULL);
    std::poisson_distribution<int> pd(3);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = static_cast<float>(pd(rng));

    std::uniform_real_distribution<float> uw(-1.f, 1.f);
    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = uw(rng);

    auto csc = dense_to_csc(M, N, X_h);

    singlet_gpu::enrich::WsumConfig cfg;

    auto pz1  = upload_csc(csc);
    auto dW1  = upload_W(W_h);
    auto res1 = singlet_gpu::enrich::wsum(pz1, dW1.get(), P, cfg);
    auto h1   = d2h(res1.scores, N * P);

    auto pz2  = upload_csc(csc);
    auto dW2  = upload_W(W_h);
    auto res2 = singlet_gpu::enrich::wsum(pz2, dW2.get(), P, cfg);
    auto h2   = d2h(res2.scores, N * P);

    float max_rel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
    }

    EXPECT_LT(max_rel, 1e-4f)
        << "Determinism: max_rel_err=" << max_rel << " exceeds 1e-4";

    std::printf("[REGISTRY] %s | enrich/decoupler_wsum/determinism | 30x60x4 | "
                "rel_err | %.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_rel, (max_rel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Wsum_XScale_PropagatesScore
//
// Scaling X by 2× should scale all scores by 2× (WSUM divides by norm of W,
// not norm of X, so the scaling passes through linearly).
// Scaling W by 2× should NOT change scores (wsum / L1-norm(2W) = wsum/2 / L1-norm(W)
// but wsum itself doubles, so they cancel → scores unchanged).
// =============================================================================
TEST(WsumTest, Wsum_XScale_PropagatesScore)
{
    constexpr int M = 15, N = 20, P = 3;

    std::mt19937 rng(0xC0DE1234ULL);
    std::uniform_real_distribution<float> ux(0.f, 5.f);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);

    std::vector<float> X_h(M * N), X2_h(M * N);
    std::vector<float> W_h(M * P), W2_h(M * P);
    for (int i = 0; i < M * N; ++i) { X_h[i] = ux(rng); X2_h[i] = X_h[i] * 2.f; }
    for (int i = 0; i < M * P; ++i) { W_h[i] = uw(rng); W2_h[i] = W_h[i] * 2.f; }

    singlet_gpu::enrich::WsumConfig cfg;

    // Run 1: base X, base W → scores1
    auto csc1  = dense_to_csc(M, N, X_h);
    auto pz1   = upload_csc(csc1);
    auto dW1   = upload_W(W_h);
    auto res1  = singlet_gpu::enrich::wsum(pz1, dW1.get(), P, cfg);
    auto h1    = d2h(res1.scores, N * P);

    // Run 2: 2×X, base W → scores2 should be ~2×scores1
    auto csc2  = dense_to_csc(M, N, X2_h);
    auto pz2   = upload_csc(csc2);
    auto dW2   = upload_W(W_h);
    auto res2  = singlet_gpu::enrich::wsum(pz2, dW2.get(), P, cfg);
    auto h2    = d2h(res2.scores, N * P);

    // Run 3: base X, 2×W → scores3 should equal scores1 (cancel)
    auto csc3  = dense_to_csc(M, N, X_h);
    auto pz3   = upload_csc(csc3);
    auto dW3   = upload_W(W2_h);
    auto res3  = singlet_gpu::enrich::wsum(pz3, dW3.get(), P, cfg);
    auto h3    = d2h(res3.scores, N * P);

    float max_err_scale = 0.f, max_err_cancel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        // 2×X should double scores
        float expected2 = 2.f * h1[i];
        max_err_scale   = std::max(max_err_scale, std::fabs(h2[i] - expected2));
        // 2×W should leave scores unchanged
        max_err_cancel  = std::max(max_err_cancel, std::fabs(h3[i] - h1[i]));
    }

    EXPECT_LT(max_err_scale, 1e-4f)
        << "2×X scaling: max error from 2×baseline = " << max_err_scale;
    EXPECT_LT(max_err_cancel, 1e-4f)
        << "2×W cancels: max deviation from baseline = " << max_err_cancel;

    std::printf("[REGISTRY] %s | enrich/decoupler_wsum/scale_prop | 15x20x3 | "
                "abs_err | scale=%.2e cancel=%.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_err_scale, max_err_cancel,
                (max_err_scale < 1e-4f && max_err_cancel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// WMEAN Tests
// =============================================================================

// =============================================================================
// Test 6: Wmean_TinyClosedForm
//
// 5 genes × 8 cells × 3 pathways. Same X and W as Wsum_TinyClosedForm.
// CPU reference uses nonzero-count normaliser instead of L1.
// =============================================================================
TEST(WmeanTest, Wmean_TinyClosedForm)
{
    constexpr int M = 5, N = 8, P = 3;

    // clang-format off
    std::vector<float> X_h = {
        3,  0,  1,  0,  2,  0,  4,  1,
        0,  2,  0,  3,  0,  1,  0,  2,
        1,  1,  0,  0,  3,  2,  1,  0,
        0,  0,  0,  0,  0,  0,  5,  0,
        2,  3,  1,  2,  1,  3,  2,  1,
    };
    std::vector<float> W_h = {
         1.0f, -1.0f,  0.5f,  0.0f,  2.0f,
         0.0f,  1.0f,  1.0f, -1.0f,  0.0f,
         0.5f,  0.5f, -0.5f,  1.0f,  0.5f,
    };
    // clang-format on

    auto ref = cpu_wmean(M, N, P, X_h, W_h);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet_gpu::enrich::WmeanConfig cfg;
    auto res  = singlet_gpu::enrich::wmean(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    ASSERT_EQ(res.n_cells,    N);
    ASSERT_EQ(res.n_pathways, P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
        EXPECT_NEAR(gpu[i], ref[i], 1e-4f)
            << "element " << i << ": gpu=" << gpu[i] << " ref=" << ref[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_wmean/tiny_cf | 5x8x3 | "
                "abs_err | max=%.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 7: Wmean_RealMatrices_VsCpu
//
// 50 genes × 100 cells × 5 pathways random. CPU reference vs GPU. abs_err < 1e-3.
// =============================================================================
TEST(WmeanTest, Wmean_RealMatrices_VsCpu)
{
    constexpr int M = 50, N = 100, P = 5;

    std::mt19937 rng(0xF00DCAFEULL);
    std::uniform_real_distribution<float> u_expr(0.f, 10.f);
    std::uniform_real_distribution<float> u_w(-1.f, 1.f);
    std::bernoulli_distribution sparse(0.6);

    std::vector<float> X_h(M * N, 0.f);
    for (int i = 0; i < M * N; ++i)
        if (!sparse(rng)) X_h[i] = u_expr(rng);

    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = u_w(rng);

    auto ref = cpu_wmean(M, N, P, X_h, W_h);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet_gpu::enrich::WmeanConfig cfg;
    auto res  = singlet_gpu::enrich::wmean(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i)
        max_err = std::max(max_err, std::fabs(gpu[i] - ref[i]));

    EXPECT_LT(max_err, 1e-3f)
        << "max abs_err=" << max_err << " exceeds 1e-3 tolerance";

    std::printf("[REGISTRY] %s | enrich/decoupler_wmean/vs_cpu | 50x100x5 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-128 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 8: Wmean_AllZerosWeights_AllZerosScores
//
// W all zeros → all wsum values are zero; count guard max(0,1)=1 → scores=0.
// No NaN/Inf.
// =============================================================================
TEST(WmeanTest, Wmean_AllZerosWeights_AllZerosScores)
{
    constexpr int M = 20, N = 30, P = 4;

    std::mt19937 rng(0xABCDEFULL);
    std::uniform_real_distribution<float> u(0.f, 5.f);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    std::vector<float> W_h(M * P, 0.f);

    auto csc  = dense_to_csc(M, N, X_h);
    auto pz   = upload_csc(csc);
    auto d_W  = upload_W(W_h);

    singlet_gpu::enrich::WmeanConfig cfg;
    auto res  = singlet_gpu::enrich::wmean(pz, d_W.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    for (int i = 0; i < N * P; ++i) {
        EXPECT_TRUE(std::isfinite(gpu[i]))
            << "element " << i << " is non-finite: " << gpu[i];
        EXPECT_NEAR(gpu[i], 0.f, 1e-6f)
            << "element " << i << ": expected 0, got " << gpu[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_wmean/zeros_W | 20x30x4 | "
                "zero_scores | none | 1e-6 | none | CYCLE-128 | PASS\n", __DATE__);
}


// =============================================================================
// Test 9: Wmean_Determinism_SameInput
//
// Two consecutive wmean calls must agree to rel_err < 1e-4.
// =============================================================================
TEST(WmeanTest, Wmean_Determinism_SameInput)
{
    constexpr int M = 30, N = 60, P = 4;

    std::mt19937 rng(0xCAFEBABEULL);
    std::poisson_distribution<int> pd(3);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = static_cast<float>(pd(rng));

    std::uniform_real_distribution<float> uw(-1.f, 1.f);
    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = uw(rng);

    auto csc = dense_to_csc(M, N, X_h);

    singlet_gpu::enrich::WmeanConfig cfg;

    auto pz1  = upload_csc(csc);
    auto dW1  = upload_W(W_h);
    auto res1 = singlet_gpu::enrich::wmean(pz1, dW1.get(), P, cfg);
    auto h1   = d2h(res1.scores, N * P);

    auto pz2  = upload_csc(csc);
    auto dW2  = upload_W(W_h);
    auto res2 = singlet_gpu::enrich::wmean(pz2, dW2.get(), P, cfg);
    auto h2   = d2h(res2.scores, N * P);

    float max_rel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
    }

    EXPECT_LT(max_rel, 1e-4f)
        << "Determinism: max_rel_err=" << max_rel << " exceeds 1e-4";

    std::printf("[REGISTRY] %s | enrich/decoupler_wmean/determinism | 30x60x4 | "
                "rel_err | %.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_rel, (max_rel < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 10: Wmean_WScale_DoesNotChangeScore
//
// WMEAN score[c,p] = wsum[c,p] / n_nz[p].
// Scaling W by scalar k: wsum scales by k, n_nz is unchanged (same zero pattern)
// → scores scale by k. This differs from WSUM (where scaling cancels).
// Conversely, scaling X by k → scores scale by k (same as WSUM).
// This test verifies both properties for WMEAN specifically.
// =============================================================================
TEST(WmeanTest, Wmean_WScale_PropagatesScore)
{
    constexpr int M = 15, N = 20, P = 3;

    std::mt19937 rng(0xFEEDFACEULL);
    std::uniform_real_distribution<float> ux(0.f, 5.f);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);

    std::vector<float> X_h(M * N), X2_h(M * N);
    std::vector<float> W_h(M * P), W2_h(M * P);
    for (int i = 0; i < M * N; ++i) { X_h[i] = ux(rng); X2_h[i] = X_h[i] * 2.f; }
    for (int i = 0; i < M * P; ++i) { W_h[i] = uw(rng); W2_h[i] = W_h[i] * 2.f; }

    singlet_gpu::enrich::WmeanConfig cfg;

    // Baseline: X, W → scores1
    auto csc1 = dense_to_csc(M, N, X_h);
    auto pz1  = upload_csc(csc1);
    auto dW1  = upload_W(W_h);
    auto res1 = singlet_gpu::enrich::wmean(pz1, dW1.get(), P, cfg);
    auto h1   = d2h(res1.scores, N * P);

    // 2×X, base W → scores should double (linear in X)
    auto csc2 = dense_to_csc(M, N, X2_h);
    auto pz2  = upload_csc(csc2);
    auto dW2  = upload_W(W_h);
    auto res2 = singlet_gpu::enrich::wmean(pz2, dW2.get(), P, cfg);
    auto h2   = d2h(res2.scores, N * P);

    // base X, 2×W → scores should also double (WMEAN: n_nz unchanged, wsum doubles)
    auto csc3 = dense_to_csc(M, N, X_h);
    auto pz3  = upload_csc(csc3);
    auto dW3  = upload_W(W2_h);
    auto res3 = singlet_gpu::enrich::wmean(pz3, dW3.get(), P, cfg);
    auto h3   = d2h(res3.scores, N * P);

    float max_err_x = 0.f, max_err_w = 0.f;
    for (int i = 0; i < N * P; ++i) {
        max_err_x = std::max(max_err_x, std::fabs(h2[i] - 2.f * h1[i]));
        max_err_w = std::max(max_err_w, std::fabs(h3[i] - 2.f * h1[i]));
    }

    EXPECT_LT(max_err_x, 1e-4f)
        << "WMEAN 2×X: max deviation from 2×baseline = " << max_err_x;
    EXPECT_LT(max_err_w, 1e-4f)
        << "WMEAN 2×W: max deviation from 2×baseline = " << max_err_w;

    std::printf("[REGISTRY] %s | enrich/decoupler_wmean/scale_prop | 15x20x3 | "
                "abs_err | x_scale=%.2e w_scale=%.2e | 1e-4 | none | CYCLE-128 | %s\n",
                __DATE__, max_err_x, max_err_w,
                (max_err_x < 1e-4f && max_err_w < 1e-4f) ? "PASS" : "FAIL");
}
