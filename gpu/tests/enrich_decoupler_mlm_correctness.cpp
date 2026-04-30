// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/enrich_decoupler_mlm_correctness.cpp
//
// Correctness harness for decoupleR MLM pathway-scoring (CYCLE-136).
//
// Tests:
//  1. Mlm_TinyClosedForm             — 5×4×2, CPU (W^T W)^{-1} W^T X, abs_err<1e-3
//  2. Mlm_OrthogonalPathways_MatchRef — orthogonal W^T W=diag, compare to CPU β, abs_err<1e-3
//  3. Mlm_CorrelatedPathways_DiffersFromULM — correlated W: MLM != ULM-based scores
//  4. Mlm_Determinism_BitIdentical   — rel_err = 0 across two runs (bit-exact)
//  5. Mlm_RankDeficient_RidgeStabilizes — near-rank-deficient W; ridge prevents failure
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="MlmTest.*"

#include <singlet-gpu/enrich/decoupler_mlm.h>
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

// ---------------------------------------------------------------------------
// SynCSC + helpers (mirror of enrich_decoupler_ulm_correctness.cpp)
// ---------------------------------------------------------------------------
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

std::vector<float> d2h(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

singlet_gpu::core::DeviceMemory<float> upload_W(const std::vector<float>& W_h)
{
    singlet_gpu::core::DeviceMemory<float> d_W(W_h.size());
    cudaMemcpy(d_W.get(), W_h.data(), W_h.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    return d_W;
}

// ---------------------------------------------------------------------------
// cpu_mlm — CPU reference MLM.
//
// Inputs:
//   X: m×n col-major (X[g + c*m]). W: m×p col-major (W[g + q*m]).
//   Returns scores: n×p col-major (scores[c + q*n]).
//
// Algorithm: for each cell c, β_c = (W^T W + ridge*I)^{-1} W^T X[:,c].
// For small p (≤ 8), we compute the 2×2 or p×p inverse directly.
//
// General approach: Cholesky on (W^T W + ridge*I) via simple CPU code,
// then back-substitution for each RHS column.  For tests with p ≤ 4 we
// use a direct formula.  Here we use Gaussian elimination (no pivoting,
// safe for positive definite matrices).
// ---------------------------------------------------------------------------

// Solve L y = b (forward substitution, L lower triangular, p×p col-major).
// L[i + j*p] for j <= i.
static void fwd_sub(const std::vector<double>& L, std::vector<double>& b, int p) {
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < i; ++j)
            b[i] -= L[i + j * p] * b[j];
        b[i] /= L[i + i * p];
    }
}

// Solve L^T y = b (backward substitution, L^T upper triangular).
static void bwd_sub(const std::vector<double>& L, std::vector<double>& b, int p) {
    for (int i = p - 1; i >= 0; --i) {
        for (int j = i + 1; j < p; ++j)
            b[i] -= L[j + i * p] * b[j];  // L^T[i,j] = L[j,i] = L[j + i*p]
        b[i] /= L[i + i * p];
    }
}

// Cholesky factorization (in-place, lower triangle, col-major, double precision).
static bool cholesky_lower(std::vector<double>& A, int p) {
    for (int j = 0; j < p; ++j) {
        double sum = A[j + j * p];
        for (int k = 0; k < j; ++k)
            sum -= A[j + k * p] * A[j + k * p];
        if (sum <= 0.0) return false;  // not positive definite
        A[j + j * p] = std::sqrt(sum);
        for (int i = j + 1; i < p; ++i) {
            double s2 = A[i + j * p];
            for (int k = 0; k < j; ++k)
                s2 -= A[i + k * p] * A[j + k * p];
            A[i + j * p] = s2 / A[j + j * p];
        }
    }
    return true;
}

std::vector<float> cpu_mlm(
    int m, int n, int p,
    const std::vector<float>& X_colmaj,   // m×n col-major: X[g + c*m]
    const std::vector<float>& W_colmaj,   // m×p col-major: W[g + q*m]
    float ridge = 1e-6f)
{
    std::vector<float> scores(n * p, 0.f);

    // Build W^T W (p×p, col-major) in double.
    std::vector<double> A(p * p, 0.0);
    for (int q1 = 0; q1 < p; ++q1) {
        for (int q2 = 0; q2 < p; ++q2) {
            double dot = 0.0;
            for (int g = 0; g < m; ++g)
                dot += static_cast<double>(W_colmaj[g + q1 * m]) *
                       static_cast<double>(W_colmaj[g + q2 * m]);
            A[q1 + q2 * p] = dot;
        }
    }
    // Ridge.
    for (int q = 0; q < p; ++q)
        A[q + q * p] += static_cast<double>(ridge);

    // Cholesky factor A.
    std::vector<double> L = A;
    bool ok = cholesky_lower(L, p);
    if (!ok) {
        // Return NaN to signal failure (test should not reach here).
        std::fill(scores.begin(), scores.end(), std::numeric_limits<float>::quiet_NaN());
        return scores;
    }

    // For each cell c, solve (W^T W + ridge*I) β_c = W^T X[:,c].
    for (int c = 0; c < n; ++c) {
        // Compute rhs = W^T X[:,c]  (p × 1, in double).
        std::vector<double> rhs(p, 0.0);
        for (int q = 0; q < p; ++q) {
            double s = 0.0;
            for (int g = 0; g < m; ++g)
                s += static_cast<double>(W_colmaj[g + q * m]) *
                     static_cast<double>(X_colmaj[g + c * m]);
            rhs[q] = s;
        }
        // Solve L y = rhs, then L^T β = y.
        fwd_sub(L, rhs, p);
        bwd_sub(L, rhs, p);
        // Store β into scores col-major (n×p): scores[c + q*n].
        for (int q = 0; q < p; ++q)
            scores[c + q * n] = static_cast<float>(rhs[q]);
    }
    return scores;
}

// Convert row-major X (m×n: X[g*n + c]) to col-major (X[g + c*m]).
std::vector<float> rowmaj_to_colmaj(const std::vector<float>& X_row, int m, int n) {
    std::vector<float> X_col(m * n);
    for (int g = 0; g < m; ++g)
        for (int c = 0; c < n; ++c)
            X_col[g + c * m] = X_row[g * n + c];
    return X_col;
}

// dense_to_csc expects row-major X (rows=genes, cols=cells): X[g*n + c].
// MLM SpMM operates on the same CSC that WSUM/ULM use.

} // anonymous namespace


// =============================================================================
// MLM Tests
// =============================================================================

// =============================================================================
// Test 1: Mlm_TinyClosedForm
//
// 5 genes × 4 cells × 2 pathways. Hand-constructed W, X.
// CPU closed-form reference β = (W^T W)^{-1} W^T X vs GPU.
// abs_err < 1e-3.
// =============================================================================
TEST(MlmTest, Mlm_TinyClosedForm)
{
    constexpr int M = 5, N = 4, P = 2;

    // Row-major X (genes × cells): X[g*N + c]
    // clang-format off
    std::vector<float> X_row = {
        // gene 0:  c0  c1  c2  c3
                    3,  0,  1,  2,
        // gene 1
                    0,  2,  0,  1,
        // gene 2
                    1,  1,  3,  0,
        // gene 3
                    0,  0,  0,  1,
        // gene 4
                    2,  3,  1,  2,
    };
    // Col-major W (m × p): W[g + q*M]
    std::vector<float> W_h = {
        // pathway 0: genes 0..4
         1.0f, -1.0f,  0.5f,  0.0f,  2.0f,
        // pathway 1: genes 0..4
         0.5f,  1.0f, -0.5f,  1.0f,  0.0f,
    };
    // clang-format on

    // CPU reference (col-major X).
    auto X_col = rowmaj_to_colmaj(X_row, M, N);
    auto ref   = cpu_mlm(M, N, P, X_col, W_h, 1e-6f);

    // GPU: dense_to_csc expects row-major X.
    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    singlet_gpu::enrich::MlmConfig cfg;
    cfg.ridge = 1e-6f;
    auto res  = singlet_gpu::enrich::mlm(pz, dW.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    ASSERT_EQ(res.n_cells,    N);
    ASSERT_EQ(res.n_pathways, P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
        EXPECT_NEAR(gpu[i], ref[i], 1e-3f)
            << "element " << i << " (pathway=" << i/N << " cell=" << i%N
            << "): gpu=" << gpu[i] << " ref=" << ref[i];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_mlm/tiny_cf | 5x4x2 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-136 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Mlm_OrthogonalPathways_MatchRef
//
// When W has orthogonal columns (W^T W = diagonal), MLM reduces to
// β_q = (W[:,q]^T X[:,c]) / ||W[:,q]||²  for each pathway q independently.
// Verify MLM scores match CPU per-pathway reference within abs_err < 1e-3.
// =============================================================================
TEST(MlmTest, Mlm_OrthogonalPathways_MatchRef)
{
    constexpr int M = 10, N = 6, P = 3;

    // Build W with orthogonal columns (simple basis-like vectors scaled).
    // W col 0: first half = 1, rest = 0.
    // W col 1: second half = 1, rest = 0.
    // W col 2: alternating +1/-1.
    std::vector<float> W_h(M * P, 0.f);
    for (int g = 0; g < M / 2; ++g)      W_h[g + 0 * M] = 1.f;    // col 0
    for (int g = M / 2; g < M; ++g)      W_h[g + 1 * M] = 1.f;    // col 1
    for (int g = 0; g < M; ++g)          W_h[g + 2 * M] = (g % 2 == 0) ? 1.f : -1.f;  // col 2

    // Random X (row-major).
    std::mt19937 rng(0x4F2A9C1BULL);
    std::uniform_real_distribution<float> u(0.f, 5.f);
    std::vector<float> X_row(M * N);
    for (auto& v : X_row) v = u(rng);

    auto X_col = rowmaj_to_colmaj(X_row, M, N);
    auto ref   = cpu_mlm(M, N, P, X_col, W_h, 1e-6f);

    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    singlet_gpu::enrich::MlmConfig cfg;
    cfg.ridge = 1e-6f;
    auto res  = singlet_gpu::enrich::mlm(pz, dW.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    float max_err = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float err = std::fabs(gpu[i] - ref[i]);
        max_err   = std::max(max_err, err);
    }

    EXPECT_LT(max_err, 1e-3f)
        << "OrthogonalPathways: max abs_err=" << max_err << " exceeds 1e-3";

    std::printf("[REGISTRY] %s | enrich/decoupler_mlm/orthogonal | 10x6x3 | "
                "abs_err | max=%.2e | 1e-3 | none | CYCLE-136 | %s\n",
                __DATE__, max_err, (max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Mlm_CorrelatedPathways_DiffersFromULM
//
// When pathways are correlated (w_2 = w_1 + small noise), MLM should give
// DIFFERENT scores than per-pathway univariate OLS (ULM).  The correlated
// pathway double-counts shared signal in ULM but not in MLM.
//
// Strategy: build two pathways where W[:,1] = W[:,0] + 0.05*noise.
// Compute MLM scores and per-pathway CPU OLS scores (as ULM would produce).
// Assert the maximum absolute difference in scores > 0.05 (exceeds noise).
// =============================================================================
TEST(MlmTest, Mlm_CorrelatedPathways_DiffersFromULM)
{
    constexpr int M = 20, N = 15, P = 2;

    std::mt19937 rng(0xABCDEF01ULL);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);
    std::uniform_real_distribution<float> ux(0.f, 5.f);

    // W[:,0]: random base pathway.
    std::vector<float> W_h(M * P);
    for (int g = 0; g < M; ++g)
        W_h[g + 0 * M] = uw(rng);
    // W[:,1] = W[:,0] + 0.05 * noise (highly correlated).
    std::uniform_real_distribution<float> unoise(-0.05f, 0.05f);
    for (int g = 0; g < M; ++g)
        W_h[g + 1 * M] = W_h[g + 0 * M] + unoise(rng);

    // Random X (row-major, dense to ensure nonzero correlation signal).
    std::vector<float> X_row(M * N);
    for (auto& v : X_row) v = ux(rng);

    auto X_col = rowmaj_to_colmaj(X_row, M, N);

    // MLM scores (joint).
    auto csc   = dense_to_csc(M, N, X_row);
    auto pz    = upload_csc(csc);
    auto dW    = upload_W(W_h);

    singlet_gpu::enrich::MlmConfig cfg;
    cfg.ridge = 1e-3f;   // more ridge for near-rank-deficient case
    auto res   = singlet_gpu::enrich::mlm(pz, dW.get(), P, cfg);
    auto gpu   = d2h(res.scores, N * P);

    // ULM-equivalent: per-pathway univariate OLS via cpu_mlm on each pathway
    // independently (P=1 for each).
    std::vector<float> ulm_scores(N * P);
    for (int q = 0; q < P; ++q) {
        // Extract W[:,q] as a separate p=1 weight matrix.
        std::vector<float> Wq(M);
        for (int g = 0; g < M; ++g)
            Wq[g] = W_h[g + q * M];
        auto ulm_q = cpu_mlm(M, N, 1, X_col, Wq, 0.f);  // no ridge for ULM
        for (int c = 0; c < N; ++c)
            ulm_scores[c + q * N] = ulm_q[c];
    }

    // For pathway 1 (highly correlated with 0), MLM and ULM must differ.
    float max_diff_p1 = 0.f;
    for (int c = 0; c < N; ++c) {
        float diff = std::fabs(gpu[c + 1 * N] - ulm_scores[c + 1 * N]);
        max_diff_p1 = std::max(max_diff_p1, diff);
    }

    // Verify all GPU scores are finite.
    for (int i = 0; i < N * P; ++i) {
        EXPECT_TRUE(std::isfinite(gpu[i]))
            << "non-finite MLM score at element " << i;
    }

    EXPECT_GT(max_diff_p1, 0.05f)
        << "CorrelatedPathways: MLM and ULM should differ for correlated pathway "
           "(max_diff=" << max_diff_p1 << "); correlation not detected by MLM";

    std::printf("[REGISTRY] %s | enrich/decoupler_mlm/correlated_vs_ulm | 20x15x2 | "
                "max_diff | max=%.2e | >0.05 | none | CYCLE-136 | %s\n",
                __DATE__, max_diff_p1, (max_diff_p1 > 0.05f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Mlm_Determinism_BitIdentical
//
// Two consecutive MLM calls on the same data must produce bit-identical
// results (rel_err = 0).  cuBLAS Sgemm + cuSPARSE SpMM + cuSOLVER Potrf/Potrs
// + cuBLAS Sgeam are all deterministic at fp32 for fixed inputs.
// =============================================================================
TEST(MlmTest, Mlm_Determinism_BitIdentical)
{
    constexpr int M = 30, N = 50, P = 4;

    std::mt19937 rng(0x12345678ULL);
    std::poisson_distribution<int> pd(3);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);

    std::vector<float> X_row(M * N);
    for (auto& v : X_row) v = static_cast<float>(pd(rng));

    std::vector<float> W_h(M * P);
    for (auto& v : W_h) v = uw(rng);

    auto csc = dense_to_csc(M, N, X_row);

    singlet_gpu::enrich::MlmConfig cfg;
    cfg.ridge = 1e-6f;

    auto pz1 = upload_csc(csc);
    auto dW1 = upload_W(W_h);
    auto r1  = singlet_gpu::enrich::mlm(pz1, dW1.get(), P, cfg);
    auto h1  = d2h(r1.scores, N * P);

    auto pz2 = upload_csc(csc);
    auto dW2 = upload_W(W_h);
    auto r2  = singlet_gpu::enrich::mlm(pz2, dW2.get(), P, cfg);
    auto h2  = d2h(r2.scores, N * P);

    float max_rel = 0.f;
    for (int i = 0; i < N * P; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
        EXPECT_EQ(h1[i], h2[i])
            << "Determinism: runs differ at element " << i
            << " (h1=" << h1[i] << " h2=" << h2[i] << ")";
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_mlm/determinism | 30x50x4 | "
                "rel_err | %.2e | 0 | none | CYCLE-136 | %s\n",
                __DATE__, max_rel, (max_rel == 0.f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Mlm_RankDeficient_RidgeStabilizes
//
// Build W where pathway 1 is nearly identical to pathway 0 (rank-deficient).
// Without sufficient ridge, Cholesky would fail.  With cfg.ridge = 1e-3,
// the regression is stabilized and produces finite scores.
// Verify no NaN/Inf and that all scores are finite.
// =============================================================================
TEST(MlmTest, Mlm_RankDeficient_RidgeStabilizes)
{
    constexpr int M = 15, N = 10, P = 2;

    std::mt19937 rng(0xDEADBEEFULL);
    std::uniform_real_distribution<float> uw(-1.f, 1.f);
    std::uniform_real_distribution<float> ux(0.f, 4.f);

    // W[:,0]: random base pathway.
    std::vector<float> W_h(M * P);
    for (int g = 0; g < M; ++g)
        W_h[g + 0 * M] = uw(rng);
    // W[:,1] = W[:,0] + 1e-4 * noise (nearly identical → near-singular W^T W).
    std::uniform_real_distribution<float> tiny(-1e-4f, 1e-4f);
    for (int g = 0; g < M; ++g)
        W_h[g + 1 * M] = W_h[g + 0 * M] + tiny(rng);

    // Dense X (row-major).
    std::vector<float> X_row(M * N);
    for (auto& v : X_row) v = ux(rng);

    auto csc = dense_to_csc(M, N, X_row);
    auto pz  = upload_csc(csc);
    auto dW  = upload_W(W_h);

    // With ridge = 1e-3 the system should be stabilized.
    singlet_gpu::enrich::MlmConfig cfg;
    cfg.ridge = 1e-3f;
    auto res  = singlet_gpu::enrich::mlm(pz, dW.get(), P, cfg);
    auto gpu  = d2h(res.scores, N * P);

    ASSERT_EQ(res.n_cells,    N);
    ASSERT_EQ(res.n_pathways, P);

    int finite_count = 0;
    for (int i = 0; i < N * P; ++i) {
        bool fin = std::isfinite(gpu[i]);
        EXPECT_TRUE(fin)
            << "non-finite score at element " << i << " value=" << gpu[i];
        if (fin) ++finite_count;
    }

    EXPECT_EQ(finite_count, N * P)
        << "RidgeStabilizes: " << (N * P - finite_count) << " non-finite scores";

    std::printf("[REGISTRY] %s | enrich/decoupler_mlm/ridge_stabilize | 15x10x2 | "
                "finite_scores | %d/%d | all | none | CYCLE-136 | %s\n",
                __DATE__, finite_count, N * P,
                (finite_count == N * P) ? "PASS" : "FAIL");
}
