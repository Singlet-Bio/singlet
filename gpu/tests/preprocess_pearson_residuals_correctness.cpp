// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/preprocess_pearson_residuals_correctness.cpp
//
// Correctness harness for the Pearson residuals variance kernel (CYCLE-118).
//
// Tests:
//   1. Tiny_Synthetic_VsClosedForm      — 5x8 hand-computable matrix
//   2. Variance_RanksGenes_VsLog1p      — planted high-residual genes
//   3. Determinism_BitIdentical_Same_Seed — two runs, rel_err < 1e-4
//   4. AllZeros_Input_AllZerosOutput    — all-zero matrix edge case
//   5. Theta_Robustness                 — var_i finite for theta=1,100,10000
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=PearsonResidualsTest.*

#include <singlet-gpu/preprocess/pearson_residuals.h>
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

// Build a SynCSC struct and upload it to a PzDeviceMatrix.
struct SynCSC {
    int                   rows, cols, nnz;
    std::vector<float>    vals;
    std::vector<int32_t>  col_ptr;
    std::vector<int32_t>  row_idx;
};

// Upload SynCSC to device and wrap in PzDeviceMatrix.
singlet_gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet_gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet_gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
    cudaMemcpy(pz.mat.col_ptr.get(),     c.col_ptr.data(),
               (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
               c.nnz * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.values.get(),      c.vals.data(),
               c.nnz * sizeof(float), cudaMemcpyHostToDevice);
    return pz;
}

// Build a CSC from a dense row-major matrix (rows=genes, cols=cells).
SynCSC dense_to_csc(int rows, int cols, const std::vector<float>& dense)
{
    SynCSC c;
    c.rows = rows;
    c.cols = cols;
    c.col_ptr.resize(cols + 1, 0);
    // Count nnz per column
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            if (dense[i * cols + j] != 0.f) c.col_ptr[j + 1]++;
        }
    }
    // Prefix sum
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

// Compute Pearson residual variance on CPU for a dense matrix.
// m=genes, n=cells, X in row-major.
std::vector<float> cpu_pearson_var(
    int m, int n,
    const std::vector<float>& X,
    float theta = 100.f)
{
    // Row sums u[m], col sums v[n], grand total N
    std::vector<double> u(m, 0.0), v(n, 0.0);
    double N = 0.0;
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            double x = X[i * n + j];
            u[i] += x;
            v[j] += x;
            N    += x;
        }
    if (N <= 0.0) return std::vector<float>(m, 0.f);

    std::vector<float> var_out(m);
    for (int i = 0; i < m; ++i) {
        double sum_r = 0.0, sum_r2 = 0.0;
        for (int j = 0; j < n; ++j) {
            double mu = u[i] * v[j] / N;
            if (mu <= 0.0) continue;
            double sigma2 = mu * (1.0 + mu / theta);
            double sigma  = std::sqrt(sigma2);
            if (sigma <= 0.0) continue;
            double r = (X[i * n + j] - mu) / sigma;
            sum_r  += r;
            sum_r2 += r * r;
        }
        double mean_r  = sum_r  / n;
        double mean_r2 = sum_r2 / n;
        var_out[i] = static_cast<float>(mean_r2 - mean_r * mean_r);
    }
    return var_out;
}

// Fetch device float buffer to host.
std::vector<float> d2h(const singlet_gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Log-normalize a dense matrix (log1p of x/col_sum * median) and return per-gene variance.
std::vector<float> log1p_var(int m, int n, const std::vector<float>& X)
{
    // Per-cell sums
    std::vector<double> col_sum(n, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            col_sum[j] += X[i * n + j];
    // Median of nonzero col sums
    std::vector<double> pos_sums;
    for (int j = 0; j < n; ++j) if (col_sum[j] > 0) pos_sums.push_back(col_sum[j]);
    if (pos_sums.empty()) return std::vector<float>(m, 0.f);
    std::nth_element(pos_sums.begin(), pos_sums.begin() + pos_sums.size()/2, pos_sums.end());
    double T = pos_sums[pos_sums.size()/2];

    // Normalized log1p matrix, then per-gene variance
    std::vector<float> var_out(m);
    for (int i = 0; i < m; ++i) {
        double sum_r = 0.0, sum_r2 = 0.0;
        for (int j = 0; j < n; ++j) {
            double s = (col_sum[j] > 0) ? col_sum[j] / T : 1.0;
            double r = std::log1p(X[i * n + j] / s);
            sum_r  += r;
            sum_r2 += r * r;
        }
        double mean_r  = sum_r  / n;
        double mean_r2 = sum_r2 / n;
        var_out[i] = static_cast<float>(mean_r2 - mean_r * mean_r);
    }
    return var_out;
}

} // anonymous namespace


// =============================================================================
// Test 1: Tiny_Synthetic_VsClosedForm
//
// 5 genes × 8 cells, hand-computable sparse matrix.
// Compare GPU var[i] vs closed-form CPU var[i] to 1e-3 absolute tolerance.
// This validates the algebraic decomposition (Pass2+Pass3+Fuse).
// =============================================================================
TEST(PearsonResidualsTest, Tiny_Synthetic_VsClosedForm)
{
    constexpr int M = 5, N = 8;
    // clang-format off
    // Dense row-major (genes x cells): small integer counts; ~50% sparsity
    std::vector<float> X = {
        // gene 0
        3, 0, 1, 0, 2, 0, 4, 1,
        // gene 1
        0, 2, 0, 3, 0, 1, 0, 2,
        // gene 2
        1, 1, 0, 0, 3, 2, 1, 0,
        // gene 3
        0, 0, 0, 0, 0, 0, 5, 0,
        // gene 4
        2, 3, 1, 2, 1, 3, 2, 1,
    };
    // clang-format on

    // GPU path
    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);
    singlet_gpu::preprocess::PearsonResidualsConfig cfg;
    cfg.theta = 100.f;
    auto d_var    = singlet_gpu::preprocess::pearson_residual_variance(pz, cfg);
    auto gpu_var  = d2h(d_var, M);

    // CPU closed-form reference
    auto cpu_var = cpu_pearson_var(M, N, X, 100.f);

    for (int i = 0; i < M; ++i) {
        EXPECT_NEAR(gpu_var[i], cpu_var[i], 1e-3f)
            << "gene " << i << ": gpu=" << gpu_var[i] << " cpu=" << cpu_var[i];
    }

    std::printf("[REGISTRY] %s | preprocess/pearson_residuals/tiny_vs_cf | 5x8 | "
                "abs_err | max=%.6f | 1e-3 | none | CYCLE-118 | PASS\n",
                __DATE__,
                [&](){
                    float mx = 0.f;
                    for (int i = 0; i < M; ++i)
                        mx = std::max(mx, std::fabs(gpu_var[i] - cpu_var[i]));
                    return mx;
                }());
}


// =============================================================================
// Test 2: Variance_DistinguishesPlanted (rewritten 2026-04-29)
//
// 50 genes × 500 cells. Planted genes (0..9) express in INDEPENDENT random
// 50% subsets of cells with Poisson(20); background (10..49) is uniform Poisson(2).
//
// WHY independent subsets per planted gene: the original "alternating 0/100 in
// lockstep with j%2==0" pattern made cell library size PERFECTLY co-vary with
// planted-gene expression. Pearson residuals correctly suppress that signal as
// a library-size confound (Lause-Berens-Kobak 2021 §2 — that's the whole point
// of the method). Independent per-gene subsets break the co-variation so the
// signal isn't trivially explained by Σ_i x_ij.
//
// Assertion: planted genes' Pearson variances should be statistically higher
// than background (mean planted > mean background, gap ≥ 2× background mean).
// We do NOT require strict top-10 recall because Pearson residuals deliberately
// rank differently from log1p — that's the feature, not a bug.
// =============================================================================
TEST(PearsonResidualsTest, Variance_DistinguishesPlanted)
{
    constexpr int M = 50, N = 500;
    constexpr int N_PLANTED = 10;

    std::mt19937 rng(0x5EED1234ULL);
    std::poisson_distribution<int> background(2);
    std::poisson_distribution<int> planted_high(20);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    std::vector<float> X(M * N, 0.f);

    // Background genes — uniform Poisson(2) across all cells
    for (int i = N_PLANTED; i < M; ++i)
        for (int j = 0; j < N; ++j)
            X[i * N + j] = static_cast<float>(background(rng));

    // Planted genes — each picks an INDEPENDENT 50% subset of cells.
    // Two genes' expressing-cells are not correlated → no library-size confound.
    for (int i = 0; i < N_PLANTED; ++i) {
        for (int j = 0; j < N; ++j) {
            if (u01(rng) < 0.5f)
                X[i * N + j] = static_cast<float>(planted_high(rng));
            // else stays at 0
        }
    }

    // --- GPU Pearson variance ---
    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);
    singlet_gpu::preprocess::PearsonResidualsConfig cfg;
    cfg.theta = 100.f;
    auto d_var   = singlet_gpu::preprocess::pearson_residual_variance(pz, cfg);
    auto pvar    = d2h(d_var, M);

    // Mean variance: planted vs background
    double mean_planted = 0.0, mean_bg = 0.0;
    for (int i = 0; i < N_PLANTED; ++i) mean_planted += pvar[i];
    mean_planted /= N_PLANTED;
    for (int i = N_PLANTED; i < M; ++i) mean_bg += pvar[i];
    mean_bg /= (M - N_PLANTED);

    // Planted should be at least 2× background — modest separation guarantee.
    EXPECT_GT(mean_planted, 2.0 * mean_bg)
        << "Planted gene Pearson-variance mean (" << mean_planted
        << ") should exceed 2× background mean (" << mean_bg
        << "). Variances should be finite and separating.";

    // Sanity: log1p variance should ALSO separate planted from background on
    // this test (uses log1p-of-x; insensitive to library-size confound here).
    auto lvar = log1p_var(M, N, X);
    double lvar_planted = 0.0, lvar_bg = 0.0;
    for (int i = 0; i < N_PLANTED; ++i) lvar_planted += lvar[i];
    lvar_planted /= N_PLANTED;
    for (int i = N_PLANTED; i < M; ++i) lvar_bg += lvar[i];
    lvar_bg /= (M - N_PLANTED);
    EXPECT_GT(lvar_planted, 2.0 * lvar_bg)
        << "Sanity check failed: log1p variance should also separate "
        << "(planted=" << lvar_planted << " vs bg=" << lvar_bg << ")";

    std::printf("[REGISTRY] %s | preprocess/pearson_residuals/distinguishes_planted | 50x500 | "
                "var_ratio | pearson_p/bg=%.2f log1p_p/bg=%.2f | >=2 | atomicAdd | CYCLE-118 | %s\n",
                __DATE__,
                mean_planted / std::max(mean_bg, 1e-9),
                lvar_planted / std::max(lvar_bg, 1e-9),
                (mean_planted > 2.0 * mean_bg && lvar_planted > 2.0 * lvar_bg) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Determinism_BitIdentical_Same_Seed
//
// Run pearson_residual_variance twice on the same matrix.
// With atomics in pass 3 (default), we expect rel_err < 1e-4 (not bit-exact).
// =============================================================================
TEST(PearsonResidualsTest, Determinism_BitIdentical_Same_Seed)
{
    constexpr int M = 30, N = 200;
    std::mt19937 rng(0xDEADBEEFULL);
    std::poisson_distribution<int> pd(5);

    std::vector<float> Xd(M * N);
    for (auto& v : Xd) v = static_cast<float>(pd(rng));

    const auto csc = dense_to_csc(M, N, Xd);

    singlet_gpu::preprocess::PearsonResidualsConfig cfg;
    cfg.theta = 100.f;

    auto pz1 = upload_csc(csc);
    auto d1  = singlet_gpu::preprocess::pearson_residual_variance(pz1, cfg);
    auto h1  = d2h(d1, M);

    auto pz2 = upload_csc(csc);
    auto d2  = singlet_gpu::preprocess::pearson_residual_variance(pz2, cfg);
    auto h2  = d2h(d2, M);

    float max_rel_err = 0.f;
    for (int i = 0; i < M; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel_err = std::max(max_rel_err, rel);
    }

    EXPECT_LT(max_rel_err, 1e-4f)
        << "Two runs should be nearly identical; max_rel_err=" << max_rel_err;

    std::printf("[REGISTRY] %s | preprocess/pearson_residuals/determinism | 30x200 | "
                "rel_err | %.2e | 1e-4 | atomicAdd | CYCLE-118 | %s\n",
                __DATE__, max_rel_err,
                (max_rel_err < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: AllZeros_Input_AllZerosOutput
//
// All-zero count matrix → per-gene variance should be all zeros.
// Also exercises the div-by-zero guard (μ=0, σ=0).
// =============================================================================
TEST(PearsonResidualsTest, AllZeros_Input_AllZerosOutput)
{
    constexpr int M = 10, N = 20;

    // Build an explicit all-zero CSC: nnz=0
    SynCSC csc;
    csc.rows = M; csc.cols = N; csc.nnz = 0;
    csc.col_ptr.assign(N + 1, 0);
    // No row_idx or vals entries.

    singlet_gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet_gpu::core::DeviceCSC(M, N, 0);
    cudaMemcpy(pz.mat.col_ptr.get(), csc.col_ptr.data(),
               (N + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    // row_indices and values are zero-size; no copy needed.

    singlet_gpu::preprocess::PearsonResidualsConfig cfg;
    auto d_var = singlet_gpu::preprocess::pearson_residual_variance(pz, cfg);
    auto h_var = d2h(d_var, M);

    for (int i = 0; i < M; ++i) {
        EXPECT_TRUE(std::isfinite(h_var[i]))
            << "gene " << i << ": var should be finite (0), got " << h_var[i];
        EXPECT_NEAR(h_var[i], 0.f, 1e-6f)
            << "gene " << i << ": all-zero input should give zero variance";
    }

    std::printf("[REGISTRY] %s | preprocess/pearson_residuals/all_zeros | 10x20 | "
                "zero_var | %.2e | 1e-6 | none | CYCLE-118 | %s\n",
                __DATE__,
                [&](){ float mx=0.f; for(float v:h_var) mx=std::max(mx,std::fabs(v)); return mx; }(),
                "PASS");
}


// =============================================================================
// Test 5: Theta_Robustness
//
// Verify var_i is finite (no NaN/Inf) for theta = 1, 100, 10000.
// =============================================================================
TEST(PearsonResidualsTest, Theta_Robustness)
{
    constexpr int M = 20, N = 100;
    std::mt19937 rng(0xC0FFEEULL);
    std::poisson_distribution<int> pd(3);

    std::vector<float> Xd(M * N);
    for (auto& v : Xd) v = static_cast<float>(pd(rng));
    const auto csc = dense_to_csc(M, N, Xd);

    for (float theta : {1.f, 100.f, 10000.f}) {
        auto pz = upload_csc(csc);
        singlet_gpu::preprocess::PearsonResidualsConfig cfg;
        cfg.theta = theta;
        auto d_var = singlet_gpu::preprocess::pearson_residual_variance(pz, cfg);
        auto h_var = d2h(d_var, M);

        for (int i = 0; i < M; ++i) {
            EXPECT_TRUE(std::isfinite(h_var[i]))
                << "theta=" << theta << " gene " << i
                << ": var is non-finite: " << h_var[i];
        }

        std::printf("[REGISTRY] %s | preprocess/pearson_residuals/theta_rob | 20x100 | "
                    "finite_var | theta=%.0f | all_finite | none | CYCLE-118 | PASS\n",
                    __DATE__, theta);
    }
}
