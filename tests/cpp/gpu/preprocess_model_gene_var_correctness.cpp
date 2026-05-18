// SPDX-License-Identifier: MIT
// singlet/gpu/tests/preprocess_model_gene_var_correctness.cpp
//
// Correctness harness for the modelGeneVarByPoisson HVG kernel (CYCLE-127).
//
// Tests:
//   1. Tiny_Closed_Form               — 5x8 hand-computable matrix vs CPU reference
//   2. Pure_Poisson_Background_Zero_BioVar — 100x500 Poisson(2): bio_var ≈ 0
//   3. Planted_HighVar_Genes_Recovered — 50x500: 10 high-var genes in top-10 HVGs
//   4. Determinism_Same_Input          — two runs rel_err < 1e-4
//   5. Min_Mean_Filter                 — genes below min_mean excluded from HVGs
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter=ModelGeneVarTest.*

#include <singlet/gpu/preprocess/model_gene_var.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <vector>

// =============================================================================
// Shared test helpers
// =============================================================================

namespace {

// Build a SynCSC struct for upload.
struct SynCSC {
    int                  rows, cols, nnz;
    std::vector<float>   vals;
    std::vector<int32_t> col_ptr;
    std::vector<int32_t> row_idx;
};

// Upload SynCSC to device and wrap in PzDeviceMatrix.
singlet::gpu::io::PzDeviceMatrix upload_csc(const SynCSC& c)
{
    singlet::gpu::io::PzDeviceMatrix pz;
    pz.mat = singlet::gpu::core::DeviceCSC(c.rows, c.cols, c.nnz);
    cudaMemcpy(pz.mat.col_ptr.get(),     c.col_ptr.data(),
               (c.cols + 1) * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.row_indices.get(), c.row_idx.data(),
               c.nnz * sizeof(int32_t),  cudaMemcpyHostToDevice);
    cudaMemcpy(pz.mat.values.get(),      c.vals.data(),
               c.nnz * sizeof(float),    cudaMemcpyHostToDevice);
    return pz;
}

// Build a CSC from a dense row-major matrix (rows=genes, cols=cells).
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
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i) {
            float v = dense[i * cols + j];
            if (v != 0.f) {
                int pos = c.col_ptr[j] + fill[j]++;
                c.row_idx[pos] = i;
                c.vals[pos]    = v;
            }
        }
    return c;
}

// Fetch device float buffer to host.
std::vector<float> d2h_f(const singlet::gpu::core::DeviceMemory<float>& d, int n)
{
    std::vector<float> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
}

// Fetch device int buffer to host.
std::vector<int> d2h_i(const singlet::gpu::core::DeviceMemory<int>& d, int n)
{
    std::vector<int> h(n);
    cudaMemcpy(h.data(), d.get(), n * sizeof(int), cudaMemcpyDeviceToHost);
    return h;
}

// CPU reference: per-gene mean, sample variance, bio_var under Poisson null.
struct CpuMGV {
    std::vector<double> mean;
    std::vector<double> total_var;
    std::vector<double> bio_var;
};

CpuMGV cpu_model_gene_var(int m, int n, const std::vector<float>& X)
{
    CpuMGV r;
    r.mean.resize(m, 0.0);
    r.total_var.resize(m, 0.0);
    r.bio_var.resize(m, 0.0);

    // Mean
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) s += X[i * n + j];
        r.mean[i] = s / n;
    }
    // Sample variance (Bessel-corrected)
    for (int i = 0; i < m; ++i) {
        double ss = 0.0;
        double mu = r.mean[i];
        for (int j = 0; j < n; ++j) {
            double d = X[i * n + j] - mu;
            ss += d * d;
        }
        r.total_var[i] = (n > 1) ? ss / (n - 1) : 0.0;
    }
    // Bio var: max(0, total_var - mean)
    for (int i = 0; i < m; ++i) {
        double bv = r.total_var[i] - r.mean[i];
        r.bio_var[i] = (bv > 0.0) ? bv : 0.0;
    }
    return r;
}

} // anonymous namespace


// =============================================================================
// Test 1: Tiny_Closed_Form
//
// 5 genes × 8 cells, small integer values. Verify GPU mean ≈ CPU mean to 1e-5
// and GPU total_var ≈ CPU total_var to 1e-3 absolute tolerance.
// =============================================================================
TEST(ModelGeneVarTest, Tiny_Closed_Form)
{
    constexpr int M = 5, N = 8;
    // clang-format off
    // Dense row-major (genes × cells)
    std::vector<float> X = {
        // gene 0: mean=(3+0+1+0+2+0+4+1)/8=11/8=1.375
        3, 0, 1, 0, 2, 0, 4, 1,
        // gene 1: mean=(0+2+0+3+0+1+0+2)/8=8/8=1.0
        0, 2, 0, 3, 0, 1, 0, 2,
        // gene 2: mean=(1+1+0+0+3+2+1+0)/8=8/8=1.0
        1, 1, 0, 0, 3, 2, 1, 0,
        // gene 3: mean=5/8=0.625 (only one nonzero)
        0, 0, 0, 0, 0, 0, 5, 0,
        // gene 4: mean=(2+3+1+2+1+3+2+1)/8=15/8=1.875
        2, 3, 1, 2, 1, 3, 2, 1,
    };
    // clang-format on

    // GPU path
    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);

    singlet::gpu::preprocess::ModelGeneVarConfig cfg;
    cfg.n_top = M; // keep all genes
    auto res  = singlet::gpu::preprocess::model_gene_var(pz, cfg);

    auto gpu_mean  = d2h_f(res.mean,      M);
    auto gpu_tvar  = d2h_f(res.total_var, M);
    auto gpu_bvar  = d2h_f(res.bio_var,   M);

    // CPU reference
    auto ref = cpu_model_gene_var(M, N, X);

    for (int i = 0; i < M; ++i) {
        EXPECT_NEAR(gpu_mean[i], static_cast<float>(ref.mean[i]), 1e-5f)
            << "gene " << i << " mean: gpu=" << gpu_mean[i]
            << " cpu=" << ref.mean[i];
        EXPECT_NEAR(gpu_tvar[i], static_cast<float>(ref.total_var[i]), 1e-3f)
            << "gene " << i << " total_var: gpu=" << gpu_tvar[i]
            << " cpu=" << ref.total_var[i];
        // bio_var must be >= 0
        EXPECT_GE(gpu_bvar[i], 0.f) << "gene " << i << " bio_var negative";
    }

    float max_mean_err = 0.f, max_var_err = 0.f;
    for (int i = 0; i < M; ++i) {
        max_mean_err = std::max(max_mean_err,
            std::fabs(gpu_mean[i] - static_cast<float>(ref.mean[i])));
        max_var_err  = std::max(max_var_err,
            std::fabs(gpu_tvar[i] - static_cast<float>(ref.total_var[i])));
    }

    std::printf("[REGISTRY] %s | preprocess/model_gene_var/tiny_closed_form | 5x8 | "
                "abs_err | mean_max=%.2e var_max=%.2e | 1e-5,1e-3 | atomicAdd | CYCLE-127 | %s\n",
                __DATE__, max_mean_err, max_var_err,
                (max_mean_err < 1e-5f && max_var_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Pure_Poisson_Background_Has_Zero_BioVar
//
// 100×500 random Poisson(2) matrix. Under Poisson sampling, E[X]=Var[X]=λ=2,
// so total_var ≈ 2 ≈ mean, giving bio_var = max(0, total_var - mean) ≈ 0.
// Assert: median(bio_var) < 0.5. (Some fluctuation expected for n=500.)
// =============================================================================
TEST(ModelGeneVarTest, Pure_Poisson_Background_Has_Zero_BioVar)
{
    constexpr int M = 100, N = 500;

    std::mt19937 rng(0xAB12CD34ULL);
    std::poisson_distribution<int> pd(2);

    std::vector<float> X(M * N);
    for (auto& v : X) v = static_cast<float>(pd(rng));

    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);

    singlet::gpu::preprocess::ModelGeneVarConfig cfg;
    cfg.n_top = M;
    auto res  = singlet::gpu::preprocess::model_gene_var(pz, cfg);

    auto gpu_bvar = d2h_f(res.bio_var, M);
    auto gpu_mean = d2h_f(res.mean,    M);

    // Mean should be near 2 for all genes
    for (int i = 0; i < M; ++i) {
        EXPECT_NEAR(gpu_mean[i], 2.f, 0.5f)
            << "gene " << i << " mean far from Poisson(2) expected value";
    }

    // Median bio_var should be near 0
    std::vector<float> sorted_bvar = gpu_bvar;
    std::nth_element(sorted_bvar.begin(),
                     sorted_bvar.begin() + M / 2,
                     sorted_bvar.end());
    float median_bvar = sorted_bvar[M / 2];

    EXPECT_LT(median_bvar, 0.5f)
        << "Median bio_var=" << median_bvar
        << " should be near 0 for Poisson(2) data (Poisson null is exact)";

    std::printf("[REGISTRY] %s | preprocess/model_gene_var/poisson_background | 100x500 | "
                "median_bio_var | %.4f | <0.5 | atomicAdd | CYCLE-127 | %s\n",
                __DATE__, median_bvar, (median_bvar < 0.5f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Planted_HighVar_Genes_Recovered
//
// 50×500 matrix.
//   Genes 0..9:  Poisson(2) backbone + burst (10% of cells get Poisson(50)).
//   Genes 10..49: pure Poisson(2).
//
// After model_gene_var with n_top=10, hvg_indices should contain ≥ 8 of {0..9}.
// Also assert: mean(bio_var[0..9]) > 5 × mean(bio_var[10..49]).
// =============================================================================
TEST(ModelGeneVarTest, Planted_HighVar_Genes_Recovered)
{
    constexpr int M = 50, N = 500;
    constexpr int N_PLANTED = 10;

    std::mt19937 rng(0xDEAD5EED);
    std::poisson_distribution<int> pd_bg(2);
    std::poisson_distribution<int> pd_burst(50);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    std::vector<float> X(M * N, 0.f);

    // Background genes: Poisson(2) everywhere
    for (int i = N_PLANTED; i < M; ++i)
        for (int j = 0; j < N; ++j)
            X[i * N + j] = static_cast<float>(pd_bg(rng));

    // Planted genes: Poisson(2) backbone + 10% burst cells get Poisson(50)
    for (int i = 0; i < N_PLANTED; ++i)
        for (int j = 0; j < N; ++j) {
            if (u01(rng) < 0.10f)
                X[i * N + j] = static_cast<float>(pd_burst(rng));
            else
                X[i * N + j] = static_cast<float>(pd_bg(rng));
        }

    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);

    singlet::gpu::preprocess::ModelGeneVarConfig cfg;
    cfg.n_top = N_PLANTED; // ask for exactly 10 HVGs
    auto res  = singlet::gpu::preprocess::model_gene_var(pz, cfg);

    ASSERT_EQ(res.n_hvg, N_PLANTED);

    auto hvg = d2h_i(res.hvg_indices, res.n_hvg);
    auto bvar = d2h_f(res.bio_var, M);

    // Count planted genes in the returned HVG set
    std::set<int> planted_set;
    for (int i = 0; i < N_PLANTED; ++i) planted_set.insert(i);

    int n_planted_recovered = 0;
    for (int idx : hvg) {
        if (planted_set.count(idx)) n_planted_recovered++;
    }

    EXPECT_GE(n_planted_recovered, 8)
        << "Expected >=8 of 10 planted genes in top-10 HVGs; got "
        << n_planted_recovered;

    // Mean bio_var separation: planted >> background
    double mean_planted = 0.0, mean_bg = 0.0;
    for (int i = 0; i < N_PLANTED; ++i)      mean_planted += bvar[i];
    for (int i = N_PLANTED; i < M; ++i) mean_bg     += bvar[i];
    mean_planted /= N_PLANTED;
    mean_bg      /= (M - N_PLANTED);

    EXPECT_GT(mean_planted, 5.0 * mean_bg)
        << "Planted bio_var mean=" << mean_planted
        << " should be >5x background mean=" << mean_bg;

    std::printf("[REGISTRY] %s | preprocess/model_gene_var/planted_hvg | 50x500 | "
                "recall+ratio | recovered=%d/10 ratio=%.1f | >=8,>=5x | atomicAdd | CYCLE-127 | %s\n",
                __DATE__, n_planted_recovered,
                mean_bg > 1e-9 ? mean_planted / mean_bg : 0.0,
                (n_planted_recovered >= 8 && mean_planted > 5.0 * mean_bg) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Determinism_Same_Input
//
// Two runs on the same matrix. With atomicAdd (default), expect rel_err < 1e-4.
// =============================================================================
TEST(ModelGeneVarTest, Determinism_Same_Input)
{
    constexpr int M = 30, N = 200;

    std::mt19937 rng(0xFEEDC0DE);
    std::poisson_distribution<int> pd(5);

    std::vector<float> Xd(M * N);
    for (auto& v : Xd) v = static_cast<float>(pd(rng));

    const auto csc = dense_to_csc(M, N, Xd);

    singlet::gpu::preprocess::ModelGeneVarConfig cfg;
    cfg.n_top = 10;

    auto pz1 = upload_csc(csc);
    auto r1  = singlet::gpu::preprocess::model_gene_var(pz1, cfg);
    auto h1_mean = d2h_f(r1.mean,      M);
    auto h1_tvar = d2h_f(r1.total_var, M);
    auto h1_bvar = d2h_f(r1.bio_var,   M);

    auto pz2 = upload_csc(csc);
    auto r2  = singlet::gpu::preprocess::model_gene_var(pz2, cfg);
    auto h2_mean = d2h_f(r2.mean,      M);
    auto h2_tvar = d2h_f(r2.total_var, M);
    auto h2_bvar = d2h_f(r2.bio_var,   M);

    float max_rel_mean = 0.f, max_rel_tvar = 0.f, max_rel_bvar = 0.f;
    for (int i = 0; i < M; ++i) {
        float denom_mean = std::max(std::fabs(h1_mean[i]), 1e-8f);
        float denom_tvar = std::max(std::fabs(h1_tvar[i]), 1e-8f);
        float denom_bvar = std::max(std::fabs(h1_bvar[i]), 1e-8f);
        max_rel_mean = std::max(max_rel_mean,
            std::fabs(h1_mean[i] - h2_mean[i]) / denom_mean);
        max_rel_tvar = std::max(max_rel_tvar,
            std::fabs(h1_tvar[i] - h2_tvar[i]) / denom_tvar);
        max_rel_bvar = std::max(max_rel_bvar,
            std::fabs(h1_bvar[i] - h2_bvar[i]) / denom_bvar);
    }

    EXPECT_LT(max_rel_mean, 1e-4f)
        << "mean: max_rel_err=" << max_rel_mean;
    EXPECT_LT(max_rel_tvar, 1e-4f)
        << "total_var: max_rel_err=" << max_rel_tvar;
    EXPECT_LT(max_rel_bvar, 1e-4f)
        << "bio_var: max_rel_err=" << max_rel_bvar;

    std::printf("[REGISTRY] %s | preprocess/model_gene_var/determinism | 30x200 | "
                "rel_err | mean=%.2e tvar=%.2e bvar=%.2e | 1e-4 | atomicAdd | CYCLE-127 | %s\n",
                __DATE__, max_rel_mean, max_rel_tvar, max_rel_bvar,
                (max_rel_mean < 1e-4f && max_rel_tvar < 1e-4f && max_rel_bvar < 1e-4f)
                    ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Min_Mean_Filter
//
// 50 genes × 300 cells.
//   Genes 0..9:  high-mean (Poisson(5) so mean ≈ 5)
//   Genes 10..49: very low-mean (Bernoulli(p=0.02) → mean ≈ 0.02)
//
// With cfg.min_mean=1.0, only genes 0..9 should appear in hvg_indices.
// =============================================================================
TEST(ModelGeneVarTest, Min_Mean_Filter)
{
    constexpr int M = 50, N = 300;
    constexpr int N_HIGH = 10;

    std::mt19937 rng(0xBEEF1234);
    std::poisson_distribution<int> pd_high(5);
    std::bernoulli_distribution bern(0.02);

    std::vector<float> X(M * N, 0.f);

    for (int i = 0; i < N_HIGH; ++i)
        for (int j = 0; j < N; ++j)
            X[i * N + j] = static_cast<float>(pd_high(rng));

    for (int i = N_HIGH; i < M; ++i)
        for (int j = 0; j < N; ++j)
            X[i * N + j] = bern(rng) ? 1.f : 0.f;

    const auto csc = dense_to_csc(M, N, X);
    auto pz        = upload_csc(csc);

    singlet::gpu::preprocess::ModelGeneVarConfig cfg;
    cfg.n_top    = M;      // request all
    cfg.min_mean = 1.0f;   // filter out low-mean genes

    auto res = singlet::gpu::preprocess::model_gene_var(pz, cfg);

    // n_hvg should be <= N_HIGH (may be exactly N_HIGH if all high genes pass)
    EXPECT_LE(res.n_hvg, N_HIGH + 2)  // allow tiny overshoot from sampling noise
        << "With min_mean=1.0, expected ~" << N_HIGH << " HVGs; got " << res.n_hvg;
    EXPECT_GT(res.n_hvg, 0)
        << "Should have at least 1 HVG after filter";

    // All returned hvg_indices should correspond to high-mean genes (0..N_HIGH-1)
    auto hvg_idx = d2h_i(res.hvg_indices, res.n_hvg);
    auto gpu_mean = d2h_f(res.mean, M);

    int n_low_mean_in_hvg = 0;
    for (int idx : hvg_idx) {
        if (gpu_mean[idx] < 1.0f) n_low_mean_in_hvg++;
    }

    EXPECT_EQ(n_low_mean_in_hvg, 0)
        << n_low_mean_in_hvg << " genes with mean<1.0 leaked into HVG set";

    std::printf("[REGISTRY] %s | preprocess/model_gene_var/min_mean_filter | 50x300 | "
                "filter_recall | n_hvg=%d low_in_hvg=%d | n_low=0 | none | CYCLE-127 | %s\n",
                __DATE__, res.n_hvg, n_low_mean_in_hvg,
                (n_low_mean_in_hvg == 0 && res.n_hvg > 0) ? "PASS" : "FAIL");
}
