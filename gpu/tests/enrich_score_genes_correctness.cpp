// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/tests/enrich_score_genes_correctness.cpp
//
// Correctness harness for scanpy.tl.score_genes GPU port (CYCLE-129).
// Satija R et al. (2015) Nat Biotechnol 33:495-502 / Seurat AddModuleScore.
//
// Tests:
//   1. ScoreGenes_TinyClosedForm          — 10×20×1, hand-traced algorithm, abs_err<1e-4
//   2. ScoreGenes_PlantedSet_HighScore    — 50×200, planted high-expression cells, delta≥1.0
//   3. ScoreGenes_MultipleSets_Independent — 50×100×3 disjoint sets, columns differ
//   4. ScoreGenes_Determinism_SameSeed    — same seed → bit-identical; diff seed → differs
//   5. ScoreGenes_AllOnesInput            — X=1, scores ≈ 0 for all cells
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="ScoreGenesTest.*"

#include <singlet-gpu/enrich/score_genes.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <unordered_set>
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

// CPU reference for score_genes:
// Returns scores[cell * n_sets + s] (row-major cells × sets).
// Uses the same bin-assignment and std::mt19937 RNG as the GPU implementation.
std::vector<float> cpu_score_genes(
    int m, int n,
    const std::vector<float>& X_row_major,        // m × n row-major
    const std::vector<std::vector<int>>& gene_sets,
    const singlet_gpu::enrich::ScoreGenesConfig& cfg)
{
    const int n_sets = static_cast<int>(gene_sets.size());

    // Compute per-gene mean.
    std::vector<float> mu(m, 0.f);
    for (int g = 0; g < m; ++g)
        for (int c = 0; c < n; ++c)
            mu[g] += X_row_major[g * n + c];
    for (int g = 0; g < m; ++g) mu[g] /= static_cast<float>(n);

    // Bin assignment.
    float mu_min = mu[0], mu_max = mu[0];
    for (int g = 1; g < m; ++g) {
        if (mu[g] < mu_min) mu_min = mu[g];
        if (mu[g] > mu_max) mu_max = mu[g];
    }
    const float range = mu_max - mu_min;
    const int   nb    = cfg.n_bins;

    std::vector<int> bin_id(m);
    for (int g = 0; g < m; ++g) {
        if (range <= 0.f) {
            bin_id[g] = 0;
        } else {
            int b = static_cast<int>(std::floor((mu[g] - mu_min) / range * static_cast<float>(nb)));
            if (b < 0)   b = 0;
            if (b >= nb) b = nb - 1;
            bin_id[g] = b;
        }
    }

    std::vector<std::vector<int>> bin_genes(nb);
    for (int g = 0; g < m; ++g) bin_genes[bin_id[g]].push_back(g);

    std::vector<float> scores(n * n_sets, 0.f);

    for (int s = 0; s < n_sets; ++s) {
        const std::vector<int>& S = gene_sets[s];
        const int set_size        = static_cast<int>(S.size());

        std::unordered_set<int> set_mask(S.begin(), S.end());

        std::set<int> set_bins_used;
        for (int g : S) set_bins_used.insert(bin_id[g]);

        std::vector<int> pool;
        for (int b : set_bins_used)
            for (int g : bin_genes[b])
                if (set_mask.find(g) == set_mask.end()) pool.push_back(g);

        int ctrl_size_s = cfg.ctrl_size;
        if (cfg.use_set_size_for_ctrl) ctrl_size_s = std::max(ctrl_size_s, set_size);
        ctrl_size_s = std::max(ctrl_size_s, 1);

        std::mt19937 rng(
            static_cast<uint32_t>((cfg.seed ^ static_cast<uint64_t>(s + 1)) & 0xFFFFFFFF)
            ^ static_cast<uint32_t>((cfg.seed ^ static_cast<uint64_t>(s + 1)) >> 32));

        std::vector<int> controls;
        if (!pool.empty()) {
            const int n_ctrl = std::min(ctrl_size_s, static_cast<int>(pool.size()));
            for (int i = 0; i < n_ctrl; ++i) {
                std::uniform_int_distribution<int> uid(i, static_cast<int>(pool.size()) - 1);
                int j = uid(rng);
                std::swap(pool[i], pool[j]);
            }
            controls.assign(pool.begin(), pool.begin() + n_ctrl);
        }

        // Per-cell score.
        for (int c = 0; c < n; ++c) {
            float set_mean  = 0.f;
            float ctrl_mean = 0.f;
            for (int g : S)        set_mean  += X_row_major[g * n + c];
            if (set_size > 0) set_mean /= static_cast<float>(set_size);
            if (!controls.empty()) {
                for (int g : controls) ctrl_mean += X_row_major[g * n + c];
                ctrl_mean /= static_cast<float>(controls.size());
            }
            // col-major output: scores[s*n + c]
            scores[static_cast<size_t>(s) * n + c] = set_mean - ctrl_mean;
        }
    }
    return scores;  // n_sets × n col-major (matches GPU output layout)
}

} // anonymous namespace


// =============================================================================
// Test 1: ScoreGenes_TinyClosedForm
//
// 10 genes × 20 cells, 1 gene set = {0, 1, 2}, ctrl_size=3, n_bins=2, seed=42.
// Manually trace the full algorithm on CPU; verify GPU output matches to 1e-4.
// =============================================================================
TEST(ScoreGenesTest, ScoreGenes_TinyClosedForm)
{
    constexpr int M = 10, N = 20;

    // Build a small, controlled expression matrix (row-major genes × cells).
    // Genes 0-2: set genes (moderate expression). Genes 3-9: background.
    std::mt19937 rng(0xABCD1234ULL);
    std::uniform_real_distribution<float> u(0.f, 5.f);

    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    std::vector<std::vector<int>> gene_sets = {{0, 1, 2}};

    singlet_gpu::enrich::ScoreGenesConfig cfg;
    cfg.ctrl_size             = 3;
    cfg.n_bins                = 2;
    cfg.seed                  = 42;
    cfg.use_set_size_for_ctrl = true;

    // CPU reference (same algorithm).
    auto ref = cpu_score_genes(M, N, X_h, gene_sets, cfg);

    // GPU path.
    auto csc = dense_to_csc(M, N, X_h);
    auto pz  = upload_csc(csc);

    auto res = singlet_gpu::enrich::score_genes(pz, gene_sets, cfg);
    auto gpu = d2h(res.scores, N * 1);

    ASSERT_EQ(res.n_cells, N);
    ASSERT_EQ(res.n_sets,  1);

    float max_err = 0.f;
    for (int c = 0; c < N; ++c) {
        // GPU: col-major scores[set=0, cell=c] = scores[0*N + c]
        float err = std::fabs(gpu[c] - ref[c]);
        max_err   = std::max(max_err, err);
        EXPECT_NEAR(gpu[c], ref[c], 1e-4f)
            << "cell=" << c << ": gpu=" << gpu[c] << " ref=" << ref[c];
    }

    std::printf("[REGISTRY] %s | enrich/score_genes/tiny_cf | 10x20x1 | "
                "abs_err | max=%.2e | 1e-4 | none | CYCLE-129 | %s\n",
                __DATE__, max_err, (max_err < 1e-4f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: ScoreGenes_PlantedSet_HighScore
//
// 50 genes × 200 cells. Set 0 = genes {0, 1, 2, 3, 4}.
// Cells 0..99: planted to express set genes at 5× background (Poisson(5) vs Poisson(1)).
// After scoring: mean(score[cells 0..99]) > mean(score[cells 100..199]) by ≥ 1.0.
// =============================================================================
TEST(ScoreGenesTest, ScoreGenes_PlantedSet_HighScore)
{
    constexpr int M = 50, N = 200;

    std::mt19937 rng(0xBEEF5555ULL);
    std::poisson_distribution<int> bg(1);     // background: Poisson(1)
    std::poisson_distribution<int> signal(5); // planted signal: Poisson(5)

    std::vector<float> X_h(M * N, 0.f);
    for (int g = 0; g < M; ++g) {
        for (int c = 0; c < N; ++c) {
            bool is_set_gene    = (g < 5);
            bool is_signal_cell = (c < 100);
            if (is_set_gene && is_signal_cell) {
                X_h[g * N + c] = static_cast<float>(signal(rng));
            } else {
                X_h[g * N + c] = static_cast<float>(bg(rng));
            }
        }
    }

    std::vector<std::vector<int>> gene_sets = {{0, 1, 2, 3, 4}};

    singlet_gpu::enrich::ScoreGenesConfig cfg;
    cfg.ctrl_size             = 10;
    cfg.n_bins                = 5;
    cfg.seed                  = 0;
    cfg.use_set_size_for_ctrl = true;

    auto csc = dense_to_csc(M, N, X_h);
    auto pz  = upload_csc(csc);

    auto res = singlet_gpu::enrich::score_genes(pz, gene_sets, cfg);
    auto gpu = d2h(res.scores, N * 1);

    ASSERT_EQ(res.n_cells, N);
    ASSERT_EQ(res.n_sets,  1);

    // Compute mean score for each population.
    float mean_signal = 0.f, mean_bg_pop = 0.f;
    for (int c = 0;   c < 100; ++c) mean_signal += gpu[c];
    for (int c = 100; c < 200; ++c) mean_bg_pop  += gpu[c];
    mean_signal /= 100.f;
    mean_bg_pop  /= 100.f;

    const float delta = mean_signal - mean_bg_pop;

    EXPECT_GT(delta, 1.0f)
        << "Planted signal: mean_signal=" << mean_signal
        << " mean_bg=" << mean_bg_pop
        << " delta=" << delta << " (expected > 1.0)";

    std::printf("[REGISTRY] %s | enrich/score_genes/planted_set | 50x200x1 | "
                "delta_means | %.3f | >=1.0 | Poisson5vs1 | CYCLE-129 | %s\n",
                __DATE__, delta, (delta > 1.0f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: ScoreGenes_MultipleSets_Independent
//
// 50 genes × 100 cells, 3 disjoint gene sets (genes 0-4, 10-14, 20-24).
// Verify the 3 output columns are different (no aliasing in W construction).
// =============================================================================
TEST(ScoreGenesTest, ScoreGenes_MultipleSets_Independent)
{
    constexpr int M = 50, N = 100;

    std::mt19937 rng(0xCAFEF00DULL);
    std::uniform_real_distribution<float> u(0.f, 5.f);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = u(rng);

    std::vector<std::vector<int>> gene_sets = {
        {0, 1, 2, 3, 4},
        {10, 11, 12, 13, 14},
        {20, 21, 22, 23, 24}
    };

    singlet_gpu::enrich::ScoreGenesConfig cfg;
    cfg.ctrl_size             = 10;
    cfg.n_bins                = 5;
    cfg.seed                  = 123;
    cfg.use_set_size_for_ctrl = true;

    auto csc = dense_to_csc(M, N, X_h);
    auto pz  = upload_csc(csc);

    auto res = singlet_gpu::enrich::score_genes(pz, gene_sets, cfg);
    auto gpu = d2h(res.scores, N * 3);

    ASSERT_EQ(res.n_cells, N);
    ASSERT_EQ(res.n_sets,  3);

    // Verify the 3 score columns are not all identical (no aliasing).
    // Compare column 0 vs column 1 (scores[0..N-1] vs scores[N..2N-1]).
    float diff_01 = 0.f, diff_02 = 0.f;
    for (int c = 0; c < N; ++c) {
        diff_01 += std::fabs(gpu[c] - gpu[N + c]);
        diff_02 += std::fabs(gpu[c] - gpu[2 * N + c]);
    }

    EXPECT_GT(diff_01, 1.0f)
        << "Set 0 and Set 1 scores are suspiciously identical (sum_diff=" << diff_01 << ")";
    EXPECT_GT(diff_02, 1.0f)
        << "Set 0 and Set 2 scores are suspiciously identical (sum_diff=" << diff_02 << ")";

    // CPU cross-check to confirm GPU shapes are correct.
    auto ref = cpu_score_genes(M, N, X_h, gene_sets, cfg);
    float max_err = 0.f;
    for (int i = 0; i < N * 3; ++i)
        max_err = std::max(max_err, std::fabs(gpu[i] - ref[i]));

    EXPECT_LT(max_err, 1e-3f)
        << "GPU vs CPU max_err=" << max_err << " exceeds 1e-3";

    std::printf("[REGISTRY] %s | enrich/score_genes/multi_sets | 50x100x3 | "
                "independence+abs_err | diff01=%.2f diff02=%.2f max_err=%.2e | "
                "none | none | CYCLE-129 | %s\n",
                __DATE__, diff_01, diff_02, max_err,
                (diff_01 > 1.f && diff_02 > 1.f && max_err < 1e-3f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: ScoreGenes_Determinism_SameSeed
//
// Two runs with the same seed must produce bit-identical scores (rel_err = 0).
// Two runs with different seeds must produce different scores (sanity: diff > 0).
// =============================================================================
TEST(ScoreGenesTest, ScoreGenes_Determinism_SameSeed)
{
    constexpr int M = 30, N = 50;

    std::mt19937 rng(0xDEADBEEFULL);
    std::poisson_distribution<int> pd(2);
    std::vector<float> X_h(M * N);
    for (auto& v : X_h) v = static_cast<float>(pd(rng));

    std::vector<std::vector<int>> gene_sets = {{0, 1, 2, 3}, {5, 6, 7}};

    auto csc = dense_to_csc(M, N, X_h);

    singlet_gpu::enrich::ScoreGenesConfig cfg_a;
    cfg_a.ctrl_size = 5;
    cfg_a.n_bins    = 5;
    cfg_a.seed      = 7777;

    singlet_gpu::enrich::ScoreGenesConfig cfg_b = cfg_a;  // same seed
    singlet_gpu::enrich::ScoreGenesConfig cfg_c = cfg_a;
    cfg_c.seed = 8888;  // different seed

    // Run 1: seed=7777
    auto pz1  = upload_csc(csc);
    auto res1 = singlet_gpu::enrich::score_genes(pz1, gene_sets, cfg_a);
    auto h1   = d2h(res1.scores, N * 2);

    // Run 2: seed=7777 (same → should be bit-identical)
    auto pz2  = upload_csc(csc);
    auto res2 = singlet_gpu::enrich::score_genes(pz2, gene_sets, cfg_b);
    auto h2   = d2h(res2.scores, N * 2);

    // Run 3: seed=8888 (different → should differ in at least some elements)
    auto pz3  = upload_csc(csc);
    auto res3 = singlet_gpu::enrich::score_genes(pz3, gene_sets, cfg_c);
    auto h3   = d2h(res3.scores, N * 2);

    // Same seed: bit-identical (rel_err = 0).
    float max_rel_same = 0.f;
    for (int i = 0; i < N * 2; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel_same = std::max(max_rel_same, rel);
    }
    EXPECT_EQ(max_rel_same, 0.f)
        << "Same seed: expected bit-identical scores but max_rel_err=" << max_rel_same;

    // Different seed: at least some scores differ.
    float max_abs_diff_seed = 0.f;
    for (int i = 0; i < N * 2; ++i)
        max_abs_diff_seed = std::max(max_abs_diff_seed, std::fabs(h1[i] - h3[i]));
    EXPECT_GT(max_abs_diff_seed, 0.f)
        << "Different seeds produced identical scores (unexpected for random control sampling)";

    std::printf("[REGISTRY] %s | enrich/score_genes/determinism | 30x50x2 | "
                "rel_err+diff | same=%.2e diff=%.2e | 0 / >0 | none | CYCLE-129 | %s\n",
                __DATE__, max_rel_same, max_abs_diff_seed,
                (max_rel_same == 0.f && max_abs_diff_seed > 0.f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: ScoreGenes_AllOnesInput
//
// X = all ones. mean_g = 1.0 for all genes → all genes land in the same bin.
// Control genes are sampled from that same bin (excluding set genes) and also
// have mean = 1.0. Per cell: mean(set) = 1.0, mean(ctrl) = 1.0 → score ≈ 0.
// =============================================================================
TEST(ScoreGenesTest, ScoreGenes_AllOnesInput)
{
    constexpr int M = 20, N = 40;

    // X = dense all-ones matrix.
    std::vector<float> X_h(M * N, 1.f);

    std::vector<std::vector<int>> gene_sets = {{0, 1, 2, 3}};

    singlet_gpu::enrich::ScoreGenesConfig cfg;
    cfg.ctrl_size             = 4;
    cfg.n_bins                = 5;
    cfg.seed                  = 0;
    cfg.use_set_size_for_ctrl = true;

    auto csc = dense_to_csc(M, N, X_h);
    auto pz  = upload_csc(csc);

    auto res = singlet_gpu::enrich::score_genes(pz, gene_sets, cfg);
    auto gpu = d2h(res.scores, N * 1);

    ASSERT_EQ(res.n_cells, N);
    ASSERT_EQ(res.n_sets,  1);

    float max_abs = 0.f;
    for (int c = 0; c < N; ++c) {
        EXPECT_TRUE(std::isfinite(gpu[c])) << "cell=" << c << " is non-finite: " << gpu[c];
        // mean(set)=1, mean(ctrl)=1 → score=0. Allow small fp32 SpMM rounding.
        EXPECT_NEAR(gpu[c], 0.f, 1e-5f) << "cell=" << c << ": score=" << gpu[c];
        max_abs = std::max(max_abs, std::fabs(gpu[c]));
    }

    std::printf("[REGISTRY] %s | enrich/score_genes/all_ones | 20x40x1 | "
                "abs_score | max=%.2e | 1e-5 | none | CYCLE-129 | %s\n",
                __DATE__, max_abs, (max_abs < 1e-5f) ? "PASS" : "FAIL");
}
