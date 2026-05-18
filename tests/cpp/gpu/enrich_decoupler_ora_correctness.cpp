// SPDX-License-Identifier: MIT
// singlet/gpu/tests/enrich_decoupler_ora_correctness.cpp
//
// Correctness harness for decoupleR ORA over-representation analysis (CYCLE-132).
//
// Tests:
//  1. Ora_TinyClosedForm        — 20×5×1 set, hand-computed hypergeometric, abs_err<0.05
//  2. Ora_NoOverlap_HighPvalue  — 50×100×1 set, no overlap → score≈0
//  3. Ora_PerfectOverlap_LowPvalue — 50×100×1 set, full overlap → score>5
//  4. Ora_MultipleSets          — 50×100×5 disjoint sets, independent enrichment
//  5. Ora_Determinism           — two runs: rel_err=0 (no atomics in output pass)
//
// Build: compiled as CUDA (see CMakeLists.txt — LANGUAGE CUDA).
// Run:   --gtest_filter="OraTest.*"

#include <singlet/gpu/enrich/decoupler_ora.h>
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

// Build CSC from a dense row-major (genes × cells) matrix.
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

// ---------------------------------------------------------------------------
// cpu_ora — CPU reference hypergeometric ORA.
//
// X: m×n row-major (genes×cells). gene_sets: list of gene index sets.
// top_k: number of top genes per cell (by value). Returns n×n_sets col-major.
//
// score[c, s] = -log10(p_hypergeom(hits, K=|set|, N=m, n=top_k))
//
// Reference formula (one-tailed, ≥ k successes):
//   p = Σ_{i=k}^{min(K,n)} C(K,i)*C(N-K,n-i) / C(N,n)
//   via lgamma closed form.
// ---------------------------------------------------------------------------

// Log of C(n, k) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1).
// Returns -Inf for invalid k.
static double log_binom(int n, int k)
{
    if (k < 0 || k > n) return -1e300;
    if (k == 0 || k == n) return 0.0;
    return std::lgamma(n + 1.0) - std::lgamma(k + 1.0) - std::lgamma(n - k + 1.0);
}

static float cpu_hypergeo_score(int k, int K, int N, int nk)
{
    if (k <= 0) return 0.f;
    if (K <= 0 || nk <= 0) return 0.f;
    const int i_max = std::min(K, nk);
    if (k > i_max) return 0.f;

    const double log_C_Nn = log_binom(N, nk);
    double log_max   = -1e300;
    double log_sum   = 0.0;
    bool   first     = true;

    for (int i = k; i <= i_max; ++i) {
        int ni = nk - i;
        if (ni < 0 || ni > N - K) continue;
        double lt = log_binom(K, i) + log_binom(N - K, ni) - log_C_Nn;
        if (first) {
            log_max = lt; log_sum = 1.0; first = false;
        } else if (lt > log_max) {
            log_sum = log_sum * std::exp(log_max - lt) + 1.0;
            log_max = lt;
        } else {
            log_sum += std::exp(lt - log_max);
        }
    }

    if (first) return 0.f;
    double log_p = log_max + std::log(log_sum);
    double score = -log_p / std::log(10.0);
    return static_cast<float>(std::min(score, 300.0));
}

std::vector<float> cpu_ora(
    int m, int n,
    const std::vector<float>& X,          // row-major m×n: X[g*n+c]
    const std::vector<std::vector<int>>& gene_sets,
    int top_k)
{
    const int n_sets = static_cast<int>(gene_sets.size());
    std::vector<float> scores(n * n_sets, 0.f);

    for (int c = 0; c < n; ++c) {
        // Extract column c and find top_k indices.
        std::vector<std::pair<float, int>> col(m);
        for (int g = 0; g < m; ++g) col[g] = { X[g * n + c], g };
        std::partial_sort(col.begin(), col.begin() + std::min(top_k, m), col.end(),
            [](const auto& a, const auto& b){ return a.first > b.first; });

        // Build top-K set.
        std::unordered_set<int> topk_set;
        for (int i = 0; i < std::min(top_k, m); ++i)
            if (col[i].first > 0.f) topk_set.insert(col[i].second);

        for (int s = 0; s < n_sets; ++s) {
            const auto& S = gene_sets[s];
            int hits = 0;
            for (int g : S) if (topk_set.count(g)) ++hits;
            const int K = static_cast<int>(S.size());
            scores[s * n + c] = cpu_hypergeo_score(hits, K, m, static_cast<int>(topk_set.size()));
        }
    }
    return scores;
}

} // anonymous namespace


// =============================================================================
// ORA Tests
// =============================================================================

// =============================================================================
// Test 1: Ora_TinyClosedForm
//
// 20 genes × 5 cells × 1 gene set (genes 0..4, size 5).
// cfg.top_k = 4 (explicit).
//
// For cell 0: X = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, ...0]  (genes 0..9 active)
//   top-4 = genes 0,1,2,3  (values 10,9,8,7).  Gene set = {0,1,2,3,4}.
//   hits = 4, K=5, N=20, n=4.
//
// Hypergeometric p = Σ_{i=4}^{min(5,4)=4} C(5,i)*C(15,4-i) / C(20,4)
//   = C(5,4)*C(15,0) / C(20,4) = 5*1 / 4845 ≈ 0.001032
//   score = -log10(0.001032) ≈ 2.986.
//
// Verified with scipy.stats.hypergeom.sf(k-1, N, K, n):
//   from scipy.stats import hypergeom
//   hypergeom.sf(3, 20, 5, 4) = 0.001032  → -log10 ≈ 2.986
//   abs_err < 0.05 (lgamma fp32 accuracy).
//
// For cells 1..4: X = uniform low values; top-4 may partially overlap set.
// We verify cell 0's score against the closed-form reference.
// =============================================================================
TEST(OraTest, Ora_TinyClosedForm)
{
    constexpr int M = 20, N_CELLS = 5;
    // Row-major: X[g * N_CELLS + c]
    std::vector<float> X_h(M * N_CELLS, 0.f);
    // Cell 0: genes 0..9 with descending values 10..1.
    for (int g = 0; g < 10; ++g) {
        X_h[g * N_CELLS + 0] = static_cast<float>(10 - g);
    }
    // Cells 1..4: low constant values so top-4 are mostly within genes 10..19.
    for (int c = 1; c < N_CELLS; ++c) {
        for (int g = 10; g < M; ++g) {
            X_h[g * N_CELLS + c] = static_cast<float>(g - 9);  // 1..10, outside set {0..4}
        }
    }

    std::vector<std::vector<int>> sets = { {0, 1, 2, 3, 4} };  // gene set: genes 0..4

    auto csc = dense_to_csc(M, N_CELLS, X_h);
    auto pz  = upload_csc(csc);

    singlet::gpu::enrich::OraConfig cfg;
    cfg.top_k = 4;
    auto res = singlet::gpu::enrich::ora(pz, sets, cfg);
    auto gpu = d2h(res.scores, N_CELLS * 1);

    ASSERT_EQ(res.n_cells, N_CELLS);
    ASSERT_EQ(res.n_sets,  1);

    // Cell 0: expected score ≈ 2.986 (scipy reference: -log10(0.001032)).
    const float expected_c0 = 2.986f;
    const float err_c0 = std::fabs(gpu[0] - expected_c0);  // scores[set=0, cell=0]
    EXPECT_LT(err_c0, 0.05f)
        << "Cell 0: gpu=" << gpu[0] << " expected≈" << expected_c0
        << " err=" << err_c0;

    // All scores finite and non-negative.
    for (int c = 0; c < N_CELLS; ++c) {
        EXPECT_TRUE(std::isfinite(gpu[c]))
            << "cell " << c << " non-finite: " << gpu[c];
        EXPECT_GE(gpu[c], 0.f)
            << "cell " << c << " negative score: " << gpu[c];
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ora/tiny_cf | 20x5x1 | "
                "abs_err | c0_err=%.3f | 0.05 | none | CYCLE-132 | %s\n",
                __DATE__, err_c0, (err_c0 < 0.05f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 2: Ora_NoOverlap_HighPvalue
//
// 50 genes × 100 cells × 1 gene set {40, 41, 42, 43, 44}.
// X is built so genes 40..44 are NEVER in any cell's top-10:
//   - Genes 0..9 have high constant expression (value = 10).
//   - Genes 10..39 have mid expression (value = 5).
//   - Genes 40..44 have zero expression.
//   - Genes 45..49 have very low expression (value = 1).
// top_k = 10: top-10 will be genes 0..9 for every cell.
// Expected: hits = 0 for all cells, p = 1, score = 0.
// =============================================================================
TEST(OraTest, Ora_NoOverlap_HighPvalue)
{
    constexpr int M = 50, N_CELLS = 100;
    std::vector<float> X_h(M * N_CELLS, 0.f);

    for (int c = 0; c < N_CELLS; ++c) {
        for (int g = 0;  g < 10; ++g) X_h[g * N_CELLS + c] = 10.f;
        for (int g = 10; g < 40; ++g) X_h[g * N_CELLS + c] =  5.f;
        // genes 40..44: zero (implicit)
        for (int g = 45; g < M;  ++g) X_h[g * N_CELLS + c] =  1.f;
    }

    std::vector<std::vector<int>> sets = { {40, 41, 42, 43, 44} };

    auto csc = dense_to_csc(M, N_CELLS, X_h);
    auto pz  = upload_csc(csc);

    singlet::gpu::enrich::OraConfig cfg;
    cfg.top_k = 10;
    auto res = singlet::gpu::enrich::ora(pz, sets, cfg);
    auto gpu = d2h(res.scores, N_CELLS * 1);

    ASSERT_EQ(res.n_cells, N_CELLS);
    ASSERT_EQ(res.n_sets,  1);

    // All scores should be 0 (hits = 0 → p = 1 → -log10(1) = 0).
    float max_score = 0.f;
    for (int c = 0; c < N_CELLS; ++c) {
        EXPECT_TRUE(std::isfinite(gpu[c]))
            << "cell " << c << " non-finite";
        EXPECT_NEAR(gpu[c], 0.f, 0.01f)
            << "cell " << c << " expected 0 (no overlap), got " << gpu[c];
        max_score = std::max(max_score, gpu[c]);
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ora/no_overlap | 50x100x1 | "
                "max_score | max=%.3f | ~0 | none | CYCLE-132 | %s\n",
                __DATE__, max_score, (max_score < 0.1f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 3: Ora_PerfectOverlap_LowPvalue
//
// 50 genes × 100 cells × 1 gene set {0, 1, 2, 3, 4}.
// X built so genes 0..4 are ALWAYS the top-5 in every cell:
//   - Genes 0..4: expression = 100 (very high).
//   - All other genes: expression = 1 (low).
// top_k = 5: top-5 = genes 0..4 exactly.
// Expected: hits = 5 (full overlap), K=5, N=50, n=5.
//
// p = C(5,5)*C(45,0) / C(50,5) = 1 / 2118760 ≈ 4.72e-7
// score = -log10(4.72e-7) ≈ 6.33   (well above threshold of 5).
// =============================================================================
TEST(OraTest, Ora_PerfectOverlap_LowPvalue)
{
    constexpr int M = 50, N_CELLS = 100;
    std::vector<float> X_h(M * N_CELLS, 0.f);

    for (int c = 0; c < N_CELLS; ++c) {
        for (int g = 0; g < 5;  ++g) X_h[g * N_CELLS + c] = 100.f;
        for (int g = 5; g < M;  ++g) X_h[g * N_CELLS + c] =   1.f;
    }

    std::vector<std::vector<int>> sets = { {0, 1, 2, 3, 4} };

    auto csc = dense_to_csc(M, N_CELLS, X_h);
    auto pz  = upload_csc(csc);

    singlet::gpu::enrich::OraConfig cfg;
    cfg.top_k = 5;
    auto res = singlet::gpu::enrich::ora(pz, sets, cfg);
    auto gpu = d2h(res.scores, N_CELLS * 1);

    ASSERT_EQ(res.n_cells, N_CELLS);
    ASSERT_EQ(res.n_sets,  1);

    // Expected score ≈ 6.33; all cells identical construction → all scores high.
    float min_score = gpu[0];
    for (int c = 0; c < N_CELLS; ++c) {
        EXPECT_TRUE(std::isfinite(gpu[c]))
            << "cell " << c << " non-finite";
        EXPECT_GT(gpu[c], 5.f)
            << "cell " << c << " expected score > 5 (perfect overlap), got " << gpu[c];
        min_score = std::min(min_score, gpu[c]);
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ora/perfect_overlap | 50x100x1 | "
                "min_score | min=%.3f | >5 | none | CYCLE-132 | %s\n",
                __DATE__, min_score, (min_score > 5.f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 4: Ora_MultipleSets
//
// 50 genes × 100 cells × 5 disjoint gene sets of size 5 each.
//   set 0: genes {0..4},   set 1: {5..9},  set 2: {10..14},
//   set 3: {15..19},       set 4: {20..24}.
//
// X is built so that:
//   - Cells 0..19:   top-5 = genes {0..4}   → set 0 fully enriched
//   - Cells 20..39:  top-5 = genes {5..9}   → set 1 fully enriched
//   - Cells 40..59:  top-5 = genes {10..14} → set 2 fully enriched
//   - Cells 60..79:  top-5 = genes {15..19} → set 3 fully enriched
//   - Cells 80..99:  top-5 = genes {20..24} → set 4 fully enriched
//
// Verify: for cells 0..19, set 0 has score > 5; sets 1..4 have score = 0.
// =============================================================================
TEST(OraTest, Ora_MultipleSets)
{
    constexpr int M = 50, N_CELLS = 100;
    std::vector<float> X_h(M * N_CELLS, 0.f);

    for (int c = 0; c < N_CELLS; ++c) {
        // Determine which gene block is "hot" for this cell.
        int block = c / 20;  // 0..4, each 20 cells
        for (int g = 0; g < M; ++g) {
            int g_block = g / 5;
            if (g_block == block && g < 25) {
                X_h[g * N_CELLS + c] = 100.f;  // hot block
            } else {
                X_h[g * N_CELLS + c] = 1.f;    // cold
            }
        }
    }

    std::vector<std::vector<int>> sets = {
        {0, 1, 2, 3, 4},
        {5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
        {15, 16, 17, 18, 19},
        {20, 21, 22, 23, 24},
    };

    auto csc = dense_to_csc(M, N_CELLS, X_h);
    auto pz  = upload_csc(csc);

    singlet::gpu::enrich::OraConfig cfg;
    cfg.top_k = 5;
    auto res = singlet::gpu::enrich::ora(pz, sets, cfg);
    auto gpu = d2h(res.scores, N_CELLS * 5);

    ASSERT_EQ(res.n_cells, N_CELLS);
    ASSERT_EQ(res.n_sets,  5);

    // Verify each cell group: enriched set should have score > 5;
    // non-enriched sets should have score ≈ 0.
    float min_enriched = 1e9f, max_non_enriched = 0.f;
    for (int block = 0; block < 5; ++block) {
        for (int c = block * 20; c < (block + 1) * 20; ++c) {
            for (int s = 0; s < 5; ++s) {
                float sc = gpu[s * N_CELLS + c];
                EXPECT_TRUE(std::isfinite(sc))
                    << "cell " << c << " set " << s << " non-finite";
                if (s == block) {
                    min_enriched = std::min(min_enriched, sc);
                } else {
                    max_non_enriched = std::max(max_non_enriched, sc);
                }
            }
        }
    }

    EXPECT_GT(min_enriched, 5.f)
        << "min enriched score=" << min_enriched << " expected > 5";
    EXPECT_LT(max_non_enriched, 0.1f)
        << "max non-enriched score=" << max_non_enriched << " expected ≈ 0";

    std::printf("[REGISTRY] %s | enrich/decoupler_ora/multi_sets | 50x100x5 | "
                "enriched>5,non_enriched<0.1 | min_enriched=%.3f max_non_enriched=%.3f | "
                "none | CYCLE-132 | %s\n",
                __DATE__, min_enriched, max_non_enriched,
                (min_enriched > 5.f && max_non_enriched < 0.1f) ? "PASS" : "FAIL");
}


// =============================================================================
// Test 5: Ora_Determinism
//
// Two consecutive ORA calls on the same data must produce bit-exact scores.
// ORA v0 has no atomics in the scoring path (Pass 4 is one thread per element,
// Pass 3 is cuBLAS Sgemm — deterministic on fixed arch for fixed input).
// Expected: rel_err = 0 for all output elements.
// =============================================================================
TEST(OraTest, Ora_Determinism)
{
    constexpr int M = 50, N_CELLS = 100;

    std::mt19937 rng(0xDEC0DEULL);
    std::uniform_real_distribution<float> ux(0.f, 10.f);
    std::bernoulli_distribution sparse(0.5);

    std::vector<float> X_h(M * N_CELLS, 0.f);
    for (int i = 0; i < M * N_CELLS; ++i)
        if (!sparse(rng)) X_h[i] = ux(rng);

    std::vector<std::vector<int>> sets = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {10, 11, 12, 13, 14},
        {20, 25, 30, 35, 40},
    };

    auto csc = dense_to_csc(M, N_CELLS, X_h);

    singlet::gpu::enrich::OraConfig cfg;
    cfg.top_k = 10;

    auto pz1  = upload_csc(csc);
    auto res1 = singlet::gpu::enrich::ora(pz1, sets, cfg);
    auto h1   = d2h(res1.scores, N_CELLS * 3);

    auto pz2  = upload_csc(csc);
    auto res2 = singlet::gpu::enrich::ora(pz2, sets, cfg);
    auto h2   = d2h(res2.scores, N_CELLS * 3);

    float max_rel = 0.f;
    for (int i = 0; i < N_CELLS * 3; ++i) {
        float denom = std::max(std::fabs(h1[i]), 1e-8f);
        float rel   = std::fabs(h1[i] - h2[i]) / denom;
        max_rel     = std::max(max_rel, rel);
        EXPECT_EQ(h1[i], h2[i])
            << "element " << i << " not bit-exact: run1=" << h1[i]
            << " run2=" << h2[i] << " rel=" << rel;
    }

    std::printf("[REGISTRY] %s | enrich/decoupler_ora/determinism | 50x100x3 | "
                "rel_err | max=%.2e | 0 | none | CYCLE-132 | %s\n",
                __DATE__, max_rel, (max_rel == 0.f) ? "PASS" : "FAIL");
}
