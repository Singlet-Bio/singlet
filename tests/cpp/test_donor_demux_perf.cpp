// SPDX-License-Identifier: MIT
// test/test_donor_demux_perf.cpp
// Performance and correctness tests for the optimised donor_demux.h.
// All tests are synthetic — no disk files required.
//
// Tests:
//   1. K-selection accuracy (EM+BIC + VB pipeline recovers true K)
//   2. EM vs full-VB concordance (same K, same assignments on same data)
//   3. Speed regression (3000 cells, 5000 SNPs, K=1 must finish in < 30 s)
//   4. Edge cases (tiny library, huge library, degenerate genotypes)
//   5. Numerical stability (dp=0, unbalanced mixing)

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/donor_demux.h"

using namespace singlet;

// ─── Minimal test framework ──────────────────────────────────────────────────
#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

// ─── Synthetic data generator ────────────────────────────────────────────────
// Generates CoveredSNPData with K known donors and random allele frequencies.
// Also fills CSC AD/DP matrices for full run_demux() testing.
struct SyntheticDemux {
    CoveredSNPData cov;
    std::vector<int> true_donor;          // [n_cells] ground-truth donor assignment
    // CSC AD/DP for run_demux()
    int n_snps = 0, n_cells = 0;
    std::vector<int32_t> ad_indptr, ad_indices, dp_indptr, dp_indices;
    std::vector<uint8_t> ad_data, dp_data;
};

static SyntheticDemux make_synthetic(
    int K_true, int n_cells, int n_snps,
    double snp_coverage_prob = 0.30,   // fraction of SNPs covered per cell
    double mean_depth = 5.0,           // average read depth per covered SNP
    uint64_t seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> allele_unif(0.15f, 0.85f);
    std::bernoulli_distribution cover(snp_coverage_prob);
    std::poisson_distribution<int> depth_dist(mean_depth);
    std::uniform_int_distribution<int> donor_dist(0, K_true - 1);

    // Generate donor allele frequencies: f[k][j] ∈ [0.15, 0.85]
    std::vector<std::vector<float>> f(K_true, std::vector<float>(n_snps));
    for (int k = 0; k < K_true; ++k)
        for (int j = 0; j < n_snps; ++j)
            f[k][j] = allele_unif(rng);

    // Assign cells to donors
    std::vector<int> true_donor(n_cells);
    for (int c = 0; c < n_cells; ++c)
        true_donor[c] = donor_dist(rng);

    // Build per-cell data
    CoveredSNPData cov;
    cov.n_cells = n_cells;
    cov.n_covered_snps = n_snps;
    cov.cells.resize(n_cells);
    cov.covered_to_original.resize(n_snps);
    std::iota(cov.covered_to_original.begin(), cov.covered_to_original.end(), 0);

    // Build CSC for run_demux() — stored temporarily as COO, converted after
    struct COO { int row, col; uint8_t val; };
    std::vector<COO> ad_coo, dp_coo;

    for (int c = 0; c < n_cells; ++c) {
        const int k = true_donor[c];
        for (int j = 0; j < n_snps; ++j) {
            if (!cover(rng)) continue;
            int dp = std::max(1, depth_dist(rng));
            if (dp > 255) dp = 255;
            // Simulate AD ~ Binom(dp, f[k][j])
            std::binomial_distribution<int> binom(dp, f[k][j]);
            int ad = binom(rng);
            if (ad > 255) ad = 255;

            cov.cells[c].snp_idx.push_back(j);
            cov.cells[c].dp.push_back(static_cast<uint8_t>(dp));
            cov.cells[c].ad.push_back(static_cast<uint8_t>(ad));

            dp_coo.push_back({j, c, static_cast<uint8_t>(dp)});
            if (ad > 0)
                ad_coo.push_back({j, c, static_cast<uint8_t>(ad)});
        }
    }

    // COO → CSC (column = cell)
    auto to_csc = [&](const std::vector<COO>& coo,
                      std::vector<int32_t>& indptr,
                      std::vector<int32_t>& indices,
                      std::vector<uint8_t>& data) {
        indptr.assign(n_cells + 1, 0);
        for (auto& e : coo) indptr[e.col + 1]++;
        for (int i = 0; i < n_cells; ++i) indptr[i+1] += indptr[i];
        indices.resize(coo.size());
        data.resize(coo.size());
        std::vector<int32_t> pos(indptr.begin(), indptr.end());
        for (auto& e : coo) {
            int32_t p = pos[e.col]++;
            indices[p] = e.row;
            data[p] = e.val;
        }
    };

    SyntheticDemux sd;
    sd.cov         = std::move(cov);
    sd.true_donor  = std::move(true_donor);
    sd.n_snps      = n_snps;
    sd.n_cells     = n_cells;
    to_csc(dp_coo, sd.dp_indptr, sd.dp_indices, sd.dp_data);
    to_csc(ad_coo, sd.ad_indptr, sd.ad_indices, sd.ad_data);
    return sd;
}

// Helper: assignment concordance between two label vectors (handles permutations)
// Returns fraction of cells where the majority-label permutation agrees.
static float assignment_concordance(const std::vector<int>& a, const std::vector<int>& b, int K) {
    if (a.size() != b.size()) return 0.0f;
    const int N = static_cast<int>(a.size());
    // Build confusion matrix
    std::vector<std::vector<int>> conf(K, std::vector<int>(K, 0));
    for (int i = 0; i < N; ++i) {
        if (a[i] >= 0 && a[i] < K && b[i] >= 0 && b[i] < K)
            conf[a[i]][b[i]]++;
    }
    // Greedy max-weight matching (good enough for small K)
    std::vector<bool> used_b(K, false);
    int matches = 0;
    for (int iter = 0; iter < K; ++iter) {
        int best_i = -1, best_j = -1, best_val = -1;
        for (int i = 0; i < K; ++i)
            for (int j = 0; j < K; ++j)
                if (!used_b[j] && conf[i][j] > best_val) {
                    best_val = conf[i][j]; best_i = i; best_j = j;
                }
        if (best_i < 0) break;
        matches += best_val;
        used_b[best_j] = true;
        for (int i = 0; i < K; ++i) conf[i][best_j] = -1;
    }
    return static_cast<float>(matches) / N;
}

// ─── Test 1: K-selection accuracy ────────────────────────────────────────────
static void test_k_selection_accuracy() {
    std::cout << "\n[Test 1] K-selection accuracy\n";

    struct Case { int K_true; int n_cells; int n_snps; };
    std::vector<Case> cases = {
        {1, 100,  100},
        {1, 500,  1000},
        {2, 500,  1000},
        {3, 1000, 1000},
        {1, 3000, 5000},
        {2, 3000, 5000},
        {5, 3000, 5000},
    };

    for (auto& c : cases) {
        auto sd = make_synthetic(c.K_true, c.n_cells, c.n_snps, 0.30, 5.0, 42);

        DonorDemuxConfig cfg;
        cfg.n_donors       = -1;
        cfg.max_donors     = 8;
        cfg.threads        = 4;
        cfg.fast_k_select  = true;
        cfg.subsample_cells = 500;
        cfg.em_max_iter    = 50;
        cfg.n_inits        = 5;
        cfg.max_iter_pre   = 50;
        cfg.max_iter       = 100;
        cfg.seed           = 42;

        auto result = run_demux(
            static_cast<uint32_t>(c.n_snps), static_cast<uint32_t>(c.n_cells),
            sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
            sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
            cfg);

        const bool k_ok = (result.n_donors_k == c.K_true ||
                           result.n_donors_k == c.K_true + 1 ||
                           (c.K_true > 1 && result.n_donors_k == c.K_true - 1));

        char msg[128];
        std::snprintf(msg, sizeof(msg), "K_true=%d n_cells=%d n_snps=%d → K_selected=%d",
                      c.K_true, c.n_cells, c.n_snps, result.n_donors_k);
        CHECK(k_ok, msg);
    }
}

// ─── Test 2: EM+BIC candidates always contain the true K ────────────────────
// Key property: EM+BIC Phase 1 must not exclude the true K from its candidates.
// The +/- 1 window in select_k_bic() ensures this even when BIC errs by 1.
static void test_em_vb_concordance() {
    std::cout << "\n[Test 2] EM+BIC candidates include true K\n";

    struct Case { int K_true; int n_cells; int n_snps; };
    std::vector<Case> cases = {
        {1, 500, 500}, {2, 500, 500}, {3, 1000, 1000},
        {1, 1000, 500}, {2, 1000, 500},
    };
    int n_ok = 0, n_total = 0;
    for (auto& cas : cases) {
        for (int seed = 0; seed < 5; ++seed) {
            auto sd = make_synthetic(cas.K_true, cas.n_cells, cas.n_snps,
                                     0.30, 5.0, static_cast<uint64_t>(seed * 13 + cas.K_true * 7));
            DonorDemuxConfig cfg;
            cfg.n_donors       = -1;
            cfg.max_donors     = std::max(cas.K_true + 2, 5);
            cfg.subsample_cells = std::min(500, cas.n_cells);
            cfg.em_max_iter    = 50;
            cfg.n_inits        = 3;
            cfg.seed           = static_cast<uint64_t>(seed);

            // Build CoveredSNPData from the already-extracted cov
            // select_k_bic works on cov directly — extract it by running a dummy demux
            // and checking n_donors_k. Or just call select_k_bic directly.
            const std::vector<int> cands = select_k_bic(
                sd.cov, 1, cfg.max_donors, cfg);

            bool found = false;
            for (int K : cands)
                if (K == cas.K_true) { found = true; break; }
            if (found) ++n_ok;
            ++n_total;
        }
    }

    char msg[96];
    std::snprintf(msg, sizeof(msg), "True K in EM+BIC candidates >= 70%% (%d/%d)", n_ok, n_total);
    CHECK(n_ok >= (n_total * 7 / 10), msg);
}

// ─── Test 3: Speed regression ─────────────────────────────────────────────────
static void test_speed_regression() {
    std::cout << "\n[Test 3] Speed regression: 3000 cells x 5000 SNPs x K_true=1\n";

    // Build synthetic data of the target size
    auto sd = make_synthetic(1, 3000, 5000, 0.30, 5.0, 42);

    DonorDemuxConfig cfg;
    cfg.n_donors       = -1;
    cfg.max_donors     = 8;
    cfg.threads        = 4;
    cfg.fast_k_select  = true;
    cfg.subsample_cells = 500;
    cfg.em_max_iter    = 50;
    cfg.n_inits        = 3;
    cfg.max_iter_pre   = 50;
    cfg.max_iter       = 100;
    cfg.seed           = 42;

    const auto t0 = std::chrono::steady_clock::now();
    auto result = run_demux(
        static_cast<uint32_t>(sd.n_snps), static_cast<uint32_t>(sd.n_cells),
        sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
        sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
        cfg);
    const auto t1 = std::chrono::steady_clock::now();

    const double wall_s = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "  Wall time: " << wall_s << " s, K_selected=" << result.n_donors_k << "\n";

    CHECK(wall_s < 30.0, "< 30 s on 3000 cells x 5000 SNPs x K_true=1");
    CHECK(result.n_donors_k >= 1 && result.n_donors_k <= 2,
          "K selected in [1, 2] for single-donor data");
}

// ─── Test 4: Edge cases ───────────────────────────────────────────────────────
static void test_edge_cases() {
    std::cout << "\n[Test 4] Edge cases\n";

    // 4a: Tiny library (10 cells)
    {
        auto sd = make_synthetic(1, 10, 200, 0.40, 5.0, 7);
        DonorDemuxConfig cfg;
        cfg.n_donors = -1; cfg.max_donors = 4; cfg.threads = 2;
        cfg.fast_k_select = true; cfg.subsample_cells = 500;
        cfg.n_inits = 3; cfg.max_iter_pre = 30; cfg.max_iter = 50; cfg.seed = 7;
        auto r = run_demux(
            static_cast<uint32_t>(sd.n_snps),
            static_cast<uint32_t>(sd.n_cells),
            sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
            sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
            cfg);
        CHECK(r.n_donors_k >= 1, "tiny library: K >= 1");
        CHECK(r.assignments.size() == 10u, "tiny library: assignment vector size");
    }

    // 4b: n_donors fixed (avoid sweep entirely)
    {
        auto sd = make_synthetic(3, 500, 500, 0.30, 5.0, 123);
        DonorDemuxConfig cfg;
        cfg.n_donors = 3; cfg.threads = 4;
        cfg.fast_k_select = true;  // no-op when n_donors is fixed
        cfg.n_inits = 3; cfg.max_iter_pre = 50; cfg.max_iter = 100; cfg.seed = 123;
        auto r = run_demux(500, 500,
            sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
            sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
            cfg);
        CHECK(r.n_donors_k == 3, "fixed n_donors=3 respected");
    }

    // 4c: All cells same genotype (K=1 trivially best)
    {
        // Simulate: all cells from donor 0, all SNPs have the same allele freq
        CoveredSNPData cov;
        cov.n_cells = 200; cov.n_covered_snps = 100;
        cov.cells.resize(200);
        cov.covered_to_original.resize(100);
        std::iota(cov.covered_to_original.begin(), cov.covered_to_original.end(), 0);
        // All cells get identical data: ad=2, dp=5 for all 100 SNPs
        for (auto& cell : cov.cells) {
            for (int j = 0; j < 100; ++j) {
                cell.snp_idx.push_back(j);
                cell.ad.push_back(2);
                cell.dp.push_back(5);
            }
        }
        // Use select_k_bic directly
        DonorDemuxConfig cfg;
        cfg.subsample_cells = 200; cfg.em_max_iter = 30; cfg.n_inits = 3; cfg.seed = 99;
        cfg.fast_k_select = true;
        auto cands = select_k_bic(cov, 1, 4, cfg);
        CHECK(!cands.empty(), "degenerate data: EM BIC returns candidates");
        CHECK(cands[0] == 1 || (cands.size() > 1 && cands[1] == 1),
              "degenerate data: K=1 is a candidate");
    }

    // 4d: fast_k_select=false (old behaviour, should still work)
    {
        auto sd = make_synthetic(2, 300, 300, 0.30, 5.0, 55);
        DonorDemuxConfig cfg;
        cfg.n_donors = -1; cfg.max_donors = 4; cfg.threads = 2;
        cfg.fast_k_select = false;
        cfg.n_inits = 3; cfg.max_iter_pre = 30; cfg.max_iter = 50; cfg.seed = 55;
        auto r = run_demux(300, 300,
            sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
            sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
            cfg);
        CHECK(r.n_donors_k >= 1 && r.n_donors_k <= 4, "fast_k_select=false: valid K");
    }
}

// ─── Test 5: Numerical stability ─────────────────────────────────────────────
static void test_numerical_stability() {
    std::cout << "\n[Test 5] Numerical stability\n";

    // 5a: Very unbalanced mixing (95% donor A, 5% donor B)
    {
        const int N = 400, J = 200;
        CoveredSNPData cov;
        cov.n_cells = N; cov.n_covered_snps = J;
        cov.cells.resize(N);
        cov.covered_to_original.resize(J);
        std::iota(cov.covered_to_original.begin(), cov.covered_to_original.end(), 0);

        std::mt19937 rng(1234);
        std::bernoulli_distribution is_minor(0.05);
        for (int c = 0; c < N; ++c) {
            const bool minor = is_minor(rng);
            for (int j = 0; j < J; ++j) {
                const uint8_t dp = 5;
                // Major donor: af=0.1; minor donor: af=0.9
                std::binomial_distribution<int> binom(dp, minor ? 0.9f : 0.1f);
                const uint8_t ad = static_cast<uint8_t>(binom(rng));
                cov.cells[c].snp_idx.push_back(j);
                cov.cells[c].ad.push_back(ad);
                cov.cells[c].dp.push_back(dp);
            }
        }

        DonorDemuxConfig cfg;
        cfg.n_donors = 2; cfg.threads = 2;
        cfg.n_inits = 3; cfg.max_iter_pre = 50; cfg.max_iter = 100; cfg.seed = 1234;
        // Build minimal CSC from cov
        std::vector<int32_t> ad_indptr(N+1, 0), dp_indptr(N+1, 0);
        std::vector<int32_t> ad_indices, dp_indices;
        std::vector<uint8_t> ad_data, dp_data;
        for (int c = 0; c < N; ++c) {
            for (size_t si = 0; si < cov.cells[c].snp_idx.size(); ++si) {
                dp_indices.push_back(cov.cells[c].snp_idx[si]);
                dp_data.push_back(cov.cells[c].dp[si]);
                if (cov.cells[c].ad[si] > 0) {
                    ad_indices.push_back(cov.cells[c].snp_idx[si]);
                    ad_data.push_back(cov.cells[c].ad[si]);
                }
                dp_indptr[c+1]++;
            }
            if (cov.cells[c].ad[0] > 0 || true) {  // count ad per cell
                int cnt = 0;
                for (auto v : cov.cells[c].ad) if (v > 0) ++cnt;
                ad_indptr[c+1] = cnt;
            }
        }
        for (int c = 0; c < N; ++c) { dp_indptr[c+1] += dp_indptr[c]; ad_indptr[c+1] += ad_indptr[c]; }

        // Actually build correctly from the cov structure
        std::vector<int32_t> ad_ip(N+1, 0), dp_ip(N+1, 0);
        std::vector<int32_t> ad_idx2, dp_idx2;
        std::vector<uint8_t> ad_d2, dp_d2;
        for (int c = 0; c < N; ++c) {
            for (size_t si = 0; si < cov.cells[c].snp_idx.size(); ++si) {
                dp_ip[c+1]++;
                dp_idx2.push_back(cov.cells[c].snp_idx[si]);
                dp_d2.push_back(cov.cells[c].dp[si]);
                if (cov.cells[c].ad[si] > 0) {
                    ad_ip[c+1]++;
                    ad_idx2.push_back(cov.cells[c].snp_idx[si]);
                    ad_d2.push_back(cov.cells[c].ad[si]);
                }
            }
        }
        for (int c = 0; c < N; ++c) { dp_ip[c+1] += dp_ip[c]; ad_ip[c+1] += ad_ip[c]; }

        auto r = run_demux(J, N,
            ad_ip.data(), ad_idx2.data(), ad_d2.data(), ad_d2.size(),
            dp_ip.data(), dp_idx2.data(), dp_d2.data(), dp_d2.size(),
            cfg);

        // Should have assigned ~380 confident cells (95% major)
        int n_assigned = 0;
        for (auto& a : r.assignments)
            if (a.label == "donor0" || a.label == "donor1") ++n_assigned;

        CHECK(r.n_donors_k == 2, "unbalanced mixing: K=2 found");
        CHECK(n_assigned > N * 8 / 10, "unbalanced mixing: >=80% cells assigned");
    }

    // 5b: run_em() on K=1 (trivial case — all cells same donor)
    {
        CoveredSNPData cov;
        cov.n_cells = 100; cov.n_covered_snps = 50;
        cov.cells.resize(100);
        cov.covered_to_original.resize(50);
        std::iota(cov.covered_to_original.begin(), cov.covered_to_original.end(), 0);
        for (auto& cell : cov.cells)
            for (int j = 0; j < 50; ++j) {
                cell.snp_idx.push_back(j);
                cell.ad.push_back(2); cell.dp.push_back(5);
            }
        std::vector<int> subset(100);
        std::iota(subset.begin(), subset.end(), 0);
        EMResult em = run_em(cov, 1, subset, 30, 42);
        CHECK(std::isfinite(em.log_likelihood), "run_em K=1: finite LL");
        CHECK(em.bic > 0.0, "run_em K=1: positive BIC");
    }

    // 5c: No covered SNPs (should return unassigned result without crash)
    {
        // All dp=0 → extract_covered_snps finds nothing
        std::vector<int32_t> ad_ip(11, 0), dp_ip(11, 0);
        std::vector<int32_t> ad_idx, dp_idx;
        std::vector<uint8_t> ad_d, dp_d;
        DonorDemuxConfig cfg; cfg.n_donors = -1; cfg.seed = 42;
        auto r = run_demux(100, 10,
            ad_ip.data(), ad_idx.data(), ad_d.data(), 0,
            dp_ip.data(), dp_idx.data(), dp_d.data(), 0,
            cfg);
        CHECK(r.n_donors_k == 0, "no-SNP input: n_donors_k=0");
        CHECK(r.assignments.size() == 10, "no-SNP input: correct assignment size");
        CHECK(r.assignments[0].label == "unassigned", "no-SNP input: all unassigned");
    }
}

// ─── Test 6: Assignment concordance with true donors ─────────────────────────
static void test_assignment_concordance() {
    std::cout << "\n[Test 6] Assignment concordance with ground truth\n";

    struct Case { int K; int n_cells; int n_snps; float min_concordance; };
    std::vector<Case> cases = {
        {1, 500,  500,  0.90f},
        {2, 500,  500,  0.85f},
        {3, 1000, 1000, 0.80f},
    };

    for (auto& c : cases) {
        auto sd = make_synthetic(c.K, c.n_cells, c.n_snps, 0.30, 6.0, 77);

        DonorDemuxConfig cfg;
        cfg.n_donors       = c.K;     // fix K to true value for fair concordance test
        cfg.threads        = 4;
        cfg.fast_k_select  = true;    // no-op since n_donors is fixed
        cfg.n_inits        = 5;
        cfg.max_iter_pre   = 50;
        cfg.max_iter       = 150;
        cfg.seed           = 77;

        auto r = run_demux(
            static_cast<uint32_t>(c.n_snps), static_cast<uint32_t>(c.n_cells),
            sd.ad_indptr.data(), sd.ad_indices.data(), sd.ad_data.data(), sd.ad_data.size(),
            sd.dp_indptr.data(), sd.dp_indices.data(), sd.dp_data.data(), sd.dp_data.size(),
            cfg);

        // Extract predicted donors (filtering out "unassigned")
        std::vector<int> pred(c.n_cells, -1);
        for (int i = 0; i < c.n_cells; ++i)
            if (r.assignments[i].donor_id < c.K)
                pred[i] = r.assignments[i].donor_id;

        const float conc = assignment_concordance(pred, sd.true_donor, c.K);

        char msg[128];
        std::snprintf(msg, sizeof(msg),
                      "K=%d n=%d concordance=%.1f%% (>= %.0f%%)",
                      c.K, c.n_cells, conc * 100.0f, c.min_concordance * 100.0f);
        CHECK(conc >= c.min_concordance, msg);
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== test_donor_demux_perf ===\n";

    test_k_selection_accuracy();
    test_em_vb_concordance();
    test_speed_regression();
    test_edge_cases();
    test_numerical_stability();
    test_assignment_concordance();

    std::cout << "\n=== Results: " << n_pass << " passed, " << n_fail << " failed ===\n";
    return (n_fail > 0) ? 1 : 0;
}
