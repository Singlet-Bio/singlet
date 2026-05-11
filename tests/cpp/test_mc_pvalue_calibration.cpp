// Monte Carlo p-value calibration tests for EmptyDrops cell calling.
//
// Tests that the MC null distribution is well-calibrated:
//   T_UNIFORMITY      — all-ambient matrix → p-values ≈ Uniform(0,1), FPR ≤ 2%
//   T_DEEP_LIBRARY    — deep cells + shallow empties → recall ≥ 80%
//   T_MC_NULL_SHAPE   — mc_null_deviances() output sanity
//   T_DEPTH_BINNING   — deeper depth → more deviance (sparse-K regime)
//   T_POISSON_VS_ALIAS— alias (N<K) vs Poisson (N≥K) give similar means

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/sparse_accumulator.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace singlet;

// Helper: build a CSC matrix from (gene, barcode, count) triplets.
static SparseAccumulator<uint16_t>::CSCMatrix make_csc(
    uint32_t n_genes,
    uint32_t n_barcodes,
    const std::vector<std::tuple<uint32_t, uint32_t, uint16_t>>& entries)
{
    SparseAccumulator<uint16_t> acc;
    std::vector<std::string> dummy_bc(n_barcodes, "X");
    acc.set_barcodes(dummy_bc);
    acc.set_n_features(n_genes);
    for (auto& [g, b, v] : entries)
        acc.increment(g, b, v);
    return acc.to_csc();
}

// Helper: build a normalized flat ambient profile and its log.
static void make_flat_profile(
    uint32_t K,
    std::vector<double>& prof,
    std::vector<double>& log_prof,
    std::vector<uint32_t>& nonzero)
{
    prof.assign(K, 1.0 / K);
    log_prof.resize(K);
    nonzero.clear();
    for (uint32_t g = 0; g < K; ++g) {
        log_prof[g] = std::log(prof[g]);
        nonzero.push_back(g);
    }
}

// Helper: compute mean and stddev of a vector.
static std::pair<double, double> mean_std(const std::vector<double>& v) {
    if (v.empty()) return {0.0, 0.0};
    double sum = 0.0, sum2 = 0.0;
    for (double x : v) { sum += x; sum2 += x * x; }
    double m = sum / v.size();
    double var = sum2 / v.size() - m * m;
    return {m, std::sqrt(std::max(0.0, var))};
}

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* name) {
        ++passed;
        std::cerr << "PASS: " << name << "\n";
    };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // ── T_UNIFORMITY ─────────────────────────────────────────────────────────
    // All-ambient matrix: 500 test barcodes drawn from the same flat profile
    // as 500 ambient barcodes.  Under H₀, FDR-corrected p-values should
    // mostly be > 0.01 and raw p-values should be ≈ Uniform(0,1).
    {
        std::cerr << "\n=== T_UNIFORMITY ===\n";
        const uint32_t N_GENES = 100;
        const uint32_t N_AMB = 500;   // ambient-pool barcodes (UMI ≤ lower)
        const uint32_t N_TEST = 500;  // test barcodes (all null)
        const uint32_t N_BC = N_AMB + N_TEST;

        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> umi_dist(200, 500);
        std::vector<double> flat_w(N_GENES, 1.0);
        std::discrete_distribution<int> gdist(flat_w.begin(), flat_w.end());

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> entries;

        // Ambient barcodes: 50 UMI each (≤ lower=100), flat profile
        for (uint32_t b = 0; b < N_AMB; ++b) {
            std::vector<uint16_t> gc(N_GENES, 0);
            for (int i = 0; i < 50; ++i)
                ++gc[gdist(rng)];
            for (uint32_t g = 0; g < N_GENES; ++g) {
                if (gc[g] > 0) {
                    entries.emplace_back(g, b, gc[g]);
                    counts[b] += gc[g];
                }
            }
        }

        // Test barcodes: 200-500 UMI each (> lower=100), same flat profile
        for (uint32_t b = N_AMB; b < N_BC; ++b) {
            int total = umi_dist(rng);
            std::vector<uint16_t> gc(N_GENES, 0);
            for (int i = 0; i < total; ++i)
                ++gc[gdist(rng)];
            for (uint32_t g = 0; g < N_GENES; ++g) {
                if (gc[g] > 0) {
                    entries.emplace_back(g, b, gc[g]);
                    counts[b] += gc[g];
                }
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);

        // lower=100, min_umi_test=101 so all test barcodes are tested
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/uint64_t(100),
            /*n_ambient_max=*/200000,
            /*min_umi_test=*/uint64_t(101),
            /*n_monte_carlo=*/10000);

        // FPR check: ≤ 10 of 500 null barcodes called as cells (≤ 2%)
        uint32_t n_called = static_cast<uint32_t>(res.cell_indices.size());
        std::cerr << "  cells_called=" << n_called
                  << " tested=" << res.tested_indices.size() << "\n";
        if (n_called <= 10)
            PASS("T_UNIFORMITY: FPR <= 2%");
        else
            FAIL("T_UNIFORMITY: FPR <= 2%",
                 ("called " + std::to_string(n_called) + "/500").c_str());

        // KS test on raw MC p-values against Uniform(0,1).
        // Uses K=500 genes (> depth 350) to force the alias path (exact
        // multinomial).  The Poisson path (N≥K) adds total-count variance
        // that compresses p-values toward 0.5 (KS≈0.24 at K=100, N=350);
        // the alias path gives proper calibration (KS≈0.05).
        // NOTE: in production K≈38606 so most cells use the alias path.
        // Deep cells with N>K hit the Poisson path and get conservative
        // p-values — a potential contributor to the 0-cell bug at >12M reads.
        {
            const uint32_t KS_GENES = 500;
            const uint32_t KS_N = 500;
            const int KS_DEPTH = 350;
            std::vector<double> exact_prof(KS_GENES, 1.0 / KS_GENES);
            std::vector<double> log_exact(KS_GENES);
            for (uint32_t g = 0; g < KS_GENES; ++g)
                log_exact[g] = std::log(exact_prof[g]);

            // Generate barcodes from flat multinomial at constant depth
            std::mt19937 ks_rng(77777);
            std::discrete_distribution<int> ks_gdist(exact_prof.begin(),
                                                      exact_prof.end());
            std::vector<uint64_t> ks_counts(KS_N, KS_DEPTH);
            std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> ks_entries;
            for (uint32_t b = 0; b < KS_N; ++b) {
                std::vector<uint16_t> gc(KS_GENES, 0);
                for (int i = 0; i < KS_DEPTH; ++i)
                    ++gc[ks_gdist(ks_rng)];
                for (uint32_t g = 0; g < KS_GENES; ++g)
                    if (gc[g] > 0)
                        ks_entries.emplace_back(g, b, gc[g]);
            }
            auto ks_csc = make_csc(KS_GENES, KS_N, ks_entries);

            // Compute deviances and MC p-values
            std::vector<uint32_t> tidx(KS_N);
            std::iota(tidx.begin(), tidx.end(), 0u);
            std::vector<double> tdev(KS_N);
            for (uint32_t b = 0; b < KS_N; ++b)
                tdev[b] = detail::compute_deviance(
                    ks_csc.indices.data(), ks_csc.data.data(),
                    ks_csc.indptr[b], ks_csc.indptr[b + 1],
                    log_exact, ks_counts[b]);

            auto raw_pv = detail::mc_emptydrops_pvalues(
                tidx, tdev, ks_counts, exact_prof, log_exact, 10000);

            // KS statistic: D = max |F_n(x) - x|
            auto sp = raw_pv;
            std::sort(sp.begin(), sp.end());
            double ks = 0.0;
            const size_t np = sp.size();
            for (size_t i = 0; i < np; ++i) {
                double d1 = std::abs(double(i + 1) / np - sp[i]);
                double d2 = std::abs(sp[i] - double(i) / np);
                ks = std::max(ks, std::max(d1, d2));
            }
            std::cerr << "  KS stat = " << ks << " (n_pvals=" << np
                      << ", K=" << KS_GENES << ", alias path)\n";
            if (ks < 0.1)
                PASS("T_UNIFORMITY: KS stat < 0.1");
            else
                FAIL("T_UNIFORMITY: KS stat < 0.1",
                     ("ks=" + std::to_string(ks)).c_str());
        }
    }

    // ── T_DEEP_LIBRARY ───────────────────────────────────────────────────────
    // Probe the large-library 0-cell bug: deep cells + shallow empties.
    {
        std::cerr << "\n=== T_DEEP_LIBRARY ===\n";
        const uint32_t N_GENES = 100;
        const uint32_t N_CELLS = 100;
        const uint32_t N_EMPTY = 200;
        const uint32_t N_BC = N_CELLS + N_EMPTY;
        const int CELL_UMI = 50000;
        const int EMPTY_UMI = 50;

        std::mt19937 rng(99999);
        // Cells: genes 0-29 (30 genes), multinomial
        std::vector<double> cell_w(N_GENES, 0.0);
        for (int g = 0; g < 30; ++g) cell_w[g] = 1.0;
        std::discrete_distribution<int> cell_gdist(cell_w.begin(), cell_w.end());

        // Empties: genes 50-99 (50 genes), multinomial
        std::vector<double> empty_w(N_GENES, 0.0);
        for (int g = 50; g < 100; ++g) empty_w[g] = 1.0;
        std::discrete_distribution<int> empty_gdist(empty_w.begin(), empty_w.end());

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> entries;

        // Cells (barcodes 0..99): 50000 UMI on genes 0-29
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            std::vector<uint16_t> gc(N_GENES, 0);
            for (int i = 0; i < CELL_UMI; ++i)
                ++gc[cell_gdist(rng)];
            for (uint32_t g = 0; g < N_GENES; ++g) {
                if (gc[g] > 0) {
                    entries.emplace_back(g, b, gc[g]);
                    counts[b] += gc[g];
                }
            }
        }

        // Empties (barcodes 100..299): 50 UMI on genes 50-99
        for (uint32_t b = N_CELLS; b < N_BC; ++b) {
            std::vector<uint16_t> gc(N_GENES, 0);
            for (int i = 0; i < EMPTY_UMI; ++i)
                ++gc[empty_gdist(rng)];
            for (uint32_t g = 0; g < N_GENES; ++g) {
                if (gc[g] > 0) {
                    entries.emplace_back(g, b, gc[g]);
                    counts[b] += gc[g];
                }
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);

        // lower=100: empties (50 UMI) → ambient, cells (50000) → tested
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/uint64_t(100),
            /*n_ambient_max=*/200000,
            /*min_umi_test=*/uint64_t(500),
            /*n_monte_carlo=*/5000);

        uint32_t n_called = static_cast<uint32_t>(res.cell_indices.size());
        std::cerr << "  cells_called=" << n_called
                  << " tested=" << res.tested_indices.size()
                  << " n_ambient=" << res.n_ambient << "\n";

        // Not the 0-cell bug
        if (n_called > 0)
            PASS("T_DEEP_LIBRARY: cell count > 0");
        else
            FAIL("T_DEEP_LIBRARY: cell count > 0",
                 "0-cell bug reproduced: EmptyDrops returned 0 cells");

        // Recall: count how many of barcodes 0..99 are called
        uint32_t true_pos = 0;
        for (uint32_t idx : res.cell_indices)
            if (idx < N_CELLS) ++true_pos;
        double recall = double(true_pos) / N_CELLS;
        std::cerr << "  recall=" << recall
                  << " (" << true_pos << "/" << N_CELLS << ")\n";
        if (recall >= 0.80)
            PASS("T_DEEP_LIBRARY: recall >= 0.80");
        else
            FAIL("T_DEEP_LIBRARY: recall >= 0.80",
                 ("recall=" + std::to_string(recall)).c_str());
    }

    // ── T_MC_NULL_SHAPE ──────────────────────────────────────────────────────
    // Directly test mc_null_deviances() output properties.
    {
        std::cerr << "\n=== T_MC_NULL_SHAPE ===\n";
        const uint32_t K = 100;
        std::vector<double> prof, log_prof;
        std::vector<uint32_t> nz;
        make_flat_profile(K, prof, log_prof, nz);

        std::mt19937 rng(42);
        auto devs = detail::mc_null_deviances(500, prof, log_prof, nz, 5000, rng);

        // All deviances >= 0
        bool all_nonneg = std::all_of(devs.begin(), devs.end(),
                                       [](double d) { return d >= 0.0; });
        if (all_nonneg) PASS("T_MC_NULL_SHAPE: all deviances >= 0");
        else FAIL("T_MC_NULL_SHAPE: all deviances >= 0", "negative deviance found");

        auto [mu, sd] = mean_std(devs);
        std::cerr << "  mean=" << mu << " std=" << sd
                  << " min=" << *std::min_element(devs.begin(), devs.end())
                  << " max=" << *std::max_element(devs.begin(), devs.end()) << "\n";

        if (mu > 0.0) PASS("T_MC_NULL_SHAPE: mean > 0");
        else FAIL("T_MC_NULL_SHAPE: mean > 0",
                  ("mean=" + std::to_string(mu)).c_str());

        if (sd > 0.0) PASS("T_MC_NULL_SHAPE: stddev > 0");
        else FAIL("T_MC_NULL_SHAPE: stddev > 0",
                  ("sd=" + std::to_string(sd)).c_str());

        double mx = *std::max_element(devs.begin(), devs.end());
        if (mx < mu + 10.0 * sd)
            PASS("T_MC_NULL_SHAPE: max < mean + 10*std");
        else
            FAIL("T_MC_NULL_SHAPE: max < mean + 10*std",
                 ("max=" + std::to_string(mx) + " threshold=" +
                  std::to_string(mu + 10.0 * sd)).c_str());
    }

    // ── T_DEPTH_BINNING ──────────────────────────────────────────────────────
    // Verify that deeper UMI depth produces larger null deviances in the
    // sparse regime (K >> N).  Uses K=1000 genes so that N=200 is in the
    // sparse regime (N << K) and N=50000 covers most genes → higher deviance.
    {
        std::cerr << "\n=== T_DEPTH_BINNING ===\n";
        const uint32_t K = 1000;
        std::vector<double> prof, log_prof;
        std::vector<uint32_t> nz;
        make_flat_profile(K, prof, log_prof, nz);

        std::mt19937 rng1(111), rng2(222);
        auto devs_shallow = detail::mc_null_deviances(200,   prof, log_prof, nz, 2000, rng1);
        auto devs_deep    = detail::mc_null_deviances(50000, prof, log_prof, nz, 2000, rng2);

        auto [mu_s, sd_s] = mean_std(devs_shallow);
        auto [mu_d, sd_d] = mean_std(devs_deep);
        std::cerr << "  shallow(N=200):  mean=" << mu_s << " std=" << sd_s << "\n"
                  << "  deep(N=50000):   mean=" << mu_d << " std=" << sd_d << "\n";

        if (mu_d > mu_s)
            PASS("T_DEPTH_BINNING: mean(deep) > mean(shallow)");
        else
            FAIL("T_DEPTH_BINNING: mean(deep) > mean(shallow)",
                 ("deep=" + std::to_string(mu_d) +
                  " shallow=" + std::to_string(mu_s)).c_str());
    }

    // ── T_POISSON_VS_ALIAS ───────────────────────────────────────────────────
    // With K=100 genes: N=50 (< K) → alias path, N=200 (≥ K) → Poisson path.
    // Both approximate the multinomial null; means should be within 20%.
    {
        std::cerr << "\n=== T_POISSON_VS_ALIAS ===\n";
        const uint32_t K = 100;
        std::vector<double> prof, log_prof;
        std::vector<uint32_t> nz;
        make_flat_profile(K, prof, log_prof, nz);

        std::mt19937 rng_a(333), rng_p(444);
        auto devs_alias   = detail::mc_null_deviances(50,  prof, log_prof, nz, 2000, rng_a);
        auto devs_poisson = detail::mc_null_deviances(200, prof, log_prof, nz, 2000, rng_p);

        auto [mu_a, sd_a] = mean_std(devs_alias);
        auto [mu_p, sd_p] = mean_std(devs_poisson);
        std::cerr << "  alias(N=50):   mean=" << mu_a << " std=" << sd_a << "\n"
                  << "  poisson(N=200): mean=" << mu_p << " std=" << sd_p << "\n";

        double ratio = (mu_a > mu_p) ? mu_a / mu_p : mu_p / mu_a;
        std::cerr << "  ratio=" << ratio << "\n";
        if (ratio < 1.20)
            PASS("T_POISSON_VS_ALIAS: means within 20%");
        else
            FAIL("T_POISSON_VS_ALIAS: means within 20%",
                 ("ratio=" + std::to_string(ratio) +
                  " alias=" + std::to_string(mu_a) +
                  " poisson=" + std::to_string(mu_p)).c_str());
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
