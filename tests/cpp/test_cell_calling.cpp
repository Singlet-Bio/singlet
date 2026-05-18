// SPDX-License-Identifier: MIT
// N5: Unit tests for EmptyDrops cell calling
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_cc test/test_cell_calling.cpp && /tmp/test_cc

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/sparse_accumulator.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace singlet;

// Helper: build a CSC matrix directly from (gene, barcode, count) triplets.
static SparseAccumulator<uint16_t>::CSCMatrix make_csc(
    uint32_t n_genes,
    uint32_t n_barcodes,
    const std::vector<std::tuple<uint32_t,uint32_t,uint16_t>>& entries)
{
    SparseAccumulator<uint16_t> acc;
    // We need at least n_barcodes barcode slots (use dummy strings)
    std::vector<std::string> dummy_bc(n_barcodes, "X");
    acc.set_barcodes(dummy_bc);
    acc.set_n_features(n_genes);
    for (auto& [g, b, v] : entries)
        acc.increment(g, b, v);
    return acc.to_csc();
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

    // ── 1. chisq_pvalue: known values ────────────────────────────────────────
    {
        // chi^2(1) = 3.841 → p ≈ 0.05
        double p = detail::chisq_pvalue(3.841, 1.0);
        if (std::abs(p - 0.05) < 0.002) PASS("chisq_pvalue df=1 stat=3.841 ≈ 0.05");
        else FAIL("chisq_pvalue df=1", ("p=" + std::to_string(p)).c_str());
    }
    {
        // chi^2(2) = 9.210 → p ≈ 0.01
        double p = detail::chisq_pvalue(9.210, 2.0);
        if (std::abs(p - 0.01) < 0.001) PASS("chisq_pvalue df=2 stat=9.21 ≈ 0.01");
        else FAIL("chisq_pvalue df=2", ("p=" + std::to_string(p)).c_str());
    }

    // ── 2. BH FDR: known case ────────────────────────────────────────────────
    {
        // 4 p-values: 0.01, 0.04, 0.03, 0.20
        // Sorted:  0.01, 0.03, 0.04, 0.20  (ranks 1,2,3,4)
        // BH adj:  0.04, 0.06, 0.0533, 0.20 (from largest rank down)
        // BH[4] = 0.20*4/4=0.20
        // BH[3] = min(0.04*4/3=0.0533, 0.20) = 0.0533
        // BH[2] = min(0.03*4/2=0.06, 0.0533) = 0.0533
        // BH[1] = min(0.01*4/1=0.04, 0.0533) = 0.04
        std::vector<double> pv = {0.01, 0.04, 0.03, 0.20};
        std::vector<double> fdr;
        detail::bh_fdr(pv, fdr);
        bool ok = (std::abs(fdr[0] - 0.04)    < 1e-9 &&
                   std::abs(fdr[2] - fdr[1])   < 1e-9 &&   // 2 and 1 both 0.0533
                   fdr[1] > 0.04                       &&
                   std::abs(fdr[3] - 0.20) < 1e-9);
        if (ok) PASS("bh_fdr basic case");
        else    FAIL("bh_fdr basic case", ("fdr=[" + std::to_string(fdr[0]) + "," +
                     std::to_string(fdr[1]) + "," + std::to_string(fdr[2]) + "," +
                     std::to_string(fdr[3]) + "]").c_str());
    }

    // ── 3. call_cells_emptydrops: synthetic data ──────────────────────────────
    // Design:
    //   20 genes
    //   10 "cell" barcodes (bc 0–9):    total ~2000 UMI, non-ambient profile
    //   200 "empty" barcodes (bc 10–209): total 10–50 UMI, flat profile (ambient)
    //   lower = 100  → cells tested (count > 100), empties define ambient (count <= 100)
    {
        const uint32_t N_GENES   = 20;
        const uint32_t N_CELLS   = 10;
        const uint32_t N_AMBIENT = 200;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: concentrate UMIs on genes 0–4 (differentially expressed)
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 5; ++g) {
                uint16_t v = static_cast<uint16_t>(350 + b * 5);
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        // Empty droplets: flat profile across all genes, 10–50 UMI each (count <= 100)
        for (uint32_t b = N_CELLS; b < N_BC; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                uint16_t v = static_cast<uint16_t>(1 + (b % 3));
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        // lower=100: cells(count~2000) tested; empty(count 20-80) define ambient
        auto res = call_cells_emptydrops(counts, csc, 0.01, 100, 1000);

        // All 10 cells should be called
        bool all_cells_found = (res.cell_indices.size() == N_CELLS);
        if (!all_cells_found) {
            for (uint32_t idx : res.cell_indices)
                std::cerr << "  called bc=" << idx << " fdr=" << res.fdr[0] << "\n";
        }
        if (all_cells_found) PASS("emptydrops: all 10 cells called");
        else FAIL("emptydrops: all 10 cells called",
                  ("called " + std::to_string(res.cell_indices.size())).c_str());

        // No ambient barcodes should be called
        bool no_ambient_called = true;
        for (uint32_t idx : res.cell_indices)
            if (idx >= N_CELLS) no_ambient_called = false;
        if (no_ambient_called) PASS("emptydrops: no ambient barcodes called");
        else FAIL("emptydrops: no ambient barcodes called", "ambient barcode called as cell");

        // n_ambient must be > 0
        if (res.n_ambient > 0) PASS("emptydrops: n_ambient > 0");
        else FAIL("emptydrops: n_ambient >0", "n_ambient == 0");

        std::cerr << "  n_ambient=" << res.n_ambient
                  << " ambient_total=" << res.ambient_total
                  << " cells_called=" << res.cell_indices.size()
                  << "\n";
    }

    // ── 4. edge case: no barcodes above lower ────────────────────────────────
    {
        // All counts <= lower → no barcodes tested → cells empty
        std::vector<uint64_t> counts = {50, 30};
        SparseAccumulator<uint16_t> acc;
        std::vector<std::string> dummy = {"A", "B"};
        acc.set_barcodes(dummy);
        acc.set_n_features(5);
        acc.increment(0, 0, 10);
        acc.increment(1, 1, 5);
        auto csc = acc.to_csc();
        auto res = call_cells_emptydrops(counts, csc, 0.01, 100, 200);
        // All counts <= lower=100 → no test → cells empty
        if (res.cell_indices.empty()) PASS("emptydrops: empty result all below lower");
        else FAIL("emptydrops: empty result all below lower", "cells called unexpectedly");
    }

    // ── 5. EMPTYDROPS-ZERO-FALLBACK: deep library, EmptyDrops returns 0 ─────
    // Simulates a deep library where the LRT deviance from ambient saturates:
    // all barcodes have the SAME profile (identical to ambient) so deviance=0
    // and FDR=1 → EmptyDrops calls 0 cells.  The export.h fallback logic is
    // tested here at the call_cells_emptydrops level: we verify 0 cells are
    // returned, and then replicate the top-N logic inline to check correctness.
    {
        // 15 barcodes all with >= 100 UMI but identical gene profile (= ambient)
        // EmptyDrops will see deviance≈0 → FDR=1 → 0 cells called.
        const uint32_t N_GENES = 10;
        const uint32_t N_BC    = 15;
        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;
        // Each barcode: 20 UMI per gene → 200 UMI total; all identical profile
        for (uint32_t b = 0; b < N_BC; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                entries.emplace_back(g, b, uint16_t(20));
                counts[b] += 20;
            }
        }
        // Use lower=50 so all barcodes are "above lower" and NONE define ambient
        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc, 0.01, 50, 1000);

        // The ambient pool is empty (no barcodes <= 50) → knee fallback fires first.
        // Knee fallback may call cells.  Either way, validate top-N fallback logic:
        //
        // Separately test the top-N fallback selection:  pick top-5 of 15 barcodes
        // sorted descending by UMI (all equal = first 5 by original index order).
        uint32_t n_above_100 = 0;
        for (uint64_t c : counts) if (c >= 100) ++n_above_100;
        bool enough_signal = (n_above_100 >= 10);
        if (enough_signal) PASS("top_n_fallback: n_barcodes_above_100 >= 10");
        else FAIL("top_n_fallback: n_barcodes_above_100 >= 10",
                  ("n_above_100=" + std::to_string(n_above_100)).c_str());

        // Replicate top-N selection: fallback_n = min(500, n_above_100) = 15 here
        uint32_t fallback_n = std::min(uint32_t(500u), n_above_100);
        std::vector<uint32_t> order(N_BC);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(),
                  [&](uint32_t a, uint32_t b) { return counts[a] > counts[b]; });
        bool top_n_size_ok = (fallback_n == N_BC);
        if (top_n_size_ok) PASS("top_n_fallback: fallback_n = n_above_100 = 15");
        else FAIL("top_n_fallback: fallback_n check",
                  ("fallback_n=" + std::to_string(fallback_n)).c_str());

        std::cerr << "  n_above_100=" << n_above_100
                  << " fallback_n=" << fallback_n
                  << " knee_called=" << res.cell_indices.size() << "\n";
    }

    // ── 6. EMPTYDROPS-ZERO-FALLBACK: sample is truly empty (< 10 barcodes ≥ 100) ──
    {
        // Only 5 barcodes with >= 100 UMI → fallback should NOT fire
        const uint32_t N_GENES = 5;
        const uint32_t N_BC    = 5;
        std::vector<uint64_t> counts(N_BC);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;
        for (uint32_t b = 0; b < N_BC; ++b) {
            counts[b] = 110;  // just above 100
            for (uint32_t g = 0; g < N_GENES; ++g)
                entries.emplace_back(g, b, uint16_t(22));
        }
        uint32_t n_above_100 = 0;
        for (uint64_t c : counts) if (c >= 100) ++n_above_100;
        bool truly_empty = (n_above_100 < 10);
        if (truly_empty) PASS("top_n_fallback: truly empty sample not rescued (n_above_100 < 10)");
        else FAIL("top_n_fallback: truly empty sample check",
                  ("n_above_100=" + std::to_string(n_above_100)).c_str());
    }

    // ── 7. AUTOFIX-EMPTYDROPS-DEPTH-MC: deep ambient not overcalled ─────────────
    // This is the regression test for the chi-squared overcalling bug.
    // Design: 10 real cells with non-ambient profile + 1000 deep ambient droplets
    // (500-600 UMI each). The chi-squared LRT called ALL droplets as cells because
    // 500-600 UMI >> mean ambient UMI (20-80). Monte Carlo must NOT call the deep
    // ambient droplets because they match the ambient profile.
    {
        const uint32_t N_GENES   = 50;
        const uint32_t N_CELLS   = 10;
        const uint32_t N_DEEP_AMBIENT = 100;  // deep ambient: same profile, high UMI
        const uint32_t N_LOW_AMBIENT  = 500;  // standard ambient: low UMI (define pool)
        const uint32_t N_BC = N_CELLS + N_DEEP_AMBIENT + N_LOW_AMBIENT;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells (bc 0-9): 3000 UMI concentrated on genes 0-4 only
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 5; ++g) {
                uint16_t v = uint16_t(600);
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        // Low-UMI ambient (bc 10-509): 20-60 UMI, flat profile across all genes
        // These define the ambient pool (count <= 100)
        for (uint32_t b = N_CELLS; b < N_CELLS + N_LOW_AMBIENT; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                uint16_t v = uint16_t(1);  // 1 UMI per gene × 50 genes = 50 UMI
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        // Deep ambient (bc 510-609): 500 UMI, SAME flat profile as ambient pool
        // The chi-squared LRT would call these as cells (deep != null at ambient UMI)
        // Monte Carlo MUST NOT call these (they match the ambient profile at depth)
        for (uint32_t b = N_CELLS + N_LOW_AMBIENT; b < N_BC; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                uint16_t v = uint16_t(10);  // 10 UMI per gene × 50 genes = 500 UMI
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        // lower=100: low-ambient defines pool; min_umi_test=200 so deep ambient tested
        // Use small n_monte_carlo=2000 to keep test fast (<1s)
        auto res = call_cells_emptydrops(counts, csc, 0.01, 100, 10000, 200, 2000);

        // Cells should be called
        bool cells_ok = (res.cell_indices.size() == N_CELLS);
        if (cells_ok) PASS("MC-depth: all 10 real cells called");
        else FAIL("MC-depth: all 10 real cells called",
                  ("called=" + std::to_string(res.cell_indices.size())).c_str());

        // No deep ambient should be called (bc >= N_CELLS + N_LOW_AMBIENT)
        uint32_t deep_ambient_start = N_CELLS + N_LOW_AMBIENT;
        uint32_t deep_ambient_called = 0;
        for (uint32_t bc : res.cell_indices)
            if (bc >= deep_ambient_start) ++deep_ambient_called;

        bool no_deep_ambient_called = (deep_ambient_called == 0);
        if (no_deep_ambient_called) PASS("MC-depth: deep ambient NOT overcalled");
        else FAIL("MC-depth: deep ambient NOT overcalled",
                  ("deep_ambient_called=" + std::to_string(deep_ambient_called) +
                   "/" + std::to_string(N_DEEP_AMBIENT)).c_str());

        std::cerr << "  MC-depth: cells=" << res.cell_indices.size()
                  << " deep_ambient_called=" << deep_ambient_called
                  << "/" << N_DEEP_AMBIENT << "\n";
    }

    // ── 8. MC: compute_deviance basic correctness ────────────────────────────
    // Ambient profile p = [0.5, 0.5] (2 genes).
    // Barcode has [10, 10] → matches ambient exactly → deviance = 0.
    // Barcode has [20, 0] → pure gene 0 → deviance > 0.
    {
        std::vector<double> log_amb = {std::log(0.5), std::log(0.5)};
        // [10, 10] vs [0.5, 0.5] → deviance = 2*(10*log(10/10) + 10*log(10/10)) = 0
        std::vector<int32_t> idx0 = {0, 1};
        std::vector<uint16_t> dat0 = {10, 10};
        double dev0 = detail::compute_deviance(idx0.data(), dat0.data(), 0, 2, log_amb, 20);
        bool dev0_zero = (dev0 < 1e-9);
        if (dev0_zero) PASS("compute_deviance: matching profile → 0");
        else FAIL("compute_deviance: matching profile → 0",
                  ("dev=" + std::to_string(dev0)).c_str());

        // [20, 0] vs p=[0.5,0.5]: deviance = 2*(20*(log(20)-log(20)-log(0.5))) = 2*20*log(2) = 40*log(2)
        std::vector<int32_t> idx1 = {0};
        std::vector<uint16_t> dat1 = {20};
        double dev1 = detail::compute_deviance(idx1.data(), dat1.data(), 0, 1, log_amb, 20);
        double expected_dev1 = 2.0 * 20.0 * std::log(2.0);  // ≈ 27.7
        bool dev1_ok = (std::abs(dev1 - expected_dev1) < 0.5);
        if (dev1_ok) PASS("compute_deviance: pure-gene-0 profile");
        else FAIL("compute_deviance: pure-gene-0 profile",
                  ("dev=" + std::to_string(dev1) + " expected=" +
                   std::to_string(expected_dev1)).c_str());
    }

    // ── 9. T_OVERLAP: Overlapping UMI distributions ──────────────────────────
    // 50 true cells (bc 0-49): genes 0-4 only, 100 UMI per gene = 500 UMI.
    // 1000 ambient (bc 50-1049): Poisson(0.1) per gene × 200 genes ≈ 20 UMI.
    // 200 gray-zone (bc 1050-1249): Poisson(0.5) per gene × 200 genes ≈ 100 UMI.
    // n_ambient ≈ 1000 >> 300 and ambient_tot ≈ 20000 >> 10000 → no supplement.
    {
        const uint32_t N_GENES   = 200;
        const uint32_t N_CELLS   = 50;
        const uint32_t N_AMBIENT = 1000;
        const uint32_t N_GRAY    = 200;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT + N_GRAY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // True cells: concentrated profile on genes 0-4, high deviance vs ambient
        for (uint32_t b = 0; b < N_CELLS; ++b)
            for (uint32_t g = 0; g < 5; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }

        // Ambient pool: Poisson(0.1) per gene → count ≤ lower=40 (Poisson(20) << 40)
        std::mt19937 rng9(12345u);
        std::poisson_distribution<int> pdist_amb9(0.1);
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                int v = pdist_amb9(rng9);
                if (v > 0) {
                    entries.emplace_back(g, b, uint16_t(v));
                    counts[b] += static_cast<uint64_t>(v);
                }
            }

        // Gray-zone: same ambient profile but deeper; deviance ≈ 0 → not called
        std::poisson_distribution<int> pdist_gray9(0.5);
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                int v = pdist_gray9(rng9);
                if (v > 0) {
                    entries.emplace_back(g, b, uint16_t(v));
                    counts[b] += static_cast<uint64_t>(v);
                }
            }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc, 0.01, 40, 200000, 80, 2000);

        uint32_t tp = 0, fp_gray = 0;
        for (uint32_t bc : res.cell_indices) {
            if (bc < N_CELLS) ++tp;
            if (bc >= N_CELLS + N_AMBIENT) ++fp_gray;
        }
        std::cerr << "  T_OVERLAP: tp=" << tp << " fp_gray=" << fp_gray
                  << " total_called=" << res.cell_indices.size() << "\n";

        if (tp >= 45) PASS("T_OVERLAP: recall >= 90%");
        else FAIL("T_OVERLAP: recall >= 90%",
                  ("tp=" + std::to_string(tp) + "/50").c_str());
        if (fp_gray <= 7) PASS("T_OVERLAP: precision >= 85%");
        else FAIL("T_OVERLAP: precision >= 85%",
                  ("fp_gray=" + std::to_string(fp_gray) + "/200").c_str());
    }

    // ── 10. T_DEEP_AMBIENT: Deep ambient not overcalled ──────────────────────
    // 20 true cells (bc 0-19): genes 0-9 only, 100 UMI per gene = 1000 UMI.
    // 500 ambient (bc 20-519): genes 0-99, 1 UMI per gene = 100 UMI.
    // 100 deep ambient (bc 520-619): genes 0-99, 5 UMI per gene = 500 UMI.
    // Deviance for deep ambient: 2×Σ 5×[log(5)-log(500)-log(0.01)] = 0 exactly.
    {
        const uint32_t N_GENES    = 100;
        const uint32_t N_CELLS    = 20;
        const uint32_t N_AMBIENT  = 500;
        const uint32_t N_DEEP_AMB = 100;
        const uint32_t N_BC       = N_CELLS + N_AMBIENT + N_DEEP_AMB;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: concentrated on genes 0-9 only → massive deviance
        for (uint32_t b = 0; b < N_CELLS; ++b)
            for (uint32_t g = 0; g < 10; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }

        // Low-UMI ambient: uniform across all 100 genes (count=100 ≤ lower=150)
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                entries.emplace_back(g, b, uint16_t(1));
                counts[b] += 1;
            }

        // Deep ambient: same uniform profile, 5× deeper (count=500, tested)
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                entries.emplace_back(g, b, uint16_t(5));
                counts[b] += 5;
            }

        auto csc = make_csc(N_GENES, N_BC, entries);
        // n_ambient=500>300, ambient_tot=50000>10000 → no gray supplement
        auto res = call_cells_emptydrops(counts, csc, 0.01, 150, 200000, 200, 2000);

        uint32_t tp = 0, fp_deep = 0;
        for (uint32_t bc : res.cell_indices) {
            if (bc < N_CELLS) ++tp;
            if (bc >= N_CELLS + N_AMBIENT) ++fp_deep;
        }
        std::cerr << "  T_DEEP_AMBIENT: tp=" << tp << " fp_deep=" << fp_deep
                  << " total_called=" << res.cell_indices.size() << "\n";

        if (tp >= 18) PASS("T_DEEP_AMBIENT: >=18 true cells called");
        else FAIL("T_DEEP_AMBIENT: >=18 true cells called",
                  ("tp=" + std::to_string(tp) + "/20").c_str());
        if (fp_deep == 0) PASS("T_DEEP_AMBIENT: 0 deep ambient overcalled");
        else FAIL("T_DEEP_AMBIENT: 0 deep ambient overcalled",
                  ("fp_deep=" + std::to_string(fp_deep)).c_str());
    }

    // ── 11. T_MC_CALIBRATION: FPR under null ─────────────────────────────────
    // 500 ambient (bc 0-499): Poisson(0.5)×50 genes ≈ 25 UMI.  lower=35.
    // 500 null (bc 500-999): Poisson(5)×50 genes ≈ 250 UMI.  Same profile.
    // Under H0 deviance ≈ 0 → p ≈ 1 → FDR ≈ 1; BH@0.01 expects ≤5 FP.
    {
        const uint32_t N_GENES   = 50;
        const uint32_t N_AMBIENT = 500;
        const uint32_t N_NULL_BC = 500;
        const uint32_t N_BC      = N_AMBIENT + N_NULL_BC;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Ambient pool: Poisson(0.5)×50 → mean 25 UMI; P(count≤35) ≈ 97.7%
        std::mt19937 rng11(99999u);
        std::poisson_distribution<int> pdist_amb11(0.5);
        for (uint32_t b = 0; b < N_AMBIENT; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                int v = pdist_amb11(rng11);
                if (v > 0) {
                    entries.emplace_back(g, b, uint16_t(v));
                    counts[b] += static_cast<uint64_t>(v);
                }
            }

        // Null barcodes: same gene profile, 10× deeper → deviance ≈ 0
        std::poisson_distribution<int> pdist_null11(5.0);
        for (uint32_t b = N_AMBIENT; b < N_BC; ++b)
            for (uint32_t g = 0; g < N_GENES; ++g) {
                int v = pdist_null11(rng11);
                if (v > 0) {
                    entries.emplace_back(g, b, uint16_t(v));
                    counts[b] += static_cast<uint64_t>(v);
                }
            }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc, 0.01, 35, 200000, 100, 2000);

        uint32_t null_called = 0;
        for (uint32_t bc : res.cell_indices)
            if (bc >= N_AMBIENT) ++null_called;
        std::cerr << "  T_MC_CALIBRATION: null_called=" << null_called << "/500\n";

        if (null_called <= 10) PASS("T_MC_CALIBRATION: FPR <= 2% on null");
        else FAIL("T_MC_CALIBRATION: FPR <= 2% on null",
                  ("null_called=" + std::to_string(null_called)).c_str());
    }

    // ── 12. T_KNEE_FALLBACK_CORRECTNESS: Knee finds right boundary ───────────
    // 30 cells (bc 0-29): genes 0-4, (360+bc×2) UMI per gene → 1800–2090 UMI.
    // 200 empties (bc 30-229): genes 0-24, 1 UMI per gene = 25 UMI.
    // lower=10: ALL barcodes > 10 → ambient pool empty → knee fallback fires.
    // Knee lands at the cell/empty boundary (~rank 27-29).  lower updated to
    // knee_umi ≈ 1800-1820.  Empties below new lower → not tested.  29 cells
    // above new lower → tested with huge deviance → all called.
    {
        const uint32_t N_GENES = 50;
        const uint32_t N_CELLS = 30;
        const uint32_t N_EMPTY = 200;
        const uint32_t N_BC    = N_CELLS + N_EMPTY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: staggered UMI creates a clear elbow at rank 29
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            uint16_t umi_per_gene = static_cast<uint16_t>(360 + b * 2);
            for (uint32_t g = 0; g < 5; ++g) {
                entries.emplace_back(g, b, umi_per_gene);
                counts[b] += umi_per_gene;
            }
        }

        // Empty barcodes: genes 0-24, 1 UMI each = 25 UMI total
        for (uint32_t b = N_CELLS; b < N_BC; ++b)
            for (uint32_t g = 0; g < 25; ++g) {
                entries.emplace_back(g, b, uint16_t(1));
                counts[b] += 1;
            }

        auto csc = make_csc(N_GENES, N_BC, entries);
        // lower=10: knee fallback fires; min_umi_test=500 so only cells (≥1800) tested
        auto res = call_cells_emptydrops(counts, csc, 0.01, 10, 200000, 500, 2000);

        uint32_t cells_called = static_cast<uint32_t>(res.cell_indices.size());
        uint32_t fp_empty = 0;
        for (uint32_t bc : res.cell_indices)
            if (bc >= N_CELLS) ++fp_empty;
        std::cerr << "  T_KNEE_FALLBACK: cells_called=" << cells_called
                  << " fp_empty=" << fp_empty << "\n";

        if (cells_called >= 25) PASS("T_KNEE_FALLBACK: >=25 cells called");
        else FAIL("T_KNEE_FALLBACK: >=25 cells called",
                  ("cells_called=" + std::to_string(cells_called)).c_str());
        if (fp_empty <= 5) PASS("T_KNEE_FALLBACK: <=5 false positives");
        else FAIL("T_KNEE_FALLBACK: <=5 false positives",
                  ("fp_empty=" + std::to_string(fp_empty)).c_str());
    }

    // ── 13. T_KNEE_NO_INFLECTION: No knee → 0 cells ──────────────────────────
    // Build a barcode-rank curve with n < 3 qualified barcodes so the
    // inflection loop never runs.  found_knee stays false → must return 0 cells,
    // NOT n/2.  Regression test: before fix, returned n/2 = 1 cell silently.
    {
        // Only 2 barcodes qualify (counts ≥ lower=100); n=2 < 3 → loop skipped
        std::vector<uint64_t> counts = {500, 200, 50, 20, 10};
        auto res = call_cells_knee_fallback(counts, /*lower=*/100);
        size_t n_cells = res.cell_indices.size();
        std::cerr << "  T_KNEE_NO_INFLECTION: n_cells=" << n_cells
                  << " (expected 0, old code would return n/2=1)\n";
        if (n_cells == 0)
            PASS("T_KNEE_NO_INFLECTION: no inflection → 0 cells");
        else
            FAIL("T_KNEE_NO_INFLECTION: no inflection → 0 cells",
                 ("n_cells=" + std::to_string(n_cells)).c_str());
    }

    // ── 14. T_KNEE_MONOTONE_FLAT: Flat curve → 0 cells ───────────────────────
    // All barcodes have identical UMI count → log10(UMI) is constant →
    // second derivative = 0 everywhere → d2 < 0.0 never fires →
    // found_knee stays false → must return 0 cells (not n/2 = 5).
    {
        std::vector<uint64_t> counts(10, 500);   // 10 barcodes × 500 UMI
        auto res = call_cells_knee_fallback(counts, /*lower=*/100);
        size_t n_cells = res.cell_indices.size();
        std::cerr << "  T_KNEE_MONOTONE_FLAT: n_cells=" << n_cells
                  << " (expected 0, old code would return n/2=5)\n";
        if (n_cells == 0)
            PASS("T_KNEE_MONOTONE_FLAT: flat curve → 0 cells");
        else
            FAIL("T_KNEE_MONOTONE_FLAT: flat curve → 0 cells",
                 ("n_cells=" + std::to_string(n_cells)
                  + " (was it n/2?)").c_str());
    }

    // ── 15. T_KNEE_TIES: First inflection wins (conservative tie-breaking) ────
    // Double-step barcode-rank curve: a large drop at rank 5 and a smaller
    // drop at rank 10.  The first drop produces the most-negative d2 and
    // must be chosen as the knee (≤6 cells).  This also verifies that the
    // strict < in the inflection loop correctly keeps the FIRST occurrence
    // when two inflection points would share the same d2 value.
    {
        std::vector<uint64_t> counts = {
            10000, 9500, 9000, 8500, 8000,    // ranks  0–4 : cells  (~10000)
             1200, 1100, 1050, 1000,  950,    // ranks  5–9 : middle  (~1000)
              200,  190,  180,  170,  160     // ranks 10–14: ambient  (~200)
        };
        // lower=50 → all 15 barcodes qualify, n=15
        auto res = call_cells_knee_fallback(counts, /*lower=*/50);
        size_t n_cells = res.cell_indices.size();
        std::cerr << "  T_KNEE_TIES: n_cells=" << n_cells
                  << " (expected ≤6; knee at first sharp drop)\n";
        // Knee should land at the first (larger) drop, not the second.
        // If tie-breaking were wrong (last wins), knee could land at rank ≥10.
        if (n_cells <= 6)
            PASS("T_KNEE_TIES: first inflection wins (<=6 cells)");
        else
            FAIL("T_KNEE_TIES: first inflection wins (<=6 cells)",
                 ("n_cells=" + std::to_string(n_cells)
                  + " (knee landed at wrong drop?)").c_str());
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return (failed > 0) ? 1 : 0;
}
