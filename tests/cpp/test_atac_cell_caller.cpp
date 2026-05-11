// test_atac_cell_caller.cpp
// Unit tests for A7: AtacCellCaller
// Tests: basic cell calling, auto-threshold, edge cases, filter isolation.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/atac_cell_caller.h"

using namespace singlet;

// ── Helpers ───────────────────────────────────────────────────────────────────

static void pass(const char* name) {
    std::printf("  PASS: %s\n", name);
}

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::exit(1);
    }
}

// ── Test 1: Basic cell calling — known metrics ────────────────────────────────

static void test_basic_cell_calling() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 500;
    cfg.min_tss_enrichment = 2.0;
    cfg.min_frac_in_peaks = 10;  // 10% FRIP

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"BC1", "BC2", "BC3", "BC4"};
    std::vector<uint64_t> frags = {1000, 200, 800, 600};
    std::vector<double> tss = {4.0, 1.5, 2.5, 0.5};
    std::vector<double> frip = {0.25, 0.30, 0.05, 0.20};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    // BC1: pass (frags=1000 ≥ 500, tss=4.0 ≥ 2.0, frip=0.25 ≥ 0.10)
    require(results[0].is_cell, "BC1 should be a cell");
    require(results[0].filter_reason == "pass", "BC1 reason=pass");

    // BC2: low_fragments (200 < 500) — fragments is priority filter
    require(!results[1].is_cell, "BC2 should not be a cell");
    require(results[1].filter_reason == "low_fragments", "BC2 reason=low_fragments");

    // BC3: low_frip (0.05 < 0.10) — frags and tss pass
    require(!results[2].is_cell, "BC3 should not be a cell");
    require(results[2].filter_reason == "low_frip", "BC3 reason=low_frip");

    // BC4: low_tss (0.5 < 2.0) — frags pass, tss fails (priority over frip)
    require(!results[3].is_cell, "BC4 should not be a cell");
    require(results[3].filter_reason == "low_tss", "BC4 reason=low_tss");

    require(summary.cells_called == 1, "1 cell called");
    require(summary.total_barcodes == 4, "4 barcodes total");

    pass("basic_cell_calling");
}

// ── Test 2: Auto-threshold via call_cells correctly separates cells from empties ─

static void test_auto_threshold_synthetic() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = true;
    cfg.min_tss_enrichment = 1.0;  // relax TSS so only fragment count filters
    cfg.min_frac_in_peaks = 0;     // relax FRIP

    AtacCellCaller caller(cfg);

    // 20 cells: fragments 50000 down to 31000 (step 1000)
    // 80 empties: fragments 50 down to 1 (cycling 50..1)
    // Gap between cells (31000) and empties (50) is enormous — auto threshold must land between them.
    std::vector<std::string> barcodes;
    std::vector<uint64_t> frags;
    std::vector<double> tss_v;
    std::vector<double> frip_v;

    for (int i = 0; i < 20; ++i) {
        barcodes.push_back("cell_" + std::to_string(i));
        frags.push_back(static_cast<uint64_t>(50000 - i * 1000));  // 50000..31000
        tss_v.push_back(5.0);
        frip_v.push_back(0.5);
    }
    for (int i = 0; i < 80; ++i) {
        barcodes.push_back("empty_" + std::to_string(i));
        frags.push_back(static_cast<uint64_t>(50 - (i % 50)));  // 50..1 (cycling)
        tss_v.push_back(5.0);
        frip_v.push_back(0.5);
    }

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss_v, frip_v, results);

    // All 20 cell barcodes must be called as cells; empties must not be
    require(summary.cells_called == 20, "auto threshold: 20 cells called");
    require(summary.total_barcodes == 100, "auto threshold: 100 barcodes");

    int cell_pass = 0, empty_pass = 0;
    for (int i = 0; i < 20; ++i)
        if (results[i].is_cell) ++cell_pass;
    for (int i = 20; i < 100; ++i)
        if (results[i].is_cell) ++empty_pass;

    require(cell_pass == 20, "auto threshold: all cells correctly classified");
    require(empty_pass == 0, "auto threshold: no empties misclassified");

    pass("auto_threshold_synthetic");
}

// ── Test 3: All barcodes pass → 100% cells ────────────────────────────────────

static void test_all_pass() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 100;
    cfg.min_tss_enrichment = 1.0;
    cfg.min_frac_in_peaks = 5;

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"A", "B", "C"};
    std::vector<uint64_t> frags = {500, 1000, 2000};
    std::vector<double> tss = {3.0, 5.0, 4.0};
    std::vector<double> frip = {0.4, 0.5, 0.6};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 3, "all 3 pass");
    require(summary.filtered_low_tss == 0, "no low_tss");
    require(summary.filtered_low_fragments == 0, "no low_fragments");
    require(summary.filtered_low_frip == 0, "no low_frip");
    for (auto& r : results)
        require(r.is_cell, "each barcode is_cell");

    pass("all_pass");
}

// ── Test 4: All barcodes fail → 0 cells ──────────────────────────────────────

static void test_all_fail() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 1000;
    cfg.min_tss_enrichment = 5.0;
    cfg.min_frac_in_peaks = 50;

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"X", "Y"};
    std::vector<uint64_t> frags = {50, 80};
    std::vector<double> tss = {1.0, 1.5};
    std::vector<double> frip = {0.05, 0.10};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 0, "0 cells called");
    for (auto& r : results)
        require(!r.is_cell, "no barcode is_cell");

    pass("all_fail");
}

// ── Test 5: Mixed quality — correct per-filter counts ────────────────────────

static void test_mixed_quality() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 500;
    cfg.min_tss_enrichment = 2.0;
    cfg.min_frac_in_peaks = 15;  // 15%

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"G", "H", "I", "J", "K"};
    //                                    pass  low_frag  low_tss  low_frip  pass
    std::vector<uint64_t> frags = {600, 100, 700, 800, 900};
    std::vector<double> tss = {3.0, 3.0, 1.0, 3.0, 4.0};
    std::vector<double> frip = {0.20, 0.20, 0.20, 0.05, 0.25};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 2, "2 cells called");
    require(summary.filtered_low_fragments == 1, "1 low_fragments");
    require(summary.filtered_low_tss == 1, "1 low_tss");
    require(summary.filtered_low_frip == 1, "1 low_frip");
    require(results[0].is_cell, "G is cell");
    require(!results[1].is_cell, "H not cell");
    require(!results[2].is_cell, "I not cell");
    require(!results[3].is_cell, "J not cell");
    require(results[4].is_cell, "K is cell");

    pass("mixed_quality");
}

// ── Test 6: Edge case — single barcode ────────────────────────────────────────

static void test_single_barcode() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 100;
    cfg.min_tss_enrichment = 2.0;
    cfg.min_frac_in_peaks = 10;

    AtacCellCaller caller(cfg);

    {
        std::vector<AtacCellCaller::CellCallResult> results;
        auto s = caller.call_cells({"BC"}, {500}, {3.0}, {0.3}, results);
        require(s.cells_called == 1, "single barcode: cell");
        require(results[0].is_cell, "single barcode is_cell");
    }
    {
        std::vector<AtacCellCaller::CellCallResult> results;
        auto s = caller.call_cells({"BC"}, {50}, {3.0}, {0.3}, results);
        require(s.cells_called == 0, "single barcode: empty");
        require(!results[0].is_cell, "single barcode not is_cell");
    }

    pass("single_barcode");
}

// ── Test 7: TSS filter alone (fragments and FRIP both pass) ──────────────────

static void test_tss_filter_alone() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 100;
    cfg.min_tss_enrichment = 3.0;
    cfg.min_frac_in_peaks = 5;

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"N1", "N2"};
    std::vector<uint64_t> frags = {1000, 1000};
    std::vector<double> tss = {4.0, 1.5};  // N2 fails TSS
    std::vector<double> frip = {0.30, 0.30};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 1, "1 cell (TSS filter)");
    require(summary.filtered_low_tss == 1, "1 filtered by TSS");
    require(results[0].filter_reason == "pass", "N1 passes");
    require(results[1].filter_reason == "low_tss", "N2 low_tss");

    pass("tss_filter_alone");
}

// ── Test 8: Fragment filter alone (TSS and FRIP both pass) ───────────────────

static void test_fragment_filter_alone() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 1000;
    cfg.min_tss_enrichment = 1.5;
    cfg.min_frac_in_peaks = 5;

    AtacCellCaller caller(cfg);

    std::vector<std::string> barcodes = {"M1", "M2"};
    std::vector<uint64_t> frags = {2000, 400};  // M2 fails fragments
    std::vector<double> tss = {5.0, 5.0};
    std::vector<double> frip = {0.50, 0.50};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 1, "1 cell (fragment filter)");
    require(summary.filtered_low_fragments == 1, "1 filtered by fragments");
    require(results[0].filter_reason == "pass", "M1 passes");
    require(results[1].filter_reason == "low_fragments", "M2 low_fragments");

    pass("fragment_filter_alone");
}

// ── Test 9: Median statistics computed correctly ──────────────────────────────

static void test_median_statistics() {
    AtacCellCaller::Config cfg;
    cfg.use_auto_threshold = false;
    cfg.min_unique_fragments = 100;
    cfg.min_tss_enrichment = 1.0;
    cfg.min_frac_in_peaks = 5;

    AtacCellCaller caller(cfg);

    // 4 cells with TSS = 2.0, 3.0, 4.0, 5.0 → median = 3.5
    // fragments = 200, 400, 600, 800 → median = 500
    std::vector<std::string> barcodes = {"P1", "P2", "P3", "P4"};
    std::vector<uint64_t> frags = {200, 400, 600, 800};
    std::vector<double> tss = {2.0, 3.0, 4.0, 5.0};
    std::vector<double> frip = {0.3, 0.3, 0.3, 0.3};

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    require(summary.cells_called == 4, "4 cells for median test");
    require(std::abs(summary.median_tss_cells - 3.5) < 1e-9, "median TSS = 3.5");
    require(std::abs(summary.median_fragments_cells - 500.0) < 1e-9, "median frags = 500");

    pass("median_statistics");
}

// ── Test 10: Empty input → no crash, zero cells ───────────────────────────────

static void test_empty_input() {
    AtacCellCaller caller;

    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells({}, {}, {}, {}, results);

    require(summary.total_barcodes == 0, "empty: 0 barcodes");
    require(summary.cells_called == 0, "empty: 0 cells");
    require(results.empty(), "empty: no results");

    pass("empty_input");
}

// ── Test 11: T_OTSU_BIMODAL — bimodal PBMC 500 ATAC simulation ──────────────

static void test_otsu_bimodal() {
    std::mt19937 rng(42);

    const int n_cells = 500;
    const int n_empties = 3000;
    const int N = n_cells + n_empties;

    std::vector<std::string> barcodes(N);
    std::vector<uint64_t> frags(N);
    std::vector<double> tss(N);
    std::vector<double> frip(N);

    // Cell barcodes: log-normal centered at log10(5000)≈3.7, σ=0.2
    std::normal_distribution<double> cell_log_frag(3.7, 0.2);
    std::normal_distribution<double> cell_tss_dist(0.25, 0.05);
    std::normal_distribution<double> cell_frip_dist(0.30, 0.05);

    for (int i = 0; i < n_cells; ++i) {
        barcodes[i] = "cell_" + std::to_string(i);
        double lf = cell_log_frag(rng);
        frags[i] = std::max(uint64_t(1), static_cast<uint64_t>(std::pow(10.0, lf)));
        tss[i] = std::max(0.0, cell_tss_dist(rng));
        frip[i] = std::max(0.0, cell_frip_dist(rng));
    }

    // Empty barcodes: log-normal centered at log10(50)≈1.7, σ=0.3
    std::normal_distribution<double> empty_log_frag(1.7, 0.3);
    std::normal_distribution<double> empty_tss_dist(0.03, 0.01);
    std::normal_distribution<double> empty_frip_dist(0.05, 0.02);

    for (int i = 0; i < n_empties; ++i) {
        int idx = n_cells + i;
        barcodes[idx] = "empty_" + std::to_string(i);
        double lf = empty_log_frag(rng);
        frags[idx] = std::max(uint64_t(1), static_cast<uint64_t>(std::pow(10.0, lf)));
        tss[idx] = std::max(0.0, empty_tss_dist(rng));
        frip[idx] = std::max(0.0, empty_frip_dist(rng));
    }

    AtacCellCaller caller;  // default config
    std::vector<AtacCellCaller::CellCallResult> results;
    auto summary = caller.call_cells(barcodes, frags, tss, frip, results);

    // Count precision and recall
    int true_pos = 0, false_pos = 0, false_neg = 0;
    for (int i = 0; i < N; ++i) {
        bool is_true_cell = (i < n_cells);
        if (results[i].is_cell && is_true_cell) ++true_pos;
        if (results[i].is_cell && !is_true_cell) ++false_pos;
        if (!results[i].is_cell && is_true_cell) ++false_neg;
    }

    double precision = (true_pos + false_pos > 0)
                           ? static_cast<double>(true_pos) / (true_pos + false_pos)
                           : 0.0;
    double recall = (n_cells > 0) ? static_cast<double>(true_pos) / n_cells : 0.0;

    require(summary.cells_called >= 400 && summary.cells_called <= 600,
            "T_OTSU_BIMODAL: cells_called in [400, 600]");
    require(precision >= 0.90,
            "T_OTSU_BIMODAL: precision >= 0.90");
    require(recall >= 0.80,
            "T_OTSU_BIMODAL: recall >= 0.80");

    pass("T_OTSU_BIMODAL");
}

// ── Test 12: T_OTSU_THRESHOLD_SEPARATES — auto_fragment_threshold finds valley

static void test_otsu_threshold_separates() {
    std::mt19937 rng(123);

    std::vector<uint64_t> counts;
    counts.reserve(3500);

    // 500 values in [3000, 8000]
    std::uniform_int_distribution<uint64_t> cell_dist(3000, 8000);
    for (int i = 0; i < 500; ++i)
        counts.push_back(cell_dist(rng));

    // 3000 values in [20, 200]
    std::uniform_int_distribution<uint64_t> empty_dist(20, 200);
    for (int i = 0; i < 3000; ++i)
        counts.push_back(empty_dist(rng));

    AtacCellCaller::Config cfg;
    cfg.min_unique_fragments = 1;  // don't clamp for this test
    AtacCellCaller caller(cfg);

    uint32_t threshold = caller.auto_fragment_threshold(counts);

    require(threshold >= 200 && threshold <= 3000,
            "T_OTSU_THRESHOLD_SEPARATES: threshold in [200, 3000]");

    pass("T_OTSU_THRESHOLD_SEPARATES");
}

// ── Test 13: T_UNIMODAL_FALLBACK — no bimodal split → min_unique_fragments ──

static void test_unimodal_fallback() {
    std::mt19937 rng(99);
    std::uniform_int_distribution<uint64_t> dist(800, 1200);

    std::vector<uint64_t> counts(1000);
    for (auto& c : counts) c = dist(rng);

    AtacCellCaller::Config cfg;
    cfg.min_unique_fragments = 500;
    AtacCellCaller caller(cfg);

    uint32_t threshold = caller.auto_fragment_threshold(counts);

    require(threshold == 500,
            "T_UNIMODAL_FALLBACK: threshold == 500 (min_unique_fragments)");

    pass("T_UNIMODAL_FALLBACK");
}

// ── Test 14: T_MIN_FLOOR — valley below min_unique_fragments is clamped ──────

static void test_min_floor() {
    std::vector<uint64_t> counts;
    // 100 cells at ~1000
    for (int i = 0; i < 100; ++i) counts.push_back(1000);
    // 500 empties at ~10
    for (int i = 0; i < 500; ++i) counts.push_back(10);

    AtacCellCaller::Config cfg;
    cfg.min_unique_fragments = 500;
    AtacCellCaller caller(cfg);

    uint32_t threshold = caller.auto_fragment_threshold(counts);

    require(threshold >= 500,
            "T_MIN_FLOOR: threshold >= 500 (clamped to min_unique_fragments)");

    pass("T_MIN_FLOOR");
}

// ── Test 15: T_EMPTY_AND_EDGE — degenerate inputs ───────────────────────────

static void test_empty_and_edge() {
    AtacCellCaller::Config cfg;
    cfg.min_unique_fragments = 500;
    AtacCellCaller caller(cfg);

    // Empty input
    {
        uint32_t t = caller.auto_fragment_threshold({});
        require(t == 500, "T_EMPTY_AND_EDGE: empty → 500");
    }

    // Single barcode above min
    {
        uint32_t t = caller.auto_fragment_threshold({1000});
        require(t >= 500, "T_EMPTY_AND_EDGE: single barcode(1000) >= 500");
    }

    // Single barcode below min
    {
        uint32_t t = caller.auto_fragment_threshold({100});
        require(t >= 500, "T_EMPTY_AND_EDGE: single barcode(100) >= 500");
    }

    // All zeros
    {
        std::vector<uint64_t> zeros(50, 0);
        uint32_t t = caller.auto_fragment_threshold(zeros);
        require(t == 500, "T_EMPTY_AND_EDGE: all zeros → 500");
    }

    pass("T_EMPTY_AND_EDGE");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    std::printf("=== test_atac_cell_caller ===\n");

    test_basic_cell_calling();
    test_auto_threshold_synthetic();
    test_all_pass();
    test_all_fail();
    test_mixed_quality();
    test_single_barcode();
    test_tss_filter_alone();
    test_fragment_filter_alone();
    test_median_statistics();
    test_empty_input();
    test_otsu_bimodal();
    test_otsu_threshold_separates();
    test_unimodal_fallback();
    test_min_floor();
    test_empty_and_edge();

    std::printf("All tests passed.\n");
    return 0;
}
