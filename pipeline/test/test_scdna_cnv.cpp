// test_scdna_cnv.cpp — Unit tests for G-SCDNA: scdna_cnv.h
//
// Tests:
//   1.  compute_mapd: known values → expected result
//   2.  compute_mapd: single bin → 0.0
//   3.  compute_mapd: empty → 0.0
//   4.  compute_mapd: constant bins → 0.0 (no pairwise difference)
//   5.  estimate_ploidy: diploid bins → 2.0
//   6.  estimate_ploidy: tetraploid bins → 4.0
//   7.  estimate_ploidy: empty → 0.0
//   8.  gc_correct_bins: flat GC → no change (uniform correction)
//   9.  gc_correct_bins: GC bias present → correction applied
//  10.  gc_correct_bins: empty → empty result
//  11.  compute_cell_cnv_qc: coverage computed correctly
//  12.  write_cnv_qc: JSON created with barcode + required keys
//  13.  write_cnv_qc: multiple cells written as JSON array

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet-pileup/scdna_cnv.h"

using namespace singlet;

static int passed = 0;
static int failed = 0;

static void PASS(const char* name) {
    ++passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}
static bool near(double a, double b, double eps = 1e-6) { return std::abs(a - b) < eps; }

// ── Test 1: MAPD known values ─────────────────────────────────────────────
// bins: [1, 2, 4, 8]
// diffs: |log2(2+1)-log2(1+1)|, |log2(4+1)-log2(2+1)|, |log2(8+1)-log2(4+1)|
//      = |1.585-1.0|, |2.322-1.585|, |3.170-2.322| ≈ 0.585, 0.737, 0.848
// sorted: 0.585, 0.737, 0.848 → median = 0.737
static void test_mapd_known() {
    std::vector<int32_t> bins = {1, 2, 4, 8};
    double mapd = compute_mapd(bins);
    // Expected median of 3 diffs (index 1 after sort)
    double d0 = std::abs(std::log2(3.0) - std::log2(2.0));
    double d1 = std::abs(std::log2(5.0) - std::log2(3.0));
    double d2 = std::abs(std::log2(9.0) - std::log2(5.0));
    // Sort: d0 ≈ 0.585, d1 ≈ 0.737, d2 ≈ 0.848 → median = d1
    (void)d0;
    (void)d2;
    if (near(mapd, d1, 1e-6))
        PASS("mapd_known");
    else
        FAIL("mapd_known", "expected median of sorted pairwise diffs");
}

// ── Test 2: MAPD single bin → 0.0 ────────────────────────────────────────
static void test_mapd_single() {
    std::vector<int32_t> bins = {42};
    if (near(compute_mapd(bins), 0.0))
        PASS("mapd_single");
    else
        FAIL("mapd_single", "expected 0.0 for single bin");
}

// ── Test 3: MAPD empty → 0.0 ─────────────────────────────────────────────
static void test_mapd_empty() {
    std::vector<int32_t> bins;
    if (near(compute_mapd(bins), 0.0))
        PASS("mapd_empty");
    else
        FAIL("mapd_empty", "expected 0.0 for empty bins");
}

// ── Test 4: MAPD constant bins → 0.0 ─────────────────────────────────────
static void test_mapd_constant() {
    std::vector<int32_t> bins(50, 100);
    if (near(compute_mapd(bins), 0.0))
        PASS("mapd_constant");
    else
        FAIL("mapd_constant", "expected 0.0 for constant bins");
}

// ── Test 5: ploidy estimation — diploid ───────────────────────────────────
// If all bins have count ≈ expected_diploid_count, ploidy = 2
static void test_ploidy_diploid() {
    std::vector<int32_t> bins(100, 50);  // mean = 50, expected_diploid = 50
    double ploidy = estimate_ploidy(bins, 50.0);
    if (near(ploidy, 2.0, 0.5))
        PASS("ploidy_diploid");
    else
        FAIL("ploidy_diploid", "expected ploidy ≈ 2.0");
}

// ── Test 6: ploidy estimation — tetraploid ────────────────────────────────
// Bins count ~100, expected_diploid=50 → ratio=100/50*2=4
static void test_ploidy_tetraploid() {
    std::vector<int32_t> bins(100, 100);  // count = 2x diploid → CN=4
    double ploidy = estimate_ploidy(bins, 50.0);
    if (near(ploidy, 4.0, 0.5))
        PASS("ploidy_tetraploid");
    else
        FAIL("ploidy_tetraploid", "expected ploidy ≈ 4.0");
}

// ── Test 7: ploidy estimation — empty ────────────────────────────────────
static void test_ploidy_empty() {
    std::vector<int32_t> bins;
    double ploidy = estimate_ploidy(bins, 50.0);
    if (near(ploidy, 0.0))
        PASS("ploidy_empty");
    else
        FAIL("ploidy_empty", "expected 0.0 for empty bins");
}

// ── Test 8: GC correction — flat GC → no change ──────────────────────────
// If all bins have identical GC content, the linear fit slope is 0,
// the correction factor is uniform, and relative values are preserved.
static void test_gc_correct_flat_gc() {
    std::vector<int32_t> bins = {10, 20, 30, 40, 50};
    std::vector<double> gc = {0.5, 0.5, 0.5, 0.5, 0.5};  // all same GC
    auto corrected = gc_correct_bins(bins, gc);
    if (corrected.size() != 5) {
        FAIL("gc_correct_flat_gc", "wrong size");
        return;
    }

    // With slope=0, the correction is uniform (all scale by the same factor).
    // Check that ratios between bins are preserved.
    // corrected[0] / corrected[1] should equal 10.0 / 20.0 = 0.5
    double ratio_raw = 10.0 / 20.0;
    double ratio_cor = corrected[0] / corrected[1];
    if (near(ratio_cor, ratio_raw, 1e-6))
        PASS("gc_correct_flat_gc");
    else
        FAIL("gc_correct_flat_gc", "expected preserved ratios for uniform GC");
}

// ── Test 9: GC correction — bias present ─────────────────────────────────
// Construct bins where count correlates perfectly with GC.
// After correction, the GC-induced trend should be reduced.
static void test_gc_correct_bias() {
    // 10 bins with GC linearly increasing from 0.3 to 0.7
    // Counts = 10 * gc (perfect GC bias) → after correction should be flat
    const int n = 10;
    std::vector<int32_t> bins(n);
    std::vector<double> gc(n);
    for (int i = 0; i < n; ++i) {
        gc[i] = 0.3 + 0.4 * static_cast<double>(i) / static_cast<double>(n - 1);
        bins[i] = static_cast<int32_t>(std::round(gc[i] * 100.0));
    }
    auto corrected = gc_correct_bins(bins, gc);
    if (corrected.size() != static_cast<size_t>(n)) {
        FAIL("gc_correct_bias", "wrong size");
        return;
    }
    // After correction the variance of corrected values should be less than
    // variance of raw values (GC trend removed).
    double raw_mean = 0.0, cor_mean = 0.0;
    for (int i = 0; i < n; ++i) {
        raw_mean += bins[i];
        cor_mean += corrected[i];
    }
    raw_mean /= n;
    cor_mean /= n;
    double raw_var = 0.0, cor_var = 0.0;
    for (int i = 0; i < n; ++i) {
        raw_var += (bins[i] - raw_mean) * (bins[i] - raw_mean);
        cor_var += (corrected[i] - cor_mean) * (corrected[i] - cor_mean);
    }
    if (cor_var < raw_var)
        PASS("gc_correct_bias");
    else
        FAIL("gc_correct_bias", "expected correction to reduce GC-driven variance");
}

// ── Test 10: GC correction — empty input ─────────────────────────────────
static void test_gc_correct_empty() {
    std::vector<int32_t> bins;
    std::vector<double> gc;
    auto corrected = gc_correct_bins(bins, gc);
    if (corrected.empty())
        PASS("gc_correct_empty");
    else
        FAIL("gc_correct_empty", "expected empty output for empty input");
}

// ── Test 11: compute_cell_cnv_qc — coverage ───────────────────────────────
static void test_cell_cnv_qc_coverage() {
    std::vector<int32_t> bins = {10, 20, 30, 40};  // mean = 25
    auto qc = compute_cell_cnv_qc(bins, 1000, 950);
    if (near(qc.coverage, 25.0, 1e-9) &&
        qc.total_reads == 1000 &&
        qc.unique_reads == 950)
        PASS("cell_cnv_qc_coverage");
    else
        FAIL("cell_cnv_qc_coverage", "unexpected coverage or read counts");
}

// ── Test 12: write_cnv_qc — single cell JSON ─────────────────────────────
static void test_write_cnv_qc_single() {
    CnvCellQC qc;
    qc.mapd = 0.15;
    qc.ploidy_estimate = 2.0;
    qc.total_reads = 5000;
    qc.unique_reads = 4800;
    qc.coverage = 12.5;

    const std::string path = "/tmp/test_cnv_qc_single.json";
    bool ok = write_cnv_qc(path, {qc}, {"ACGT-1"});
    if (!ok) {
        FAIL("write_cnv_qc_single", "write returned false");
        return;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
        FAIL("write_cnv_qc_single", "file not created");
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), {});

    bool has_barcode = content.find("\"barcode\"") != std::string::npos;
    bool has_mapd = content.find("\"mapd\"") != std::string::npos;
    bool has_ploidy = content.find("\"ploidy_estimate\"") != std::string::npos;
    bool has_coverage = content.find("\"coverage\"") != std::string::npos;
    bool has_bc_val = content.find("ACGT-1") != std::string::npos;

    if (has_barcode && has_mapd && has_ploidy && has_coverage && has_bc_val)
        PASS("write_cnv_qc_single");
    else
        FAIL("write_cnv_qc_single", "missing required JSON keys or barcode");
}

// ── Test 13: write_cnv_qc — multiple cells ───────────────────────────────
static void test_write_cnv_qc_multi() {
    std::vector<CnvCellQC> cells(3);
    std::vector<std::string> barcodes = {"BC1", "BC2", "BC3"};
    cells[0].mapd = 0.1;
    cells[1].mapd = 0.2;
    cells[2].mapd = 0.3;

    const std::string path = "/tmp/test_cnv_qc_multi.json";
    bool ok = write_cnv_qc(path, cells, barcodes);
    if (!ok) {
        FAIL("write_cnv_qc_multi", "write returned false");
        return;
    }

    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)), {});
    // Should start with '[' (JSON array) and contain all 3 barcodes
    bool is_array = content.front() == '[';
    bool has_bc1 = content.find("BC1") != std::string::npos;
    bool has_bc2 = content.find("BC2") != std::string::npos;
    bool has_bc3 = content.find("BC3") != std::string::npos;

    if (is_array && has_bc1 && has_bc2 && has_bc3)
        PASS("write_cnv_qc_multi");
    else
        FAIL("write_cnv_qc_multi", "expected JSON array with 3 barcodes");
}

int main() {
    test_mapd_known();
    test_mapd_single();
    test_mapd_empty();
    test_mapd_constant();
    test_ploidy_diploid();
    test_ploidy_tetraploid();
    test_ploidy_empty();
    test_gc_correct_flat_gc();
    test_gc_correct_bias();
    test_gc_correct_empty();
    test_cell_cnv_qc_coverage();
    test_write_cnv_qc_single();
    test_write_cnv_qc_multi();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
