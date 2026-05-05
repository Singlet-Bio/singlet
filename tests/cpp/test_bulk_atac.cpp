// test_bulk_atac.cpp — Unit tests for G-BULK-ATAC: bulk_atac.h
//
// Tests:
//   1. is_bulk_atac() — cblen=0, paired-end, no assay_type → true
//   2. is_bulk_atac() — cblen>0 (has barcodes) → false
//   3. is_bulk_atac() — single-end → false
//   4. is_bulk_atac() — explicit assay_type "ATAC" → true
//   5. is_bulk_atac() — explicit assay_type "10xv3" → false
//   6. Fragment size classification: NFR / mono / di fractions
//   7. Median fragment size (odd count)
//   8. Median fragment size (even count)
//   9. Library complexity (Lander-Waterman, low saturation)
//  10. Library complexity (saturated: unique == total)
//  11. Mitochondrial fragment counting
//  12. QC JSON output: all required fields present

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet-pileup/bulk_atac.h"

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

// Helper: build an ATACFragment with given coordinates
static ATACFragment make_frag(int32_t chrom, int32_t start, int32_t end) {
    ATACFragment f;
    f.chrom_idx = chrom;
    f.start = start;
    f.end = end;
    f.barcode_idx = 0;  // unused in bulk mode
    return f;
}

// ── Test 1: bulk detection — cblen=0, paired, no assay type ────────────────
static void test_detect_bulk_pe_no_bc() {
    bool r = is_bulk_atac(0, true, "");
    if (r)
        PASS("detect_bulk_pe_no_bc");
    else
        FAIL("detect_bulk_pe_no_bc", "expected true for cblen=0 paired-end");
}

// ── Test 2: not bulk — has barcodes ────────────────────────────────────────
static void test_detect_has_barcodes() {
    bool r = is_bulk_atac(16, true, "");
    if (!r)
        PASS("detect_has_barcodes");
    else
        FAIL("detect_has_barcodes", "expected false when cblen>0");
}

// ── Test 3: not bulk — single-end ──────────────────────────────────────────
static void test_detect_single_end() {
    bool r = is_bulk_atac(0, false, "");
    if (!r)
        PASS("detect_single_end");
    else
        FAIL("detect_single_end", "expected false for single-end");
}

// ── Test 4: explicit assay_type "ATAC" → true ──────────────────────────────
static void test_detect_explicit_atac() {
    bool r = is_bulk_atac(0, true, "ATAC");
    if (r)
        PASS("detect_explicit_atac");
    else
        FAIL("detect_explicit_atac", "expected true for assay_type=ATAC");
}

// ── Test 5: explicit assay_type "10xv3" → false ────────────────────────────
static void test_detect_explicit_10xv3() {
    bool r = is_bulk_atac(0, true, "10xv3");
    if (!r)
        PASS("detect_explicit_10xv3");
    else
        FAIL("detect_explicit_10xv3", "expected false for assay_type=10xv3");
}

// ── Test 6: fragment size classification ───────────────────────────────────
// Build 6 fragments: 2 NFR (<147), 2 mono (147-294), 2 di (294-441)
static void test_size_class_fractions() {
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 100),  // NFR: 100 bp
        make_frag(0, 0, 120),  // NFR: 120 bp
        make_frag(0, 0, 200),  // mono: 200 bp
        make_frag(0, 0, 250),  // mono: 250 bp
        make_frag(0, 0, 350),  // di: 350 bp
        make_frag(0, 0, 400),  // di: 400 bp
    };

    BulkAtacQC qc = compute_bulk_atac_qc(frags, 6);

    const double eps = 1e-9;
    bool ok = true;
    if (std::abs(qc.nfr_fraction - 2.0 / 6) > eps) ok = false;
    if (std::abs(qc.mono_fraction - 2.0 / 6) > eps) ok = false;
    if (std::abs(qc.di_fraction - 2.0 / 6) > eps) ok = false;

    if (ok)
        PASS("size_class_fractions");
    else
        FAIL("size_class_fractions", "fraction mismatch");
}

// ── Test 7: median fragment size (odd count) ───────────────────────────────
static void test_median_odd() {
    // Sizes: 100, 200, 300 → sorted median = 200
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 300),
        make_frag(0, 0, 100),
        make_frag(0, 0, 200),
    };

    BulkAtacQC qc = compute_bulk_atac_qc(frags, 3);
    if (std::abs(qc.median_fragment_size - 200.0) < 1e-9)
        PASS("median_odd");
    else
        FAIL("median_odd", "expected median=200");
}

// ── Test 8: median fragment size (even count) ──────────────────────────────
static void test_median_even() {
    // Sizes: 100, 200, 300, 400 → sorted median = (200+300)/2 = 250
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 400),
        make_frag(0, 0, 100),
        make_frag(0, 0, 300),
        make_frag(0, 0, 200),
    };

    BulkAtacQC qc = compute_bulk_atac_qc(frags, 4);
    if (std::abs(qc.median_fragment_size - 250.0) < 1e-9)
        PASS("median_even");
    else
        FAIL("median_even", "expected median=250");
}

// ── Test 9: library complexity (low saturation) ────────────────────────────
// With total_reads=1000 and unique_fragments=100 (10% saturation),
//   C = -1000 * ln(1 - 0.1) = -1000 * ln(0.9) ≈ 105.36
static void test_library_complexity_low_sat() {
    std::vector<ATACFragment> frags;
    frags.reserve(100);
    for (int i = 0; i < 100; ++i) {
        frags.push_back(make_frag(0, i * 10, i * 10 + 150));
    }

    BulkAtacQC qc = compute_bulk_atac_qc(frags, 1000);
    const double expected = -1000.0 * std::log(1.0 - 100.0 / 1000.0);
    if (std::abs(qc.library_complexity - expected) < 0.01)
        PASS("library_complexity_low_sat");
    else
        FAIL("library_complexity_low_sat", "Lander-Waterman mismatch");
}

// ── Test 10: library complexity (saturated) ────────────────────────────────
// When unique == total_reads, x_over_n = 1.0 → fallback to unique_fragments
static void test_library_complexity_saturated() {
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 200),
        make_frag(0, 200, 400),
    };

    // Pass total_reads == unique_fragments == 2 → saturated path
    BulkAtacQC qc = compute_bulk_atac_qc(frags, 2);
    if (qc.library_complexity == 2.0)
        PASS("library_complexity_saturated");
    else
        FAIL("library_complexity_saturated", "expected fallback to unique_fragments");
}

// ── Test 11: mitochondrial fragment counting ───────────────────────────────
static void test_chrm_filtering() {
    // chrom 0 = chr1, chrom 1 = chrM
    std::vector<ATACFragment> frags = {
        make_frag(0, 1000, 1200),  // chr1
        make_frag(1, 500, 700),    // chrM
        make_frag(1, 100, 300),    // chrM
    };

    std::vector<int32_t> mito_idxs = {1};
    BulkAtacQC qc = compute_bulk_atac_qc(frags, 3, mito_idxs);

    if (qc.mitochondrial_fragments == 2 && qc.total_fragments == 3)
        PASS("chrm_filtering");
    else
        FAIL("chrm_filtering", "expected 2 mito out of 3 total");
}

// ── Test 12: JSON output — all required fields present ─────────────────────
static void test_json_output() {
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 180),
        make_frag(0, 200, 450),
    };
    BulkAtacQC qc = compute_bulk_atac_qc(frags, 10);

    const std::string path = "/tmp/test_bulk_atac_qc.json";
    bool wrote = write_bulk_atac_qc(path, qc);
    if (!wrote) {
        FAIL("json_output", "write_bulk_atac_qc returned false");
        return;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
        FAIL("json_output", "cannot re-open JSON file");
        return;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    const std::string content = ss.str();

    // Check all required fields are present
    const std::vector<std::string> required_fields = {
        "total_fragments",
        "unique_fragments",
        "mitochondrial_fragments",
        "tss_enrichment",
        "frip",
        "median_fragment_size",
        "library_complexity",
        "nfr_fraction",
        "mono_fraction",
        "di_fraction",
    };

    bool ok = true;
    for (const auto& field : required_fields) {
        if (content.find(field) == std::string::npos) {
            std::cerr << "  missing field: " << field << "\n";
            ok = false;
        }
    }

    if (ok)
        PASS("json_output");
    else
        FAIL("json_output", "one or more required JSON fields missing");
}

// ── Main ───────────────────────────────────────────────────────────────────
int main() {
    test_detect_bulk_pe_no_bc();
    test_detect_has_barcodes();
    test_detect_single_end();
    test_detect_explicit_atac();
    test_detect_explicit_10xv3();
    test_size_class_fractions();
    test_median_odd();
    test_median_even();
    test_library_complexity_low_sat();
    test_library_complexity_saturated();
    test_chrm_filtering();
    test_json_output();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
