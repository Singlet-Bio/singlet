// test_chipseq.cpp — Unit tests for G-CHIPSEQ: chipseq.h
//
// Tests:
//   1.  compute_pbc1: zero total → 0.0
//   2.  compute_pbc1: all unique → 1.0
//   3.  compute_pbc1: 50% unique → 0.5
//   4.  compute_pbc2: zero two-read positions → large value
//   5.  compute_pbc2: equal counts → 1.0
//   6.  compute_pbc2: 10:2 ratio → 5.0
//   7.  Lorenz curve: uniform coverage → diagonal (AUC ≈ 0.5)
//   8.  Lorenz curve: single-position spike → curve hugs bottom (AUC < 0.5)
//   9.  Lorenz curve: empty input → all zeros
//  10.  lorenz_auc: linear diagonal → 0.5
//  11.  strand_cross_correlation: identical fwd == rev → cc[0] == 1.0
//  12.  strand_cross_correlation: empty input → empty result
//  13.  compute_chipseq_qc: FRiP calculation
//  14.  compute_chipseq_qc: pbc1 matches expected
//  15.  write_chipseq_qc: file created with required keys

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/chipseq.h"

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
static bool near(double a, double b, double eps = 1e-9) { return std::abs(a - b) < eps; }

// ── Test 1: pbc1 zero total ────────────────────────────────────────────────
static void test_pbc1_zero_total() {
    double r = compute_pbc1(0, 0);
    if (near(r, 0.0))
        PASS("pbc1_zero_total");
    else
        FAIL("pbc1_zero_total", "expected 0.0");
}

// ── Test 2: pbc1 all unique ────────────────────────────────────────────────
static void test_pbc1_all_unique() {
    double r = compute_pbc1(100, 100);
    if (near(r, 1.0))
        PASS("pbc1_all_unique");
    else
        FAIL("pbc1_all_unique", "expected 1.0");
}

// ── Test 3: pbc1 half unique ───────────────────────────────────────────────
static void test_pbc1_half() {
    double r = compute_pbc1(50, 100);
    if (near(r, 0.5))
        PASS("pbc1_half");
    else
        FAIL("pbc1_half", "expected 0.5");
}

// ── Test 4: pbc2 zero two-read positions ──────────────────────────────────
static void test_pbc2_zero_denom_nonzero_num() {
    double r = compute_pbc2(10, 0);
    if (r > 1e6)
        PASS("pbc2_zero_denom_nonzero_num");
    else
        FAIL("pbc2_zero_denom_nonzero_num", "expected large value");
}

// ── Test 5: pbc2 equal counts → 1.0 ──────────────────────────────────────
static void test_pbc2_equal() {
    double r = compute_pbc2(5, 5);
    if (near(r, 1.0))
        PASS("pbc2_equal");
    else
        FAIL("pbc2_equal", "expected 1.0");
}

// ── Test 6: pbc2 10:2 ratio → 5.0 ────────────────────────────────────────
static void test_pbc2_ratio() {
    double r = compute_pbc2(10, 2);
    if (near(r, 5.0))
        PASS("pbc2_ratio");
    else
        FAIL("pbc2_ratio", "expected 5.0");
}

// ── Test 7: Lorenz curve — uniform coverage ────────────────────────────────
// With constant coverage the Lorenz curve is the diagonal: y[b] = b/N.
// AUC should be very close to 0.5.
static void test_lorenz_uniform() {
    std::vector<uint32_t> cov(1000, 10);  // all positions have count 10
    auto curve = compute_lorenz_curve(cov, 100);
    double auc = lorenz_auc(curve);
    // AUC of the identity line = 0.5
    if (auc > 0.45 && auc < 0.55)
        PASS("lorenz_uniform");
    else
        FAIL("lorenz_uniform", "AUC should be ~0.5 for uniform");
}

// ── Test 8: Lorenz curve — single spike (high enrichment) ─────────────────
// One position with count 10000, rest with count 1 → highly concentrated.
// AUC should be much less than 0.5 (signal concentrated at one position).
static void test_lorenz_spike() {
    std::vector<uint32_t> cov(1000, 1);
    cov[500] = 10000;  // single hot spot
    auto curve = compute_lorenz_curve(cov, 100);
    double auc = lorenz_auc(curve);
    if (auc < 0.45)
        PASS("lorenz_spike");
    else
        FAIL("lorenz_spike", "AUC should be < 0.45 for concentrated signal");
}

// ── Test 9: Lorenz curve — empty input → all zeros ────────────────────────
static void test_lorenz_empty() {
    std::vector<uint32_t> cov;
    auto curve = compute_lorenz_curve(cov, 100);
    bool all_zero = true;
    for (auto v : curve)
        if (v != 0.0) {
            all_zero = false;
            break;
        }
    if (all_zero && curve.size() == 101)
        PASS("lorenz_empty");
    else
        FAIL("lorenz_empty", "expected 101 zeros");
}

// ── Test 10: lorenz_auc diagonal → 0.5 ────────────────────────────────────
static void test_lorenz_auc_diagonal() {
    // Manually build a perfect diagonal: points = 0, 0.01, 0.02, …, 1.0
    std::vector<double> diag(101);
    for (int i = 0; i <= 100; ++i) diag[i] = static_cast<double>(i) / 100.0;
    double auc = lorenz_auc(diag);
    if (near(auc, 0.5, 1e-6))
        PASS("lorenz_auc_diagonal");
    else
        FAIL("lorenz_auc_diagonal", "expected 0.5");
}

// ── Test 11: cross-correlation identical fwd==rev ─────────────────────────
static void test_cc_identical() {
    std::vector<uint32_t> sig(200);
    for (size_t i = 0; i < sig.size(); i++) sig[i] = static_cast<uint32_t>((i % 10) + 1);
    auto cc = strand_cross_correlation(sig, sig, 50);
    // At shift 0, identical vectors → Pearson r = 1.0
    if (!cc.empty() && near(cc[0], 1.0, 1e-9))
        PASS("cc_identical");
    else
        FAIL("cc_identical", "expected cc[0] == 1.0 for identical vectors");
}

// ── Test 12: cross-correlation empty input ───────────────────────────────
static void test_cc_empty() {
    std::vector<uint32_t> empty_vec;
    auto cc = strand_cross_correlation(empty_vec, empty_vec, 50);
    if (cc.empty())
        PASS("cc_empty");
    else
        FAIL("cc_empty", "expected empty result for empty input");
}

// ── Test 13: compute_chipseq_qc FRiP ─────────────────────────────────────
static void test_chipseq_qc_frip() {
    std::vector<uint32_t> fwd(500, 1), rev(500, 1);
    auto qc = compute_chipseq_qc(fwd, rev, 1000, 900, 500);
    if (near(qc.frip, 0.5, 1e-9))
        PASS("chipseq_qc_frip");
    else
        FAIL("chipseq_qc_frip", "expected frip = reads_in_peaks/total = 0.5");
}

// ── Test 14: compute_chipseq_qc PBC1 ─────────────────────────────────────
static void test_chipseq_qc_pbc1() {
    std::vector<uint32_t> fwd(100, 2), rev(100, 2);
    auto qc = compute_chipseq_qc(fwd, rev, 200, 180, 50);
    if (near(qc.pbc1, 0.9, 1e-9))
        PASS("chipseq_qc_pbc1");
    else
        FAIL("chipseq_qc_pbc1", "expected pbc1 = 180/200 = 0.9");
}

// ── Test 15: write_chipseq_qc JSON output ────────────────────────────────
static void test_write_chipseq_qc() {
    ChipSeqQC qc;
    qc.frip = 0.42;
    qc.nsc = 1.15;
    qc.rsc = 1.05;
    qc.pbc1 = 0.92;
    qc.pbc2 = 4.5;
    qc.total_reads = 20000000;
    qc.unique_reads = 18000000;
    qc.reads_in_peaks = 8400000;
    qc.auc_lorenz = 0.38;

    const std::string path = "/tmp/test_chipseq_qc.json";
    bool ok = write_chipseq_qc(path, qc);
    if (!ok) {
        FAIL("write_chipseq_qc", "write_chipseq_qc returned false");
        return;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
        FAIL("write_chipseq_qc", "file not created");
        return;
    }
    std::string content((std::istreambuf_iterator<char>(f)), {});

    // Check required fields
    bool has_frip = content.find("\"frip\"") != std::string::npos;
    bool has_nsc = content.find("\"nsc\"") != std::string::npos;
    bool has_rsc = content.find("\"rsc\"") != std::string::npos;
    bool has_pbc1 = content.find("\"pbc1\"") != std::string::npos;
    bool has_pbc2 = content.find("\"pbc2\"") != std::string::npos;
    bool has_total = content.find("\"total_reads\"") != std::string::npos;
    bool has_auc = content.find("\"auc_lorenz\"") != std::string::npos;

    if (has_frip && has_nsc && has_rsc && has_pbc1 && has_pbc2 && has_total && has_auc)
        PASS("write_chipseq_qc");
    else
        FAIL("write_chipseq_qc", "missing required JSON keys");
}

int main() {
    test_pbc1_zero_total();
    test_pbc1_all_unique();
    test_pbc1_half();
    test_pbc2_zero_denom_nonzero_num();
    test_pbc2_equal();
    test_pbc2_ratio();
    test_lorenz_uniform();
    test_lorenz_spike();
    test_lorenz_empty();
    test_lorenz_auc_diagonal();
    test_cc_identical();
    test_cc_empty();
    test_chipseq_qc_frip();
    test_chipseq_qc_pbc1();
    test_write_chipseq_qc();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
