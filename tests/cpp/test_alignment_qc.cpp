// test_alignment_qc.cpp — Unit tests for alignment_qc.h (SS2 + B2)
//
// Tests:
//   1. mapping_rate and duplication_rate computed correctly
//   2. genes_detected and median_gene_coverage
//   3. Lander-Waterman estimate: no duplicates → T ≈ C; high duplication → T >> C
//   4. five_prime_three_prime_ratio: 5'-biased profile > 1, 3'-biased < 1, flat ≈ 1
//   5. compute_gene_body_coverage: normalisation and zero-coverage gene skipping
//   6. write_report: file produced with expected key count
//   7. mito_fraction passed through correctly
//   8. Gene body coverage with single gene

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/alignment_qc.h"

using namespace singlet;

// ── Helpers ───────────────────────────────────────────────────────────────────

static int passed = 0, failed = 0;

static void PASS(const char* name) {
    ++passed;
    std::cerr << "PASS: " << name << "\n";
}

static void FAIL(const char* name, const char* msg) {
    ++failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

static bool near(double a, double b, double tol = 1e-4) {
    return std::fabs(a - b) <= tol;
}

// ── Test 1: mapping_rate and duplication_rate ─────────────────────────────────

static void test_rates() {
    AlignmentQCComputer qcc;
    // total=1000, mapped=900, unique=800, multi=100, dup=180
    AlignmentQC qc = qcc.compute_from_stats(
        1000, 900, 800, 100, 180,
        /*mito=*/0.05f,
        /*gene_counts=*/{},
        /*profile=*/{});

    if (near(qc.mapping_rate, 0.9f))
        PASS("mapping_rate=0.9");
    else
        FAIL("mapping_rate=0.9", "wrong value");

    // duplication_rate = 180/900
    if (near(qc.duplication_rate, 180.0 / 900.0))
        PASS("duplication_rate=180/900");
    else
        FAIL("duplication_rate=180/900", "wrong value");
}

// ── Test 2: genes_detected and median_gene_coverage ──────────────────────────

static void test_gene_detection() {
    AlignmentQCComputer qcc;
    // min_reads=5; gene counts: 1,3,5,10,20 → detected: 5,10,20 (3 genes)
    std::vector<uint32_t> gcounts = {1, 3, 5, 10, 20};
    AlignmentQC qc = qcc.compute_from_stats(
        100, 80, 70, 10, 5, 0.f, gcounts, {}, /*min_reads=*/5);

    if (qc.genes_detected == 3)
        PASS("genes_detected=3");
    else
        FAIL("genes_detected=3", "wrong count");

    // sorted detected: [5,10,20] → median = 10
    if (near(qc.median_gene_coverage, 10.f))
        PASS("median_gene_coverage=10");
    else
        FAIL("median_gene_coverage=10", "wrong median");
}

// ── Test 3: Lander-Waterman library complexity ────────────────────────────────

static void test_lander_waterman() {
    // Case A: no duplicates → distinct == total → T returns distinct
    {
        float T = AlignmentQCComputer::lander_waterman_complexity(1000, 1000);
        if (near(T, 1000.f, 1.f))
            PASS("lw_no_dups: T ≈ 1000");
        else
            FAIL("lw_no_dups: T ≈ 1000", "wrong estimate");
    }

    // Case B: ~38% duplication rate → distinct=500, total=1000
    // Solve: T*(1 - e^(-1000/T)) = 500  →  T ≈ 625  (x = 1000/625 = 1.6)
    {
        float T = AlignmentQCComputer::lander_waterman_complexity(1000, 500);
        // T should be in [550, 750]
        if (T > 550.f && T < 750.f)
            PASS("lw_50pct_dup: 550 < T < 750");
        else
            FAIL("lw_50pct_dup: 550 < T < 750", "out of expected range");
    }

    // Case C: 90% duplication → distinct=100k from 1M total
    // Solve: T*(1 - e^(-1M/T)) = 100k  →  T ≈ 100k  (fully sequenced library)
    {
        float T = AlignmentQCComputer::lander_waterman_complexity(1000000, 100000);
        if (T > 80000.f && T < 150000.f)
            PASS("lw_high_dup: 80k < T < 150k");
        else
            FAIL("lw_high_dup: 80k < T < 150k", "out of expected range");
    }

    // Case D: zero mapped → returns 0
    {
        float T = AlignmentQCComputer::lander_waterman_complexity(0, 0);
        if (T == 0.f)
            PASS("lw_zero: T=0");
        else
            FAIL("lw_zero: T=0", "should be 0");
    }
}

// ── Test 4: five_prime_three_prime_ratio ──────────────────────────────────────

static void test_53_ratio() {
    AlignmentQCComputer qcc;

    // 5'-biased: high at start, low at end → ratio > 1
    std::vector<double> five_biased(100, 0.1);
    for (int i = 0; i < 10; ++i) five_biased[static_cast<size_t>(i)] = 1.0;     // head high
    for (int i = 90; i < 100; ++i) five_biased[static_cast<size_t>(i)] = 0.01;  // tail low

    float r5 = qcc.five_prime_three_prime_ratio(five_biased, 10);
    if (r5 > 1.f)
        PASS("53_ratio_5prime_biased > 1");
    else
        FAIL("53_ratio_5prime_biased > 1", "should be > 1");

    // 3'-biased: low at start, high at end → ratio < 1
    std::vector<double> three_biased(100, 0.1);
    for (int i = 0; i < 10; ++i) three_biased[static_cast<size_t>(i)] = 0.01;
    for (int i = 90; i < 100; ++i) three_biased[static_cast<size_t>(i)] = 1.0;

    float r3 = qcc.five_prime_three_prime_ratio(three_biased, 10);
    if (r3 < 1.f)
        PASS("53_ratio_3prime_biased < 1");
    else
        FAIL("53_ratio_3prime_biased < 1", "should be < 1");

    // Flat profile → ratio ≈ 1
    std::vector<double> flat(100, 1.0);
    float rf = qcc.five_prime_three_prime_ratio(flat, 10);
    if (near(rf, 1.f, 0.01f))
        PASS("53_ratio_flat ≈ 1");
    else
        FAIL("53_ratio_flat ≈ 1", "should be near 1");
}

// ── Test 5: compute_gene_body_coverage normalisation ─────────────────────────

static void test_gene_body_coverage() {
    AlignmentQCComputer qcc;

    // Three genes with 10 bins each
    std::vector<std::vector<double>> genes = {
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 10},  // uniform → each bin = 0.1
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0},            // zero → skipped
        {20, 10, 10, 10, 10, 10, 10, 10, 10, 20},  // higher at ends
    };

    auto profile = qcc.compute_gene_body_coverage(genes, 10);

    if (profile.size() == 10)
        PASS("gene_body_coverage: size=10");
    else
        FAIL("gene_body_coverage: size=10", "wrong size");

    // First gene: each bin = 10/100 = 0.1; third gene: bin[0] = 20/120 ≈ 0.1667
    // Average (2 genes): bin[0] = (0.1 + 0.1667) / 2 ≈ 0.1333
    if (profile[0] > 0.0 && profile[0] < 1.0)
        PASS("gene_body_coverage: bin[0] in (0,1)");
    else
        FAIL("gene_body_coverage: bin[0] in (0,1)", "out of range");

    // Zero-coverage gene skipped: profile should be average of 2, not 3
    if (profile[0] > 0.12 && profile[0] < 0.20)
        PASS("gene_body_coverage: zero gene skipped");
    else
        FAIL("gene_body_coverage: zero gene skipped", "expected ~0.133");
}

// ── Test 6: write_report ──────────────────────────────────────────────────────

static void test_write_report() {
    AlignmentQCComputer qcc;
    AlignmentQC qc = qcc.compute_from_stats(
        1000, 900, 800, 100, 90, 0.05f,
        {10, 20, 30, 1}, {}, 5);

    const char* tmpfile = "/tmp/test_alignment_qc_report.tsv";
    qcc.write_report(tmpfile, qc);

    std::ifstream f(tmpfile);
    int n_lines = 0;
    std::string line;
    while (std::getline(f, line)) ++n_lines;
    std::remove(tmpfile);

    // Expect exactly 12 key-value lines
    if (n_lines == 12)
        PASS("write_report: 12 lines");
    else
        FAIL("write_report: 12 lines", "wrong line count");
}

// ── Test 7: mito_fraction passthrough ────────────────────────────────────────

static void test_mito_fraction() {
    AlignmentQCComputer qcc;
    AlignmentQC qc = qcc.compute_from_stats(100, 80, 70, 10, 0, 0.15f, {}, {});
    if (near(qc.mito_fraction, 0.15f))
        PASS("mito_fraction=0.15");
    else
        FAIL("mito_fraction=0.15", "wrong value");
}

// ── Test 8: single gene coverage profile ─────────────────────────────────────

static void test_single_gene_profile() {
    AlignmentQCComputer qcc;
    // Single gene, 10 bins, linear ramp 1..10
    std::vector<std::vector<double>> genes = {
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}};
    auto profile = qcc.compute_gene_body_coverage(genes, 10);

    // sum = 55; normalised bin[0] = 1/55 ≈ 0.01818, bin[9] = 10/55 ≈ 0.1818
    if (near(profile[0], 1.0 / 55.0, 1e-5) && near(profile[9], 10.0 / 55.0, 1e-5))
        PASS("single_gene_profile: correct normalisation");
    else
        FAIL("single_gene_profile: correct normalisation", "wrong bin values");

    // 5'/3' ratio: 3'-biased (ramp up to end) → ratio < 1
    float r = qcc.five_prime_three_prime_ratio(profile, 3);
    if (r < 1.f)
        PASS("single_gene_profile: 53_ratio < 1 for ramp");
    else
        FAIL("single_gene_profile: 53_ratio < 1 for ramp", "should be < 1");
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main() {
    test_rates();
    test_gene_detection();
    test_lander_waterman();
    test_53_ratio();
    test_gene_body_coverage();
    test_write_report();
    test_mito_fraction();
    test_single_gene_profile();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return failed ? 1 : 0;
}
