// SPDX-License-Identifier: MIT
// test/test_rna_variant_caller.cpp
// Unit tests for RNAVariantCaller (SS3 + B3 de novo variant discovery).
// Tests exercise call_position() directly — no BAM I/O required.
// Built with PILEUP_INCLUDE_DIRS + PILEUP_LINK_LIBS so real htslib is available.

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "singlet/pileup/rna_variant_caller.h"

using namespace singlet;

#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

// ── Test 1: clear het variant — 30 ref (A), 20 alt (T) → VAF=0.4 ────────────
static void test_clear_het_variant() {
    std::cout << "Test 1: clear het variant\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(10);
    vc.set_min_alt_count(3);
    vc.set_min_vaf(0.1f);

    AlleleCount ac;
    // 30 A (ref), 20 T (alt)
    for (int i = 0; i < 30; ++i) ac.add(0);  // A
    for (int i = 0; i < 20; ++i) ac.add(3);  // T

    RNAVariant v;
    bool called = vc.call_position(0, 1000, ac, v, /*ref_idx=*/0);
    CHECK(called, "variant called");
    CHECK(v.ref_allele == 'A', "ref allele = A");
    CHECK(v.alt_allele == 'T', "alt allele = T");
    CHECK(v.ref_count == 30, "ref_count = 30");
    CHECK(v.alt_count == 20, "alt_count = 20");
    CHECK(std::fabs(v.vaf - 0.4f) < 0.001f, "VAF ≈ 0.4");
    CHECK(v.position == 1000, "position stored");
    CHECK(v.chrom_idx == 0, "chrom_idx stored");
}

// ── Test 2: below coverage threshold → no call ───────────────────────────────
static void test_below_coverage_threshold() {
    std::cout << "Test 2: below coverage threshold\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(10);
    vc.set_min_alt_count(1);
    vc.set_min_vaf(0.1f);

    AlleleCount ac;
    for (int i = 0; i < 5; ++i) ac.add(0);  // A ×5
    for (int i = 0; i < 2; ++i) ac.add(3);  // T ×2 — VAF=0.29 but coverage=7 < 10

    RNAVariant v;
    bool called = vc.call_position(0, 500, ac, v, 0);
    CHECK(!called, "no call when coverage < min_cov");
}

// ── Test 3: low VAF below threshold → no call ───────────────────────────────
static void test_low_vaf_no_call() {
    std::cout << "Test 3: low VAF below threshold\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(10);
    vc.set_min_alt_count(1);
    vc.set_min_vaf(0.1f);

    AlleleCount ac;
    for (int i = 0; i < 90; ++i) ac.add(1);  // C ×90 (ref)
    for (int i = 0; i < 1; ++i) ac.add(2);   // G ×1  (alt) → VAF=0.011 < 0.1

    RNAVariant v;
    bool called = vc.call_position(0, 200, ac, v, 1 /*C=ref*/);
    CHECK(!called, "no call when VAF < min_vaf");
}

// ── Test 4: multiple positions → correct variant count ──────────────────────
static void test_multiple_variants() {
    std::cout << "Test 4: multiple variants across positions\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(10);
    vc.set_min_alt_count(3);
    vc.set_min_vaf(0.1f);

    // Build 5 positions: 3 should be called, 2 should not
    struct PosSpec {
        int ref_cnt;
        int alt_cnt;
        int ref_idx;
        int alt_idx;
        bool expect;
    };
    std::vector<PosSpec> specs = {
        {50, 10, 0, 3, true},  // A/T het; VAF=0.17 ✓
        {80, 2, 1, 2, false},  // C/G: alt_cnt=2 < min_alt=3 → no call
        {20, 20, 2, 0, true},  // G/A 50% ✓
        {5, 5, 0, 1, false},   // coverage=10 but alt_cnt=5≥3 AND VAF=0.5 ✓... wait coverage=10 == min_cov, should call
        {30, 15, 3, 0, true},  // T/A VAF=0.33 ✓
    };
    // Fix spec[3]: coverage==10 == min_cov so it IS called; set expect=true
    specs[3].expect = true;

    int called_count = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        const auto& s = specs[i];
        AlleleCount ac;
        for (int j = 0; j < s.ref_cnt; ++j) ac.add(s.ref_idx);
        for (int j = 0; j < s.alt_cnt; ++j) ac.add(s.alt_idx);
        RNAVariant v;
        bool c = vc.call_position(0, (int32_t)i * 100, ac, v, s.ref_idx);
        if (c) ++called_count;
        if (c != s.expect) {
            std::cout << "  FAIL: position " << i << " expected called=" << s.expect
                      << " got=" << c << "\n";
            ++n_fail;
        } else {
            ++n_pass;
        }
    }
    CHECK(called_count == 4, "4 of 5 positions called");
}

// ── Test 5: min_alt_count filter ────────────────────────────────────────────
static void test_min_alt_count_filter() {
    std::cout << "Test 5: min_alt_count filter\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(5);
    vc.set_min_alt_count(3);
    vc.set_min_vaf(0.01f);  // low VAF threshold so only alt_count matters

    AlleleCount ac;
    for (int i = 0; i < 98; ++i) ac.add(0);  // A ×98
    for (int i = 0; i < 2; ++i) ac.add(1);   // C ×2 → alt_cnt=2 < 3

    RNAVariant v;
    bool called = vc.call_position(0, 1, ac, v, 0);
    CHECK(!called, "no call when alt_count < min_alt");

    // Now bump alt to 3 → should call
    ac.add(1);  // C ×3 total now
    called = vc.call_position(0, 1, ac, v, 0);
    CHECK(called, "call when alt_count == min_alt");
}

// ── Test 6: write_tsv and write_vcf smoke test ───────────────────────────────
static void test_output_writers() {
    std::cout << "Test 6: output writers (TSV + VCF)\n";
    RNAVariantCaller vc;
    vc.set_min_coverage(5);
    vc.set_min_alt_count(1);
    vc.set_min_vaf(0.05f);

    AlleleCount ac;
    for (int i = 0; i < 70; ++i) ac.add(0);  // A
    for (int i = 0; i < 30; ++i) ac.add(3);  // T

    RNAVariant v;
    bool c = vc.call_position(1, 999, ac, v, 0);
    // Manually add to a fresh caller result for writer test
    (void)c;

    // Create a tiny caller with one synthetic variant
    RNAVariantCaller vc2;
    vc2.set_min_coverage(5);
    vc2.set_min_alt_count(1);
    vc2.set_min_vaf(0.05f);

    AlleleCount ac2;
    for (int i = 0; i < 60; ++i) ac2.add(2);  // G
    for (int i = 0; i < 40; ++i) ac2.add(0);  // A

    std::vector<std::string> chroms = {"chr1", "chr2"};
    RNAVariant vout;
    bool ok = vc2.call_position(0, 5000, ac2, vout, 2 /*G=ref*/);
    CHECK(ok, "variant called for writer test");

    // Test write paths to /tmp
    if (ok) {
        // Inject the variant manually by running through process path
        // (write_tsv / write_vcf tested via direct call to the variant-less vc2)
        // The writers iterate over variants_; with no BAM run variants_ is empty.
        // This smoke test just confirms the file I/O doesn't throw.
        try {
            vc2.write_tsv("/tmp/test_rvc.tsv", chroms);
            vc2.write_vcf("/tmp/test_rvc.vcf", chroms);
            CHECK(true, "write_tsv and write_vcf do not throw");
        } catch (...) {
            CHECK(false, "write_tsv or write_vcf threw exception");
        }
    }
}

int main() {
    std::cout << "=== test_rna_variant_caller ===\n";
    test_clear_het_variant();
    test_below_coverage_threshold();
    test_low_vaf_no_call();
    test_multiple_variants();
    test_min_alt_count_filter();
    test_output_writers();

    std::cout << "\nResult: " << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
