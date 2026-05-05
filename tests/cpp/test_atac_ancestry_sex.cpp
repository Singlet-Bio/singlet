// test_atac_ancestry_sex.cpp
// A5: Tests for ATAC ancestry classification + sex/karyotype calling.

#include "singlet-pileup/atac_ancestry_sex.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

using namespace singlet;

// ─────────────────────────────────────────────────────────────────────────────
// Sex calling tests
// ─────────────────────────────────────────────────────────────────────────────

// Test 1: XX — no chrY fragments → female
static void test_sex_xx_no_y() {
    AtacSexCaller sex;
    sex.add_counts(10000, 0, 100000);
    auto r = sex.call();
    assert(r.sex == "XX");
    assert(r.confidence > 0.9);
    assert(r.chry_frac == 0.0);
    printf("PASS test_sex_xx_no_y\n");
}

// Test 2: XY — high chrY → male
static void test_sex_xy_high_y() {
    AtacSexCaller sex;
    sex.add_counts(10000, 8000, 100000);  // chrY/(chrX+chrY) = 0.444
    auto r = sex.call();
    assert(r.sex == "XY");
    assert(r.confidence > 0.0);
    assert(r.chry_frac > 0.25);
    printf("PASS test_sex_xy_high_y\n");
}

// Test 3: Ambiguous chrY (borderline) → uncertain
static void test_sex_uncertain() {
    AtacSexCaller sex;
    sex.add_counts(10000, 1500, 100000);  // chrY/(chrX+chrY) = 0.13 (between thresholds)
    auto r = sex.call();
    assert(r.sex == "uncertain");
    printf("PASS test_sex_uncertain\n");
}

// Test 4: Edge — zero fragments on sex chromosomes → no_data
static void test_sex_no_data() {
    AtacSexCaller sex;
    sex.add_counts(0, 0, 50000);
    auto r = sex.call();
    assert(r.sex == "no_data");
    assert(r.confidence == 0.0);
    assert(std::isnan(r.chry_frac));
    printf("PASS test_sex_no_data\n");
}

// Test 5: add_fragment() string API for XX
static void test_sex_add_fragment_api() {
    AtacSexCaller sex;
    for (int i = 0; i < 500; ++i) sex.add_fragment("chrX", i * 200, i * 200 + 150);
    for (int i = 0; i < 10; ++i) sex.add_fragment("chr1", i * 1000, i * 1000 + 200);
    auto r = sex.call();
    assert(r.sex == "XX");
    assert(r.chrx_frags == 500);
    assert(r.chry_frags == 0);
    assert(r.auto_frags == 10);
    printf("PASS test_sex_add_fragment_api\n");
}

// Test 6: clear() resets state
static void test_sex_clear() {
    AtacSexCaller sex;
    sex.add_counts(0, 5000, 0);  // would be XY
    sex.clear();
    sex.add_counts(5000, 0, 0);  // now XX
    auto r = sex.call();
    assert(r.sex == "XX");
    printf("PASS test_sex_clear\n");
}

// ─────────────────────────────────────────────────────────────────────────────
// Ancestry calling tests
// ─────────────────────────────────────────────────────────────────────────────

// Test 7: EUR-like allele frequencies → EUR call
static void test_ancestry_eur() {
    const auto& aims = get_aim_panel();
    AtacAncestryClassifier anc;
    // Feed 20 ALT observations at index 0 (rs1426654, EUR alt freq ~0.989)
    // and 1 ALT + 19 REF at index 1 (rs16891982, EUR alt freq ~0.879)
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(0, 1);  // alt (EUR-like)
    for (int i = 0; i < 18; ++i) anc.add_aim_observation(1, 1);  // alt (EUR-like)
    for (int i = 0; i < 2;  ++i) anc.add_aim_observation(1, 0);  // ref
    // rs2814778 (idx 5): AFR alt freq 0.944, EUR alt freq 0.017 → feed mostly REF (EUR-like)
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(5, 0);  // ref (EUR-like)
    // rs4988235 (idx 6): EUR lactase, alt freq EUR 0.753 → feed mostly ALT
    for (int i = 0; i < 15; ++i) anc.add_aim_observation(6, 1);
    for (int i = 0; i < 5;  ++i) anc.add_aim_observation(6, 0);
    // rs17822931 (idx 7): EAS earwax, alt freq EAS 0.776, EUR 0.151 → mostly REF
    for (int i = 0; i < 18; ++i) anc.add_aim_observation(7, 0);
    for (int i = 0; i < 2;  ++i) anc.add_aim_observation(7, 1);
    auto r = anc.classify();
    assert(r.ancestry == "EUR");
    assert(r.confidence > 0.6);
    assert(!r.low_data);
    printf("PASS test_ancestry_eur (conf=%.3f)\n", r.confidence);
}

// Test 8: AFR-like allele frequencies → AFR call
static void test_ancestry_afr() {
    AtacAncestryClassifier anc;
    // rs2814778 (idx 5): AFR alt freq 0.944 → feed mostly ALT
    for (int i = 0; i < 19; ++i) anc.add_aim_observation(5, 1);
    for (int i = 0; i < 1;  ++i) anc.add_aim_observation(5, 0);
    // rs1426654 (idx 0): AFR alt freq 0.023 → feed mostly REF (AFR-like)
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(0, 0);
    // rs4833103 (idx 15): AFR alt freq 0.508 → feed ~50% alt
    for (int i = 0; i < 10; ++i) anc.add_aim_observation(15, 1);
    for (int i = 0; i < 10; ++i) anc.add_aim_observation(15, 0);
    // rs7784487 (idx 14): AFR alt freq 0.583 → feed mostly alt
    for (int i = 0; i < 12; ++i) anc.add_aim_observation(14, 1);
    for (int i = 0; i < 8;  ++i) anc.add_aim_observation(14, 0);
    // rs937277 (idx 12): AFR 0.480, EUR 0.096 → feed mostly alt
    for (int i = 0; i < 10; ++i) anc.add_aim_observation(12, 1);
    for (int i = 0; i < 5;  ++i) anc.add_aim_observation(12, 0);
    auto r = anc.classify();
    assert(r.ancestry == "AFR");
    assert(r.confidence > 0.5);
    printf("PASS test_ancestry_afr (conf=%.3f)\n", r.confidence);
}

// Test 9: Too few AIMs → low_data flag
static void test_ancestry_low_data() {
    AtacAncestryClassifier anc;
    // Only 3 observations (< kAncestryMinInformative = 5)
    anc.add_aim_observation(0, 1);
    anc.add_aim_observation(1, 0);
    anc.add_aim_observation(2, 1);
    auto r = anc.classify();
    assert(r.low_data == true);
    assert(r.n_informative == 3);
    printf("PASS test_ancestry_low_data\n");
}

// Test 10: Single SNP for ancestry — runs without crash
static void test_ancestry_single_snp() {
    AtacAncestryClassifier anc;
    anc.add_aim_observation(0, 1);  // rs1426654 alt (EUR-like)
    auto r = anc.classify();
    assert(r.n_informative == 1);
    assert(r.low_data == true);
    // probabilities should sum to 1
    double sum = 0.0;
    for (int p = 0; p < 5; ++p) sum += r.prob[p];
    assert(std::abs(sum - 1.0) < 1e-9);
    printf("PASS test_ancestry_single_snp (sum_prob=%.10f)\n", sum);
}

// Test 11: batch add_observations API
static void test_ancestry_batch_add() {
    AtacAncestryClassifier anc;
    std::vector<AtacAimObservation> obs;
    for (int i = 0; i < 10; ++i) obs.push_back({0u, 1u}); // aim 0, alt
    for (int i = 0; i < 10; ++i) obs.push_back({5u, 0u}); // aim 5, ref (EUR-like)
    for (int i = 0; i < 10; ++i) obs.push_back({6u, 1u}); // aim 6, alt
    for (int i = 0; i < 10; ++i) obs.push_back({7u, 0u}); // aim 7, ref
    for (int i = 0; i < 10; ++i) obs.push_back({1u, 1u}); // aim 1, alt (EUR-like)
    anc.add_observations(obs);
    auto r = anc.classify();
    assert(r.n_informative == 5);
    assert(!r.low_data);
    printf("PASS test_ancestry_batch_add\n");
}

// Test 12: JSON output — sex
static void test_json_sex_output() {
    AtacSexCaller sex;
    sex.add_counts(10000, 0, 80000);
    auto r = sex.call();
    write_atac_sex_json("/tmp/test_atac_sex.json", r);
    FILE* f = fopen("/tmp/test_atac_sex.json", "r");
    assert(f != nullptr);
    fclose(f);
    printf("PASS test_json_sex_output\n");
}

// Test 13: clear and reuse classifier
static void test_ancestry_clear() {
    AtacAncestryClassifier anc;
    // First: EUR
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(0, 1);
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(1, 1);
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(5, 0);
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(6, 1);
    for (int i = 0; i < 20; ++i) anc.add_aim_observation(7, 0);
    auto r1 = anc.classify();
    anc.clear();
    assert(anc.classify().n_informative == 0);
    printf("PASS test_ancestry_clear\n");
}

int main() {
    printf("=== A5 ATAC Ancestry + Sex Tests ===\n");
    test_sex_xx_no_y();
    test_sex_xy_high_y();
    test_sex_uncertain();
    test_sex_no_data();
    test_sex_add_fragment_api();
    test_sex_clear();
    test_ancestry_eur();
    test_ancestry_afr();
    test_ancestry_low_data();
    test_ancestry_single_snp();
    test_ancestry_batch_add();
    test_json_sex_output();
    test_ancestry_clear();
    printf("=== All 13 tests PASSED ===\n");
    return 0;
}
