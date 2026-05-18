// SPDX-License-Identifier: MIT
// test/test_atac_donor_demux.cpp
// Unit tests for AtacDonorDemux (A4 feature).
// All tests synthetic — no BAM / disk files required.

#include <cassert>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/atac_donor_demux.h"

using namespace singlet;

#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

// ── Helpers ──────────────────────────────────────────────────────────────────

// Fill observations for a barcode: every SNP in snp_list gets n_reads reads,
// each with the given is_alt value (uniform allele).
static void fill_bc(AtacDonorDemux& demux,
                    uint32_t bc_idx,
                    const std::vector<int32_t>& snp_list,
                    uint8_t is_alt,
                    int n_reads) {
    for (auto snp : snp_list)
        for (int r = 0; r < n_reads; ++r)
            demux.add_snp_observation(bc_idx, snp, is_alt);
}

// ── Test 1: two donors with distinct SNP profiles ────────────────────────────
// Donor A: all ALT at SNPs 0–4   Donor B: all REF at SNPs 0–4
// Barcodes 0–2 → donor A,  barcodes 3–5 → donor B.
static void test_two_donors_distinct() {
    std::cout << "Test 1: two donors with distinct SNP profiles\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 2;
    cfg.n_inits = 3;
    cfg.threads = 1;
    demux.set_config(cfg);

    const int n_snps = 5, n_barcodes = 6;
    std::vector<int32_t> all_snps = {0, 1, 2, 3, 4};

    // Donor A barcodes: heavy ALT signal
    for (uint32_t bc = 0; bc < 3; ++bc)
        fill_bc(demux, bc, all_snps, /*is_alt=*/1, 20);

    // Donor B barcodes: heavy REF signal
    for (uint32_t bc = 3; bc < 6; ++bc)
        fill_bc(demux, bc, all_snps, /*is_alt=*/0, 20);

    DemuxResult res = demux.run(n_snps, n_barcodes);

    CHECK(res.assignments.size() == n_barcodes, "assignment count == n_barcodes");

    // All barcodes should be assigned (not "unassigned")
    bool all_assigned = true;
    for (const auto& a : res.assignments)
        if (a.label == "unassigned") {
            all_assigned = false;
            break;
        }
    CHECK(all_assigned, "all barcodes assigned");

    // Group A and group B should receive the SAME donor label within each group
    int da = res.assignments[0].donor_id;
    int db = res.assignments[3].donor_id;
    bool groups_consistent = true;
    for (int bc = 0; bc < 3; ++bc)
        if (res.assignments[bc].donor_id != da) {
            groups_consistent = false;
            break;
        }
    for (int bc = 3; bc < 6; ++bc)
        if (res.assignments[bc].donor_id != db) {
            groups_consistent = false;
            break;
        }
    CHECK(groups_consistent, "each group assigned same donor");
    CHECK(da != db, "two groups assigned different donors");
}

// ── Test 2: single donor — all barcodes assigned to donor 0 ──────────────────
static void test_single_donor() {
    std::cout << "Test 2: single donor\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 1;
    cfg.n_inits = 2;
    cfg.threads = 1;
    demux.set_config(cfg);

    const int n_snps = 4, n_barcodes = 5;
    std::vector<int32_t> snps = {0, 1, 2, 3};

    // All barcodes: mixed alleles (50/50 matches one donor's posterior)
    for (uint32_t bc = 0; bc < (uint32_t)n_barcodes; ++bc) {
        fill_bc(demux, bc, snps, 1, 10);
        fill_bc(demux, bc, snps, 0, 10);
    }

    DemuxResult res = demux.run(n_snps, n_barcodes);

    CHECK(res.assignments.size() == (size_t)n_barcodes, "assignment count correct");
    bool all_donor0 = true;
    for (const auto& a : res.assignments)
        if (a.donor_id != 0) {
            all_donor0 = false;
            break;
        }
    CHECK(all_donor0, "all barcodes → donor 0 (K=1)");
}

// ── Test 3: unknown donor — low-confidence / unassigned barcodes ─────────────
// K=2 but one barcode has no observations → should be unassigned or low-prob
static void test_unknown_donor_unassigned() {
    std::cout << "Test 3: barcode with no observations → unassigned\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 2;
    cfg.n_inits = 2;
    cfg.threads = 1;
    cfg.min_prob = 0.9f;  // strict threshold
    demux.set_config(cfg);

    const int n_snps = 4, n_barcodes = 4;

    // Barcodes 0–2: strong signal
    std::vector<int32_t> snps = {0, 1, 2, 3};
    fill_bc(demux, 0, snps, 1, 30);
    fill_bc(demux, 1, snps, 1, 30);
    fill_bc(demux, 2, snps, 0, 30);
    // Barcode 3: no observations at all → VB posterior = uniform = 0.5 per donor

    DemuxResult res = demux.run(n_snps, n_barcodes);

    CHECK(res.assignments.size() == (size_t)n_barcodes, "4 assignments returned");

    // Barcode 3 has no coverage → max posterior ≤ 0.5 for K=2
    float p3 = res.assignments[3].prob_max;
    CHECK(p3 < cfg.min_prob, "no-coverage barcode below confidence threshold");

    // Check label for barcode 3 (should be unassigned given strict threshold)
    CHECK(res.assignments[3].label == "unassigned",
          "no-coverage barcode labeled unassigned");
}

// ── Test 4: no SNPs in fragments — all barcodes unassigned ───────────────────
static void test_no_snps_in_fragments() {
    std::cout << "Test 4: no SNP observations → all unassigned\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 2;
    cfg.threads = 1;
    demux.set_config(cfg);

    // No observations added
    DemuxResult res = demux.run(/*n_snps=*/10, /*n_barcodes=*/5);

    CHECK(res.assignments.size() == 5u, "5 assignments returned");
    bool all_unassigned = true;
    for (const auto& a : res.assignments)
        if (a.label != "unassigned") {
            all_unassigned = false;
            break;
        }
    CHECK(all_unassigned, "all barcodes unassigned when no observations");
}

// ── Test 5: edge cases — empty barcodes and zero-coverage SNPs ────────────────
static void test_edge_cases() {
    std::cout << "Test 5: edge cases — empty fragment list, out-of-range indices\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 1;
    cfg.threads = 1;
    demux.set_config(cfg);

    // Add out-of-range observations (should be silently ignored)
    demux.add_snp_observation(999, 0, 1);    // barcode out of range
    demux.add_snp_observation(0, 999, 1);    // snp out of range
    demux.add_snp_observation(999, 999, 1);  // both out of range
    demux.add_snp_observation(0, -1, 1);     // negative snp index

    CHECK(demux.observation_count() == 4u, "4 observations recorded (before filtering)");

    // Run with small sizes — out-of-range ones are filtered during run()
    DemuxResult res = demux.run(/*n_snps=*/5, /*n_barcodes=*/3);

    CHECK(res.assignments.size() == 3u, "3 assignments for 3 barcodes");
    // After filtering, no valid observations → all unassigned
    for (const auto& a : res.assignments)
        CHECK(a.label == "unassigned", "out-of-range-only bc → unassigned");

    // clear() resets observations
    demux.clear();
    CHECK(demux.observation_count() == 0u, "clear() resets observation count");
}

// ── Test 6: mixed quality — high vs low confidence barcodes ──────────────────
// Barcodes with many reads → high prob_max, few reads → low prob_max
static void test_mixed_confidence() {
    std::cout << "Test 6: mixed read depth — high vs low confidence assignments\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 2;
    cfg.n_inits = 3;
    cfg.threads = 1;
    demux.set_config(cfg);

    const int n_snps = 6, n_barcodes = 6;
    std::vector<int32_t> snps = {0, 1, 2, 3, 4, 5};

    // High-confidence barcodes 0–1: 50 reads each, all ALT
    fill_bc(demux, 0, snps, 1, 50);
    fill_bc(demux, 1, snps, 1, 50);

    // High-confidence barcodes 2–3: 50 reads each, all REF
    fill_bc(demux, 2, snps, 0, 50);
    fill_bc(demux, 3, snps, 0, 50);

    // Low-confidence barcodes 4–5: zero observations → uniform posterior (prob_max = 0.5)
    // (No fill_bc calls for bc 4 and 5)

    DemuxResult res = demux.run(n_snps, n_barcodes);

    CHECK(res.assignments.size() == (size_t)n_barcodes, "6 assignments");

    // High-confidence barcodes should have higher prob_max than zero-coverage barcodes
    float prob_high = std::min(res.assignments[0].prob_max, res.assignments[2].prob_max);
    float prob_low = std::max(res.assignments[4].prob_max, res.assignments[5].prob_max);
    CHECK(prob_high > prob_low, "high-depth barcodes have higher confidence than zero-coverage");

    // High-confidence barcodes should pass the default min_prob threshold
    DonorDemuxConfig default_cfg;
    CHECK(prob_high >= default_cfg.min_prob, "high-depth barcodes exceed default min_prob");
}

// ── Test 7: add_observations batch API ───────────────────────────────────────
static void test_batch_add() {
    std::cout << "Test 7: batch add_observations API\n";

    AtacDonorDemux demux;
    DonorDemuxConfig cfg;
    cfg.n_donors = 1;
    cfg.threads = 1;
    demux.set_config(cfg);

    std::vector<AtacSnpObservation> batch;
    for (int i = 0; i < 20; ++i)
        batch.push_back({0u, 0, 1u});  // barcode 0, snp 0, alt

    demux.add_observations(batch);
    CHECK(demux.observation_count() == 20u, "batch add stores 20 observations");

    DemuxResult res = demux.run(/*n_snps=*/1, /*n_barcodes=*/1);
    CHECK(res.assignments.size() == 1u, "1 barcode assigned");
    CHECK(res.assignments[0].donor_id == 0, "single barcode → donor 0");
}

// ── main ─────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== test_atac_donor_demux ===\n";

    test_two_donors_distinct();
    test_single_donor();
    test_unknown_donor_unassigned();
    test_no_snps_in_fragments();
    test_edge_cases();
    test_mixed_confidence();
    test_batch_add();

    std::cout << "\nResults: " << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail > 0 ? 1 : 0;
}
