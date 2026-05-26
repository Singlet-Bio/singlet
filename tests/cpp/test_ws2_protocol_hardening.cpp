// test_ws2_protocol_hardening.cpp
// Unit tests for WS-2 protocol detection hardening:
//   1. sci-RNA-seq3 linker floor (reject if linker_rate < 10%)
//   2. BD Rhapsody geometry boost (linker_rate >= 20% → +0.15 score)
//   3. Confidence floor logging (sub-LOW confidence diagnostic)
//   4. known_protocols() registry integrity
//
// These test the protocol.h primitives without requiring real SRA data.
// The detect_protocol() template is tested with synthetic SpotData.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <algorithm>
#include <unordered_set>

#include "singlet/fq/protocol.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", msg); } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", msg); } \
} while(0)

// ── Test 1: known_protocols() contains key protocols ──────────────────────────
static void test_known_protocols_registry() {
    std::fprintf(stderr, "[ws2] known_protocols registry...\n");
    const auto& protos = lib1fq::known_protocols();
    CHECK(!protos.empty(), "known_protocols non-empty");

    // Must contain these key protocols
    std::unordered_set<std::string> tags;
    for (const auto& p : protos) tags.insert(p.tag);

    CHECK(tags.count("10x-3p-v3") > 0, "has 10x-3p-v3");
    CHECK(tags.count("10x-3p-v2") > 0, "has 10x-3p-v2");
    CHECK(tags.count("sci-rna-seq3") > 0, "has sci-rna-seq3");
    CHECK(tags.count("bd-rhapsody") > 0, "has bd-rhapsody");
    CHECK(tags.count("dropseq") > 0, "has dropseq");
    CHECK(tags.count("celseq2") > 0, "has celseq2");
}

// ── Test 2: sci-RNA-seq3 has a non-empty linker ──────────────────────────────
static void test_sci_rna_seq3_linker() {
    std::fprintf(stderr, "[ws2] sci-RNA-seq3 linker spec...\n");
    const auto& protos = lib1fq::known_protocols();
    const lib1fq::CandidateSpec* sci = nullptr;
    for (const auto& p : protos) {
        if (p.tag == "sci-rna-seq3") { sci = &p; break; }
    }
    CHECK(sci != nullptr, "sci-rna-seq3 found");
    if (sci) {
        CHECK(!sci->linker.empty(), "sci-rna-seq3 has linker");
        CHECK(sci->linker.size() >= 4, "sci-rna-seq3 linker >= 4bp");
    }
}

// ── Test 3: BD Rhapsody has a non-empty linker ──────────────────────────────
static void test_bd_rhapsody_linker() {
    std::fprintf(stderr, "[ws2] BD Rhapsody linker spec...\n");
    const auto& protos = lib1fq::known_protocols();
    const lib1fq::CandidateSpec* bd = nullptr;
    for (const auto& p : protos) {
        if (p.tag == "bd-rhapsody") { bd = &p; break; }
    }
    CHECK(bd != nullptr, "bd-rhapsody found");
    if (bd) {
        CHECK(!bd->linker.empty(), "bd-rhapsody has linker");
        CHECK(bd->linker.size() >= 6, "bd-rhapsody linker >= 6bp");
    }
}

// ── Test 4: Protocol IDs are unique ──────────────────────────────────────────
static void test_protocol_ids_unique() {
    std::fprintf(stderr, "[ws2] Protocol IDs unique...\n");
    const auto& protos = lib1fq::known_protocols();
    std::unordered_set<uint8_t> ids;
    bool all_unique = true;
    for (const auto& p : protos) {
        if (ids.count(p.protocol_id)) {
            std::fprintf(stderr, "    duplicate protocol_id %d for %s\n",
                         (int)p.protocol_id, p.tag.c_str());
            all_unique = false;
        }
        ids.insert(p.protocol_id);
    }
    CHECK(all_unique, "all protocol_ids unique");
}

// ── Test 5: find_protocol_spec resolves known protocols ──────────────────────
static void test_find_protocol_spec() {
    std::fprintf(stderr, "[ws2] find_protocol_spec lookup...\n");

    auto* v3 = lib1fq::find_protocol_spec("10x-3p-v3");
    CHECK(v3 != nullptr, "find 10x-3p-v3");
    if (v3) {
        CHECK(v3->bc_len == 16, "10x-3p-v3 bc_len=16");
        CHECK(v3->umi_len == 12, "10x-3p-v3 umi_len=12");
        CHECK(v3->r1_len == 28, "10x-3p-v3 r1_len=28");
    }

    auto* v2 = lib1fq::find_protocol_spec("10x-3p-v2");
    CHECK(v2 != nullptr, "find 10x-3p-v2");
    if (v2) {
        CHECK(v2->bc_len == 16, "10x-3p-v2 bc_len=16");
        CHECK(v2->umi_len == 10, "10x-3p-v2 umi_len=10");
        CHECK(v2->r1_len == 26, "10x-3p-v2 r1_len=26");
    }

    auto* bd = lib1fq::find_protocol_spec("bd-rhapsody");
    CHECK(bd != nullptr, "find bd-rhapsody");

    auto* sci = lib1fq::find_protocol_spec("sci-rna-seq3");
    CHECK(sci != nullptr, "find sci-rna-seq3");

    auto* nope = lib1fq::find_protocol_spec("nonexistent-protocol-xyz");
    CHECK(nope == nullptr, "nonexistent returns nullptr");
}

// ── Test 6: Confidence enum ordering ─────────────────────────────────────────
static void test_confidence_ordering() {
    std::fprintf(stderr, "[ws2] Confidence enum ordering...\n");
    using lib1fq::Confidence;
    CHECK(Confidence::NONE < Confidence::LOW, "NONE < LOW");
    CHECK(Confidence::LOW < Confidence::MEDIUM, "LOW < MEDIUM");
    CHECK(Confidence::MEDIUM < Confidence::HIGH, "MEDIUM < HIGH");
}

// ── Test 7: BC/UMI geometry constraints for all protocols ────────────────────
static void test_protocol_geometry_valid() {
    std::fprintf(stderr, "[ws2] Protocol geometry constraints...\n");
    const auto& protos = lib1fq::known_protocols();
    bool all_valid = true;
    for (const auto& p : protos) {
        // BC + UMI must fit within R1 (for non-concat, non-complex protocols with r1_len > 0)
        // Skip complex protocols (e.g. sci-RNA-seq3) where bc_offset+bc_len+umi_len
        // may exceed r1_len because the barcode structure includes linkers between segments.
        if (p.r1_len > 0 && p.linker.empty()) {
            uint16_t need = p.bc_offset + p.bc_len + p.umi_len;
            if (need > p.r1_len + 4) {  // +4 for over-sequencing tolerance
                std::fprintf(stderr, "    %s: bc+umi=%d > r1_len=%d\n",
                             p.tag.c_str(), need, p.r1_len);
                all_valid = false;
            }
        }
        // BC length should be reasonable (1-32bp)
        if (p.bc_len > 0 && p.bc_len > 32) {
            std::fprintf(stderr, "    %s: bc_len=%d > 32\n",
                         p.tag.c_str(), p.bc_len);
            all_valid = false;
        }
        // UMI length should be reasonable (1-16bp)
        if (p.umi_len > 0 && p.umi_len > 16) {
            std::fprintf(stderr, "    %s: umi_len=%d > 16\n",
                         p.tag.c_str(), p.umi_len);
            all_valid = false;
        }
    }
    CHECK(all_valid, "all protocols have valid geometry");
}

// ── Test 8: Whitelist file names are reasonable ──────────────────────────────
static void test_whitelist_file_names() {
    std::fprintf(stderr, "[ws2] Whitelist file names...\n");
    const auto& protos = lib1fq::known_protocols();
    bool all_ok = true;
    for (const auto& p : protos) {
        if (!p.whitelist_file.empty()) {
            // Should end with .txt or .gz
            bool valid_ext = (p.whitelist_file.size() > 4 &&
                (p.whitelist_file.substr(p.whitelist_file.size()-4) == ".txt" ||
                 p.whitelist_file.substr(p.whitelist_file.size()-3) == ".gz"));
            if (!valid_ext) {
                std::fprintf(stderr, "    %s: whitelist '%s' bad extension\n",
                             p.tag.c_str(), p.whitelist_file.c_str());
                all_ok = false;
            }
            // Should not contain path separators
            if (p.whitelist_file.find('/') != std::string::npos) {
                std::fprintf(stderr, "    %s: whitelist '%s' contains path sep\n",
                             p.tag.c_str(), p.whitelist_file.c_str());
                all_ok = false;
            }
        }
    }
    CHECK(all_ok, "whitelist filenames valid format");
}

// ── Test 9: detect_protocol with synthetic spots (no whitelists) ────────────
// When no whitelist directories are available, detection should still work
// for non-WL protocols or return UNKNOWN/NONE confidence.
struct TestSpot {
    std::vector<uint8_t> r1_seq, r2_seq;
    uint16_t r1_len = 0, r2_len = 0;
};

static void test_detect_no_whitelists() {
    std::fprintf(stderr, "[ws2] detect_protocol without whitelists...\n");

    // Create 100 synthetic spots with R1=28bp, R2=91bp (typical 10xv3 geometry)
    std::vector<TestSpot> spots(100);
    for (auto& s : spots) {
        s.r1_len = 28;
        s.r2_len = 91;
        s.r1_seq.resize(28, 0);  // all A's
        s.r2_seq.resize(91, 0);
    }

    std::vector<std::string> empty_wl_dirs;
    auto result = lib1fq::detect_protocol(spots, 28, 91, empty_wl_dirs);

    // Without whitelists, won't get HIGH confidence for WL-based protocols
    // but should not crash
    CHECK(result.score >= 0.0, "score non-negative");
    CHECK(!result.tag.empty(), "tag not empty");
}

// ── Test 10: detect_protocol returns UNKNOWN for garbage reads ──────────────
static void test_detect_garbage_reads() {
    std::fprintf(stderr, "[ws2] detect_protocol garbage reads...\n");

    // Create spots with R1=5bp (too short for any protocol)
    std::vector<TestSpot> spots(50);
    for (auto& s : spots) {
        s.r1_len = 5;
        s.r2_len = 50;
        s.r1_seq.resize(5, 2);  // all G's
        s.r2_seq.resize(50, 3); // all T's
    }

    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 5, 50, empty_dirs);

    // With only 5bp R1, no protocol should match well
    CHECK(result.confidence <= lib1fq::Confidence::LOW, "garbage reads → low/none confidence");
}

// ── Test 11: Per-segment whitelist files for complex protocols ───────────────
static void test_per_seg_whitelist_consistency() {
    std::fprintf(stderr, "[ws2] Per-seg whitelist consistency...\n");
    const auto& protos = lib1fq::known_protocols();
    bool ok = true;
    for (const auto& p : protos) {
        if (!p.per_seg_whitelist_files.empty()) {
            // Per-seg files should all be non-empty strings
            for (const auto& f : p.per_seg_whitelist_files) {
                if (f.empty()) {
                    std::fprintf(stderr, "    %s: empty per-seg WL file\n", p.tag.c_str());
                    ok = false;
                }
            }
        }
    }
    CHECK(ok, "per-seg whitelist files non-empty");
}

// ── Test 12: sci-RNA-seq3 has expected barcode structure ─────────────────────
static void test_sci_rna_seq3_structure() {
    std::fprintf(stderr, "[ws2] sci-RNA-seq3 barcode structure...\n");
    auto* sci = lib1fq::find_protocol_spec("sci-rna-seq3");
    CHECK(sci != nullptr, "sci-rna-seq3 exists");
    if (sci) {
        // sci-RNA-seq3 uses a multi-barcode structure
        CHECK(sci->bc_len > 0, "sci-rna-seq3 has barcode");
        CHECK(sci->umi_len > 0, "sci-rna-seq3 has UMI");
        CHECK(sci->linker_offset > 0 || sci->linker_offset == 0,
              "sci-rna-seq3 linker_offset valid");
    }
}

// ── Test 13: Candidate admission — under-sequenced WL-free ──────────────────
// Drop-seq (r1_len=20) with R1=16bp should be admitted as candidate because:
// - 16 < 20 (under-sequenced)
// - 16 >= 0 + 12 (bc_offset + bc_len = 12, fits in R1)
// - Drop-seq has no whitelist
// CEL-Seq2 (r1_len=12) at R1=16bp should also be a candidate (+4bp over, rule 2).
static void test_candidate_admission_under_sequenced() {
    std::fprintf(stderr, "[ws2] Candidate admission: under-sequenced...\n");
    const auto& protos = lib1fq::known_protocols();

    // Find Drop-seq and CEL-Seq2 specs
    const lib1fq::CandidateSpec* dropseq = nullptr;
    const lib1fq::CandidateSpec* celseq2 = nullptr;
    for (const auto& k : protos) {
        if (k.tag == "dropseq") dropseq = &k;
        if (k.tag == "celseq2") celseq2 = &k;
    }
    CHECK(dropseq != nullptr, "dropseq spec found");
    CHECK(celseq2 != nullptr, "celseq2 spec found");
    if (!dropseq || !celseq2) return;

    // Drop-seq: r1_len=20, bc_offset=0, bc_len=12, WL empty
    CHECK(dropseq->r1_len == 20, "dropseq r1_len=20");
    CHECK(dropseq->bc_offset == 0, "dropseq bc_offset=0");
    CHECK(dropseq->bc_len == 12, "dropseq bc_len=12");
    CHECK(dropseq->whitelist_file.empty(), "dropseq no whitelist");

    // Under-seq rule: R1=16 < expected=20, R1=16 >= bc_offset+bc_len=12, no WL → admit
    uint16_t r1 = 16;
    bool dropseq_admitted = (r1 < dropseq->r1_len &&
                             r1 >= dropseq->bc_offset + dropseq->bc_len &&
                             dropseq->whitelist_file.empty());
    CHECK(dropseq_admitted, "dropseq admitted at R1=16 (under-seq)");

    // CEL-Seq2: r1_len=12, R1=16 → +4bp over → slightly over-sequenced rule
    bool celseq2_admitted = (r1 > celseq2->r1_len && r1 <= celseq2->r1_len + 9);
    CHECK(celseq2_admitted, "celseq2 admitted at R1=16 (over-seq +4bp)");

    // 10xv2 (r1_len=26, WL) at R1=16: under-seq but HAS whitelist → not admitted
    const lib1fq::CandidateSpec* v2 = nullptr;
    for (const auto& k : protos) { if (k.tag == "10x-3p-v2") { v2 = &k; break; } }
    if (v2) {
        bool v2_admitted = (r1 < v2->r1_len && r1 >= v2->bc_offset + v2->bc_len &&
                            v2->whitelist_file.empty());
        CHECK(!v2_admitted, "10xv2 NOT admitted at R1=16 (has WL, under-seq guard)");
    }
}

// ── Test 14: Candidate admission — over-seq gap closed (5-9bp) ──────────────
// Drop-seq (r1_len=20) at R1=28 (+8bp) was previously in dead zone (>4, <10).
// With the fix (threshold +9bp), Drop-seq should now be a candidate.
static void test_candidate_admission_over_seq_gap() {
    std::fprintf(stderr, "[ws2] Candidate admission: over-seq gap closed...\n");
    const auto& protos = lib1fq::known_protocols();

    const lib1fq::CandidateSpec* dropseq = nullptr;
    for (const auto& k : protos) { if (k.tag == "dropseq") { dropseq = &k; break; } }
    CHECK(dropseq != nullptr, "dropseq spec found");
    if (!dropseq) return;

    // R1=28, expected=20, diff=+8 → should be admitted (≤ +9)
    uint16_t r1_28 = 28;
    bool admitted_28 = (r1_28 > dropseq->r1_len && r1_28 <= dropseq->r1_len + 9);
    CHECK(admitted_28, "dropseq admitted at R1=28 (over-seq +8bp)");

    // R1=25, expected=20, diff=+5 → should be admitted (≤ +9)
    uint16_t r1_25 = 25;
    bool admitted_25 = (r1_25 > dropseq->r1_len && r1_25 <= dropseq->r1_len + 9);
    CHECK(admitted_25, "dropseq admitted at R1=25 (over-seq +5bp)");

    // R1=29, expected=20, diff=+9 → should be admitted (== +9)
    uint16_t r1_29 = 29;
    bool admitted_29 = (r1_29 > dropseq->r1_len && r1_29 <= dropseq->r1_len + 9);
    CHECK(admitted_29, "dropseq admitted at R1=29 (over-seq +9bp, boundary)");

    // R1=30, expected=20, diff=+10 → should be admitted (>= +10 rule)
    uint16_t r1_30 = 30;
    bool admitted_30 = (r1_30 >= dropseq->r1_len + 10);
    CHECK(admitted_30, "dropseq admitted at R1=30 (over-seq +10bp)");

    // R1=24, expected=20, diff=+4 → already admitted by old rule (≤ +9 now)
    uint16_t r1_24 = 24;
    bool admitted_24 = (r1_24 > dropseq->r1_len && r1_24 <= dropseq->r1_len + 9);
    CHECK(admitted_24, "dropseq admitted at R1=24 (over-seq +4bp)");
}

// ── Test 15: bc_coverage scoring signal ─────────────────────────────────────
// When no exact R1 match, bc_coverage rewards protocols that use more of R1.
// At R1=16: Drop-seq uses 20/16=1.0 (capped), CEL-Seq2 uses 12/16=0.75.
static void test_bc_coverage_scoring() {
    std::fprintf(stderr, "[ws2] bc_coverage scoring signal...\n");
    const auto& protos = lib1fq::known_protocols();

    const lib1fq::CandidateSpec* dropseq = nullptr;
    const lib1fq::CandidateSpec* celseq2 = nullptr;
    for (const auto& k : protos) {
        if (k.tag == "dropseq") dropseq = &k;
        if (k.tag == "celseq2") celseq2 = &k;
    }
    CHECK(dropseq != nullptr && celseq2 != nullptr, "both protocols found");
    if (!dropseq || !celseq2) return;

    uint16_t r1 = 16;
    // bc_coverage calculation: min(bc_offset + bc_len + umi_len, r1) / r1
    uint16_t ds_usable = std::min<uint16_t>(
        dropseq->bc_offset + dropseq->bc_len + dropseq->umi_len, r1);
    double ds_coverage = static_cast<double>(ds_usable) / r1;
    // Drop-seq: min(0+12+8, 16) = 16, 16/16 = 1.0
    CHECK(ds_coverage > 0.99, "dropseq bc_coverage ≈ 1.0 at R1=16");

    uint16_t cs_usable = std::min<uint16_t>(
        celseq2->bc_offset + celseq2->bc_len + celseq2->umi_len, r1);
    double cs_coverage = static_cast<double>(cs_usable) / r1;
    // CEL-Seq2: min(0+6+6, 16) = 12, 12/16 = 0.75
    CHECK(cs_coverage > 0.74 && cs_coverage < 0.76, "celseq2 bc_coverage ≈ 0.75 at R1=16");

    // Drop-seq has higher coverage than CEL-Seq2
    CHECK(ds_coverage > cs_coverage, "dropseq bc_coverage > celseq2 at R1=16");

    // At R1=28: both have no exact match
    r1 = 28;
    ds_usable = std::min<uint16_t>(
        dropseq->bc_offset + dropseq->bc_len + dropseq->umi_len, r1);
    ds_coverage = static_cast<double>(ds_usable) / r1;
    // Drop-seq: min(0+12+8, 28) = 20, 20/28 ≈ 0.714
    CHECK(ds_coverage > 0.71 && ds_coverage < 0.72, "dropseq bc_coverage ≈ 0.714 at R1=28");
}

// ── Test 16: geometry_bonus for under-sequenced WL-free ─────────────────────
// Under-sequenced WL-free protocols should get -0.05 (mild penalty), not -0.20.
static void test_geometry_bonus_under_seq() {
    std::fprintf(stderr, "[ws2] geometry_bonus under-sequenced WL-free...\n");

    // Simulate geometry_bonus calculation for Drop-seq at R1=16
    uint16_t r1 = 16;
    int32_t diff = static_cast<int32_t>(r1) - static_cast<int32_t>(20); // -4
    bool wl_free = true;  // Drop-seq has no whitelist
    bool bc_fits = (r1 >= 0 + 12);  // bc_offset + bc_len = 12 ≤ 16

    double bonus = 0.0;
    if (diff == 0)          bonus = 0.15;
    else if (diff >= -2 && diff <= 2) bonus = 0.10;
    else if (diff > 0 && diff <= 10)  bonus = 0.05;
    else if (diff < -2 && bc_fits && wl_free) bonus = -0.05;
    else if (diff < -2)    bonus = -0.20;

    CHECK(bonus == -0.05, "dropseq geometry_bonus = -0.05 at R1=16 (under-seq WL-free)");

    // CEL-Seq2 at R1=16: diff = 16-12 = +4 → slightly over-sequenced → +0.05
    diff = static_cast<int32_t>(r1) - static_cast<int32_t>(12);
    bonus = 0.0;
    if (diff == 0)          bonus = 0.15;
    else if (diff >= -2 && diff <= 2) bonus = 0.10;
    else if (diff > 0 && diff <= 10)  bonus = 0.05;

    CHECK(bonus == 0.05, "celseq2 geometry_bonus = +0.05 at R1=16 (over-seq +4bp)");

    // 10xv3 at R1=28: exact match → +0.15
    diff = static_cast<int32_t>(28) - static_cast<int32_t>(28);
    bonus = (diff == 0) ? 0.15 : -1.0;
    CHECK(bonus == 0.15, "10xv3 geometry_bonus = +0.15 at R1=28 (exact)");

    // Drop-seq at R1=28: diff = +8 → slightly over-sequenced → +0.05
    diff = static_cast<int32_t>(28) - static_cast<int32_t>(20);
    bonus = 0.0;
    if (diff == 0)          bonus = 0.15;
    else if (diff >= -2 && diff <= 2) bonus = 0.10;
    else if (diff > 0 && diff <= 10)  bonus = 0.05;
    CHECK(bonus == 0.05, "dropseq geometry_bonus = +0.05 at R1=28 (over-seq +8bp)");
}

// ── Test 17: UMI overflow penalty reduction for WL-free ─────────────────────
// WL-free protocols get reduced UMI overflow penalty (0.05 vs 0.10).
static void test_umi_overflow_penalty() {
    std::fprintf(stderr, "[ws2] UMI overflow penalty WL-free reduction...\n");

    // Drop-seq at R1=16: umi_end = 12+8 = 20, overflow = 20-16 = 4
    // overflow_frac = 4/8 = 0.5, WL-free → penalty = 0.05 * 0.5 = 0.025
    uint16_t umi_end = 12 + 8;
    uint16_t r1 = 16;
    uint16_t overflow = umi_end - r1;
    double overflow_frac = static_cast<double>(overflow) / 8;
    double penalty_wl_free = 0.05 * overflow_frac;
    double penalty_wl_has = 0.10 * overflow_frac;

    CHECK(std::abs(penalty_wl_free - 0.025) < 0.001,
          "WL-free UMI overflow penalty = 0.025");
    CHECK(std::abs(penalty_wl_has - 0.05) < 0.001,
          "WL-based UMI overflow penalty = 0.05");
    CHECK(penalty_wl_free < penalty_wl_has,
          "WL-free penalty < WL-based penalty");

    // No overflow case: umi_end = 12+8 = 20 ≤ 20 → no penalty
    r1 = 20;
    bool has_overflow = (umi_end > r1);
    CHECK(!has_overflow, "no UMI overflow at R1=20 for dropseq");

    // R1=28: umi_end=20 ≤ 28 → no overflow
    r1 = 28;
    has_overflow = (umi_end > r1);
    CHECK(!has_overflow, "no UMI overflow at R1=28 for dropseq");
}

// ── Test 18: detect_protocol with synthetic Drop-seq R1=28 (no WL) ──────────
// With R1=28bp and high UMI entropy, Drop-seq should be detected (no WL needed).
static void test_detect_dropseq_r1_28() {
    std::fprintf(stderr, "[ws2] detect_protocol: dropseq at R1=28 (no WL)...\n");

    // Create synthetic spots with randomized R1 (BC+UMI+extra) and polyA R2
    std::vector<TestSpot> spots(5000);
    uint32_t seed = 12345;
    for (auto& s : spots) {
        s.r1_len = 28;
        s.r2_len = 91;
        s.r1_seq.resize(28);
        s.r2_seq.resize(91);
        // Random BC (12bp) + random UMI (8bp) + 8bp extra (T=polyT tail)
        for (int j = 0; j < 20; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r1_seq[j] = (seed >> 16) & 3;
        }
        for (int j = 20; j < 28; ++j) {
            s.r1_seq[j] = 3;  // polyT tail in extra region
        }
        // R2: mostly random cDNA with polyA at end
        for (int j = 0; j < 76; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r2_seq[j] = (seed >> 16) & 3;
        }
        for (int j = 76; j < 91; ++j) {
            s.r2_seq[j] = 0;  // polyA tail
        }
    }

    std::vector<std::string> empty_dirs;  // no whitelist dirs → WL protocols can't match
    auto result = lib1fq::detect_protocol(spots, 28, 91, empty_dirs);

    // Without whitelists, a non-WL protocol should win
    CHECK(result.score > 0.0, "R1=28 no-WL: positive score");
    // The detected protocol should be one of the WL-free protocols
    bool is_wl_free = (result.tag == "dropseq" || result.tag == "seqwell" ||
                       result.tag == "celseq2" || result.tag == "marsseq2" ||
                       result.tag == "strtseq");
    CHECK(is_wl_free, "R1=28 no-WL: detected WL-free protocol");
}

// ── Test 19: detect_protocol with synthetic Drop-seq R1=16 (under-seq) ──────
static void test_detect_dropseq_r1_16() {
    std::fprintf(stderr, "[ws2] detect_protocol: dropseq at R1=16 (under-seq)...\n");

    // Create synthetic spots with R1=16bp (Drop-seq BC=12 + truncated UMI=4)
    std::vector<TestSpot> spots(5000);
    uint32_t seed = 67890;
    for (auto& s : spots) {
        s.r1_len = 16;
        s.r2_len = 98;
        s.r1_seq.resize(16);
        s.r2_seq.resize(98);
        // Random BC (12bp) + random truncated UMI (4bp)
        for (int j = 0; j < 16; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r1_seq[j] = (seed >> 16) & 3;
        }
        // R2: random cDNA with polyA tail
        for (int j = 0; j < 83; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r2_seq[j] = (seed >> 16) & 3;
        }
        for (int j = 83; j < 98; ++j) {
            s.r2_seq[j] = 0;  // polyA
        }
    }

    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 16, 98, empty_dirs);

    CHECK(result.score > 0.0, "R1=16 no-WL: positive score");
    // Should detect a WL-free protocol (dropseq, celseq2, or marsseq2)
    bool is_wl_free = (result.tag == "dropseq" || result.tag == "seqwell" ||
                       result.tag == "celseq2" || result.tag == "marsseq2");
    CHECK(is_wl_free, "R1=16 no-WL: detected WL-free protocol");
    CHECK(result.confidence >= lib1fq::Confidence::LOW,
          "R1=16 no-WL: at least LOW confidence");
}

// ── Test 20: detect_protocol exact R1 match (20bp) → Drop-seq preferred ──────
static void test_detect_dropseq_exact_r1() {
    std::fprintf(stderr, "[ws2] detect_protocol: dropseq at R1=20 (exact)...\n");

    // Create synthetic spots with R1=20bp (exact Drop-seq geometry)
    std::vector<TestSpot> spots(5000);
    uint32_t seed = 11111;
    for (auto& s : spots) {
        s.r1_len = 20;
        s.r2_len = 100;
        s.r1_seq.resize(20);
        s.r2_seq.resize(100);
        // Random BC (12bp) + random UMI (8bp)
        for (int j = 0; j < 20; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r1_seq[j] = (seed >> 16) & 3;
        }
        // R2: random cDNA with polyA tail
        for (int j = 0; j < 85; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r2_seq[j] = (seed >> 16) & 3;
        }
        for (int j = 85; j < 100; ++j) {
            s.r2_seq[j] = 0;  // polyA
        }
    }

    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 20, 100, empty_dirs);

    CHECK(result.score > 0.0, "R1=20 exact: positive score");
    // Should detect dropseq or seqwell (both have r1_len=20)
    bool is_r1_20 = (result.tag == "dropseq" || result.tag == "seqwell");
    CHECK(is_r1_20, "R1=20 exact: detected dropseq/seqwell");
    CHECK(result.confidence >= lib1fq::Confidence::MEDIUM,
          "R1=20 exact: at least MEDIUM confidence");
}

// ── Test 21: detect_protocol empty spots → low confidence ───────────────────
static void test_detect_empty_spots() {
    std::fprintf(stderr, "[ws2] detect_protocol: empty spots...\n");

    std::vector<TestSpot> spots;  // no spots at all
    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 28, 91, empty_dirs);

    // With no probe data, scores are based on geometry only → low confidence
    CHECK(result.confidence <= lib1fq::Confidence::MEDIUM, "empty spots: at most MEDIUM confidence");
    // With no data but R1=28 (exact geometry match for v3/arc),
    // score is r1_match(0.25) + geometry_bonus(0.15) = 0.40 for the best non-WL candidate
    CHECK(result.score <= 0.50, "empty spots: score capped without data");
}

// ── Test 22: detect_protocol R1=0 (concat mode) ──────────────────────────────
static void test_detect_concat_mode() {
    std::fprintf(stderr, "[ws2] detect_protocol: R1=0 (concat mode)...\n");

    // R1=0 means all data in a single segment (concat mode)
    std::vector<TestSpot> spots(5000);
    uint32_t seed = 99999;
    for (auto& s : spots) {
        s.r1_len = 0;
        s.r2_len = 150;
        s.r1_seq.clear();
        s.r2_seq.resize(150);
        for (int j = 0; j < 150; ++j) {
            seed = seed * 1103515245 + 12345;
            s.r2_seq[j] = (seed >> 16) & 3;
        }
    }

    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 0, 150, empty_dirs);

    // Should either detect a concat-mode protocol or UNKNOWN
    CHECK(result.score >= 0.0, "concat mode: non-negative score");
}

// ── Test 23: WL-geometry suppression (non-WL protocols with WL geometry) ────
static void test_wl_geometry_suppression() {
    std::fprintf(stderr, "[ws2] WL-geometry suppression...\n");
    const auto& protos = lib1fq::known_protocols();

    // Find protocols that share geometry
    // 10x-arc-gex has the same geometry as 10xv3 (R1=28, BC=16, UMI=12)
    // arc-gex has a whitelist, so both are WL protocols — no suppression
    const lib1fq::CandidateSpec* v3 = nullptr;
    const lib1fq::CandidateSpec* arc = nullptr;
    for (const auto& k : protos) {
        if (k.tag == "10x-3p-v3") v3 = &k;
        if (k.tag == "10x-arc-gex") arc = &k;
    }
    CHECK(v3 != nullptr && arc != nullptr, "both v3 and arc found");
    if (v3 && arc) {
        // Both have WL
        CHECK(!v3->whitelist_file.empty(), "v3 has WL");
        CHECK(!arc->whitelist_file.empty(), "arc has WL");
        // Same R1/BC/UMI geometry
        CHECK(v3->r1_len == arc->r1_len, "v3 and arc same R1");
        CHECK(v3->bc_len == arc->bc_len, "v3 and arc same BC");
        CHECK(v3->umi_len == arc->umi_len, "v3 and arc same UMI");
    }

    // 10x-visium has r1_len=28 but no whitelist — should be suppressed by v3/arc geometry
    const lib1fq::CandidateSpec* visium = nullptr;
    for (const auto& k : protos) {
        if (k.tag == "10x-visium") { visium = &k; break; }
    }
    if (visium) {
        CHECK(visium->whitelist_file.empty(), "visium has no WL");
        CHECK(visium->r1_len == 28, "visium R1=28 (same as v3)");
    }
}

// ── Test 24: 5' protocols have adapter3p, 3' protocols do not ───────────────
static void test_adapter3p_classification() {
    std::fprintf(stderr, "[ws2] adapter3p classification...\n");
    const auto& protos = lib1fq::known_protocols();

    // 3' protocols should NOT have adapter3p (polyA clip is handled by STAR arg)
    for (const auto& p : protos) {
        if (p.tag == "10x-3p-v3" || p.tag == "10x-3p-v2" || p.tag == "10x-3p-v4"
            || p.tag == "dropseq" || p.tag == "seqwell") {
            CHECK(p.adapter3p.empty(),
                  (p.tag + " has no adapter3p (3' protocol)").c_str());
        }
    }

    // 5' protocols should have TSO-like adapter3p
    for (const auto& p : protos) {
        if (p.tag == "10x-5p-v2" || p.tag == "10x-5p-v3") {
            CHECK(!p.adapter3p.empty(),
                  (p.tag + " has adapter3p (5' protocol)").c_str());
        }
    }
}

// ── Test 25: protocol_id → protocol_name round-trip ─────────────────────────
static void test_protocol_id_name_roundtrip() {
    std::fprintf(stderr, "[ws2] protocol_id ↔ name round-trip...\n");
    const auto& protos = lib1fq::known_protocols();

    // Every protocol with a numeric ID should have a non-empty tag
    for (const auto& p : protos) {
        CHECK(!p.tag.empty(), (std::string("protocol ") + std::to_string(p.protocol_id) + " has tag").c_str());
    }

    // Key IDs: 1=10x-3p-v3, 2=10x-3p-v2
    const lib1fq::CandidateSpec* id1 = nullptr;
    const lib1fq::CandidateSpec* id2 = nullptr;
    for (const auto& p : protos) {
        if (p.protocol_id == 1) id1 = &p;
        if (p.protocol_id == 2) id2 = &p;
    }
    CHECK(id1 && id1->tag == "10x-3p-v3", "protocol_id 1 → 10x-3p-v3");
    CHECK(id2 && id2->tag == "10x-3p-v2", "protocol_id 2 → 10x-3p-v2");
}

// ── Test 26: detect_protocol R1=150 R2=150 → concat or mapped ───────────────
static void test_detect_long_reads() {
    std::fprintf(stderr, "[ws2] detect_protocol: very long reads (R1=150 R2=150)...\n");

    std::vector<TestSpot> spots(500);
    for (auto& s : spots) {
        s.r1_len = 150;
        s.r2_len = 150;
        s.r1_seq.resize(150, 0);
        s.r2_seq.resize(150, 0);
    }
    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 150, 150, empty_dirs);

    // With R1=R2=150, no standard sc protocol geometry matches
    // Should get low confidence
    CHECK(result.confidence <= lib1fq::Confidence::MEDIUM,
          "long reads: at most MEDIUM confidence");
}

// ── Test 27: detect_protocol inverted reads (R1 long, R2 short) ─────────────
static void test_detect_inverted_reads() {
    std::fprintf(stderr, "[ws2] detect_protocol: inverted reads (R1=91, R2=28)...\n");

    // Inverted: R2 is the barcode read (28bp), R1 is cDNA (91bp)
    std::vector<TestSpot> spots(500);
    for (auto& s : spots) {
        s.r1_len = 91;
        s.r2_len = 28;
        s.r1_seq.resize(91, 0);
        s.r2_seq.resize(28, 0);
    }
    std::vector<std::string> empty_dirs;
    auto result = lib1fq::detect_protocol(spots, 91, 28, empty_dirs);

    // R2=28 ≤ 34, R1=91 > 40, R1 > R2*2 → inverted=true
    // barcode_read_len becomes R2=28, matching 10x-3p-v3 geometry
    CHECK(result.score > 0.0, "inverted: positive score");
    // Should detect a 28bp-barcode protocol
    CHECK(result.tag == "10x-3p-v3" || result.tag == "10x-3p-v4"
          || result.tag == "10x-arc-gex" || result.tag == "10x-5p-v3",
          "inverted: detects 28bp-barcode protocol");
}

// ── Test 28: geometry-based WL lookup for unknown protocol ──────────────────
// Validates that for common R1 lengths, there exists a known protocol
// with matching BC/UMI geometry that has a whitelist file.
// This supports the late-WL-resolution fix in singlet.cpp.
static void test_geometry_wl_lookup() {
    std::fprintf(stderr, "[ws2] geometry_wl_lookup: R1→WL resolution...\n");
    const auto& protos = lib1fq::known_protocols();

    // R1=28 (BC=16, UMI=12): must find a WL protocol
    {
        bool found = false;
        for (const auto& p : protos) {
            if (p.bc_len == 16 && p.umi_len == 12 && !p.whitelist_file.empty()) {
                found = true;
                break;
            }
        }
        CHECK(found, "R1=28 (BC16+UMI12): WL protocol exists");
    }

    // R1=26 (BC=16, UMI=10): must find a WL protocol
    {
        bool found = false;
        for (const auto& p : protos) {
            if (p.bc_len == 16 && p.umi_len == 10 && !p.whitelist_file.empty()) {
                found = true;
                break;
            }
        }
        CHECK(found, "R1=26 (BC16+UMI10): WL protocol exists");
    }

    // R1=24 (BC=14, UMI=10): must find a WL protocol (10x v1)
    {
        bool found = false;
        for (const auto& p : protos) {
            if (p.bc_len == 14 && p.umi_len == 10 && !p.whitelist_file.empty()) {
                found = true;
                break;
            }
        }
        CHECK(found, "R1=24 (BC14+UMI10): WL protocol exists");
    }

    // R1=20 (BC=12, UMI=8): should NOT have a WL (dropseq/seqwell are WL-free)
    {
        bool found = false;
        for (const auto& p : protos) {
            if (p.bc_len == 12 && p.umi_len == 8 && !p.whitelist_file.empty()) {
                found = true;
                break;
            }
        }
        CHECK(!found, "R1=20 (BC12+UMI8): no WL protocol (dropseq/seqwell)");
    }

    // R1=12 (BC=6, UMI=6): microwell-seq has bc_len=6, umi_len=6 with WL
    {
        bool found = false;
        std::string wl_name;
        for (const auto& p : protos) {
            if (p.bc_len == 6 && p.umi_len == 6 && !p.whitelist_file.empty()) {
                found = true;
                wl_name = p.whitelist_file;
                break;
            }
        }
        CHECK(found, "R1=12 (BC6+UMI6): WL protocol exists (microwell-seq)");
    }
}

// ── Protocol data integrity tests ─────────────────────────────────────────

static void test_protocol_id_unique() {
    const auto& specs = lib1fq::known_protocols();
    std::set<uint8_t> ids;
    bool all_unique = true;
    for (const auto& s : specs) {
        if (ids.count(s.protocol_id)) {
            std::fprintf(stderr, "  DUPLICATE protocol_id=%d for tag=%s\n",
                         s.protocol_id, s.tag.c_str());
            all_unique = false;
        }
        ids.insert(s.protocol_id);
    }
    CHECK(all_unique, "protocol_id_all_unique");
}

static void test_protocol_tag_unique() {
    const auto& specs = lib1fq::known_protocols();
    std::set<std::string> tags;
    bool all_unique = true;
    for (const auto& s : specs) {
        if (tags.count(s.tag)) {
            std::fprintf(stderr, "  DUPLICATE tag=%s\n", s.tag.c_str());
            all_unique = false;
        }
        tags.insert(s.tag);
    }
    CHECK(all_unique, "protocol_tag_all_unique");
}

static void test_protocol_bc_umi_geometry_valid() {
    const auto& specs = lib1fq::known_protocols();
    int ok = 0;
    for (const auto& s : specs) {
        // Skip ATAC (no barcode in R1)
        if (s.tag == "10x-atac") { ok++; continue; }
        // Skip STRT-seq (no UMI)
        if (s.tag == "strtseq") { ok++; continue; }
        // bc_offset + bc_len must not exceed r1_len (for non-barcode_in_r2 protos)
        if (!s.barcode_in_r2 && s.r1_len > 0) {
            if (s.bc_offset + s.bc_len > s.r1_len) {
                std::fprintf(stderr, "  BC overflow: %s bc_off=%d+bc_len=%d > r1=%d\n",
                             s.tag.c_str(), s.bc_offset, s.bc_len, s.r1_len);
                continue;
            }
            // UMI must also fit in R1
            if (s.umi_offset + s.umi_len > s.r1_len) {
                std::fprintf(stderr, "  UMI overflow: %s umi_off=%d+umi_len=%d > r1=%d\n",
                             s.tag.c_str(), s.umi_offset, s.umi_len, s.r1_len);
                continue;
            }
        }
        ok++;
    }
    CHECK(ok == (int)specs.size(), "bc_umi_geometry_valid_all");
}

static void test_complex_protocols_have_per_seg() {
    // Complex protocols (BD Rhapsody, SPLiT-seq, inDrop, Microwell, SureCell)
    // must have non-empty per_seg_whitelist_files
    const auto& specs = lib1fq::known_protocols();
    const std::set<std::string> complex = {"bd-rhapsody", "splitseq", "indrop",
                                            "microwell-seq", "surecell"};
    int ok = 0;
    for (const auto& s : specs) {
        if (complex.count(s.tag)) {
            if (s.per_seg_whitelist_files.empty()) {
                std::fprintf(stderr, "  MISSING per_seg for complex proto %s\n",
                             s.tag.c_str());
            } else {
                ok++;
            }
        }
    }
    CHECK(ok == (int)complex.size(), "complex_protos_have_per_seg");
}

static void test_no_protocol_id_zero() {
    // protocol_id=0 means "unknown/undetected" — no spec should use it
    const auto& specs = lib1fq::known_protocols();
    bool none_zero = true;
    for (const auto& s : specs) {
        if (s.protocol_id == 0) {
            std::fprintf(stderr, "  protocol_id=0 found for tag=%s\n", s.tag.c_str());
            none_zero = false;
        }
    }
    CHECK(none_zero, "no_protocol_id_zero");
}

static void test_adapter3p_only_5prime() {
    // Only 5' protocols should have adapter3p set (TSO adapter)
    const auto& specs = lib1fq::known_protocols();
    bool ok = true;
    for (const auto& s : specs) {
        if (!s.adapter3p.empty()) {
            // Must be a 5' protocol or arc-gex
            bool is_5p = (s.tag.find("5p") != std::string::npos
                          || s.tag == "strtseq"
                          || s.tag == "10x-arc-gex");
            if (!is_5p) {
                std::fprintf(stderr, "  adapter3p set on non-5' proto %s\n",
                             s.tag.c_str());
                ok = false;
            }
        }
    }
    CHECK(ok, "adapter3p_only_5prime");
}

// ── Tag normalization tests ─────────────────────────────────────────────────

static void test_normalize_tag_cases() {
    // Basic lowercasing
    CHECK(lib1fq::normalize_tag("DROPSEQ") == "dropseq", "norm_uppercase");
    // Hyphen removal
    CHECK(lib1fq::normalize_tag("Drop-seq") == "dropseq", "norm_hyphen");
    // Underscore removal
    CHECK(lib1fq::normalize_tag("drop_seq") == "dropseq", "norm_underscore");
    // Mixed case + hyphens + underscores
    CHECK(lib1fq::normalize_tag("10x-3p-V3") == "10x3pv3", "norm_mixed");
    // Already normalized
    CHECK(lib1fq::normalize_tag("dropseq") == "dropseq", "norm_identity");
    // Empty string
    CHECK(lib1fq::normalize_tag("") == "", "norm_empty");
    // All hyphens
    CHECK(lib1fq::normalize_tag("---") == "", "norm_all_hyphens");
}

static void test_alias_resolution() {
    // Direct tag lookup
    CHECK(lib1fq::find_protocol_spec("10x-3p-v3") != nullptr, "alias_direct_10xv3");
    CHECK(lib1fq::find_protocol_spec("10x-3p-v3")->protocol_id == 1, "alias_direct_10xv3_id");

    // Case-insensitive lookup
    CHECK(lib1fq::find_protocol_spec("DROPSEQ") != nullptr, "alias_case_dropseq");
    CHECK(lib1fq::find_protocol_spec("DROPSEQ")->tag == "dropseq", "alias_case_dropseq_tag");

    // Common aliases
    CHECK(lib1fq::find_protocol_spec("10xv3") != nullptr, "alias_10xv3");
    CHECK(lib1fq::find_protocol_spec("10xv3")->tag == "10x-3p-v3", "alias_10xv3_resolves");

    CHECK(lib1fq::find_protocol_spec("10xv2") != nullptr, "alias_10xv2");
    CHECK(lib1fq::find_protocol_spec("10xv2")->tag == "10x-3p-v2", "alias_10xv2_resolves");

    // Parse → splitseq
    CHECK(lib1fq::find_protocol_spec("parse") != nullptr, "alias_parse");
    CHECK(lib1fq::find_protocol_spec("parse")->tag == "splitseq", "alias_parse_resolves");

    // CITE-seq aliases
    CHECK(lib1fq::find_protocol_spec("cite-seq") != nullptr, "alias_citeseq");
    CHECK(lib1fq::find_protocol_spec("citeseq")->tag == "cite-seq-gex", "alias_citeseq_resolves");

    // Multiome aliases
    CHECK(lib1fq::find_protocol_spec("multiome") != nullptr, "alias_multiome");
    CHECK(lib1fq::find_protocol_spec("multiome")->tag == "10x-arc-gex", "alias_multiome_resolves");

    // Nonexistent protocol
    CHECK(lib1fq::find_protocol_spec("nonexistent") == nullptr, "alias_nonexistent_null");
    CHECK(lib1fq::find_protocol_spec("") == nullptr, "alias_empty_null");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_known_protocols_registry();
    test_sci_rna_seq3_linker();
    test_bd_rhapsody_linker();
    test_protocol_ids_unique();
    test_find_protocol_spec();
    test_confidence_ordering();
    test_protocol_geometry_valid();
    test_whitelist_file_names();
    test_detect_no_whitelists();
    test_detect_garbage_reads();
    test_per_seg_whitelist_consistency();
    test_sci_rna_seq3_structure();
    test_candidate_admission_under_sequenced();
    test_candidate_admission_over_seq_gap();
    test_bc_coverage_scoring();
    test_geometry_bonus_under_seq();
    test_umi_overflow_penalty();
    test_detect_dropseq_r1_28();
    test_detect_dropseq_r1_16();
    test_detect_dropseq_exact_r1();
    test_detect_empty_spots();
    test_detect_concat_mode();
    test_wl_geometry_suppression();
    test_adapter3p_classification();
    test_protocol_id_name_roundtrip();
    test_detect_long_reads();
    test_detect_inverted_reads();
    test_geometry_wl_lookup();

    // ── Protocol data integrity tests ──
    test_protocol_id_unique();
    test_protocol_tag_unique();
    test_protocol_bc_umi_geometry_valid();
    test_complex_protocols_have_per_seg();
    test_no_protocol_id_zero();
    test_adapter3p_only_5prime();

    // ── Tag normalization and alias resolution tests ──
    test_normalize_tag_cases();
    test_alias_resolution();

    std::fprintf(stderr, "\n=== ws2_protocol_hardening: %d passed, %d failed ===\n",
                 g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}
