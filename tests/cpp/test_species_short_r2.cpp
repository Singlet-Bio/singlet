// test/test_species_short_r2.cpp — Unit tests for B-G2-5 AUTOFIX-SPECIES-VAL-R2-SHORT
//
// Tests the logic that should reject LOW-confidence protocol detections when
// R2 length < 50 bp.  Because the rejection happens inside singlet main's
// .1fq processing block (which requires a full .1fq file + genome), we test
// the guard condition in isolation using the same comparands singlet.cpp uses:
//
//   hdr.confidence <= singlet::fq::Confidence::LOW (=1)
//   detected_r2_len > 0 && detected_r2_len < 50
//
// We also verify the Confidence enum values haven't drifted, and test all
// (confidence × r2_len) boundary combinations expected to pass/reject.

#include "singlet/fq/types.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", msg); } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", msg); } \
} while(0)

// Mirror of the guard condition in singlet.cpp B-G2-5 block.
// Returns true when the sample should be REJECTED (autodetect_species_fail).
static bool should_reject_r2_short(singlet::fq::Confidence conf, int r2_len, bool is_atac) {
    if (is_atac) return false;  // ATAC short-R2 is expected
    return conf <= singlet::fq::Confidence::LOW
        && r2_len > 0
        && r2_len < 50;
}

static void test_confidence_enum_values() {
    std::fprintf(stderr, "[test_species_short_r2] Confidence enum values...\n");
    // These must stay stable — singlet.cpp casts hdr.confidence (uint8_t) against them
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::NONE)   == 0, "NONE == 0");
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::LOW)    == 1, "LOW == 1");
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::MEDIUM) == 2, "MEDIUM == 2");
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::HIGH)   == 3, "HIGH == 3");
}

static void test_reject_low_conf_short_r2() {
    std::fprintf(stderr, "[test_species_short_r2] Reject: LOW confidence + short R2...\n");

    // Case from failure registry: R2=30bp, confidence=LOW, non-ATAC → REJECT
    CHECK(should_reject_r2_short(singlet::fq::Confidence::LOW, 30, false),
          "LOW conf + 30bp R2 → reject");
    CHECK(should_reject_r2_short(singlet::fq::Confidence::LOW, 49, false),
          "LOW conf + 49bp R2 → reject (boundary)");
    CHECK(should_reject_r2_short(singlet::fq::Confidence::NONE, 30, false),
          "NONE conf + 30bp R2 → reject");
}

static void test_pass_cases() {
    std::fprintf(stderr, "[test_species_short_r2] Pass: should NOT reject...\n");

    // R2 ≥ 50 bp even with LOW confidence → allow (could be a valid short-R2 protocol)
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 50, false),
          "LOW conf + 50bp R2 → allow (at threshold)");
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 91, false),
          "LOW conf + 91bp R2 → allow");

    // High / medium confidence → always allow regardless of R2 length
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::MEDIUM, 30, false),
          "MEDIUM conf + 30bp R2 → allow");
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::HIGH, 30, false),
          "HIGH conf + 30bp R2 → allow");
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::MANUAL, 30, false),
          "MANUAL conf + 30bp R2 → allow");

    // r2_len == 0 (unknown) → allow (no length info, cannot decide)
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 0, false),
          "LOW conf + r2_len=0 → allow (unknown R2 len)");

    // ATAC mode → always allow (ATAC R2 barcode is legitimately short)
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 16, true),
          "LOW conf + 16bp R2 + is_atac → allow");
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 30, true),
          "LOW conf + 30bp R2 + is_atac → allow");
}

static void test_boundary_exact_50() {
    std::fprintf(stderr, "[test_species_short_r2] Boundary at exactly 50bp...\n");

    // < 50 → reject (if low conf)
    CHECK(should_reject_r2_short(singlet::fq::Confidence::LOW, 49, false),
          "r2_len=49 < 50 → reject");
    // >= 50 → pass
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 50, false),
          "r2_len=50 == 50 → pass");
    CHECK(!should_reject_r2_short(singlet::fq::Confidence::LOW, 51, false),
          "r2_len=51 > 50 → pass");
}

int main() {
    test_confidence_enum_values();
    test_reject_low_conf_short_r2();
    test_pass_cases();
    test_boundary_exact_50();

    std::fprintf(stderr, "\n[test_species_short_r2] Results: %d passed, %d failed\n",
                 g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
