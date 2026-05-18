// SPDX-License-Identifier: MIT
// test_protocol_confidence_override.cpp
// Unit tests for AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE logic.
//
// The override fires inside SraEncoder::encode() which requires real SRA data,
// so we test the two reusable primitives it relies on:
//   1. find_protocol_spec() correctly resolves catalog protocol names.
//   2. The override decision predicate produces the right outcome for all
//      combinations of (detected_confidence, metadata_protocol, detected_tag).
//
// We also test the Confidence enum ordering that the override depends on.

#include <iostream>
#include <string>

#include "singlet/fq/protocol.h"
#include "singlet/fq/types.h"

#define SINGLET_TEST_HARNESS_TERSE
#include "test_harness.h"  // CHECK(cond) + g_pass / g_fail

// ── Helper: simulate the override decision ───────────────────────────────────
// Mirrors the AUTOFIX-PROTOCOL-CONFIDENCE-OVERRIDE logic in sra_encoder.h.
// Returns {final_protocol_tag, final_protocol_id, override_fired}.
struct OverrideResult {
    std::string tag;
    uint8_t     protocol_id;
    bool        override_fired;
};

static OverrideResult apply_override(
    const std::string& cli_protocol,       // ecfg.protocol_tag ("" = auto)
    const std::string& metadata_protocol,  // ecfg.metadata_protocol
    const std::string& detected_tag,       // from auto-detection
    uint8_t            detected_id,
    singlet::fq::Confidence confidence)
{
    // If --protocol was given, committed; do not override.
    if (!cli_protocol.empty()) {
        return {detected_tag, detected_id, false};
    }

    // Catalog metadata always wins over VDB auto-detection at any confidence level.
    if (!metadata_protocol.empty()) {
        const singlet::fq::CandidateSpec* spec =
            singlet::fq::find_protocol_spec(metadata_protocol);
        if (spec && spec->tag != detected_tag) {
            return {spec->tag, spec->protocol_id, true};
        }
    }
    return {detected_tag, detected_id, false};
}

// ── Confidence enum ordering ──────────────────────────────────────────────────

static void test_confidence_ordering() {
    // Verify enum numeric values used throughout singlet.
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::NONE)   == 0);
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::LOW)    == 1);
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::MEDIUM) == 2);
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::HIGH)   == 3);
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::MANUAL) == 4);
    CHECK(static_cast<uint8_t>(singlet::fq::Confidence::FORCE)  == 5);
    // Ordering comparisons used by the override condition:
    CHECK(singlet::fq::Confidence::NONE   <  singlet::fq::Confidence::LOW);
    CHECK(singlet::fq::Confidence::LOW    <  singlet::fq::Confidence::MEDIUM);
    CHECK(singlet::fq::Confidence::MEDIUM <  singlet::fq::Confidence::HIGH);
    CHECK(singlet::fq::Confidence::HIGH   <  singlet::fq::Confidence::MANUAL);
}

// ── Witness sample 1: seqwell catalog vs 10x-visium detected (c=1) ──────────
// GSM3587947 / SRR8521680: detected=10x-visium c=1, catalog=seqwell

static void test_override_fires_seqwell_vs_visium_c1() {
    auto r = apply_override(
        "",             // no --protocol flag
        "seqwell",      // metadata.protocol from catalog
        "10x-visium",   // what the detector returned
        24,             // protocol_id for 10x-visium
        singlet::fq::Confidence::LOW  // c=1
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "seqwell");

    // Verify the spec has the correct CBlen/UMIlen (seqwell: CB=12, UMI=8)
    const singlet::fq::CandidateSpec* spec = singlet::fq::find_protocol_spec("seqwell");
    CHECK(spec != nullptr);
    if (spec) {
        CHECK(spec->bc_len  == 12);
        CHECK(spec->umi_len ==  8);
        CHECK(spec->r1_len  == 20);
    }
}

// ── Witness sample 2: 10xv3 catalog vs 10x-visium detected (c=2) ────────────
// GSM8197519 / SRR23974415: detected=10x-visium c=2, catalog=10xv3
// MEDIUM confidence → catalog WINS (catalog always overrides VDB at any confidence)

static void test_override_fires_medium_confidence() {
    auto r = apply_override(
        "",
        "10xv3",       // metadata.protocol
        "10x-visium",  // detected
        24,
        singlet::fq::Confidence::MEDIUM  // c=2 — catalog still wins
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "10x-3p-v3");  // 10xv3 alias resolves to canonical tag
}

// ── Witness sample 3: 10xv2 catalog vs quartzseq2 detected (c=1) ────────────
// GSM8808563 / SRR32452119: detected=quartzseq2 c=1, catalog=10xv2

static void test_override_fires_10xv2_vs_quartzseq2_c1() {
    auto r = apply_override(
        "",
        "10xv2",       // catalog alias for 10x-3p-v2
        "quartzseq2",  // wrong detection
        18,
        singlet::fq::Confidence::LOW
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "10x-3p-v2");  // alias resolved to canonical tag

    const singlet::fq::CandidateSpec* spec = singlet::fq::find_protocol_spec("10xv2");
    CHECK(spec != nullptr);
    if (spec) {
        CHECK(spec->bc_len  == 16);
        CHECK(spec->umi_len == 10);
    }
}

// ── No metadata: behavior unchanged (no override) ───────────────────────────

static void test_no_metadata_no_override() {
    auto r = apply_override(
        "",           // no --protocol
        "",           // no metadata protocol
        "10x-visium",
        24,
        singlet::fq::Confidence::LOW
    );
    CHECK(!r.override_fired);
    CHECK(r.tag == "10x-visium");
}

// ── Explicit --protocol wins even at low confidence ──────────────────────────

static void test_cli_protocol_wins() {
    auto r = apply_override(
        "dropseq",     // explicit --protocol flag
        "seqwell",     // metadata says otherwise
        "dropseq",     // detector would say dropseq (forced)
        6,
        singlet::fq::Confidence::MANUAL
    );
    CHECK(!r.override_fired);  // no override when --protocol was given
}

// ── High confidence: catalog STILL wins (VDB is never trusted over catalog) ───

static void test_override_fires_high_confidence() {
    auto r = apply_override(
        "",
        "dropseq",
        "10x-3p-v3",
        1,
        singlet::fq::Confidence::HIGH  // c=3 — catalog still overrides
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "dropseq");
}

// ── NONE confidence: override fires (even lower than LOW) ────────────────────

static void test_none_confidence_fires() {
    auto r = apply_override(
        "",
        "seqwell",
        "UNKNOWN",
        0,
        singlet::fq::Confidence::NONE  // c=0
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "seqwell");
}

// ── Unknown metadata protocol: no override, warning path ────────────────────

static void test_unknown_metadata_protocol_no_override() {
    // "totally-made-up-protocol" is not in known_protocols()
    CHECK(singlet::fq::find_protocol_spec("totally-made-up-protocol") == nullptr);

    auto r = apply_override(
        "",
        "totally-made-up-protocol",
        "10x-visium",
        24,
        singlet::fq::Confidence::LOW
    );
    // find_protocol_spec returns nullptr → override does not fire
    CHECK(!r.override_fired);
    CHECK(r.tag == "10x-visium");
}

// ── Same tag detected and in metadata: no override needed ────────────────────

static void test_same_tag_no_override() {
    auto r = apply_override(
        "",
        "seqwell",
        "seqwell",   // already correct
        16,
        singlet::fq::Confidence::LOW
    );
    CHECK(!r.override_fired);  // tags match → override did not fire
}

// ── DNBELAB: VDB returns quartzseq2 at MEDIUM confidence, catalog says dnbelab ──
// Root cause of AUTOFIX-DNBELAB-QUARTZSEQ2-MISDETECT:
// DNBelab-C4 samples mismatch quartzseq2 at MEDIUM confidence so the old
// confidence<=LOW gate silently kept the wrong protocol. Catalog must win.

static void test_override_fires_dnbelab_vs_quartzseq2_medium() {
    auto r = apply_override(
        "",
        "dnbelab",     // catalog: DNBelab (alias for dnbelab-c4)
        "quartzseq2",  // VDB misdetection at MEDIUM confidence
        18,
        singlet::fq::Confidence::MEDIUM
    );
    CHECK(r.override_fired);
    CHECK(r.tag == "dnbelab-c4");  // alias resolved to canonical tag

    // Verify the spec geometry (CB=10+10+10=30 split, r1_len=56)
    const singlet::fq::CandidateSpec* spec = singlet::fq::find_protocol_spec("dnbelab");
    CHECK(spec != nullptr);
    if (spec) {
        CHECK(spec->bc_len  == 10);  // single segment bc_len in CandidateSpec
        CHECK(spec->umi_len == 10);
    }
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_confidence_ordering();
    test_override_fires_seqwell_vs_visium_c1();
    test_override_fires_medium_confidence();
    test_override_fires_10xv2_vs_quartzseq2_c1();
    test_no_metadata_no_override();
    test_cli_protocol_wins();
    test_override_fires_high_confidence();
    test_none_confidence_fires();
    test_unknown_metadata_protocol_no_override();
    test_same_tag_no_override();
    test_override_fires_dnbelab_vs_quartzseq2_medium();

    std::cout << "protocol_confidence_override: "
              << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
