// SPDX-License-Identifier: MIT
// test_seqwell_detect.cpp — Unit tests for lib1fq/seqwell_detect.h
#include <iostream>

#include "singlet/fq/seqwell_detect.h"

#define SINGLET_TEST_HARNESS_TERSE
#include "test_harness.h"  // CHECK(cond) + g_pass / g_fail

// ── is_seqwell detection ──────────────────────────────────────────────────────

static void test_is_seqwell_tags() {
    CHECK(is_seqwell("seq-well"));
    CHECK(is_seqwell("Seq-Well"));
    CHECK(is_seqwell("SEQ_WELL"));
    CHECK(is_seqwell("seqwell"));
    CHECK(is_seqwell("seqwell-s3"));
    CHECK(is_seqwell("SEQWELL-S3"));
    CHECK(!is_seqwell("dropseq"));
    CHECK(!is_seqwell("10xv3"));
}

// ── CBlen validation ──────────────────────────────────────────────────────────

static void test_cblen_validation() {
    CHECK(validate_seqwell_cblen(12));    // canonical
    CHECK(validate_seqwell_cblen(10));    // minimum acceptable
    CHECK(validate_seqwell_cblen(16));    // maximum acceptable
    CHECK(!validate_seqwell_cblen(150));  // VAL1-SEQWELL-DETECT bug value
    CHECK(!validate_seqwell_cblen(0));    // zero is invalid
    CHECK(!validate_seqwell_cblen(-1));   // negative is invalid
}

// ── Layout constants ──────────────────────────────────────────────────────────

static void test_layout_constants() {
    CHECK(SeqWellLayout::CB_LEN == 12);
    CHECK(SeqWellLayout::UMI_LEN == 8);
    CHECK(SeqWellLayout::TOTAL == 20);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_is_seqwell_tags();
    test_cblen_validation();
    test_layout_constants();

    std::cout << "seqwell_detect: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
