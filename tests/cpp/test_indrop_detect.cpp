// SPDX-License-Identifier: MIT
// test_indrop_detect.cpp — Unit tests for lib1fq/indrop_detect.h
#include <cassert>
#include <iostream>

#include "singlet/fq/indrop_detect.h"

#define SINGLET_TEST_HARNESS_TERSE
#include "test_harness.h"  // CHECK(cond) + g_pass / g_fail

// ── Version detection ─────────────────────────────────────────────────────────

static void test_v1_detection() {
    // Normal v1: cblen=16 (8+8), umilen=6, r1_length=22 — fits
    CHECK(detect_indrop_version(22, 16, 6) == InDropVersion::V1);

    // v1 with longer R1 (padding present)
    CHECK(detect_indrop_version(34, 16, 6) == InDropVersion::V1);
}

static void test_v2_detection() {
    // v2: very short R1 carrying only CB1 (8bp)
    CHECK(detect_indrop_version(8, 8, 0) == InDropVersion::V2);
    CHECK(detect_indrop_version(10, 8, 0) == InDropVersion::V2);
}

static void test_v3_detection() {
    // v3: long R1 with linker; cblen=38+
    CHECK(detect_indrop_version(60, 38, 6) == InDropVersion::V3);
    CHECK(detect_indrop_version(50, 24, 8) == InDropVersion::V3);
}

static void test_overflow_fallback() {
    // VAL1-INDROP-VARIANT bug: claimed CB+UMI end (41) > r1_length (35)
    // R1=35bp falls in the V1 range (13–35)
    CHECK(detect_indrop_version(35, 35, 6) == InDropVersion::V1);

    // Another overflow case: R1=8bp → V2
    CHECK(detect_indrop_version(8, 16, 6) == InDropVersion::V2);

    // Overflow but R1 long enough → V3
    CHECK(detect_indrop_version(40, 50, 10) == InDropVersion::V3);
}

// ── Layout retrieval ──────────────────────────────────────────────────────────

static void test_v1_layout() {
    InDropLayout l = get_indrop_layout(InDropVersion::V1);
    CHECK(l.split_reads == false);
    CHECK(l.cb1_len == 8);
    CHECK(l.cb2_len == 8);
    CHECK(l.umi_len == 6);
    CHECK(l.cb2_start == 8);   // CB2 immediately after CB1
    CHECK(l.umi_start == 16);  // UMI after CB1+CB2
}

static void test_v2_layout() {
    InDropLayout l = get_indrop_layout(InDropVersion::V2);
    CHECK(l.split_reads == true);
    CHECK(l.cb1_len == 8);
    CHECK(l.cb2_len == 8);
    CHECK(l.umi_len == 6);
}

static void test_v3_layout() {
    InDropLayout l = get_indrop_layout(InDropVersion::V3);
    CHECK(l.split_reads == false);
    CHECK(l.cb1_len == 8);
    CHECK(l.cb2_start == 30);  // after CB1(8) + linker(22)
    CHECK(l.umi_start == 38);
}

static void test_unknown_layout() {
    InDropLayout l = get_indrop_layout(InDropVersion::UNKNOWN);
    CHECK(l.cb1_len == 0);
    CHECK(l.umi_len == 0);
    CHECK(l.split_reads == false);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_v1_detection();
    test_v2_detection();
    test_v3_detection();
    test_overflow_fallback();
    test_v1_layout();
    test_v2_layout();
    test_v3_layout();
    test_unknown_layout();

    std::cout << "indrop_detect: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
