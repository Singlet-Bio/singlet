// SPDX-License-Identifier: MIT
// test/test_rrna_detect.cpp — Unit tests for G-RRNA rRNA contamination detection
// Tests: detect(), batch_detect(), subunit accounting, write_rrna_report().

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/rrna_detect.h"

using namespace singlet;

#define SINGLET_TEST_HARNESS_QUIET
#include "test_harness.h"  // CHECK(cond, msg) + g_pass / g_fail

// Grab a k-mer that is definitely in the index from the human 28S array.
// We embed two consecutive 21-mers from the human_28S list so detect() fires.
static std::string make_rrna_read() {
    // Concatenate first two human 28S kmers to get a ≥42-bp string with ≥2 hits
    return std::string("GTTGGTGGAGCGATTTGTCTG") + std::string("CGGGTCGGGAGGCGAACGAGT");
}

static std::string make_clean_read(int len = 50) {
    // Poly-A — no rRNA k-mer
    return std::string(len, 'A');
}

// ── detect() ─────────────────────────────────────────────────────────────────

static void test_detect_positive() {
    RrnaDetector det;
    std::string read = make_rrna_read();
    CHECK(det.detect(read), "detect: rRNA read positive");
}

static void test_detect_negative() {
    RrnaDetector det;
    CHECK(!det.detect(make_clean_read()), "detect: clean read negative");
}

static void test_detect_short_read() {
    RrnaDetector det;
    CHECK(!det.detect("ACGT"), "detect: <21 bp returns false");
}

static void test_detect_single_kmer_not_enough() {
    RrnaDetector det;
    // Only one k-mer present — threshold is ≥2
    std::string one = "GTTGGTGGAGCGATTTGTCTG" + std::string(30, 'N');
    // With NNN padding the second kmer won't match
    CHECK(!det.detect(one), "detect: single kmer not enough (threshold=2)");
}

static void test_detect_empty_string() {
    RrnaDetector det;
    CHECK(!det.detect(""), "detect: empty string false");
}

// ── batch_detect() ────────────────────────────────────────────────────────────

static void test_batch_counts() {
    RrnaDetector det;
    std::vector<std::string_view> reads;
    std::string pos = make_rrna_read();
    std::string neg = make_clean_read();
    reads.push_back(pos);
    reads.push_back(neg);
    reads.push_back(pos);
    auto stats = det.batch_detect(reads);
    CHECK(stats.total_reads == 3, "batch: total_reads = 3");
    CHECK(stats.rrna_reads == 2, "batch: rrna_reads = 2");
    CHECK(stats.rrna_fraction > 0.65 && stats.rrna_fraction < 0.68,
          "batch: fraction ~0.667");
}

static void test_batch_all_clean() {
    RrnaDetector det;
    std::vector<std::string_view> reads;
    std::string neg = make_clean_read();
    for (int i = 0; i < 5; ++i) reads.push_back(neg);
    auto stats = det.batch_detect(reads);
    CHECK(stats.rrna_reads == 0, "batch_clean: no rRNA reads");
    CHECK(stats.rrna_fraction == 0.0, "batch_clean: fraction 0");
}

static void test_batch_empty() {
    RrnaDetector det;
    std::vector<std::string_view> reads;
    auto stats = det.batch_detect(reads);
    CHECK(stats.total_reads == 0, "batch_empty: total 0");
    CHECK(stats.rrna_fraction == 0.0, "batch_empty: fraction 0");
}

// ── subunit_counts ────────────────────────────────────────────────────────────

static void test_subunit_reported() {
    RrnaDetector det;
    std::string pos = make_rrna_read();  // hits human_28S k-mers
    std::vector<std::string_view> reads = {pos};
    auto stats = det.batch_detect(reads);
    CHECK(!stats.subunit_counts.empty(), "subunit: at least one subunit recorded");
    CHECK(stats.subunit_counts.count("human_28S") > 0, "subunit: human_28S present");
}

// ── write_rrna_report() ───────────────────────────────────────────────────────

static void test_write_report() {
    RrnaStats stats;
    stats.total_reads = 1000;
    stats.rrna_reads = 150;
    stats.rrna_fraction = 0.15;
    stats.subunit_counts["human_28S"] = 90;
    stats.subunit_counts["human_18S"] = 60;

    const char* tmpfile = "/tmp/test_rrna_report.json";
    write_rrna_report(stats, tmpfile);

    std::ifstream f(tmpfile);
    CHECK(f.is_open(), "write_report: file created");
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    CHECK(content.find("\"total_reads\": 1000") != std::string::npos,
          "write_report: total_reads in JSON");
    CHECK(content.find("\"rrna_reads\": 150") != std::string::npos,
          "write_report: rrna_reads in JSON");
    CHECK(content.find("human_28S") != std::string::npos,
          "write_report: subunit key in JSON");
    std::remove(tmpfile);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_detect_positive();
    test_detect_negative();
    test_detect_short_read();
    test_detect_single_kmer_not_enough();
    test_detect_empty_string();

    test_batch_counts();
    test_batch_all_clean();
    test_batch_empty();

    test_subunit_reported();

    test_write_report();

    std::cout << "rrna_detect: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail > 0) ? 1 : 0;
}
