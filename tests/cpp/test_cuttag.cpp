// SPDX-License-Identifier: MIT
// test_cuttag.cpp — Unit tests for G-CUTTAG: cuttag.h
//
// Tests:
//   1. detect_mode — CUT&RUN signature (sub-nucl fraction > 40%)
//   2. detect_mode — CUT&TAG signature (sub-nucl fraction < 20%)
//   3. detect_mode — ambiguous fraction (20–40%) → conservative CUT_AND_TAG
//   4. detect_mode — empty input → CUT_AND_TAG
//   5. spike_in_scale_factor: correct reciprocal
//   6. spike_in_scale_factor: zero spike-in → 1.0
//   7. spike_in_scale_factor: zero total reads → 1.0
//   8. compute_cuttag_qc: sub-nucleosomal fraction
//   9. compute_cuttag_qc: mono-nucleosomal fraction
//  10. compute_cuttag_qc: large-fragment fraction
//  11. compute_cuttag_qc: signal-to-noise ratio
//  12. compute_cuttag_qc: spike-in chrom counting
//  13. compute_cuttag_qc: size histogram bins
//  14. compute_cuttag_qc: empty fragment list
//  15. write_cuttag_qc: JSON file contains required keys

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/cuttag.h"

using namespace singlet;

static int passed = 0;
static int failed = 0;

static void PASS(const char* name) {
    ++passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// Helper: build a Fragment
static Fragment make_frag(int32_t chrom, int32_t start, int32_t end) {
    Fragment f;
    f.chrom_idx = chrom;
    f.start = start;
    f.end = end;
    return f;
}

// Helper: check double approximately equal
static bool approx(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) < eps;
}

// ── Test 1: detect_mode CUT&RUN (> 40% sub-nucleosomal) ───────────────────
static void test_detect_cutrun() {
    // 50 fragments @ 80 bp (sub-nucl), 50 fragments @ 400 bp (large)
    // sub-nucl fraction = 0.50 > 0.40 → CUT&RUN
    std::vector<uint32_t> sizes(50, 80u);
    sizes.insert(sizes.end(), 50, 400u);
    auto mode = detect_mode(sizes);
    if (mode == CutTagMode::CUT_AND_RUN)
        PASS("detect_cutrun: 50% sub-nucl → CUT_AND_RUN");
    else
        FAIL("detect_cutrun", "expected CUT_AND_RUN for 50% sub-nucl fraction");
}

// ── Test 2: detect_mode CUT&TAG (< 20% sub-nucleosomal) ──────────────────
static void test_detect_cuttag() {
    // 10 fragments @ 80 bp, 90 at 200 bp (mono)
    // sub-nucl fraction = 0.10 < 0.20 → CUT&TAG
    std::vector<uint32_t> sizes(10, 80u);
    sizes.insert(sizes.end(), 90, 200u);
    auto mode = detect_mode(sizes);
    if (mode == CutTagMode::CUT_AND_TAG)
        PASS("detect_cuttag: 10% sub-nucl → CUT_AND_TAG");
    else
        FAIL("detect_cuttag", "expected CUT_AND_TAG for 10% sub-nucl fraction");
}

// ── Test 3: detect_mode ambiguous (20–40%) → conservative CUT_AND_TAG ─────
static void test_detect_ambiguous() {
    // 30% sub-nucl — falls in ambiguous zone (20–40%), conservative → CUT&TAG
    std::vector<uint32_t> sizes(30, 80u);
    sizes.insert(sizes.end(), 70, 200u);
    auto mode = detect_mode(sizes);
    if (mode == CutTagMode::CUT_AND_TAG)
        PASS("detect_ambiguous: 30% sub-nucl → conservative CUT_AND_TAG");
    else
        FAIL("detect_ambiguous", "expected conservative CUT_AND_TAG for 30% sub-nucl");
}

// ── Test 4: detect_mode empty ─────────────────────────────────────────────
static void test_detect_empty() {
    std::vector<uint32_t> sizes;
    auto mode = detect_mode(sizes);
    if (mode == CutTagMode::CUT_AND_TAG)
        PASS("detect_empty: empty → CUT_AND_TAG");
    else
        FAIL("detect_empty", "expected CUT_AND_TAG for empty size list");
}

// ── Test 5: spike-in scale factor — correct reciprocal ────────────────────
static void test_spike_in_scale_factor() {
    // 1000 spike-in / 10000 total → frac = 0.1 → scale = 10.0
    double sf = compute_spike_in_scale_factor(1000, 10000);
    if (approx(sf, 10.0, 1e-9))
        PASS("spike_in_scale_factor: 10% → 10.0");
    else
        FAIL("spike_in_scale_factor", "expected 10.0 for 10% spike-in");
}

// ── Test 6: spike-in scale factor — zero spike-in ─────────────────────────
static void test_spike_in_zero() {
    double sf = compute_spike_in_scale_factor(0, 10000);
    if (approx(sf, 1.0))
        PASS("spike_in_zero: zero spike-in → 1.0");
    else
        FAIL("spike_in_zero", "expected 1.0 when no spike-in reads");
}

// ── Test 7: spike-in scale factor — zero total reads ──────────────────────
static void test_spike_in_zero_total() {
    double sf = compute_spike_in_scale_factor(100, 0);
    if (approx(sf, 1.0))
        PASS("spike_in_zero_total: zero total reads → 1.0");
    else
        FAIL("spike_in_zero_total", "expected 1.0 when total_reads = 0");
}

// ── Test 8: QC sub-nucleosomal fraction ───────────────────────────────────
static void test_qc_sub_nucl_fraction() {
    // 60 frags @ 80 bp (< 120 → sub-nucl), 40 frags @ 400 bp
    std::vector<Fragment> frags;
    for (int i = 0; i < 60; ++i) frags.push_back(make_frag(0, 0, 80));
    for (int i = 0; i < 40; ++i) frags.push_back(make_frag(0, 0, 400));

    auto qc = compute_cuttag_qc(frags, 100);
    if (approx(qc.sub_nucleosomal_fraction, 0.6, 1e-9))
        PASS("qc_sub_nucl_fraction: 60/100 = 0.6");
    else
        FAIL("qc_sub_nucl_fraction", "wrong sub-nucleosomal fraction");
}

// ── Test 9: QC mono-nucleosomal fraction ──────────────────────────────────
static void test_qc_mono_fraction() {
    // 20 @ 80 bp (sub-nucl), 50 @ 200 bp (mono: 150–300), 30 @ 400 bp (large)
    std::vector<Fragment> frags;
    for (int i = 0; i < 20; ++i) frags.push_back(make_frag(0, 0, 80));
    for (int i = 0; i < 50; ++i) frags.push_back(make_frag(0, 0, 200));
    for (int i = 0; i < 30; ++i) frags.push_back(make_frag(0, 0, 400));

    auto qc = compute_cuttag_qc(frags, 100);
    if (approx(qc.mono_nucleosomal_fraction, 0.5, 1e-9))
        PASS("qc_mono_fraction: 50/100 = 0.5");
    else
        FAIL("qc_mono_fraction", "wrong mono-nucleosomal fraction");
}

// ── Test 10: QC large-fragment fraction ───────────────────────────────────
static void test_qc_large_fraction() {
    // 70 @ 80 bp, 30 @ 500 bp (large: > 300)
    std::vector<Fragment> frags;
    for (int i = 0; i < 70; ++i) frags.push_back(make_frag(0, 0, 80));
    for (int i = 0; i < 30; ++i) frags.push_back(make_frag(0, 0, 500));

    auto qc = compute_cuttag_qc(frags, 100);
    if (approx(qc.large_fragment_fraction, 0.3, 1e-9))
        PASS("qc_large_fraction: 30/100 = 0.3");
    else
        FAIL("qc_large_fraction", "wrong large-fragment fraction");
}

// ── Test 11: QC signal-to-noise ratio ─────────────────────────────────────
static void test_qc_signal_to_noise() {
    // 60 signal (sub + mono), 20 large → SNR = 3.0
    std::vector<Fragment> frags;
    for (int i = 0; i < 60; ++i) frags.push_back(make_frag(0, 0, 80));
    for (int i = 0; i < 20; ++i) frags.push_back(make_frag(0, 0, 400));

    auto qc = compute_cuttag_qc(frags, 80);
    // signal = 60 (sub-nucl), large = 20 → SNR = 3.0
    if (approx(qc.signal_to_noise, 3.0, 1e-9))
        PASS("qc_signal_to_noise: SNR = 60/20 = 3.0");
    else
        FAIL("qc_signal_to_noise", "wrong signal-to-noise ratio");
}

// ── Test 12: QC spike-in chrom counting ────────────────────────────────────
static void test_qc_spike_in() {
    // chrom 0 = experiment, chrom 1 = spike-in (E. coli)
    // 80 frags on chrom 0, 20 on chrom 1
    std::vector<Fragment> frags;
    for (int i = 0; i < 80; ++i) frags.push_back(make_frag(0, 0, 200));
    for (int i = 0; i < 20; ++i) frags.push_back(make_frag(1, 0, 200));

    auto qc = compute_cuttag_qc(frags, 100, {1});
    if (qc.spike_in_reads == 20 &&
        approx(qc.spike_in_fraction, 0.20, 1e-9) &&
        approx(qc.scale_factor, 5.0, 1e-9))
        PASS("qc_spike_in: reads=20 frac=0.2 scale=5.0");
    else
        FAIL("qc_spike_in", "wrong spike-in metrics");
}

// ── Test 13: QC size histogram bins ───────────────────────────────────────
static void test_qc_size_histogram() {
    // 100 fragments of size 85 → bin = 85/10 = 8
    std::vector<Fragment> frags;
    for (int i = 0; i < 100; ++i) frags.push_back(make_frag(0, 0, 85));

    auto qc = compute_cuttag_qc(frags, 100);
    if (qc.size_histogram.size() == 100 &&
        qc.size_histogram[8] == 100 &&
        qc.size_histogram[0] == 0)
        PASS("qc_size_histogram: bin 8 = 100");
    else
        FAIL("qc_size_histogram", "wrong histogram count");
}

// ── Test 14: QC empty fragment list ───────────────────────────────────────
static void test_qc_empty() {
    std::vector<Fragment> frags;
    auto qc = compute_cuttag_qc(frags, 0);
    if (qc.total_fragments == 0 &&
        approx(qc.sub_nucleosomal_fraction, 0.0) &&
        approx(qc.mono_nucleosomal_fraction, 0.0) &&
        approx(qc.large_fragment_fraction, 0.0) &&
        approx(qc.scale_factor, 1.0))
        PASS("qc_empty: all fractions 0, scale=1");
    else
        FAIL("qc_empty", "failed on empty fragment list");
}

// ── Test 15: JSON output — required fields present ────────────────────────
static void test_json_output() {
    std::vector<Fragment> frags;
    for (int i = 0; i < 50; ++i) frags.push_back(make_frag(0, 0, 80));
    for (int i = 0; i < 50; ++i) frags.push_back(make_frag(0, 0, 200));

    auto qc = compute_cuttag_qc(frags, 100);
    const std::string path = "/tmp/test_cuttag_qc.json";
    bool ok = write_cuttag_qc(path, qc, CutTagMode::CUT_AND_RUN);
    if (!ok) {
        FAIL("json_output", "write_cuttag_qc returned false");
        return;
    }

    std::ifstream ifs(path);
    std::ostringstream buf;
    buf << ifs.rdbuf();
    const std::string json = buf.str();

    // Check required keys
    const std::vector<std::string> keys = {
        "\"mode\"", "\"total_fragments\"", "\"unique_fragments\"",
        "\"sub_nucleosomal_fraction\"", "\"mono_nucleosomal_fraction\"",
        "\"large_fragment_fraction\"", "\"signal_to_noise\"",
        "\"frip\"", "\"spike_in_reads\"", "\"spike_in_fraction\"",
        "\"scale_factor\"", "\"size_histogram\""};
    bool all_found = true;
    for (const auto& key : keys) {
        if (json.find(key) == std::string::npos) {
            all_found = false;
            std::cerr << "  missing key: " << key << "\n";
        }
    }
    // Check mode value
    if (json.find("CUT_AND_RUN") == std::string::npos) {
        all_found = false;
        std::cerr << "  missing mode value CUT_AND_RUN\n";
    }

    if (all_found)
        PASS("json_output: all required fields present");
    else
        FAIL("json_output", "missing required fields in JSON");
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_detect_cutrun();
    test_detect_cuttag();
    test_detect_ambiguous();
    test_detect_empty();
    test_spike_in_scale_factor();
    test_spike_in_zero();
    test_spike_in_zero_total();
    test_qc_sub_nucl_fraction();
    test_qc_mono_fraction();
    test_qc_large_fraction();
    test_qc_signal_to_noise();
    test_qc_spike_in();
    test_qc_size_histogram();
    test_qc_empty();
    test_json_output();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
