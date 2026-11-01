// SPDX-License-Identifier: MIT
// test_hto_demux.cpp — unit tests for T3 HTO demultiplexing
// Tests: perfect HTO separation → all singlets, all-low HTO → negatives,
//        two HTOs above threshold → doublet, write_tsv output.

#include <cassert>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <string>

#include "singlet/pileup/hto_demux.h"
#include "singlet/pileup/sparse_accumulator.h"

using namespace singlet;

static int g_passed = 0, g_failed = 0;

static void PASS(const char* name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// Helper: build a SparseAccumulator<uint32_t> from a dense (n_tags × n_cells) array
static SparseAccumulator<uint32_t> make_mat(const std::vector<uint32_t>& vals,
                                            size_t n_tags, size_t n_cells) {
    SparseAccumulator<uint32_t> acc;
    acc.set_n_features(static_cast<uint32_t>(n_tags));
    std::vector<std::string> dummy(n_cells);
    acc.set_barcodes(dummy);
    for (size_t t = 0; t < n_tags; ++t)
        for (size_t c = 0; c < n_cells; ++c) {
            uint32_t v = vals[t * n_cells + c];
            if (v > 0) acc.increment(static_cast<uint32_t>(t),
                                     static_cast<uint32_t>(c), v);
        }
    return acc;
}

// ── Test 1: perfect HTO separation → all singlets ─────────────────────────
// 4 cells, 2 HTOs. Cell 0 & 1 have high HTO-A, Cell 2 & 3 have high HTO-B.
static void test_perfect_separation() {
    // Row 0 = HTO-A, Row 1 = HTO-B; cols = cells
    //         c0    c1    c2    c3
    // HTO-A: 1000  1000    1     1
    // HTO-B:    1     1 1000  1000
    std::vector<uint32_t> vals = {
        1000, 1000, 1, 1,  // HTO-A
        1, 1, 1000, 1000   // HTO-B
    };
    auto mat = make_mat(vals, 2, 4);
    HtoDemux demux;
    auto res = demux.demux(mat, {"HTO-A", "HTO-B"}, 4);

    bool ok = (res[0].classification == "SINGLET" && res[0].assignment == "HTO-A" &&
               res[1].classification == "SINGLET" && res[1].assignment == "HTO-A" &&
               res[2].classification == "SINGLET" && res[2].assignment == "HTO-B" &&
               res[3].classification == "SINGLET" && res[3].assignment == "HTO-B");
    if (ok)
        PASS("perfect_separation all singlets");
    else {
        FAIL("perfect_separation all singlets",
             (res[0].classification + "/" + res[0].assignment + " " +
              res[2].classification + "/" + res[2].assignment)
                 .c_str());
    }
}

// ── Test 2: all-zero HTO counts → all NEGATIVE ───────────────────────────
static void test_all_zeros_are_negative() {
    std::vector<uint32_t> vals(2 * 4, 0);  // 2 tags × 4 cells, all zero
    auto mat = make_mat(vals, 2, 4);
    HtoDemux demux;
    auto res = demux.demux(mat, {"HTO-A", "HTO-B"}, 4);

    bool ok = true;
    for (const auto& r : res)
        if (r.classification != "NEGATIVE") ok = false;
    if (ok)
        PASS("all_zeros_are_negative");
    else
        FAIL("all_zeros_are_negative", "expected all NEGATIVE");
}

// ── Test 3: two HTOs both high → DOUBLET ─────────────────────────────────
// 5 cells, 2 HTOs.
// Cells 0-1: HTO-A=1000, HTO-B=5   → SINGLET HTO-A
// Cells 2-3: HTO-A=5,    HTO-B=1000 → SINGLET HTO-B
// Cell  4:   HTO-A=1000, HTO-B=1000 → DOUBLET
// CLR for cell 4 = (0, 0); threshold from neg mode ≈ -2.56 → both above → DOUBLET ✓
static void test_doublet_detection() {
    //         c0    c1    c2    c3    c4
    // HTO-A: 1000  1000     5     5  1000
    // HTO-B:    5     5  1000  1000  1000
    std::vector<uint32_t> vals = {
        1000, 1000, 5, 5, 1000,  // HTO-A
        5, 5, 1000, 1000, 1000   // HTO-B
    };
    auto mat = make_mat(vals, 2, 5);
    HtoDemux demux;
    auto res = demux.demux(mat, {"HTO-A", "HTO-B"}, 5);

    bool c0_ok = (res[0].classification == "SINGLET" && res[0].assignment == "HTO-A");
    bool c2_ok = (res[2].classification == "SINGLET" && res[2].assignment == "HTO-B");
    bool c4_ok = (res[4].classification == "DOUBLET");
    if (c0_ok && c2_ok && c4_ok)
        PASS("doublet_detection");
    else
        FAIL("doublet_detection",
             (std::string("c0=") + res[0].classification + "/" + res[0].assignment +
              " c2=" + res[2].classification + "/" + res[2].assignment +
              " c4=" + res[4].classification)
                 .c_str());
}

// ── Test 4: write_tsv produces correct output ────────────────────────────
static void test_write_tsv() {
    std::vector<HtoDemuxResult> res = {
        {"SINGLET", "HTO-A", 2.5f},
        {"NEGATIVE", "", -1.0f},
        {"DOUBLET", "", 1.8f}};
    std::vector<std::string> barcodes = {"BC1", "BC2", "BC3"};
    const char* path = "/tmp/test_hto_demux_out.tsv";
    HtoDemux demux;
    demux.write_tsv(path, barcodes, res);

    std::ifstream ifs(path);
    std::string line;
    std::getline(ifs, line);  // header
    bool ok = (line.find("barcode") != std::string::npos &&
               line.find("classification") != std::string::npos);
    std::getline(ifs, line);  // BC1
    ok &= (line.find("BC1") != std::string::npos &&
           line.find("SINGLET") != std::string::npos &&
           line.find("HTO-A") != std::string::npos);
    std::getline(ifs, line);  // BC2
    ok &= (line.find("NEGATIVE") != std::string::npos);
    std::getline(ifs, line);  // BC3
    ok &= (line.find("DOUBLET") != std::string::npos);
    std::remove(path);

    if (ok)
        PASS("write_tsv correct output");
    else
        FAIL("write_tsv correct output", "TSV format wrong");
}

// ── Test 5: 3-HTO panel, one singlet per HTO ─────────────────────────────
static void test_three_hto_panel() {
    // 3 cells × 3 HTOs: each cell positive for exactly one HTO
    //         c0    c1    c2
    // HTO-A: 1000    1     1
    // HTO-B:    1 1000     1
    // HTO-C:    1    1  1000
    std::vector<uint32_t> vals = {
        1000, 1, 1,
        1, 1000, 1,
        1, 1, 1000};
    auto mat = make_mat(vals, 3, 3);
    HtoDemux demux;
    auto res = demux.demux(mat, {"HTO-A", "HTO-B", "HTO-C"}, 3);

    bool ok = (res[0].classification == "SINGLET" && res[0].assignment == "HTO-A" &&
               res[1].classification == "SINGLET" && res[1].assignment == "HTO-B" &&
               res[2].classification == "SINGLET" && res[2].assignment == "HTO-C");
    if (ok)
        PASS("three_hto_panel");
    else
        FAIL("three_hto_panel",
             (res[0].assignment + " " + res[1].assignment + " " + res[2].assignment).c_str());
}

// -- Test 6: a realistic pool (400 cells, 4 tags): accuracy, and the calls do not depend on the cell order --
// Cells 0-339 are singlets of tag (c % 4) with a noisy bound signal over a low ambient; cells 340-399 are doublets of two tags.
static std::vector<uint32_t> synth_pool(size_t n_tags, size_t n_cells) {
    uint64_t s = 12345;
    auto rnd = [&]() {  // LCG in [0, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(s >> 11) / 9007199254740992.0;
    };
    std::vector<uint32_t> v(n_tags * n_cells, 0);
    for (size_t c = 0; c < n_cells; ++c) {
        const size_t a = c % n_tags, b = (c + 1 + (c / n_tags) % (n_tags - 1)) % n_tags;
        const bool dbl = c >= 340;
        for (size_t t = 0; t < n_tags; ++t) {
            double x = 2.0 + 6.0 * rnd();  // ambient
            if (t == a || (dbl && t == b)) x += 150.0 + 250.0 * rnd();
            v[t * n_cells + c] = static_cast<uint32_t>(x);
        }
    }
    return v;
}

static void test_pool_accuracy_and_order_invariance() {
    const size_t T = 4, N = 400;
    const std::vector<std::string> names = {"HTO-1", "HTO-2", "HTO-3", "HTO-4"};
    auto vals = synth_pool(T, N);
    HtoDemux demux;
    auto res = demux.demux(make_mat(vals, T, N), names, N);
    size_t ok = 0, dbl_ok = 0;
    for (size_t c = 0; c < 340; ++c) ok += (res[c].classification == "SINGLET" && res[c].assignment == names[c % T]);
    for (size_t c = 340; c < N; ++c) dbl_ok += (res[c].classification == "DOUBLET");
    std::vector<uint32_t> rev(vals.size());
    for (size_t t = 0; t < T; ++t)
        for (size_t c = 0; c < N; ++c) rev[t * N + c] = vals[t * N + (N - 1 - c)];
    auto res2 = demux.demux(make_mat(rev, T, N), names, N);
    bool same = true;
    for (size_t c = 0; c < N; ++c)
        same &= (res[c].classification == res2[N - 1 - c].classification && res[c].assignment == res2[N - 1 - c].assignment);
    if (ok >= 320 && dbl_ok >= 50 && same)
        PASS("pool accuracy and order invariance");
    else
        FAIL("pool accuracy and order invariance",
             (std::to_string(ok) + "/340 singlets, " + std::to_string(dbl_ok) + "/60 doublets, same=" + std::to_string(same)).c_str());
}

int main() {
    test_perfect_separation();
    test_all_zeros_are_negative();
    test_doublet_detection();
    test_write_tsv();
    test_three_hto_panel();
    test_pool_accuracy_and_order_invariance();

    std::cerr << "\n"
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed ? 1 : 0;
}
