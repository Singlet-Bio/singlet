// test_cellplex.cpp — unit tests for G-CELLPLEX demultiplexing
// Tests: 3 CMOs, 5 cells using Otsu-threshold CLR demux.

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

#include "singlet-pileup/cellplex_demux.h"
#include "singlet-pileup/sparse_accumulator.h"

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

// Helper: build SparseAccumulator<uint32_t> from dense (n_cmos × n_cells) array
static SparseAccumulator<uint32_t> make_cmo_mat(const std::vector<uint32_t>& vals,
                                                size_t n_cmos, size_t n_cells) {
    SparseAccumulator<uint32_t> acc;
    acc.set_n_features(static_cast<uint32_t>(n_cmos));
    std::vector<std::string> dummy(n_cells);
    acc.set_barcodes(dummy);
    for (size_t k = 0; k < n_cmos; ++k)
        for (size_t c = 0; c < n_cells; ++c) {
            uint32_t v = vals[k * n_cells + c];
            if (v > 0) acc.increment(static_cast<uint32_t>(k),
                                     static_cast<uint32_t>(c), v);
        }
    return acc;
}

// ── Test 1: 5-cell, 3-CMO scenario from spec ──────────────────────────────────
//  Cell 0: CMO1=100, CMO2=1, CMO3=1  → SINGLET (sample 0, CMO1)
//  Cell 1: CMO1=1,   CMO2=80, CMO3=1 → SINGLET (sample 1, CMO2)
//  Cell 2: CMO1=50,  CMO2=50, CMO3=1 → MULTIPLET
//  Cell 3: CMO1=1,   CMO2=1,  CMO3=1 → UNASSIGNED
//  Cell 4: CMO1=1,   CMO2=1,  CMO3=90→ SINGLET (sample 2, CMO3)
static void test_basic_assignment() {
    //           c0   c1   c2  c3   c4
    // CMO1:    100    1   50   1    1
    // CMO2:      1   80   50   1    1
    // CMO3:      1    1    1   1   90
    std::vector<uint32_t> vals = {
        100, 1, 50, 1, 1,  // CMO1
        1, 80, 50, 1, 1,   // CMO2
        1, 1, 1, 1, 90     // CMO3
    };
    auto mat = make_cmo_mat(vals, 3, 5);
    CellPlexDemux demux;
    auto res = demux.demux(mat, {"CMO1", "CMO2", "CMO3"}, 5);

    bool ok = true;
    if (res[0].call != "SINGLET" || res[0].sample != "CMO1" || res[0].sample_idx != 0) {
        ok = false;
        std::cerr << "  cell0: got " << res[0].call << "/" << res[0].sample << "\n";
    }
    if (res[1].call != "SINGLET" || res[1].sample != "CMO2" || res[1].sample_idx != 1) {
        ok = false;
        std::cerr << "  cell1: got " << res[1].call << "/" << res[1].sample << "\n";
    }
    if (res[2].call != "MULTIPLET") {
        ok = false;
        std::cerr << "  cell2: got " << res[2].call << "\n";
    }
    if (res[3].call != "UNASSIGNED") {
        ok = false;
        std::cerr << "  cell3: got " << res[3].call << "\n";
    }
    if (res[4].call != "SINGLET" || res[4].sample != "CMO3" || res[4].sample_idx != 2) {
        ok = false;
        std::cerr << "  cell4: got " << res[4].call << "/" << res[4].sample << "\n";
    }
    if (ok)
        PASS("basic_assignment");
    else
        FAIL("basic_assignment", "wrong call/sample");
}

// ── Test 2: CLR correctness check ─────────────────────────────────────────────
// 2 CMOs, 2 cells. When both cells have identical counts, CLR = 0.
// Equal signal → UNASSIGNED (both CLR = 0, below Otsu threshold for both CMOs).
static void test_clr_equal_counts() {
    // All cells same → CLR=0 everywhere → everything UNASSIGNED
    std::vector<uint32_t> vals = {
        10, 10,  // CMO1 cells 0,1
        10, 10   // CMO2 cells 0,1
    };
    auto mat = make_cmo_mat(vals, 2, 2);
    CellPlexDemux demux;
    auto res = demux.demux(mat, {"CMO1", "CMO2"}, 2);
    bool ok = (res[0].call == "UNASSIGNED" && res[1].call == "UNASSIGNED");
    if (ok)
        PASS("clr_equal_counts");
    else
        FAIL("clr_equal_counts", "expected UNASSIGNED");
}

// ── Test 3: all cells same CMO → all singlets with same assignment ────────────
static void test_all_same_cmo() {
    // 4 cells all positive for CMO1
    //         c0    c1    c2    c3
    // CMO1: 1000  1000  1000  1000
    // CMO2:    1     1     1     1
    std::vector<uint32_t> vals = {
        1000, 1000, 1000, 1000,
        1, 1, 1, 1};
    auto mat = make_cmo_mat(vals, 2, 4);
    CellPlexDemux demux;
    auto res = demux.demux(mat, {"CMO1", "CMO2"}, 4);
    bool ok = true;
    for (auto& r : res)
        if (r.call != "SINGLET" || r.sample != "CMO1") ok = false;
    if (ok)
        PASS("all_same_cmo");
    else
        FAIL("all_same_cmo", "expected all SINGLET/CMO1");
}

// ── Test 4: write_tsv output ───────────────────────────────────────────────────
static void test_write_tsv() {
    std::vector<uint32_t> vals = {
        100, 1, 50, 1, 1,
        1, 80, 50, 1, 1,
        1, 1, 1, 1, 90};
    auto mat = make_cmo_mat(vals, 3, 5);
    CellPlexDemux demux;
    auto res = demux.demux(mat, {"CMO1", "CMO2", "CMO3"}, 5);

    const char* tmpfile = "/tmp/test_cellplex_out.tsv";
    std::vector<std::string> barcodes = {"BC0", "BC1", "BC2", "BC3", "BC4"};
    demux.write_tsv(tmpfile, barcodes, res);

    std::ifstream f(tmpfile);
    std::string header;
    std::getline(f, header);
    bool ok = (header == "barcode\tcall\tsample\ttop_cmo\ttop_clr");
    if (!ok) std::cerr << "  header: [" << header << "]\n";

    // Count lines
    int lines = 0;
    std::string line;
    while (std::getline(f, line)) ++lines;
    ok = ok && (lines == 5);
    if (!ok) std::cerr << "  lines=" << lines << " expected 5\n";

    std::remove(tmpfile);
    if (ok)
        PASS("write_tsv");
    else
        FAIL("write_tsv", "header or line count wrong");
}

// ── Test 5: empty input returns empty result ──────────────────────────────────
static void test_empty_input() {
    SparseAccumulator<uint32_t> acc;
    acc.set_n_features(3);
    std::vector<std::string> dummy;
    acc.set_barcodes(dummy);
    CellPlexDemux demux;
    auto res = demux.demux(acc, {"CMO1", "CMO2", "CMO3"}, 0);
    bool ok = res.empty();
    if (ok)
        PASS("empty_input");
    else
        FAIL("empty_input", "expected empty result");
}

int main() {
    test_basic_assignment();
    test_clr_equal_counts();
    test_all_same_cmo();
    test_write_tsv();
    test_empty_input();

    std::cerr << "\n"
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}
