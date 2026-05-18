// SPDX-License-Identifier: MIT
// test_atac_qc.cpp — Unit tests for A3: ATAC QC metrics (atac_qc.h)
//
// Tests:
//   1. Fragment on chrM → counted as mito
//   2. Fragment midpoint near TSS → counted as TSS hit
//   3. Fragment far from TSS → not counted as TSS hit
//   4. Fragment size / median correctly computed
//   5. QC TSV output format: header + correct columns
//   6. FRIP approximation: signal vs background bin

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/atac_qc.h"

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

// Helper: build a minimal ATACFragment
static ATACFragment make_frag(int32_t chrom, int32_t start, int32_t end, uint32_t bc) {
    ATACFragment f;
    f.chrom_idx = chrom;
    f.start = start;
    f.end = end;
    f.barcode_idx = bc;
    return f;
}

// ── Test 1: Mito fragment detection ────────────────────────────────────────
static void test_mito_fragment() {
    // chrom 0 = chr1, chrom 1 = chrM
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}, {"chrM", 1}};

    ATACQCComputer qc;
    qc.set_chrom_map(cmap);
    qc.set_mito_chroms({"chrM", "MT"});

    std::vector<ATACFragment> frags = {
        make_frag(0, 1000, 1200, 0),  // chr1, cell 0 — not mito
        make_frag(1, 500, 700, 0),    // chrM, cell 0 — mito
        make_frag(1, 100, 300, 1),    // chrM, cell 1 — mito
    };

    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 2);

    const auto& cqc = qc.cell_qc();
    if (cqc[0].total_fragments == 2 && cqc[0].mito_fragments == 1)
        PASS("mito_fragment cell0: total=2 mito=1");
    else
        FAIL("mito_fragment cell0", "wrong counts");

    if (std::abs(cqc[0].mito_fraction - 0.5f) < 1e-5f)
        PASS("mito_fraction cell0=0.5");
    else
        FAIL("mito_fraction cell0=0.5", "wrong fraction");

    if (cqc[1].total_fragments == 1 && cqc[1].mito_fragments == 1)
        PASS("mito_fragment cell1: total=1 mito=1");
    else
        FAIL("mito_fragment cell1", "wrong counts");
}

// ── Test 2: TSS enrichment — near vs far ────────────────────────────────────
static void test_tss_enrichment() {
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}};

    // TSS at position 5000
    std::unordered_map<int32_t, std::vector<int32_t>> tss = {{0, {5000}}};

    ATACQCComputer qc;
    qc.set_chrom_map(cmap);
    qc.set_tss_positions(tss);
    qc.set_tss_window(1000);

    std::vector<ATACFragment> frags = {
        // Midpoint 5100 → within ±1000 of 5000 → TSS hit
        make_frag(0, 5000, 5200, 0),
        // Midpoint 500 → far from TSS 5000 → NOT a hit
        make_frag(0, 400, 600, 0),
    };

    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 1);

    const auto& cqc = qc.cell_qc();
    if (cqc[0].tss_fragments == 1)
        PASS("tss_near counted");
    else
        FAIL("tss_near counted", "expected tss_fragments=1");

    if (std::abs(cqc[0].tss_enrichment - 0.5f) < 1e-5f)
        PASS("tss_enrichment=0.5");
    else
        FAIL("tss_enrichment=0.5", "wrong value");
}

// ── Test 3: Fragment far from TSS not counted ───────────────────────────────
static void test_tss_far_not_counted() {
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}};
    std::unordered_map<int32_t, std::vector<int32_t>> tss = {{0, {5000}}};

    ATACQCComputer qc;
    qc.set_chrom_map(cmap);
    qc.set_tss_positions(tss);
    qc.set_tss_window(500);

    std::vector<ATACFragment> frags = {
        // Midpoint 100 → |100 - 5000| = 4900 > 500 → not near
        make_frag(0, 0, 200, 0),
    };

    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 1);

    const auto& cqc = qc.cell_qc();
    if (cqc[0].tss_fragments == 0 && std::abs(cqc[0].tss_enrichment) < 1e-5f)
        PASS("tss_far_not_counted");
    else
        FAIL("tss_far_not_counted", "expected tss_fragments=0");
}

// ── Test 4: Median fragment size ────────────────────────────────────────────
static void test_median_fragment_size() {
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}};
    ATACQCComputer qc;
    qc.set_chrom_map(cmap);

    // Sizes: 100, 200, 300 → median = 200
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 100, 0),
        make_frag(0, 0, 200, 0),
        make_frag(0, 0, 300, 0),
    };

    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 1);

    const auto& cqc = qc.cell_qc();
    if (std::abs(cqc[0].median_fragment_size - 200.0f) < 1e-3f)
        PASS("median_fragment_size=200");
    else
        FAIL("median_fragment_size=200",
             ("got " + std::to_string(cqc[0].median_fragment_size)).c_str());

    // Even count: sizes 100, 300 → median = 200
    ATACQCComputer qc2;
    qc2.set_chrom_map(cmap);
    std::vector<ATACFragment> frags2 = {
        make_frag(0, 0, 100, 0),
        make_frag(0, 0, 300, 0),
    };
    qc2.compute(frags2, ext, 1);
    const auto& cqc2 = qc2.cell_qc();
    if (std::abs(cqc2[0].median_fragment_size - 200.0f) < 1e-3f)
        PASS("median_fragment_size_even=200");
    else
        FAIL("median_fragment_size_even=200",
             ("got " + std::to_string(cqc2[0].median_fragment_size)).c_str());
}

// ── Test 5: QC TSV output format ────────────────────────────────────────────
static void test_qc_tsv_output() {
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}};
    ATACQCComputer qc;
    qc.set_chrom_map(cmap);

    std::vector<ATACFragment> frags = {
        make_frag(0, 100, 300, 0),
        make_frag(0, 200, 400, 1),
    };
    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 2);

    const std::string path = "/tmp/test_atac_qc_out.tsv";
    std::vector<std::string> barcodes = {"CELL_A", "CELL_B"};
    qc.write_qc_tsv(path, barcodes);

    std::ifstream f(path);
    if (!f.is_open()) {
        FAIL("qc_tsv_open", "cannot open output file");
        return;
    }

    std::string header;
    std::getline(f, header);
    // Check expected columns in header
    bool ok = header.find("barcode") != std::string::npos &&
              header.find("total_fragments") != std::string::npos &&
              header.find("mito_fraction") != std::string::npos &&
              header.find("tss_enrichment") != std::string::npos;
    if (ok)
        PASS("qc_tsv_header_columns");
    else
        FAIL("qc_tsv_header_columns", header.c_str());

    std::string row1;
    std::getline(f, row1);
    if (row1.find("CELL_A") != std::string::npos)
        PASS("qc_tsv_barcode_in_row1");
    else
        FAIL("qc_tsv_barcode_in_row1", row1.c_str());

    std::string row2;
    std::getline(f, row2);
    if (row2.find("CELL_B") != std::string::npos)
        PASS("qc_tsv_barcode_in_row2");
    else
        FAIL("qc_tsv_barcode_in_row2", row2.c_str());
}

// ── Test 6: Global histogram ────────────────────────────────────────────────
static void test_global_histogram() {
    std::unordered_map<std::string, int32_t> cmap = {{"chr1", 0}};
    ATACQCComputer qc;
    qc.set_chrom_map(cmap);
    qc.set_hist_bin_width(100);

    // Sizes: 50 (bin 0), 150 (bin 1), 250 (bin 2)
    std::vector<ATACFragment> frags = {
        make_frag(0, 0, 50, 0),
        make_frag(0, 0, 150, 0),
        make_frag(0, 0, 250, 0),
    };
    ATACFragmentExtractor ext;
    qc.compute(frags, ext, 1);

    auto hist = qc.fragment_size_histogram(100);
    if (hist.size() >= 3 && hist[0] == 1 && hist[1] == 1 && hist[2] == 1)
        PASS("global_histogram bins");
    else
        FAIL("global_histogram bins",
             ("size=" + std::to_string(hist.size())).c_str());
}

// ── main ────────────────────────────────────────────────────────────────────
int main() {
    test_mito_fragment();
    test_tss_enrichment();
    test_tss_far_not_counted();
    test_median_fragment_size();
    test_qc_tsv_output();
    test_global_histogram();

    std::cerr << "\n"
              << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
