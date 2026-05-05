#include "singlet/pileup/slamseq.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstdio>

using namespace singlet;

static const SLAMSeqParams DEFAULT_PARAMS{};  // min_tc=2, min_qual=27

// Helper: build quality string of specified length at good quality (Phred 30 → ASCII 63)
static std::string good_qual(size_t len) {
    return std::string(len, static_cast<char>(30 + 33));  // '?'
}

// ── Test 1: count_tc_conversions basic cases ──────────────────────────────────

static void test_basic_tc() {
    // Read: ACCCGA  Ref: ACTCGA
    // pos 2: read=C, ref=T → T→C conversion
    // pos 3: read=C, ref=C → no conversion (ref is not T)
    // Actually: ACTCGA ref, ACCCGA read
    // pos 0: A vs A — no event
    // pos 1: C vs C — no event
    // pos 2: C vs T — T→C ✓
    // pos 3: C vs C — no event (ref not T)
    // pos 4: G vs G — no event
    // pos 5: A vs A — no event
    // T positions in ref = 1 (pos 2), tc_count = 1
    {
        std::string read = "ACCCGA";
        std::string ref  = "ACTCGA";
        auto q = good_qual(6);
        auto conv = count_tc_conversions(read, ref, q, DEFAULT_PARAMS);
        assert(conv.t_positions == 1);
        assert(conv.tc_count == 1);
        assert(conv.other_conversions == 0);
    }

    // Acceptance criteria example: Read ACCCGA, Ref ACTCGA → task doc says 2 T→C at positions 2,3
    // Let's re-read: ref=ACTCGA: A C T C G A — only one T at pos 2.
    // The task description example has ref "ACTCGA" and shows 2 conversions which actually
    // requires two T's. Let's use the ref from the spec literally: ACTCGA has 1 T.
    // Use a cleaner example with 2 T→C:
    {
        // Read: ACCGCA  Ref: ATCGTA  → T at pos 1(→C) and pos 4(→C) → tc=2, t_pos=2
        std::string read = "ACCGCA";
        std::string ref  = "ATCGTA";
        auto q = good_qual(6);
        auto conv = count_tc_conversions(read, ref, q, DEFAULT_PARAMS);
        assert(conv.t_positions == 2);
        assert(conv.tc_count == 2);
        assert(conv.other_conversions == 0);
    }

    // No T→C: identical sequence
    {
        std::string read = "ACTTGA";
        std::string ref  = "ACTTGA";
        auto q = good_qual(6);
        auto conv = count_tc_conversions(read, ref, q, DEFAULT_PARAMS);
        assert(conv.tc_count == 0);
        assert(conv.t_positions == 2);  // two T's in ref
        assert(conv.other_conversions == 0);
    }

    std::cout << "PASS test_basic_tc\n";
}

// ── Test 2: is_new_rna threshold ─────────────────────────────────────────────

static void test_is_new_rna() {
    SLAMSeqParams p;
    p.min_tc_conversions = 2;

    ConversionCount c2{2, 5, 0};
    ConversionCount c1{1, 5, 0};
    ConversionCount c0{0, 5, 0};

    assert(is_new_rna(c2, p) == true);
    assert(is_new_rna(c1, p) == false);
    assert(is_new_rna(c0, p) == false);

    // min_tc=1
    p.min_tc_conversions = 1;
    assert(is_new_rna(c1, p) == true);

    std::cout << "PASS test_is_new_rna\n";
}

// ── Test 3: quality filtering ─────────────────────────────────────────────────

static void test_quality_filter() {
    SLAMSeqParams p;
    p.min_base_quality = 27;  // min Phred 27 → ASCII 60 = '<'

    // Quality below threshold at position of T→C: should be ignored
    std::string read = "ACCCGA";
    std::string ref  = "ACTCGA";
    // Put low quality at position 2 (the T→C site)
    std::string q = good_qual(6);
    q[2] = static_cast<char>(20 + 33);  // Phred 20 = '5', below 27

    auto conv = count_tc_conversions(read, ref, q, p);
    assert(conv.tc_count == 0);  // filtered out
    assert(conv.t_positions == 0);  // T at pos 2 skipped due to quality

    // High quality: conversion counted
    q[2] = static_cast<char>(30 + 33);
    auto conv2 = count_tc_conversions(read, ref, q, p);
    assert(conv2.tc_count == 1);

    std::cout << "PASS test_quality_filter\n";
}

// ── Test 4: background rate (non-T→C mismatches) ─────────────────────────────

static void test_background_rate() {
    // Read has an A→G mismatch (not T→C) and a T→C
    std::string read = "GCCCGA";
    std::string ref  = "ACTCGA";
    // pos 0: G vs A → other mismatch
    // pos 2: C vs T → T→C
    auto q = good_qual(6);
    auto conv = count_tc_conversions(read, ref, q, DEFAULT_PARAMS);
    assert(conv.tc_count == 1);
    assert(conv.t_positions == 1);
    assert(conv.other_conversions == 1);  // the A→G at pos 0

    std::cout << "PASS test_background_rate\n";
}

// ── Test 5: per-cell stats computation ───────────────────────────────────────

static void test_compute_slam_stats() {
    SLAMSeqParams p;
    p.min_tc_conversions = 2;

    // Cell 0: 3 reads — 2 new (tc≥2), 1 old
    // Cell 1: 2 reads — 0 new
    std::vector<std::vector<ConversionCount>> per_cell = {
        { {2, 4, 0}, {3, 5, 0}, {1, 3, 0} },  // cell 0
        { {0, 2, 0}, {1, 3, 1} }                // cell 1
    };

    auto stats = compute_slam_stats(per_cell, p);
    assert(stats.size() == 2);

    // Cell 0
    assert(stats[0].total_reads == 3);
    assert(stats[0].new_reads == 2);
    assert(stats[0].old_reads == 1);
    float frac0 = 2.0f / 3.0f;
    assert(std::fabs(stats[0].new_fraction - frac0) < 1e-5f);

    // mean_tc_rate: (2/4 + 3/5 + 1/3) / 3 = (0.5 + 0.6 + 0.333) / 3 = 1.433/3 ≈ 0.4778
    float expected_tc = (2.0f/4 + 3.0f/5 + 1.0f/3) / 3.0f;
    assert(std::fabs(stats[0].mean_tc_rate - expected_tc) < 1e-4f);

    // background_rate cell 0 = 0 (no other mismatches)
    assert(stats[0].background_rate == 0.0f);

    // Cell 1: 0 new
    assert(stats[1].total_reads == 2);
    assert(stats[1].new_reads == 0);
    assert(stats[1].old_reads == 2);
    assert(stats[1].new_fraction == 0.0f);
    // background_rate cell 1: (0/2 + 1/3) / 2 = 0.1667
    float expected_bg1 = (0.0f/2 + 1.0f/3) / 2.0f;
    assert(std::fabs(stats[1].background_rate - expected_bg1) < 1e-4f);

    std::cout << "PASS test_compute_slam_stats\n";
}

// ── Test 6: write_slam_stats_tsv ─────────────────────────────────────────────

static void test_write_tsv() {
    std::vector<CellSLAMStats> stats(2);
    stats[0] = {10, 4, 6, 0.4f, 0.12f, 0.01f};
    stats[1] = {8,  2, 6, 0.25f, 0.08f, 0.02f};

    std::vector<std::string> barcodes = {"AAACCC", "TTTGGG"};

    std::string path = "/tmp/test_slam_stats.tsv";
    bool ok = write_slam_stats_tsv(path, stats, barcodes);
    assert(ok);

    std::ifstream f(path);
    assert(f.is_open());
    std::string header;
    std::getline(f, header);
    assert(header.find("barcode") != std::string::npos);
    assert(header.find("total_reads") != std::string::npos);
    assert(header.find("new_reads") != std::string::npos);
    assert(header.find("old_reads") != std::string::npos);
    assert(header.find("new_fraction") != std::string::npos);
    assert(header.find("mean_tc_rate") != std::string::npos);
    assert(header.find("background_rate") != std::string::npos);

    std::string line1;
    std::getline(f, line1);
    assert(line1.find("AAACCC") != std::string::npos);
    assert(line1.find("10") != std::string::npos);

    std::string line2;
    std::getline(f, line2);
    assert(line2.find("TTTGGG") != std::string::npos);

    std::remove(path.c_str());
    std::cout << "PASS test_write_tsv\n";
}

// ── Test 7: minus-strand T→C via stranded function ───────────────────────────

static void test_stranded() {
    SLAMSeqParams p;
    p.min_tc_conversions = 1;

    // On + strand read: ref T → read C = T→C
    std::string read_plus = "ACCCGA";
    std::string ref_plus  = "ACTCGA";
    auto q = good_qual(6);
    auto conv_plus = count_tc_conversions_stranded(read_plus, ref_plus, q, p, false);
    assert(conv_plus.tc_count == 1);

    // On - strand read: ref (+ strand) A → read G = A→G (≡ T→C on minus strand)
    // ref: ACGAGT (has one A at pos 0 and one A at pos 4)
    // read: GCGAGT → pos 0: G vs A → T→C on minus strand
    std::string ref_minus  = "ACGAGT";
    std::string read_minus = "GCGAGT";
    auto conv_minus = count_tc_conversions_stranded(read_minus, ref_minus, q, p, true);
    assert(conv_minus.tc_count == 1);
    assert(conv_minus.t_positions == 2);  // A at pos 0 and pos 4

    // Unstranded (+ orientation) should not count A→G
    auto conv_unstranded = count_tc_conversions(read_minus, ref_minus, q, p);
    // ref has no T, so t_positions=0, tc_count=0
    assert(conv_unstranded.t_positions == 0);
    assert(conv_unstranded.tc_count == 0);

    std::cout << "PASS test_stranded\n";
}

// ── Test 8: mismatched lengths throw ─────────────────────────────────────────

static void test_length_mismatch() {
    bool threw = false;
    try {
        count_tc_conversions("ACG", "AC", good_qual(3), DEFAULT_PARAMS);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    assert(threw);
    std::cout << "PASS test_length_mismatch\n";
}

// ── Test 9: build_slam_count_matrix basic ────────────────────────────────────

static void test_count_matrix() {
    // 2 genes, 3 cells
    // gene 0: cell 0 → (2 new, 5 total), cell 2 → (1 new, 3 total)
    // gene 1: cell 1 → (3 new, 4 total)
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> gc = {
        { {2, 5}, {0, 0}, {1, 3} },  // gene 0
        { {0, 0}, {3, 4}, {0, 0} }   // gene 1
    };
    auto res = build_slam_count_matrix(gc, 2, 3);
    assert(res.nrows == 2);
    assert(res.ncols == 3);
    // new nnz: gene0→cell0(2), gene0→cell2(1), gene1→cell1(3) = 3 entries
    assert(res.new_data.size() == 3);
    // total nnz: same cells with non-zero total
    assert(res.total_data.size() == 3);
    // Check indptr for new: gene0 has 2 entries, gene1 has 1
    assert(res.new_indptr[0] == 0);
    assert(res.new_indptr[1] == 2);  // gene 0 has 2 new nonzeros
    assert(res.new_indptr[2] == 3);  // gene 1 has 1 new nonzero
    // Values: gene0: cell0=2, cell2=1
    assert(res.new_data[0] == 2);
    assert(res.new_data[1] == 1);
    assert(res.new_data[2] == 3);

    std::cout << "PASS test_count_matrix\n";
}

int main() {
    test_basic_tc();
    test_is_new_rna();
    test_quality_filter();
    test_background_rate();
    test_compute_slam_stats();
    test_write_tsv();
    test_stranded();
    test_length_mismatch();
    test_count_matrix();

    std::cout << "All slamseq tests passed.\n";
    return 0;
}
