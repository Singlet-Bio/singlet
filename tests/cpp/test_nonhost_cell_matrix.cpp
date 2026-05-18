// SPDX-License-Identifier: MIT
// test_nonhost_cell_matrix.cpp — Unit tests for nonhost_cell_matrix.h
//
// Tests NonHostCellMatrix::build(), ::read_mate_pairs(), and ::write().
// All tests are self-contained with no external files.
//
// Build:
//   ninja -C build test_nonhost_cell_matrix && ./build/test_nonhost_cell_matrix
//
// Tests:
//   1. empty_input           : zero tagged reads → empty matrix
//   2. single_species        : single cell, 3 reads to one species → correct n_reads
//   3. umi_dedup             : 3 reads with same CB/UMI → n_reads=3, n_umis=1
//   4. barcode_filter        : reads from cells not in valid_barcodes excluded
//   5. multi_species         : 2 cells × 2 species → 4 records, correct per-cell counts
//   6. kingdom_map_override  : kingdom_map overrides MultiHit's per-read inference
//   7. read_mate_pairs_basic : parse paired FASTQ files, extract CB+UMI correctly
//   8. write_output_format   : write() produces correct TSV header + rows
//   9. empty_umi             : reads with empty UMI tracked in n_reads but n_umis=0
//  10. no_valid_barcodes     : valid_barcodes empty set means ALL barcodes pass

#include "singlet/pileup/nonhost/nonhost_cell_matrix.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace singlet::nonhost;

// ── Test infrastructure ───────────────────────────────────────────────────────

static int g_passed = 0, g_failed = 0;

static void PASS(const std::string& name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const std::string& name, const std::string& msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}
#define CHECK(name, cond, msg) do { if (cond) PASS(name); else FAIL(name, msg); } while (0)

// ── Helper: build a single MultiHit with one viral species hit ───────────────

static NonHostScreener::MultiHit make_viral_hit(uint32_t sid, float rate) {
    NonHostScreener::MultiHit mh;
    mh.top_category = NonHostCategory::VIRAL;
    mh.viral_hits.emplace_back(sid, rate);
    return mh;
}

static NonHostScreener::MultiHit make_bacterial_hit(uint32_t sid, float rate) {
    NonHostScreener::MultiHit mh;
    mh.top_category = NonHostCategory::BACTERIAL;
    mh.microbial_hits.emplace_back(sid, rate);
    return mh;
}

[[maybe_unused]] static NonHostScreener::MultiHit make_no_hit() {
    return {};   // UNKNOWN, empty hit lists
}

// ── Test 1: empty_input ───────────────────────────────────────────────────────

static void test_empty_input() {
    std::vector<TaggedRead> tagged;
    std::vector<NonHostScreener::MultiHit> hits;
    std::unordered_set<std::string> valid{ "AAACCCAAGAAACACT" };
    std::unordered_map<uint32_t, std::string> names;
    std::unordered_map<uint32_t, NonHostCategory> kmap;

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);
    CHECK("empty_input.n_records", mat.n_records() == 0u,
          "expected 0 records for empty input, got " + std::to_string(mat.n_records()));
}

// ── Test 2: single_species ────────────────────────────────────────────────────

static void test_single_species() {
    const std::string CB = "AAACCCAAGAAACACT";
    const uint32_t SID = 1001;

    std::vector<TaggedRead> tagged = {
        {"ACTG....", CB, "AAAAAAAAAA"},
        {"ACTG....", CB, "BBBBBBBBBB"},
        {"ACTG....", CB, "CCCCCCCCCC"},
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_viral_hit(SID, 0.8f),
        make_viral_hit(SID, 0.7f),
        make_viral_hit(SID, 0.6f),
    };
    std::unordered_set<std::string> valid{ CB };
    std::unordered_map<uint32_t, std::string> names{{SID, "TestVirus"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::VIRAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("single_species.n_records", mat.n_records() == 1u,
          "expected 1 record, got " + std::to_string(mat.n_records()));
    if (mat.n_records() == 1u) {
        const auto& r = mat.records()[0];
        CHECK("single_species.n_reads", r.n_reads == 3u,
              "expected n_reads=3, got " + std::to_string(r.n_reads));
        CHECK("single_species.n_umis", r.n_umis == 3u,
              "expected n_umis=3 (3 distinct UMIs), got " + std::to_string(r.n_umis));
        CHECK("single_species.kingdom", r.kingdom == NonHostCategory::VIRAL,
              "expected VIRAL kingdom");
        CHECK("single_species.species_name", r.species_name == "TestVirus",
              "expected 'TestVirus', got '" + r.species_name + "'");
        CHECK("single_species.n_genes", r.n_genes == 0u,
              "expected n_genes=0, got " + std::to_string(r.n_genes));
    }
}

// ── Test 3: umi_dedup ─────────────────────────────────────────────────────────

static void test_umi_dedup() {
    const std::string CB = "TTTTGGTTCAAGATCC";
    const uint32_t SID = 2002;
    const std::string UMI = "GGGGGGGGGGGG";  // same UMI for all 3 reads

    std::vector<TaggedRead> tagged = {
        {"ACGTACGT", CB, UMI},
        {"ACGTACGT", CB, UMI},
        {"ACGTACGT", CB, UMI},
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_bacterial_hit(SID, 0.5f),
        make_bacterial_hit(SID, 0.5f),
        make_bacterial_hit(SID, 0.5f),
    };
    std::unordered_set<std::string> valid{ CB };
    std::unordered_map<uint32_t, std::string> names{{SID, "E. coli"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::BACTERIAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("umi_dedup.n_records", mat.n_records() == 1u,
          "expected 1 record");
    if (mat.n_records() == 1u) {
        const auto& r = mat.records()[0];
        CHECK("umi_dedup.n_reads", r.n_reads == 3u,
              "expected n_reads=3, got " + std::to_string(r.n_reads));
        CHECK("umi_dedup.n_umis", r.n_umis == 1u,
              "expected n_umis=1 (same UMI x3), got " + std::to_string(r.n_umis));
    }
}

// ── Test 4: barcode_filter ────────────────────────────────────────────────────

static void test_barcode_filter() {
    const std::string GOOD_CB  = "AAACCCAAGAAACACT";
    const std::string BAD_CB   = "TTTTTTTTTTTTTTTT";
    const uint32_t SID = 3003;

    std::vector<TaggedRead> tagged = {
        {"ACGTACGT", GOOD_CB, "AAAAAAAAAA"},  // in valid set
        {"ACGTACGT", BAD_CB,  "BBBBBBBBBB"},  // NOT in valid set
        {"ACGTACGT", GOOD_CB, "CCCCCCCCCC"},  // in valid set
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_viral_hit(SID, 0.9f),
        make_viral_hit(SID, 0.9f),
        make_viral_hit(SID, 0.9f),
    };
    std::unordered_set<std::string> valid{ GOOD_CB };   // BAD_CB excluded
    std::unordered_map<uint32_t, std::string> names{{SID, "FilterVirus"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::VIRAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("barcode_filter.n_records", mat.n_records() == 1u,
          "expected 1 record (BAD_CB filtered), got " + std::to_string(mat.n_records()));
    if (mat.n_records() == 1u) {
        const auto& r = mat.records()[0];
        CHECK("barcode_filter.cb_correct", r.cb == GOOD_CB,
              "expected GOOD_CB in output, got '" + r.cb + "'");
        CHECK("barcode_filter.n_reads", r.n_reads == 2u,
              "expected n_reads=2 (GOOD_CB reads only), got " + std::to_string(r.n_reads));
    }
}

// ── Test 5: multi_species ─────────────────────────────────────────────────────

static void test_multi_species() {
    const std::string CB1 = "AAACCCAAGAAACACT";
    const std::string CB2 = "GGGGTTTTCCCCAAAA";
    const uint32_t SID_V = 101;  // viral
    const uint32_t SID_B = 202;  // bacterial

    // 4 reads: (CB1→viral), (CB1→bacterial), (CB2→viral), (CB2→bacterial)
    std::vector<TaggedRead> tagged = {
        {"ACGT", CB1, "UMI1"},
        {"ACGT", CB1, "UMI2"},
        {"ACGT", CB2, "UMI3"},
        {"ACGT", CB2, "UMI4"},
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_viral_hit(SID_V, 0.9f),
        make_bacterial_hit(SID_B, 0.8f),
        make_viral_hit(SID_V, 0.9f),
        make_bacterial_hit(SID_B, 0.8f),
    };
    std::unordered_set<std::string> valid{ CB1, CB2 };
    std::unordered_map<uint32_t, std::string> names{
        {SID_V, "VirusA"}, {SID_B, "BacteriumB"}
    };
    std::unordered_map<uint32_t, NonHostCategory> kmap{
        {SID_V, NonHostCategory::VIRAL}, {SID_B, NonHostCategory::BACTERIAL}
    };

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("multi_species.n_records", mat.n_records() == 4u,
          "expected 4 records (2 cells × 2 species), got " + std::to_string(mat.n_records()));

    // Count n_reads per (CB, kind) to validate
    uint32_t cb1_v = 0, cb1_b = 0, cb2_v = 0, cb2_b = 0;
    for (const auto& r : mat.records()) {
        if (r.cb == CB1 && r.kingdom == NonHostCategory::VIRAL)     cb1_v = r.n_reads;
        if (r.cb == CB1 && r.kingdom == NonHostCategory::BACTERIAL) cb1_b = r.n_reads;
        if (r.cb == CB2 && r.kingdom == NonHostCategory::VIRAL)     cb2_v = r.n_reads;
        if (r.cb == CB2 && r.kingdom == NonHostCategory::BACTERIAL) cb2_b = r.n_reads;
    }
    CHECK("multi_species.cb1_viral",      cb1_v == 1u, "CB1 viral n_reads should be 1");
    CHECK("multi_species.cb1_bacterial",  cb1_b == 1u, "CB1 bacterial n_reads should be 1");
    CHECK("multi_species.cb2_viral",      cb2_v == 1u, "CB2 viral n_reads should be 1");
    CHECK("multi_species.cb2_bacterial",  cb2_b == 1u, "CB2 bacterial n_reads should be 1");
}

// ── Test 6: kingdom_map_override ──────────────────────────────────────────────

static void test_kingdom_map_override() {
    const std::string CB  = "ACTGACTGACTGACTG";
    const uint32_t SID = 999;

    // MultiHit says BACTERIAL, but kingdom_map says FUNGAL (Candida, for example)
    std::vector<TaggedRead> tagged = { {"ACGT", CB, "UMIUMI"} };
    std::vector<NonHostScreener::MultiHit> hits = { make_bacterial_hit(SID, 0.7f) };
    std::unordered_set<std::string> valid{ CB };
    std::unordered_map<uint32_t, std::string> names{{SID, "Candida albicans"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::FUNGAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("kingdom_override.n_records", mat.n_records() == 1u, "expected 1 record");
    if (mat.n_records() == 1u) {
        CHECK("kingdom_override.kingdom", mat.records()[0].kingdom == NonHostCategory::FUNGAL,
              "expected FUNGAL from kingdom_map override");
    }
}

// ── Test 7: read_mate_pairs_basic ─────────────────────────────────────────────

static void test_read_mate_pairs_basic() {
    // Write two paired FASTQ files into /tmp
    const std::string m1 = "/tmp/test_nhcm_mate1.fastq";
    const std::string m2 = "/tmp/test_nhcm_mate2.fastq";

    {
        std::ofstream f1(m1);
        f1 << "@read1\n" << "ACGTACGT\n" << "+\n" << "IIIIIIII\n";
        f1 << "@read2\n" << "TTTTTTTT\n" << "+\n" << "IIIIIIII\n";
    }
    {
        // mate2: 16nt barcode + 12nt UMI = 28nt
        std::ofstream f2(m2);
        f2 << "@read1\n" << "AAACCCAAGAAACACT" "AAAAAAAAAAAA\n" << "+\n" << std::string(28, 'I') << "\n";
        f2 << "@read2\n" << "TTTTGGTTCAAGATCC" "BBBBBBBBBBBB\n" << "+\n" << std::string(28, 'I') << "\n";
    }

    auto tagged = NonHostCellMatrix::read_mate_pairs(m1, m2, 16, 12);

    CHECK("mate_pairs.count", tagged.size() == 2u,
          "expected 2 tagged reads, got " + std::to_string(tagged.size()));
    if (tagged.size() == 2u) {
        CHECK("mate_pairs.seq0",  tagged[0].seq == "ACGTACGT",
              "expected 'ACGTACGT', got '" + tagged[0].seq + "'");
        CHECK("mate_pairs.cb0",   tagged[0].cb  == "AAACCCAAGAAACACT",
              "wrong CB[0]: " + tagged[0].cb);
        CHECK("mate_pairs.umi0",  tagged[0].umi == "AAAAAAAAAAAA",
              "wrong UMI[0]: " + tagged[0].umi);
        CHECK("mate_pairs.cb1",   tagged[1].cb  == "TTTTGGTTCAAGATCC",
              "wrong CB[1]: " + tagged[1].cb);
        CHECK("mate_pairs.umi1",  tagged[1].umi == "BBBBBBBBBBBB",
              "wrong UMI[1]: " + tagged[1].umi);
    }

    std::remove(m1.c_str());
    std::remove(m2.c_str());
}

// ── Test 8: write_output_format ───────────────────────────────────────────────

static void test_write_output_format() {
    const std::string CB  = "AAACCCAAGAAACACT";
    const uint32_t SID = 42;

    std::vector<TaggedRead> tagged = { {"ACGT", CB, "UMI1"} };
    std::vector<NonHostScreener::MultiHit> hits = { make_viral_hit(SID, 0.9f) };
    std::unordered_set<std::string> valid{ CB };
    std::unordered_map<uint32_t, std::string> names{{SID, "WriteTestVirus"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::VIRAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    // Write to /tmp
    const std::string out_dir = "/tmp/test_nhcm_write";
    std::filesystem::create_directories(out_dir);
    mat.write(out_dir);

    const std::string tsv_path = out_dir + "/nonhost_per_cell.tsv";
    std::ifstream tsv(tsv_path);
    bool file_exists = tsv.is_open();
    CHECK("write_format.file_created", file_exists, "nonhost_per_cell.tsv not created");

    if (file_exists) {
        std::string header, row;
        std::getline(tsv, header);
        std::getline(tsv, row);
        CHECK("write_format.header",
              header == "barcode\tspecies_name\tspecies_taxon_id\tkingdom\tn_reads\tn_umis\tn_genes",
              "unexpected header: " + header);
        // Row should contain the CB and species name
        CHECK("write_format.row_has_cb",     row.find(CB) != std::string::npos,
              "row missing CB: " + row);
        CHECK("write_format.row_has_name",   row.find("WriteTestVirus") != std::string::npos,
              "row missing species name: " + row);
        CHECK("write_format.row_has_viral",  row.find("VIRAL") != std::string::npos,
              "row missing VIRAL kingdom: " + row);
    }

    std::filesystem::remove_all(out_dir);
}

// ── Test 9: empty_umi ─────────────────────────────────────────────────────────
// Reads with empty UMI still counted in n_reads; n_umis stays 0

static void test_empty_umi() {
    const std::string CB = "ACGTACGTACGTACGT";
    const uint32_t SID = 55;

    // UMI is intentionally empty (cb_len ≥ r1_len, so UMI extraction returns empty)
    std::vector<TaggedRead> tagged = {
        {"ACGT", CB, ""},
        {"ACGT", CB, ""},
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_viral_hit(SID, 0.8f),
        make_viral_hit(SID, 0.8f),
    };
    std::unordered_set<std::string> valid{ CB };
    std::unordered_map<uint32_t, std::string> names{{SID, "ShortRead"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::VIRAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    CHECK("empty_umi.n_records", mat.n_records() == 1u, "expected 1 record");
    if (mat.n_records() == 1u) {
        CHECK("empty_umi.n_reads", mat.records()[0].n_reads == 2u,
              "expected n_reads=2");
        // Empty UMIs: set contains one empty string → n_umis=0 because
        // empty strings are NOT inserted (implementation choice: skip empty UMI)
        CHECK("empty_umi.n_umis_zero", mat.records()[0].n_umis == 0u,
              "expected n_umis=0 for all-empty UMIs");
    }
}

// ── Test 10: no_valid_barcodes → all pass ─────────────────────────────────────

static void test_no_valid_barcodes_all_pass() {
    const std::string CB1 = "AAACCCAAGAAACACT";
    const std::string CB2 = "TTTTGGTTCAAGATCC";
    const uint32_t SID = 77;

    std::vector<TaggedRead> tagged = {
        {"ACGT", CB1, "UMI1"},
        {"ACGT", CB2, "UMI2"},
    };
    std::vector<NonHostScreener::MultiHit> hits = {
        make_viral_hit(SID, 0.9f),
        make_viral_hit(SID, 0.9f),
    };
    std::unordered_set<std::string> valid{};  // EMPTY — all cells pass
    std::unordered_map<uint32_t, std::string> names{{SID, "Virus"}};
    std::unordered_map<uint32_t, NonHostCategory> kmap{{SID, NonHostCategory::VIRAL}};

    auto mat = NonHostCellMatrix::build(tagged, hits, valid, names, kmap);

    // With empty valid_barcodes, build() passes all reads through
    CHECK("no_valid_barcodes.n_records", mat.n_records() == 2u,
          "expected 2 records when valid_barcodes is empty, got "
          + std::to_string(mat.n_records()));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cerr << "=== test_nonhost_cell_matrix ===\n";

    test_empty_input();
    test_single_species();
    test_umi_dedup();
    test_barcode_filter();
    test_multi_species();
    test_kingdom_map_override();
    test_read_mate_pairs_basic();
    test_write_output_format();
    test_empty_umi();
    test_no_valid_barcodes_all_pass();

    std::cerr << "\n--- Results: "
              << g_passed << " passed, " << g_failed << " failed ---\n";

    return g_failed == 0 ? 0 : 1;
}
