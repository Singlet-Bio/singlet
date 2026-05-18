// SPDX-License-Identifier: MIT
// test/test_tagged_bam.cpp
// Unit tests for TaggedBamWriter (G-TAGGED-BAM).
//
// Synthetic BAM records are constructed with htslib (no real BAM on disk).
// A small in-memory BAM is written and read back with sam_open() to verify
// all Cell Ranger-style tags are present and correct.
//
// Built with PILEUP_INCLUDE_DIRS + PILEUP_LINK_LIBS (real htslib required).

#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

#include "singlet/pileup/tagged_bam.h"

using namespace singlet;
namespace fs = std::filesystem;

// ── Helpers ────────────────────────────────────────────────────────────────

#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

// Build a minimal sam_hdr_t with 1 reference "chr1" and SO:coordinate tag.
static sam_hdr_t* make_header() {
    sam_hdr_t* h = sam_hdr_init();
    sam_hdr_add_line(h, "HD", "VN", "1.6", "SO", "coordinate", nullptr);
    sam_hdr_add_line(h, "SQ", "SN", "chr1", "LN", "248956422", nullptr);
    sam_hdr_add_line(h, "SQ", "SN", "chr2", "LN", "242193529", nullptr);
    return h;
}

// Build a minimal mapped BAM record.
// Sequence and quality are set to `seq_len` characters for realism but
// the content doesn't matter for tag-testing purposes.
static bam1_t* make_bam_record(const char* qname,
                                int32_t     tid,
                                int32_t     pos,
                                uint16_t    flag    = 0,
                                int         seq_len = 20) {
    bam1_t* b = bam_init1();

    // Fill with bam_set1 to get a properly structured record
    // cigar: one M operation spanning seq_len bases
    uint32_t cigar_op = seq_len << 4 | BAM_CMATCH;  // e.g. "20M"
    std::string seq(seq_len, 'A');  // dummy sequence
    std::string qual(seq_len, (char)30);  // dummy qual (Phred 30)

    bam_set1(b,
             strlen(qname),        // l_qname (without NUL — bam_set1 adds it)
             qname,
             flag,
             tid,
             pos,
             60,                   // mapq
             1, &cigar_op,         // n_cigar, cigar
             -1, -1, 0,            // mtid, mpos, isize
             seq_len,              // l_seq
             seq.c_str(),
             qual.c_str(),
             0);                   // l_aux (no initial aux)
    return b;
}

static void free_record(bam1_t* b) { bam_destroy1(b); }

// Read back a BAM file and collect all records.
static std::vector<bam1_t*> read_bam(const std::string& path, sam_hdr_t** hdr_out) {
    samFile* fp = sam_open(path.c_str(), "rb");
    assert(fp);
    *hdr_out = sam_hdr_read(fp);
    assert(*hdr_out);
    std::vector<bam1_t*> recs;
    bam1_t* b = bam_init1();
    while (sam_read1(fp, *hdr_out, b) >= 0) {
        recs.push_back(bam_dup1(b));
    }
    bam_destroy1(b);
    sam_close(fp);
    return recs;
}

static void free_records(std::vector<bam1_t*>& recs) {
    for (auto* b : recs) bam_destroy1(b);
    recs.clear();
}

// Temporary path in /dev/shm to keep tests fast
static std::string tmp_path(const char* name) {
    return std::string("/dev/shm/test_tagged_bam_") + name + ".bam";
}

// ── Test 1: open/close — file exists after close ───────────────────────────
static void test_open_close() {
    std::cout << "Test 1: open/close\n";
    std::string path = tmp_path("open_close");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    bool ok = w.open(cfg, hdr);
    CHECK(ok, "open returns true");
    CHECK(w.is_open(), "is_open() after open");
    w.close();
    CHECK(!w.is_open(), "is_open() after close");
    CHECK(fs::exists(path), "BAM file exists on disk");
    CHECK(fs::file_size(path) > 0, "BAM file non-empty");

    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 2: tag presence — CB, UB, GX, RE present in written record ────────
static void test_tag_presence() {
    std::cout << "Test 2: tag presence\n";
    std::string path = tmp_path("tag_presence");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("READ001", 0, 1000);
    bool wrote = w.write_record(b, "ACGTACGT", "ACGTACGT", "", "ATCGATCG",
                                 "ATCGATCG", "", "ENSG00000001", "GENE1", 'E');
    CHECK(wrote, "write_record returns true");
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 1u, "one record in output");

    if (!recs.empty()) {
        bam1_t* r = recs[0];
        CHECK(bam_aux_get(r, "CB") != nullptr, "CB tag present");
        CHECK(bam_aux_get(r, "CR") != nullptr, "CR tag present");
        CHECK(bam_aux_get(r, "UB") != nullptr, "UB tag present");
        CHECK(bam_aux_get(r, "UR") != nullptr, "UR tag present");
        CHECK(bam_aux_get(r, "GX") != nullptr, "GX tag present");
        CHECK(bam_aux_get(r, "GN") != nullptr, "GN tag present");
        CHECK(bam_aux_get(r, "RE") != nullptr, "RE tag present");
        CHECK(bam_aux_get(r, "xf") != nullptr, "xf tag present");
    }

    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 3: tag values — verify content matches what was written ────────────
static void test_tag_values() {
    std::cout << "Test 3: tag values\n";
    std::string path = tmp_path("tag_values");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("READ002", 0, 5000);
    w.write_record(b, "TTTTAAAA", "TTTTGGGG", "", "CCCCCCCC", "CCCCTTTT",
                   "", "ENSG00000099", "BRCA2", 'N');
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 1u, "one record");

    if (!recs.empty()) {
        bam1_t* r = recs[0];

        // CB:Z should be barcode + "-1"
        uint8_t* cb_ptr = bam_aux_get(r, "CB");
        if (cb_ptr) {
            std::string cb_val = bam_aux2Z(cb_ptr);
            CHECK(cb_val == "TTTTAAAA-1", "CB = barcode + '-1'");
        } else {
            CHECK(false, "CB tag found for value check");
        }

        uint8_t* cr_ptr = bam_aux_get(r, "CR");
        if (cr_ptr) {
            CHECK(std::string(bam_aux2Z(cr_ptr)) == "TTTTGGGG", "CR = raw barcode");
        } else {
            CHECK(false, "CR tag found for value check");
        }

        uint8_t* ub_ptr = bam_aux_get(r, "UB");
        if (ub_ptr) {
            CHECK(std::string(bam_aux2Z(ub_ptr)) == "CCCCCCCC", "UB = corrected UMI");
        } else {
            CHECK(false, "UB tag found for value check");
        }

        uint8_t* gn_ptr = bam_aux_get(r, "GN");
        if (gn_ptr) {
            CHECK(std::string(bam_aux2Z(gn_ptr)) == "BRCA2", "GN = gene name");
        } else {
            CHECK(false, "GN tag found for value check");
        }

        uint8_t* gx_ptr = bam_aux_get(r, "GX");
        if (gx_ptr) {
            CHECK(std::string(bam_aux2Z(gx_ptr)) == "ENSG00000099", "GX = gene ID");
        } else {
            CHECK(false, "GX tag found for value check");
        }

        uint8_t* re_ptr = bam_aux_get(r, "RE");
        if (re_ptr) {
            CHECK(bam_aux2A(re_ptr) == 'N', "RE = 'N' (intronic)");
        } else {
            CHECK(false, "RE tag found for value check");
        }
    }

    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 4: cell calling filter — only called-barcode reads written ─────────
static void test_cell_filter() {
    std::cout << "Test 4: cell calling filter\n";
    std::string path = tmp_path("cell_filter");

    sam_hdr_t* hdr = make_header();
    std::unordered_set<std::string> called = {"AAAAAAAA", "CCCCCCCC"};

    TaggedBamConfig cfg;
    cfg.output_path    = path;
    cfg.add_index      = false;
    cfg.threads        = 1;
    cfg.called_barcodes = &called;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    // 3 reads: 2 from called cells, 1 from uncalled
    for (int i = 0; i < 3; ++i) {
        bam1_t* b = make_bam_record(("R" + std::to_string(i)).c_str(), 0, i * 100);
        std::string bc = (i == 0) ? "AAAAAAAA" : (i == 1) ? "CCCCCCCC" : "GGGGGGGG";
        w.write_record(b, bc, bc, "", "UUUUUUUU", "UUUUUUUU", "",
                       "ENSG00000001", "GENE1", 'E');
        free_record(b);
    }
    w.close();

    CHECK(w.records_written() == 2u, "2 records written (called cells)");
    CHECK(w.records_skipped() == 1u, "1 record skipped (uncalled cell)");

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 2u, "2 records in output BAM");
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 5: stats — records_written() and records_skipped() ────────────────
static void test_stats() {
    std::cout << "Test 5: stats\n";
    std::string path = tmp_path("stats");

    sam_hdr_t* hdr = make_header();
    std::unordered_set<std::string> called = {"BC1", "BC2", "BC3"};

    TaggedBamConfig cfg;
    cfg.output_path    = path;
    cfg.add_index      = false;
    cfg.threads        = 1;
    cfg.called_barcodes = &called;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    // 5 from called, 3 from uncalled
    for (int i = 0; i < 8; ++i) {
        bam1_t* b = make_bam_record(("S" + std::to_string(i)).c_str(), 0, i * 50);
        std::string bc = (i < 5) ? (std::string("BC") + std::to_string((i % 3) + 1)) : "ZZZZZZZZ";
        w.write_record(b, bc, bc, "", "UUUU", "UUUU", "", "GENE_ID", "GENE", 'E');
        free_record(b);
    }
    w.close();

    CHECK(w.records_written() == 5u, "5 records written");
    CHECK(w.records_skipped() == 3u, "3 records skipped");

    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 6: intergenic read — RE='I', GX/GN absent ─────────────────────────
static void test_intergenic_read() {
    std::cout << "Test 6: intergenic read\n";
    std::string path = tmp_path("intergenic");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("READ_IG", 0, 9000);
    // No gene_id or gene_name → intergenic
    w.write_record(b, "ACGTACGT", "ACGTACGT", "ATCGATCG", "ATCGATCG",
                   /*gene_name=*/"", /*gene_id=*/"", 'I');
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 1u, "one record");

    if (!recs.empty()) {
        bam1_t* r = recs[0];
        uint8_t* re_ptr = bam_aux_get(r, "RE");
        if (re_ptr) {
            CHECK(bam_aux2A(re_ptr) == 'I', "RE = 'I' for intergenic");
        } else {
            CHECK(false, "RE tag present for intergenic");
        }
        // GX and GN should be absent (empty string means no tag written)
        CHECK(bam_aux_get(r, "GX") == nullptr, "GX absent for intergenic");
        CHECK(bam_aux_get(r, "GN") == nullptr, "GN absent for intergenic");
    }

    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 7: multiple records — write 100, read back 100 ────────────────────
static void test_multiple_records() {
    std::cout << "Test 7: 100 records round-trip\n";
    std::string path = tmp_path("multi100");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 2;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    for (int i = 0; i < 100; ++i) {
        bam1_t* b = make_bam_record(("M" + std::to_string(i)).c_str(),
                                    0, i * 1000);
        w.write_record(b, "ATCGATCG", "ATCGATCG", "TTTTTTTT", "TTTTTTTT",
                       "ACTB", "ENSG00000075624", 'E');
        free_record(b);
    }
    w.close();

    CHECK(w.records_written() == 100u, "100 records written");

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 100u, "100 records readable");
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 8: all three region types ─────────────────────────────────────────
static void test_region_types() {
    std::cout << "Test 8: region types E / N / I\n";
    std::string path = tmp_path("regions");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    const char types[] = {'E', 'N', 'I'};
    for (int i = 0; i < 3; ++i) {
        bam1_t* b = make_bam_record(("RG" + std::to_string(i)).c_str(), 0, i * 200);
        bool ok = (types[i] != 'I')
            ? w.write_record(b, "AABBCCDD", "AABBCCDD", "EEFFGGHH", "EEFFGGHH",
                             "GAPDH", "ENSG00000111640", types[i])
            : w.write_record(b, "AABBCCDD", "AABBCCDD", "EEFFGGHH", "EEFFGGHH",
                             "", "", types[i]);
        CHECK(ok, std::string("record ") + types[i] + " written");
        free_record(b);
    }
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 3u, "3 records in output");

    const char expected[] = {'E', 'N', 'I'};
    for (size_t i = 0; i < recs.size() && i < 3; ++i) {
        uint8_t* re_ptr = bam_aux_get(recs[i], "RE");
        if (re_ptr) {
            char got = bam_aux2A(re_ptr);
            CHECK(got == expected[i],
                  std::string("RE[") + std::to_string(i) + "] = " + expected[i]);
        } else {
            CHECK(false, std::string("RE tag present for record ") + std::to_string(i));
        }
    }

    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 9: BAM validity — samtools can read it (samtools view returns 0) ───
static void test_bam_validity() {
    std::cout << "Test 9: BAM validity (samtools)\n";
    std::string path = tmp_path("validity");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    for (int i = 0; i < 5; ++i) {
        bam1_t* b = make_bam_record(("V" + std::to_string(i)).c_str(), 0, i * 500);
        w.write_record(b, "ACGTACGT", "ACGTACGT", "GCGCGCGC", "GCGCGCGC",
                       "ACTB", "ENSG00000075624", 'E');
        free_record(b);
    }
    w.close();
    sam_hdr_destroy(hdr);

    // Use htslib itself to re-open and count records (no samtools binary needed)
    samFile* fp = sam_open(path.c_str(), "rb");
    CHECK(fp != nullptr, "sam_open succeeds on written BAM");
    if (fp) {
        sam_hdr_t* h2 = sam_hdr_read(fp);
        CHECK(h2 != nullptr, "sam_hdr_read succeeds");
        int count = 0;
        bam1_t* b = bam_init1();
        while (sam_read1(fp, h2, b) >= 0) ++count;
        bam_destroy1(b);
        CHECK(count == 5, "5 records readable by htslib");
        if (h2) sam_hdr_destroy(h2);
        sam_close(fp);
    }

    fs::remove(path);
}

// ── Test 10: SO:coordinate header ──────────────────────────────────────────
static void test_coordinate_sorted_header() {
    std::cout << "Test 10: SO:coordinate in header\n";
    std::string path = tmp_path("sorted");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);
    // Write one record — we just need a valid BAM to read back
    bam1_t* b = make_bam_record("SORTED_R1", 0, 1000);
    w.write_record(b, "TTTTTTTT", "TTTTTTTT", "AAAAAAAA", "AAAAAAAA",
                   "TP53", "ENSG00000141510", 'E');
    free_record(b);
    w.close();
    sam_hdr_destroy(hdr);

    samFile* fp = sam_open(path.c_str(), "rb");
    CHECK(fp != nullptr, "re-open for header check");
    if (fp) {
        sam_hdr_t* h2 = sam_hdr_read(fp);
        CHECK(h2 != nullptr, "header readable");
        if (h2) {
            // sam_hdr_str() returns the full SAM text header — search for SO:coordinate
            const char* hdr_txt = sam_hdr_str(h2);
            bool has_coord = (hdr_txt != nullptr &&
                              std::string(hdr_txt).find("SO:coordinate") != std::string::npos);
            CHECK(has_coord, "SO:coordinate in @HD line");
            sam_hdr_destroy(h2);
        }
        sam_close(fp);
    }

    fs::remove(path);
}

// ── Test 11: quality tags — CY and UY written when provided ────────────────
static void test_quality_tags() {
    std::cout << "Test 11: quality tags CY and UY\n";
    std::string path = tmp_path("qual_tags");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("QUAL_R1", 0, 2000);
    w.write_record(b,
                   "ACGTACGT",   // barcode
                   "ACGTACGT",   // raw_barcode
                   "FFFFFFFF",   // bc_qual
                   "GCGCGCGC",   // umi
                   "GCGCGCGC",   // raw_umi
                   "HHHHHHHH",   // umi_qual
                   "ENSG00000001", // gene_id
                   "MYC",          // gene_name
                   'E',
                   1);  // xf = GEX
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 1u, "one record");
    if (!recs.empty()) {
        bam1_t* r = recs[0];
        uint8_t* cy = bam_aux_get(r, "CY");
        uint8_t* uy = bam_aux_get(r, "UY");
        CHECK(cy != nullptr, "CY quality tag present");
        CHECK(uy != nullptr, "UY quality tag present");
        if (cy) CHECK(std::string(bam_aux2Z(cy)) == "FFFFFFFF", "CY value correct");
        if (uy) CHECK(std::string(bam_aux2Z(uy)) == "HHHHHHHH", "UY value correct");
    }
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 12: empty barcode — no CB/CR tags written ─────────────────────────
static void test_empty_barcode() {
    std::cout << "Test 12: empty barcode — no CB/CR tags\n";
    std::string path = tmp_path("empty_bc");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("NOBC_R1", 0, 3000);
    // Empty barcode and raw_barcode
    w.write_record(b, "", "", "ATCGATCG", "ATCGATCG",
                   "ACTB", "ENSG00000075624", 'E');
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 1u, "one record written");
    if (!recs.empty()) {
        bam1_t* r = recs[0];
        CHECK(bam_aux_get(r, "CB") == nullptr, "CB absent when barcode empty");
        CHECK(bam_aux_get(r, "CR") == nullptr, "CR absent when raw_barcode empty");
        CHECK(bam_aux_get(r, "GX") != nullptr, "GX present regardless");
    }
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 13: no-filter mode — all reads written regardless of barcode ───────
static void test_no_filter() {
    std::cout << "Test 13: no filter — all reads written\n";
    std::string path = tmp_path("no_filter");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path    = path;
    cfg.add_index      = false;
    cfg.threads        = 1;
    cfg.called_barcodes = nullptr;  // no filter

    TaggedBamWriter w;
    w.open(cfg, hdr);

    for (int i = 0; i < 10; ++i) {
        bam1_t* b = make_bam_record(("NF" + std::to_string(i)).c_str(), 0, i * 300);
        std::string bc = "RANDOM_" + std::to_string(i);
        w.write_record(b, bc, bc, "GCGCGCGC", "GCGCGCGC",
                       "GAPDH", "ENSG00000111640", 'E');
        free_record(b);
    }
    w.close();

    CHECK(w.records_written() == 10u, "10 records written (no filter)");
    CHECK(w.records_skipped() == 0u, "0 records skipped (no filter)");

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    CHECK(recs.size() == 10u, "10 records in output");
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 14: xf tag value ───────────────────────────────────────────────────
static void test_xf_tag() {
    std::cout << "Test 14: xf feature type tag\n";
    std::string path = tmp_path("xf_tag");

    sam_hdr_t* hdr = make_header();
    TaggedBamConfig cfg;
    cfg.output_path = path;
    cfg.add_index   = false;
    cfg.threads     = 1;

    TaggedBamWriter w;
    w.open(cfg, hdr);

    bam1_t* b = make_bam_record("XF_R1", 0, 7000);
    // GEX: xf=1
    w.write_record(b, "AACCTTGG", "AACCTTGG", "",
                   "GCGCGCGC", "GCGCGCGC", "",
                   "ENSG00000075624", "ACTB", 'E', /*xf=*/1);
    free_record(b);
    w.close();

    sam_hdr_t* hdr2 = nullptr;
    auto recs = read_bam(path, &hdr2);
    if (!recs.empty()) {
        uint8_t* xf_ptr = bam_aux_get(recs[0], "xf");
        CHECK(xf_ptr != nullptr, "xf tag present");
        if (xf_ptr) {
            CHECK(bam_aux2i(xf_ptr) == 1, "xf = 1 (GEX)");
        }
    }
    CHECK(recs.size() == 1u, "one record");
    free_records(recs);
    if (hdr2) sam_hdr_destroy(hdr2);
    sam_hdr_destroy(hdr);
    fs::remove(path);
}

// ── Test 15: null header — open returns false ───────────────────────────────
static void test_null_header() {
    std::cout << "Test 15: null header → open returns false\n";
    TaggedBamConfig cfg;
    cfg.output_path = tmp_path("null_hdr");
    TaggedBamWriter w;
    bool ok = w.open(cfg, nullptr);
    CHECK(!ok, "open returns false for null header");
    CHECK(!w.is_open(), "not open after null-header open");
}

// ── main ────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== test_tagged_bam ===\n";
    test_open_close();
    test_tag_presence();
    test_tag_values();
    test_cell_filter();
    test_stats();
    test_intergenic_read();
    test_multiple_records();
    test_region_types();
    test_bam_validity();
    test_coordinate_sorted_header();
    test_quality_tags();
    test_empty_barcode();
    test_no_filter();
    test_xf_tag();
    test_null_header();

    std::cout << "\n=== Results: " << n_pass << " passed, "
              << n_fail << " failed ===\n";
    return n_fail > 0 ? 1 : 0;
}
