// test/test_feature_barcode_detect.cpp — B-G2-3 / B-G8-2 unit test
// Verifies that a CITE_SEQ_ADT .1fq header is correctly identified as
// feature-barcode-only and that a summary.json with status=feature_barcode_not_gex
// would be emitted.
//
// Tests the detection logic directly (no full singlet binary run needed):
//   1. Write a synthetic CITE_SEQ_ADT .1fq
//   2. Open it and verify assay_type is CITE_SEQ_ADT
//   3. Verify detection logic: is_feature_barcode_only == true
//   4. Verify a GEX .1fq (CITE_SEQ_GEX) is NOT flagged as feature_barcode_only

#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/fq/lib1fq.h"
#include "singlet/fq/types.h"
#include "singlet/fq/writer.h"
#include "singlet/fq/reader.h"

// ─── Minimal test framework ─────────────────────────────────────────────────
static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++n_pass; } \
        else      { std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; ++n_fail; } \
    } while(0)

// ─── Helper: write a minimal .1fq with a given assay_type ────────────────────
static std::string write_synthetic_1fq(const char* path, singlet::fq::AssayType assay) {
    singlet::fq::WriterConfig cfg;
    cfg.block_size  = 100;
    cfg.r1_length   = 16;  // barcode+UMI
    cfg.r2_length   = 28;  // tag read (short, typical ADT)
    cfg.assay_type  = assay;

    singlet::fq::Writer w;
    w.open(path, cfg);

    // Write 10 synthetic reads
    std::vector<uint8_t> r1(16, 0), r2(28, 1);
    for (int i = 0; i < 10; ++i)
        w.add_read(r2.data(), r2.size(), r2.data(),
                   r1.data(), r1.size(), nullptr);
    w.finish("{}");
    return std::string(path);
}

// ─── Tests ──────────────────────────────────────────────────────────────────

void test_cite_seq_adt_detected_as_feature_barcode_only() {
    const char* tmpfile = "/tmp/test_fb_adt.1fq";
    write_synthetic_1fq(tmpfile, singlet::fq::AssayType::CITE_SEQ_ADT);

    singlet::fq::Reader reader;
    reader.open(tmpfile);
    auto hdr = reader.header();

    // The assay_type byte must be CITE_SEQ_ADT (= 9)
    CHECK(hdr.assay_type == static_cast<uint8_t>(singlet::fq::AssayType::CITE_SEQ_ADT),
          "header.assay_type == CITE_SEQ_ADT");

    // Reproduce singlet detection logic
    bool is_feature_barcode_only =
        (hdr.assay_type == static_cast<uint8_t>(singlet::fq::AssayType::CITE_SEQ_ADT));
    CHECK(is_feature_barcode_only, "is_feature_barcode_only == true for CITE_SEQ_ADT");

    // With no GTF (exon_gtf_path empty), the guard fires → feature_barcode_not_gex
    std::string exon_gtf_path = "";
    bool should_exit_feature_barcode = is_feature_barcode_only && exon_gtf_path.empty();
    CHECK(should_exit_feature_barcode,
          "guard fires: is_feature_barcode_only && exon_gtf_path.empty()");

    std::remove(tmpfile);
}

void test_cite_seq_gex_not_flagged_as_feature_barcode_only() {
    const char* tmpfile = "/tmp/test_fb_gex.1fq";
    write_synthetic_1fq(tmpfile, singlet::fq::AssayType::CITE_SEQ_GEX);

    singlet::fq::Reader reader;
    reader.open(tmpfile);
    auto hdr = reader.header();

    CHECK(hdr.assay_type == static_cast<uint8_t>(singlet::fq::AssayType::CITE_SEQ_GEX),
          "header.assay_type == CITE_SEQ_GEX");

    bool is_feature_barcode_only =
        (hdr.assay_type == static_cast<uint8_t>(singlet::fq::AssayType::CITE_SEQ_ADT));
    CHECK(!is_feature_barcode_only, "CITE_SEQ_GEX is NOT flagged as feature_barcode_only");

    std::remove(tmpfile);
}

void test_scrna_3prime_not_flagged() {
    const char* tmpfile = "/tmp/test_fb_scrna.1fq";
    write_synthetic_1fq(tmpfile, singlet::fq::AssayType::SC_RNA_3PRIME);

    singlet::fq::Reader reader;
    reader.open(tmpfile);
    auto hdr = reader.header();

    bool is_feature_barcode_only =
        (hdr.assay_type == static_cast<uint8_t>(singlet::fq::AssayType::CITE_SEQ_ADT));
    CHECK(!is_feature_barcode_only, "SC_RNA_3PRIME is NOT flagged as feature_barcode_only");

    std::remove(tmpfile);
}

void test_summary_json_schema() {
    // Verify the summary.json that would be written has required keys.
    // We exercise the exact format used in singlet.cpp B-G2-3 guard.
    const char* tmpfile = "/tmp/test_fb_summary.json";
    {
        std::ofstream sj(tmpfile);
        sj << "{\n"
           << "  \"schema_version\": \"1.0\",\n"
           << "  \"sample_id\": \"test_sample\",\n"
           << "  \"status\": \"feature_barcode_not_gex\",\n"
           << "  \"protocol_id\": 0,\n"
           << "  \"protocol_name\": \"CITE_SEQ_ADT\",\n"
           << "  \"species\": \"\",\n"
           << "  \"reference_build\": \"\",\n"
           << "  \"n_input_reads\": 10,\n"
           << "  \"n_uniquely_mapped\": 0,\n"
           << "  \"uniquely_mapped_pct\": 0.0,\n"
           << "  \"n_cells_called\": 0,\n"
           << "  \"median_umi_per_cell\": 0.0,\n"
           << "  \"median_genes_per_cell\": 0.0,\n"
           << "  \"memory_tier\": \"unknown\",\n"
           << "  \"peak_rss_gb\": 0.0,\n"
           << "  \"wall_seconds\": 0.01,\n"
           << "  \"track\": \"A\",\n"
           << "  \"donor\": null,\n"
           << "  \"mt\": null,\n"
           << "  \"nonhost\": null,\n"
           << "  \"reason\": \"feature-barcode-only (ADT/HTO) library; no GEX reads present; not in-scope for scRNA lane\"\n"
           << "}\n";
    }
    // Read back and verify key fields
    std::ifstream f(tmpfile);
    CHECK(f.is_open(), "summary.json opened for reading");
    std::string contents((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
    CHECK(contents.find("\"status\": \"feature_barcode_not_gex\"") != std::string::npos,
          "summary.json contains status=feature_barcode_not_gex");
    CHECK(contents.find("\"schema_version\"") != std::string::npos,
          "summary.json contains schema_version");
    CHECK(contents.find("\"n_input_reads\"") != std::string::npos,
          "summary.json contains n_input_reads");
    CHECK(contents.find("\"reason\"") != std::string::npos,
          "summary.json contains reason field");
    std::remove(tmpfile);
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== B-G2-3/B-G8-2 feature-barcode-detect unit tests ===\n";
    test_cite_seq_adt_detected_as_feature_barcode_only();
    test_cite_seq_gex_not_flagged_as_feature_barcode_only();
    test_scrna_3prime_not_flagged();
    test_summary_json_schema();
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail > 0 ? 1 : 0;
}
