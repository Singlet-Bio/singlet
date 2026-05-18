// SPDX-License-Identifier: MIT
// test_interop.cpp — Unit tests for singlet/pileup/interop.h
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "singlet/pileup/interop.h"

using namespace singlet::pileup::interop;
namespace fs = std::filesystem;

#define SINGLET_TEST_HARNESS_TERSE
#include "test_harness.h"  // CHECK(cond) + g_pass / g_fail

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string tmp_dir() {
    std::string d = "/tmp/test_interop_" + std::to_string(std::rand());
    fs::create_directories(d);
    return d;
}

static void write_file(const std::string& path, const std::string& content) {
    std::ofstream f(path);
    f << content;
}

// ── 1. MTX directory validation ───────────────────────────────────────────────

static void test_mtx_valid() {
    std::string d = tmp_dir() + "/mtx_valid";
    generate_test_mtx(d, 10, 5, 20);
    auto r = validate_cellranger_mtx(d);
    CHECK(r.valid);
    CHECK(r.error.empty());
    fs::remove_all(d);
}

static void test_mtx_missing_dir() {
    auto r = validate_cellranger_mtx("/tmp/does_not_exist_xyz_123");
    CHECK(!r.valid);
    CHECK(!r.error.empty());
}

static void test_mtx_missing_matrix() {
    std::string d = tmp_dir() + "/mtx_no_matrix";
    fs::create_directories(d);
    write_file(d + "/features.tsv", "ENSG1\tGene1\tGene Expression\n");
    write_file(d + "/barcodes.tsv", "ACGT-1\n");
    auto r = validate_cellranger_mtx(d);
    CHECK(!r.valid);
    CHECK(r.error.find("matrix") != std::string::npos);
    fs::remove_all(d);
}

static void test_mtx_bad_header() {
    std::string d = tmp_dir() + "/mtx_bad_hdr";
    fs::create_directories(d);
    write_file(d + "/matrix.mtx", "NOT_MARKET_HEADER\n10 5 20\n");
    write_file(d + "/features.tsv", "ENSG1\tGene1\tGene Expression\n");
    write_file(d + "/barcodes.tsv", "ACGT-1\n");
    auto r = validate_cellranger_mtx(d);
    CHECK(!r.valid);
    fs::remove_all(d);
}

static void test_mtx_gz_accepted() {
    // .gz files should be accepted without content validation
    std::string d = tmp_dir() + "/mtx_gz";
    fs::create_directories(d);
    write_file(d + "/matrix.mtx.gz", "pretend_gzip");
    write_file(d + "/features.tsv.gz", "pretend_gzip");
    write_file(d + "/barcodes.tsv.gz", "pretend_gzip");
    auto r = validate_cellranger_mtx(d);
    CHECK(r.valid);
    fs::remove_all(d);
}

// ── 2. Fragments.tsv validation ──────────────────────────────────────────────

static void test_fragments_valid() {
    std::string p = "/tmp/test_interop_frags_valid.tsv";
    generate_test_fragments(p, 5);
    auto r = validate_fragments_tsv(p);
    CHECK(r.valid);
    fs::remove(p);
}

static void test_fragments_missing_file() {
    auto r = validate_fragments_tsv("/tmp/does_not_exist_frags.tsv");
    CHECK(!r.valid);
}

static void test_fragments_bad_coords() {
    std::string p = "/tmp/test_interop_frags_bad.tsv";
    write_file(p, "chr1\t500\t100\tACGT-1\t1\n");  // start > end
    auto r = validate_fragments_tsv(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_fragments_comment_skipped() {
    std::string p = "/tmp/test_interop_frags_comment.tsv";
    write_file(p, "# header comment\nchr1\t0\t100\tACGT-1\t1\n");
    auto r = validate_fragments_tsv(p);
    CHECK(r.valid);
    fs::remove(p);
}

// ── 3. .pairs validation ─────────────────────────────────────────────────────

static void test_pairs_valid() {
    std::string p = "/tmp/test_interop_pairs_valid.pairs";
    generate_test_pairs(p, 3);
    auto r = validate_4dn_pairs(p);
    CHECK(r.valid);
    fs::remove(p);
}

static void test_pairs_missing_header() {
    std::string p = "/tmp/test_interop_pairs_bad.pairs";
    write_file(p, "readID\tchr1\t100\tchr1\t5000\t+\t-\n");
    auto r = validate_4dn_pairs(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_pairs_missing_file() {
    auto r = validate_4dn_pairs("/tmp/does_not_exist.pairs");
    CHECK(!r.valid);
}

// ── 4. Bismark BED validation ─────────────────────────────────────────────────

static void test_bismark_valid() {
    std::string p = "/tmp/test_interop_bismark.bed";
    write_file(p, "chr1\t100\t101\t85.5\t17\t3\n");
    auto r = validate_bismark_bed(p);
    CHECK(r.valid);
    fs::remove(p);
}

static void test_bismark_bad_methylation() {
    std::string p = "/tmp/test_interop_bismark_bad.bed";
    write_file(p, "chr1\t100\t101\t150.0\t30\t0\n");  // meth > 100
    auto r = validate_bismark_bed(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_bismark_missing_file() {
    auto r = validate_bismark_bed("/tmp/does_not_exist.bed");
    CHECK(!r.valid);
}

// ── 5. metrics_summary.csv validation ────────────────────────────────────────

static void test_metrics_valid() {
    std::string p = "/tmp/test_interop_metrics.csv";
    write_file(p, "Estimated Number of Cells,Median Genes per Cell\n3000,2500\n");
    auto r = validate_metrics_csv(p);
    CHECK(r.valid);
    fs::remove(p);
}

static void test_metrics_missing_field() {
    std::string p = "/tmp/test_interop_metrics_bad.csv";
    write_file(p, "Total Reads,Mean Reads per Cell\n100000,33.3\n");
    auto r = validate_metrics_csv(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_metrics_missing_file() {
    auto r = validate_metrics_csv("/tmp/does_not_exist_metrics.csv");
    CHECK(!r.valid);
}

// ── 6. VCF validation ─────────────────────────────────────────────────────────

static void test_vcf_valid() {
    std::string p = "/tmp/test_interop_valid.vcf";
    write_file(p,
               "##fileformat=VCFv4.2\n"
               "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Depth\">\n"
               "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
               "chr1\t100\t.\tA\tT\t30\tPASS\t.\n");
    auto r = validate_vcf(p);
    CHECK(r.valid);
    fs::remove(p);
}

static void test_vcf_bad_header() {
    std::string p = "/tmp/test_interop_bad.vcf";
    write_file(p, "# not a vcf\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n");
    auto r = validate_vcf(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_vcf_missing_chrom() {
    std::string p = "/tmp/test_interop_nochrom.vcf";
    write_file(p, "##fileformat=VCFv4.2\nchr1\t100\t.\tA\tT\t30\tPASS\t.\n");
    auto r = validate_vcf(p);
    CHECK(!r.valid);
    fs::remove(p);
}

static void test_vcf_missing_file() {
    auto r = validate_vcf("/tmp/does_not_exist.vcf");
    CHECK(!r.valid);
}

// ── 7. Generator round-trips ──────────────────────────────────────────────────

static void test_generators_roundtrip() {
    std::string mtx_d = "/tmp/test_interop_gen_mtx";
    generate_test_mtx(mtx_d, 5, 3, 10);
    CHECK(validate_cellranger_mtx(mtx_d).valid);
    fs::remove_all(mtx_d);

    std::string frags_p = "/tmp/test_interop_gen_frags.tsv";
    generate_test_fragments(frags_p, 10);
    CHECK(validate_fragments_tsv(frags_p).valid);
    fs::remove(frags_p);

    std::string pairs_p = "/tmp/test_interop_gen_pairs.pairs";
    generate_test_pairs(pairs_p, 5);
    CHECK(validate_4dn_pairs(pairs_p).valid);
    fs::remove(pairs_p);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_mtx_valid();
    test_mtx_missing_dir();
    test_mtx_missing_matrix();
    test_mtx_bad_header();
    test_mtx_gz_accepted();

    test_fragments_valid();
    test_fragments_missing_file();
    test_fragments_bad_coords();
    test_fragments_comment_skipped();

    test_pairs_valid();
    test_pairs_missing_header();
    test_pairs_missing_file();

    test_bismark_valid();
    test_bismark_bad_methylation();
    test_bismark_missing_file();

    test_metrics_valid();
    test_metrics_missing_field();
    test_metrics_missing_file();

    test_vcf_valid();
    test_vcf_bad_header();
    test_vcf_missing_chrom();
    test_vcf_missing_file();

    test_generators_roundtrip();

    std::cout << "interop: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
