// test/test_mirna.cpp — Unit tests for G-MIRNA miRNA quantification
// Tests: parse_mirbase_gff3, detect_small_rna, MirnaCounter build/count/RPM.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/mirna.h"

using namespace singlet;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                               \
    do {                                                               \
        if (!(cond)) {                                                 \
            std::cerr << "FAIL [" << __LINE__ << "]: " << msg << "\n"; \
            ++g_fail;                                                  \
        } else {                                                       \
            ++g_pass;                                                  \
        }                                                              \
    } while (0)

// ── Minimal synthetic GFF3 ────────────────────────────────────────────────────

// Minimal synthetic GFF3 — use regular string literals so \t is a real tab.
static const char* MINI_GFF3 =
    "##gff-version 3\n"
    "##date 2022-10-01\n"
    "chr1\t.\tmiRNA_primary_transcript\t17369\t17436\t.\t-\t.\t"
    "ID=MI0022705;Alias=MI0022705;Name=hsa-mir-6859-1\n"
    "chr1\t.\tmiRNA\t17369\t17391\t.\t-\t.\t"
    "ID=MIMAT0027618;Alias=MIMAT0027618;Name=hsa-miR-6859-3p;"
    "Derives_from=MI0022705;Accession=MIMAT0027618\n"
    "chr1\t.\tmiRNA\t17409\t17430\t.\t-\t.\t"
    "ID=MIMAT0027619;Alias=MIMAT0027619;Name=hsa-miR-6859-5p;"
    "Derives_from=MI0022705;Accession=MIMAT0027619\n"
    "chrX\t.\tmiRNA\t100\t122\t.\t+\t.\t"
    "ID=MIMAT0000001;Alias=MIMAT0000001;Name=mmu-miR-1;Accession=MIMAT0000001\n";

// ── GFF3 parser tests ──────────────────────────────────────────────────────────

static void test_gff3_parse_all() {
    auto entries = parse_mirbase_gff3(MINI_GFF3, "");
    // Should find 3 miRNA lines (primary transcript skipped)
    CHECK(entries.size() == 3, "gff3: 3 miRNA entries (primary skipped)");
}

static void test_gff3_species_filter_hsa() {
    auto entries = parse_mirbase_gff3(MINI_GFF3, "hsa");
    CHECK(entries.size() == 2, "gff3: 2 hsa entries");
    CHECK(entries[0].name == "hsa-miR-6859-3p", "gff3: first entry name");
    CHECK(entries[0].accession == "MIMAT0027618", "gff3: first entry accession");
    CHECK(entries[0].mature_start == 17369, "gff3: start coordinate");
    CHECK(entries[0].mature_end == 17391, "gff3: end coordinate");
}

static void test_gff3_species_filter_mmu() {
    auto entries = parse_mirbase_gff3(MINI_GFF3, "mmu");
    CHECK(entries.size() == 1, "gff3: 1 mmu entry");
    CHECK(entries[0].name == "mmu-miR-1", "gff3: mmu name");
}

static void test_gff3_empty_content() {
    auto entries = parse_mirbase_gff3("", "hsa");
    CHECK(entries.empty(), "gff3: empty content → no entries");
}

// ── detect_small_rna tests ────────────────────────────────────────────────────

static void test_small_rna_detection_positive() {
    std::vector<int> short_reads(100, 22);  // all 22 bp — miRNA-like
    CHECK(detect_small_rna(short_reads, 100), "small_rna: short reads detected");
}

static void test_small_rna_detection_negative() {
    std::vector<int> long_reads(100, 90);  // all 90 bp — mRNA-like
    CHECK(!detect_small_rna(long_reads, 100), "small_rna: long reads not detected");
}

static void test_small_rna_empty() {
    std::vector<int> empty;
    CHECK(!detect_small_rna(empty, 0), "small_rna: empty vector → false");
}

static void test_small_rna_boundary() {
    std::vector<int> boundary_reads(10, 34);  // 34 < 35 → small RNA
    CHECK(detect_small_rna(boundary_reads, 10), "small_rna: 34bp boundary positive");
    std::vector<int> over_reads(10, 35);  // 35 == 35 → NOT small RNA (< strictly)
    CHECK(!detect_small_rna(over_reads, 10), "small_rna: 35bp boundary negative");
}

// ── MirnaCounter tests ────────────────────────────────────────────────────────

static MirnaEntry make_entry(const std::string& name, const std::string& acc,
                             const std::string& seq) {
    MirnaEntry e;
    e.name = name;
    e.accession = acc;
    e.sequence = seq;
    e.mature_start = 1;
    e.mature_end = static_cast<int>(seq.size());
    return e;
}

static void test_exact_match() {
    MirnaCounter ctr;
    // Sequence with clearly distinguishable 18-mer seed.
    std::string seq = "ACGTACGTACGTACGTACGTACGT";  // 24 bp
    ctr.build_index({make_entry("test-miR-1", "ACC001", seq)});
    auto hit = ctr.count_read(seq);
    CHECK(hit.has_value(), "exact_match: hit found");
    CHECK(*hit == "test-miR-1", "exact_match: correct name");
    CHECK(ctr.get_counts().at("test-miR-1") == 1, "exact_match: count = 1");
}

static void test_no_match() {
    MirnaCounter ctr;
    std::string seq = "ACGTACGTACGTACGTACGTACGT";
    ctr.build_index({make_entry("test-miR-1", "ACC001", seq)});
    // Completely different 24-mer
    auto hit = ctr.count_read("TTTTTTTTTTTTTTTTTTTTTTTT");
    CHECK(!hit.has_value(), "no_match: should not hit");
}

static void test_one_mismatch() {
    MirnaCounter ctr;
    std::string seq = "GCATGCATGCATGCATGCAT";  // 20 bp
    ctr.build_index({make_entry("miR-mm", "ACC002", seq)});
    // Change 1 base in the seed (position 3)
    std::string query = seq;
    query[3] = (query[3] == 'A') ? 'C' : 'A';
    auto hit = ctr.count_read(query);
    CHECK(hit.has_value(), "one_mismatch: 1-mismatch indexed");
}

static void test_rpm_calculation() {
    MirnaCounter ctr;
    std::string seqA = "AAAAAAAAAAAAAAAAAACCC";  // 21 bp seed
    std::string seqB = "CCCCCCCCCCCCCCCCCCGGG";  // 21 bp seed
    ctr.build_index({
        make_entry("miR-A", "A001", seqA),
        make_entry("miR-B", "B001", seqB),
    });
    // Count 3 reads for miR-A, 1 for miR-B
    for (int i = 0; i < 3; ++i) ctr.count_read(seqA);
    ctr.count_read(seqB);
    CHECK(ctr.total_counted() == 4, "rpm: total count = 4");
    // Write to temp file and verify it exists
    const char* tmpfile = "/tmp/test_mirna_counts.tsv";
    ctr.write_mirna_counts(tmpfile);
    FILE* f = std::fopen(tmpfile, "r");
    CHECK(f != nullptr, "rpm: output file created");
    if (f) std::fclose(f);
    std::remove(tmpfile);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_gff3_parse_all();
    test_gff3_species_filter_hsa();
    test_gff3_species_filter_mmu();
    test_gff3_empty_content();

    test_small_rna_detection_positive();
    test_small_rna_detection_negative();
    test_small_rna_empty();
    test_small_rna_boundary();

    test_exact_match();
    test_no_match();
    test_one_mismatch();
    test_rpm_calculation();

    std::cout << "mirna: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail > 0) ? 1 : 0;
}
