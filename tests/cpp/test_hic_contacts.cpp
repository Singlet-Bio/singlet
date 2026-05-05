// tests/test_hic_contacts.cpp — Unit tests for G-HIC contact pair extraction
// Tests: classify_contact, compute_hic_qc, write_pairs, write_hic_qc
// No htslib dependency; all synthetic ContactPair data.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/hic_contacts.h"

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

// ---------------------------------------------------------------------------
// Helper: build a ContactPair
// ---------------------------------------------------------------------------
static ContactPair make_pair(const std::string& c1, uint32_t p1, bool s1,
                             const std::string& c2, uint32_t p2, bool s2,
                             uint32_t mq1 = 30, uint32_t mq2 = 30,
                             const std::string& bc = "") {
    ContactPair cp;
    cp.chrom1 = c1;
    cp.pos1 = p1;
    cp.strand1 = s1;
    cp.chrom2 = c2;
    cp.pos2 = p2;
    cp.strand2 = s2;
    cp.mapq1 = mq1;
    cp.mapq2 = mq2;
    cp.barcode = bc;
    return cp;
}

// ---------------------------------------------------------------------------
// Group 1: classify_contact — basic cases
// ---------------------------------------------------------------------------
static void test_classify_cis_short() {
    // Same chrom, 5 kb apart => CIS_SHORT
    auto cp = make_pair("chr1", 1000000, true, "chr1", 1005000, false);
    CHECK(classify_contact(cp) == ContactType::CIS_SHORT, "5kb same chrom should be CIS_SHORT");
}

static void test_classify_cis_long() {
    // Same chrom, 50 kb apart => CIS_LONG
    auto cp = make_pair("chr1", 1000000, true, "chr1", 1050000, false);
    CHECK(classify_contact(cp) == ContactType::CIS_LONG, "50kb same chrom should be CIS_LONG");
}

static void test_classify_cis_exactly_at_threshold() {
    // Exactly 20 kb => CIS_LONG (>= threshold)
    auto cp = make_pair("chr2", 500000, true, "chr2", 520000, false);
    CHECK(classify_contact(cp) == ContactType::CIS_LONG, "Exactly 20kb should be CIS_LONG");
}

static void test_classify_trans() {
    // Different chromosomes => TRANS
    auto cp = make_pair("chr1", 1000000, true, "chr2", 2000000, false);
    CHECK(classify_contact(cp) == ContactType::TRANS, "Different chroms should be TRANS");
}

static void test_classify_low_mapq_both() {
    // MAPQ below threshold => INVALID
    auto cp = make_pair("chr1", 1000000, true, "chr1", 1050000, false, 5, 5);
    CHECK(classify_contact(cp) == ContactType::INVALID, "Low MAPQ both should be INVALID");
}

static void test_classify_low_mapq_one() {
    // One end low MAPQ => INVALID
    auto cp = make_pair("chr1", 1000000, true, "chr1", 1050000, false, 30, 5);
    CHECK(classify_contact(cp) == ContactType::INVALID, "One end low MAPQ should be INVALID");
}

static void test_classify_mapq_at_threshold() {
    // MAPQ exactly at threshold (10) => valid
    auto cp = make_pair("chr1", 1000000, true, "chr1", 1050000, false, 10, 10);
    CHECK(classify_contact(cp) == ContactType::CIS_LONG, "MAPQ==10 should be valid");
}

static void test_classify_self_ligation() {
    // Same chrom, <1kb, outward facing (pos1 < pos2, strand1=-, strand2=+)
    auto cp = make_pair("chr1", 1000, false, "chr1", 1500, true);
    CHECK(classify_contact(cp) == ContactType::SELF_LIGATION, "Outward-facing near pair should be SELF_LIGATION");
}

static void test_classify_dangling_end() {
    // Same chrom, <1kb, inward facing (pos1 < pos2, strand1=+, strand2=-)
    auto cp = make_pair("chr1", 1000, true, "chr1", 1500, false);
    CHECK(classify_contact(cp) == ContactType::DANGLING_END, "Inward-facing near pair should be DANGLING_END");
}

static void test_classify_self_ligation_reversed() {
    // pos1 > pos2, outward = strand1=+, strand2=-
    auto cp = make_pair("chr1", 1500, true, "chr1", 1000, false);
    CHECK(classify_contact(cp) == ContactType::SELF_LIGATION, "Reversed outward-facing should be SELF_LIGATION");
}

static void test_classify_dangling_end_reversed() {
    // pos1 > pos2, inward = strand1=-, strand2=+
    auto cp = make_pair("chr1", 1500, false, "chr1", 1000, true);
    CHECK(classify_contact(cp) == ContactType::DANGLING_END, "Reversed inward-facing should be DANGLING_END");
}

static void test_classify_cis_short_not_artefact() {
    // Same chrom, 5 kb (>1kb threshold for artefacts) => CIS_SHORT, not artefact
    auto cp = make_pair("chr1", 1000, true, "chr1", 6000, false);
    CHECK(classify_contact(cp) == ContactType::CIS_SHORT, ">1kb same chrom low distance is CIS_SHORT not artefact");
}

static void test_classify_custom_threshold() {
    // Custom cis_threshold = 50000; 30kb pair should be CIS_SHORT
    auto cp = make_pair("chr1", 0, true, "chr1", 30000, false);
    CHECK(classify_contact(cp, 10, 50000) == ContactType::CIS_SHORT, "30kb with 50kb threshold should be CIS_SHORT");
    // 60kb pair should be CIS_LONG
    auto cp2 = make_pair("chr1", 0, true, "chr1", 60000, false);
    CHECK(classify_contact(cp2, 10, 50000) == ContactType::CIS_LONG, "60kb with 50kb threshold should be CIS_LONG");
}

// ---------------------------------------------------------------------------
// Group 2: compute_hic_qc — aggregate stats
// ---------------------------------------------------------------------------
static void test_qc_empty() {
    std::vector<ContactPair> empty;
    auto qc = compute_hic_qc(empty);
    CHECK(qc.total_pairs == 0, "Empty: total_pairs == 0");
    CHECK(qc.valid_pairs == 0, "Empty: valid_pairs == 0");
    CHECK(qc.valid_fraction == 0.0, "Empty: valid_fraction == 0");
}

static void test_qc_all_valid_cis_short() {
    std::vector<ContactPair> pairs;
    for (int i = 0; i < 5; ++i) {
        pairs.push_back(make_pair("chr1", 1000000 + i * 100, true, "chr1", 1000000 + i * 100 + 5000, false));
    }
    auto qc = compute_hic_qc(pairs);
    CHECK(qc.total_pairs == 5, "5 pairs total");
    CHECK(qc.valid_pairs == 5, "all valid");
    CHECK(qc.cis_short == 5, "all CIS_SHORT");
    CHECK(qc.cis_long == 0, "no CIS_LONG");
    CHECK(qc.trans == 0, "no trans");
    CHECK(std::abs(qc.valid_fraction - 1.0) < 1e-9, "valid_fraction == 1.0");
}

static void test_qc_mixed_contacts() {
    // 3 cis_long, 2 trans, 1 low_mapq, 1 self_ligation, 1 dangling_end
    std::vector<ContactPair> pairs;
    // cis_long x3
    pairs.push_back(make_pair("chr1", 0, true, "chr1", 50000, false));
    pairs.push_back(make_pair("chr2", 0, true, "chr2", 100000, false));
    pairs.push_back(make_pair("chr3", 0, true, "chr3", 200000, false));
    // trans x2
    pairs.push_back(make_pair("chr1", 0, true, "chr2", 0, false));
    pairs.push_back(make_pair("chr3", 0, true, "chr4", 0, false));
    // invalid (low mapq) x1
    pairs.push_back(make_pair("chr1", 0, true, "chr1", 50000, false, 5, 5));
    // self-ligation x1
    pairs.push_back(make_pair("chr1", 100, false, "chr1", 500, true));
    // dangling end x1
    pairs.push_back(make_pair("chr1", 100, true, "chr1", 500, false));

    auto qc = compute_hic_qc(pairs);
    CHECK(qc.total_pairs == 8, "total_pairs == 8");
    CHECK(qc.valid_pairs == 5, "valid_pairs == 5 (cis_long+trans)");
    CHECK(qc.cis_long == 3, "cis_long == 3");
    CHECK(qc.trans == 2, "trans == 2");
    CHECK(qc.low_mapq == 1, "low_mapq == 1");
    CHECK(qc.self_ligation == 1, "self_ligation == 1");
    CHECK(qc.dangling_end == 1, "dangling_end == 1");
    CHECK(std::abs(qc.valid_fraction - 5.0 / 8.0) < 1e-9, "valid_fraction == 5/8");
    // cis_trans_ratio: 3 cis / 2 trans = 1.5
    CHECK(std::abs(qc.cis_trans_ratio - 1.5) < 1e-9, "cis_trans_ratio == 1.5");
    // long_range_fraction: cis_long / (cis_long + cis_short) = 3 / 3 = 1.0
    CHECK(std::abs(qc.long_range_fraction - 1.0) < 1e-9, "long_range_fraction == 1.0");
}

static void test_qc_long_range_fraction() {
    // 2 cis_short, 8 cis_long => LRF = 8/10 = 0.8
    std::vector<ContactPair> pairs;
    for (int i = 0; i < 2; ++i)
        pairs.push_back(make_pair("chr1", 0, true, "chr1", 5000, false));
    for (int i = 0; i < 8; ++i)
        pairs.push_back(make_pair("chr1", 0, true, "chr1", 50000, false));
    auto qc = compute_hic_qc(pairs);
    CHECK(qc.cis_short == 2, "cis_short == 2");
    CHECK(qc.cis_long == 8, "cis_long == 8");
    CHECK(std::abs(qc.long_range_fraction - 0.8) < 1e-9, "long_range_fraction == 0.8");
}

static void test_qc_only_trans_cis_ratio() {
    // Only trans => cis_trans_ratio = 0/N... let's check: cis_total=0, trans>0 => ratio = 0.0
    std::vector<ContactPair> pairs;
    pairs.push_back(make_pair("chr1", 0, true, "chr2", 0, false));
    auto qc = compute_hic_qc(pairs);
    CHECK(qc.cis_trans_ratio == 0.0, "All trans: cis_trans_ratio == 0.0");
}

static void test_qc_only_cis_ratio_infinite() {
    // No trans => ratio = inf
    std::vector<ContactPair> pairs;
    pairs.push_back(make_pair("chr1", 0, true, "chr1", 50000, false));
    auto qc = compute_hic_qc(pairs);
    CHECK(std::isinf(qc.cis_trans_ratio), "No trans: cis_trans_ratio == inf");
}

static void test_qc_barcode_field_preserved() {
    // scHi-C barcode should not affect classification
    auto cp = make_pair("chr1", 0, true, "chr1", 50000, false, 30, 30, "ACGTACGT");
    auto qc = compute_hic_qc({cp});
    CHECK(qc.valid_pairs == 1, "scHi-C barcode pair is valid");
    CHECK(qc.cis_long == 1, "scHi-C barcode pair is CIS_LONG");
}

// ---------------------------------------------------------------------------
// Group 3: write_pairs — 4DN format output
// ---------------------------------------------------------------------------
static void test_write_pairs_basic() {
    std::vector<ContactPair> pairs;
    pairs.push_back(make_pair("chr1", 1000, true, "chr2", 2000, false));
    pairs.push_back(make_pair("chr3", 5000, false, "chr3", 55000, true));

    std::vector<std::string> chrom_names = {"chr1", "chr2", "chr3"};
    std::vector<uint32_t> chrom_sizes = {248956422, 242193529, 198295559};

    const std::string tmpfile = "/tmp/test_hic_pairs.pairs";
    bool ok = write_pairs(tmpfile, pairs, chrom_names, chrom_sizes);
    CHECK(ok, "write_pairs returns true");

    std::ifstream in(tmpfile);
    CHECK(in.is_open(), "output file opens");

    std::string line;
    bool found_header = false, found_chromsize = false, found_columns = false;
    int data_lines = 0;
    bool found_chr1 = false, found_chr3 = false;
    bool found_plus = false, found_minus = false;
    while (std::getline(in, line)) {
        if (line.find("## pairs format") != std::string::npos) found_header = true;
        if (line.find("#chromsize:") != std::string::npos) found_chromsize = true;
        if (line.find("#columns:") != std::string::npos) found_columns = true;
        if (!line.empty() && line[0] != '#') {
            ++data_lines;
            if (line.find("chr1") != std::string::npos) found_chr1 = true;
            if (line.find("chr3") != std::string::npos && line.find("55000") != std::string::npos) found_chr3 = true;
            if (line.find('+') != std::string::npos) found_plus = true;
            if (line.find('-') != std::string::npos) found_minus = true;
        }
    }
    CHECK(found_header, ".pairs has version header");
    CHECK(found_chromsize, ".pairs has chromsize lines");
    CHECK(found_columns, ".pairs has #columns line");
    CHECK(data_lines == 2, ".pairs has 2 data lines");
    CHECK(found_chr1, ".pairs contains chr1 entry");
    CHECK(found_chr3, ".pairs contains chr3 CIS_LONG entry");
    CHECK(found_plus, ".pairs contains '+' strand");
    CHECK(found_minus, ".pairs contains '-' strand");

    std::remove(tmpfile.c_str());
}

static void test_write_pairs_empty() {
    std::vector<ContactPair> pairs;
    std::vector<std::string> chrom_names;
    std::vector<uint32_t> chrom_sizes;
    const std::string tmpfile = "/tmp/test_hic_pairs_empty.pairs";
    bool ok = write_pairs(tmpfile, pairs, chrom_names, chrom_sizes);
    CHECK(ok, "write_pairs with empty input returns true");
    std::ifstream in(tmpfile);
    std::string line;
    int data_lines = 0;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') ++data_lines;
    }
    CHECK(data_lines == 0, "empty write has 0 data lines");
    std::remove(tmpfile.c_str());
}

// ---------------------------------------------------------------------------
// Group 4: write_hic_qc — JSON output
// ---------------------------------------------------------------------------
static void test_write_hic_qc_json() {
    HicQC qc;
    qc.total_pairs = 100;
    qc.valid_pairs = 80;
    qc.cis_short = 20;
    qc.cis_long = 40;
    qc.trans = 20;
    qc.self_ligation = 5;
    qc.dangling_end = 5;
    qc.low_mapq = 10;
    qc.duplicate = 3;
    qc.valid_fraction = 0.8;
    qc.cis_trans_ratio = 3.0;
    qc.long_range_fraction = 0.667;

    const std::string tmpfile = "/tmp/test_hic_qc.json";
    bool ok = write_hic_qc(tmpfile, qc);
    CHECK(ok, "write_hic_qc returns true");

    std::ifstream in(tmpfile);
    CHECK(in.is_open(), "JSON file opens");
    std::string contents((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    CHECK(contents.find("\"total_pairs\"") != std::string::npos, "JSON has total_pairs");
    CHECK(contents.find("\"valid_pairs\"") != std::string::npos, "JSON has valid_pairs");
    CHECK(contents.find("\"cis_trans_ratio\"") != std::string::npos, "JSON has cis_trans_ratio");
    CHECK(contents.find("\"long_range_fraction\"") != std::string::npos, "JSON has long_range_fraction");
    CHECK(contents.find("100") != std::string::npos, "JSON has total pair count");
    CHECK(contents.find('{') != std::string::npos && contents.find('}') != std::string::npos, "JSON is wrapped in braces");
    std::remove(tmpfile.c_str());
}

static void test_write_hic_qc_inf() {
    HicQC qc;
    qc.total_pairs = 1;
    qc.valid_pairs = 1;
    qc.cis_long = 1;
    qc.cis_trans_ratio = std::numeric_limits<double>::infinity();
    qc.long_range_fraction = 1.0;
    qc.valid_fraction = 1.0;

    const std::string tmpfile = "/tmp/test_hic_qc_inf.json";
    bool ok = write_hic_qc(tmpfile, qc);
    CHECK(ok, "write_hic_qc with inf returns true");
    std::ifstream in(tmpfile);
    std::string contents((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
    CHECK(contents.find("\"inf\"") != std::string::npos, "JSON serializes inf as string \"inf\"");
    std::remove(tmpfile.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    // Group 1: classify_contact
    test_classify_cis_short();
    test_classify_cis_long();
    test_classify_cis_exactly_at_threshold();
    test_classify_trans();
    test_classify_low_mapq_both();
    test_classify_low_mapq_one();
    test_classify_mapq_at_threshold();
    test_classify_self_ligation();
    test_classify_dangling_end();
    test_classify_self_ligation_reversed();
    test_classify_dangling_end_reversed();
    test_classify_cis_short_not_artefact();
    test_classify_custom_threshold();

    // Group 2: compute_hic_qc
    test_qc_empty();
    test_qc_all_valid_cis_short();
    test_qc_mixed_contacts();
    test_qc_long_range_fraction();
    test_qc_only_trans_cis_ratio();
    test_qc_only_cis_ratio_infinite();
    test_qc_barcode_field_preserved();

    // Group 3: write_pairs
    test_write_pairs_basic();
    test_write_pairs_empty();

    // Group 4: write_hic_qc
    test_write_hic_qc_json();
    test_write_hic_qc_inf();

    std::cout << "Results: " << g_pass << " passed, " << g_fail << " failed\n";
    return (g_fail == 0) ? 0 : 1;
}
