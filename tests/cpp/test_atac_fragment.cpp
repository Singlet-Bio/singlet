// SPDX-License-Identifier: MIT
// test_atac_fragment.cpp — unit tests for atac_fragment.h
// Synthetic bam1_t construction, no BAM file on disk required.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

#include "singlet/pileup/atac_fragment.h"

static bam1_t* make_record(const char* qname,
                           int32_t tid, int32_t pos, int32_t isize,
                           uint16_t flag) {
    bam1_t* b = bam_init1();
    const int qname_len = static_cast<int>(std::strlen(qname)) + 1;  // include NUL
    const int seq_len = 0;                                           // no sequence needed for these tests
    const int data_len = qname_len + 1 /*NUL after qname null term*/;

    b->data = static_cast<uint8_t*>(std::calloc(static_cast<size_t>(data_len + 16), 1));
    b->l_data = data_len;
    b->m_data = static_cast<uint32_t>(data_len + 16);

    b->core.l_qname = static_cast<uint8_t>(qname_len);
    b->core.l_extranul = 0;
    b->core.n_cigar = 0;
    b->core.l_qseq = seq_len;

    b->core.tid = tid;
    b->core.pos = pos;
    b->core.isize = isize;
    b->core.flag = flag;
    b->core.qual = 60;
    b->core.mtid = tid;
    b->core.mpos = pos + std::abs(isize) - 50;

    std::memcpy(b->data, qname, static_cast<size_t>(qname_len));
    return b;
}

static void free_record(bam1_t* b) {
    std::free(b->data);
    b->data = nullptr;
    bam_destroy1(b);
}

// ---- Tests ----------------------------------------------------------------

static void test_basic_fragment() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    // Flag: proper pair | read1 = 0x2 | 0x40 = 0x42
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_READ001",
                            0, 1000, 200, BAM_FPROPER_PAIR | BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);

    // Expected: start = 1000+4=1004, end = 1000+200-5=1195
    assert(ex.unique_fragments() == 1);
    assert(ex.fragments()[0].start == 1004);
    assert(ex.fragments()[0].end == 1195);
    assert(ex.fragments()[0].barcode_idx == 0);
    std::printf("  test_basic_fragment: PASS\n");
}

static void test_deduplication() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    uint16_t flag = BAM_FPROPER_PAIR | BAM_FREAD1;
    bam1_t* b1 = make_record("BC:AACCGGTTAACCGGTT_R1", 0, 500, 300, flag);
    bam1_t* b2 = make_record("BC:AACCGGTTAACCGGTT_R2", 0, 500, 300, flag);  // duplicate
    ex.process_record(b1, nullptr);
    ex.process_record(b2, nullptr);
    free_record(b1);
    free_record(b2);

    assert(ex.unique_fragments() == 1);
    assert(ex.duplicate_fragments() == 1);
    std::printf("  test_deduplication: PASS\n");
}

static void test_read2_skipped() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    // FLAG: proper pair | read2 → should be skipped
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_READ",
                            0, 1000, 200, BAM_FPROPER_PAIR | BAM_FREAD2);
    ex.process_record(b, nullptr);
    free_record(b);

    assert(ex.unique_fragments() == 0);
    std::printf("  test_read2_skipped: PASS\n");
}

static void test_missing_barcode() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    // No BC: prefix
    bam1_t* b = make_record("PLAINREADNAME",
                            0, 1000, 200, BAM_FPROPER_PAIR | BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);

    assert(ex.unique_fragments() == 0);
    assert(ex.no_barcode_reads() == 1);
    std::printf("  test_missing_barcode: PASS\n");
}

static void test_all_n_barcode() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AAAA"] = 0;
    ex.set_barcode_map(bc_map);

    bam1_t* b = make_record("BC:NNNNNNNNNNNNNNNN_READ",
                            0, 1000, 200, BAM_FPROPER_PAIR | BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);

    assert(ex.unique_fragments() == 0);
    assert(ex.no_barcode_reads() == 1);
    std::printf("  test_all_n_barcode: PASS\n");
}

static void test_large_fragment_skipped() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);
    ex.set_max_fragment_size(2000);

    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_R",
                            0, 1000, 2500, BAM_FPROPER_PAIR | BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);

    assert(ex.unique_fragments() == 0);
    std::printf("  test_large_fragment_skipped: PASS\n");
}

static void test_not_proper_pair_skipped() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    // No BAM_FPROPER_PAIR bit
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_R",
                            0, 1000, 200, BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);

    assert(ex.unique_fragments() == 0);
    std::printf("  test_not_proper_pair_skipped: PASS\n");
}

static void test_sort_order() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AAAAAAAAAAAAAAAA"] = 0;
    bc_map["CCCCCCCCCCCCCCCC"] = 1;
    ex.set_barcode_map(bc_map);

    uint16_t flag = BAM_FPROPER_PAIR | BAM_FREAD1;
    // Insert in chrom/pos reverse order
    bam1_t* b1 = make_record("BC:AAAAAAAAAAAAAAAA_R1", 1, 5000, 200, flag);
    bam1_t* b2 = make_record("BC:CCCCCCCCCCCCCCCC_R2", 0, 100, 300, flag);
    bam1_t* b3 = make_record("BC:AAAAAAAAAAAAAAAA_R3", 0, 2000, 150, flag);
    ex.process_record(b1, nullptr);
    ex.process_record(b2, nullptr);
    ex.process_record(b3, nullptr);
    free_record(b1);
    free_record(b2);
    free_record(b3);

    ex.finalise();
    const auto& frags = ex.fragments();
    (void)frags;
    assert(frags.size() == 3);
    // Sorted by chrom_idx then start
    assert(frags[0].chrom_idx == 0 && frags[1].chrom_idx == 0);
    assert(frags[0].start < frags[1].start);
    assert(frags[2].chrom_idx == 1);
    std::printf("  test_sort_order: PASS\n");
}

static void test_low_mapq_filtered() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);

    uint16_t flag = BAM_FPROPER_PAIR | BAM_FREAD1;

    // MAPQ=29 → below default threshold of 30 → filtered
    bam1_t* b1 = make_record("BC:AACCGGTTAACCGGTT_LOW", 0, 1000, 200, flag);
    b1->core.qual = 29;
    ex.process_record(b1, nullptr);
    free_record(b1);
    assert(ex.unique_fragments() == 0);
    assert(ex.low_mapq_reads() == 1);

    // MAPQ=30 → passes
    bam1_t* b2 = make_record("BC:AACCGGTTAACCGGTT_OK", 0, 2000, 200, flag);
    b2->core.qual = 30;
    ex.process_record(b2, nullptr);
    free_record(b2);
    assert(ex.unique_fragments() == 1);

    std::printf("  test_low_mapq_filtered: PASS\n");
}

static void test_custom_mapq_threshold() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);
    ex.set_min_mapq(10);  // lower threshold

    uint16_t flag = BAM_FPROPER_PAIR | BAM_FREAD1;

    // MAPQ=10 → passes with custom threshold
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_R", 0, 1000, 200, flag);
    b->core.qual = 10;
    ex.process_record(b, nullptr);
    free_record(b);
    assert(ex.unique_fragments() == 1);

    std::printf("  test_custom_mapq_threshold: PASS\n");
}

int main() {
    std::printf("=== test_atac_fragment ===\n");
    test_basic_fragment();
    test_deduplication();
    test_read2_skipped();
    test_missing_barcode();
    test_all_n_barcode();
    test_large_fragment_skipped();
    test_not_proper_pair_skipped();
    test_sort_order();
    test_low_mapq_filtered();
    test_custom_mapq_threshold();
    std::printf("All tests PASSED\n");
    return 0;
}
