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

    // FLAG: proper pair | read2, rightmost mate (TLEN < 0) → skipped; the leftmost mate carries the fragment
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTT_READ",
                            0, 1000, -200, BAM_FPROPER_PAIR | BAM_FREAD2);
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

// Read1 on the reverse strand: the fragment comes from the leftmost mate (read2, TLEN > 0), never from read1's pos.
static void test_reverse_read1_uses_leftmost_mate() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);
    // read2 forward at 1000 (leftmost, TLEN +300), read1 reverse at 1250 (TLEN -300)
    bam1_t* r2 = make_record("BC:AACCGGTTAACCGGTT_P", 0, 1000, 300, BAM_FPROPER_PAIR | BAM_FREAD2 | BAM_FMREVERSE);
    bam1_t* r1 = make_record("BC:AACCGGTTAACCGGTT_P", 0, 1250, -300, BAM_FPROPER_PAIR | BAM_FREAD1 | BAM_FREVERSE);
    ex.process_record(r2, nullptr);
    ex.process_record(r1, nullptr);
    free_record(r2);
    free_record(r1);
    assert(ex.unique_fragments() == 1);
    assert(ex.fragments()[0].start == 1004);
    assert(ex.fragments()[0].end == 1295);
    std::printf("  test_reverse_read1_uses_leftmost_mate: PASS\n");
}

// Duplicates are counted in read_pairs (fragments.tsv.gz column 5).
static void test_duplicate_count() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    bc_map["CCCCCCCCCCCCCCCC"] = 1;
    ex.set_barcode_map(bc_map);
    const uint16_t f = BAM_FPROPER_PAIR | BAM_FREAD1;
    const char* q[] = {"BC:AACCGGTTAACCGGTT_a", "BC:AACCGGTTAACCGGTT_b", "BC:CCCCCCCCCCCCCCCC_c",
                       "BC:AACCGGTTAACCGGTT_d"};
    for (const char* n : q) {
        bam1_t* b = make_record(n, 0, 700, 250, f);
        ex.process_record(b, nullptr);
        free_record(b);
    }
    ex.finalise();
    assert(ex.unique_fragments() == 2);
    assert(ex.fragments()[0].barcode_idx == 0 && ex.fragments()[0].read_pairs == 3);
    assert(ex.fragments()[1].barcode_idx == 1 && ex.fragments()[1].read_pairs == 1);
    std::printf("  test_duplicate_count: PASS\n");
}

// Whitelist correction: reverse-complement 24 bp Multiome layout, exact and Hamming-1 reads, GEX translation.
static std::string revcomp(const std::string& s) {
    std::string r(s.rbegin(), s.rend());
    for (char& c : r) c = c == 'A' ? 'T' : c == 'C' ? 'G' : c == 'G' ? 'C' : 'A';
    return r;
}

static void test_corrector_layout_and_correction() {
    singlet::AtacBarcodeCorrector cor;
    const std::string w0 = "AAACAGCCAAACAACA", w1 = "TTTGTTGGTTTGTTGT", w2 = "ACGTACGTACGTACGT";
    cor.add_entry(w0, "GGGGGGGGGGGGGGGG");
    cor.add_entry(w1, "CCCCCCCCCCCCCCCC");
    cor.add_entry(w2, "");
    // raw read = revcomp(spacer + barcode): the barcode is the first 16 bases after reverse-complementing.
    auto raw = [](const std::string& bc) { return revcomp(bc + "TCAGTCAG"); };
    for (int i = 0; i < 100; ++i) cor.observe(raw(i % 2 ? w0 : w1));
    auto lay = cor.detect_layout();
    assert(lay.revcomp && lay.offset == 0 && lay.hit_rate == 1.0);
    for (int i = 0; i < 100; ++i) cor.count(raw(w0));
    cor.count(raw(w1));
    assert(cor.resolve(raw(w0)) == 0);
    std::string mm = w0; mm[5] = 'T';                  // one mismatch from w0 only
    assert(cor.resolve(raw(mm)) == 0);
    std::string nn = w0; nn[3] = 'N';                  // one N
    assert(cor.resolve(raw(nn)) == 0);
    std::string far = "GGGGGGGGGGGGGGGG";              // no neighbour
    assert(cor.resolve(raw(far)) == -1);
    assert(cor.output_barcode(0) == "GGGGGGGGGGGGGGGG");
    assert(cor.output_barcode(2) == w2);               // unpaired entry falls back to the ATAC barcode
    std::printf("  test_corrector_layout_and_correction: PASS\n");
}

// The extractor resolves barcodes through the corrector when one is set.
static void test_extractor_with_corrector() {
    singlet::AtacBarcodeCorrector cor;
    cor.add_entry("AACCGGTTAACCGGTT");
    cor.observe("AACCGGTTAACCGGTT");
    cor.detect_layout();
    cor.count("AACCGGTTAACCGGTT");
    singlet::ATACFragmentExtractor ex;
    ex.set_barcode_corrector(&cor);
    bam1_t* b = make_record("BC:AACCGGTTAACCGGTA_x", 0, 1000, 200, BAM_FPROPER_PAIR | BAM_FREAD1);
    ex.process_record(b, nullptr);
    free_record(b);
    assert(ex.unique_fragments() == 1 && ex.fragments()[0].barcode_idx == 0);
    std::printf("  test_extractor_with_corrector: PASS\n");
}

// A1 (NOT COMPILED when written): duplicates that are not adjacent in the stream are still merged.
static void test_dedup_unsorted_order() {
    singlet::ATACFragmentExtractor ex;
    std::unordered_map<std::string, uint32_t> bc_map;
    bc_map["AACCGGTTAACCGGTT"] = 0;
    ex.set_barcode_map(bc_map);
    uint16_t flag = BAM_FPROPER_PAIR | BAM_FREAD1;
    // same fragment (pos 500, TLEN 300) three times with other starts in between: the old per-start table kept all three
    bam1_t* r[5] = {make_record("BC:AACCGGTTAACCGGTT_a", 0, 500, 300, flag), make_record("BC:AACCGGTTAACCGGTT_b", 0, 900, 200, flag),
                    make_record("BC:AACCGGTTAACCGGTT_c", 0, 500, 300, flag), make_record("BC:AACCGGTTAACCGGTT_d", 0, 700, 250, flag),
                    make_record("BC:AACCGGTTAACCGGTT_e", 0, 500, 300, flag)};
    for (auto* b : r) { ex.process_record(b, nullptr); free_record(b); }
    ex.finalise();
    assert(ex.unique_fragments() == 3);
    assert(ex.duplicate_fragments() == 2);
    bool found = false;
    for (const auto& f : ex.fragments()) if (f.start == 504) { assert(f.read_pairs == 3); found = true; }
    assert(found);
    std::printf("  test_dedup_unsorted_order: PASS
");
}

// A2 (NOT COMPILED when written): the layout sample is every hash-sampled record, not the first N.
static void test_layout_sample_is_hash_sample_not_prefix() {
    singlet::AtacBarcodeCorrector cor;
    cor.add_entry("AACCGGTTAACCGGTT");
    for (int i = 0; i < 300000; ++i) cor.observe("AACCGGTTAACCGGTT");   // more than the old 200,000 cap
    auto lay = cor.detect_layout();
    assert(lay.hit_rate == 1.0 && lay.offset == 0 && !lay.revcomp);
    // the sampling hash is fixed by the QNAME
    assert(singlet::AtacBarcodeCorrector::qname_hash("BC:AACC_r1") == singlet::AtacBarcodeCorrector::qname_hash("BC:AACC_r1"));
    int in = 0;
    for (int i = 0; i < 6400; ++i) in += singlet::AtacBarcodeCorrector::is_layout_sample(("BC:AACC_r" + std::to_string(i)).c_str());
    assert(in > 40 && in < 200);   // about 1/64
    std::printf("  test_layout_sample_is_hash_sample_not_prefix: PASS
");
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
    test_reverse_read1_uses_leftmost_mate();
    test_duplicate_count();
    test_corrector_layout_and_correction();
    test_extractor_with_corrector();
    test_dedup_unsorted_order();
    test_layout_sample_is_hash_sample_not_prefix();
    std::printf("All tests PASSED\n");
    return 0;
}
