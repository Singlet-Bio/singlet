// test_combinatorial_barcode.cpp — Unit tests for G-SHARESEQ combinatorial barcode decoder
// Tests SHARE-seq 3-level structure, linker matching, mismatch tolerance, and UMI extraction.

#include <cassert>
#include <iostream>
#include <string>

#include "singlet-pileup/combinatorial_barcode.h"

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Build a synthetic SHARE-seq R1 read:
//   [BC1:8][Linker1:22+8pad][BC2:8][Linker2:22+8pad][BC3:8][UMI:10]
// Total: 8+30+8+30+8+10 = 94bp
static std::string make_shareseq_r1(
    const std::string& bc1,                              // 8bp
    const std::string& bc2,                              // 8bp
    const std::string& bc3,                              // 8bp
    const std::string& umi,                              // 10bp
    const std::string& lnk1 = "ATCCACGTGCTTGAGACTGTGG",  // 22bp
    const std::string& lnk2 = "GTGGCCGATGTTTCGCATCGGC"   // 22bp
) {
    // Pad linkers to 30bp with fixed sequence
    std::string pad8(8, 'N');
    return bc1 + lnk1 + pad8 + bc2 + lnk2 + pad8 + bc3 + umi;
}

// ── Test 1: SHARE-seq correct decode ─────────────────────────────────────────
static void test_shareseq_correct() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    std::string bc1 = "ACGTACGT";
    std::string bc2 = "TTGGCCAA";
    std::string bc3 = "GGATCCTA";
    std::string umi = "AAACCCGGGT";

    std::string r1 = make_shareseq_r1(bc1, bc2, bc3, umi);
    assert(r1.size() == 94);

    auto res = dec.decode(r1);
    assert(res.valid);
    assert(res.barcode == bc1 + bc2 + bc3);
    assert(res.umi == umi);
    assert(res.linker_mismatches == 0);

    std::cout << "PASS test_shareseq_correct\n";
}

// ── Test 2: Wrong linker causes invalid decode ────────────────────────────────
static void test_shareseq_wrong_linker() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    // Replace linker1 with random sequence (many mismatches)
    std::string r1 = make_shareseq_r1(
        "ACGTACGT", "TTGGCCAA", "GGATCCTA", "AAACCCGGGT",
        "GGGGGGGGGGGGGGGGGGGGGG",  // wrong linker1 (22bp)
        "GTGGCCGATGTTTCGCATCGGC"   // correct linker2
    );

    auto res = dec.decode(r1);
    assert(!res.valid);

    std::cout << "PASS test_shareseq_wrong_linker\n";
}

// ── Test 3: One mismatch in linker is tolerated ───────────────────────────────
static void test_shareseq_one_mismatch_ok() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    // Introduce 1 mismatch at position 0 of linker1
    std::string lnk1_mut = "GTCCACGTGCTTGAGACTGTGG";  // A→G at pos 0
    std::string r1 = make_shareseq_r1(
        "ACGTACGT", "TTGGCCAA", "GGATCCTA", "AAACCCGGGT",
        lnk1_mut,
        "GTGGCCGATGTTTCGCATCGGC");

    auto res = dec.decode(r1);
    assert(res.valid);
    assert(res.linker_mismatches == 1);

    std::cout << "PASS test_shareseq_one_mismatch_ok\n";
}

// ── Test 4: Two mismatches in linker exceed tolerance ─────────────────────────
static void test_shareseq_two_mismatches_fail() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    // 2 mismatches in linker1 (pos 0 and 1)
    std::string lnk1_2mm = "GTACGTGCTTGAGACTGTGGAA";  // positions 0,1,2 changed
    // Let's be precise: original = ATCCACGTGCTTGAGACTGTGG
    //                   mutated  = GTGCACGTGCTTGAGACTGTGG  (A→G at 0, T→G at 1)
    std::string lnk1_mut2 = "GTGCACGTGCTTGAGACTGTGG";
    std::string r1 = make_shareseq_r1(
        "ACGTACGT", "TTGGCCAA", "GGATCCTA", "AAACCCGGGT",
        lnk1_mut2,
        "GTGGCCGATGTTTCGCATCGGC");

    auto res = dec.decode(r1);
    assert(!res.valid);

    std::cout << "PASS test_shareseq_two_mismatches_fail\n";
}

// ── Test 5: Correct composite barcode concatenation ───────────────────────────
static void test_shareseq_composite_barcode() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    std::string bc1 = "AAAACCCC";
    std::string bc2 = "GGGGTTTT";
    std::string bc3 = "ACACACAC";
    std::string umi = "TTTTAAAACC";

    std::string r1 = make_shareseq_r1(bc1, bc2, bc3, umi);
    auto res = dec.decode(r1);

    assert(res.valid);
    assert(res.barcode.size() == 24);
    assert(res.barcode.substr(0, 8) == bc1);
    assert(res.barcode.substr(8, 8) == bc2);
    assert(res.barcode.substr(16, 8) == bc3);

    std::cout << "PASS test_shareseq_composite_barcode\n";
}

// ── Test 6: UMI extraction ────────────────────────────────────────────────────
static void test_shareseq_umi_extraction() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    std::string umi = "CGATCGATCG";
    std::string r1 = make_shareseq_r1("ACGTACGT", "TTGGCCAA", "GGATCCTA", umi);
    auto res = dec.decode(r1);

    assert(res.valid);
    assert(res.umi == umi);
    assert(res.umi.size() == 10);

    std::cout << "PASS test_shareseq_umi_extraction\n";
}

// ── Test 7: Read too short returns invalid ────────────────────────────────────
static void test_shareseq_short_read() {
    auto dec = CombinatorialBarcodeDecoder::share_seq();

    // Only 50bp — too short for the full SHARE-seq structure
    std::string r1(50, 'A');
    auto res = dec.decode(r1);
    assert(!res.valid);

    std::cout << "PASS test_shareseq_short_read\n";
}

// ── Test 8: Custom decoder (2-level) ─────────────────────────────────────────
static void test_custom_two_level() {
    CombinatorialBarcodeDecoder dec;
    dec.umi_offset = 20;
    dec.umi_length = 6;
    dec.max_linker_mismatches = 1;

    BarcodeLevel l1;
    l1.offset = 0;
    l1.length = 6;
    l1.linker = "ATCGATCG";  // 8bp linker
    l1.linker_offset = 6;
    dec.levels.push_back(l1);

    BarcodeLevel l2;
    l2.offset = 14;
    l2.length = 6;
    l2.linker = "";
    l2.linker_offset = 0;
    dec.levels.push_back(l2);

    // Build: BC1(6) + linker(8) + BC2(6) + UMI(6) = 26bp
    std::string r1 = "ACGTACATCGATCGTTGGCCAAACGT";
    assert(r1.size() == 26);

    auto res = dec.decode(r1);
    assert(res.valid);
    assert(res.barcode == "ACGTAC" + std::string("TTGGCC"));
    assert(res.umi == "AAACGT");
    assert(res.linker_mismatches == 0);

    std::cout << "PASS test_custom_two_level\n";
}

// ── Test 9: scifi-RNA-seq factory ─────────────────────────────────────────────
static void test_scifi_seq_factory() {
    auto dec = CombinatorialBarcodeDecoder::scifi_seq();

    // R1: [BC1:8][Linker:18+2pad][BC2:8][UMI:8]
    // scifi linker (18bp): TCTTCAGCGTTCCCGAGA
    // Offset: BC1=0, linker=8, BC2=28, UMI=36
    std::string bc1 = "ACGTACGT";
    std::string lnk = "TCTTCAGCGTTCCCGAGA" + std::string(2, 'N');  // 18 + 2 pad = 20bp
    std::string bc2 = "TTGGCCAA";
    std::string umi = "CGATCGAT";
    std::string r1 = bc1 + lnk + bc2 + umi;
    assert(r1.size() == 44);

    auto res = dec.decode(r1);
    assert(res.valid);
    assert(res.barcode == bc1 + bc2);
    assert(res.umi == umi);

    std::cout << "PASS test_scifi_seq_factory\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main() {
    test_shareseq_correct();
    test_shareseq_wrong_linker();
    test_shareseq_one_mismatch_ok();
    test_shareseq_two_mismatches_fail();
    test_shareseq_composite_barcode();
    test_shareseq_umi_extraction();
    test_shareseq_short_read();
    test_custom_two_level();
    test_scifi_seq_factory();

    std::cout << "All combinatorial_barcode tests passed.\n";
    return 0;
}
