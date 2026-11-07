// SPDX-License-Identifier: MIT
// test_crispr_guide.cpp — unit tests for N18 CRISPR guide capture
// Tests GuideRef::load_csv, GuideRef::match, GuideCounter::count + UMI dedup
#include "singlet/pileup/crispr_guide.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string write_temp_csv(const char* content) {
    char path[] = "/tmp/test_guides_XXXXXX.csv";
    int fd = mkstemps(path, 4);
    if (fd < 0) { perror("mkstemps"); abort(); }
    FILE* f = fdopen(fd, "w");
    fputs(content, f);
    fclose(f);
    return path;
}

// ── Test 1: load CSV with header, exact match ─────────────────────────────

static void test_load_and_exact_match() {
    const char* csv =
        "name,sequence\n"
        "GUIDE_A,AGTCCGAATTAGGCGTACGA\n"
        "GUIDE_B,TACGGCAATCGGTACGCAAT\n"
        "GUIDE_C,CCGTATAGCAATCGGAATCG\n";
    auto path = write_temp_csv(csv);

    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded && "load_csv failed");
    assert(ref.size() == 3);

    // Exact match (guide as substring of a longer read)
    std::string r1 = "NNNNAGTCCGAATTAGGCGTACGANNNN";
    assert(ref.match(r1) == 0 && "GUIDE_A not matched");

    std::string r2 = "TACGGCAATCGGTACGCAAT";
    assert(ref.match(r2) == 1 && "GUIDE_B not matched");

    std::string r3 = "AAAAAAAAAAAAAAAAAAAAAA";
    assert(ref.match(r3) == -1 && "spurious match");

    std::cout << "Test 1 PASS: load_csv + exact substring match\n";
}

// ── Test 2: CSV without header, lowercase input ───────────────────────────
// NOTE: avoid "guide" / "name" / "sequence" in the first name field — those
// trigger the automatic header-detection heuristic in load_csv().

static void test_no_header_lowercase() {
    const char* csv =
        "Ctrl_X,GATTACAGATTACAGATTAC\n"
        "Ctrl_Y,CGTACGCAATCGGTACGGCA\n";
    auto path = write_temp_csv(csv);

    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);
    assert(ref.size() == 2);

    // Input lowercase — match() should uppercase before comparing
    std::string r = "gattacagattacagattac";
    assert(ref.match(r) == 0 && "lowercase guide not matched");

    std::cout << "Test 2 PASS: no-header CSV + lowercase input\n";
}

// ── Test 3: empty / invalid files ────────────────────────────────────────

static void test_empty_and_invalid() {
    GuideRef ref;
    bool r1 = ref.load_csv("/tmp/nonexistent_guide_file_12345.csv");
    (void)r1;
    assert(!r1);
    assert(ref.empty());

    auto path = write_temp_csv("name,sequence\n"); // header only, no data
    GuideRef ref2;
    bool ok = ref2.load_csv(path);
    (void)ok;
    // Either false (empty) or true with 0 guides — both acceptable
    assert(ref2.empty());
    std::remove(path.c_str());

    // match on empty ref must return -1
    assert(ref.match("AGTCCGAATTAGGCGTACGA") == -1);

    std::cout << "Test 3 PASS: empty/invalid file graceful\n";
}

// ── Test 4: GuideCounter with UMI dedup ──────────────────────────────────

static void test_guide_counter_umi_dedup() {
    const char* csv =
        "name,sequence\n"
        "G0,AAAAAAAAAAAAAAAAAAAAA\n"
        "G1,CCCCCCCCCCCCCCCCCCCCC\n";
    auto path = write_temp_csv(csv);

    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);

    std::vector<std::string> barcodes = {"BC0", "BC1", "BC2"};
    GuideCounter ctr;
    ctr.init(ref.size(), barcodes);

    const char* umi1 = "AAAAAAAAAA";
    const char* umi2 = "CCCCCCCCCC";

    // BC0, G0, UMI1 — should count
    ctr.count(0, 0, umi1, 10);
    // BC0, G0, UMI1 — duplicate → should NOT count
    ctr.count(0, 0, umi1, 10);
    // BC0, G0, UMI2 — different UMI → should count
    ctr.count(0, 0, umi2, 10);
    // BC1, G1, UMI1 → should count
    ctr.count(1, 1, umi1, 10);

    auto& acc = ctr.accumulator();
    assert(acc.nnz_raw() == 3 && "expected 3 unique (bc, guide, umi) entries");

    auto csc = acc.to_csc();
    // nrows = 2 guides, ncols = 3 barcodes
    assert(csc.nrows == 2);
    assert(csc.ncols == 3);

    // BC0 should have G0 count=2 (two unique UMIs)
    // BC1 should have G1 count=1
    bool found_bc0_g0 = false, found_bc1_g1 = false;
    for (int col = 0; col < (int)csc.ncols; ++col) {
        for (int p = csc.indptr[col]; p < csc.indptr[col + 1]; ++p) {
            int feat = csc.indices[p];
            int val  = csc.data[p];
            (void)val;
            if (col == 0 && feat == 0) { assert(val == 2); found_bc0_g0 = true; }
            if (col == 1 && feat == 1) { assert(val == 1); found_bc1_g1 = true; }
        }
    }
    assert(found_bc0_g0 && "BC0/G0 entry missing");
    (void)found_bc0_g0;
    (void)found_bc1_g1;
    assert(found_bc1_g1 && "BC1/G1 entry missing");

    std::cout << "Test 4 PASS: GuideCounter UMI dedup + CSC export\n";
}

// ── Test 5: no-UMI path (umi_len=0) ──────────────────────────────────────

static void test_no_umi() {
    GuideRef ref;
    const char* csv = "G0,TTTTTTTTTTTTTTTTTTTT\n";
    auto path = write_temp_csv(csv);
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);

    std::vector<std::string> bcs = {"X"};
    GuideCounter ctr;
    ctr.init(1, bcs);

    // Count 3× same guide with no UMI — all should go through
    for (int i = 0; i < 3; ++i)
        ctr.count(0, 0, nullptr, 0);

    auto csc = ctr.accumulator().to_csc();
    assert(csc.data[0] == 3 && "no-UMI path should count every observation");

    std::cout << "Test 5 PASS: no-UMI path counts all observations\n";
}

// ── Test 6: merge() for parallel accumulator merging ─────────────────────

static void test_merge() {
    const char* csv = "G0,GGGGGGGGGGGGGGGGGGGG\n";
    auto path = write_temp_csv(csv);

    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);

    std::vector<std::string> bcs = {"A", "B"};
    GuideCounter ctr_a, ctr_b;
    ctr_a.init(1, bcs);
    ctr_b.init(1, bcs);

    ctr_a.count(0, 0, nullptr, 0);
    ctr_b.count(1, 0, nullptr, 0);

    ctr_a.merge(ctr_b);
    auto csc = ctr_a.accumulator().to_csc();

    // Both BC0 and BC1 should have G0=1
    assert(csc.data.size() == 2);

    std::cout << "Test 6 PASS: merge() combines two worker accumulators\n";
}

// ── Test 7: guide names vector ────────────────────────────────────────────

static void test_guide_names() {
    const char* csv = "name,sequence\nAlpha,ACGT\nBeta,TGCA\n";
    auto path = write_temp_csv(csv);
    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);

    auto names = ref.names();
    assert(names.size() == 2);
    assert(names[0] == "Alpha");
    assert(names[1] == "Beta");

    std::cout << "Test 7 PASS: guide names() in order\n";
}

// -- Test 8: a read containing two guides is ambiguous, whatever the hash order --

static void test_ambiguous_guides() {
    const char* csv =
        "Long_1,GACGGATCGGGAGATCTCCC\n"
        "Inner_2,GGGAGATC\n"          // occurs inside Long_1
        "Other_3,TTTTGGGGCCCCAAAATTTT\n"
        "Dup_4,TTTTGGGGCCCCAAAATTTT\n"; // same sequence as Other_3
    auto path = write_temp_csv(csv);
    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded && ref.size() == 4);

    assert(ref.match("NNGACGGATCGGGAGATCTCCCNN") == GuideRef::kAmbiguous);  // Long_1 and Inner_2
    assert(ref.match("AAGGGAGATCAA") == 1);                                  // Inner_2 alone
    assert(ref.match("TTTTGGGGCCCCAAAATTTT") == GuideRef::kAmbiguous);       // duplicated sequence
    assert(ref.match("ACGTACGT") == -1);
    std::cout << "Test 8 PASS: ambiguous reads are never counted for a winner\n";
}

// -- Test 9: guides match in the read's sequencing orientation --

static void test_bam_orientation() {
    const char* csv = "G_1,GACGGATCGGGAGATCTCCC\nG_2,AAAACCCCGGGGTTTTAAAC\n";
    auto path = write_temp_csv(csv);
    GuideRef ref;
    bool loaded = ref.load_csv(path);
    (void)loaded;
    std::remove(path.c_str());
    assert(loaded);

    // A forward-strand record stores the read as sequenced.
    std::string fwd = "TTGACGGATCGGGAGATCTCCCTT";
    std::string b1 = fwd;
    assert(ref.match_bam(&b1[0], (int)b1.size(), false) == 0);

    // A reverse-strand record stores its reverse complement; match_bam restores the read orientation.
    std::string rev = fwd;
    GuideRef::reverse_complement(&rev[0], (int)rev.size());
    assert(rev != fwd);
    std::string b2 = rev;
    assert(ref.match(b2.data(), (int)b2.size()) == -1);                      // the raw BAM sequence does not match
    assert(ref.match_bam(&b2[0], (int)b2.size(), true) == 0);

    std::string x = "ACGTNacgt";
    GuideRef::reverse_complement(&x[0], (int)x.size());
    assert(x == "ACGTNACGT");
    std::cout << "Test 9 PASS: reverse-strand BAM records are matched in read orientation\n";
}

// ── main ─────────────────────────────────────────────────────────────────

int main() {
    test_load_and_exact_match();
    test_no_header_lowercase();
    test_empty_and_invalid();
    test_guide_counter_umi_dedup();
    test_no_umi();
    test_merge();
    test_guide_names();
    test_ambiguous_guides();
    test_bam_orientation();
    std::cout << "All 9 CRISPR guide tests passed.\n";
    return 0;
}
