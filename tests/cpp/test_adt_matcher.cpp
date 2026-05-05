// test_adt_matcher.cpp — unit tests for T1 ADT antibody barcode matching
// Tests AdtMatcher: CSV loading, exact match, Hamming-1, no-match, ambiguous,
// multi-offset scanning, stats counters, and thread-safety (read-only phase).
#include <cassert>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "singlet/pileup/adt_matcher.h"

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string write_temp(const char* content, const char* suffix = ".csv") {
    std::string tmpl = std::string("/tmp/test_adt_XXXXXX") + suffix;
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkstemps(buf.data(), static_cast<int>(strlen(suffix)));
    if (fd < 0) {
        perror("mkstemps");
        abort();
    }
    FILE* f = fdopen(fd, "w");
    fputs(content, f);
    fclose(f);
    return std::string(buf.data());
}

// ── Test 1: CSV loading with header + exact match ─────────────────────────

static void test_csv_load_and_exact_match() {
    const char* csv =
        "name,sequence\n"
        "CD4,AACAAGACCCCTCCT\n"
        "CD8a,TGCATGTCATCGGTGG\n"
        "CD19,CTGGGCAATTCATGGC\n";
    auto path = write_temp(csv);

    AdtMatcher m;
    bool loaded = m.load_reference(path);
    std::remove(path.c_str());
    assert(loaded && "load_reference failed");
    assert(m.num_tags() == 3);

    m.build_index();

    // Exact match at offset 0
    assert(m.match("AACAAGACCCCTCCT") == 0 && "CD4 not matched");
    assert(m.match("TGCATGTCATCGGTGG") == 1 && "CD8a not matched");
    assert(m.match("CTGGGCAATTCATGGC") == 2 && "CD19 not matched");

    // Completely different sequence → no match
    assert(m.match("NNNNNNNNNNNNNNNN") == -1 && "spurious match");

    assert(m.tag_name(0) == "CD4");
    assert(m.tag_name(1) == "CD8a");
    assert(m.tag_name(2) == "CD19");

    std::cout << "Test 1 PASS: CSV load + exact match\n";
}

// ── Test 2: Hamming-1 match (single substitution) ─────────────────────────

static void test_hamming1_match() {
    AdtMatcher m;
    m.add_tag("TAG_A", "AACAAGACCCCTCCT");
    m.add_tag("TAG_B", "TGCATGTCATCGGTGG");
    m.build_index();

    // 1 substitution at position 0: A→C
    assert(m.match("CACAAGACCCCTCCT") == 0 && "Hamm-1 pos0 failed");

    // 1 substitution at position 7: A→G
    assert(m.match("AACAAGACGCCTCCT") == 0 && "Hamm-1 pos7 failed");

    // 1 substitution at last position
    assert(m.match("AACAAGACCCCTCCG") == 0 && "Hamm-1 last pos failed");

    // 2 substitutions → no match
    assert(m.match("CACAAGACCCCTCCG") == -1 && "2-sub should not match");

    std::cout << "Test 2 PASS: Hamming-1 substitution matching\n";
}

// ── Test 3: No match (too many mismatches) ────────────────────────────────

static void test_no_match() {
    AdtMatcher m;
    m.add_tag("ONLY", "AACAAGACCCCTCCT");
    m.build_index();

    // 2 mismatches
    assert(m.match("TTCAAGACCCCTCCT") == -1 && "2-mm should not match");

    // Completely random
    assert(m.match("GCGCGCGCGCGCGCG") == -1);

    // Read too short
    assert(m.match("AACAAG") == -1 && "short read should not match");

    // stats
    assert(m.unmatched() == 3);

    std::cout << "Test 3 PASS: no-match (mismatches, short read)\n";
}

// ── Test 4: Ambiguous (neighbour of two tags) ─────────────────────────────

static void test_ambiguous() {
    // TAG_A and TAG_B differ only at position 14 (last base).
    // Their Hamming-1 neighbourhoods overlap: e.g. position-14 substitution
    // of TAG_A and position-14 substitution of TAG_B can share a neighbour.
    // More directly: construct two tags that share a Hamming-1 neighbour.
    //   TAG_A = AACAAGACCCCTCCA
    //   TAG_B = AACAAGACCCCTCCG
    //   Shared Hamming-1 neighbour at pos 14: AACAAGACCCCTCCT
    //   (AACAAGACCCCTCCT is H1 from both A and G at pos14)
    AdtMatcher m;
    m.add_tag("TAG_A", "AACAAGACCCCTCCA");
    m.add_tag("TAG_B", "AACAAGACCCCTCCG");
    m.build_index();

    // AACAAGACCCCTCCT is H1 from TAG_A (A→T) and H1 from TAG_B (G→T): ambiguous
    int r = m.match("AACAAGACCCCTCCT");
    assert(r == -2 && "expected -2 (ambiguous)");
    assert(m.ambiguous() == 1);

    // Exact matches still resolve unambiguously
    assert(m.match("AACAAGACCCCTCCA") == 0);
    assert(m.match("AACAAGACCCCTCCG") == 1);

    std::cout << "Test 4 PASS: ambiguous neighbour → -2\n";
}

// ── Test 5: TSV loading + offset scanning ─────────────────────────────────

static void test_tsv_and_offset() {
    // TSV, no header; tag is at offset 3 in the read
    const char* tsv =
        "HTO_1\tAACAGATCTCGCTTG\n"
        "HTO_2\tTGGATATGCTACGAC\n";
    auto path = write_temp(tsv, ".tsv");

    AdtMatcher m;
    bool loaded = m.load_reference(path);
    std::remove(path.c_str());
    assert(loaded);
    assert(m.num_tags() == 2);
    m.build_index();

    // Tag starts at offset 3 (within default max_offset=5)
    std::string read = "NNNAAACAGATCTCGCTTGNNNNNN";
    // offset 3: "AACAGATCTCGCTTG" — wait, let me count:
    // read[3..17] = "AACAGATCTCGCTTG" — that's HTO_1
    assert(m.match(read) == 0 && "offset scan failed for HTO_1");

    // Offset 0 exact
    assert(m.match("TGGATTATGCTACGAC", 16, 0) == -1);  // 1 sub at pos5 but offset=0 only
    assert(m.match("TGGATATGCTACGAC") == 1 && "HTO_2 exact");

    std::cout << "Test 5 PASS: TSV load + offset scanning\n";
}

// ── Test 6: names() accessor ──────────────────────────────────────────────

static void test_names() {
    AdtMatcher m;
    m.add_tag("Hashtag1", "GTCAACTCTTTAGCG");
    m.add_tag("Hashtag2", "TGATGGCCTATTGGG");
    m.add_tag("Hashtag3", "TTCCGCCTCTCTTTG");
    m.build_index();

    auto names = m.names();
    assert(names.size() == 3);
    assert(names[0] == "Hashtag1");
    assert(names[1] == "Hashtag2");
    assert(names[2] == "Hashtag3");

    std::cout << "Test 6 PASS: names() in index order\n";
}

// ── Test 7: thread-safety — concurrent reads after build_index() ──────────

static void test_thread_safe_concurrent_match() {
    AdtMatcher m;
    m.add_tag("T0", "AACAAGACCCCTCCT");
    m.add_tag("T1", "TGCATGTCATCGGTGG");
    m.build_index();

    static constexpr int N_THREADS = 8;
    static constexpr int N_QUERIES = 10000;

    std::vector<std::thread> threads;
    std::atomic<int> errors{0};

    for (int t = 0; t < N_THREADS; ++t) {
        threads.emplace_back([&m, &errors]() {
            for (int i = 0; i < N_QUERIES; ++i) {
                int r0 = m.match("AACAAGACCCCTCCT");
                int r1 = m.match("TGCATGTCATCGGTGG");
                int rn = m.match("NNNNNNNNNNNNNNNN");
                if (r0 != 0 || r1 != 1 || rn != -1)
                    ++errors;
            }
        });
    }
    for (auto& th : threads) th.join();

    assert(errors.load() == 0 && "concurrent match() returned wrong results");
    std::cout << "Test 7 PASS: " << N_THREADS << " threads × "
              << N_QUERIES << " queries — no races\n";
}

// ── Test 8: empty reference graceful ──────────────────────────────────────

static void test_empty_ref() {
    AdtMatcher m;
    m.build_index();

    assert(m.match("AACAAGACCCCTCCT") == -1 && "empty ref should return -1");
    assert(m.num_tags() == 0);
    assert(m.empty());

    bool loaded = m.load_reference("/tmp/nonexistent_adt_ref_99999.csv");
    assert(!loaded && "nonexistent file should return false");

    std::cout << "Test 8 PASS: empty/missing reference graceful\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_csv_load_and_exact_match();
    test_hamming1_match();
    test_no_match();
    test_ambiguous();
    test_tsv_and_offset();
    test_names();
    test_thread_safe_concurrent_match();
    test_empty_ref();
    std::cout << "All 8 ADT matcher tests passed.\n";
    return 0;
}
