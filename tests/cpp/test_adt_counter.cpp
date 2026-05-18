// SPDX-License-Identifier: MIT
// test_adt_counter.cpp — unit tests for T2 ADT UMI deduplication + counting
// Tests: single observation, duplicate UMI dedup, multi-UMI counting,
//        multi-tag independence, cross-cell isolation, merge().

#include <cassert>
#include <iostream>

#include "singlet/pileup/adt_counter.h"

using namespace singlet;

static int g_passed = 0, g_failed = 0;

static void PASS(const char* name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// Helper: retrieve count for (tag, cell) from a finalized AdtCounter
static uint32_t get_count(const AdtCounter& ac, uint32_t tag, uint32_t cell) {
    auto csc = ac.counts().to_csc();
    for (int32_t j = csc.indptr[cell]; j < csc.indptr[cell + 1]; ++j) {
        if (static_cast<uint32_t>(csc.indices[j]) == tag)
            return static_cast<uint32_t>(csc.data[j]);
    }
    return 0;
}

// ── Test 1: single observation → count 1 ─────────────────────────────────
static void test_single_observation() {
    AdtCounter ac;
    ac.set_num_tags(2);
    ac.set_num_cells(5);
    ac.add(0, 0, "AAAAAAAAAAAA");
    ac.finalize();

    uint32_t c = get_count(ac, 0, 0);
    if (c == 1)
        PASS("single_observation count=1");
    else
        FAIL("single_observation count=1", "expected 1");
}

// ── Test 2: duplicate UMI → count stays 1 ────────────────────────────────
static void test_duplicate_umi_dedup() {
    AdtCounter ac;
    ac.set_num_tags(2);
    ac.set_num_cells(5);
    ac.add(1, 0, "TTTTTTTTTTTT");
    ac.add(1, 0, "TTTTTTTTTTTT");  // exact duplicate
    ac.add(1, 0, "TTTTTTTTTTTT");  // triplicate
    ac.finalize();

    uint32_t c = get_count(ac, 0, 1);
    if (c == 1)
        PASS("duplicate_umi_dedup count=1");
    else
        FAIL("duplicate_umi_dedup count=1", "expected 1");

    if (ac.total_reads() == 3)
        PASS("duplicate_umi_dedup total_reads=3");
    else
        FAIL("duplicate_umi_dedup total_reads=3", "wrong");
}

// ── Test 3: two distinct UMIs → count 2 ──────────────────────────────────
static void test_two_distinct_umis() {
    AdtCounter ac;
    ac.set_num_tags(3);
    ac.set_num_cells(10);
    ac.add(2, 1, "AAACCCGGGTTT");
    ac.add(2, 1, "TTTGGGCCCAAA");  // different UMI
    ac.finalize();

    uint32_t c = get_count(ac, 1, 2);
    if (c == 2)
        PASS("two_distinct_umis count=2");
    else
        FAIL("two_distinct_umis count=2", "expected 2");
}

// ── Test 4: different tags are independent ────────────────────────────────
static void test_tag_independence() {
    AdtCounter ac;
    ac.set_num_tags(4);
    ac.set_num_cells(5);
    // same cell, same UMI but different tags → should each count as 1
    ac.add(0, 0, "AAACCCGGGTTT");
    ac.add(0, 1, "AAACCCGGGTTT");
    ac.add(0, 2, "AAACCCGGGTTT");
    ac.finalize();

    bool ok = (get_count(ac, 0, 0) == 1 &&
               get_count(ac, 1, 0) == 1 &&
               get_count(ac, 2, 0) == 1);
    if (ok)
        PASS("tag_independence");
    else
        FAIL("tag_independence", "counts wrong");
}

// ── Test 5: different cells are independent ───────────────────────────────
static void test_cell_independence() {
    AdtCounter ac;
    ac.set_num_tags(2);
    ac.set_num_cells(10);
    // same tag, same UMI, different cells → each cell gets count 1
    ac.add(0, 0, "GGGGGGGGGGGG");
    ac.add(1, 0, "GGGGGGGGGGGG");
    ac.add(2, 0, "GGGGGGGGGGGG");
    ac.finalize();

    bool ok = (get_count(ac, 0, 0) == 1 &&
               get_count(ac, 0, 1) == 1 &&
               get_count(ac, 0, 2) == 1);
    if (ok)
        PASS("cell_independence");
    else
        FAIL("cell_independence", "counts wrong");
}

// ── Test 6: merge two shards ──────────────────────────────────────────────
static void test_merge() {
    AdtCounter shard1, shard2;
    shard1.set_num_tags(2);
    shard1.set_num_cells(5);
    shard2.set_num_tags(2);
    shard2.set_num_cells(5);

    shard1.add(0, 0, "AAAAAAAAAAAA");
    shard1.add(0, 0, "CCCCCCCCCCCC");
    shard2.add(0, 0, "CCCCCCCCCCCC");  // duplicate across shards
    shard2.add(0, 0, "GGGGGGGGGGGG");  // new UMI in shard2

    AdtCounter merged;
    merged.set_num_tags(2);
    merged.set_num_cells(5);
    merged.merge(shard1);
    merged.merge(shard2);
    merged.finalize();

    // 3 unique UMIs: AAAA, CCCC, GGGG
    uint32_t c = get_count(merged, 0, 0);
    if (c == 3)
        PASS("merge count=3");
    else
        FAIL("merge count=3", "expected 3");
}

// ── Test 7: no_match tag_idx=-1 is ignored ────────────────────────────────
static void test_no_match_ignored() {
    AdtCounter ac;
    ac.set_num_tags(2);
    ac.set_num_cells(5);
    ac.add(0, -1, "AAAAAAAAAAAA");  // tag_idx < 0 → should be dropped
    ac.finalize();

    bool ok = (ac.total_reads() == 0);
    if (ok)
        PASS("no_match_ignored");
    else
        FAIL("no_match_ignored", "total_reads should be 0");
}

int main() {
    test_single_observation();
    test_duplicate_umi_dedup();
    test_two_distinct_umis();
    test_tag_independence();
    test_cell_independence();
    test_merge();
    test_no_match_ignored();

    std::cerr << "\n"
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed ? 1 : 0;
}
