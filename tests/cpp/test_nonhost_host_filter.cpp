// test_nonhost_host_filter.cpp — Unit tests for HostKmerFilter (NONHOST-HOST-SUBTRACT)
//
// Tests:
//   1. Basic insert/contains round-trip on small synthetic sequences
//   2. Save/load binary round-trip (file written to /tmp)
//   3. FPR estimate: random k-mer hashes not inserted → FPR within expected range
//   4. remove_host_minimizers: correctly partitions minimizer lists
//   5. Non-host minimizer threshold: read with all host minimizers → UNKNOWN in classify_multi_filtered
//   6. Genuine viral hit survives filtering when viral k-mers are non-host
//   7. Host-shared viral hit is suppressed when all viral minimizers are host-derived
//   8. count_host_minimizers: correct counts returned
//
// Build:
//   ninja -C build test_nonhost_host_filter && ./build/test_nonhost_host_filter

#include "singlet/pileup/nonhost/host_kmer_filter.h"
#include "singlet/pileup/nonhost/nonhost_screener.h"
#include "singlet/pileup/nonhost/min_sketch.h"
#include "singlet/pileup/nonhost/nonhost_em.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_set>
#include <cmath>

using namespace singlet::nonhost;

// ── Test helpers ─────────────────────────────────────────────────────────────

static int g_passed = 0, g_failed = 0;

static void PASS(const char* name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// Generate a non-repetitive sequence by cycling through the pattern.
static std::string make_seq(const char* pattern, int len) {
    std::string s;
    s.reserve(static_cast<size_t>(len));
    int plen = static_cast<int>(std::strlen(pattern));
    for (int i = 0; i < len; ++i) s += pattern[i % plen];
    return s;
}

// ── Test 1: Basic insert/contains ────────────────────────────────────────────

static void test1_insert_contains() {
    const char* name = "test1_insert_contains";

    HostKmerFilter flt(1000u, 21, 11);

    // Insert some canonical hashes directly
    std::vector<uint64_t> inserted = {
        12345678901234567ULL,
        99887766554433221ULL,
        11111111111111111ULL,
        0xDEADBEEFCAFEULL,
        0x0102030405060708ULL,
    };
    for (uint64_t h : inserted)
        flt.insert(h);

    for (uint64_t h : inserted) {
        if (!flt.contains(h)) {
            FAIL(name, "inserted hash not found");
            return;
        }
    }

    // Definitely absent (not inserted)
    // Use values unlikely to be false positives for such a small filter
    bool any_fp = false;
    for (uint64_t h : {0xABCDEF1234567890ULL, 0xFEDCBA0987654321ULL,
                       0x1234ABCD5678EFULL,   0xDEAD0000BEEFULL}) {
        if (flt.contains(h)) any_fp = true;  // technically possible but very unlikely at 1000 elems
    }
    (void)any_fp;  // not flagging FP in this tiny synthetic test

    PASS(name);
}

// ── Test 2: Save/load round-trip ─────────────────────────────────────────────

static void test2_save_load() {
    const char* name = "test2_save_load";

    HostKmerFilter flt(5000u, 21, 11);
    std::vector<uint64_t> keys;
    for (uint64_t i = 0; i < 200; ++i)
        keys.push_back(i * 1000003ULL + 7ULL);
    for (uint64_t h : keys) flt.insert(h);

    const std::string path = "/tmp/test_host_filter_roundtrip.bloom";
    flt.save(path);

    // exists() must return true
    if (!HostKmerFilter::exists(path)) {
        FAIL(name, "exists() returned false after save");
        return;
    }

    // Load and verify all inserted keys
    HostKmerFilter flt2 = HostKmerFilter::load(path);
    if (flt2.k() != 21 || flt2.w() != 11) {
        FAIL(name, "k or w mismatch after load");
        return;
    }
    if (flt2.n_bits() != flt.n_bits()) {
        FAIL(name, "n_bits mismatch after load");
        return;
    }

    bool ok = true;
    for (uint64_t h : keys) {
        if (!flt2.contains(h)) { ok = false; break; }
    }
    if (!ok) {
        FAIL(name, "inserted key not found after load");
        return;
    }

    std::remove(path.c_str());
    PASS(name);
}

// ── Test 3: FPR estimate ─────────────────────────────────────────────────────

static void test3_fpr_estimate() {
    const char* name = "test3_fpr_estimate";

    // Insert 10,000 known hashes; query 100,000 random NOT-inserted hashes.
    // Expected FPR ≈ 1% for 10 bits/element.
    const uint64_t N_INSERT  = 10000ULL;
    const uint64_t N_QUERY   = 100000ULL;
    const double   FPR_MAX   = 0.05;   // allow up to 5× headroom for the small dataset

    HostKmerFilter flt(N_INSERT, 21, 11);

    // Insert: use sequential keys from a "known" range
    for (uint64_t i = 0; i < N_INSERT; ++i)
        flt.insert(i * 1000003ULL + 1021ULL);

    // Query: use a completely different range (very unlikely to collide with above)
    uint64_t fp = 0;
    for (uint64_t i = 0; i < N_QUERY; ++i) {
        uint64_t h = (i + 500000ULL) * 999999999989ULL + 3ULL;
        if (flt.contains(h)) ++fp;
    }
    double fpr = static_cast<double>(fp) / static_cast<double>(N_QUERY);
    if (fpr > FPR_MAX) {
        std::cerr << "[" << name << "] FPR = " << fpr << " (max " << FPR_MAX << ")\n";
        FAIL(name, "FPR above 5x expected");
        return;
    }
    std::cerr << "[" << name << "] FPR = " << fpr * 100.0 << "% (expected ~1%, OK ≤5%)\n";
    PASS(name);
}

// ── Test 4: remove_host_minimizers ───────────────────────────────────────────

static void test4_remove_host_minimizers() {
    const char* name = "test4_remove_host_minimizers";

    HostKmerFilter flt(1000u, 21, 11);
    std::vector<uint64_t> host_hashes   = {100ULL, 200ULL, 300ULL, 400ULL};
    std::vector<uint64_t> nonhost_hashes = {500ULL, 600ULL, 700ULL};

    for (uint64_t h : host_hashes) flt.insert(h);

    std::vector<uint64_t> all;
    for (uint64_t h : host_hashes)   all.push_back(h);
    for (uint64_t h : nonhost_hashes) all.push_back(h);

    auto removed = flt.remove_host_minimizers(all);

    // All non-host hashes should remain; host hashes should be absent
    if (removed.size() != nonhost_hashes.size()) {
        FAIL(name, "wrong number of non-host minimizers after removal");
        return;
    }
    for (uint64_t h : removed) {
        bool was_host = false;
        for (uint64_t hf : host_hashes) if (h == hf) was_host = true;
        if (was_host) {
            FAIL(name, "host hash leaked through remove_host_minimizers");
            return;
        }
    }
    PASS(name);
}

// ── Test 5: count_host_minimizers ────────────────────────────────────────────

static void test5_count_host_minimizers() {
    const char* name = "test5_count_host_minimizers";

    HostKmerFilter flt(1000u, 21, 11);
    // Insert 6 hashes as "host"
    for (uint64_t h : {10ULL, 20ULL, 30ULL, 40ULL, 50ULL, 60ULL}) flt.insert(h);

    std::vector<uint64_t> mins = {10ULL, 20ULL, 30ULL, 99ULL, 88ULL, 77ULL};  // 3 host, 3 non-host
    auto [n_host, n_total] = flt.count_host_minimizers(mins);

    if (n_total != 6) {
        FAIL(name, "n_total wrong");
        return;
    }
    if (n_host != 3) {
        FAIL(name, "n_host wrong (expected 3)");
        return;
    }
    PASS(name);
}

// ── Test 6: classify_multi_filtered — all-host minimizer read → UNKNOWN ──────

static void test6_all_host_read_filtered() {
    const char* name = "test6_all_host_read_filtered";

    // k=21, w=11  → minimum sequence length is k+w-1 = 31
    // Build a viral index with 1 species, sequence that shares all minimizers with host.
    const int K = 21, W = 11;

    // Shared sequence used for both the host genome and the viral index.
    // By putting the SAME sequence in both the host filter and the viral index,
    // all of its minimizers will appear in both → classify_multi_filtered should
    // return UNKNOWN because all viral hits come from host-shared k-mers.
    const std::string shared_seq = make_seq("ACGTACGTCGATCGATCGAT", 100);

    // Build viral index from shared sequence
    MinSketchIndex viral_idx(K, W);
    viral_idx.add_sequence(shared_seq, /*species_id=*/1u, 50000u);
    viral_idx.finalize();

    // Build host filter containing ALL minimizers of shared_seq
    auto shared_mins = extract_minimizers(shared_seq, K, W);
    HostKmerFilter flt(static_cast<uint64_t>(shared_mins.size() * 2 + 100), K, W);
    for (uint64_t h : shared_mins) flt.insert(h);

    // Empty microbial index
    MinSketchIndex empty_idx(K, W);
    empty_idx.finalize();

    NonHostScreener screener(viral_idx, empty_idx, 0.05f);

    // Without host filter: should get viral hit
    auto mh_plain = screener.classify_multi(shared_seq, 0.25f);
    if (mh_plain.top_category != NonHostCategory::VIRAL || mh_plain.viral_hits.empty()) {
        FAIL(name, "without host filter, shared sequence should get viral hits");
        return;
    }

    // With host filter: all minimizers are host → not enough non-host mins → UNKNOWN
    auto mh_filtered = screener.classify_multi_filtered(shared_seq, flt, 0.25f);
    if (mh_filtered.top_category != NonHostCategory::UNKNOWN) {
        FAIL(name, "with host filter, all-host read should return UNKNOWN");
        return;
    }
    PASS(name);
}

// ── Test 7: classify_multi_filtered with empty filter == classify_multi ──────
// An empty Bloom filter (nothing inserted, FPR=0) should pass ALL minimizers
// through as non-host.  classify_multi_filtered must agree with classify_multi
// in this degenerate case.

static void test7_genuine_viral_passes() {
    const char* name = "test7_genuine_viral_passes";

    const int K = 21, W = 11;

    // Build a 400-bp viral sequence with a period > k=21 so the sequence
    // produces many distinct k-mers.  Period-23 ensures gcd(23,21)=1.
    const std::string viral_seq = make_seq("ACGTGCTAGCATCGATCGAATCG", 400);  // period=23

    MinSketchIndex viral_idx(K, W);
    viral_idx.add_sequence(viral_seq, /*species_id=*/42u, 50000u);
    viral_idx.finalize();

    // Empty host filter: nothing inserted -> FPR=0 -> all minimizers are "non-host"
    HostKmerFilter empty_flt(100u, K, W);  // nothing inserted

    MinSketchIndex empty_idx(K, W);
    empty_idx.finalize();
    NonHostScreener screener(viral_idx, empty_idx, 0.05f);

    // Without filter
    auto mh_plain = screener.classify_multi(viral_seq, 0.25f);
    // With empty filter (should be equivalent)
    auto mh_filtered = screener.classify_multi_filtered(viral_seq, empty_flt, 0.25f);

    if (mh_plain.viral_hits.empty()) {
        FAIL(name, "plain: viral_seq has no viral hits in its own index");
        return;
    }
    if (mh_filtered.viral_hits.empty()) {
        FAIL(name, "filtered (empty host): viral_seq lost all viral hits");
        return;
    }
    if (mh_filtered.top_category != NonHostCategory::VIRAL) {
        FAIL(name, "filtered (empty host): top_category not VIRAL");
        return;
    }
    // The hit rates should be the same (same non-host minimizer set)
    float rate_plain    = mh_plain.viral_hits.empty()    ? 0.f : mh_plain.viral_hits[0].second;
    float rate_filtered = mh_filtered.viral_hits.empty() ? 0.f : mh_filtered.viral_hits[0].second;
    if (std::abs(rate_plain - rate_filtered) > 0.01f) {
        std::cerr << "[" << name << "] rate_plain=" << rate_plain
                  << " rate_filtered=" << rate_filtered << "\n";
        FAIL(name, "hit rates unexpectedly differ with empty host filter");
        return;
    }
    PASS(name);
}

// ── Test 8: batch filtered matches per-read filtered ─────────────────────────

static void test8_batch_consistency() {
    const char* name = "test8_batch_consistency";

    const int K = 21, W = 11;
    const std::string host_seq  = make_seq("GGCTAGTCAG", 300);
    const std::string viral_seq = make_seq("TTTAAACGTCAGTTTTAAACGTCAGT", 300);
    const std::string host_seq2 = make_seq("CCGATCAGTATCCGATCAGTAT", 300);

    auto host_mins  = extract_minimizers(host_seq, K, W);
    auto host_mins2 = extract_minimizers(host_seq2, K, W);
    uint64_t filter_capacity = host_mins.size() + host_mins2.size() + 10000;
    HostKmerFilter flt(filter_capacity, K, W);
    for (uint64_t h : host_mins)  flt.insert(h);
    for (uint64_t h : host_mins2) flt.insert(h);

    MinSketchIndex viral_idx(K, W);
    viral_idx.add_sequence(viral_seq, /*species_id=*/7u, 50000u);
    viral_idx.finalize();

    MinSketchIndex empty_idx(K, W);
    empty_idx.finalize();
    NonHostScreener screener(viral_idx, empty_idx, 0.05f);

    std::vector<std::string> reads = {host_seq, viral_seq, host_seq2, viral_seq};

    // Per-read
    std::vector<NonHostScreener::MultiHit> per_read;
    for (const auto& r : reads)
        per_read.push_back(screener.classify_multi_filtered(r, flt, 0.25f));

    // Batch
    auto batch = screener.classify_multi_batch_filtered(reads, flt, 0.25f);

    if (batch.size() != per_read.size()) {
        FAIL(name, "batch size mismatch");
        return;
    }
    for (size_t i = 0; i < reads.size(); ++i) {
        if (batch[i].top_category != per_read[i].top_category) {
            FAIL(name, "batch vs per-read category mismatch");
            return;
        }
    }
    PASS(name);
}

// ── Test 9: exists() returns false for missing / corrupt paths ────────────────

static void test9_exists_checks() {
    const char* name = "test9_exists_checks";

    if (HostKmerFilter::exists("/nonexistent/path/filter.bloom")) {
        FAIL(name, "exists() returned true for missing file");
        return;
    }

    // Write a file with wrong magic
    const std::string path = "/tmp/test_bad_magic.bloom";
    {
        std::ofstream f(path, std::ios::binary);
        f.write("BADMAGIC", 8);
    }
    if (HostKmerFilter::exists(path)) {
        FAIL(name, "exists() returned true for wrong magic");
        std::remove(path.c_str());
        return;
    }
    std::remove(path.c_str());
    PASS(name);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test1_insert_contains();
    test2_save_load();
    test3_fpr_estimate();
    test4_remove_host_minimizers();
    test5_count_host_minimizers();
    test6_all_host_read_filtered();
    test7_genuine_viral_passes();
    test8_batch_consistency();
    test9_exists_checks();

    std::cerr << "\n";
    if (g_failed == 0) {
        std::cerr << "ALL " << g_passed << " tests PASSED\n";
        return 0;
    } else {
        std::cerr << g_passed << " passed, " << g_failed << " FAILED\n";
        return 1;
    }
}
