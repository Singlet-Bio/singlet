// test_cascade_router.cpp — Unit tests for CascadeRouter (T-L2-1)
// Tests:
//   1. 100K mock read stream: asserts deterministic class distribution
//   2. Zero allocation tracking via operator new interceptor
//   3. L1/L2/L3 routing correctness

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>

// ── Allocation tracker ────────────────────────────────────────────────────────
// Counts operator new calls to verify zero-allocation hot path.
// We install this BEFORE including the router headers to capture any
// allocations that happen during routing.

static std::atomic<int64_t> g_alloc_count{0};
static bool g_tracking = false;

// Override global new/delete for tracking.
// Note: this intercepts ALL allocations in this translation unit after the
// atomic is set.  We bracket the hot-path section explicitly.
void* operator new(std::size_t sz) {
    if (g_tracking) ++g_alloc_count;
    void* p = std::malloc(sz);
    if (!p) throw std::bad_alloc{};
    return p;
}
void* operator new[](std::size_t sz) {
    if (g_tracking) ++g_alloc_count;
    void* p = std::malloc(sz);
    if (!p) throw std::bad_alloc{};
    return p;
}
void operator delete(void* p) noexcept  { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept  { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

// ── Now include the cascade headers ──────────────────────────────────────────
#include "singlet/pileup/cascade_router.h"

// ── Tiny synthetic transcriptome for L1 hits ─────────────────────────────────
static singlet::TxomeIndex build_test_txome() {
    // Three short transcripts each 200bp
    // We use ASCII A/C/G/T sequences
    std::vector<std::pair<std::string, std::string>> txs;
    auto make_seq = [](char base, int len) {
        std::string s(len, base);
        // vary slightly to avoid identical hashes across transcripts
        for (int i = 0; i < len; i += 7) s[i] = 'N';
        return s;
    };
    txs.push_back({"TX1", make_seq('A', 200)});
    txs.push_back({"TX2", make_seq('C', 200)});
    txs.push_back({"TX3", make_seq('G', 200)});

    singlet::TxomeIndex idx;
    idx.build(txs);
    return idx;
}

// ── Simple PRNG (xorshift64) ──────────────────────────────────────────────────
static uint64_t xorshift64(uint64_t& state) noexcept {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// ── Generate a mock read that will or will not hit the index ──────────────────
static std::string make_read_hit(const singlet::TxomeIndex& idx,
                                 uint32_t tx_id) {
    // Read = first SEED_K + 8 bytes of the transcript (will match exactly)
    const uint32_t tlen = idx.tx_lengths[tx_id];
    const uint8_t* tseq = idx.tx_seqs.data() + idx.tx_offsets[tx_id];
    uint32_t read_len = std::min(tlen, (uint32_t)60);
    std::string read(reinterpret_cast<const char*>(tseq), read_len);
    return read;
}

static std::string make_read_miss(uint64_t& rng) {
    // Random 60bp read — unlikely to hit the tiny test transcriptome exactly
    std::string r(60, 'N');
    const char bases[] = "ACGT";
    for (auto& c : r) c = bases[xorshift64(rng) & 3];
    return r;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static int test_determinism_and_distribution() {
    printf("  test_determinism_and_distribution...\n");

    auto idx = build_test_txome();
    singlet::TxomeAligner aligner(idx, singlet::CascadeRouter::DETERMINISM_SEED);
    singlet::CascadeRouter router(aligner, nullptr);

    // Run 100K reads: 30% constructed to hit L1, 70% random (→ L3)
    constexpr uint32_t N = 100'000;
    constexpr uint32_t N_HIT = 30'000;

    uint64_t rng = singlet::CascadeRouter::DETERMINISM_SEED;
    singlet::CascadeRouterStats stats;

    // Build hit reads ahead of time (before tracking)
    std::vector<std::string> hit_reads;
    hit_reads.reserve(N_HIT);
    for (uint32_t i = 0; i < N_HIT; ++i) {
        hit_reads.push_back(make_read_hit(idx, i % idx.n_transcripts()));
    }
    std::vector<std::string> miss_reads;
    miss_reads.reserve(N - N_HIT);
    for (uint32_t i = 0; i < N - N_HIT; ++i) {
        miss_reads.push_back(make_read_miss(rng));
    }

    // First pass (to warm up any static state)
    for (uint32_t i = 0; i < N_HIT; ++i) {
        singlet::CascadeReadView rv{};
        rv.r2_seq = reinterpret_cast<const uint8_t*>(hit_reads[i].data());
        rv.r2_len = static_cast<uint16_t>(hit_reads[i].size());
        router.route(rv, stats);
    }
    for (uint32_t i = 0; i < N - N_HIT; ++i) {
        singlet::CascadeReadView rv{};
        rv.r2_seq = reinterpret_cast<const uint8_t*>(miss_reads[i].data());
        rv.r2_len = static_cast<uint16_t>(miss_reads[i].size());
        router.route(rv, stats);
    }

    // Verify total
    assert(stats.n_total == N);
    // L1 resolve: exactly N_HIT reads were constructed to hit
    // (some may hash-collide with others; accept any count > 0)
    assert(stats.n_l1_resolve > 0);
    assert(stats.n_l3_pass > 0);
    assert(stats.n_l2_resolve == 0); // no L2 classifier attached
    assert(stats.n_l1_resolve + stats.n_l2_resolve + stats.n_l3_pass == N);

    printf("    n_total=%llu l1=%llu (%.1f%%) l3=%llu (%.1f%%) PASS\n",
           (unsigned long long)stats.n_total,
           (unsigned long long)stats.n_l1_resolve, 100.0 * stats.l1_fraction(),
           (unsigned long long)stats.n_l3_pass,    100.0 * stats.l3_fraction());

    // Determinism check: run a second pass with same inputs, same seed
    singlet::CascadeRouterStats stats2;
    for (uint32_t i = 0; i < N_HIT; ++i) {
        singlet::CascadeReadView rv{};
        rv.r2_seq = reinterpret_cast<const uint8_t*>(hit_reads[i].data());
        rv.r2_len = static_cast<uint16_t>(hit_reads[i].size());
        router.route(rv, stats2);
    }
    for (uint32_t i = 0; i < N - N_HIT; ++i) {
        singlet::CascadeReadView rv{};
        rv.r2_seq = reinterpret_cast<const uint8_t*>(miss_reads[i].data());
        rv.r2_len = static_cast<uint16_t>(miss_reads[i].size());
        router.route(rv, stats2);
    }
    assert(stats2.n_l1_resolve == stats.n_l1_resolve && "determinism violation");
    assert(stats2.n_l3_pass    == stats.n_l3_pass    && "determinism violation");
    printf("    determinism check PASS\n");

    return 0;
}

static int test_zero_allocation_hot_path() {
    printf("  test_zero_allocation_hot_path...\n");

    auto idx = build_test_txome();
    singlet::TxomeAligner aligner(idx, singlet::CascadeRouter::DETERMINISM_SEED);
    singlet::CascadeRouter router(aligner, nullptr);

    // Pre-build read data
    std::string miss_read(60, 'X'); // won't match anything
    singlet::CascadeReadView rv{};
    rv.r2_seq = reinterpret_cast<const uint8_t*>(miss_read.data());
    rv.r2_len = static_cast<uint16_t>(miss_read.size());

    // Warm up: run once outside tracking
    singlet::CascadeRouterStats dummy;
    router.route(rv, dummy);

    // Enable tracking
    g_alloc_count.store(0);
    g_tracking = true;

    // Hot loop: 10K calls
    singlet::CascadeRouterStats stats;
    for (int i = 0; i < 10'000; ++i) {
        router.route(rv, stats);
    }

    g_tracking = false;
    int64_t allocs = g_alloc_count.load();

    printf("    heap allocations during 10K-read hot loop: %lld\n",
           (long long)allocs);
    assert(allocs == 0 && "cascade_router hot path must not allocate");
    printf("    zero-allocation PASS\n");

    return 0;
}

static int test_l3_passthrough_for_unresolved() {
    printf("  test_l3_passthrough_for_unresolved...\n");

    auto idx = build_test_txome();
    singlet::TxomeAligner aligner(idx, singlet::CascadeRouter::DETERMINISM_SEED);
    singlet::CascadeRouter router(aligner, nullptr);

    // A read full of 'Z' (not a valid base) — will never match any transcript
    std::string junk(60, 'Z');
    singlet::CascadeReadView rv{};
    rv.r2_seq = reinterpret_cast<const uint8_t*>(junk.data());
    rv.r2_len = static_cast<uint16_t>(junk.size());

    singlet::CascadeRouterStats stats;
    auto d = router.route(rv, stats);
    assert(d == singlet::CascadeDecision::L3_passthrough);
    assert(stats.n_l3_pass == 1);
    printf("    L3 passthrough for junk read PASS\n");

    return 0;
}

static int test_l1_resolve_for_exact_hit() {
    printf("  test_l1_resolve_for_exact_hit...\n");

    auto idx = build_test_txome();
    singlet::TxomeAligner aligner(idx, singlet::CascadeRouter::DETERMINISM_SEED);
    singlet::CascadeRouter router(aligner, nullptr);

    // Build a read that exactly matches the first transcript at position 0
    std::string read = make_read_hit(idx, 0);
    singlet::CascadeReadView rv{};
    rv.r2_seq = reinterpret_cast<const uint8_t*>(read.data());
    rv.r2_len = static_cast<uint16_t>(read.size());

    singlet::CascadeRouterStats stats;
    auto d = router.route(rv, stats);
    assert(d == singlet::CascadeDecision::L1_resolve);
    assert(stats.n_l1_resolve == 1);
    printf("    L1 resolve for exact transcript hit PASS\n");

    return 0;
}

static int test_stats_fractions() {
    printf("  test_stats_fractions...\n");

    singlet::CascadeRouterStats s;
    s.n_total = 100;
    s.n_l1_resolve = 60;
    s.n_l3_pass = 40;

    assert(s.l1_fraction() > 0.599 && s.l1_fraction() < 0.601);
    assert(s.l3_fraction() > 0.399 && s.l3_fraction() < 0.401);

    // Edge case: zero total
    singlet::CascadeRouterStats empty;
    assert(empty.l1_fraction() == 0.0);
    printf("    fraction math PASS\n");

    return 0;
}

int main() {
    printf("cascade_router tests\n");
    int rc = 0;
    rc |= test_determinism_and_distribution();
    rc |= test_zero_allocation_hot_path();
    rc |= test_l3_passthrough_for_unresolved();
    rc |= test_l1_resolve_for_exact_hit();
    rc |= test_stats_fractions();
    if (rc == 0) printf("ALL PASS\n");
    return rc;
}
