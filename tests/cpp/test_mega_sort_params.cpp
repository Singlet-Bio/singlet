// test_mega_sort_params.cpp — Unit tests for mega_sort_params.h
//
// Verifies the three-tier STAR BAM-sort parameter policy enforced for the
// AUTOFIX-MEGA-SORT-RSS-OVERAGE fix:
//
//   normal  (<=200M reads)            compression=0, cap=75% of SLURM_MEM
//   mega    (>200M, <500M)            compression=1, cap=50% of SLURM_MEM
//   ultra   (>=500M)                  compression=0, cap=15% + hard cap 64 GiB, bins=100
//   complex (CB_UMI_Complex)          compression=1, cap=15% + hard cap 64 GiB
//
// Witness samples driving the policy:
//   - GSM3743501 (408M, 10x-5p-v3): must stay in the mega tier (50%/comp=1).
//   - GSM7102845 (666M), GSM5239644 (817M): must land in ultra
//     (15%/comp=0/bins=100).
//
// Iteration #2 (2026-04-16): bins dropped 500 → 100 after GSM7102845 hit
// STAR's "bin size on disk != expected" pathology at bin #494. Compression
// is 0 (uncompressed) in the ultra tier — memory safety comes from the 15%
// /64-GiB limitBAMsortRAM cap plus bins=100, not from BGZF compression.

#include <cassert>
#include <cstdint>
#include <iostream>

#include "singlet-pileup/mega_sort_params.h"

using singlet_pileup::mega_sort::compression_level;
using singlet_pileup::mega_sort::ram_cap_bytes;
using singlet_pileup::mega_sort::tier_for;
using singlet_pileup::mega_sort::Tier;
using singlet_pileup::mega_sort::MEGA_READ_THRESHOLD;
using singlet_pileup::mega_sort::ULTRA_READ_THRESHOLD;
using singlet_pileup::mega_sort::ULTRA_SORT_BINS;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (cond) {                                                         \
            ++g_pass;                                                       \
        } else {                                                            \
            ++g_fail;                                                       \
            std::cerr << "FAIL: " << #cond                                  \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
        }                                                                   \
    } while (0)

static constexpr uint64_t GiB = 1ULL << 30;

// ── 1. Thresholds ────────────────────────────────────────────────────────────
// Sanity check: the constants are what we think they are.
static void test_thresholds() {
    CHECK(MEGA_READ_THRESHOLD  == 200'000'000ULL);
    CHECK(ULTRA_READ_THRESHOLD == 500'000'000ULL);
    CHECK(ULTRA_READ_THRESHOLD > MEGA_READ_THRESHOLD);

    // ULTRA_SORT_BINS: iteration #2 dropped this from 500 to 100 after
    // GSM7102845 hit STAR's bin-size-on-disk pathology at bin #494.
    // 100 is 2× STAR's default (50) and inside its tested range; NEVER
    // set this above ~200 without re-validating against a real 500M+ sample.
    CHECK(ULTRA_SORT_BINS == 100);
    CHECK(ULTRA_SORT_BINS > 50);   // strictly above STAR default
    CHECK(ULTRA_SORT_BINS < 200);  // inside safe range
}

// ── 2. compression_level ─────────────────────────────────────────────────────
static void test_compression_level() {
    // Normal small sample: uncompressed sort.
    CHECK(compression_level(50'000'000ULL, /*use_complex*/ false) == 0);
    CHECK(compression_level(5'000'000ULL,  /*use_complex*/ false) == 0);
    CHECK(compression_level(0ULL,          /*use_complex*/ false) == 0);

    // Exactly at 200M threshold: still normal (strict > 200M to enter mega).
    CHECK(compression_level(200'000'000ULL, /*use_complex*/ false) == 0);

    // Mega tier: 200M < n_reads < 500M.
    CHECK(compression_level(200'000'001ULL, /*use_complex*/ false) == 1);
    CHECK(compression_level(400'000'000ULL, /*use_complex*/ false) == 1);
    CHECK(compression_level(408'000'000ULL, /*use_complex*/ false) == 1);  // GSM3743501
    CHECK(compression_level(499'999'999ULL, /*use_complex*/ false) == 1);

    // Ultra tier: >= 500M.  compression=0 (uncompressed). Memory safety in
    // the ultra tier comes from the 15%/64-GiB limitBAMsortRAM cap plus
    // bins=100, NOT from BGZF compression. Iteration #2 established this
    // after GSM7102845 (666M) ran successfully through STAR alignment at
    // comp=0 with the tight RAM cap.
    CHECK(compression_level(500'000'000ULL, /*use_complex*/ false) == 0);
    CHECK(compression_level(666'000'000ULL, /*use_complex*/ false) == 0);  // GSM7102845
    CHECK(compression_level(817'000'000ULL, /*use_complex*/ false) == 0);  // GSM5239644
    CHECK(compression_level(1'000'000'000ULL, /*use_complex*/ false) == 0);

    // CB_UMI_Complex at any read count below ULTRA → comp=1.
    CHECK(compression_level(30'000'000ULL,  /*use_complex*/ true) == 1);
    CHECK(compression_level(81'000'000ULL,  /*use_complex*/ true) == 1);
    CHECK(compression_level(200'000'000ULL, /*use_complex*/ true) == 1);
    CHECK(compression_level(400'000'000ULL, /*use_complex*/ true) == 1);

    // CB_UMI_Complex at ULTRA sizes → ultra wins (comp=0).
    CHECK(compression_level(600'000'000ULL, /*use_complex*/ true) == 0);
}

// ── 3. ram_cap_bytes ─────────────────────────────────────────────────────────
static void test_ram_cap_bytes() {
    const uint64_t mem_192g = 192ULL * GiB;
    const uint64_t mem_384g = 384ULL * GiB;

    // Normal small sample at 192G: 75% = 144 GiB.
    CHECK(ram_cap_bytes(50'000'000ULL, mem_192g, false) == (144ULL * GiB));
    CHECK(ram_cap_bytes(0ULL,          mem_192g, false) == (144ULL * GiB));

    // Normal sample at 384G: 75% = 288 GiB.
    CHECK(ram_cap_bytes(100'000'000ULL, mem_384g, false) == (288ULL * GiB));

    // Mega at 384G: 50% = 192 GiB (regression guard for GSM3743501 408M).
    CHECK(ram_cap_bytes(400'000'000ULL, mem_384g, false) == (192ULL * GiB));
    CHECK(ram_cap_bytes(408'000'000ULL, mem_384g, false) == (192ULL * GiB));
    CHECK(ram_cap_bytes(499'999'999ULL, mem_384g, false) == (192ULL * GiB));

    // Ultra at 384G: 15% = ~57.6 GiB (< 64 GiB hard cap, so hard cap not active).
    // Primary fix target for AUTOFIX-MEGA-SORT-RSS-OVERAGE.
    CHECK(ram_cap_bytes(500'000'000ULL, mem_384g, false) == (mem_384g * 15 / 100));
    CHECK(ram_cap_bytes(666'000'000ULL, mem_384g, false) == (mem_384g * 15 / 100));
    CHECK(ram_cap_bytes(800'000'000ULL, mem_384g, false) == (mem_384g * 15 / 100));
    CHECK(ram_cap_bytes(817'000'000ULL, mem_384g, false) == (mem_384g * 15 / 100));

    // CB_UMI_Complex at 192G: 15% = ~28.8 GiB regardless of read count.
    CHECK(ram_cap_bytes(30'000'000ULL,  mem_192g, true) == (mem_192g * 15 / 100));
    CHECK(ram_cap_bytes(81'000'000ULL,  mem_192g, true) == (mem_192g * 15 / 100));

    // slurm_mem_bytes == 0 → 0 (caller handles as "no cap").
    CHECK(ram_cap_bytes(100'000'000ULL, 0ULL, false) == 0ULL);
    CHECK(ram_cap_bytes(800'000'000ULL, 0ULL, false) == 0ULL);

    // Threshold boundaries.
    // Exactly 200M: still normal → 75%.
    CHECK(ram_cap_bytes(MEGA_READ_THRESHOLD,    mem_384g, false) == (288ULL * GiB));
    // 200M + 1: mega tier → 50%.
    CHECK(ram_cap_bytes(MEGA_READ_THRESHOLD + 1, mem_384g, false) == (192ULL * GiB));
    // 500M exactly: ultra → 15% (< 64 GiB hard cap for 384G).
    CHECK(ram_cap_bytes(ULTRA_READ_THRESHOLD,   mem_384g, false) == (mem_384g * 15 / 100));

    // Hard-cap test: large allocation where 15% > 64 GiB must be clamped.
    // 512 GiB * 15% = 76.8 GiB > 64 GiB → result must be exactly 64 GiB.
    const uint64_t mem_512g = 512ULL * GiB;
    CHECK(ram_cap_bytes(600'000'000ULL, mem_512g, false) == (64ULL * GiB));
    CHECK(ram_cap_bytes(600'000'000ULL, mem_512g, true)  == (64ULL * GiB));
}

// ── 4. tier_for ──────────────────────────────────────────────────────────────
static void test_tier_for() {
    CHECK(tier_for(50'000'000ULL,  false) == Tier::Normal);
    CHECK(tier_for(200'000'000ULL, false) == Tier::Normal);  // strict > for mega
    CHECK(tier_for(300'000'000ULL, false) == Tier::Mega);
    CHECK(tier_for(408'000'000ULL, false) == Tier::Mega);
    CHECK(tier_for(500'000'000ULL, false) == Tier::Ultra);
    CHECK(tier_for(666'000'000ULL, false) == Tier::Ultra);
    CHECK(tier_for(817'000'000ULL, false) == Tier::Ultra);

    // Complex always wins over Normal/Mega; Ultra wins over Complex only when n_reads>=500M.
    CHECK(tier_for(30'000'000ULL,  true) == Tier::Complex);
    CHECK(tier_for(100'000'000ULL, true) == Tier::Complex);
    CHECK(tier_for(400'000'000ULL, true) == Tier::Complex);
    // Note: the Tier enum is purely for logging; ram_cap_bytes and compression_level
    // both correctly handle the (use_complex==true, n_reads>=500M) overlap case by
    // returning 15%/hard-cap / comp=6, so we don't rely on Tier here for correctness.
}

// ── 5. Acceptance-test assertions from AUTOFIX-MEGA-SORT-RSS-OVERAGE spec ────
// Iteration #2 (2026-04-16) — after GSM7102845 666M validator run succeeded
// through STAR alignment with no OOM but crashed at sort due to bins=500
// pathology. The spec for this iteration asserts:
//   - ULTRA_SORT_BINS == 100 (down from 500; default is 50; 100 is 2×)
//   - ultra tier compression == 0 (uncompressed; matches what STAR actually
//     receives and what successfully ran in iteration #1 — the old log
//     printed "6" while the arg cascade wrote "0")
//   - 400M mega tier still uses comp=1 / 50% cap (regression guard)
//   - 50M normal tier still uses comp=0 / 75% cap
static void test_spec_acceptance() {
    // 800M @ 384G: ultra tier.  15% cap (~57.6 GiB) < 64 GiB hard cap.
    CHECK(ram_cap_bytes(800'000'000ULL, 384ULL * GiB, false) == (384ULL * GiB * 15 / 100));
    CHECK(compression_level(800'000'000ULL, false) == 0);

    // 400M @ 384G: mega tier — regression guard for GSM3743501 baseline.
    CHECK(ram_cap_bytes(400'000'000ULL, 384ULL * GiB, false) == (192ULL * GiB));
    CHECK(compression_level(400'000'000ULL, false) == 1);

    // 50M @ 192G: normal tier — uncompressed fast path.
    CHECK(ram_cap_bytes(50'000'000ULL, 192ULL * GiB, false) == (144ULL * GiB));
    CHECK(compression_level(50'000'000ULL, false) == 0);

    // Ultra sort bins: the primary acceptance check for iteration #2.
    // Samples at or above ULTRA_READ_THRESHOLD MUST use ULTRA_SORT_BINS = 100.
    // This is asserted structurally: the header constant is the single source
    // of truth for singlify.cpp's --outBAMsortingBinsN value in the ultra tier.
    CHECK(ULTRA_SORT_BINS == 100);

    // GSM7102845 witness: 666M on a 384G node should tier-classify to Ultra,
    // which then selects ULTRA_SORT_BINS via singlify.cpp.
    CHECK(tier_for(666'000'000ULL, false) == Tier::Ultra);
    CHECK(tier_for(800'000'000ULL, false) == Tier::Ultra);
}

int main() {
    test_thresholds();
    test_compression_level();
    test_ram_cap_bytes();
    test_tier_for();
    test_spec_acceptance();

    std::cout << "mega_sort_params: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
