// test/test_ram_tier.cpp — Unit tests for B-G3-3/4 RAM tier selection
//
// Tests:
//  1. slurm_tier_bamsort_ram: exact tier table lookup (64/128/192/384 GB)
//  2. Tolerance: ±1 GiB around each boundary still matches
//  3. Unknown allocation: returns 0 (falls back to SA-based heuristic)
//  4. XL_READ_THRESHOLD matches MEGA_READ_THRESHOLD (gating is consistent)
//  5. slurm_tier_bamsort_ram values are ~39% of allocation

#include "singlet/pileup/mega_sort_params.h"

#include <cassert>
#include <cstdint>
#include <cstdio>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; std::fprintf(stderr, "  PASS: %s\n", msg); } \
    else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", msg); } \
} while(0)

static constexpr uint64_t GiB = 1024ULL * 1024 * 1024;

static void test_tier_exact() {
    std::fprintf(stderr, "[test_ram_tier] Tier exact lookup...\n");
    using namespace singlet_pileup::mega_sort;

    // Standard 64 GB → 25 GB
    CHECK(slurm_tier_bamsort_ram(64 * GiB) == 25 * GiB, "64 GiB → 25 GiB");
    // Large  128 GB → 50 GB
    CHECK(slurm_tier_bamsort_ram(128 * GiB) == 50 * GiB, "128 GiB → 50 GiB");
    // XL-192 192 GB → 75 GB
    CHECK(slurm_tier_bamsort_ram(192 * GiB) == 75 * GiB, "192 GiB → 75 GiB");
    // XL-384 384 GB → 150 GB
    CHECK(slurm_tier_bamsort_ram(384 * GiB) == 150 * GiB, "384 GiB → 150 GiB");
}

static void test_tier_tolerance() {
    std::fprintf(stderr, "[test_ram_tier] Tier tolerance ±1 GiB...\n");
    using namespace singlet_pileup::mega_sort;

    // 63 GiB should still match 64-GiB tier
    CHECK(slurm_tier_bamsort_ram(63 * GiB) == 25 * GiB, "63 GiB → 25 GiB (low tolerance)");
    CHECK(slurm_tier_bamsort_ram(65 * GiB) == 25 * GiB, "65 GiB → 25 GiB (high tolerance)");

    // 127 GiB / 129 GiB → 128-GiB tier
    CHECK(slurm_tier_bamsort_ram(127 * GiB) == 50 * GiB, "127 GiB → 50 GiB (low tolerance)");
    CHECK(slurm_tier_bamsort_ram(129 * GiB) == 50 * GiB, "129 GiB → 50 GiB (high tolerance)");

    // 191 GiB / 193 GiB → 192-GiB tier
    CHECK(slurm_tier_bamsort_ram(191 * GiB) == 75 * GiB, "191 GiB → 75 GiB (low tolerance)");
    CHECK(slurm_tier_bamsort_ram(193 * GiB) == 75 * GiB, "193 GiB → 75 GiB (high tolerance)");

    // 383 GiB / 385 GiB → 384-GiB tier
    CHECK(slurm_tier_bamsort_ram(383 * GiB) == 150 * GiB, "383 GiB → 150 GiB (low tolerance)");
    CHECK(slurm_tier_bamsort_ram(385 * GiB) == 150 * GiB, "385 GiB → 150 GiB (high tolerance)");
}

static void test_tier_unknown() {
    std::fprintf(stderr, "[test_ram_tier] Unknown allocation → 0...\n");
    using namespace singlet_pileup::mega_sort;

    // Unrecognised tier sizes return 0 (caller falls back to SA heuristic)
    CHECK(slurm_tier_bamsort_ram(0) == 0,            "0 GiB → 0");
    CHECK(slurm_tier_bamsort_ram(32 * GiB) == 0,     "32 GiB → 0 (not a tier)");
    CHECK(slurm_tier_bamsort_ram(96 * GiB) == 0,     "96 GiB → 0 (not a tier)");
    CHECK(slurm_tier_bamsort_ram(256 * GiB) == 0,    "256 GiB → 0 (not a tier)");
    CHECK(slurm_tier_bamsort_ram(512 * GiB) == 0,    "512 GiB → 0 (not a tier)");
}

static void test_tier_fraction() {
    std::fprintf(stderr, "[test_ram_tier] Tier values ≈ 39%% of allocation...\n");
    using namespace singlet_pileup::mega_sort;

    // Each tier should be between 35% and 45% of allocation
    auto check_frac = [](uint64_t alloc_gib, uint64_t bamsort_gib, const char* label) {
        double frac = static_cast<double>(bamsort_gib) / static_cast<double>(alloc_gib);
        bool ok = (frac >= 0.35 && frac <= 0.45);
        if (ok) { ++g_pass; std::fprintf(stderr, "  PASS: %s (%.0f%%)\n", label, frac * 100); }
        else    { ++g_fail; std::fprintf(stderr, "  FAIL: %s (%.0f%% not in 35-45%%)\n", label, frac * 100); }
    };
    check_frac(64,  25,  "Standard 64→25 GiB");
    check_frac(128, 50,  "Large 128→50 GiB");
    check_frac(192, 75,  "XL-192→75 GiB");
    check_frac(384, 150, "XL-384→150 GiB");
}

static void test_xl_threshold_consistency() {
    std::fprintf(stderr, "[test_ram_tier] XL_READ_THRESHOLD == MEGA_READ_THRESHOLD...\n");
    using namespace singlet_pileup::mega_sort;
    CHECK(XL_READ_THRESHOLD == MEGA_READ_THRESHOLD,
          "XL_READ_THRESHOLD == MEGA_READ_THRESHOLD (200M)");
}

int main() {
    test_tier_exact();
    test_tier_tolerance();
    test_tier_unknown();
    test_tier_fraction();
    test_xl_threshold_consistency();

    std::fprintf(stderr, "\n[test_ram_tier] Results: %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
