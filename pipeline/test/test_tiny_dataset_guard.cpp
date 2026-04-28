// test_tiny_dataset_guard.cpp — Unit tests for singlet-pileup/tiny_dataset_guard.h
#include <cassert>
#include <iostream>
#include <string>

#include "singlet-pileup/tiny_dataset_guard.h"

using singlet_pileup::TinyDatasetGuard;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (cond) {                                                     \
            ++g_pass;                                                   \
        } else {                                                        \
            ++g_fail;                                                   \
            std::cerr << "FAIL: " << #cond                              \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        }                                                               \
    } while (0)

// ── 1. is_too_small ───────────────────────────────────────────────────────────

static void test_is_too_small() {
    CHECK(TinyDatasetGuard::is_too_small(0) == true);
    CHECK(TinyDatasetGuard::is_too_small(50) == true);
    CHECK(TinyDatasetGuard::is_too_small(999) == true);
    CHECK(TinyDatasetGuard::is_too_small(1000) == false);     // exactly at threshold
    CHECK(TinyDatasetGuard::is_too_small(110'076) == false);  // SRR23027738 read count
    CHECK(TinyDatasetGuard::is_too_small(40'000'000) == false);
}

// ── 2. recommended_bam_sort_ram ───────────────────────────────────────────────

static void test_recommended_bam_sort_ram() {
    // < 1M reads → 1 GB
    uint64_t r1 = TinyDatasetGuard::recommended_bam_sort_ram(110'076);
    CHECK(r1 == (1ULL << 30));

    // Exactly 1M → still < 1M threshold? No, 1M is not < 1M, goes to next tier
    uint64_t r2 = TinyDatasetGuard::recommended_bam_sort_ram(1'000'000);
    CHECK(r2 == (4ULL << 30));

    // 5M reads → 4 GB
    uint64_t r3 = TinyDatasetGuard::recommended_bam_sort_ram(5'000'000);
    CHECK(r3 == (4ULL << 30));

    // 10M reads (at boundary) → UINT64_MAX (no cap; MEGA formula handles these)
    uint64_t r4 = TinyDatasetGuard::recommended_bam_sort_ram(10'000'000);
    CHECK(r4 == UINT64_MAX);

    // 613M reads → UINT64_MAX (no cap; MEGA formula must not be overridden)
    uint64_t r5 = TinyDatasetGuard::recommended_bam_sort_ram(613'000'000);
    CHECK(r5 == UINT64_MAX);
}

// ── 3. enough_for_cell_calling ────────────────────────────────────────────────

static void test_enough_for_cell_calling() {
    CHECK(TinyDatasetGuard::enough_for_cell_calling(0) == false);
    CHECK(TinyDatasetGuard::enough_for_cell_calling(9) == false);
    CHECK(TinyDatasetGuard::enough_for_cell_calling(10) == true);
    CHECK(TinyDatasetGuard::enough_for_cell_calling(100) == true);
}

// ── 4. warning_message ────────────────────────────────────────────────────────

static void test_warning_message() {
    // Fewer than MIN_READS_FOR_PILEUP (100) → "too small" message
    std::string w1 = TinyDatasetGuard::warning_message(50);
    CHECK(w1.find("too small") != std::string::npos);

    // Between pileup and alignment thresholds → "may be unreliable"
    std::string w2 = TinyDatasetGuard::warning_message(500);
    CHECK(w2.find("unreliable") != std::string::npos);

    // Above alignment threshold → empty string
    std::string w3 = TinyDatasetGuard::warning_message(5'000);
    CHECK(w3.empty());

    // Large normal dataset → empty string
    std::string w4 = TinyDatasetGuard::warning_message(40'000'000);
    CHECK(w4.empty());
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_is_too_small();
    test_recommended_bam_sort_ram();
    test_enough_for_cell_calling();
    test_warning_message();

    std::cout << "tiny_dataset_guard: " << g_pass << " passed, " << g_fail << " failed\n";
    return g_fail == 0 ? 0 : 1;
}
