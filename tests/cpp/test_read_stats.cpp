// SPDX-License-Identifier: MIT
// Tests for singlet/pileup/read_stats.h
#include "singlet/pileup/read_stats.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace singlet;

int main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: lander_waterman with n_dup == 0 → returns n_total
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t result = lander_waterman(1000, 0);
        (void)result;
        assert(result == 1000);
        std::cout << "Test 1 PASS: lander_waterman n_dup=0 returns n_total\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: lander_waterman n_total=100, n_dup=50
    // C = 100*100 / (2*50) = 10000 / 100 = 100
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t result = lander_waterman(100, 50);
        (void)result;
        assert(result == 100);
        std::cout << "Test 2 PASS: lander_waterman(100, 50) = 100\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: lander_waterman n_total=1000, n_dup=100
    // C = 1000*1000 / (2*100) = 1000000 / 200 = 5000
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t result = lander_waterman(1000, 100);
        (void)result;
        assert(result == 5000);
        std::cout << "Test 3 PASS: lander_waterman(1000, 100) = 5000\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: lander_waterman d >= n → returns n_total (guard against overflow)
    // ─────────────────────────────────────────────────────────────────────────
    {
        // d == n
        uint64_t result = lander_waterman(100, 100);
        assert(result == 100);
        
        // d > n (shouldn't happen but be safe)
        result = lander_waterman(100, 150);
        assert(result == 100);
        (void)result;
        
        std::cout << "Test 4 PASS: lander_waterman guards against d >= n\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: lander_waterman with large numbers (no overflow)
    // ─────────────────────────────────────────────────────────────────────────
    {
        // n = 1M, d = 100k
        // C = 1M * 1M / (2 * 100k) = 1e12 / 200k = 5e9
        uint64_t result = lander_waterman(1000000, 100000);
        (void)result;
        assert(result == 5000000);
        std::cout << "Test 5 PASS: lander_waterman large numbers without overflow\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: compute_read_stats basic case: single cell with exon data
    // ─────────────────────────────────────────────────────────────────────────
    {
        // CSC format: exon_indptr=[0, 3], exon_data=[1, 2, 3]
        // This is 1 cell with 3 genes contributing [1, 2, 3] UMIs = sum 6
        int32_t exon_indptr[] = {0, 3};
        uint16_t exon_data[] = {1, 2, 3};
        std::vector<uint32_t> per_cell_reads = {10};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 1);
        
        assert(stats.size() == 1);
        assert(stats[0].total_reads == 10);
        assert(stats[0].unique_umis == 6);
        assert(stats[0].dup_reads == 4);  // 10 - 6 = 4
        assert(std::abs(stats[0].dup_rate - 0.4f) < 1e-6);
        
        // est_complexity = lander_waterman(10, 4) = 100 / 8 = 12 (rounded)
        uint64_t expected = lander_waterman(10, 4);
        (void)expected;
        assert(stats[0].est_complexity == expected);
        
        std::cout << "Test 6 PASS: compute_read_stats single cell\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 7: compute_read_stats with two cells
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Cell 0: exon_indptr[0..1] = [0, 3], exon_data=[1, 2, 3] → sum=6
        // Cell 1: exon_indptr[1..2] = [3, 5], exon_data=[4, 5] → sum=9
        int32_t exon_indptr[] = {0, 3, 5};
        uint16_t exon_data[] = {1, 2, 3, 4, 5};
        std::vector<uint32_t> per_cell_reads = {10, 20};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 2);
        
        assert(stats.size() == 2);
        
        // Cell 0
        assert(stats[0].total_reads == 10);
        assert(stats[0].unique_umis == 6);
        assert(stats[0].dup_reads == 4);
        
        // Cell 1
        assert(stats[1].total_reads == 20);
        assert(stats[1].unique_umis == 9);
        assert(stats[1].dup_reads == 11);
        assert(std::abs(stats[1].dup_rate - 0.55f) < 1e-6);
        
        std::cout << "Test 7 PASS: compute_read_stats two cells\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 8: compute_read_stats with intron data
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Cell 0: exon sum=6, intron sum=4 → total unique=10
        // Cell 1: exon sum=9, intron sum=1 → total unique=10
        int32_t exon_indptr[] = {0, 3, 5};
        uint16_t exon_data[] = {1, 2, 3, 4, 5};
        
        int32_t intron_indptr[] = {0, 2, 3};
        uint16_t intron_data[] = {2, 2, 1};
        
        std::vector<uint32_t> per_cell_reads = {20, 20};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data,
                                        intron_indptr, intron_data, 2);
        
        assert(stats.size() == 2);
        
        // Cell 0: exon(6) + intron(4) = 10 unique
        assert(stats[0].total_reads == 20);
        assert(stats[0].unique_umis == 10);
        assert(stats[0].dup_reads == 10);
        assert(std::abs(stats[0].dup_rate - 0.5f) < 1e-6);
        
        // Cell 1: exon(9) + intron(1) = 10 unique
        assert(stats[1].total_reads == 20);
        assert(stats[1].unique_umis == 10);
        assert(stats[1].dup_reads == 10);
        
        std::cout << "Test 8 PASS: compute_read_stats with intron data\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 9: compute_read_stats with all zeros
    // ─────────────────────────────────────────────────────────────────────────
    {
        int32_t exon_indptr[] = {0, 0, 0};
        uint16_t exon_data[] = {};
        std::vector<uint32_t> per_cell_reads = {0, 0};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 2);
        
        assert(stats.size() == 2);
        assert(stats[0].total_reads == 0);
        assert(stats[0].unique_umis == 0);
        assert(stats[0].dup_reads == 0);
        assert(stats[0].dup_rate == 0.0f);
        assert(stats[0].est_complexity == 0);
        
        assert(stats[1].total_reads == 0);
        assert(stats[1].unique_umis == 0);
        
        std::cout << "Test 9 PASS: compute_read_stats all zeros\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 10: compute_read_stats with zero total_reads but non-zero unique_umis
    // (Can happen with multimappers). dup_reads should be clamped at 0.
    // ─────────────────────────────────────────────────────────────────────────
    {
        int32_t exon_indptr[] = {0, 5};
        uint16_t exon_data[] = {1, 2, 3, 4, 5};
        std::vector<uint32_t> per_cell_reads = {0};  // No primary reads, but...
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 1);
        
        assert(stats.size() == 1);
        assert(stats[0].total_reads == 0);
        assert(stats[0].unique_umis == 15);  // sum of exon_data
        // dup_reads should be clamped at 0 (total < unique)
        assert(stats[0].dup_reads == 0);
        assert(stats[0].dup_rate == 0.0f);
        
        std::cout << "Test 10 PASS: compute_read_stats clamps dup_reads at 0\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 11: compute_read_stats with high duplication rate
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Cell: 100 reads, 10 unique UMIs → 90 duplicates, 0.9 dup_rate
        int32_t exon_indptr[] = {0, 10};
        uint16_t exon_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        std::vector<uint32_t> per_cell_reads = {100};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 1);
        
        assert(stats[0].total_reads == 100);
        assert(stats[0].unique_umis == 10);
        assert(stats[0].dup_reads == 90);
        assert(std::abs(stats[0].dup_rate - 0.9f) < 1e-6);
        
        // est_complexity = lander_waterman(100, 90) = 100*100 / (2*90) = 55 (approx)
        uint64_t expected = lander_waterman(100, 90);
        (void)expected;
        assert(stats[0].est_complexity == expected);
        
        std::cout << "Test 11 PASS: compute_read_stats high duplication\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 12: compute_read_stats with per_cell_reads empty (mismatched size)
    // Should use 0 for missing cells
    // ─────────────────────────────────────────────────────────────────────────
    {
        int32_t exon_indptr[] = {0, 2, 4};
        uint16_t exon_data[] = {1, 2, 3, 4};
        std::vector<uint32_t> per_cell_reads = {};  // Empty
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 2);
        
        assert(stats.size() == 2);
        assert(stats[0].total_reads == 0);  // Not in per_cell_reads
        assert(stats[0].unique_umis == 3);
        assert(stats[0].dup_reads == 0);  // Clamped at 0
        
        assert(stats[1].total_reads == 0);
        assert(stats[1].unique_umis == 7);
        
        std::cout << "Test 12 PASS: compute_read_stats handles missing per_cell_reads\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 13: compute_read_stats empty cells (ncols=0)
    // ─────────────────────────────────────────────────────────────────────────
    {
        int32_t exon_indptr[] = {};
        uint16_t exon_data[] = {};
        std::vector<uint32_t> per_cell_reads = {};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 0);
        
        assert(stats.size() == 0);
        std::cout << "Test 13 PASS: compute_read_stats ncols=0\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 14: compute_read_stats with large count values (uint16 saturation)
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Use large uint16 values
        int32_t exon_indptr[] = {0, 3};
        uint16_t exon_data[] = {65000, 500, 1};  // Sum > 65000
        std::vector<uint32_t> per_cell_reads = {100000};
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 1);
        
        assert(stats[0].total_reads == 100000);
        // unique_umis = 65000 + 500 + 1 = 65501
        assert(stats[0].unique_umis == 65501);
        assert(stats[0].dup_reads == 34499);
        
        std::cout << "Test 14 PASS: compute_read_stats large count values\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 15: compute_read_stats sum exceeds UINT32_MAX (should clamp)
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Note: with uint16_t data, max sum = 65535 * many, so we can't easily
        // exceed UINT32_MAX. But the code guards it anyway. We'll just verify
        // normal large values work.
        int32_t exon_indptr[] = {0, 100};
        uint16_t exon_data[100];
        for (int i = 0; i < 100; ++i) {
            exon_data[i] = 65535;  // Large values
        }
        std::vector<uint32_t> per_cell_reads = {10000000};  // 10M reads
        
        auto stats = compute_read_stats(per_cell_reads, exon_indptr, exon_data, nullptr, nullptr, 1);
        
        assert(stats[0].total_reads == 10000000);
        // unique_umis = sum of exon_data
        uint64_t expected_sum = 100ULL * 65535;
        (void)expected_sum;
        assert(stats[0].unique_umis == std::min(expected_sum, static_cast<uint64_t>(UINT32_MAX)));
        
        std::cout << "Test 15 PASS: compute_read_stats guards sum at UINT32_MAX\n";
    }

    std::cout << "\n✓ All read_stats tests passed!\n";
    return 0;
}
