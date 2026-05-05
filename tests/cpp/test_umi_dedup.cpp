// Test for UmiDedup and directional UMI error correction
#include "singlet/pileup/umi_dedup.h"
#include <cassert>
#include <iostream>

using namespace singlet;

// Mock accumulator for testing DirectionalUmiStore::finalize()
struct MockAccumulator {
    std::vector<std::pair<uint32_t, uint32_t>> calls;  // (exon_idx, bc_idx) pairs
    
    void increment(uint32_t exon_idx, uint32_t bc_idx) {
        calls.push_back({exon_idx, bc_idx});
    }
};

int main() {
    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: UmiDedup — insert new returns true
    // ─────────────────────────────────────────────────────────────────────────
    {
        UmiDedup dedup;
        bool inserted = dedup.insert(0, 0, "ACGTACGT", 8);
        assert(inserted == true);
        assert(dedup.n_unique() == 1);
        assert(dedup.n_duplicate() == 0);
        std::cout << "Test 1 PASS: UmiDedup insert new returns true\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: UmiDedup — insert duplicate returns false
    // ─────────────────────────────────────────────────────────────────────────
    {
        UmiDedup dedup;
        dedup.insert(0, 0, "ACGTACGT", 8);
        bool inserted = dedup.insert(0, 0, "ACGTACGT", 8);
        assert(inserted == false);
        assert(dedup.n_unique() == 1);
        assert(dedup.n_duplicate() == 1);
        std::cout << "Test 2 PASS: UmiDedup insert duplicate returns false\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: UmiDedup — stats correct after multiple inserts
    // ─────────────────────────────────────────────────────────────────────────
    {
        UmiDedup dedup;
        dedup.insert(0, 0, "AAAA", 4);  // unique
        dedup.insert(0, 0, "AAAA", 4);  // dup
        dedup.insert(0, 0, "CCCC", 4);  // unique (different UMI)
        dedup.insert(0, 0, "GGGG", 4);  // unique
        dedup.insert(0, 0, "GGGG", 4);  // dup
        dedup.insert(0, 0, "GGGG", 4);  // dup
        
        assert(dedup.n_unique() == 3);
        assert(dedup.n_duplicate() == 3);
        assert(dedup.size() == 3);
        std::cout << "Test 3 PASS: UmiDedup stats correct (3 unique, 3 duplicates)\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: UmiDedup — clear resets state
    // ─────────────────────────────────────────────────────────────────────────
    {
        UmiDedup dedup;
        dedup.insert(0, 0, "AAAA", 4);
        dedup.insert(0, 0, "AAAA", 4);
        assert(dedup.size() > 0);
        
        dedup.clear();
        assert(dedup.n_unique() == 0);
        assert(dedup.n_duplicate() == 0);
        assert(dedup.size() == 0);
        
        bool inserted = dedup.insert(0, 0, "AAAA", 4);
        assert(inserted == true);
        assert(dedup.n_unique() == 1);
        std::cout << "Test 4 PASS: UmiDedup clear resets state\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: UmiDedup — reserve doesn't crash
    // ─────────────────────────────────────────────────────────────────────────
    {
        UmiDedup dedup;
        dedup.reserve(1000);
        dedup.insert(0, 0, "AAAA", 4);
        dedup.insert(0, 0, "CCCC", 4);
        assert(dedup.n_unique() == 2);
        std::cout << "Test 5 PASS: UmiDedup reserve doesn't crash\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 6: umi_pack_2bit — known packing for "ACGT"
    // ─────────────────────────────────────────────────────────────────────────
    {
        // A=0, C=1, G=2, T=3
        // Little-endian: first base at LSBs
        // "ACGT" → A(0) at bits [1:0], C(1) at bits [3:2], G(2) at bits [5:4], T(3) at bits [7:6]
        // = 0b11100100 = 0xE4
        uint64_t packed = umi_pack_2bit("ACGT", 4);
        assert(packed == 0xE4);
        
        // "AA" → A(0) at [1:0], A(0) at [3:2] = 0b00000000 = 0x00
        assert(umi_pack_2bit("AA", 2) == 0x00);
        
        // "CC" → C(1) at [1:0], C(1) at [3:2] = 0b00000101 = 0x05
        assert(umi_pack_2bit("CC", 2) == 0x05);
        
        // "TT" → T(3) at [1:0], T(3) at [3:2] = 0b00001111 = 0x0F
        assert(umi_pack_2bit("TT", 2) == 0x0F);
        
        // Case-insensitive: "acgt" should equal "ACGT"
        assert(umi_pack_2bit("acgt", 4) == umi_pack_2bit("ACGT", 4));
        
        std::cout << "Test 6 PASS: umi_pack_2bit known packing correct\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 7: umi_hamming_2bit — identical UMIs have distance 0
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t a = umi_pack_2bit("ACGT", 4);
        uint64_t b = umi_pack_2bit("ACGT", 4);
        int dist = umi_hamming_2bit(a, b, 4);
        assert(dist == 0);
        std::cout << "Test 7 PASS: umi_hamming_2bit identical distance=0\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 8: umi_hamming_2bit — 1-base difference has distance 1
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t a = umi_pack_2bit("ACGT", 4);  // A=0, C=1, G=2, T=3
        uint64_t b = umi_pack_2bit("CCGT", 4);  // C=1, C=1, G=2, T=3 (first base differs)
        int dist = umi_hamming_2bit(a, b, 4);
        assert(dist == 1);
        
        // Two positions differ
        uint64_t c = umi_pack_2bit("ACCT", 4);  // A=0, C=1, C=1, T=3 (position 2 differs)
        dist = umi_hamming_2bit(a, c, 4);
        assert(dist == 1);
        
        std::cout << "Test 8 PASS: umi_hamming_2bit 1-base diff=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 9: umi_hamming_2bit — 2-base difference has distance 2
    // ─────────────────────────────────────────────────────────────────────────
    {
        uint64_t a = umi_pack_2bit("ACGT", 4);
        uint64_t b = umi_pack_2bit("CCCT", 4);  // positions 0 and 2 differ
        int dist = umi_hamming_2bit(a, b, 4);
        assert(dist == 2);
        std::cout << "Test 9 PASS: umi_hamming_2bit 2-base diff=2\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 10: directional_dedup — empty vector returns 0
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        uint32_t components = directional_dedup(umi_counts, 4);
        assert(components == 0);
        std::cout << "Test 10 PASS: directional_dedup empty vector=0\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 11: directional_dedup — single element returns 1
    // ─────────────────────────────────────────────────────────────────────────
    {
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        uint64_t packed = umi_pack_2bit("ACGT", 4);
        umi_counts.push_back({packed, 5});
        
        uint32_t components = directional_dedup(umi_counts, 4);
        assert(components == 1);
        std::cout << "Test 11 PASS: directional_dedup single element=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 12: directional_dedup — two UMIs with hamming=1, count ratio merges
    // ─────────────────────────────────────────────────────────────────────────
    {
        // UMI1 (higher count) and UMI2 (lower count) with hamming=1
        // UMI1 count = 10, UMI2 count = 3
        // Merge condition: count1 >= 2*count2 - 1 → 10 >= 2*3 - 1 = 5 ✓
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        uint64_t umi1 = umi_pack_2bit("ACGT", 4);
        uint64_t umi2 = umi_pack_2bit("CCGT", 4);  // hamming=1 from umi1
        
        // Must be sorted by count descending
        umi_counts.push_back({umi1, 10});
        umi_counts.push_back({umi2, 3});
        
        uint32_t components = directional_dedup(umi_counts, 4);
        assert(components == 1);  // merged into one component
        std::cout << "Test 12 PASS: directional_dedup hamming=1 with merge (ratio OK)=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 13: directional_dedup — two UMIs with hamming=1, count ratio doesn't merge
    // ─────────────────────────────────────────────────────────────────────────
    {
        // UMI1 count = 5, UMI2 count = 3
        // Merge condition: count1 >= 2*count2 - 1 → 5 >= 2*3 - 1 = 5 ✓ (equal, should merge!)
        // Let's use count1 = 4 instead
        // 4 >= 2*3 - 1 = 5? NO, 4 < 5, so no merge
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        uint64_t umi1 = umi_pack_2bit("ACGT", 4);
        uint64_t umi2 = umi_pack_2bit("CCGT", 4);  // hamming=1
        
        umi_counts.push_back({umi1, 4});
        umi_counts.push_back({umi2, 3});
        
        uint32_t components = directional_dedup(umi_counts, 4);
        assert(components == 2);  // not merged, stay separate
        std::cout << "Test 13 PASS: directional_dedup hamming=1 but count ratio rejects merge=2\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 14: directional_dedup — two UMIs with hamming=2, no merge
    // ─────────────────────────────────────────────────────────────────────────
    {
        // Even with good count ratio, hamming=2 shouldn't merge
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        uint64_t umi1 = umi_pack_2bit("ACGT", 4);
        uint64_t umi2 = umi_pack_2bit("CCCT", 4);  // hamming=2 (positions 0 and 2 differ)
        
        umi_counts.push_back({umi1, 100});
        umi_counts.push_back({umi2, 1});
        
        uint32_t components = directional_dedup(umi_counts, 4);
        assert(components == 2);  // hamming != 1, so no merge
        std::cout << "Test 14 PASS: directional_dedup hamming=2 no merge=2\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 15: DirectionalUmiStore — record single UMI → finalize emits 1
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("ACGT", 4);
        store.record(0, 0, 10, umi);
        
        MockAccumulator acc;
        uint64_t total = store.finalize(acc, 4);
        
        assert(total == 1);
        assert(acc.calls.size() == 1);
        assert(acc.calls[0].first == 10);   // exon_idx
        assert(acc.calls[0].second == 0);   // bc_idx
        std::cout << "Test 15 PASS: DirectionalUmiStore record single UMI finalize=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 16: DirectionalUmiStore — record same UMI twice → finalize emits 1
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("ACGT", 4);
        store.record(0, 0, 10, umi);
        store.record(0, 0, 10, umi);  // same UMI
        
        MockAccumulator acc;
        uint64_t total = store.finalize(acc, 4);
        
        assert(total == 1);
        assert(acc.calls.size() == 1);
        std::cout << "Test 16 PASS: DirectionalUmiStore record same UMI twice finalize=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 17: DirectionalUmiStore — two different UMIs same (bc,gene) with hamming=1 merges
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi1 = umi_pack_2bit("ACGT", 4);
        uint64_t umi2 = umi_pack_2bit("CCGT", 4);  // hamming=1
        
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 20, umi2);
        store.record(0, 0, 20, umi2);
        
        // umi1 count=3, umi2 count=2
        // Merge if: 3 >= 2*2 - 1 = 3? YES
        
        MockAccumulator acc;
        uint64_t total = store.finalize(acc, 4);
        
        assert(total == 1);  // merged to 1 component
        assert(acc.calls.size() == 1);
        // The root (umi1) should use its exon (10)
        assert(acc.calls[0].first == 10);
        std::cout << "Test 17 PASS: DirectionalUmiStore two UMIs hamming=1 merge=1\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 18: DirectionalUmiStore::per_cell_stats — single UMI returns correct stats
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("ACGT", 4);
        store.record(0, 0, 10, umi);
        store.record(0, 0, 10, umi);
        
        auto stats = store.per_cell_stats(4);
        
        assert(stats.size() == 1);
        assert(stats[0].total_reads == 2);
        assert(stats[0].unique_umis == 1);
        std::cout << "Test 18 PASS: DirectionalUmiStore per_cell_stats single UMI correct\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 19: DirectionalUmiStore::per_cell_stats — multiple cells/genes
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi1 = umi_pack_2bit("ACGT", 4);
        uint64_t umi2 = umi_pack_2bit("CCGT", 4);
        
        // Cell 0, gene 0
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 10, umi1);
        store.record(0, 0, 20, umi2);
        
        // Cell 1, gene 0
        store.record(1, 0, 10, umi1);
        store.record(1, 0, 10, umi1);
        
        auto stats = store.per_cell_stats(4);
        
        assert(stats.size() == 2);
        // Cell 0: 3+1=4 total reads, 1 unique (merged)
        assert(stats[0].total_reads == 4);
        assert(stats[0].unique_umis == 1);
        // Cell 1: 2 total reads, 1 unique
        assert(stats[1].total_reads == 2);
        assert(stats[1].unique_umis == 1);
        std::cout << "Test 19 PASS: DirectionalUmiStore per_cell_stats multiple cells correct\n";
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 20: DirectionalUmiStore::clear
    // ─────────────────────────────────────────────────────────────────────────
    {
        DirectionalUmiStore store;
        uint64_t umi = umi_pack_2bit("ACGT", 4);
        store.record(0, 0, 10, umi);
        
        assert(store.n_groups() == 1);
        assert(!store.empty());
        
        store.clear();
        assert(store.n_groups() == 0);
        assert(store.empty());
        std::cout << "Test 20 PASS: DirectionalUmiStore clear works\n";
    }

    std::cout << "\nAll UMI dedup tests passed!\n";
    return 0;
}
