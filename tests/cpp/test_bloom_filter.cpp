// Test for BloomFilter
#include "singlet/pileup/bloom_filter.h"
#include <cassert>
#include <iostream>
#include <cstdio>
#include <cstdlib>

int main() {
    using namespace singlet_pileup;

    // ── Test 1: Constructor rounds up to 64-bit boundary ──────────────────────
    {
        BloomFilter bf1(100);
        assert(bf1.n_bits() == 128);  // 100 rounded up to 128
        std::cout << "Test 1 PASS: BloomFilter(100).n_bits() == 128\n";

        BloomFilter bf2(64);
        assert(bf2.n_bits() == 64);   // Already aligned
        std::cout << "Test 1 PASS: BloomFilter(64).n_bits() == 64\n";

        BloomFilter bf3(1);
        assert(bf3.n_bits() == 64);   // Rounded up to 64
        std::cout << "Test 1 PASS: BloomFilter(1).n_bits() == 64\n";

        BloomFilter bf4(128);
        assert(bf4.n_bits() == 128);  // Already aligned
        std::cout << "Test 1 PASS: BloomFilter(128).n_bits() == 128\n";
    }

    // ── Test 2: Insert then query ──────────────────────────────────────────────
    {
        BloomFilter bf(1000);
        uint64_t kmer = 0x123456789ABCDEFULL;
        
        // Query before insert should return false
        assert(!bf.query(kmer));
        std::cout << "Test 2 PASS: query before insert returns false\n";
        
        // Insert and query
        bf.insert(kmer);
        assert(bf.query(kmer));
        std::cout << "Test 2 PASS: query after insert returns true\n";
    }

    // ── Test 3: Query not inserted (no false negative) ──────────────────────
    {
        BloomFilter bf(10000);
        uint64_t kmer1 = 0x123456789ABCDEFULL;
        uint64_t kmer2 = 0xFEDCBA9876543210ULL;
        
        bf.insert(kmer1);
        assert(bf.query(kmer1));
        
        // This should usually return false (may have rare false positives in bloom filters)
        // For a reasonably sized filter, false positive rate is low
        // We test the determinism: if we don't insert kmer2, it should not query true
        // (or at least, requerying the same kmer2 gives same result)
        bool result1 = bf.query(kmer2);
        bool result2 = bf.query(kmer2);
        (void)result2;
        assert(result1 == result2);  // Must be deterministic
        std::cout << "Test 3 PASS: query is deterministic (kmer2 result=" << result1 << ")\n";
    }

    // ── Test 4: Multiple inserts ──────────────────────────────────────────────
    {
        BloomFilter bf(5000);
        std::vector<uint64_t> kmers = {
            0x0000000000000001ULL,
            0x0000000000000002ULL,
            0x0000000000000004ULL,
            0x0000000000000008ULL,
            0x0000000000000010ULL,
        };
        
        for (auto kmer : kmers) {
            bf.insert(kmer);
        }
        
        for (auto kmer : kmers) {
            (void)kmer;
            assert(bf.query(kmer));
        }
        std::cout << "Test 4 PASS: all " << kmers.size() << " inserted k-mers query true\n";
    }

    // ── Test 5: Save/Load round-trip ───────────────────────────────────────
    {
        const char* test_file = "/tmp/test_bloom.bloom";
        
        // Create, populate, and save
        {
            BloomFilter bf(2000);
            std::vector<uint64_t> kmers = {
                0xDEADBEEFCAFEBABEULL,
                0x0123456789ABCDEFULL,
                0xFEDCBA9876543210ULL,
                0x1111111111111111ULL,
                0x2222222222222222ULL,
            };
            
            for (auto kmer : kmers) {
                bf.insert(kmer);
            }
            
            bf.save(test_file);
            std::cout << "Test 5a PASS: saved filter to " << test_file << "\n";
        }
        
        // Load into new filter and verify queries
        {
            BloomFilter bf_loaded;
            bool ok = bf_loaded.load(test_file);
            (void)ok;
            assert(ok);
            std::cout << "Test 5b PASS: loaded filter from " << test_file << "\n";
            
            assert(bf_loaded.n_bits() == 2048);  // 2000 rounded up to 64-bit boundary
            std::cout << "Test 5c PASS: loaded filter has correct n_bits() == " << bf_loaded.n_bits() << "\n";
            
            // Verify all previously inserted kmers still query true
            assert(bf_loaded.query(0xDEADBEEFCAFEBABEULL));
            assert(bf_loaded.query(0x0123456789ABCDEFULL));
            assert(bf_loaded.query(0xFEDCBA9876543210ULL));
            assert(bf_loaded.query(0x1111111111111111ULL));
            assert(bf_loaded.query(0x2222222222222222ULL));
            std::cout << "Test 5d PASS: all saved k-mers query true after load\n";
        }
        
        // Clean up
        std::remove(test_file);
        std::cout << "Test 5e PASS: save/load round-trip complete\n";
    }

    // ── Test 6: count_hits_numeric ────────────────────────────────────────
    {
        BloomFilter bf(5000);
        
        // Create a short byte-numeric sequence: A=0, C=1, G=2, T=3
        // Sequence: AAAAA... (all 0's)
        uint8_t seq[25];
        for (int i = 0; i < 25; ++i) {
            seq[i] = 0;  // All A's
        }
        
        // The k-mer of all A's (k=21) is 0 (all 21 bits are 0)
        // Both forward and reverse complement of all A's are the same (0)
        uint64_t all_a_kmer = 0;
        bf.insert(all_a_kmer);
        
        int hits = bf.count_hits_numeric(seq, 25, 21);
        assert(hits >= 1);  // Should find at least one hit
        std::cout << "Test 6 PASS: count_hits_numeric found " << hits << " hit(s) in sequence\n";
    }

    // ── Test 7: count_hits_numeric with mixed sequence ──────────────────────
    {
        BloomFilter bf(8000);
        
        // Create a sequence: ACGTACGTACGTACGTACGTACGT... (repeating pattern)
        // Pattern: A=0, C=1, G=2, T=3, A=0, C=1, G=2, T=3, ...
        uint8_t seq[50];
        const uint8_t pattern[] = {0, 1, 2, 3};
        for (int i = 0; i < 50; ++i) {
            seq[i] = pattern[i % 4];
        }
        
        // Build first k-mer (k=21): ACGTACGTACGTACGTACGTA
        // In 2-bit encoding: bits from positions 0-20 of the sequence
        uint64_t first_kmer = 0;
        uint64_t mask = (1ULL << 42) - 1;  // 21 * 2 = 42 bits
        for (int i = 0; i < 21; ++i) {
            first_kmer = ((first_kmer << 2) | seq[i]) & mask;
        }
        
        bf.insert(first_kmer);
        int hits = bf.count_hits_numeric(seq, 50, 21);
        assert(hits >= 1);
        std::cout << "Test 7 PASS: count_hits_numeric found " << hits 
                  << " hit(s) in mixed sequence\n";
    }

    // ── Test 8: murmurhash3_mix is deterministic ──────────────────────────
    {
        uint64_t key = 0x123456789ABCDEFULL;
        uint64_t seed = 0x9e3779b97f4a7c15ULL;
        
        uint64_t hash1 = murmurhash3_mix(key, seed);
        (void)hash1;
        uint64_t hash2 = murmurhash3_mix(key, seed);
        (void)hash2;
        
        assert(hash1 == hash2);
        std::cout << "Test 8 PASS: murmurhash3_mix(key, seed) is deterministic\n";
    }

    // ── Test 9: murmurhash3_mix different seeds produce different outputs ────
    {
        uint64_t key = 0x123456789ABCDEFULL;
        uint64_t seed1 = 1;
        uint64_t seed2 = 2;
        
        uint64_t hash1 = murmurhash3_mix(key, seed1);
        (void)hash1;
        uint64_t hash2 = murmurhash3_mix(key, seed2);
        (void)hash2;
        
        assert(hash1 != hash2);
        std::cout << "Test 9 PASS: murmurhash3_mix different seeds produce different outputs\n";
    }

    // ── Test 10: murmurhash3_mix different keys produce different outputs ────
    {
        uint64_t key1 = 0x123456789ABCDEFULL;
        uint64_t key2 = 0xFEDCBA9876543210ULL;
        uint64_t seed = 0x9e3779b97f4a7c15ULL;
        
        uint64_t hash1 = murmurhash3_mix(key1, seed);
        (void)hash1;
        uint64_t hash2 = murmurhash3_mix(key2, seed);
        (void)hash2;
        
        assert(hash1 != hash2);
        std::cout << "Test 10 PASS: murmurhash3_mix different keys produce different outputs\n";
    }

    // ── Test 11: Default constructor ───────────────────────────────────────
    {
        BloomFilter bf;
        assert(bf.n_bits() == 0);
        assert(bf.empty());
        std::cout << "Test 11 PASS: default constructor creates empty filter with n_bits() == 0\n";
    }

    // ── Test 12: Large n_bits ──────────────────────────────────────────────
    {
        BloomFilter bf(1000000);  // 1M bits = 15625 64-bit words
        assert(bf.n_bits() == 1000000);
        std::cout << "Test 12 PASS: BloomFilter(1000000).n_bits() == 1000000\n";
        
        // Insert and query should still work
        uint64_t kmer = 0xABCDEF1234567890ULL;
        bf.insert(kmer);
        assert(bf.query(kmer));
        std::cout << "Test 12 PASS: large filter insert/query works\n";
    }

    std::cout << "\nAll Bloom filter tests passed!\n";
    return 0;
}
