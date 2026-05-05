// singlet-pileup/bloom_filter.h — N1: Species Bloom filter
//
// Simple bit-vector Bloom filter for k-mer sets (~50MB per species).
// Uses 3 MurmurHash3 variants for low collision rate.
//
// Usage (reading):
//   BloomFilter bf;
//   bf.load("/path/to/human_21mer.bloom");
//   bool hit = bf.query(canonical_kmer);
//
// Usage (building, offline via Python script):
//   BloomFilter bf(BLOOM_BITS);
//   for (auto kmer : kmers) bf.insert(kmer);
//   bf.save("/path/to/filter.bloom");

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_pileup {

// ── MurmurHash3 mixing ─────────────────────────────────────────────────────
inline uint64_t murmurhash3_mix(uint64_t key, uint64_t seed) {
    key ^= seed;
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

// ── BloomFilter ────────────────────────────────────────────────────────────

class BloomFilter {
public:
    static constexpr uint32_t MAGIC   = 0x424C4F4D; // "BLOM"
    static constexpr uint32_t VERSION = 1;
    static constexpr int      N_HASH  = 3;           // hash functions

    BloomFilter() = default;

    /// Construct an empty filter for insertion (n_bits must be a multiple of 64)
    explicit BloomFilter(uint64_t n_bits) {
        n_bits_ = (n_bits + 63) & ~63ULL;  // round up to 64-bit boundary
        bits_.assign(n_bits_ / 64, 0ULL);
    }

    uint64_t n_bits() const { return n_bits_; }

    /// Insert a canonical k-mer (64-bit encoded)
    void insert(uint64_t kmer) {
        for (int i = 0; i < N_HASH; ++i) {
            uint64_t h = murmurhash3_mix(kmer, static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
            uint64_t pos = h % n_bits_;
            bits_[pos >> 6] |= (1ULL << (pos & 63));
        }
    }

    /// Query a canonical k-mer; returns true if possibly in set (may false-positive)
    bool query(uint64_t kmer) const {
        for (int i = 0; i < N_HASH; ++i) {
            uint64_t h = murmurhash3_mix(kmer, static_cast<uint64_t>(i + 1) * 0x9e3779b97f4a7c15ULL);
            uint64_t pos = h % n_bits_;
            if (!(bits_[pos >> 6] & (1ULL << (pos & 63)))) return false;
        }
        return true;
    }

    /// Count how many k-mers in query_set hit this filter
    /// Input: byte-numeric STAR sequence (A=0, C=1, G=2, T=3), length len, K=21
    int count_hits_numeric(const uint8_t* seq, int len, int k = 21) const {
        if (len < k) return 0;
        static const uint8_t COMP[4] = {3, 2, 1, 0};
        const uint64_t mask = (1ULL << (2 * k)) - 1;
        int hits = 0;
        uint64_t kmer = 0;
        int valid = 0;
        for (int i = 0; i < len; ++i) {
            uint8_t b = seq[i];
            if (b > 3) { valid = 0; kmer = 0; continue; }
            kmer = ((kmer << 2) | b) & mask;
            if (++valid >= k) {
                // Compute canonical
                uint64_t tmp = kmer, rc = 0;
                for (int j = 0; j < k; ++j) {
                    rc = (rc << 2) | COMP[tmp & 3];
                    tmp >>= 2;
                }
                rc &= mask;
                uint64_t canon = (kmer < rc) ? kmer : rc;
                if (query(canon)) ++hits;
            }
        }
        return hits;
    }

    /// Save filter to binary file
    /// Format: [MAGIC:u32][VERSION:u32][n_bits:u64][bits:n_bits/8 bytes]
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("Cannot open for write: " + path);
        uint32_t magic   = MAGIC;
        uint32_t version = VERSION;
        std::fwrite(&magic,   4, 1, f);
        std::fwrite(&version, 4, 1, f);
        std::fwrite(&n_bits_, 8, 1, f);
        std::fwrite(bits_.data(), 8, bits_.size(), f);
        std::fclose(f);
    }

    /// Load filter from binary file (returns false on failure)
    bool load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t magic = 0, version = 0;
        if (std::fread(&magic,   4, 1, f) != 1 || magic   != MAGIC)   { std::fclose(f); return false; }
        if (std::fread(&version, 4, 1, f) != 1 || version != VERSION) { std::fclose(f); return false; }
        if (std::fread(&n_bits_, 8, 1, f) != 1) { std::fclose(f); return false; }
        bits_.resize(n_bits_ / 64);
        if (std::fread(bits_.data(), 8, bits_.size(), f) != bits_.size()) {
            std::fclose(f); return false;
        }
        std::fclose(f);
        return true;
    }

    bool empty() const { return bits_.empty(); }

private:
    uint64_t              n_bits_ = 0;
    std::vector<uint64_t> bits_;
};

} // namespace singlet_pileup
