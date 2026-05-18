// SPDX-License-Identifier: MIT
// lib1fq/folding.h — R2 sequence folding for alignment cache (Phase 4)
//
// Builds a mapping from read_idx → unique_seq_id based on R2 cDNA identity.
// During alignment, only unique R2 sequences need to be aligned; the results
// are cached and replayed for all reads sharing the same R2.

#pragma once

#include "reader.h"

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace singlet::fq {

// ── R2 sequence fold map ──
// Maps each read in a block to its unique R2 sequence ID.
// Unique R2s are emitted for alignment; results are replayed via the map.

struct FoldedBlock {
    uint32_t n_reads = 0;          // total reads in block
    uint32_t n_unique_r2 = 0;      // distinct R2 sequences

    // Per-read: index into unique_r2_* arrays
    std::vector<uint32_t> read_to_unique;   // [n_reads]

    // Per-unique-R2: the representative read index and accumulated info
    std::vector<uint32_t> unique_r2_rep;    // [n_unique_r2] representative read_idx
    std::vector<uint32_t> unique_r2_count;  // [n_unique_r2] how many reads share this R2

    // Alignment cache (filled after STAR alignment)
    struct CachedAlignment {
        int32_t  ref_id = -1;        // chromosome / contig ID (-1 = unmapped)
        int32_t  pos = -1;           // 0-based position
        uint8_t  mapq = 0;           // mapping quality
        uint16_t n_cigar = 0;        // CIGAR operations count
        uint16_t flags = 4;          // SAM flags (4 = unmapped)
        int32_t  n_hits = 0;         // NH tag (multi-mapper count)
        int32_t  n_mismatches = 0;   // nM tag
        int32_t  align_score = 0;    // AS tag
        bool     valid = false;
    };
    std::vector<CachedAlignment> alignments; // [n_unique_r2]
};

class ReadFolder {
public:

    // Fold a decoded block: build read_to_unique mapping based on R2 identity
    static FoldedBlock fold(const DecodedBlock& blk) {
        FoldedBlock fb;
        fb.n_reads = blk.n_reads;
        fb.read_to_unique.resize(blk.n_reads);

        // Hash R2 sequences to find unique groups
        struct R2Key {
            uint64_t hash;
            uint32_t first_idx;  // representative
        };

        std::unordered_map<uint64_t, uint32_t> hash_to_uid;
        hash_to_uid.reserve(blk.n_reads);

        for (uint32_t i = 0; i < blk.n_reads; ++i) {
            uint64_t h = fnv1a(blk.r2_seq(i), blk.r2_len(i));

            auto it = hash_to_uid.find(h);
            if (it != hash_to_uid.end()) {
                // Check true equality (collision safety)
                uint32_t uid = it->second;
                uint32_t rep = fb.unique_r2_rep[uid];
                if (blk.r2_len(i) == blk.r2_len(rep) &&
                    std::memcmp(blk.r2_seq(i), blk.r2_seq(rep),
                                blk.r2_len(i)) == 0) {
                    fb.read_to_unique[i] = uid;
                    fb.unique_r2_count[uid]++;
                    continue;
                }
                // Hash collision — fall through to create new unique
            }

            uint32_t uid = static_cast<uint32_t>(fb.unique_r2_rep.size());
            hash_to_uid[h] = uid;
            fb.unique_r2_rep.push_back(i);
            fb.unique_r2_count.push_back(1);
            fb.read_to_unique[i] = uid;
        }

        fb.n_unique_r2 = static_cast<uint32_t>(fb.unique_r2_rep.size());
        fb.alignments.resize(fb.n_unique_r2);

        return fb;
    }

    // Get the fold ratio for a block
    static double fold_ratio(const FoldedBlock& fb) {
        return (fb.n_unique_r2 > 0) ?
            static_cast<double>(fb.n_reads) / fb.n_unique_r2 : 1.0;
    }

private:
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    static uint64_t fnv1a(const uint8_t* data, size_t len) {
        uint64_t h = FNV_OFFSET;
        for (size_t i = 0; i < len; ++i) {
            h ^= data[i];
            h *= FNV_PRIME;
        }
        return h;
    }
};

// ── Alignment Result Cache ──
// Global cache across blocks: maps R2 hash → alignment result.
// If the same R2 appears in multiple blocks, the cached result is reused.

class AlignmentCache {
public:
    using CachedAlignment = FoldedBlock::CachedAlignment;

    // Check if an R2 sequence has a cached alignment
    bool has(const uint8_t* r2_seq, uint16_t r2_len) const {
        uint64_t h = hash(r2_seq, r2_len);
        return cache_.count(h) > 0;
    }

    // Get cached alignment (returns nullptr if not found)
    const CachedAlignment* get(const uint8_t* r2_seq, uint16_t r2_len) const {
        uint64_t h = hash(r2_seq, r2_len);
        auto it = cache_.find(h);
        return (it != cache_.end()) ? &it->second : nullptr;
    }

    // Store alignment result for an R2 sequence
    void put(const uint8_t* r2_seq, uint16_t r2_len,
             const CachedAlignment& aln) {
        uint64_t h = hash(r2_seq, r2_len);
        cache_[h] = aln;
    }

    size_t size() const { return cache_.size(); }
    void clear() { cache_.clear(); }

    // Estimated memory usage
    size_t memory_bytes() const {
        return cache_.size() * (sizeof(uint64_t) + sizeof(CachedAlignment) + 32);
    }

private:
    static uint64_t hash(const uint8_t* data, size_t len) {
        uint64_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < len; ++i) {
            h ^= data[i];
            h *= 1099511628211ULL;
        }
        return h;
    }

    std::unordered_map<uint64_t, CachedAlignment> cache_;
};

} // namespace singlet::fq
