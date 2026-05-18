// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: umi_dedup.h
// UMI deduplication for gene/exon/intron counting.
//
// For each (barcode, gene, UMI) triple, count only once.
// SNP counting is NOT UMI-deduped (needs raw depth for AD/DP).
//
// Design: hash-based dedup using 64-bit hash of (barcode_idx, gene_idx, UMI string).
// Memory: ~16 bytes per unique UMI × ~5M UMIs = ~80MB typical.
//
// Directional UMI error correction (N6, Smith et al. 2017):
//   DirectionalUmiStore accumulates per-(cell, gene) UMI observations during BAM
//   reading, then applies directional 1-Hamming graph clustering at finalize().
//   Each connected component = one true molecule.

#include <algorithm>     // std::sort, std::iota
#include <cstdint>
#include <numeric>       // std::iota
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>       // std::pair
#include <vector>

namespace singlet {

class UmiDedup {
public:
    UmiDedup() = default;

    // Check if this (barcode, gene, UMI) has been seen before.
    // Returns true if this is a NEW UMI (should count), false if duplicate.
    bool insert(uint32_t barcode_idx, uint32_t gene_idx, const char* umi, int umi_len) {
        uint64_t h = hash_triple(barcode_idx, gene_idx, umi, umi_len);
        auto [_, inserted] = seen_.insert(h);
        if (inserted) n_unique_++;
        else n_duplicate_++;
        return inserted;
    }

    // Simplified version for string UMIs
    bool insert(uint32_t barcode_idx, uint32_t gene_idx, const std::string& umi) {
        return insert(barcode_idx, gene_idx, umi.c_str(), static_cast<int>(umi.size()));
    }

    // Clear all state (for per-chromosome processing, not needed for streaming)
    void clear() {
        seen_.clear();
        n_unique_ = 0;
        n_duplicate_ = 0;
    }

    // Reserve capacity hint
    void reserve(size_t n) { seen_.reserve(n); }

    // Stats
    uint64_t n_unique() const { return n_unique_; }
    uint64_t n_duplicate() const { return n_duplicate_; }
    size_t size() const { return seen_.size(); }

private:
    std::unordered_set<uint64_t> seen_;
    uint64_t n_unique_ = 0;
    uint64_t n_duplicate_ = 0;

    // Hash a (barcode_idx, gene_idx, UMI) triple into 64 bits.
    // Uses FNV-1a for the UMI string, then mixes with barcode and gene.
    static uint64_t hash_triple(uint32_t barcode_idx, uint32_t gene_idx,
                                 const char* umi, int umi_len) {
        // FNV-1a hash of UMI string
        uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < umi_len; ++i) {
            h ^= static_cast<uint64_t>(umi[i]);
            h *= 1099511628211ULL;
        }

        // Mix in barcode and gene indices
        h ^= static_cast<uint64_t>(barcode_idx) * 0x517cc1b727220a95ULL;
        h ^= static_cast<uint64_t>(gene_idx) * 0x6c62272e07bb0142ULL;

        // Final avalanche
        h ^= h >> 33;
        h *= 0xff51afd7ed558ccdULL;
        h ^= h >> 33;
        h *= 0xc4ceb9fe1a85ec53ULL;
        h ^= h >> 33;

        return h;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// N6: Directional 1-Hamming UMI error correction (Smith et al. 2017)
// ─────────────────────────────────────────────────────────────────────────────

/// Pack a UMI string to 2-bit representation for fast XOR-based Hamming.
/// A=0, C=1, G=2, T/N=3.  Little-endian: first base stored at LSBs.
/// Supports up to 32-mer UMIs (64 bits total).
inline uint64_t umi_pack_2bit(const char* umi, int len) {
    uint64_t packed = 0;
    for (int i = 0; i < len && i < 32; ++i) {
        uint64_t b;
        char c = umi[i] | 32;   // to lowercase
        if      (c == 'a') b = 0;
        else if (c == 'c') b = 1;
        else if (c == 'g') b = 2;
        else               b = 3;  // t, n, or other
        packed |= (b << (i * 2));
    }
    return packed;
}

/// Count Hamming distance between two 2-bit-packed UMIs of given length.
/// Uses XOR + popcount: each non-zero 2-bit pair = 1 substitution.
inline int umi_hamming_2bit(uint64_t a, uint64_t b, int umi_len) {
    uint64_t diff = a ^ b;
    // Mask to valid UMI bits only
    if (umi_len < 32) diff &= (1ULL << (2 * umi_len)) - 1;
    // Count non-zero 2-bit pairs: set low bit of each pair if either bit differs
    return __builtin_popcountll((diff | (diff >> 1)) & 0x5555555555555555ULL);
}

/// Directional UMI deduplication (Smith et al. 2017).
///
/// Builds a directed adjacency graph on UMIs: edge A→B if
///   hamming(A,B) == 1  AND  count(A) >= 2 * count(B) - 1
/// Returns the number of connected components (= deduplicated molecule count).
///
/// @param umi_counts  (packed_umi, count) pairs, sorted by count descending
/// @param umi_len     UMI length in bases (for Hamming masking)
inline uint32_t directional_dedup(
    const std::vector<std::pair<uint64_t, uint32_t>>& umi_counts,
    int umi_len)
{
    const int n = static_cast<int>(umi_counts.size());
    if (n <= 1) return static_cast<uint32_t>(n);

    // Union-Find with path compression (two-step halving)
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);

    auto find = [&](int x) -> int {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];  // path halving
            x = parent[x];
        }
        return x;
    };

    // Build edges: umi_counts is sorted descending, so umi_counts[i].second >= umi_counts[j].second
    // Directional edge i→j (i "absorbs" j) when hamming=1 AND count_i >= 2*count_j - 1
    for (int i = 0; i < n - 1; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (umi_hamming_2bit(umi_counts[i].first, umi_counts[j].first, umi_len) == 1) {
                if (umi_counts[i].second >= 2 * umi_counts[j].second - 1) {
                    int ri = find(i), rj = find(j);
                    if (ri != rj) parent[rj] = ri;  // root of j → root of i
                }
            }
        }
    }

    // Count distinct component roots
    int roots = 0;
    for (int i = 0; i < n; ++i) if (find(i) == i) roots++;
    return static_cast<uint32_t>(roots);
}

/// Per-(barcode, gene) UMI accumulator for directional error correction.
///
/// During BAM reading: record(bc_idx, gene_idx, exon_idx, umi_packed) accumulates
/// UMIs per cell×gene group.  After BAM loop: finalize(acc, umi_len) applies
/// directional dedup and emits corrected counts to the sparse accumulator.
///
/// Memory: one UmiEntry (16 bytes) per unique (bc, gene, umi) observation.
/// Typical: ~12M entries × 16B = ~192 MB for a 40M-read 10x v3 run.
///
/// The finalize() method is templated so umi_dedup.h does not need to depend on
/// sparse_accumulator.h: any type AccT with increment(uint32_t, uint32_t) works.
class DirectionalUmiStore {
    struct UmiEntry {
        uint64_t packed;    ///< 2-bit packed UMI
        uint32_t exon_idx;  ///< exon this molecule maps to (for accumulator write)
        uint32_t count;     ///< number of reads with this exact UMI
    };

    // key = (uint64_t(bc_idx) << 32) | gene_idx  →  per-group UMI list
    std::unordered_map<uint64_t, std::vector<UmiEntry>> groups_;

public:
    /// Record a single UMI observation for a (barcode, gene) group.
    void record(uint32_t bc_idx, uint32_t gene_idx, uint32_t exon_idx, uint64_t umi_packed) {
        uint64_t key = (static_cast<uint64_t>(bc_idx) << 32) | gene_idx;
        auto& group = groups_[key];
        // Linear search (groups are small: typically 1-10 distinct UMIs)
        for (auto& e : group) {
            if (e.packed == umi_packed) {
                e.count++;
                return;
            }
        }
        group.push_back({umi_packed, exon_idx, 1u});
    }

    /// Apply directional dedup for all groups, emit corrected counts to accumulator.
    ///
    /// For each (bc, gene) group:
    ///   1. Sort UMIs by count descending
    ///   2. Apply directional Union-Find
    ///   3. Each component root emits 1 count to its exon in the accumulator
    ///
    /// @param acc      Sparse accumulator (SparseAccumulator<uint16_t> or similar)
    /// @param umi_len  UMI length in bases (for Hamming distance)
    /// @returns        Total number of deduplicated molecules emitted
    template <typename AccT>
    uint64_t finalize(AccT& acc, int umi_len) const {
        uint64_t total_deduped = 0;

        for (const auto& [key, entries] : groups_) {
            const uint32_t bc_idx = static_cast<uint32_t>(key >> 32);

            if (entries.size() == 1) {
                // Fast path: single UMI — no dedup needed
                acc.increment(entries[0].exon_idx, bc_idx);
                total_deduped++;
                continue;
            }

            // Sort by count descending (copy for sort)
            std::vector<UmiEntry> sorted(entries.begin(), entries.end());
            std::sort(sorted.begin(), sorted.end(),
                [](const UmiEntry& a, const UmiEntry& b) { return a.count > b.count; });

            // Union-Find
            const int n = static_cast<int>(sorted.size());
            std::vector<int> parent(n);
            std::iota(parent.begin(), parent.end(), 0);

            auto find = [&](int x) -> int {
                while (parent[x] != x) {
                    parent[x] = parent[parent[x]];
                    x = parent[x];
                }
                return x;
            };

            // Build directed edges: sorted[i] has higher count → absorbs sorted[j]
            for (int i = 0; i < n - 1; ++i) {
                for (int j = i + 1; j < n; ++j) {
                    if (umi_hamming_2bit(sorted[i].packed, sorted[j].packed, umi_len) == 1 &&
                        sorted[i].count >= 2 * sorted[j].count - 1) {
                        int ri = find(i), rj = find(j);
                        if (ri != rj) parent[rj] = ri;
                    }
                }
            }

            // Emit one count per component root (using root's exon assignment)
            for (int i = 0; i < n; ++i) {
                if (find(i) == i) {
                    acc.increment(sorted[i].exon_idx, bc_idx);
                    total_deduped++;
                }
            }
        }

        return total_deduped;
    }

    void clear()               { groups_.clear(); }
    size_t n_groups()   const  { return groups_.size(); }
    bool   empty()      const  { return groups_.empty(); }

    // ── N7: Per-cell UMI statistics for sequencing saturation ────────────────
    struct PerCellUmiStats {
        uint64_t total_reads = 0;   ///< Raw read count (pre-dedup)
        uint64_t unique_umis = 0;   ///< Deduplicated molecule count
    };

    /// Compute per-cell total reads and unique UMI counts.
    /// Applies the same directional dedup as finalize() — no extra BAM pass needed.
    /// @param umi_len  UMI length in bases (for Hamming distance masking)
    /// @returns        Map from barcode_index → (total_reads, unique_umis)
    std::unordered_map<uint32_t, PerCellUmiStats> per_cell_stats(int umi_len) const {
        std::unordered_map<uint32_t, PerCellUmiStats> result;
        result.reserve(groups_.size());

        for (const auto& [key, entries] : groups_) {
            const uint32_t bc_idx = static_cast<uint32_t>(key >> 32);
            auto& s = result[bc_idx];

            for (const auto& e : entries) s.total_reads += e.count;

            if (entries.size() == 1) {
                s.unique_umis++;
            } else {
                std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
                umi_counts.reserve(entries.size());
                for (const auto& e : entries)
                    umi_counts.push_back({e.packed, e.count});
                std::sort(umi_counts.begin(), umi_counts.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
                s.unique_umis += directional_dedup(umi_counts, umi_len);
            }
        }
        return result;
    }
};

} // namespace singlet
