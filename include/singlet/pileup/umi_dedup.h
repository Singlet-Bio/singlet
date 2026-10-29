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

/// Union every directional edge of a (barcode, gene) group.
///
/// `packed[r]` / `count[r]` are the UMIs in rank order (count descending, ties in any
/// fixed order). Edge r -> t exists when r < t, hamming(r, t) == 1 and
/// count[r] >= 2 * count[t] - 1; `parent` (union-find, size n, initialised to the
/// identity by the caller) ends up holding the connected components, so the result
/// does not depend on the order edges are visited. Small groups test every pair;
/// big ones (deep genes: MALAT1, mitochondrial, ribosomal) look the 3 * umi_len
/// single-base neighbours of each UMI up in a packed-sorted index, O(n L log n)
/// instead of O(n^2).
template <typename CountT>
inline void umi_directional_union(const uint64_t* packed, const CountT* count, int n,
                                  int umi_len, std::vector<int>& parent) {
    auto find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto join = [&](int i, int j) {
        int ri = find(i), rj = find(j);
        if (ri != rj) parent[rj] = ri;
    };
    constexpr int kBrute = 512;
    if (n <= kBrute || umi_len > 32) {
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                if (umi_hamming_2bit(packed[i], packed[j], umi_len) == 1 &&
                    count[i] >= 2 * count[j] - 1)
                    join(i, j);
        return;
    }
    const uint64_t mask = umi_len < 32 ? (1ULL << (2 * umi_len)) - 1 : ~0ULL;
    std::vector<std::pair<uint64_t, int>> index(n);  // (masked packed, rank)
    for (int r = 0; r < n; ++r) index[r] = {packed[r] & mask, r};
    std::sort(index.begin(), index.end());
    for (int i = 0; i < n - 1; ++i) {
        const uint64_t u = packed[i] & mask;
        for (int p = 0; p < umi_len; ++p) {
            const uint64_t cur = (u >> (2 * p)) & 3ULL;
            for (uint64_t b = 0; b < 4; ++b) {
                if (b == cur) continue;
                const uint64_t cand = u ^ ((cur ^ b) << (2 * p));
                auto it = std::lower_bound(index.begin(), index.end(),
                                           std::make_pair(cand, -1));
                // several ranks can share a masked value only if umi_len is shorter
                // than the packed UMIs; visit all of them like the pairwise loop does
                for (; it != index.end() && it->first == cand; ++it) {
                    const int j = it->second;
                    if (j > i && count[i] >= 2 * count[j] - 1) join(i, j);
                }
            }
        }
    }
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

    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<uint64_t> pk(n);
    std::vector<uint32_t> ct(n);
    for (int i = 0; i < n; ++i) { pk[i] = umi_counts[i].first; ct[i] = umi_counts[i].second; }
    umi_directional_union(pk.data(), ct.data(), n, umi_len, parent);
    auto find = [&](int x) -> int {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };

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
        // Kept sorted by packed UMI: binary search, so deep genes with thousands of
        // UMIs per cell stay O(log n) per read.
        auto it = std::lower_bound(group.begin(), group.end(), umi_packed,
            [](const UmiEntry& e, uint64_t v) { return e.packed < v; });
        if (it != group.end() && it->packed == umi_packed) { it->count++; return; }
        group.insert(it, {umi_packed, exon_idx, 1u});
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
                [](const UmiEntry& a, const UmiEntry& b) {
                    return a.count != b.count ? a.count > b.count : a.packed < b.packed;
                });

            const int n = static_cast<int>(sorted.size());
            std::vector<int> parent(n);
            std::iota(parent.begin(), parent.end(), 0);
            {
                std::vector<uint64_t> pk(n);
                std::vector<uint32_t> ct(n);
                for (int i = 0; i < n; ++i) { pk[i] = sorted[i].packed; ct[i] = sorted[i].count; }
                umi_directional_union(pk.data(), ct.data(), n, umi_len, parent);
            }
            auto find = [&](int x) -> int {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            };

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

    // ── Molecule labelling: spliced / unspliced / ambiguous ──────────────────
    //
    // record() above keeps one exon index per UMI and is used once per feature
    // type, so a molecule whose reads touch both an exon and an intron of a gene
    // used to be counted once in the exon matrix AND once in the intron matrix.
    // record_read() keeps every read of a (barcode, gene, UMI) together with
    // what that read touched; finalize_molecules() then labels each molecule
    // exactly once (velocyto / STARsolo Velocyto rules on the collapsed gene
    // model, see docs/pipeline-validation.md):
    //   spliced    every read lies in exons only
    //   unspliced  every read touches an intron (intronic or exon-intron spanning)
    //   ambiguous  both kinds of reads
    // and writes spliced + ambiguous to the exon matrix, unspliced to the intron
    // matrix, so exon + intron counts every molecule once (= STARsolo GeneFull)
    // and the exon matrix alone stays STARsolo Gene-like.

    /// Read evidence bits passed to record_read().
    static constexpr uint8_t kReadExonOnly    = 1;  ///< all aligned bases in exons of the gene
    static constexpr uint8_t kReadIntron      = 2;  ///< some aligned bases in an intron of the gene
    static constexpr uint8_t kReadTouchesExon = 4;  ///< some aligned bases in an exon of the gene

    enum class MoleculeLabel : uint8_t { Spliced, Unspliced, Ambiguous };

    static MoleculeLabel label_of(uint8_t flags) {
        const bool e = (flags & kReadExonOnly) != 0;
        const bool i = (flags & kReadIntron) != 0;
        if (e && i) return MoleculeLabel::Ambiguous;
        return i ? MoleculeLabel::Unspliced : MoleculeLabel::Spliced;
    }

    struct MoleculeStats {
        uint64_t spliced = 0;
        uint64_t unspliced = 0;
        uint64_t ambiguous = 0;
        /// Molecules the pre-fix engine counted in both matrices: some read
        /// touched an exon and some read touched an intron of the same gene.
        uint64_t legacy_double = 0;
        uint64_t total() const { return spliced + unspliced + ambiguous; }
        MoleculeStats& operator+=(const MoleculeStats& o) {
            spliced += o.spliced; unspliced += o.unspliced;
            ambiguous += o.ambiguous; legacy_double += o.legacy_double;
            return *this;
        }
    };

    /// Record one read of a molecule. exon_idx / intron_idx are the features the
    /// read hit (UINT32_MAX if none); flags are kRead* bits.
    void record_read(uint32_t bc_idx, uint32_t gene_idx, uint64_t umi_packed,
                     uint32_t exon_idx, uint32_t intron_idx, uint8_t flags) {
        uint64_t key = (static_cast<uint64_t>(bc_idx) << 32) | gene_idx;
        auto& group = mol_groups_[key];
        auto it = std::lower_bound(group.begin(), group.end(), umi_packed,
            [](const MolEntry& e, uint64_t v) { return e.packed < v; });  // sorted by packed
        if (it != group.end() && it->packed == umi_packed) {
            it->count++;
            it->flags |= flags;
            // smallest index, not first seen: read order (STAR threads) must not matter
            it->exon_idx = std::min(it->exon_idx, exon_idx);
            it->intron_idx = std::min(it->intron_idx, intron_idx);
            return;
        }
        group.insert(it, {umi_packed, exon_idx, intron_idx, 1u, flags});
    }

    /// True if (bc, gene) already has molecules here (used to route reads that
    /// were resolved outside the store's worker to the right store).
    bool has_group(uint32_t bc_idx, uint32_t gene_idx) const {
        return mol_groups_.count((static_cast<uint64_t>(bc_idx) << 32) | gene_idx) != 0;
    }

    /// Collapse UMIs (directional 1-Hamming when `directional`, exact otherwise),
    /// label each molecule once and emit it.
    ///   exon_acc.increment(exon_idx, bc)      spliced and ambiguous molecules
    ///   intron_acc.increment(intron_idx, bc)  unspliced molecules
    ///   on_ambiguous(bc, gene)                 ambiguous molecules (also in exon_acc)
    template <typename ExonAccT, typename IntronAccT, typename AmbFn>
    MoleculeStats finalize_molecules(ExonAccT& exon_acc, IntronAccT& intron_acc,
                                     AmbFn&& on_ambiguous, int umi_len,
                                     bool directional) const {
        MoleculeStats st;
        std::vector<MolEntry> sorted;
        std::vector<int> parent;
        std::vector<uint64_t> pk;
        std::vector<uint32_t> ct;
        std::vector<uint8_t> comp_flags;
        std::vector<uint32_t> comp_exon, comp_intron;

        auto emit = [&](uint32_t bc, uint32_t gene, uint8_t flags,
                        uint32_t exon_idx, uint32_t intron_idx) {
            if ((flags & kReadTouchesExon) && (flags & kReadIntron)) st.legacy_double++;
            switch (label_of(flags)) {
                case MoleculeLabel::Spliced:
                    exon_acc.increment(exon_idx, bc); st.spliced++; break;
                case MoleculeLabel::Unspliced:
                    intron_acc.increment(intron_idx, bc); st.unspliced++; break;
                case MoleculeLabel::Ambiguous:
                    exon_acc.increment(exon_idx, bc); on_ambiguous(bc, gene);
                    st.ambiguous++; break;
            }
        };

        for (const auto& [key, entries] : mol_groups_) {
            const uint32_t bc = static_cast<uint32_t>(key >> 32);
            const uint32_t gene = static_cast<uint32_t>(key & 0xffffffffu);

            if (!directional || entries.size() == 1) {
                for (const auto& e : entries)
                    emit(bc, gene, e.flags, e.exon_idx, e.intron_idx);
                continue;
            }

            sorted.assign(entries.begin(), entries.end());
            std::sort(sorted.begin(), sorted.end(),
                [](const MolEntry& a, const MolEntry& b) {
                    return a.count != b.count ? a.count > b.count : a.packed < b.packed;
                });
            const int n = static_cast<int>(sorted.size());
            parent.resize(n);
            std::iota(parent.begin(), parent.end(), 0);
            pk.resize(n); ct.resize(n);
            for (int i = 0; i < n; ++i) { pk[i] = sorted[i].packed; ct[i] = sorted[i].count; }
            umi_directional_union(pk.data(), ct.data(), n, umi_len, parent);
            auto find = [&](int x) -> int {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            };

            // A molecule's evidence is the union over every UMI merged into it.
            comp_flags.assign(n, 0);
            comp_exon.assign(n, UINT32_MAX);
            comp_intron.assign(n, UINT32_MAX);
            for (int i = 0; i < n; ++i) {   // smallest feature index over the merged UMIs
                int r = find(i);
                comp_flags[r] |= sorted[i].flags;
                comp_exon[r] = std::min(comp_exon[r], sorted[i].exon_idx);
                comp_intron[r] = std::min(comp_intron[r], sorted[i].intron_idx);
            }
            for (int i = 0; i < n; ++i)
                if (find(i) == i) emit(bc, gene, comp_flags[i], comp_exon[i], comp_intron[i]);
        }
        return st;
    }

    size_t n_molecule_groups() const { return mol_groups_.size(); }
    bool   molecules_empty()   const { return mol_groups_.empty(); }
    void   clear_molecules()         { mol_groups_.clear(); }

private:
    struct MolEntry {
        uint64_t packed;      ///< 2-bit packed UMI
        uint32_t exon_idx;    ///< first exon a read of this UMI hit
        uint32_t intron_idx;  ///< first intron a read of this UMI hit
        uint32_t count;       ///< reads with this exact UMI
        uint8_t  flags;       ///< OR of kRead* over those reads
    };
    std::unordered_map<uint64_t, std::vector<MolEntry>> mol_groups_;

public:

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

    /// per_cell_stats() over the molecule store (record_read()).
    std::unordered_map<uint32_t, PerCellUmiStats> per_cell_molecule_stats(int umi_len) const {
        std::unordered_map<uint32_t, PerCellUmiStats> result;
        std::vector<std::pair<uint64_t, uint32_t>> umi_counts;
        for (const auto& [key, entries] : mol_groups_) {
            auto& s = result[static_cast<uint32_t>(key >> 32)];
            umi_counts.clear();
            for (const auto& e : entries) {
                s.total_reads += e.count;
                umi_counts.push_back({e.packed, e.count});
            }
            std::sort(umi_counts.begin(), umi_counts.end(),
                [](const auto& a, const auto& b) { return a.second > b.second; });
            s.unique_umis += directional_dedup(umi_counts, umi_len);
        }
        return result;
    }
};

} // namespace singlet
