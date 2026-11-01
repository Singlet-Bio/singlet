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
#include <cstdlib>
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

/// The pair loop `for i < j: if hamming(i, j) == 1 && cond(i, j) join(i, j)` over n 2-bit packed UMIs, visiting the same
/// edges in the same (i, then j ascending) order, so an order-dependent union-find ends in the same roots. Groups above
/// 512 UMIs find the edges through a sorted index of the 3 * umi_len single-base neighbours of each UMI, O(n L log n).
template <typename PackedAt, typename CondFn, typename JoinFn>
inline void umi_directional_union_ordered(int n, int umi_len, PackedAt packed, CondFn cond, JoinFn join) {
    constexpr int kBrute = 512;
    if (n <= kBrute || umi_len > 32) {
        for (int i = 0; i < n - 1; ++i)
            for (int j = i + 1; j < n; ++j)
                if (umi_hamming_2bit(packed(i), packed(j), umi_len) == 1 && cond(i, j)) join(i, j);
        return;
    }
    const uint64_t mask = umi_len < 32 ? (1ULL << (2 * umi_len)) - 1 : ~0ULL;
    std::vector<std::pair<uint64_t, int>> index(n);
    for (int r = 0; r < n; ++r) index[r] = {packed(r) & mask, r};
    std::sort(index.begin(), index.end());
    std::vector<int> nb;
    for (int i = 0; i < n - 1; ++i) {
        const uint64_t u = packed(i) & mask;
        nb.clear();
        for (int p = 0; p < umi_len; ++p) {
            const uint64_t cur = (u >> (2 * p)) & 3ULL;
            for (uint64_t b = 0; b < 4; ++b) {
                if (b == cur) continue;
                const uint64_t cand = u ^ ((cur ^ b) << (2 * p));
                for (auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(cand, -1));
                     it != index.end() && it->first == cand; ++it)
                    if (it->second > i && cond(i, it->second)) nb.push_back(it->second);
            }
        }
        std::sort(nb.begin(), nb.end());
        for (int j : nb) join(i, j);
    }
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

/// Seed-based directional collapse (UMI-tools "directional", Smith et al. 2017), made independent of read order.
///
/// `packed[r]` / `count[r]` are in rank order (count descending, packed UMI ascending: a total order of the UMI set).
/// Seeds are taken in rank order; a UMI that no earlier seed has claimed becomes a seed and claims every unclaimed UMI
/// reachable from it through directional edges (hamming 1, count[u] >= 2 count[v] - 1). `owner[r]` is the rank of the
/// seed that claimed r. Because the rank order is total and fixed by the (UMI, reads) multiset, the result depends on
/// nothing else. Unlike the connected components of umi_directional_union, two well-supported UMIs 2 mismatches apart
/// stay two molecules when a one-read error copy sits between them (the component rule merges all three), and the
/// molecule's UMI is its best-supported UMI, not whichever union-find root the edge order happened to leave.
template <typename CountT>
inline void umi_directional_seeds(const uint64_t* packed, const CountT* count, int n, int umi_len, std::vector<int>& owner) {
    owner.assign(n, -1);
    constexpr int kBrute = 512;
    const bool brute = n <= kBrute || umi_len > 32;
    const uint64_t mask = umi_len < 32 ? (1ULL << (2 * umi_len)) - 1 : ~0ULL;
    std::vector<std::pair<uint64_t, int>> index;
    if (!brute) {
        index.resize(n);
        for (int r = 0; r < n; ++r) index[r] = {packed[r] & mask, r};
        std::sort(index.begin(), index.end());
    }
    std::vector<int> stack;
    auto visit = [&](int u, int seed) {
        if (brute) {
            for (int v = 0; v < n; ++v)
                if (owner[v] < 0 && umi_hamming_2bit(packed[u], packed[v], umi_len) == 1 && count[u] >= 2 * count[v] - 1) {
                    owner[v] = seed; stack.push_back(v);
                }
            return;
        }
        const uint64_t x = packed[u] & mask;
        for (int p = 0; p < umi_len; ++p) {
            const uint64_t cur = (x >> (2 * p)) & 3ULL;
            for (uint64_t b = 0; b < 4; ++b) {
                if (b == cur) continue;
                const uint64_t cand = x ^ ((cur ^ b) << (2 * p));
                for (auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(cand, -1));
                     it != index.end() && it->first == cand; ++it) {
                    const int v = it->second;
                    if (owner[v] < 0 && count[u] >= 2 * count[v] - 1) { owner[v] = seed; stack.push_back(v); }
                }
            }
        }
    };
    for (int i = 0; i < n; ++i) {
        if (owner[i] >= 0) continue;
        owner[i] = i;
        stack.assign(1, i);
        while (!stack.empty()) { const int u = stack.back(); stack.pop_back(); visit(u, i); }
    }
}

/// UMI collapse rule of the molecule store: false = connected components of the directional graph (the default, as
/// always); true = umi_directional_seeds. SINGLET_UMI_COLLAPSE=seed switches it (read once).
inline bool umi_collapse_seed() {
    static const bool on = [] { const char* e = getenv("SINGLET_UMI_COLLAPSE"); return e && std::string(e) == "seed"; }();
    return on;
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
    // model, alevin-fry USA semantics since the long-read check of docs/research-beta/sota-expression.md):
    //   unspliced  at least one read touches an intron (intronic or exon-intron spanning)
    //   spliced    no intron-touching read and at least one read crosses a junction (CIGAR N)
    //   ambiguous  exon-only reads and no junction evidence (compatible with both)
    // (before: unspliced = every read touches an intron, ambiguous = reads of both kinds, spliced = all exonic; on PBMC-E
    // long-read truth that sent 43% of the truly unspliced molecules to ambiguous: unspliced recall 54% vs 94-98% for
    // STARsolo Velocyto and alevin-fry, and 100% of molecules with any intron-touching read are unspliced by the long read)
    // and writes spliced + ambiguous to the exon matrix, unspliced to the intron
    // matrix, so exon + intron counts every molecule once (= STARsolo GeneFull,
    // unchanged by the labels), the exon matrix alone stays STARsolo Gene-like, and
    // velocyto-style "spliced" counts are spliced + ambiguous.

    /// Read evidence bits passed to record_read().
    static constexpr uint8_t kReadExonOnly    = 1;  ///< all aligned bases in exons of the gene
    static constexpr uint8_t kReadIntron      = 2;  ///< some aligned bases in an intron of the gene
    static constexpr uint8_t kReadTouchesExon = 4;  ///< some aligned bases in an exon of the gene

    enum class MoleculeLabel : uint8_t { Spliced, Unspliced, Ambiguous };

    /// flags: OR of kRead* over the molecule's reads; diag: OR of kDiag* (only kDiagSpliced is read here).
    static MoleculeLabel label_of(uint8_t flags, uint8_t diag) {
        if (flags & kReadIntron) return MoleculeLabel::Unspliced;
        return (diag & kDiagSpliced) ? MoleculeLabel::Spliced : MoleculeLabel::Ambiguous;
    }

    struct MoleculeStats {
        uint64_t spliced = 0;
        uint64_t unspliced = 0;
        uint64_t ambiguous = 0;
        /// Molecules the pre-fix engine counted in both matrices: some read
        /// touched an exon and some read touched an intron of the same gene.
        uint64_t legacy_double = 0;
        /// Molecules not emitted (molecule-level intron rules, pileup_engine.h).
        uint64_t dropped = 0;
        uint64_t total() const { return spliced + unspliced + ambiguous; }
        MoleculeStats& operator+=(const MoleculeStats& o) {
            spliced += o.spliced; unspliced += o.unspliced;
            ambiguous += o.ambiguous; legacy_double += o.legacy_double;
            dropped += o.dropped;
            return *this;
        }
    };

    /// Diagnostic bits passed to record_read() (OR over a molecule's reads).
    /// They never change a label; they describe what the reads looked like so a
    /// molecule dump can be joined with other tools' calls (SINGLET_MOLECULE_DUMP).
    static constexpr uint8_t kDiagMulti      = 1;   ///< read had NH > 1
    static constexpr uint8_t kDiagSpliced    = 2;   ///< read has a CIGAR N gap
    static constexpr uint8_t kDiagFlank      = 4;   ///< intronic part of an exon-touching read <= 5 bp
    static constexpr uint8_t kDiagOtherGene  = 8;   ///< read also lies in another same-strand gene's exon/intron
    static constexpr uint8_t kDiagAntisense  = 16;  ///< read also overlaps an opposite-strand gene
    static constexpr uint8_t kDiagOutside    = 32;  ///< some aligned bases outside the gene body

    /// Record one read of a molecule. exon_idx / intron_idx are the features the
    /// read hit (UINT32_MAX if none); flags are kRead* bits, diag kDiag* bits.
    void record_read(uint32_t bc_idx, uint32_t gene_idx, uint64_t umi_packed,
                     uint32_t exon_idx, uint32_t intron_idx, uint8_t flags,
                     uint8_t diag = 0) {
        uint64_t key = (static_cast<uint64_t>(bc_idx) << 32) | gene_idx;
        auto& group = mol_groups_[key];
        auto it = std::lower_bound(group.begin(), group.end(), umi_packed,
            [](const MolEntry& e, uint64_t v) { return e.packed < v; });  // sorted by packed
        if (it != group.end() && it->packed == umi_packed) {
            it->count++;
            it->flags |= flags;
            it->diag |= diag;
            // smallest index, not first seen: read order (STAR threads) must not matter
            it->exon_idx = std::min(it->exon_idx, exon_idx);
            it->intron_idx = std::min(it->intron_idx, intron_idx);
            return;
        }
        group.insert(it, {umi_packed, exon_idx, intron_idx, 1u, flags, diag});
    }

    /// True if (bc, gene) already has molecules here (used to route reads that
    /// were resolved outside the store's worker to the right store).
    bool has_group(uint32_t bc_idx, uint32_t gene_idx) const {
        return mol_groups_.count((static_cast<uint64_t>(bc_idx) << 32) | gene_idx) != 0;
    }

    /// One collapsed molecule: the union of the evidence of every UMI merged into it.
    struct Molecule {
        uint64_t umi;         ///< 2-bit packed UMI of the component root
        uint32_t bc;
        uint32_t gene;
        uint32_t exon_idx;    ///< smallest exon index over the merged reads
        uint32_t intron_idx;  ///< smallest intron index over the merged reads
        uint32_t count;       ///< reads
        uint8_t  flags;       ///< OR of kRead*
        uint8_t  diag;        ///< OR of kDiag*
        MoleculeLabel label() const { return label_of(flags, diag); }
    };

    /// Collapse UMIs per (barcode, gene) (directional 1-Hamming when
    /// `directional`, exact otherwise) and append one Molecule per component.
    void collapse_molecules(int umi_len, bool directional, std::vector<Molecule>& out) const {
        collapse_groups<false>(mol_groups_, umi_len, directional, out);
    }

    /// collapse_molecules() that gives each (barcode, gene) group back as soon as it is collapsed, so the group
    /// store and its molecules never coexist in full (the finalize peak was store + molecules + the multi-gene
    /// table). Same molecules in the same order. Afterwards the store holds no molecule groups, and the
    /// per-cell saturation statistics (per_cell_molecule_stats()) are kept, computed before the first group goes.
    /// Nothing may read the groups after this (gene_root(), has_group(), collapse_molecules()).
    void consume_molecules(int umi_len, bool directional, std::vector<Molecule>& out) {
        if (mol_groups_.empty()) return;
        mol_cell_stats_ = per_cell_molecule_stats(umi_len);
        mol_stats_cached_ = true;
        collapse_groups<true>(mol_groups_, umi_len, directional, out);
        std::unordered_map<uint64_t, std::vector<MolEntry>>().swap(mol_groups_);
        root_cache_.clear();
    }

private:
    template <bool Consume, class Map>
    static void collapse_groups(Map& groups, int umi_len, bool directional, std::vector<Molecule>& out) {
        std::vector<MolEntry> sorted;
        std::vector<int> parent;
        std::vector<uint64_t> pk;
        std::vector<uint32_t> ct;
        std::vector<Molecule> comp;
        auto process = [&](uint64_t key, const std::vector<MolEntry>& entries) {
            const uint32_t bc = static_cast<uint32_t>(key >> 32);
            const uint32_t gene = static_cast<uint32_t>(key & 0xffffffffu);

            if (!directional || entries.size() == 1) {
                for (const auto& e : entries)
                    out.push_back({e.packed, bc, gene, e.exon_idx, e.intron_idx, e.count, e.flags, e.diag});
                return;
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
            if (umi_collapse_seed()) {
                umi_directional_seeds(pk.data(), ct.data(), n, umi_len, parent);   // parent[i] = the seed that claimed i (a seed is its own parent)
            } else {
                umi_directional_union(pk.data(), ct.data(), n, umi_len, parent);
            }
            auto find = [&](int x) -> int {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            };

            // A molecule's evidence is the union over every UMI merged into it
            // (smallest feature index over the merged UMIs).
            comp.assign(n, Molecule{0, bc, gene, UINT32_MAX, UINT32_MAX, 0, 0, 0});
            for (int i = 0; i < n; ++i) {
                int r = find(i);
                comp[r].flags |= sorted[i].flags;
                comp[r].diag |= sorted[i].diag;
                comp[r].count += sorted[i].count;
                comp[r].exon_idx = std::min(comp[r].exon_idx, sorted[i].exon_idx);
                comp[r].intron_idx = std::min(comp[r].intron_idx, sorted[i].intron_idx);
            }
            for (int i = 0; i < n; ++i)
                if (find(i) == i) { comp[i].umi = sorted[i].packed; out.push_back(comp[i]); }
        };
        if constexpr (Consume) {
            for (auto it = groups.begin(); it != groups.end();) {
                process(it->first, it->second);
                it = groups.erase(it);
            }
        } else {
            for (const auto& [key, entries] : groups) process(key, entries);
        }
    }

public:

    /// Emit one labelled molecule:
    ///   exon_acc.increment(exon_idx, bc)      spliced and ambiguous molecules
    ///   intron_acc.increment(intron_idx, bc)  unspliced molecules
    ///   on_ambiguous(bc, gene)                 ambiguous molecules (also in exon_acc)
    ///   on_reads(intronic, feature, bc, reads) every emitted molecule, with the reads merged into it
    ///                                          (the reads-per-UMI matrix and histogram; never touches the counts)
    struct NoReads { void operator()(bool, uint32_t, uint32_t, uint32_t) const {} };
    struct NoMixed { void operator()(const Molecule&) const {} };

    /// `on_mixed(m)` is called for an unspliced molecule that also has an exon-only read: the molecules the pre-USA labels
    /// counted in the exon matrix (as ambiguous, at m.exon_idx) rather than the intron matrix (at m.intron_idx). Every
    /// method validated on the pre-USA split (knee, ambient, doublets, sex, cell cycle, QC metrics) reads the legacy pair
    /// legacy_exon = exon + mixed_exon, legacy_intron = intron - mixed_intron, which the engine rebuilds from these
    /// molecules (mixed_exon()/mixed_intron()); only the published exon/intron/ambiguous layers carry the USA labels.
    template <typename ExonAccT, typename IntronAccT, typename AmbFn, typename ReadsFn = NoReads, typename MixFn = NoMixed>
    static void emit_molecule(const Molecule& m, ExonAccT& exon_acc, IntronAccT& intron_acc,
                              AmbFn&& on_ambiguous, MoleculeStats& st, ReadsFn&& on_reads = ReadsFn(), MixFn&& on_mixed = MixFn()) {
        if ((m.flags & kReadTouchesExon) && (m.flags & kReadIntron)) st.legacy_double++;
        switch (m.label()) {
            case MoleculeLabel::Spliced:
                exon_acc.increment(m.exon_idx, m.bc); on_reads(false, m.exon_idx, m.bc, m.count); st.spliced++; break;
            case MoleculeLabel::Unspliced:
                intron_acc.increment(m.intron_idx, m.bc); on_reads(true, m.intron_idx, m.bc, m.count); st.unspliced++;
                if (m.flags & kReadExonOnly) on_mixed(m);
                break;
            case MoleculeLabel::Ambiguous:
                exon_acc.increment(m.exon_idx, m.bc); on_ambiguous(m.bc, m.gene); on_reads(false, m.exon_idx, m.bc, m.count);
                st.ambiguous++; break;
        }
    }

    /// Collapse UMIs, label each molecule once and emit it (emit_molecule()).
    template <typename ExonAccT, typename IntronAccT, typename AmbFn, typename ReadsFn = NoReads, typename MixFn = NoMixed>
    MoleculeStats finalize_molecules(ExonAccT& exon_acc, IntronAccT& intron_acc,
                                     AmbFn&& on_ambiguous, int umi_len,
                                     bool directional, ReadsFn&& on_reads = ReadsFn(), MixFn&& on_mixed = MixFn()) const {
        MoleculeStats st;
        std::vector<Molecule> mols;
        collapse_molecules(umi_len, directional, mols);
        for (const auto& m : mols) emit_molecule(m, exon_acc, intron_acc, on_ambiguous, st, on_reads, on_mixed);
        return st;
    }

    /// R9 per (barcode, gene): the root (2-bit packed) of `packed` after the collapse finalize_molecules() applies to
    /// its (barcode, gene) group, i.e. over every read of that gene in the cell. False when the group does not hold
    /// the UMI. Roots are computed once per group and cached until clear_molecules().
    bool gene_root(uint32_t bc, uint32_t gene, uint64_t packed, int umi_len, bool directional, uint64_t& root) const {
        const uint64_t key = (static_cast<uint64_t>(bc) << 32) | gene;
        auto c = root_cache_.find(key);
        if (c == root_cache_.end()) {
            auto g = mol_groups_.find(key);
            if (g == mol_groups_.end()) return false;
            std::vector<MolEntry> sorted(g->second.begin(), g->second.end());
            std::sort(sorted.begin(), sorted.end(), [](const MolEntry& a, const MolEntry& b) {
                return a.count != b.count ? a.count > b.count : a.packed < b.packed;
            });
            const int n = static_cast<int>(sorted.size());
            std::vector<int> parent(n);
            std::iota(parent.begin(), parent.end(), 0);
            auto find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
            if (directional && umi_collapse_seed()) {
                std::vector<uint64_t> spk(n); std::vector<uint32_t> sct(n);
                for (int i = 0; i < n; ++i) { spk[i] = sorted[i].packed; sct[i] = sorted[i].count; }
                umi_directional_seeds(spk.data(), sct.data(), n, umi_len, parent);
            } else if (directional)
                umi_directional_union_ordered(n, umi_len, [&](int i) { return sorted[i].packed; },
                                              [&](int i, int j) { return sorted[i].count >= 2 * sorted[j].count - 1; },
                                              [&](int i, int j) {
                                                  int ri = find(i), rj = find(j);
                                                  if (ri != rj) parent[rj] = ri;
                                              });
            std::vector<std::pair<uint64_t, uint64_t>> roots(n);
            for (int i = 0; i < n; ++i) roots[i] = {sorted[i].packed, sorted[find(i)].packed};
            std::sort(roots.begin(), roots.end());
            c = root_cache_.emplace(key, std::move(roots)).first;
        }
        auto it = std::lower_bound(c->second.begin(), c->second.end(), std::make_pair(packed, uint64_t(0)));
        if (it == c->second.end() || it->first != packed) return false;
        root = it->second;
        return true;
    }

    size_t n_molecule_groups() const { return mol_groups_.size(); }
    bool   molecules_empty()   const { return mol_groups_.empty() && !mol_stats_cached_; }
    void   clear_molecules()         { mol_groups_.clear(); root_cache_.clear(); }

private:
    struct MolEntry {
        uint64_t packed;      ///< 2-bit packed UMI
        uint32_t exon_idx;    ///< first exon a read of this UMI hit
        uint32_t intron_idx;  ///< first intron a read of this UMI hit
        uint32_t count;       ///< reads with this exact UMI
        uint8_t  flags;       ///< OR of kRead* over those reads
        uint8_t  diag;        ///< OR of kDiag* over those reads
    };
    std::unordered_map<uint64_t, std::vector<MolEntry>> mol_groups_;
    mutable std::unordered_map<uint64_t, std::vector<std::pair<uint64_t, uint64_t>>> root_cache_;  ///< gene_root()

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
        if (mol_stats_cached_) return mol_cell_stats_;   // the groups were consumed (consume_molecules())
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

private:
    std::unordered_map<uint32_t, PerCellUmiStats> mol_cell_stats_;   ///< kept by consume_molecules()
    bool mol_stats_cached_ = false;
};

// ── R9: UMI correction for junction, V(D)J and SNP evidence (docs/research-beta/order-free-rules.md) ──
//
// UMIs are packed 3 bits per base (A C G T N other) behind a leading 1 bit that fixes the length, for UMIs of up
// to 21 bases: exact, no hashing, and two UMIs of one length differ at one base iff one 3-bit group differs.

inline uint64_t umi_pack3(const char* umi, int len) {
    uint64_t packed = 1;
    for (int i = 0; i < len; ++i) {
        uint64_t b;
        switch (umi[i] | 32) { case 'a': b = 0; break; case 'c': b = 1; break; case 'g': b = 2; break;
                               case 't': b = 3; break; case 'n': b = 4; break; default: b = 5; }
        packed = (packed << 3) | b;
    }
    return packed;
}

/// Same length and exactly one base differs (UMIs from umi_pack3).
inline bool umi3_one_mismatch(uint64_t a, uint64_t b) {
    if (a == b || __builtin_clzll(a) != __builtin_clzll(b)) return false;
    const uint64_t d = a ^ b;
    return __builtin_popcountll((d | (d >> 1) | (d >> 2)) & 0x1249249249249249ULL) == 1;
}

/// Directional 1-mismatch collapse of one (barcode, feature) group, the rule DirectionalUmiStore applies to exon and
/// intron molecules. Entries [0, n) must be sorted by read count descending, then packed UMI ascending (so the result
/// does not depend on read order). UMI i absorbs a later UMI j that differs at one base when count_i >= 2 count_j - 1.
/// On return root[i] is the component root of entry i; the number of roots is the number of molecules.
template <typename UmiAt, typename CountAt>
inline void umi3_directional_roots(size_t n, UmiAt umi, CountAt count, std::vector<uint32_t>& root) {
    root.resize(n);
    std::iota(root.begin(), root.end(), 0u);
    auto find = [&](uint32_t x) { while (root[x] != x) { root[x] = root[root[x]]; x = root[x]; } return x; };
    constexpr size_t kBrute = 512;
    auto join = [&](size_t i, size_t j) {
        const uint32_t ri = find(static_cast<uint32_t>(i)), rj = find(static_cast<uint32_t>(j));
        if (ri != rj) root[rj] = ri;
    };
    if (n <= kBrute) {
        for (size_t i = 0; i + 1 < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
                if (count(i) >= 2 * count(j) - 1 && umi3_one_mismatch(umi(i), umi(j))) join(i, j);
    } else {
        // Deep groups: the same joins in the same (i, then j ascending) order, found through a sorted index of the
        // 5-7 single-group neighbours of each UMI instead of the n^2 pair loop, so the roots are identical.
        std::vector<std::pair<uint64_t, uint32_t>> index(n);
        for (size_t r = 0; r < n; ++r) index[r] = {umi(r), static_cast<uint32_t>(r)};
        std::sort(index.begin(), index.end());
        std::vector<uint32_t> nb;
        for (size_t i = 0; i + 1 < n; ++i) {
            const uint64_t u = umi(i);
            const int len = (63 - __builtin_clzll(u)) / 3;
            nb.clear();
            for (int p = 0; p < len; ++p) {
                const uint64_t cur = (u >> (3 * p)) & 7ULL;
                for (uint64_t b = 0; b < 8; ++b) {
                    if (b == cur) continue;
                    const uint64_t cand = u ^ ((cur ^ b) << (3 * p));
                    for (auto it = std::lower_bound(index.begin(), index.end(), std::make_pair(cand, uint32_t(0)));
                         it != index.end() && it->first == cand; ++it)
                        if (it->second > i && count(i) >= 2 * count(it->second) - 1) nb.push_back(it->second);
                }
            }
            std::sort(nb.begin(), nb.end());
            for (uint32_t j : nb) join(i, j);
        }
    }
    for (size_t i = 0; i < n; ++i) root[i] = find(static_cast<uint32_t>(i));
}

/// umi_pack_2bit() of a UMI given in umi_pack3() form; *len receives its length.
inline uint64_t umi3_to_2bit(uint64_t p3, int* len) {
    const int n = (63 - __builtin_clzll(p3)) / 3;
    uint64_t out = 0;
    for (int i = 0; i < n && i < 32; ++i) {
        const uint64_t b = (p3 >> (3 * (n - 1 - i))) & 7;
        out |= (b < 3 ? b : 3) << (i * 2);
    }
    *len = n;
    return out;
}

/// R9 molecules of one (barcode, feature) group whose entries [0, n) are sorted as for umi3_directional_roots.
/// An entry whose reads were assigned a gene (gene(i) != UINT32_MAX) and whose UMI that gene's molecule store holds is
/// identified by gene_root(gene, umi, root): its root over every read of the gene in the cell. The other entries are
/// collapsed among themselves within the feature. On return id[i] in [0, k) numbers the molecules; returns k.
template <typename UmiAt, typename CountAt, typename GeneAt, typename GeneRoot>
inline uint32_t r9_molecules(size_t n, UmiAt umi, CountAt count, GeneAt gene, GeneRoot&& gene_root,
                             bool directional, std::vector<uint32_t>& id) {
    struct M { uint32_t gene; uint64_t key; uint32_t i; };
    std::vector<M> m;
    m.reserve(n);
    std::vector<uint32_t> f;   // entries collapsed within the feature
    for (size_t i = 0; i < n; ++i) {
        uint64_t r;
        const uint32_t g = gene(i);
        if (g != UINT32_MAX && gene_root(g, umi(i), r)) m.push_back({g, r, static_cast<uint32_t>(i)});
        else f.push_back(static_cast<uint32_t>(i));
    }
    std::vector<uint32_t> root;
    if (directional && f.size() > 1) {
        umi3_directional_roots(f.size(), [&](size_t k) { return umi(f[k]); }, [&](size_t k) { return count(f[k]); }, root);
    } else {
        root.resize(f.size());
        std::iota(root.begin(), root.end(), 0u);
    }
    for (size_t k = 0; k < f.size(); ++k) m.push_back({UINT32_MAX, umi(f[root[k]]), f[k]});
    std::sort(m.begin(), m.end(), [](const M& a, const M& b) {
        return a.gene != b.gene ? a.gene < b.gene : (a.key != b.key ? a.key < b.key : a.i < b.i);
    });
    id.assign(n, 0);
    uint32_t k = 0;
    for (size_t j = 0; j < m.size(); ++j) {
        if (j > 0 && (m[j].gene != m[j - 1].gene || m[j].key != m[j - 1].key)) ++k;
        id[m[j].i] = k;
    }
    return m.empty() ? 0 : k + 1;
}

/// gene_root for r9_molecules when no gene store is given: every entry is collapsed within its feature.
struct NoGeneRoot {
    bool operator()(uint32_t, uint32_t, uint64_t, uint64_t&) const { return false; }
};

/// Molecule counting for one feature kind (splice junctions, V(D)J genes) under R9: add() keeps the read count of
/// every exact (barcode, feature, UMI) during the pass, with the gene its unique reads were assigned to; finalize()
/// collapses each (barcode, feature) group (r9_molecules: per cell x gene where a gene is known, else per feature)
/// and emits its molecule count. Order-free: the pass only counts, and finalize sorts before collapsing.
class UmiFeatureCollapse {
public:
    void add(uint32_t bc_idx, uint32_t feature, const char* umi, int umi_len, uint32_t gene = UINT32_MAX) {
        const uint64_t g = (static_cast<uint64_t>(bc_idx) << 32) | feature;
        if (!umi || umi_len <= 0) { ++no_umi_[g]; return; }        // a read without a UMI is its own molecule
        if (umi_len <= 21) {
            Val& c = mols_[Key{g, umi_pack3(umi, umi_len)}];
            if (c.n != UINT32_MAX) ++c.n;
            c.gene = std::min(c.gene, gene);                      // smallest gene, not first seen
        } else {
            long_.insert(std::to_string(g) + ':' + std::string(umi, static_cast<size_t>(umi_len)));  // exact only
        }
    }

    /// emit(bc_idx, feature, molecules) once per (barcode, feature) group with molecules > 0. Returns the total.
    /// gene_root(bc, gene, umi3, root&): see r9_molecules (NoGeneRoot = per feature only).
    template <typename Emit, typename GeneRoot = NoGeneRoot>
    uint64_t finalize(Emit&& emit, bool directional, GeneRoot&& gene_root = GeneRoot()) const {
        struct E { uint64_t g, umi; uint32_t n, gene; };
        std::vector<E> v;
        v.reserve(mols_.size());
        for (const auto& kv : mols_) v.push_back({kv.first.g, kv.first.umi, kv.second.n, kv.second.gene});
        std::sort(v.begin(), v.end(), [](const E& a, const E& b) {
            if (a.g != b.g) return a.g < b.g;
            if (a.n != b.n) return a.n > b.n;
            return a.umi < b.umi;
        });
        std::unordered_map<uint64_t, uint32_t> out;               // group -> molecules
        out.reserve(v.size() / 2 + no_umi_.size() + long_.size());
        std::vector<uint32_t> id;
        for (size_t s = 0; s < v.size();) {
            size_t e = s;
            while (e < v.size() && v[e].g == v[s].g) ++e;
            const uint32_t bc = static_cast<uint32_t>(v[s].g >> 32);
            out[v[s].g] += r9_molecules(e - s, [&](size_t i) { return v[s + i].umi; },
                                        [&](size_t i) { return v[s + i].n; }, [&](size_t i) { return v[s + i].gene; },
                                        [&](uint32_t gene, uint64_t u, uint64_t& r) { return gene_root(bc, gene, u, r); },
                                        directional, id);
            s = e;
        }
        for (const auto& kv : no_umi_) out[kv.first] += kv.second;
        for (const auto& k : long_) out[std::stoull(k.substr(0, k.find(':')))] += 1;
        uint64_t total = 0;
        for (const auto& kv : out) {
            emit(static_cast<uint32_t>(kv.first >> 32), static_cast<uint32_t>(kv.first & 0xffffffffu), kv.second);
            total += kv.second;
        }
        return total;
    }

    bool empty() const { return mols_.empty() && no_umi_.empty() && long_.empty(); }
    void clear() {
        decltype(mols_)().swap(mols_);
        decltype(no_umi_)().swap(no_umi_);
        decltype(long_)().swap(long_);
    }

private:
    struct Key {
        uint64_t g, umi;
        bool operator==(const Key& o) const { return g == o.g && umi == o.umi; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.g * 0x9e3779b97f4a7c15ULL ^ (k.umi + 0x632be59bd9b4e019ULL + (k.g << 6) + (k.g >> 2));
            h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
            return static_cast<size_t>(h);
        }
    };
    struct Val { uint32_t n = 0; uint32_t gene = UINT32_MAX; };
    std::unordered_map<Key, Val, KeyHash> mols_;
    std::unordered_map<uint64_t, uint32_t> no_umi_;
    std::unordered_set<std::string> long_;
};

/// Per-molecule allele consensus at panel SNP sites (docs/research-beta/order-free-rules.md R1).
/// add() counts each read's base class (REF / ALT / other) for its (barcode, SNP, UMI) molecule;
/// finalize() emits one DP (and one AD if ALT) per molecule whose strict majority is REF or ALT.
/// A tie between the two largest classes leaves the molecule ambiguous, and an "other" majority is
/// evidence for neither allele: neither enters AD or DP. Counts commute, so the result does not
/// depend on read order. A read without a UMI is its own molecule. With `directional`, the UMIs of one (barcode, SNP)
/// are first collapsed by R9 (umi3_directional_roots) and the class counts summed over each merged molecule.
class SnpConsensus {
public:
    static constexpr uint8_t kRef = 0, kAlt = 1, kOther = 2;
    static uint8_t base_class(char base, char ref, char alt) {
        return base == alt ? kAlt : (base == ref ? kRef : kOther);
    }

    void add(uint32_t bc_idx, uint32_t snp_idx, const char* umi, int umi_len, uint8_t cls,
             uint32_t gene = UINT32_MAX) {
        if (!umi || umi_len <= 0) { single_.push_back({bc_idx, snp_idx, cls}); return; }
        // Exact molecule key, no hashing collisions: (barcode, site) and the UMI packed 3 bits per base
        // (A C G T N other) behind a leading 1 bit that fixes the length; UMIs over 21 bases are kept as strings.
        const uint64_t site = (static_cast<uint64_t>(bc_idx) << 32) | snp_idx;
        uint8_t* n;
        if (umi_len <= 21) {
            Counts& c = mols_[Key{site, umi_pack3(umi, umi_len)}];
            c.gene = std::min(c.gene, gene);
            n = c.n;
        } else {
            n = long_mols_[std::to_string(site) + ':' + std::string(umi, static_cast<size_t>(umi_len))].n;
        }
        if (n[cls] < 255) ++n[cls];
    }

    /// AccT: increment(feature, barcode, delta). StatsT: snp_molecules, snp_molecules_ambiguous, snp_molecules_other.
    /// gene_root(bc, gene, umi3, root&): see r9_molecules (NoGeneRoot = per site only).
    template <typename AccT, typename StatsT, typename GeneRoot = NoGeneRoot>
    void finalize(AccT& ad, AccT& dp, StatsT& st, bool directional = true, GeneRoot&& gene_root = GeneRoot()) const {
        auto emit = [&](uint32_t bc, uint32_t snp, uint8_t cls) {
            if (cls == kOther) { ++st.snp_molecules_other; return; }
            ++st.snp_molecules;
            dp.increment(snp, bc, 1);
            if (cls == kAlt) ad.increment(snp, bc, 1);
        };
        auto vote = [&](uint64_t site, const uint32_t* n) {
            uint8_t best = 0;
            for (uint8_t c = 1; c < 3; ++c) if (n[c] > n[best]) best = c;
            for (uint8_t c = 0; c < 3; ++c)
                if (c != best && n[c] == n[best]) { ++st.snp_molecules_ambiguous; return; }
            emit(static_cast<uint32_t>(site >> 32), static_cast<uint32_t>(site & 0xffffffffu), best);
        };
        for (const auto& r : single_) emit(r.bc, r.snp, r.cls);
        // R9: per (barcode, SNP) site, UMIs sorted by reads descending then UMI, collapsed, class counts merged
        struct E { uint64_t site, umi; uint32_t reads, gene; uint8_t n[3]; };
        std::vector<E> v;
        v.reserve(mols_.size());
        for (const auto& kv : mols_) {
            const uint8_t* c = kv.second.n;
            v.push_back({kv.first.site, kv.first.umi, uint32_t(c[0]) + c[1] + c[2], kv.second.gene, {c[0], c[1], c[2]}});
        }
        std::sort(v.begin(), v.end(), [](const E& a, const E& b) {
            if (a.site != b.site) return a.site < b.site;
            if (a.reads != b.reads) return a.reads > b.reads;
            return a.umi < b.umi;
        });
        std::vector<uint32_t> id;
        std::vector<uint32_t> merged;
        for (size_t s = 0; s < v.size();) {
            size_t e = s;
            while (e < v.size() && v[e].site == v[s].site) ++e;
            const size_t m = e - s;
            const uint32_t bc = static_cast<uint32_t>(v[s].site >> 32);
            const uint32_t k = r9_molecules(m, [&](size_t i) { return v[s + i].umi; },
                                            [&](size_t i) { return v[s + i].reads; },
                                            [&](size_t i) { return v[s + i].gene; },
                                            [&](uint32_t gene, uint64_t u, uint64_t& r) { return gene_root(bc, gene, u, r); },
                                            directional, id);
            merged.assign(3 * static_cast<size_t>(k), 0);
            for (size_t i = 0; i < m; ++i)
                for (int c = 0; c < 3; ++c) merged[3 * id[i] + c] += v[s + i].n[c];
            for (uint32_t j = 0; j < k; ++j) vote(v[s].site, &merged[3 * j]);
            s = e;
        }
        for (const auto& kv : long_mols_) {
            const uint32_t n[3] = {kv.second.n[0], kv.second.n[1], kv.second.n[2]};
            vote(std::stoull(kv.first.substr(0, kv.first.find(':'))), n);
        }
    }

    void clear() {
        decltype(mols_)().swap(mols_);
        decltype(long_mols_)().swap(long_mols_);
        std::vector<Single>().swap(single_);
    }

private:
    struct Key {
        uint64_t site, umi;
        bool operator==(const Key& o) const { return site == o.site && umi == o.umi; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            uint64_t h = k.site * 0x9e3779b97f4a7c15ULL ^ (k.umi + 0x632be59bd9b4e019ULL + (k.site << 6) + (k.site >> 2));
            h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
            return static_cast<size_t>(h);
        }
    };
    struct Counts { uint8_t n[3] = {0, 0, 0}; uint32_t gene = UINT32_MAX; };
    struct Single { uint32_t bc; uint32_t snp; uint8_t cls; };
    std::unordered_map<Key, Counts, KeyHash> mols_;
    std::unordered_map<std::string, Counts> long_mols_;
    std::vector<Single> single_;
};

} // namespace singlet
