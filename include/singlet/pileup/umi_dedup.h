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
        for (auto& e : group) {
            if (e.packed == umi_packed) {
                e.count++;
                e.flags |= flags;
                // smallest index, not first seen: read order (STAR threads) must not matter
                e.exon_idx = std::min(e.exon_idx, exon_idx);
                e.intron_idx = std::min(e.intron_idx, intron_idx);
                return;
            }
        }
        group.push_back({umi_packed, exon_idx, intron_idx, 1u, flags});
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
            auto find = [&](int x) -> int {
                while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
                return x;
            };
            for (int i = 0; i < n - 1; ++i)
                for (int j = i + 1; j < n; ++j)
                    if (umi_hamming_2bit(sorted[i].packed, sorted[j].packed, umi_len) == 1 &&
                        sorted[i].count >= 2 * sorted[j].count - 1) {
                        int ri = find(i), rj = find(j);
                        if (ri != rj) parent[rj] = ri;
                    }

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
    for (size_t i = 0; i + 1 < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            if (count(i) >= 2 * count(j) - 1 && umi3_one_mismatch(umi(i), umi(j))) {
                const uint32_t ri = find(static_cast<uint32_t>(i)), rj = find(static_cast<uint32_t>(j));
                if (ri != rj) root[rj] = ri;
            }
    for (size_t i = 0; i < n; ++i) root[i] = find(static_cast<uint32_t>(i));
}

/// Molecule counting for one feature kind (splice junctions, V(D)J genes) under R9: add() keeps the read count of
/// every exact (barcode, feature, UMI) during the pass; finalize() collapses each (barcode, feature) group and emits
/// its molecule count. Order-free: the pass only counts, and finalize sorts before collapsing.
class UmiFeatureCollapse {
public:
    void add(uint32_t bc_idx, uint32_t feature, const char* umi, int umi_len) {
        const uint64_t g = (static_cast<uint64_t>(bc_idx) << 32) | feature;
        if (!umi || umi_len <= 0) { ++no_umi_[g]; return; }        // a read without a UMI is its own molecule
        if (umi_len <= 21) { uint32_t& c = mols_[Key{g, umi_pack3(umi, umi_len)}]; if (c != UINT32_MAX) ++c; }
        else long_.insert(std::to_string(g) + ':' + std::string(umi, static_cast<size_t>(umi_len)));  // exact only
    }

    /// emit(bc_idx, feature, molecules) once per (barcode, feature) group with molecules > 0. Returns the total.
    template <typename Emit>
    uint64_t finalize(Emit&& emit, bool directional) const {
        struct E { uint64_t g, umi; uint32_t n; };
        std::vector<E> v;
        v.reserve(mols_.size());
        for (const auto& kv : mols_) v.push_back({kv.first.g, kv.first.umi, kv.second});
        std::sort(v.begin(), v.end(), [](const E& a, const E& b) {
            if (a.g != b.g) return a.g < b.g;
            if (a.n != b.n) return a.n > b.n;
            return a.umi < b.umi;
        });
        std::unordered_map<uint64_t, uint32_t> out;               // group -> molecules
        out.reserve(v.size() / 2 + no_umi_.size() + long_.size());
        std::vector<uint32_t> root;
        for (size_t s = 0; s < v.size();) {
            size_t e = s;
            while (e < v.size() && v[e].g == v[s].g) ++e;
            uint32_t mols = static_cast<uint32_t>(e - s);
            if (directional && mols > 1) {
                umi3_directional_roots(e - s, [&](size_t i) { return v[s + i].umi; },
                                       [&](size_t i) { return v[s + i].n; }, root);
                mols = 0;
                for (size_t i = 0; i < root.size(); ++i) mols += (root[i] == i);
            }
            out[v[s].g] += mols;
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
    std::unordered_map<Key, uint32_t, KeyHash> mols_;
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

    void add(uint32_t bc_idx, uint32_t snp_idx, const char* umi, int umi_len, uint8_t cls) {
        if (!umi || umi_len <= 0) { single_.push_back({bc_idx, snp_idx, cls}); return; }
        // Exact molecule key, no hashing collisions: (barcode, site) and the UMI packed 3 bits per base
        // (A C G T N other) behind a leading 1 bit that fixes the length; UMIs over 21 bases are kept as strings.
        const uint64_t site = (static_cast<uint64_t>(bc_idx) << 32) | snp_idx;
        uint8_t* n;
        if (umi_len <= 21) {
            n = mols_[Key{site, umi_pack3(umi, umi_len)}].n;
        } else {
            n = long_mols_[std::to_string(site) + ':' + std::string(umi, static_cast<size_t>(umi_len))].n;
        }
        if (n[cls] < 255) ++n[cls];
    }

    /// AccT: increment(feature, barcode, delta). StatsT: snp_molecules, snp_molecules_ambiguous, snp_molecules_other.
    template <typename AccT, typename StatsT>
    void finalize(AccT& ad, AccT& dp, StatsT& st, bool directional = true) const {
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
        struct E { uint64_t site, umi; uint32_t reads; uint8_t n[3]; };
        std::vector<E> v;
        v.reserve(mols_.size());
        for (const auto& kv : mols_) {
            const uint8_t* c = kv.second.n;
            v.push_back({kv.first.site, kv.first.umi, uint32_t(c[0]) + c[1] + c[2], {c[0], c[1], c[2]}});
        }
        std::sort(v.begin(), v.end(), [](const E& a, const E& b) {
            if (a.site != b.site) return a.site < b.site;
            if (a.reads != b.reads) return a.reads > b.reads;
            return a.umi < b.umi;
        });
        std::vector<uint32_t> root;
        std::vector<uint32_t> merged;
        for (size_t s = 0; s < v.size();) {
            size_t e = s;
            while (e < v.size() && v[e].site == v[s].site) ++e;
            const size_t m = e - s;
            if (directional && m > 1) {
                umi3_directional_roots(m, [&](size_t i) { return v[s + i].umi; },
                                       [&](size_t i) { return v[s + i].reads; }, root);
            } else {
                root.resize(m);
                std::iota(root.begin(), root.end(), 0u);
            }
            merged.assign(3 * m, 0);
            for (size_t i = 0; i < m; ++i)
                for (int c = 0; c < 3; ++c) merged[3 * root[i] + c] += v[s + i].n[c];
            for (size_t i = 0; i < m; ++i)
                if (root[i] == i) vote(v[s].site, &merged[3 * i]);
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
    struct Counts { uint8_t n[3] = {0, 0, 0}; };
    struct Single { uint32_t bc; uint32_t snp; uint8_t cls; };
    std::unordered_map<Key, Counts, KeyHash> mols_;
    std::unordered_map<std::string, Counts> long_mols_;
    std::vector<Single> single_;
};

} // namespace singlet
