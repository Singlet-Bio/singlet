#pragma once
// singlet-pileup: em_rescue.h
// G-EM: Equivalence-class EM multi-mapper rescue.
//
// Rescues reads that were dropped by resolve_mm() because they mapped to
// multiple genes. Uses an EM algorithm to distribute fractional counts based
// on a unique-count prior derived from already-counted gene expression.
//
// Algorithm (STARsolo/alevin-fry style):
//   1. Collect AmbigReads: (bc_idx, umi, gene_a, gene_b) for each dropped read
//   2. UMI-deduplicate within each (bc_idx, gene_a, gene_b) equivalence class
//   3. Global EM: learn gene expression proportions from unique counts
//   4. Distribute per-cell per-EC counts using learned proportions
//   5. Round to integer and add to gene_counts_em output matrix

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {
namespace em_rescue {

// ── Data structures ───────────────────────────────────────────────────────

/// Ambiguous read buffered when resolve_mm() would drop a gene_ambiguous entry.
/// gene_a and gene_b are the two candidate genes. gene_b may be UINT32_MAX if
/// only one candidate is known (those reads are skipped in EM).
struct AmbigRead {
    int32_t bc_idx = -1;
    char umi[16] = {};
    uint8_t umi_len = 0;
    uint32_t gene_a = UINT32_MAX;  // first candidate gene
    uint32_t gene_b = UINT32_MAX;  // second candidate gene (UINT32_MAX = skip)
};

/// Statistics reported after EM rescue.
struct EMRescueStats {
    uint64_t n_ambig_reads = 0;    // total AmbigReads submitted
    uint64_t n_valid_reads = 0;    // reads with both gene_a and gene_b known
    uint64_t n_deduped_umis = 0;   // unique UMIs after per-cell dedup
    uint64_t n_rescued_ints = 0;   // integer counts added to output matrix
    uint64_t n_equiv_classes = 0;  // distinct (gene_a, gene_b) EC pairs
    uint32_t n_iterations = 0;     // EM iterations until convergence
};

/// CSC sparse matrix for EM output (matches export.h GeneCSC<uint32_t> layout).
struct EMGeneCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint32_t> data;
    uint32_t nrows = 0;
    uint32_t ncols = 0;
};

// ── Internal helpers ──────────────────────────────────────────────────────

namespace detail {

// 64-bit FNV-1a hash over raw bytes.
inline uint64_t fnv64(const void* data, size_t len, uint64_t seed = 14695981039346656037ULL) {
    const auto* p = static_cast<const uint8_t*>(data);
    uint64_t h = seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/// Key for UMI deduplication: (bc_idx, gene_a_sorted, gene_b_sorted, umi bytes).
struct UmiKey {
    int32_t bc_idx;
    uint32_t gene_lo;  // min(gene_a, gene_b)
    uint32_t gene_hi;  // max(gene_a, gene_b)
    char umi[16];
    uint8_t umi_len;

    bool operator==(const UmiKey& o) const {
        return bc_idx == o.bc_idx && gene_lo == o.gene_lo && gene_hi == o.gene_hi && umi_len == o.umi_len && std::memcmp(umi, o.umi, umi_len) == 0;
    }
};

struct UmiKeyHash {
    size_t operator()(const UmiKey& k) const {
        uint64_t h = fnv64(&k.bc_idx, sizeof(k.bc_idx));
        h = fnv64(&k.gene_lo, sizeof(k.gene_lo), h);
        h = fnv64(&k.gene_hi, sizeof(k.gene_hi), h);
        h = fnv64(k.umi, k.umi_len, h);
        return static_cast<size_t>(h);
    }
};

/// Per-cell per-EC entry accumulated during EM.
/// Key: (bc_idx, gene_lo, gene_hi).  UMI dedup via UmiKey set.
struct ECCellKey {
    int32_t bc_idx;
    uint32_t gene_lo;
    uint32_t gene_hi;
    bool operator==(const ECCellKey& o) const {
        return bc_idx == o.bc_idx && gene_lo == o.gene_lo && gene_hi == o.gene_hi;
    }
};
struct ECCellKeyHash {
    size_t operator()(const ECCellKey& k) const {
        uint64_t h = fnv64(&k.bc_idx, sizeof(k.bc_idx));
        h = fnv64(&k.gene_lo, sizeof(k.gene_lo), h);
        h = fnv64(&k.gene_hi, sizeof(k.gene_hi), h);
        return static_cast<size_t>(h);
    }
};

}  // namespace detail

// ── Main entry point ──────────────────────────────────────────────────────

/// Run EM rescue on a collection of ambiguous reads.
///
/// @param ambig_reads   AmbigRead vector collected during pileup
/// @param gene_totals   Per-gene total unique counts across all cells (length n_genes)
///                      — used as the EM prior (sum of gene_csc over cells)
/// @param cell_gene_unique  Flat array [gene * n_cells + cell] of unique counts per cell
///                          (used to produce the per-cell rescue output).
///                          If empty, cell_gene_rescue will also be empty.
/// @param n_genes       Number of genes
/// @param n_cells       Number of cells (barcodes)
/// @param max_iter      Maximum EM iterations
/// @param tol           Convergence tolerance (fractional change in expression)
/// @param[out] stats    Statistics about the rescue run
/// @return              CSC sparse matrix (n_genes × n_cells) with rescued integer counts
inline EMGeneCSC run_em_rescue(
    const std::vector<AmbigRead>& ambig_reads,
    const std::vector<double>& gene_totals,
    uint32_t n_genes,
    uint32_t n_cells,
    int max_iter = 50,
    double tol = 0.01,
    EMRescueStats* stats_out = nullptr) {
    EMRescueStats stats;
    stats.n_ambig_reads = ambig_reads.size();

    // ── Phase 1: UMI dedup and per-cell per-EC count accumulation ──

    // Map (bc_idx, gene_lo, gene_hi) → UMI-deduped count
    std::unordered_map<detail::ECCellKey, uint32_t, detail::ECCellKeyHash> ec_cell_counts;
    ec_cell_counts.reserve(ambig_reads.size() / 2);

    // Track seen UMIs
    std::unordered_set<detail::UmiKey, detail::UmiKeyHash>* umi_seen = nullptr;
    std::unordered_set<detail::UmiKey, detail::UmiKeyHash> umi_seen_impl;
    umi_seen_impl.reserve(ambig_reads.size());
    umi_seen = &umi_seen_impl;

    for (const auto& ar : ambig_reads) {
        if (ar.bc_idx < 0) continue;
        if (ar.gene_a == UINT32_MAX || ar.gene_b == UINT32_MAX) continue;  // skip incomplete
        if (ar.gene_a >= n_genes || ar.gene_b >= n_genes) continue;

        stats.n_valid_reads++;

        uint32_t glo = std::min(ar.gene_a, ar.gene_b);
        uint32_t ghi = std::max(ar.gene_a, ar.gene_b);

        // UMI dedup
        detail::UmiKey uk;
        uk.bc_idx = ar.bc_idx;
        uk.gene_lo = glo;
        uk.gene_hi = ghi;
        uk.umi_len = ar.umi_len;
        std::memset(uk.umi, 0, sizeof(uk.umi));
        if (ar.umi_len > 0) std::memcpy(uk.umi, ar.umi, ar.umi_len);

        if (ar.umi_len > 0) {
            auto [it, inserted] = umi_seen->insert(uk);
            if (!inserted) continue;  // duplicate UMI
        }

        // Accumulate per-cell per-EC count
        detail::ECCellKey ck{ar.bc_idx, glo, ghi};
        ec_cell_counts[ck]++;
        stats.n_deduped_umis++;
    }

    if (ec_cell_counts.empty()) {
        if (stats_out) *stats_out = stats;
        EMGeneCSC empty;
        empty.nrows = n_genes;
        empty.ncols = n_cells;
        empty.indptr.assign(n_cells + 1, 0);
        return empty;
    }

    // ── Phase 2: Build equivalence class global counts ──
    // Map (gene_lo, gene_hi) → global count (sum over cells)
    struct ECKey {
        uint32_t gene_lo, gene_hi;
        bool operator==(const ECKey& o) const {
            return gene_lo == o.gene_lo && gene_hi == o.gene_hi;
        }
    };
    struct ECKeyHash {
        size_t operator()(const ECKey& k) const {
            return std::hash<uint64_t>{}((static_cast<uint64_t>(k.gene_lo) << 32) | k.gene_hi);
        }
    };
    std::unordered_map<ECKey, uint32_t, ECKeyHash> ec_global;
    ec_global.reserve(ec_cell_counts.size());

    for (const auto& [ck, cnt] : ec_cell_counts) {
        ec_global[{ck.gene_lo, ck.gene_hi}] += cnt;
    }
    stats.n_equiv_classes = ec_global.size();

    // ── Phase 3: Global EM ──
    // theta[g] = expression estimate for gene g (initialized to unique counts)
    std::vector<double> theta(n_genes, 0.0);
    for (uint32_t g = 0; g < n_genes && g < static_cast<uint32_t>(gene_totals.size()); ++g)
        theta[g] = gene_totals[g];

    // Add a pseudocount of 1 for genes with unique count of 0 that appear in ECs
    for (const auto& [ec, _cnt] : ec_global) {
        if (theta[ec.gene_lo] == 0.0) theta[ec.gene_lo] = 1.0;
        if (theta[ec.gene_hi] == 0.0) theta[ec.gene_hi] = 1.0;
    }

    double prev_total_change = std::numeric_limits<double>::max();
    for (int iter = 0; iter < max_iter; ++iter) {
        // E-step: compute per-EC weights given current theta
        // M-step: new_theta[g] = unique[g] + sum_EC( count_EC * weight_g_in_EC )
        std::vector<double> new_theta(n_genes, 0.0);
        for (uint32_t g = 0; g < n_genes && g < static_cast<uint32_t>(gene_totals.size()); ++g)
            new_theta[g] = gene_totals[g];  // re-add unique counts as base

        for (const auto& [ec, global_cnt] : ec_global) {
            double ta = theta[ec.gene_lo];
            double tb = theta[ec.gene_hi];
            double total = ta + tb;
            if (total <= 0.0) continue;
            double wa = ta / total;
            double wb = tb / total;
            new_theta[ec.gene_lo] += wa * static_cast<double>(global_cnt);
            new_theta[ec.gene_hi] += wb * static_cast<double>(global_cnt);
        }

        // Check convergence: max relative change
        double max_change = 0.0;
        for (uint32_t g = 0; g < n_genes; ++g) {
            if (theta[g] > 0.0) {
                double rel = std::abs(new_theta[g] - theta[g]) / theta[g];
                max_change = std::max(max_change, rel);
            }
        }

        theta = std::move(new_theta);
        stats.n_iterations = static_cast<uint32_t>(iter + 1);

        if (max_change < tol && iter >= 2) break;
        if (std::abs(prev_total_change - max_change) < 1e-9) break;  // stalled
        prev_total_change = max_change;
    }

    // ── Phase 4: Distribute per-cell per-EC counts using EM proportions ──
    // Output: COO entries (gene_idx, cell_idx, count)
    struct COOEntry {
        uint32_t gene;
        uint32_t cell;
        uint32_t count;
    };
    std::vector<COOEntry> coo;
    coo.reserve(ec_cell_counts.size() * 2);

    for (const auto& [ck, cell_cnt] : ec_cell_counts) {
        double ta = theta[ck.gene_lo];
        double tb = theta[ck.gene_hi];
        double total = ta + tb;
        if (total <= 0.0) continue;

        double wa = ta / total;
        double wb = tb / total;

        // Round fractional counts preserving total: use stochastic rounding deterministically
        uint32_t cnt_a = static_cast<uint32_t>(std::round(wa * static_cast<double>(cell_cnt)));
        uint32_t cnt_b = static_cast<uint32_t>(std::round(wb * static_cast<double>(cell_cnt)));
        // Clamp so sum doesn't exceed cell_cnt (rounding artifacts)
        if (cnt_a + cnt_b > cell_cnt) {
            if (cnt_a >= cnt_b)
                cnt_a = cell_cnt - cnt_b;
            else
                cnt_b = cell_cnt - cnt_a;
        }

        if (cnt_a > 0) {
            coo.push_back({ck.gene_lo, static_cast<uint32_t>(ck.bc_idx), cnt_a});
            stats.n_rescued_ints += cnt_a;
        }
        if (cnt_b > 0) {
            coo.push_back({ck.gene_hi, static_cast<uint32_t>(ck.bc_idx), cnt_b});
            stats.n_rescued_ints += cnt_b;
        }
    }

    // ── Phase 5: COO → CSC ──
    EMGeneCSC out;
    out.nrows = n_genes;
    out.ncols = n_cells;
    out.indptr.assign(n_cells + 1, 0);

    if (coo.empty()) {
        if (stats_out) *stats_out = stats;
        return out;
    }

    // Sort COO by cell (column), then gene (row)
    std::sort(coo.begin(), coo.end(), [](const COOEntry& a, const COOEntry& b) {
        if (a.cell != b.cell) return a.cell < b.cell;
        return a.gene < b.gene;
    });

    // Aggregate (same gene in same cell can appear from multiple ECs)
    std::vector<COOEntry> merged;
    merged.reserve(coo.size());
    for (size_t i = 0; i < coo.size();) {
        size_t j = i + 1;
        uint32_t total_cnt = coo[i].count;
        while (j < coo.size() && coo[j].cell == coo[i].cell && coo[j].gene == coo[i].gene) {
            total_cnt += coo[j].count;
            ++j;
        }
        merged.push_back({coo[i].gene, coo[i].cell, total_cnt});
        i = j;
    }

    out.indices.reserve(merged.size());
    out.data.reserve(merged.size());

    for (const auto& e : merged) {
        out.indptr[e.cell + 1]++;
    }
    // Prefix sum
    for (uint32_t c = 0; c < n_cells; ++c)
        out.indptr[c + 1] += out.indptr[c];

    // Fill indices/data using sorted merged entries
    for (const auto& e : merged) {
        out.indices.push_back(static_cast<int32_t>(e.gene));
        out.data.push_back(e.count);
    }

    if (stats_out) *stats_out = stats;
    return out;
}

/// Convenience overload: compute gene_totals from a GeneCSC (sums across cells).
template <typename GeneCSCType>
inline EMGeneCSC run_em_rescue(
    const std::vector<AmbigRead>& ambig_reads,
    const GeneCSCType& gene_csc,
    int max_iter = 50,
    double tol = 0.01,
    EMRescueStats* stats_out = nullptr) {
    const uint32_t n_genes = gene_csc.nrows;
    const uint32_t n_cells = gene_csc.ncols;

    // Compute per-gene totals across all cells
    std::vector<double> gene_totals(n_genes, 0.0);
    for (uint32_t c = 0; c < n_cells; ++c) {
        for (int32_t k = gene_csc.indptr[c]; k < gene_csc.indptr[c + 1]; ++k) {
            uint32_t g = static_cast<uint32_t>(gene_csc.indices[k]);
            if (g < n_genes)
                gene_totals[g] += static_cast<double>(gene_csc.data[k]);
        }
    }

    return run_em_rescue(ambig_reads, gene_totals, n_genes, n_cells,
                         max_iter, tol, stats_out);
}

}  // namespace em_rescue
}  // namespace singlet
