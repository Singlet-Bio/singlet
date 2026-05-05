#pragma once
// singlet-pileup: saturation_curve.h
// G-SATCURVE: Sequencing saturation downsampling curves.
//
// Produces saturation_curve.tsv showing UMIs/cell and genes/cell as a function
// of sequencing depth fractions: 0.1, 0.25, 0.5, 0.75, 0.9, 1.0.
//
// APPROACH: Analytical post-hoc from existing CSC + CellReadStats.
// Zero pileup overhead — uses data already in memory.
//
// UMI formula (Lander-Waterman / Poisson):
//   umis_at_f = U * (1 - exp(-N * f / U))
//   where N = total_reads, U = unique_umis (both from CellReadStats).
//
// Gene formula (Poisson per gene-level grouping):
//   For each gene g with c_g unique UMIs in a cell (c_g from exon CSC collapse):
//     P(gene g detected at f) = 1 - exp(-c_g * rpu * f)
//   where rpu = N / U (reads per unique molecule for the cell).
//   Expected genes at f = sum_g P(gene g detected at f).
//
// Outputs: saturation_curve.tsv
//   fraction | sampled_reads | median_umis | median_genes | mean_umis | mean_genes

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "gene_model.h"
#include "read_stats.h"
#include "sparse_accumulator.h"

namespace singlet {

/// One point on the saturation curve.
struct SaturationPoint {
    double fraction = 0.0;       ///< Depth fraction (0.1 … 1.0)
    uint64_t sampled_reads = 0;  ///< Expected total reads at this fraction
    double median_umis = 0.0;    ///< Median unique UMIs per cell
    double median_genes = 0.0;   ///< Median unique genes per cell
    double mean_umis = 0.0;      ///< Mean unique UMIs per cell
    double mean_genes = 0.0;     ///< Mean unique genes per cell
};

namespace detail {

// Compute median of a SORTED vector; returns 0 if empty.
inline double sorted_median(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    const size_t m = n / 2;
    return (n % 2 == 1) ? v[m] : (v[m - 1] + v[m]) * 0.5;
}

}  // namespace detail

/// Compute sequencing saturation downsampling curves.
///
/// @param cell_stats    Per-cell read statistics from compute_read_stats()
///                      (provides total_reads and unique_umis per cell).
/// @param exon_csc      Exon-interval CSC matrix (n_exon_intervals × n_cells)
///                      from SparseAccumulator<uint16_t>::to_csc().
/// @param gm            Gene model — provides exon_to_gene() mapping.
/// @param fractions     Depth fractions to evaluate (default: standard 6-point).
/// @return              Vector of SaturationPoint, one per fraction.
inline std::vector<SaturationPoint> compute_saturation_curve(
    const std::vector<CellReadStats>& cell_stats,
    const SparseAccumulator<uint16_t>::CSCMatrix& exon_csc,
    const GeneModel& gm,
    const std::vector<double>& fractions = {0.1, 0.25, 0.5, 0.75, 0.9, 1.0}) {
    std::vector<SaturationPoint> result;
    if (cell_stats.empty() || exon_csc.ncols == 0) return result;

    const uint32_t n_cells = exon_csc.ncols;
    const uint32_t n_fracs = static_cast<uint32_t>(fractions.size());
    (void)n_fracs;

    // Pre-compute total reads across all cells (for sampled_reads column)
    uint64_t total_all_reads = 0;
    for (const auto& cs : cell_stats)
        total_all_reads += cs.total_reads;

    // Per-cell gene-level sums: gene_sums_per_cell[c] = map from gene_idx → summed UMIs.
    // We build a temporary per-cell gene accumulation on the fly per fraction.
    // To avoid O(n_cells * n_genes) memory, we accumulate lazily using existing CSC.

    // Step 1: Build a flat list of (cell_idx, gene_idx, umi_count) triples
    // from exon_csc. This is O(nnz) memory which is already present.
    struct GeneCellEntry {
        uint32_t cell_idx;
        uint32_t gene_idx;
        uint16_t count;
    };
    std::vector<GeneCellEntry> entries;
    entries.reserve(exon_csc.data.size());

    for (uint32_t j = 0; j < n_cells; ++j) {
        for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k) {
            uint32_t exon_idx = static_cast<uint32_t>(exon_csc.indices[k]);
            uint32_t gene_idx = gm.exon_to_gene(exon_idx);
            if (gene_idx == UINT32_MAX) continue;  // sentinel
            entries.push_back({j, gene_idx, exon_csc.data[k]});
        }
    }

    // Step 2: Sort by (cell_idx, gene_idx) so we can group efficiently.
    std::sort(entries.begin(), entries.end(),
              [](const GeneCellEntry& a, const GeneCellEntry& b) {
                  return a.cell_idx != b.cell_idx
                             ? a.cell_idx < b.cell_idx
                             : a.gene_idx < b.gene_idx;
              });

    // Step 3: Reduce to (cell_idx, gene_idx, total_gene_umis) pairs.
    struct GeneCellCount {
        uint32_t cell_idx;
        uint32_t gene_idx;
        uint32_t total_umis;  // sum of exon-interval UMIs for this gene in this cell
    };
    std::vector<GeneCellCount> gene_cell_counts;
    gene_cell_counts.reserve(entries.size());
    for (size_t i = 0; i < entries.size();) {
        size_t j = i;
        uint32_t sum = 0;
        while (j < entries.size() && entries[j].cell_idx == entries[i].cell_idx && entries[j].gene_idx == entries[i].gene_idx) {
            sum += entries[j].count;
            ++j;
        }
        gene_cell_counts.push_back({entries[i].cell_idx, entries[i].gene_idx, sum});
        i = j;
    }

    // Build offset array: start index in gene_cell_counts for each cell.
    std::vector<uint32_t> cell_offset(n_cells + 1, 0);
    for (const auto& gc : gene_cell_counts)
        if (gc.cell_idx < n_cells) cell_offset[gc.cell_idx + 1]++;
    for (uint32_t c = 0; c < n_cells; ++c)
        cell_offset[c + 1] += cell_offset[c];

    // Step 4: For each fraction, compute expected UMIs and genes per cell.
    for (double frac : fractions) {
        std::vector<double> umi_vec, gene_vec;
        umi_vec.reserve(n_cells);
        gene_vec.reserve(n_cells);

        for (uint32_t c = 0; c < n_cells; ++c) {
            if (c >= static_cast<uint32_t>(cell_stats.size())) break;
            const CellReadStats& cs = cell_stats[c];
            if (cs.total_reads == 0 || cs.unique_umis == 0) continue;

            const double N = static_cast<double>(cs.total_reads);
            const double U = static_cast<double>(cs.unique_umis);

            // UMI saturation curve: Lander-Waterman formula
            // Expected unique UMIs at depth f = U * (1 - exp(-N*f/U))
            double umis_at_f = U * (1.0 - std::exp(-N * frac / U));

            // Gene curve: Poisson per gene
            // rpu = reads per unique molecule = N / U
            // For gene g with c_g unique UMIs: raw reads ≈ c_g * rpu
            // P(detect gene g at f) = 1 - exp(-c_g * rpu * f)
            const double rpu = N / U;
            double genes_at_f = 0.0;
            for (uint32_t idx = cell_offset[c]; idx < cell_offset[c + 1]; ++idx) {
                const double cg = static_cast<double>(gene_cell_counts[idx].total_umis);
                genes_at_f += 1.0 - std::exp(-cg * rpu * frac);
            }

            umi_vec.push_back(umis_at_f);
            gene_vec.push_back(genes_at_f);
        }

        if (umi_vec.empty()) continue;

        std::sort(umi_vec.begin(), umi_vec.end());
        std::sort(gene_vec.begin(), gene_vec.end());

        SaturationPoint pt;
        pt.fraction = frac;
        pt.sampled_reads = static_cast<uint64_t>(total_all_reads * frac + 0.5);
        pt.median_umis = detail::sorted_median(umi_vec);
        pt.median_genes = detail::sorted_median(gene_vec);
        pt.mean_umis = std::accumulate(umi_vec.begin(), umi_vec.end(), 0.0) / static_cast<double>(umi_vec.size());
        pt.mean_genes = std::accumulate(gene_vec.begin(), gene_vec.end(), 0.0) / static_cast<double>(gene_vec.size());
        result.push_back(pt);
    }

    return result;
}

/// Write saturation_curve.tsv.
/// Columns: fraction, sampled_reads, median_umis, median_genes, mean_umis, mean_genes
inline bool write_saturation_curve_tsv(
    const std::string& path,
    const std::vector<SaturationPoint>& curve) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[saturation_curve] Warning: cannot write " << path << "\n";
        return false;
    }
    f << "fraction\tsampled_reads\tmedian_umis\tmedian_genes\tmean_umis\tmean_genes\n";
    for (const auto& pt : curve) {
        f << pt.fraction << '\t'
          << pt.sampled_reads << '\t'
          << pt.median_umis << '\t'
          << pt.median_genes << '\t'
          << pt.mean_umis << '\t'
          << pt.mean_genes << '\n';
    }
    return true;
}

}  // namespace singlet
