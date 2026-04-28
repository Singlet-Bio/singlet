#pragma once
// singlet-pileup: metrics_summary.h
// G-METRICS: Write metrics_summary.csv compatible with Cell Ranger output.
// Provides a single-row CSV summary of key pipeline statistics.
//
// Fields mirror Cell Ranger's metrics_summary.csv for Seurat/Scanpy compatibility.

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

#include "gene_model.h"
#include "pileup_engine.h"
#include "sparse_accumulator.h"

namespace singlet {

/// Format a percentage string (e.g. "73.2%").
inline std::string fmt_pct(double fraction) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%", fraction * 100.0);
    return buf;
}

/// Format an integer with comma thousands separator.
inline std::string fmt_int(uint64_t v) {
    std::string s = std::to_string(v);
    int n = static_cast<int>(s.size());
    std::string out;
    out.reserve(n + (n - 1) / 3);
    for (int i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) out += ',';
        out += s[static_cast<size_t>(i)];
    }
    return out;
}

/// Format a float with fixed decimal places.
inline std::string fmt_float(double v, int decimals = 1) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*f", decimals, v);
    return buf;
}

/// Compute the median of a sorted-or-unsorted vector of values.
/// Returns 0 if vec is empty.
template <typename T>
inline double median_of(std::vector<T> v) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    std::nth_element(v.begin(), v.begin() + n / 2, v.end());
    if (n % 2 == 1) return static_cast<double>(v[n / 2]);
    // Even: average of two middle elements
    double upper = static_cast<double>(v[n / 2]);
    std::nth_element(v.begin(), v.begin() + n / 2 - 1, v.end());
    return (static_cast<double>(v[n / 2 - 1]) + upper) / 2.0;
}

/// Write metrics_summary.csv to `path`.
///
/// @param path          Output file path (e.g. outdir/metrics_summary.csv)
/// @param stats         PileupStats from the completed pileup run
/// @param cell_indices  Barcode indices of called cells (from CellCallResult)
/// @param bc_totals     Per-barcode total UMI count (size == exon_csc.ncols)
/// @param exon_csc      Exon-interval count matrix (ncols == n_barcodes)
/// @param gene_model    GeneModel (for exon→gene mapping and total-gene count)
inline void write_metrics_summary_csv(
    const std::string& path,
    const PileupStats& stats,
    const std::vector<uint32_t>& cell_indices,
    const std::vector<uint64_t>& bc_totals,
    const SparseAccumulator<uint16_t>::CSCMatrix& exon_csc,
    const GeneModel& gene_model) {
    const uint64_t n_cells = cell_indices.size();
    const uint64_t total_reads = stats.total_reads;

    // ── Mean Reads per Cell ───────────────────────────────────────────────────
    const double mean_reads = (n_cells > 0)
                                  ? static_cast<double>(total_reads) / static_cast<double>(n_cells)
                                  : 0.0;

    // ── Median Genes per Cell and Median UMI per Cell ─────────────────────────
    // Computed over called cells only. Map exon intervals → gene IDs.
    const uint32_t n_genes = gene_model.n_genes();
    std::vector<uint32_t> genes_per_cell;
    std::vector<uint64_t> umi_per_cell;
    genes_per_cell.reserve(n_cells);
    umi_per_cell.reserve(n_cells);

    if (!exon_csc.data.empty() && n_genes > 0) {
        std::unordered_set<uint32_t> gene_set;
        for (uint32_t ci : cell_indices) {
            if (ci >= exon_csc.ncols) continue;
            gene_set.clear();
            uint64_t umi_sum = 0;
            for (int32_t k = exon_csc.indptr[ci]; k < exon_csc.indptr[ci + 1]; ++k) {
                uint32_t exon_idx = static_cast<uint32_t>(exon_csc.indices[k]);
                uint32_t gene_idx = gene_model.exon_to_gene(exon_idx);
                if (gene_idx < n_genes) gene_set.insert(gene_idx);
                umi_sum += static_cast<uint64_t>(exon_csc.data[k]);
            }
            genes_per_cell.push_back(static_cast<uint32_t>(gene_set.size()));
            umi_per_cell.push_back(umi_sum);
        }
    } else if (!bc_totals.empty()) {
        // Fallback: use bc_totals for UMI; genes are unknown
        for (uint32_t ci : cell_indices) {
            if (ci < bc_totals.size()) umi_per_cell.push_back(bc_totals[ci]);
        }
    }

    const double median_genes = median_of(genes_per_cell);
    const double median_umi = median_of(umi_per_cell);

    // ── Valid Barcodes ────────────────────────────────────────────────────────
    const double valid_barcodes_pct = (total_reads > 0)
                                          ? static_cast<double>(stats.barcoded_reads) / static_cast<double>(total_reads)
                                          : 0.0;

    // ── Sequencing Saturation ─────────────────────────────────────────────────
    // Saturation = fraction of deduplicated UMIs that are duplicates
    const uint64_t total_umi = stats.umi_unique + stats.umi_duplicate;
    const double saturation_pct = (total_umi > 0)
                                      ? static_cast<double>(stats.umi_duplicate) / static_cast<double>(total_umi)
                                      : 0.0;

    // ── Reads Mapped to Genome ────────────────────────────────────────────────
    const double mapped_pct = (total_reads > 0)
                                  ? static_cast<double>(stats.mapped_reads) / static_cast<double>(total_reads)
                                  : 0.0;

    // ── Reads Mapped Confidently to Transcriptome ─────────────────────────────
    const double exon_pct = (total_reads > 0)
                                ? static_cast<double>(stats.exon_hits) / static_cast<double>(total_reads)
                                : 0.0;

    // ── Reads Mapped Confidently to Intronic Regions ─────────────────────────
    const double intron_pct = (total_reads > 0)
                                  ? static_cast<double>(stats.intron_hits) / static_cast<double>(total_reads)
                                  : 0.0;

    // ── Fraction Reads in Cells ───────────────────────────────────────────────
    uint64_t reads_in_cells = 0;
    if (!bc_totals.empty()) {
        for (uint32_t ci : cell_indices)
            if (ci < bc_totals.size()) reads_in_cells += bc_totals[ci];
    }
    const double frac_in_cells = (total_reads > 0)
                                     ? static_cast<double>(reads_in_cells) / static_cast<double>(total_reads)
                                     : 0.0;

    // ── Total Genes Detected (across all called cells) ────────────────────────
    std::unordered_set<uint32_t> all_genes;
    if (!exon_csc.data.empty() && n_genes > 0) {
        for (uint32_t ci : cell_indices) {
            if (ci >= exon_csc.ncols) continue;
            for (int32_t k = exon_csc.indptr[ci]; k < exon_csc.indptr[ci + 1]; ++k) {
                uint32_t gene_idx = gene_model.exon_to_gene(
                    static_cast<uint32_t>(exon_csc.indices[k]));
                if (gene_idx < n_genes) all_genes.insert(gene_idx);
            }
        }
    }
    const uint64_t total_genes_detected = all_genes.size();

    // ── Write CSV ─────────────────────────────────────────────────────────────
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[metrics_summary] ERROR: cannot write " << path << "\n";
        return;
    }

    out << "Metric,Value\n"
        << "Estimated Number of Cells," << fmt_int(n_cells) << "\n"
        << "Mean Reads per Cell," << fmt_float(mean_reads, 0) << "\n"
        << "Median Genes per Cell," << fmt_float(median_genes, 0) << "\n"
        << "Number of Reads," << fmt_int(total_reads) << "\n"
        << "Valid Barcodes," << fmt_pct(valid_barcodes_pct) << "\n"
        << "Sequencing Saturation," << fmt_pct(saturation_pct) << "\n"
        << "Q30 Bases in Barcode,N/A\n"
        << "Q30 Bases in RNA Read,N/A\n"
        << "Q30 Bases in UMI,N/A\n"
        << "Reads Mapped to Genome," << fmt_pct(mapped_pct) << "\n"
        << "Reads Mapped Confidently to Transcriptome," << fmt_pct(exon_pct) << "\n"
        << "Reads Mapped Confidently to Intronic Regions," << fmt_pct(intron_pct) << "\n"
        << "Fraction Reads in Cells," << fmt_pct(frac_in_cells) << "\n"
        << "Total Genes Detected," << fmt_int(total_genes_detected) << "\n"
        << "Median UMI Counts per Cell," << fmt_float(median_umi, 0) << "\n";

    std::cerr << "[metrics_summary] n_cells=" << n_cells
              << " median_genes=" << static_cast<uint64_t>(median_genes)
              << " median_umi=" << static_cast<uint64_t>(median_umi)
              << " saturation=" << fmt_pct(saturation_pct)
              << " path=" << path << "\n";
}

}  // namespace singlet
