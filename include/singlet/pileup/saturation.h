// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: saturation.h
// N7: Sequencing saturation metrics — CellRanger-compatible definition.
//
// saturation = 1 - (unique_umis / total_reads) per cell
// Computed directly from DirectionalUmiStore after UMI dedup — no extra BAM pass.
//
// Outputs: saturation_metrics.tsv
//   barcode | total_reads | unique_umis | saturation | reads_per_umi

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "umi_dedup.h"

namespace singlet {

/// Per-cell sequencing saturation metrics.
struct CellSaturation {
    uint32_t barcode_idx = 0;
    uint64_t total_reads = 0;   ///< Raw reads mapping to any gene in this cell
    uint64_t unique_umis = 0;   ///< Unique molecules after directional dedup
    double saturation = 0.0;    ///< 1 - (unique_umis / total_reads)
    double reads_per_umi = 0.0; ///< total_reads / unique_umis
};

/// Compute per-cell saturation from a DirectionalUmiStore.
///
/// Uses per_cell_stats() — same dedup logic as finalize(), no extra BAM pass.
/// Only valid after the BAM pileup loop completes (groups are fully populated).
///
/// @param store    Completed DirectionalUmiStore (exon OR exon+intron)
/// @param umi_len  UMI length in bases (for Hamming distance masking)
/// @returns        Per-cell saturation, sorted by barcode_idx
inline std::vector<CellSaturation> compute_saturation(
    const DirectionalUmiStore& store,
    int umi_len)
{
    auto cell_stats = store.per_cell_stats(umi_len);

    std::vector<CellSaturation> result;
    result.reserve(cell_stats.size());

    for (const auto& [bc_idx, stats] : cell_stats) {
        CellSaturation cs;
        cs.barcode_idx = bc_idx;
        cs.total_reads = stats.total_reads;
        cs.unique_umis = stats.unique_umis;
        if (stats.total_reads > 0 && stats.unique_umis > 0) {
            cs.saturation = 1.0 - static_cast<double>(stats.unique_umis)
                                / static_cast<double>(stats.total_reads);
            cs.reads_per_umi = static_cast<double>(stats.total_reads)
                             / static_cast<double>(stats.unique_umis);
        }
        result.push_back(cs);
    }

    std::sort(result.begin(), result.end(),
        [](const CellSaturation& a, const CellSaturation& b) {
            return a.barcode_idx < b.barcode_idx;
        });

    return result;
}

/// Compute median saturation across all cells with at least one read.
inline double median_saturation(std::vector<CellSaturation>& cells) {
    std::vector<double> sats;
    sats.reserve(cells.size());
    for (const auto& c : cells)
        if (c.total_reads > 0) sats.push_back(c.saturation);

    if (sats.empty()) return 0.0;
    const size_t n = sats.size();
    std::nth_element(sats.begin(), sats.begin() + n / 2, sats.end());
    if (n % 2 == 1) return sats[n / 2];
    const double upper = sats[n / 2];
    std::nth_element(sats.begin(), sats.begin() + n / 2 - 1, sats.end());
    return (sats[n / 2 - 1] + upper) * 0.5;
}

/// Write saturation_metrics.tsv.
/// Columns: barcode, total_reads, unique_umis, saturation, reads_per_umi
inline void write_saturation_tsv(
    const std::string& path,
    const std::vector<CellSaturation>& cells,
    const std::vector<std::string>& barcodes)
{
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[saturation] Warning: cannot write " << path << "\n";
        return;
    }
    f << "barcode\ttotal_reads\tunique_umis\tsaturation\treads_per_umi\n";
    for (const auto& c : cells) {
        const std::string& bc = (c.barcode_idx < static_cast<uint32_t>(barcodes.size()))
                                ? barcodes[c.barcode_idx]
                                : std::to_string(c.barcode_idx);
        f << bc << '\t' << c.total_reads << '\t' << c.unique_umis
          << '\t' << c.saturation << '\t' << c.reads_per_umi << '\n';
    }
    std::cerr << "[saturation] Wrote saturation_metrics.tsv: "
              << cells.size() << " cells\n";
}

} // namespace singlet
