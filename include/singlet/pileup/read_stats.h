// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: read_stats.h  (N20)
// Per-cell read statistics: total reads, unique UMIs, duplication rate,
// and Lander-Waterman library complexity estimate.
//
// All values are derived from data already collected during pileup — zero
// additional BAM passes required.
//
//   total_reads[bc]   = barcoded, mapped primary reads passing barcode filter
//                       (before UMI dedup, before MAPQ filter on multimappers)
//   unique_umis[bc]   = sum of exon+intron count matrix columns (post-dedup UMIs)
//   dup_reads         = total_reads - unique_umis  (reads lost to UMI dedup)
//   dup_rate          = dup_reads / total_reads
//   est_complexity    = Lander-Waterman: n²/(2·dup_reads) where n = total_reads
//                       (clamped to [unique_umis, UINT64_MAX/4] for safety)

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace singlet {

struct CellReadStats {
    uint32_t total_reads = 0;     ///< Barcoded mapped primary reads (pre-dedup)
    uint32_t unique_umis = 0;     ///< Post-dedup UMIs (exon + intron column sum)
    uint32_t dup_reads = 0;       ///< total_reads - unique_umis (floor 0)
    float dup_rate = 0.0f;        ///< dup_reads / total_reads (0 if total==0)
    uint64_t est_complexity = 0;  ///< Lander-Waterman library size estimate
};

/// Compute per-cell Lander-Waterman library complexity.
/// Formula: C ≈ n² / (2 · d)  where n = total reads, d = duplicate reads.
/// Returns n (= unique_umis lower bound) when d == 0.
static inline uint64_t lander_waterman(uint32_t n_total, uint32_t n_dup) {
    if (n_dup == 0) return static_cast<uint64_t>(n_total);
    // Use uint64_t to avoid overflow for n up to ~65k
    uint64_t n = static_cast<uint64_t>(n_total);
    uint64_t d = static_cast<uint64_t>(n_dup);
    // Guard against absurd values (shouldn't happen but be safe)
    if (d >= n) return static_cast<uint64_t>(n_total);
    return (n * n) / (2 * d);
}

/// Compute per-cell read statistics.
///
/// @param per_cell_reads  Per-barcode read counts (from PileupEngine::per_cell_reads())
/// @param exon_indptr     CSC column pointer array from exon count matrix (size ncols+1)
/// @param exon_data       CSC value array from exon count matrix
/// @param intron_indptr   CSC column pointer array from intron matrix (nullptr if no introns)
/// @param intron_data     CSC value array from intron matrix (empty if no introns)
/// @param ncols           Number of barcodes (columns)
/// @return                Per-barcode vector of CellReadStats
inline std::vector<CellReadStats> compute_read_stats(
    const std::vector<uint32_t>& per_cell_reads,
    const int32_t* exon_indptr,
    const uint16_t* exon_data,
    const int32_t* intron_indptr,  // may be nullptr
    const uint16_t* intron_data,   // may be nullptr
    uint32_t ncols) {
    std::vector<CellReadStats> out(ncols);

    for (uint32_t j = 0; j < ncols; ++j) {
        uint64_t umi_sum = 0;

        // Sum exon column
        for (int32_t k = exon_indptr[j]; k < exon_indptr[j + 1]; ++k)
            umi_sum += static_cast<uint64_t>(exon_data[k]);

        // Sum intron column (if available)
        if (intron_indptr && intron_data) {
            for (int32_t k = intron_indptr[j]; k < intron_indptr[j + 1]; ++k)
                umi_sum += static_cast<uint64_t>(intron_data[k]);
        }

        uint32_t total = (j < per_cell_reads.size()) ? per_cell_reads[j] : 0;
        uint32_t unique = static_cast<uint32_t>(std::min(umi_sum,
                                                         static_cast<uint64_t>(UINT32_MAX)));
        // dup_reads can't be negative; clamp at 0 (total < unique can happen for
        // multimappers counted in unique but not in barcoded_reads counter path)
        uint32_t dup = (total > unique) ? (total - unique) : 0;
        float rate = (total > 0) ? static_cast<float>(dup) / static_cast<float>(total) : 0.0f;

        out[j] = {total, unique, dup, rate, lander_waterman(total, dup)};
    }
    return out;
}

/// Write per-cell read statistics to a TSV file.
///
/// Columns: barcode, total_reads, unique_umis, dup_reads, dup_rate, est_complexity
inline bool write_read_stats_tsv(
    const std::string& path,
    const std::vector<CellReadStats>& stats,
    const std::vector<std::string>& barcodes) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[read_stats] ERROR: Cannot write " << path << "\n";
        return false;
    }
    f << "barcode\ttotal_reads\tunique_umis\tdup_reads\tdup_rate\test_complexity\n";
    f << std::fixed << std::setprecision(4);

    uint32_t n = static_cast<uint32_t>(
        std::min(stats.size(), barcodes.size()));
    for (uint32_t i = 0; i < n; ++i) {
        const auto& s = stats[i];
        f << barcodes[i] << '\t'
          << s.total_reads << '\t'
          << s.unique_umis << '\t'
          << s.dup_reads << '\t'
          << s.dup_rate << '\t'
          << s.est_complexity << '\n';
    }
    return f.good();
}

/// Compute median duplication rate across all cells.
/// Returns -1.0 if stats is empty.
inline double median_dup_rate(const std::vector<CellReadStats>& stats) {
    if (stats.empty()) return -1.0;
    std::vector<float> rates;
    rates.reserve(stats.size());
    for (const auto& s : stats)
        if (s.total_reads > 0) rates.push_back(s.dup_rate);
    if (rates.empty()) return 0.0;
    std::nth_element(rates.begin(), rates.begin() + rates.size() / 2, rates.end());
    return static_cast<double>(rates[rates.size() / 2]);
}

// ─── Per-stage pipeline read statistics (§3.2 schema) ─────────────────────────

/// One row per processing stage in the pipeline.
struct PipelineStageStats {
    std::string stage;        ///< "encode", "align", "pileup"
    uint64_t reads_in = 0;   ///< Reads entering this stage
    uint64_t reads_out = 0;  ///< Reads passing this stage
    std::string fail_reason;  ///< Empty if no failure; else e.g. "unmapped", "no_barcode"
    uint64_t bytes_in = 0;   ///< Input bytes (0 if not tracked)
    uint64_t bytes_out = 0;  ///< Output bytes (0 if not tracked)
    double wall_seconds = 0.0;
};

/// Write per-stage read statistics to read_stats.tsv (§3.2 schema).
inline bool write_stage_read_stats_tsv(
    const std::string& path,
    const std::vector<PipelineStageStats>& stages) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[read_stats] ERROR: Cannot write " << path << "\n";
        return false;
    }
    f << "stage\treads_in\treads_out\tfail_reason\tbytes_in\tbytes_out\twall_seconds\n";
    f << std::fixed << std::setprecision(2);
    for (const auto& s : stages) {
        f << s.stage << '\t'
          << s.reads_in << '\t'
          << s.reads_out << '\t'
          << s.fail_reason << '\t'
          << s.bytes_in << '\t'
          << s.bytes_out << '\t'
          << s.wall_seconds << '\n';
    }
    return f.good();
}

}  // namespace singlet
