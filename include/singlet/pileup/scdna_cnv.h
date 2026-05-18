// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: scdna_cnv.h
// G-SCDNA — Single-cell DNA copy-number variant (CNV) bin counting and QC.
//
// Provides cell-level QC for scDNA-seq experiments:
//   - MAPD (Median Absolute Pairwise Difference) — noise metric
//   - Ploidy estimation from count distribution
//   - GC correction (binned linear-regression approach)
//   - Per-cell QC JSON output
//
// No htslib dependency — operates on integer bin-count vectors so it can
// be called from the pileup engine without BAM reading infrastructure.
//
// Reference:
//   Garvin et al. 2015 (MAPD metric)
//   Navin et al. 2011 (scDNA-seq copy-number profiling)
//   Baslan et al. 2012 (GC correction for scDNA-seq)
//
// Integration: see INTEGRATION_NOTES.md § G-SCDNA

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

struct ScDnaCnvConfig {
    uint32_t bin_size = 500000;  // 500 kb default
    bool gc_correction = true;
    uint32_t min_mapq = 20;
};

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell QC
// ─────────────────────────────────────────────────────────────────────────────

struct CnvCellQC {
    double mapd = 0.0;  // Median Absolute Pairwise Difference (noise)
    double ploidy_estimate = 0.0;
    uint64_t total_reads = 0;
    uint64_t unique_reads = 0;
    double coverage = 0.0;  // mean bin read count
};

// ─────────────────────────────────────────────────────────────────────────────
// MAPD computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute MAPD from a vector of per-bin read counts.
///
/// MAPD = median of |log2(count[i]+1) - log2(count[i-1]+1)| over all adjacent
/// pairs. Pseudocount of 1 avoids log2(0). Lower values indicate less noise.
///
/// ENCODE / clinical thresholds: MAPD < 0.2 → low noise; > 0.4 → high noise.
inline double compute_mapd(const std::vector<int32_t>& bin_counts) {
    const size_t n = bin_counts.size();
    if (n < 2) return 0.0;

    std::vector<double> diffs;
    diffs.reserve(n - 1);
    for (size_t i = 1; i < n; ++i) {
        double a = std::log2(static_cast<double>(std::max(0, bin_counts[i - 1])) + 1.0);
        double b = std::log2(static_cast<double>(std::max(0, bin_counts[i])) + 1.0);
        diffs.push_back(std::abs(b - a));
    }

    size_t mid = diffs.size() / 2;
    std::nth_element(diffs.begin(), diffs.begin() + static_cast<ptrdiff_t>(mid), diffs.end());
    if (diffs.size() % 2 == 1) {
        return diffs[mid];
    } else {
        // Even count: average two middle values
        double hi = diffs[mid];
        std::nth_element(diffs.begin(),
                         diffs.begin() + static_cast<ptrdiff_t>(mid) - 1,
                         diffs.begin() + static_cast<ptrdiff_t>(mid));
        return (diffs[mid - 1] + hi) * 0.5;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ploidy estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate ploidy from the modal bin count relative to the expected diploid count.
///
/// Procedure:
///   1. Compute ratio = count / expected_diploid for each bin.
///   2. Round each ratio to the nearest 0.5 (CNV resolution).
///   3. Return the mode of rounded ratios × 2 (copy-number estimate).
///
/// @param bin_counts           Read counts per genomic bin.
/// @param expected_diploid_count  Expected mean count for a diploid (CN=2) bin.
///                                Pass 0 to use the mean of bin_counts.
inline double estimate_ploidy(const std::vector<int32_t>& bin_counts,
                              double expected_diploid_count) {
    if (bin_counts.empty()) return 0.0;

    // Auto-compute expected if not provided
    if (expected_diploid_count <= 0.0) {
        double sum = 0.0;
        int cnt = 0;
        for (auto v : bin_counts) {
            if (v > 0) {
                sum += v;
                ++cnt;
            }
        }
        if (cnt == 0) return 2.0;
        expected_diploid_count = sum / static_cast<double>(cnt);
    }

    // Build histogram of rounded copy-number ratios
    // Ratio = count / (expected/2): normalise so diploid bins → 2.0
    const double scale = (expected_diploid_count > 0.0)
                             ? 2.0 / expected_diploid_count
                             : 1.0;

    std::vector<int32_t> cn_rounded;
    cn_rounded.reserve(bin_counts.size());
    for (auto v : bin_counts) {
        if (v < 0) continue;
        double cn = static_cast<double>(v) * scale;
        // Round to nearest integer (whole copy-number)
        int32_t cn_int = static_cast<int32_t>(std::round(cn));
        if (cn_int < 0) cn_int = 0;
        cn_rounded.push_back(cn_int);
    }
    if (cn_rounded.empty()) return 2.0;

    // Mode of the distribution
    std::sort(cn_rounded.begin(), cn_rounded.end());
    int32_t mode_cn = cn_rounded[cn_rounded.size() / 2];  // fallback: median
    int32_t best_count = 0;
    int32_t cur_val = cn_rounded[0];
    int32_t cur_count = 0;
    for (auto v : cn_rounded) {
        if (v == cur_val) {
            ++cur_count;
        } else {
            if (cur_count > best_count) {
                best_count = cur_count;
                mode_cn = cur_val;
            }
            cur_val = v;
            cur_count = 1;
        }
    }
    if (cur_count > best_count) mode_cn = cur_val;

    return static_cast<double>(mode_cn);
}

// ─────────────────────────────────────────────────────────────────────────────
// GC correction (binned linear-regression approach)
// ─────────────────────────────────────────────────────────────────────────────

/// GC-correct genomic bin counts using a linear regression of log(count+1) ~ GC.
///
/// Procedure:
///   1. Fit: B0 + B1 * gc_content[i]  →  log2(count[i]+1)  (ordinary least squares).
///   2. Predicted[i] = B0 + B1 * gc_content[i].
///   3. Corrected[i] = count[i] / exp2(Predicted[i]) * exp2(mean_predicted).
///
/// Bins with count == 0 are included in the regression (contributes log2(1)=0).
///
/// @param bin_counts  Raw read counts per bin (may be negative: treated as 0).
/// @param gc_content  GC fraction [0,1] for each bin (must be same length as bin_counts).
/// @returns           GC-corrected counts as double (may be fractional).
inline std::vector<double> gc_correct_bins(const std::vector<int32_t>& bin_counts,
                                           const std::vector<double>& gc_content) {
    const size_t n = bin_counts.size();
    if (n == 0 || gc_content.size() != n) {
        return std::vector<double>(n, 0.0);
    }

    // OLS: y = B0 + B1 * x  where y = log2(count+1), x = gc_content
    double sum_x = 0.0, sum_y = 0.0;
    double sum_xx = 0.0, sum_xy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double x = gc_content[i];
        double y = std::log2(static_cast<double>(std::max(0, bin_counts[i])) + 1.0);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    const double fn = static_cast<double>(n);
    const double denom = fn * sum_xx - sum_x * sum_x;

    double b1 = (std::abs(denom) > 1e-12)
                    ? (fn * sum_xy - sum_x * sum_y) / denom
                    : 0.0;
    double b0 = (sum_y - b1 * sum_x) / fn;

    // Compute mean predicted value (used to re-centre after correction)
    double mean_pred = b0 + b1 * (sum_x / fn);

    std::vector<double> corrected(n);
    for (size_t i = 0; i < n; ++i) {
        double raw = static_cast<double>(std::max(0, bin_counts[i]));
        double pred = b0 + b1 * gc_content[i];  // log2 space prediction
        // Move count to have the same expected signal regardless of GC
        double scale = std::exp2(mean_pred - pred);  // = 2^(mean - pred)
        corrected[i] = raw * scale;
    }
    return corrected;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell QC helper
// ─────────────────────────────────────────────────────────────────────────────

/// Compute CnvCellQC for a single cell given its bin counts.
inline CnvCellQC compute_cell_cnv_qc(const std::vector<int32_t>& bin_counts,
                                     uint64_t total_reads = 0,
                                     uint64_t unique_reads = 0) {
    CnvCellQC qc;
    qc.total_reads = total_reads;
    qc.unique_reads = unique_reads;
    qc.mapd = compute_mapd(bin_counts);

    if (!bin_counts.empty()) {
        double sum = 0.0;
        for (auto v : bin_counts) sum += std::max(0, v);
        qc.coverage = sum / static_cast<double>(bin_counts.size());
    }

    qc.ploidy_estimate = estimate_ploidy(bin_counts, 0.0 /* auto */);
    return qc;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON output (per-cell)
// ─────────────────────────────────────────────────────────────────────────────

/// Write per-cell CNV QC metrics to a JSON file at `path`.
/// Output format: array of cell objects with barcode + metrics.
/// Returns true on success.
inline bool write_cnv_qc(const std::string& path,
                         const std::vector<CnvCellQC>& cell_qc,
                         const std::vector<std::string>& barcodes) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[scdna_cnv] Cannot open for writing: " << path << "\n";
        return false;
    }

    const size_t n = cell_qc.size();
    f << "[\n";
    for (size_t i = 0; i < n; ++i) {
        const auto& qc = cell_qc[i];
        const std::string& bc = (i < barcodes.size()) ? barcodes[i] : "";
        f << "  {\n"
          << "    \"barcode\": \"" << bc << "\",\n"
          << "    \"mapd\": " << qc.mapd << ",\n"
          << "    \"ploidy_estimate\": " << qc.ploidy_estimate << ",\n"
          << "    \"total_reads\": " << qc.total_reads << ",\n"
          << "    \"unique_reads\": " << qc.unique_reads << ",\n"
          << "    \"coverage\": " << qc.coverage << "\n"
          << "  }";
        if (i + 1 < n) f << ",";
        f << "\n";
    }
    f << "]\n";
    return true;
}

}  // namespace singlet
