#pragma once
// singlet-pileup: atac_cell_caller.h
// A7 — ATAC-seq cell calling via TSS enrichment + unique fragment count thresholding.
//
// Algorithm:
//   1. Sort barcodes by unique fragment count (descending).
//   2. Auto-threshold (if enabled): log-log rank vs fragment count curve;
//      find inflection point (max second derivative) to set fragment threshold.
//   3. Apply three-way filter:
//        TSS enrichment >= min_tss_enrichment
//        unique fragments >= min_unique_fragments (or auto-threshold)
//        FRIP            >= min_frac_in_peaks / 100.0
//   4. Barcodes passing all three filters → is_cell = true.
//
// Input: pre-computed per-barcode metrics (from ATACQCComputer / ATACFragmentExtractor).
// Output: CellCallResult per barcode + Summary statistics.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// AtacCellCaller
// ─────────────────────────────────────────────────────────────────────────────
class AtacCellCaller {
   public:
    // ── Configuration ────────────────────────────────────────────────────────

    struct Config {
        // tss_enrichment is the FRACTION of unique fragments near any TSS (±1000bp),
        // i.e., tss_fragments / total_fragments ∈ [0, 1].
        // Typical values for real cells: 0.10 – 0.40 (PBMC ATAC: ~0.25–0.35).
        // Background barcodes: < 0.05.
        double min_tss_enrichment = 0.10;     ///< Minimum TSS-proximal fragment fraction [0,1]
        uint32_t min_unique_fragments = 500;  ///< Minimum unique fragments (used when auto disabled)
        uint32_t min_frac_in_peaks = 10;      ///< Minimum FRIP percentage (0-100)
        double quantile_threshold = 0.99;     ///< Fraction of barcodes to consider for auto-threshold
        bool use_auto_threshold = true;       ///< Auto-determine fragment threshold via inflection point
    };

    // ── Per-barcode result ────────────────────────────────────────────────────

    struct CellCallResult {
        std::string barcode;
        uint64_t unique_fragments = 0;
        double tss_enrichment = 0.0;
        double frip = 0.0;  ///< Fraction reads in peaks (0–1)
        bool is_cell = false;
        std::string filter_reason;  ///< "pass", "low_tss", "low_fragments", "low_frip"
    };

    // ── Aggregate summary ─────────────────────────────────────────────────────

    struct Summary {
        uint32_t total_barcodes = 0;
        uint32_t cells_called = 0;
        uint32_t filtered_low_tss = 0;
        uint32_t filtered_low_fragments = 0;
        uint32_t filtered_low_frip = 0;
        double median_tss_cells = 0.0;
        double median_fragments_cells = 0.0;
        uint32_t fragment_threshold = 0;  ///< Threshold actually applied
    };

    // ── Constructor ───────────────────────────────────────────────────────────

    AtacCellCaller() = default;
    explicit AtacCellCaller(Config cfg) : cfg_(cfg) {}

    const Config& config() const { return cfg_; }
    Config& config() { return cfg_; }

    // ── Auto fragment threshold (Otsu bimodal + inflection fallback) ──────────
    //
    // Primary: Otsu's method on log₁₀(fragment_count) distribution to find
    // the optimal split between the "cells" and "empty" modes.
    // This is robust and data-adaptive (no fixed quantile assumptions).
    //
    // Fallback: log-log inflection point (smoothed, to avoid tail noise).
    //
    // Returns the count threshold; always ≥ min_unique_fragments.

    uint32_t auto_fragment_threshold(const std::vector<uint64_t>& counts) const {
        if (counts.empty()) return cfg_.min_unique_fragments;

        // Build sorted (descending) copy, filter zeros
        std::vector<uint64_t> sorted;
        sorted.reserve(counts.size());
        for (auto c : counts)
            if (c > 0) sorted.push_back(c);

        if (sorted.empty()) return cfg_.min_unique_fragments;

        std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());

        // ── Otsu's method on log₁₀(count) histogram ─────────────────────────
        // Bin log₁₀(count) into 100 bins spanning [0, max_log10]
        const double max_log = std::log10(static_cast<double>(sorted.front()));
        const double min_log = std::log10(static_cast<double>(sorted.back()));
        const int n_bins = 100;
        const double bin_width = (max_log - min_log) / n_bins;

        if (bin_width <= 0.0 || sorted.size() < 10) {
            // Degenerate: all same value or too few barcodes
            return std::max(static_cast<uint32_t>(sorted.back()), cfg_.min_unique_fragments);
        }

        // Build histogram
        std::vector<int> hist(n_bins, 0);
        for (uint64_t c : sorted) {
            int bin = static_cast<int>((std::log10(static_cast<double>(c)) - min_log) / bin_width);
            bin = std::max(0, std::min(n_bins - 1, bin));
            ++hist[bin];
        }

        // Otsu: find threshold that maximizes between-class variance
        const int total = static_cast<int>(sorted.size());
        double sum_all = 0.0;
        for (int i = 0; i < n_bins; ++i)
            sum_all += static_cast<double>(i) * hist[i];

        int w0 = 0;
        double sum0 = 0.0;
        double best_var = -1.0;
        int best_bin = n_bins / 2;

        for (int t = 0; t < n_bins - 1; ++t) {
            w0 += hist[t];
            if (w0 == 0) continue;
            int w1 = total - w0;
            if (w1 == 0) break;

            sum0 += static_cast<double>(t) * hist[t];
            double mu0 = sum0 / w0;
            double mu1 = (sum_all - sum0) / w1;
            double var = static_cast<double>(w0) * w1 * (mu0 - mu1) * (mu0 - mu1);

            if (var > best_var) {
                best_var = var;
                best_bin = t;
            }
        }

        // Separability check: require the two class means to be at least
        // 0.5 log₁₀ units apart (~3× in count space).  Unimodal or narrow
        // distributions won't meet this threshold.
        {
            int w0_f = 0;
            double s0_f = 0.0;
            for (int i = 0; i <= best_bin; ++i) {
                w0_f += hist[i];
                s0_f += static_cast<double>(i) * hist[i];
            }
            int w1_f = total - w0_f;
            if (w0_f > 0 && w1_f > 0) {
                double mu0_log = min_log + (s0_f / w0_f) * bin_width;
                double mu1_log = min_log + ((sum_all - s0_f) / w1_f) * bin_width;
                if (std::abs(mu1_log - mu0_log) < 0.5) {
                    return cfg_.min_unique_fragments;
                }
            }
        }

        // Convert bin back to count threshold
        double threshold_log = min_log + (best_bin + 0.5) * bin_width;
        uint32_t threshold = static_cast<uint32_t>(std::pow(10.0, threshold_log));

        // Sanity: Otsu threshold must be between the 5th and 95th percentile
        // of the data. If outside, fall back to min_unique_fragments.
        uint64_t p5 = sorted[std::min(sorted.size() - 1, sorted.size() * 95 / 100)];
        uint64_t p95 = sorted[sorted.size() * 5 / 100];
        if (threshold < p5 || threshold > p95) {
            // Otsu failed (unimodal distribution) — use min_unique_fragments
            return cfg_.min_unique_fragments;
        }

        return std::max(threshold, cfg_.min_unique_fragments);
    }

    // ── Main cell calling function ────────────────────────────────────────────
    //
    // All input vectors must have the same length (one entry per barcode).
    // frip_scores should be in [0, 1] range.
    // Results are appended in input order (not sorted by rank).

    Summary call_cells(
        const std::vector<std::string>& barcodes,
        const std::vector<uint64_t>& unique_fragment_counts,
        const std::vector<double>& tss_enrichment_scores,
        const std::vector<double>& frip_scores,
        std::vector<CellCallResult>& results) {
        const size_t N = barcodes.size();

        Summary summary;
        summary.total_barcodes = static_cast<uint32_t>(N);
        results.clear();
        results.reserve(N);

        if (N == 0) return summary;

        // Determine fragment threshold
        uint32_t frag_thresh = cfg_.min_unique_fragments;
        if (cfg_.use_auto_threshold) {
            frag_thresh = auto_fragment_threshold(unique_fragment_counts);
        }
        summary.fragment_threshold = frag_thresh;

        const double frip_min = cfg_.min_frac_in_peaks / 100.0;

        // Classify each barcode
        std::vector<double> cell_tss;
        std::vector<double> cell_frags;
        cell_tss.reserve(N);
        cell_frags.reserve(N);

        for (size_t i = 0; i < N; ++i) {
            CellCallResult r;
            r.barcode = barcodes[i];
            r.unique_fragments = unique_fragment_counts[i];
            r.tss_enrichment = tss_enrichment_scores[i];
            r.frip = frip_scores[i];

            bool low_frags = (r.unique_fragments < frag_thresh);
            bool low_tss = (r.tss_enrichment < cfg_.min_tss_enrichment);
            bool low_frip = (r.frip < frip_min);

            if (!low_frags && !low_tss && !low_frip) {
                r.is_cell = true;
                r.filter_reason = "pass";
                ++summary.cells_called;
                cell_tss.push_back(r.tss_enrichment);
                cell_frags.push_back(static_cast<double>(r.unique_fragments));
            } else {
                r.is_cell = false;
                // Report the first failing filter (priority: fragments > tss > frip)
                if (low_frags) {
                    r.filter_reason = "low_fragments";
                    ++summary.filtered_low_fragments;
                } else if (low_tss) {
                    r.filter_reason = "low_tss";
                    ++summary.filtered_low_tss;
                } else {
                    r.filter_reason = "low_frip";
                    ++summary.filtered_low_frip;
                }
            }
            results.push_back(std::move(r));
        }

        // Compute medians for called cells
        if (!cell_tss.empty()) {
            std::sort(cell_tss.begin(), cell_tss.end());
            std::sort(cell_frags.begin(), cell_frags.end());
            size_t mid = cell_tss.size() / 2;
            summary.median_tss_cells = (cell_tss.size() % 2 == 0)
                                           ? (cell_tss[mid - 1] + cell_tss[mid]) * 0.5
                                           : cell_tss[mid];
            summary.median_fragments_cells = (cell_frags.size() % 2 == 0)
                                                 ? (cell_frags[mid - 1] + cell_frags[mid]) * 0.5
                                                 : cell_frags[mid];
        }

        return summary;
    }

   private:
    Config cfg_;
};

}  // namespace singlet
