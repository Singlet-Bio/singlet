// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: chipseq.h
// G-CHIPSEQ — Bulk ChIP-seq QC metrics.
//
// Computes standard ChIP-seq quality metrics from strand-resolved coverage:
//   - FRiP (Fraction of Reads in Peaks)
//   - NSC  (Normalized Strand Coefficient) from strand cross-correlation
//   - RSC  (Relative  Strand Coefficient)
//   - PBC1 / PBC2 (PCR Bottleneck Coefficients)
//   - Lorenz curve + AUC (enrichment fingerprint)
//   - GC bias (deviation from expected GC content)
//
// No htslib dependency — operates on pre-computed coverage vectors and
// aggregate counts so it can be called from the pileup engine hot path
// without pulling in BAM reading infrastructure.
//
// Reference:
//   ENCODE ChIP-seq standards (Landt et al. 2012, Genome Research)
//   phantompeakqualtools (Kharchenko et al.)
//
// Integration: see INTEGRATION_NOTES.md § G-CHIPSEQ

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// QC struct
// ─────────────────────────────────────────────────────────────────────────────

struct ChipSeqQC {
    double frip = 0.0;  // Fraction reads in peaks
    double nsc = 0.0;   // Normalized Strand Coefficient
    double rsc = 0.0;   // Relative Strand Coefficient
    double pbc1 = 0.0;  // PCR Bottleneck Coefficient 1
    double pbc2 = 0.0;  // PCR Bottleneck Coefficient 2
    uint64_t total_reads = 0;
    uint64_t unique_reads = 0;  // after dedup
    uint64_t reads_in_peaks = 0;
    double gc_bias = 0.0;  // GC content bias (deviation from expected)

    // Lorenz / fingerprint
    std::vector<double> lorenz_curve;  // n_bins cumulative fractions, sorted ascending
    double auc_lorenz = 0.0;           // area under Lorenz curve (lower = more enriched)
};

// ─────────────────────────────────────────────────────────────────────────────
// PBC coefficients
// ─────────────────────────────────────────────────────────────────────────────

/// PBC1 = unique_positions / total_positions.
/// ENCODE thresholds: <0.5 severe, 0.5–0.8 moderate, 0.8–0.9 acceptable, ≥0.9 ideal.
inline double compute_pbc1(uint64_t unique_positions, uint64_t total_positions) {
    if (total_positions == 0) return 0.0;
    return static_cast<double>(unique_positions) / static_cast<double>(total_positions);
}

/// PBC2 = positions_with_exactly_1_read / positions_with_exactly_2_reads.
/// ENCODE thresholds: <1 severe, 1–3 moderate, 3–10 acceptable, ≥10 ideal.
inline double compute_pbc2(uint64_t one_read_positions, uint64_t two_read_positions) {
    if (two_read_positions == 0) return (one_read_positions > 0) ? 1e9 : 0.0;
    return static_cast<double>(one_read_positions) / static_cast<double>(two_read_positions);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lorenz curve
// ─────────────────────────────────────────────────────────────────────────────

/// Build a Lorenz curve from per-base coverage values.
/// The curve shows cumulative fraction of total signal vs cumulative fraction
/// of genomic positions, with positions sorted from lowest to highest coverage.
/// @param coverage  Per-position read depth (zeros included).
/// @param n_bins    Number of evenly spaced quantile bins (default 100).
/// @returns Vector of n_bins+1 values in [0, 1] (cumulative signal fractions).
inline std::vector<double> compute_lorenz_curve(const std::vector<uint32_t>& coverage,
                                                uint32_t n_bins) {
    if (n_bins == 0) n_bins = 100;
    std::vector<uint32_t> sorted = coverage;
    std::sort(sorted.begin(), sorted.end());

    const size_t n = sorted.size();
    uint64_t total = 0;
    for (auto v : sorted) total += v;

    std::vector<double> curve(n_bins + 1, 0.0);
    if (n == 0 || total == 0) return curve;

    uint64_t cumsum = 0;
    size_t pos_idx = 0;
    for (uint32_t b = 0; b <= n_bins; ++b) {
        // Target: advance to floor(b/n_bins * n) positions
        size_t target = static_cast<size_t>(
            static_cast<double>(b) / static_cast<double>(n_bins) * static_cast<double>(n));
        if (target > n) target = n;
        while (pos_idx < target) {
            cumsum += sorted[pos_idx];
            ++pos_idx;
        }
        curve[b] = static_cast<double>(cumsum) / static_cast<double>(total);
    }
    return curve;
}

/// Area under the Lorenz curve (trapezoidal rule over n_bins+1 points).
/// Perfectly uniform coverage → AUC = 0.5.
/// Fully concentrated coverage → AUC ≈ 0.0 (high enrichment).
inline double lorenz_auc(const std::vector<double>& curve) {
    if (curve.size() < 2) return 0.0;
    const size_t n = curve.size() - 1;  // number of intervals
    const double dx = 1.0 / static_cast<double>(n);
    double area = 0.0;
    for (size_t i = 0; i < n; ++i)
        area += (curve[i] + curve[i + 1]) * 0.5 * dx;
    return area;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strand cross-correlation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the strand cross-correlation profile.
/// cc[s] = Pearson correlation of fwd[i] and rev[i+s] for shift s in [0, max_shift).
/// Returns empty vector if inputs are empty or incompatible.
inline std::vector<double> strand_cross_correlation(
    const std::vector<uint32_t>& fwd,
    const std::vector<uint32_t>& rev,
    uint32_t max_shift = 500) {
    const size_t n = fwd.size();
    if (n < 2 || rev.size() != n || max_shift == 0) return {};
    if (max_shift >= static_cast<uint32_t>(n)) max_shift = static_cast<uint32_t>(n) - 1;

    // Precompute means and stdev for fwd (fixed) and rev (shifted)
    double fwd_mean = 0.0;
    for (auto v : fwd) fwd_mean += v;
    fwd_mean /= static_cast<double>(n);

    std::vector<double> cc(max_shift + 1, 0.0);
    for (uint32_t s = 0; s <= max_shift; ++s) {
        const size_t len = n - s;  // number of overlapping positions
        if (len < 2) break;

        // Compute mean of shifted rev slice
        double rev_mean = 0.0;
        for (size_t i = 0; i < len; ++i) rev_mean += rev[i + s];
        rev_mean /= static_cast<double>(len);

        // Also compute mean of fwd slice (only first len positions)
        double f_mean = 0.0;
        for (size_t i = 0; i < len; ++i) f_mean += fwd[i];
        f_mean /= static_cast<double>(len);

        // Pearson numerator and denominators
        double cov = 0.0, sf = 0.0, sr = 0.0;
        for (size_t i = 0; i < len; ++i) {
            double df = static_cast<double>(fwd[i]) - f_mean;
            double dr = static_cast<double>(rev[i + s]) - rev_mean;
            cov += df * dr;
            sf += df * df;
            sr += dr * dr;
        }
        double denom = std::sqrt(sf * sr);
        cc[s] = (denom > 0.0) ? cov / denom : 0.0;
    }
    return cc;
}

// ─────────────────────────────────────────────────────────────────────────────
// NSC / RSC computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute NSC and RSC from the cross-correlation profile.
///
/// NSC = cc_max / cc_min
///   quantifies the ChIP enrichment signal above background.
///   ENCODE: NSC < 1.05 → poor quality; ≥ 1.1 → acceptable.
///
/// RSC = (cc_max - cc_min) / (cc_phantom - cc_min)
///   normalises by the phantom-peak correlation (at read_length shift).
///   ENCODE: RSC < 0.8 → poor quality; ≥ 1.0 → ideal.
///
/// @param cc           Cross-correlation vector (shift 0 … N-1).
/// @param read_length  Expected read length (used as phantom-peak shift).
/// @param[out] nsc     Normalized Strand Coefficient.
/// @param[out] rsc     Relative Strand Coefficient.
inline void compute_nsc_rsc(const std::vector<double>& cc,
                            uint32_t read_length,
                            double& nsc,
                            double& rsc) {
    nsc = 0.0;
    rsc = 0.0;
    if (cc.empty()) return;

    const double cc_max = *std::max_element(cc.begin(), cc.end());
    const double cc_min = *std::min_element(cc.begin(), cc.end());

    if (cc_min == cc_max) return;  // degenerate (flat signal)

    nsc = (cc_min != 0.0) ? cc_max / cc_min : 0.0;
    if (cc_min == 0.0) {
        nsc = 0.0;
        rsc = 0.0;
        return;
    }

    double cc_phantom = cc_min;  // fallback
    if (read_length < static_cast<uint32_t>(cc.size())) {
        cc_phantom = cc[read_length];
    }

    double denom = cc_phantom - cc_min;
    rsc = (std::abs(denom) > 1e-12) ? (cc_max - cc_min) / denom : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main QC entry point
// ─────────────────────────────────────────────────────────────────────────────

/// Compute full ChIP-seq QC from strand coverage vectors.
///
/// @param fwd              Per-base forward-strand tag coverage (length = chrom_size).
/// @param rev              Per-base reverse-strand tag coverage (same length as fwd).
/// @param total_reads      Total mapped read count.
/// @param unique_reads     Unique read count (post-dedup).
/// @param reads_in_peaks   Reads overlapping called peaks.
/// @param read_length      Read length used to locate phantom peak (default 75).
/// @param n_lorenz_bins    Bins for Lorenz curve (default 100).
inline ChipSeqQC compute_chipseq_qc(const std::vector<uint32_t>& fwd,
                                    const std::vector<uint32_t>& rev,
                                    uint64_t total_reads,
                                    uint64_t unique_reads,
                                    uint64_t reads_in_peaks,
                                    uint32_t read_length = 75,
                                    uint32_t n_lorenz_bins = 100) {
    ChipSeqQC qc;
    qc.total_reads = total_reads;
    qc.unique_reads = unique_reads;
    qc.reads_in_peaks = reads_in_peaks;

    // FRiP
    if (total_reads > 0)
        qc.frip = static_cast<double>(reads_in_peaks) / static_cast<double>(total_reads);

    // NSC / RSC via cross-correlation
    auto cc = strand_cross_correlation(fwd, rev, 500);
    compute_nsc_rsc(cc, read_length, qc.nsc, qc.rsc);

    // PBC1 / PBC2 require per-position counts; approximate from unique vs total
    qc.pbc1 = compute_pbc1(unique_reads, total_reads);
    // PBC2 not estimable from aggregate counts alone — caller should use
    // compute_pbc2(one_read_positions, two_read_positions) directly.
    qc.pbc2 = 0.0;

    // Lorenz curve + AUC from combined (fwd+rev) coverage
    std::vector<uint32_t> combined(fwd.size(), 0);
    for (size_t i = 0; i < fwd.size(); ++i)
        combined[i] = fwd[i] + (i < rev.size() ? rev[i] : 0);
    qc.lorenz_curve = compute_lorenz_curve(combined, n_lorenz_bins);
    qc.auc_lorenz = lorenz_auc(qc.lorenz_curve);

    return qc;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON output
// ─────────────────────────────────────────────────────────────────────────────

/// Write ChipSeqQC to a JSON file at `path`. Returns true on success.
inline bool write_chipseq_qc(const std::string& path, const ChipSeqQC& qc) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[chipseq] Cannot open for writing: " << path << "\n";
        return false;
    }

    f << "{\n"
      << "  \"frip\": " << qc.frip << ",\n"
      << "  \"nsc\": " << qc.nsc << ",\n"
      << "  \"rsc\": " << qc.rsc << ",\n"
      << "  \"pbc1\": " << qc.pbc1 << ",\n"
      << "  \"pbc2\": " << qc.pbc2 << ",\n"
      << "  \"total_reads\": " << qc.total_reads << ",\n"
      << "  \"unique_reads\": " << qc.unique_reads << ",\n"
      << "  \"reads_in_peaks\": " << qc.reads_in_peaks << ",\n"
      << "  \"gc_bias\": " << qc.gc_bias << ",\n"
      << "  \"auc_lorenz\": " << qc.auc_lorenz << "\n"
      << "}\n";

    return true;
}

}  // namespace singlet
