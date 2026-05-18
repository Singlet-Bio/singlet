// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: cuttag.h
// G-CUTTAG — CUT&TAG / CUT&RUN chromatin profiling QC.
//
// Computes specialized QC metrics for CUT&TAG and CUT&RUN experiments.
// Key difference from ATAC-seq: fragment size class boundaries differ because
// CUT&TAG (histone mods) retains nucleosomal laddering while CUT&RUN
// (H3K4me3, TFs) enriches short sub-nucleosomal fragments (~120 bp).
//
// No htslib dependency — operates on fragment size vectors or a simple
// Fragment struct so bulk/sc ATAC pipelines can call this header without
// pulling in BAM reading infrastructure.
//
// Integration: see INTEGRATION_NOTES.md § G-CUTTAG
// Reuses bulk-ATAC fragment lists (ATACFragment → extract sizes and
// spike-in chrom counts before calling compute_cuttag_qc).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// Mode enum
// ─────────────────────────────────────────────────────────────────────────────

enum class CutTagMode {
    CUT_AND_TAG,  // H3K27me3, H3K4me1 — expect nucleosomal fragments
    CUT_AND_RUN   // H3K4me3, TFs     — expect sub-nucleosomal fragments (~120 bp)
};

// ─────────────────────────────────────────────────────────────────────────────
// Fragment struct (no htslib dependency)
// ─────────────────────────────────────────────────────────────────────────────

struct Fragment {
    int32_t chrom_idx = 0;
    int32_t start = 0;         // 0-based inclusive
    int32_t end = 0;           // 0-based exclusive
    uint32_t barcode_idx = 0;  // unused in bulk mode
};

// ─────────────────────────────────────────────────────────────────────────────
// Size class boundaries (CUT&TAG / CUT&RUN conventions)
// ─────────────────────────────────────────────────────────────────────────────

// Sub-nucleosomal (CUT&RUN signature): < 120 bp
static constexpr int32_t kCutSubNuclMax = 120;
// Mono-nucleosomal window: 150–300 bp (CUT&TAG H3K27me3 signature)
static constexpr int32_t kCutMonoMin = 150;
static constexpr int32_t kCutMonoMax = 300;
// Large / background: > 300 bp
static constexpr int32_t kCutLargeMin = 300;
// Histogram: 10 bp bins [0, 1000 bp)
static constexpr int32_t kCutHistBinSize = 10;
static constexpr int32_t kCutHistBins = 100;  // 0..990 bp, 100 entries

// Mode-detection thresholds
static constexpr double kCutRunSubNuclFracThreshold = 0.40;  // > 40% → CUT&RUN
static constexpr double kCutTagSubNuclFracThreshold = 0.20;  // < 20% → CUT&TAG

// ─────────────────────────────────────────────────────────────────────────────
// QC struct
// ─────────────────────────────────────────────────────────────────────────────

struct CutTagQC {
    // Fragment size class fractions (of total_fragments)
    double sub_nucleosomal_fraction = 0.0;   // < 120 bp  (CUT&RUN signature)
    double mono_nucleosomal_fraction = 0.0;  // 150–300 bp (CUT&TAG signature)
    double large_fragment_fraction = 0.0;    // > 300 bp  (background)

    // Enrichment
    double signal_to_noise = 0.0;  // (sub-nucl + mono) / large
    double frip = 0.0;             // fraction in peaks / signal bins

    // Spike-in normalization (optional — yeast / E. coli genome reads)
    uint64_t spike_in_reads = 0;
    double spike_in_fraction = 0.0;
    double scale_factor = 1.0;  // 1.0 / spike_in_fraction

    // Fragment-size histogram: 10 bp bins [0, 1000 bp)
    std::vector<uint32_t> size_histogram;  // length = kCutHistBins

    uint64_t total_fragments = 0;
    uint64_t unique_fragments = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
// Core functions
// ─────────────────────────────────────────────────────────────────────────────

/// Detect CUT&TAG vs CUT&RUN from a list of fragment sizes (bp).
/// Returns CUT_AND_RUN when sub-nucleosomal fraction > 40%.
/// Returns CUT_AND_TAG when sub-nucleosomal fraction < 20%.
/// Returns CUT_AND_TAG for ambiguous intermediate fractions (conservative).
inline CutTagMode detect_mode(const std::vector<uint32_t>& fragment_sizes) {
    if (fragment_sizes.empty()) return CutTagMode::CUT_AND_TAG;

    uint64_t sub_nucl = 0;
    for (uint32_t sz : fragment_sizes) {
        if (static_cast<int32_t>(sz) < kCutSubNuclMax) ++sub_nucl;
    }
    const double frac = static_cast<double>(sub_nucl) /
                        static_cast<double>(fragment_sizes.size());

    return (frac > kCutRunSubNuclFracThreshold)
               ? CutTagMode::CUT_AND_RUN
               : CutTagMode::CUT_AND_TAG;
}

/// Compute spike-in scale factor: 1.0 / spike_in_fraction.
/// If spike_in_reads == 0 or total_reads == 0, returns 1.0.
inline double compute_spike_in_scale_factor(uint64_t spike_in_reads,
                                            uint64_t total_reads) {
    if (spike_in_reads == 0 || total_reads == 0) return 1.0;
    const double frac = static_cast<double>(spike_in_reads) /
                        static_cast<double>(total_reads);
    return (frac > 0.0) ? (1.0 / frac) : 1.0;
}

/// Compute CUT&TAG / CUT&RUN QC from a fragment list.
///
/// @param fragments        Fragment vector (deduplicated by caller)
/// @param total_reads      Total input read pairs before any filtering
/// @param spike_in_chrom_idxs  Chromosome indices corresponding to the
///                         spike-in genome (E. coli / yeast / dm6).
///                         When empty, spike-in metrics are left at 0.
inline CutTagQC compute_cuttag_qc(
    const std::vector<Fragment>& fragments,
    uint64_t total_reads,
    const std::vector<int32_t>& spike_in_chrom_idxs = {}) {
    CutTagQC qc;
    qc.total_fragments = static_cast<uint64_t>(fragments.size());
    qc.unique_fragments = qc.total_fragments;  // caller deduplicates
    qc.size_histogram.assign(static_cast<size_t>(kCutHistBins), 0u);

    if (fragments.empty()) {
        qc.scale_factor = 1.0;
        return qc;
    }

    // Build spike-in chrom set for O(log n) lookup
    std::vector<int32_t> spike_sorted = spike_in_chrom_idxs;
    std::sort(spike_sorted.begin(), spike_sorted.end());

    uint64_t sub_nucl = 0;
    uint64_t mono = 0;
    uint64_t large = 0;
    uint64_t spike_in = 0;

    for (const auto& f : fragments) {
        // Spike-in reads
        if (!spike_sorted.empty() &&
            std::binary_search(spike_sorted.begin(), spike_sorted.end(), f.chrom_idx)) {
            ++spike_in;
        }

        const int32_t len = f.end - f.start;
        if (len <= 0) continue;

        // Histogram (10 bp bins)
        const int32_t bin = len / kCutHistBinSize;
        if (bin >= 0 && bin < kCutHistBins) {
            ++qc.size_histogram[static_cast<size_t>(bin)];
        }

        // Size class
        if (len < kCutSubNuclMax)
            ++sub_nucl;
        else if (len >= kCutMonoMin && len < kCutMonoMax)
            ++mono;
        else if (len >= kCutLargeMin)
            ++large;
    }

    const double n = static_cast<double>(qc.total_fragments);
    qc.sub_nucleosomal_fraction = static_cast<double>(sub_nucl) / n;
    qc.mono_nucleosomal_fraction = static_cast<double>(mono) / n;
    qc.large_fragment_fraction = static_cast<double>(large) / n;

    // Signal-to-noise: (sub-nucl + mono) / large; 0 if no large fragments
    const double signal = static_cast<double>(sub_nucl + mono);
    qc.signal_to_noise = (large > 0)
                             ? signal / static_cast<double>(large)
                             : (signal > 0 ? std::numeric_limits<double>::infinity() : 0.0);

    // Spike-in metrics
    qc.spike_in_reads = spike_in;
    if (total_reads > 0) {
        qc.spike_in_fraction = static_cast<double>(spike_in) /
                               static_cast<double>(total_reads);
    }
    qc.scale_factor = compute_spike_in_scale_factor(spike_in, total_reads);

    return qc;
}

/// Write CUT&TAG QC to a JSON file at `path`.
/// Returns true on success.
inline bool write_cuttag_qc(const std::string& path,
                            const CutTagQC& qc,
                            CutTagMode mode) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << "{\n"
        << "  \"mode\": \""
        << (mode == CutTagMode::CUT_AND_RUN ? "CUT_AND_RUN" : "CUT_AND_TAG")
        << "\",\n"
        << "  \"total_fragments\": " << qc.total_fragments << ",\n"
        << "  \"unique_fragments\": " << qc.unique_fragments << ",\n"
        << "  \"sub_nucleosomal_fraction\": " << qc.sub_nucleosomal_fraction << ",\n"
        << "  \"mono_nucleosomal_fraction\": " << qc.mono_nucleosomal_fraction << ",\n"
        << "  \"large_fragment_fraction\": " << qc.large_fragment_fraction << ",\n"
        << "  \"signal_to_noise\": " << qc.signal_to_noise << ",\n"
        << "  \"frip\": " << qc.frip << ",\n"
        << "  \"spike_in_reads\": " << qc.spike_in_reads << ",\n"
        << "  \"spike_in_fraction\": " << qc.spike_in_fraction << ",\n"
        << "  \"scale_factor\": " << qc.scale_factor << ",\n"
        << "  \"size_histogram\": [";
    for (size_t i = 0; i < qc.size_histogram.size(); ++i) {
        ofs << qc.size_histogram[i];
        if (i + 1 < qc.size_histogram.size()) ofs << ", ";
    }
    ofs << "]\n}\n";
    return ofs.good();
}

}  // namespace singlet
