#pragma once
// singlet-pileup: bulk_atac.h
// G-BULK-ATAC — Bulk ATAC-seq auto-detection and QC.
//
// Adapts existing scATAC fragment infrastructure (A1 / A3) for bulk mode:
//   - No cell barcodes (cblen == 0)
//   - Paired-end DNA reads
//   - Fragment size class fractions (NFR / mono / di-nucleosome)
//   - Lander-Waterman library complexity estimate
//   - JSON QC output
//
// All functions are header-only; no BAM/htslib dependency required.
// Fragment list comes from ATACFragmentExtractor (atac_fragment.h) or
// any source that produces std::vector<ATACFragment>.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "atac_fragment.h"

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────
struct BulkAtacConfig {
    uint32_t min_mapq = 20;
    uint32_t max_fragment_size = 1000;  // typical ATAC upper bound
    bool remove_duplicates = true;      // Picard-style position dedup
    bool remove_chrm = false;           // exclude mitochondrial fragments
};

// ─────────────────────────────────────────────────────────────────────────────
// Detection
// ─────────────────────────────────────────────────────────────────────────────

/// Detect if input is bulk ATAC-seq.
/// Returns true when:
///   (1) cblen == 0   — no cell barcode in .1fq metadata
///   (2) is_paired_end — paired-end library
///   (3) assay_type is "ATAC", "bulk-atac", "bulk_atac", or "" (unspecified)
///       with paired-end + no barcode being the deciding gate
inline bool is_bulk_atac(uint32_t cblen,
                         bool is_paired_end,
                         const std::string& assay_type = "") {
    if (!is_paired_end) return false;
    if (cblen != 0) return false;

    // If an explicit assay_type is declared, it must be ATAC-related.
    // Empty string means auto-detect from the two structural signals above.
    if (!assay_type.empty()) {
        const std::string& t = assay_type;
        if (t == "ATAC" || t == "atac" ||
            t == "bulk-atac" || t == "bulk_atac" ||
            t == "BULK-ATAC" || t == "BULK_ATAC") {
            return true;
        }
        // Explicitly declared as something else — not bulk ATAC
        return false;
    }
    return true;  // cblen==0, paired-end, no contradicting assay_type
}

// ─────────────────────────────────────────────────────────────────────────────
// QC struct
// ─────────────────────────────────────────────────────────────────────────────
struct BulkAtacQC {
    uint64_t total_fragments = 0;
    uint64_t unique_fragments = 0;  // after dedup
    uint64_t mitochondrial_fragments = 0;
    double tss_enrichment = 0.0;  // TSS fragment fraction
    double frip = 0.0;            // fraction in peaks / signal bins
    double median_fragment_size = 0.0;
    double library_complexity = 0.0;  // Lander-Waterman estimate

    // ENCODE nucleosomal size class fractions (applied to unique fragments)
    double nfr_fraction = 0.0;   // < 147 bp  — nucleosome-free region
    double mono_fraction = 0.0;  // 147–294 bp — mono-nucleosome
    double di_fraction = 0.0;    // 294–441 bp — di-nucleosome
};

// ─────────────────────────────────────────────────────────────────────────────
// Size class boundaries (ENCODE ATAC-seq conventions)
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int32_t kNFRMax = 147;   // < 147 bp
static constexpr int32_t kMonoMax = 294;  // 147–294 bp
static constexpr int32_t kDiMax = 441;    // 294–441 bp

// ─────────────────────────────────────────────────────────────────────────────
// Core computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute bulk ATAC QC metrics from a fragment list.
///
/// @param fragments    ATACFragment vector (may include mito / large fragments)
/// @param total_reads  Total input read pairs before any filtering
/// @param mito_chrom_idxs  Set of chromosome indices considered mitochondrial
/// @param tss_hit_set      (optional) set of fragment indices that overlap a TSS ±1 kbp
///                         If empty, tss_enrichment is left at 0.
///
/// Lander-Waterman model:
///   C = 1 − X/n   where X = unique_fragments, n = total_fragments
///   library_complexity = −n * ln(1 − X/n)  ≈ X when saturation is low
///   (Simplified; full PBC uses pairs-of-reads, but fragment-level is common.)
inline BulkAtacQC compute_bulk_atac_qc(
    const std::vector<ATACFragment>& fragments,
    uint64_t total_reads,
    const std::vector<int32_t>& mito_chrom_idxs = {},
    uint64_t tss_overlapping_fragments = 0) {
    BulkAtacQC qc;
    qc.total_fragments = static_cast<uint64_t>(fragments.size());

    // Sizes for median + size-class fractions
    std::vector<int32_t> sizes;
    sizes.reserve(fragments.size());

    // Build mito set for O(log n) lookup
    std::vector<int32_t> mito_sorted = mito_chrom_idxs;
    std::sort(mito_sorted.begin(), mito_sorted.end());

    uint64_t nfr = 0;
    uint64_t mono = 0;
    uint64_t di = 0;

    for (const auto& f : fragments) {
        const int32_t len = f.end - f.start;
        sizes.push_back(len);

        // Mito count
        if (std::binary_search(mito_sorted.begin(), mito_sorted.end(), f.chrom_idx)) {
            ++qc.mitochondrial_fragments;
        }

        // Size class — ENCODE boundaries
        if (len < kNFRMax)
            ++nfr;
        else if (len < kMonoMax)
            ++mono;
        else if (len < kDiMax)
            ++di;
    }

    // Unique fragments = total after dedup (caller deduplicates via
    // ATACFragmentExtractor; these fragments are already unique)
    qc.unique_fragments = qc.total_fragments;

    // Fragment size fractions
    if (qc.total_fragments > 0) {
        qc.nfr_fraction = static_cast<double>(nfr) / qc.total_fragments;
        qc.mono_fraction = static_cast<double>(mono) / qc.total_fragments;
        qc.di_fraction = static_cast<double>(di) / qc.total_fragments;
    }

    // Median fragment size
    if (!sizes.empty()) {
        std::sort(sizes.begin(), sizes.end());
        const size_t n = sizes.size();
        if (n % 2 == 0) {
            qc.median_fragment_size = 0.5 * (sizes[n / 2 - 1] + sizes[n / 2]);
        } else {
            qc.median_fragment_size = sizes[n / 2];
        }
    }

    // TSS enrichment: fraction of fragments overlapping a TSS ±1 kbp
    if (qc.total_fragments > 0 && tss_overlapping_fragments > 0) {
        qc.tss_enrichment = static_cast<double>(tss_overlapping_fragments) / qc.total_fragments;
    }

    // Lander-Waterman library complexity
    // total_reads = n (total read-pairs sequenced)
    // unique_fragments = X (distinct molecules observed)
    // C_lw = -n * ln(1 - X/n)
    if (total_reads > 0 && qc.unique_fragments > 0) {
        const double x_over_n = static_cast<double>(qc.unique_fragments) / static_cast<double>(total_reads);
        if (x_over_n < 1.0) {
            qc.library_complexity = -static_cast<double>(total_reads) * std::log(1.0 - x_over_n);
        } else {
            // Saturated: estimate equals unique fragments observed
            qc.library_complexity = static_cast<double>(qc.unique_fragments);
        }
    }

    return qc;
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON output
// ─────────────────────────────────────────────────────────────────────────────

/// Write BulkAtacQC to a JSON file at `path`.
/// Returns true on success.
inline bool write_bulk_atac_qc(const std::string& path, const BulkAtacQC& qc) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[bulk_atac] Cannot open for writing: " << path << "\n";
        return false;
    }

    f << "{\n"
      << "  \"total_fragments\": " << qc.total_fragments << ",\n"
      << "  \"unique_fragments\": " << qc.unique_fragments << ",\n"
      << "  \"mitochondrial_fragments\": " << qc.mitochondrial_fragments << ",\n"
      << "  \"tss_enrichment\": " << qc.tss_enrichment << ",\n"
      << "  \"frip\": " << qc.frip << ",\n"
      << "  \"median_fragment_size\": " << qc.median_fragment_size << ",\n"
      << "  \"library_complexity\": " << qc.library_complexity << ",\n"
      << "  \"nfr_fraction\": " << qc.nfr_fraction << ",\n"
      << "  \"mono_fraction\": " << qc.mono_fraction << ",\n"
      << "  \"di_fraction\": " << qc.di_fraction << "\n"
      << "}\n";

    return true;
}

}  // namespace singlet
