// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: ase.h
// N15: Allele-specific expression (ASE) from existing SNP pileup AD/DP matrices.
//
// Computes per-cell, per-SNP allele counts from the snp_ad (alt depth) and
// snp_dp (total depth) CSC matrices already produced by PileupEngine.
// No extra BAM pass required — this is a pure post-processing step.
//
// Output: ase_counts.tsv with columns:
//   barcode  snp_id  ref_count  alt_count  allelic_ratio
//
// Only heterozygous sites are reported: dp >= min_depth AND ad >= 1 AND ref >= 1.
// allelic_ratio = alt / (ref + alt); expected ~0.5 for balanced expression.

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {
namespace ase {

/// One heterozygous SNP observation in one cell.
struct ASEEntry {
    uint32_t cell_idx;      ///< Column index into the CSC matrix (barcode index)
    uint32_t snp_idx;       ///< Row index into the CSC matrix (SNP index)
    uint32_t ref_count;     ///< Reference allele depth  (dp - ad)
    uint32_t alt_count;     ///< Alternate allele depth  (ad)
    float    allelic_ratio; ///< alt / (ref + alt); range [0, 1]
};

/// Compute allele-specific expression entries from SNP pileup CSC matrices.
///
/// Iterates over every (cell, SNP) pair with sufficient coverage and emits an
/// ASEEntry for each heterozygous site:
///   total_depth >= min_depth  AND  alt_depth >= 1  AND  ref_depth >= 1
///
/// Within each CSC column the row indices are sorted (guaranteed by to_csc()),
/// so this uses a single-pass merge-join — O(nnz) with no allocations.
///
/// @param snp_ad_csc  Alt-allele depth (uint8_t, capped at 255) — from PileupEngine::snp_ad()
/// @param snp_dp_csc  Total depth     (uint8_t, capped at 255) — from PileupEngine::snp_dp()
/// @param min_depth   Minimum total depth to consider a site (default: 10)
/// @return            Flat vector of ASEEntry, ordered by (cell_idx, snp_idx)
inline std::vector<ASEEntry> compute_ase(
    const SparseAccumulator<uint8_t>::CSCMatrix& snp_ad_csc,
    const SparseAccumulator<uint8_t>::CSCMatrix& snp_dp_csc,
    int min_depth = 10)
{
    std::vector<ASEEntry> entries;
    if (snp_dp_csc.ncols == 0 || snp_dp_csc.data.empty()) return entries;

    const uint32_t ncols = snp_dp_csc.ncols;
    const auto md = static_cast<uint8_t>(min_depth <= 255 ? min_depth : 255);

    for (uint32_t j = 0; j < ncols; ++j) {
        const int32_t dp_start = snp_dp_csc.indptr[j];
        const int32_t dp_end   = snp_dp_csc.indptr[j + 1];
        const int32_t ad_start = snp_ad_csc.indptr[j];
        const int32_t ad_end   = snp_ad_csc.indptr[j + 1];

        // Single-pass merge-join over two sorted row-index lists.
        // DP is the primary (drives the outer loop); AD is secondary.
        int32_t qi = ad_start; // cursor into AD column
        for (int32_t pi = dp_start; pi < dp_end; ++pi) {
            const int32_t dp_row = snp_dp_csc.indices[pi];
            const uint8_t dp_val = snp_dp_csc.data[pi];

            // Skip depth-insufficient sites early
            if (dp_val < md) continue;

            // Advance AD cursor to dp_row
            while (qi < ad_end && snp_ad_csc.indices[qi] < dp_row)
                ++qi;

            const uint8_t ad_val =
                (qi < ad_end && snp_ad_csc.indices[qi] == dp_row)
                ? snp_ad_csc.data[qi]
                : uint8_t(0);

            // Heterozygous check: both alleles observed
            if (ad_val >= 1 && dp_val > ad_val) {
                ASEEntry e;
                e.cell_idx      = j;
                e.snp_idx       = static_cast<uint32_t>(dp_row);
                e.ref_count     = static_cast<uint32_t>(dp_val - ad_val);
                e.alt_count     = static_cast<uint32_t>(ad_val);
                e.allelic_ratio = static_cast<float>(ad_val) /
                                  static_cast<float>(dp_val);
                entries.push_back(e);
            }
        }
    }

    return entries;
}

/// Write ASE entries to a tab-separated file.
///
/// Output columns: barcode, snp_id, ref_count, alt_count, allelic_ratio
/// Only called when entries is non-empty; a zero-entry file is still written
/// with the header so downstream tools can safely read it.
///
/// @param path      Destination file path (e.g. "<out_prefix>/ase_counts.tsv")
/// @param entries   Result of compute_ase()
/// @param barcodes  Barcode strings, indexed by ASEEntry::cell_idx
/// @param snp_names SNP identifier strings, indexed by ASEEntry::snp_idx
inline void write_ase_tsv(
    const std::string&             path,
    const std::vector<ASEEntry>&   entries,
    const std::vector<std::string>& barcodes,
    const std::vector<std::string>& snp_names)
{
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[ase] ERROR: Cannot open for writing: " << path << "\n";
        return;
    }

    f << "barcode\tsnp_id\tref_count\talt_count\tallelic_ratio\n";

    char ratio_buf[16];
    for (const auto& e : entries) {
        std::snprintf(ratio_buf, sizeof(ratio_buf), "%.4f", e.allelic_ratio);
        f << barcodes[e.cell_idx] << '\t'
          << snp_names[e.snp_idx] << '\t'
          << e.ref_count          << '\t'
          << e.alt_count          << '\t'
          << ratio_buf            << '\n';
    }

    std::cerr << "[ase] " << entries.size()
              << " het-SNP×cell entries → " << path << "\n";
}

} // namespace ase
} // namespace singlet
