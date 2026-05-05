#pragma once
// singlet-pileup: ambient_profile.h
// G-AMBIENT — Ambient RNA profile export (SoupX / CellBender compatible).
//
// Computes the average gene expression in empty droplets — the "soup" profile
// that SoupX and CellBender require for ambient contamination correction.
//
// Algorithm:
//   1. Identify "ambient" barcodes: UMI count in [lower_umi, cell_threshold)
//   2. Sum their gene counts column-wise from the CSC count matrix
//   3. Normalise to expression fractions (sum-to-1 across genes)
//   4. Export as TSV: gene_name\texpression_fraction\traw_count
//
// API:
//   AmbientProfile build_ambient_profile(gene_names, counts, barcode_totals,
//                                        lower_umi, cell_threshold)
//   bool write_ambient_profile(prof, filepath)

#include "sparse_accumulator.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Config + result types
// ---------------------------------------------------------------------------

struct AmbientProfileConfig {
    std::string filepath;        ///< Output TSV path
    uint64_t    lower_umi  = 10; ///< Min UMI count to be counted as ambient
    uint64_t    upper_umi  = 0;  ///< Max UMI count to be counted as ambient
                                 ///< (0 = use cell_threshold from build call)
};

struct AmbientProfile {
    std::vector<std::string> gene_names;           ///< Gene names (same order as CSC rows)
    std::vector<double>      expression_fraction;  ///< Per-gene fraction in ambient pool
    std::vector<uint64_t>    raw_counts;           ///< Per-gene raw UMI sum in ambient pool
    uint64_t n_ambient_barcodes = 0;               ///< Empty droplets used
    uint64_t total_ambient_umis  = 0;              ///< UMIs summed over ambient pool
};

// ---------------------------------------------------------------------------
// build_ambient_profile
// ---------------------------------------------------------------------------

/// Build an ambient RNA profile from barcode UMI counts and the CSC count matrix.
///
/// @param gene_names      Gene names (nrows of the CSC matrix)
/// @param counts          Exon CSC count matrix (nrows=genes, ncols=barcodes)
/// @param barcode_totals  Per-barcode total UMI count (length = ncols)
/// @param cell_indices    Barcodes called as cells (excluded from ambient pool)
/// @param lower_umi       Min UMI for ambient pool (inclusive, default 10)
/// @param cell_threshold  Max UMI for ambient pool (exclusive = knee threshold)
///
/// @return AmbientProfile with expression fractions and raw counts.
inline AmbientProfile build_ambient_profile(
    const std::vector<std::string>&               gene_names,
    const SparseAccumulator<uint16_t>::CSCMatrix& counts,
    const std::vector<uint64_t>&                  barcode_totals,
    const std::unordered_set<uint32_t>&           cell_indices,
    uint64_t                                       lower_umi      = 10,
    uint64_t                                       cell_threshold = 0)
{
    AmbientProfile prof;
    prof.gene_names = gene_names;
    const uint32_t n_genes = static_cast<uint32_t>(gene_names.size());
    prof.raw_counts.assign(n_genes, 0ULL);
    prof.expression_fraction.assign(n_genes, 0.0);

    const uint32_t n_barcodes = static_cast<uint32_t>(barcode_totals.size());
    if (n_barcodes == 0 || n_genes == 0) return prof;

    // Auto-determine upper bound: max over non-cell barcodes if not supplied
    uint64_t upper = cell_threshold;
    if (upper == 0) {
        for (uint32_t c = 0; c < n_barcodes; ++c) {
            if (cell_indices.count(c) == 0)
                upper = std::max(upper, barcode_totals[c]);
        }
        upper += 1; // exclusive
    }

    // Accumulate gene counts from ambient barcodes
    for (uint32_t c = 0; c < n_barcodes; ++c) {
        if (cell_indices.count(c) > 0) continue; // skip called cells
        const uint64_t t = barcode_totals[c];
        if (t < lower_umi || t >= upper) continue;

        ++prof.n_ambient_barcodes;
        prof.total_ambient_umis += t;

        for (int32_t k = counts.indptr[c]; k < counts.indptr[c + 1]; ++k) {
            const uint32_t gene = static_cast<uint32_t>(counts.indices[k]);
            if (gene < n_genes)
                prof.raw_counts[gene] += counts.data[k];
        }
    }

    // Normalise to fractions
    if (prof.total_ambient_umis > 0) {
        const double inv = 1.0 / static_cast<double>(prof.total_ambient_umis);
        for (uint32_t g = 0; g < n_genes; ++g)
            prof.expression_fraction[g] = prof.raw_counts[g] * inv;
    }

    return prof;
}

// ---------------------------------------------------------------------------
// write_ambient_profile
// ---------------------------------------------------------------------------

/// Write ambient RNA profile to a TSV file.
/// Header: gene_name\texpression_fraction\traw_count
///
/// @return true on success, false on IO error.
inline bool write_ambient_profile(const AmbientProfile& prof,
                                   const std::string&    filepath)
{
    std::ofstream out(filepath);
    if (!out) {
        std::cerr << "[ambient_profile] ERROR: cannot open " << filepath << "\n";
        return false;
    }

    out << "gene_name\texpression_fraction\traw_count\n";
    const uint32_t n = static_cast<uint32_t>(prof.gene_names.size());
    for (uint32_t g = 0; g < n; ++g) {
        out << prof.gene_names[g] << '\t'
            << prof.expression_fraction[g] << '\t'
            << prof.raw_counts[g] << '\n';
    }
    return out.good();
}

} // namespace singlet
