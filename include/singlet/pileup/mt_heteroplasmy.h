// SPDX-License-Identifier: MIT
// singlet-pileup: mt_heteroplasmy.h
// Mitochondrial heteroplasmy computation from inline chrM base pileup.
// Replaces mgatk (Python/Snakemake) with a single-pass C++ implementation.
//
// Data model: SparseAccumulator<uint16_t> with feature = pos*4 + base_idx
//   base_idx: A=0, C=1, G=2, T=3
//   n_features = 16569 * 4 = 66276 (rCRS reference length)
//
// After streaming, compute per-cell heteroplasmy at variable positions.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nucleotide_encoding.h"

namespace singlet {
namespace mt {

// rCRS reference length
static constexpr uint32_t MT_LEN = 16569;
static constexpr uint32_t MT_N_FEATURES = MT_LEN * 4;  // pos*4 + base

// Base encoding tables — canonical definitions live in nucleotide_encoding.h.
//   BASE_TO_IDX: htslib bam_seqi 4-bit code (1=A,2=C,4=G,8=T,15=N) -> 0-3 index
//   IDX_TO_BASE: 0-3 index -> ASCII char
using ::singlet::pileup::nt::BASE_TO_IDX;
using ::singlet::pileup::nt::IDX_TO_BASE;

// Inline feature index computation
inline uint32_t mt_feature(uint32_t pos, int base_idx) {
    return pos * 4 + static_cast<uint32_t>(base_idx);
}

// A variable mt position detected across cells
struct MtVariant {
    uint32_t pos;           // 0-based mt position
    char ref_allele;        // reference allele (rCRS)
    char alt_allele;        // most common non-ref allele
    uint32_t n_cells_covered;   // cells with ≥min_coverage
    uint32_t n_cells_het;       // cells showing heteroplasmy
    float mean_vaf;         // mean VAF across heteroplasmic cells
};

// Per-cell heteroplasmy result at variable positions
struct MtHetResult {
    std::vector<MtVariant> variants;        // variable positions found
    // Sparse matrix: cells × variants (float VAF values)
    // Stored as CSC for direct .1pz writing
    std::vector<int32_t> indptr;   // [n_cells + 1]
    std::vector<int32_t> indices;  // row indices (variant indices)
    std::vector<float> data;       // VAF values
    uint32_t n_variants = 0;       // nrows
    uint32_t n_cells = 0;          // ncols
};

// Load rCRS reference from a file (16,569 bases, upper-case ACGT)
// Returns empty string on failure.
inline std::string load_rcrs(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::string ref;
    ref.reserve(MT_LEN);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '>') continue;  // skip FASTA header
        for (char c : line) {
            if (c >= 'A' && c <= 'Z') ref.push_back(c);
            else if (c >= 'a' && c <= 'z') ref.push_back(c - 32);
        }
    }
    return ref;
}

// Generate rCRS reference from the BAM header's SQ md5/length
// (fallback: use a hardcoded consensus of the most common allele per position
//  from the mt_acc data itself)
inline std::string infer_ref_from_data(
    const int32_t* indptr, const int32_t* indices,
    const uint16_t* data, uint32_t n_cells)
{
    std::string ref(MT_LEN, 'N');
    // For each position, find the most common base across all cells
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        uint64_t counts[4] = {0, 0, 0, 0};
        (void)counts;
        for (int b = 0; b < 4; ++b) {
            uint32_t feat = pos * 4 + b;
            (void)feat;
            // Sum across all cells for this feature
            // We need to iterate the CSC column-wise
            // Since mt_csc is features × cells (row=feature, col=cell),
            // we need to scan all columns 
        }
        // Actually, with CSC (feature × cell), columns are cells.
        // For a given feature row, we need to check every column's entries.
        // This is expensive. Better approach: convert to CSR or do a single pass.
    }
    // More efficient: single pass through all entries
    uint64_t pos_counts[MT_LEN][4] = {};
    int64_t total_nnz = indptr[n_cells]; // total entries
    for (int64_t i = 0; i < total_nnz; ++i) {
        uint32_t feat = static_cast<uint32_t>(indices[i]);
        if (feat >= MT_N_FEATURES) continue;
        uint32_t pos = feat / 4;
        uint32_t base = feat % 4;
        pos_counts[pos][base] += data[i];
    }
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        int best = 0;
        for (int b = 1; b < 4; ++b) {
            if (pos_counts[pos][b] > pos_counts[pos][best]) best = b;
        }
        if (pos_counts[pos][best] > 0) {
            ref[pos] = IDX_TO_BASE[best];
        }
    }
    return ref;
}

// Compute mt heteroplasmy from the mt allele count CSC matrix.
// mt_csc is features × cells where feature = pos*4 + base_idx.
//
// Parameters:
//   indptr, indices, data: CSC arrays (n_features × n_cells)
//   n_cells: number of barcodes
//   ref_sequence: rCRS reference (16569 chars) or empty for auto-infer
//   min_af: minimum allele fraction to report as heteroplasmic
//   max_af: maximum (exclude homoplasmic alt)
//   min_coverage: minimum total reads at position per cell
//   min_cells_het: minimum cells showing heteroplasmy to report variant
inline MtHetResult compute_heteroplasmy(
    const int32_t* indptr, const int32_t* indices,
    const uint16_t* data, uint32_t n_features, uint32_t n_cells,
    const std::string& ref_sequence = "",
    float min_af = 0.01f,
    float max_af = 0.95f,
    int min_coverage = 10,
    int min_cells_het = 2)
{
    MtHetResult result;
    result.n_cells = n_cells;

    // Get reference sequence
    std::string ref = ref_sequence;
    if (ref.empty() || ref.size() != MT_LEN) {
        ref = infer_ref_from_data(indptr, indices, data, n_cells);
    }

    // Build per-cell, per-position allele counts
    // We extract from CSC (features × cells):
    //   For each cell (column), iterate its non-zero entries,
    //   group by position, compute coverage and VAF.

    // First pass: find variable positions across all cells
    // Aggregate total per-position base counts
    uint64_t global_counts[MT_LEN][4] = {};
    int64_t total_nnz = indptr[n_cells];
    for (int64_t i = 0; i < total_nnz; ++i) {
        uint32_t feat = static_cast<uint32_t>(indices[i]);
        if (feat >= MT_N_FEATURES) continue;
        uint32_t pos = feat / 4;
        uint32_t base = feat % 4;
        global_counts[pos][base] += data[i];
    }

    // Identify candidate variable positions: any non-ref allele with ≥1% global frequency
    std::vector<uint32_t> candidate_positions;
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        uint64_t total = 0;
        for (int b = 0; b < 4; ++b) total += global_counts[pos][b];
        if (total < static_cast<uint64_t>(min_coverage)) continue;

        char ref_base = ref[pos];
        int ref_idx = -1;
        for (int b = 0; b < 4; ++b) {
            if (IDX_TO_BASE[b] == ref_base) { ref_idx = b; break; }
        }
        if (ref_idx < 0) continue;

        // Check if any non-ref allele has sufficient frequency globally
        for (int b = 0; b < 4; ++b) {
            if (b == ref_idx) continue;
            if (global_counts[pos][b] > 0) {
                double af = static_cast<double>(global_counts[pos][b]) / total;
                if (af >= 0.001) {  // very permissive global filter
                    candidate_positions.push_back(pos);
                    break;
                }
            }
        }
    }

    if (candidate_positions.empty()) {
        result.n_variants = 0;
        result.indptr.assign(n_cells + 1, 0);
        return result;
    }

    // Second pass: per-cell VAF computation at candidate positions
    // Build a fast lookup: position → candidate index
    std::vector<int32_t> pos_to_cand(MT_LEN, -1);
    for (size_t i = 0; i < candidate_positions.size(); ++i) {
        pos_to_cand[candidate_positions[i]] = static_cast<int32_t>(i);
    }

    uint32_t n_cand = static_cast<uint32_t>(candidate_positions.size());

    // Per-cell, per-candidate: store allele counts
    // Memory: n_cells × n_cand × 4 × 2 bytes ≈ manageable for typical <500 candidates
    // For large sets, use sparse iteration
    struct CellPosCounts {
        uint16_t counts[4] = {0, 0, 0, 0};
    };

    // Build per-cell heteroplasmy data (COO → then convert to CSC)
    std::vector<int32_t> coo_rows;  // variant index
    std::vector<int32_t> coo_cols;  // cell index
    std::vector<float> coo_vals;    // VAF

    // Track per-variant stats
    std::vector<uint32_t> var_n_covered(n_cand, 0);
    std::vector<uint32_t> var_n_het(n_cand, 0);
    std::vector<double> var_sum_vaf(n_cand, 0.0);

    // Process cell by cell (CSC column iteration)
    std::vector<CellPosCounts> cell_counts(n_cand);
    for (uint32_t cell = 0; cell < n_cells; ++cell) {
        // Reset counts for this cell
        for (uint32_t c = 0; c < n_cand; ++c) {
            cell_counts[c] = {};
        }

        // Iterate this cell's non-zero features
        for (int32_t j = indptr[cell]; j < indptr[cell + 1]; ++j) {
            uint32_t feat = static_cast<uint32_t>(indices[j]);
            if (feat >= MT_N_FEATURES) continue;
            uint32_t pos = feat / 4;
            uint32_t base = feat % 4;
            int32_t cand = pos_to_cand[pos];
            if (cand >= 0) {
                cell_counts[cand].counts[base] = data[j];
            }
        }

        // Compute VAF at each candidate position for this cell
        for (uint32_t ci = 0; ci < n_cand; ++ci) {
            uint32_t pos = candidate_positions[ci];
            auto& cc = cell_counts[ci];
            uint32_t total = 0;
            for (int b = 0; b < 4; ++b) total += cc.counts[b];
            if (total < static_cast<uint32_t>(min_coverage)) continue;

            var_n_covered[ci]++;

            char ref_base = ref[pos];
            int ref_idx = -1;
            for (int b = 0; b < 4; ++b) {
                if (IDX_TO_BASE[b] == ref_base) { ref_idx = b; break; }
            }
            if (ref_idx < 0) continue;

            // Find the most common non-ref allele
            int best_alt = -1;
            uint32_t best_count = 0;
            for (int b = 0; b < 4; ++b) {
                if (b == ref_idx && cc.counts[b] > best_count) continue;
                if (b != ref_idx && cc.counts[b] > best_count) {
                    best_alt = b;
                    best_count = cc.counts[b];
                }
            }
            if (best_alt < 0 || best_count == 0) continue;

            float vaf = static_cast<float>(best_count) / static_cast<float>(total);
            if (vaf >= min_af && vaf <= max_af) {
                coo_rows.push_back(static_cast<int32_t>(ci));
                coo_cols.push_back(static_cast<int32_t>(cell));
                coo_vals.push_back(vaf);
                var_n_het[ci]++;
                var_sum_vaf[ci] += vaf;
            }
        }
    }

    // Filter to positions meeting min_cells_het threshold
    std::vector<uint32_t> keep_indices;
    for (uint32_t ci = 0; ci < n_cand; ++ci) {
        if (var_n_het[ci] >= static_cast<uint32_t>(min_cells_het)) {
            keep_indices.push_back(ci);
        }
    }

    if (keep_indices.empty()) {
        result.n_variants = 0;
        result.indptr.assign(n_cells + 1, 0);
        return result;
    }

    // Build old→new variant index mapping
    std::vector<int32_t> old_to_new(n_cand, -1);
    for (size_t i = 0; i < keep_indices.size(); ++i) {
        old_to_new[keep_indices[i]] = static_cast<int32_t>(i);
    }

    // Build variant metadata
    for (uint32_t ki : keep_indices) {
        uint32_t pos = candidate_positions[ki];
        char ref_base = ref[pos];

        // Find most common non-ref allele globally
        int ref_idx = -1;
        for (int b = 0; b < 4; ++b) {
            if (IDX_TO_BASE[b] == ref_base) { ref_idx = b; break; }
        }
        int best_alt = -1;
        uint64_t best_count = 0;
        for (int b = 0; b < 4; ++b) {
            if (b != ref_idx && global_counts[pos][b] > best_count) {
                best_alt = b;
                best_count = global_counts[pos][b];
            }
        }

        MtVariant v;
        v.pos = pos;
        v.ref_allele = ref_base;
        v.alt_allele = (best_alt >= 0) ? IDX_TO_BASE[best_alt] : 'N';
        v.n_cells_covered = var_n_covered[ki];
        v.n_cells_het = var_n_het[ki];
        v.mean_vaf = (var_n_het[ki] > 0)
                     ? static_cast<float>(var_sum_vaf[ki] / var_n_het[ki])
                     : 0.0f;
        result.variants.push_back(v);
    }
    result.n_variants = static_cast<uint32_t>(keep_indices.size());

    // Build CSC from filtered COO (variants × cells)
    // Remap rows and filter
    std::vector<int32_t> filt_rows, filt_cols;
    std::vector<float> filt_vals;
    for (size_t i = 0; i < coo_rows.size(); ++i) {
        int32_t new_row = old_to_new[coo_rows[i]];
        if (new_row >= 0) {
            filt_rows.push_back(new_row);
            filt_cols.push_back(coo_cols[i]);
            filt_vals.push_back(coo_vals[i]);
        }
    }

    // Convert COO → CSC (variants × cells)
    result.indptr.assign(n_cells + 1, 0);
    for (int32_t c : filt_cols) {
        result.indptr[c + 1]++;
    }
    for (uint32_t c = 0; c < n_cells; ++c) {
        result.indptr[c + 1] += result.indptr[c];
    }

    size_t nnz = filt_rows.size();
    result.indices.resize(nnz);
    result.data.resize(nnz);
    std::vector<int32_t> cursor(result.indptr.begin(), result.indptr.end() - 1);
    for (size_t i = 0; i < nnz; ++i) {
        int32_t col = filt_cols[i];
        int32_t pos = cursor[col]++;
        result.indices[pos] = filt_rows[i];
        result.data[pos] = filt_vals[i];
    }

    // Sort rows within each column (required for CSC)
    for (uint32_t c = 0; c < n_cells; ++c) {
        int32_t start = result.indptr[c];
        int32_t end = result.indptr[c + 1];
        if (end - start <= 1) continue;
        // Simple insertion sort (typically <50 entries per cell)
        for (int32_t i = start + 1; i < end; ++i) {
            int32_t ri = result.indices[i];
            float vi = result.data[i];
            int32_t j = i - 1;
            while (j >= start && result.indices[j] > ri) {
                result.indices[j + 1] = result.indices[j];
                result.data[j + 1] = result.data[j];
                j--;
            }
            result.indices[j + 1] = ri;
            result.data[j + 1] = vi;
        }
    }

    return result;
}

// Write mt heteroplasmy summary TSV
inline void write_mt_variants(const std::string& path, const MtHetResult& het) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[mt] Cannot open: " << path << "\n";
        return;
    }
    f << "pos\tref\talt\tn_cells_covered\tn_cells_het\tmean_vaf\n";
    for (const auto& v : het.variants) {
        f << (v.pos + 1) << "\t"  // 1-based for output
          << v.ref_allele << "\t"
          << v.alt_allele << "\t"
          << v.n_cells_covered << "\t"
          << v.n_cells_het << "\t"
          << v.mean_vaf << "\n";
    }
    std::cerr << "[mt] Wrote " << het.variants.size() << " variable mt positions to "
              << path << "\n";
}

// Write per-cell allele count matrix (all 4 bases) as feature names
inline std::vector<std::string> mt_feature_names() {
    std::vector<std::string> names;
    names.reserve(MT_N_FEATURES);
    for (uint32_t pos = 0; pos < MT_LEN; ++pos) {
        for (int b = 0; b < 4; ++b) {
            names.push_back("chrM:" + std::to_string(pos + 1) + ":" + IDX_TO_BASE[b]);
        }
    }
    return names;
}

// Write variant feature names for the heteroplasmy matrix
inline std::vector<std::string> het_feature_names(const MtHetResult& het) {
    std::vector<std::string> names;
    names.reserve(het.variants.size());
    for (const auto& v : het.variants) {
        names.push_back("chrM:" + std::to_string(v.pos + 1) + ":" +
                         v.ref_allele + ">" + v.alt_allele);
    }
    return names;
}

/// A single mt indel event captured during CIGAR parsing.
struct MtIndelEvent {
    uint32_t bc_idx;  ///< cell barcode column index
    uint32_t pos;     ///< 0-based chrM anchor position
    uint16_t len;     ///< CIGAR op length in bases
    bool     is_ins;  ///< true=BAM_CINS, false=BAM_CDEL
};

} // namespace mt
} // namespace singlet
