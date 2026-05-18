// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: velocity.h
// G-VELOCITY: per-gene spliced/unspliced/ambiguous matrices for RNA velocity.
// Compatible with scVelo and velocyto.
//
// API:
//   collapse_intron_to_gene(intron_csc, gm)  →  GeneCSC<uint32_t>
//   make_empty_gene_csc<T>(nrows, ncols)      →  GeneCSC<T>
//
// Note: collapse_intron_to_gene is templated on both the CSC type and
// the GeneModel type, so it works with any duck-compatible GeneModel
// (real singlet::GeneModel or test mocks). This header has no htslib
// dependency and can be included in lightweight unit tests.

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {

/// Gene-level sparse matrix in CSC format.
/// Canonical layout lives in sparse_accumulator.h as
/// SparseAccumulator<OutT>::CSCMatrix; GeneCSC is just an alias of it.
#ifndef SINGLET_GENE_CSC_DEFINED
#define SINGLET_GENE_CSC_DEFINED
template <typename OutT>
using GeneCSC = typename SparseAccumulator<OutT>::CSCMatrix;
#endif  // SINGLET_GENE_CSC_DEFINED

/// Collapse intron-interval-level CSC to gene-level CSC by summing counts.
/// Mirror of collapse_exon_to_gene() using intron_to_gene() mapping.
/// @param intron_csc  Intron-interval sparse matrix (n_intron_intervals × n_cells)
/// @param gm          Gene model providing intron_to_gene(idx) and n_genes()
/// @return            Gene-level sparse matrix (n_genes × n_cells)
template <typename CSCType, typename GeneModelType>
auto collapse_intron_to_gene(const CSCType& intron_csc, const GeneModelType& gm) {
    using InT = typename std::remove_reference_t<decltype(intron_csc.data)>::value_type;
    using OutT = std::conditional_t<std::is_integral_v<InT>, uint32_t, double>;

    const uint32_t n_genes = gm.n_genes();
    const uint32_t n_cells = intron_csc.ncols;

    GeneCSC<OutT> out;
    out.nrows = n_genes;
    out.ncols = n_cells;
    out.indptr.resize(n_cells + 1, 0);

    std::vector<OutT> acc(n_genes, OutT(0));
    std::vector<uint32_t> active;
    active.reserve(1024);

    for (uint32_t j = 0; j < n_cells; ++j) {
        for (int32_t k = intron_csc.indptr[j]; k < intron_csc.indptr[j + 1]; ++k) {
            uint32_t intron_idx = static_cast<uint32_t>(intron_csc.indices[k]);
            uint32_t gene_idx = gm.intron_to_gene(intron_idx);
            if (gene_idx < n_genes) {
                if (acc[gene_idx] == OutT(0)) active.push_back(gene_idx);
                acc[gene_idx] += static_cast<OutT>(intron_csc.data[k]);
            }
        }
        std::sort(active.begin(), active.end());
        for (uint32_t g : active) {
            out.indices.push_back(static_cast<int32_t>(g));
            out.data.push_back(acc[g]);
            acc[g] = OutT(0);
        }
        active.clear();
        out.indptr[j + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

/// Create an empty (all-zero) gene-level CSC matrix.
/// Used for placeholder ambiguous_counts in velocity output.
template <typename T>
GeneCSC<T> make_empty_gene_csc(uint32_t nrows, uint32_t ncols) {
    GeneCSC<T> csc;
    csc.nrows = nrows;
    csc.ncols = ncols;
    csc.indptr.assign(ncols + 1, 0);
    return csc;
}

}  // namespace singlet
