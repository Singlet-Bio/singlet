// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (DecoupleR-style scoring + CellTypist projection)
//
// anno/types.h — Shared structs for the annotation module.
//
// This header is the only dependency that marker_score.h and reference_map.h
// share.  Kept minimal; no CUDA or cuBLAS includes needed here.

#pragma once

#include <singlet-gpu/core/types.h>

#include <string>
#include <vector>

namespace singlet_gpu {
namespace anno {

// ---------------------------------------------------------------------------
// Gene-set database
// ---------------------------------------------------------------------------

// Host-side description of a gene-set collection (e.g., MSigDB, PanglaoDB).
// All index and weight vectors are host-side at construction; they are copied
// to device on the first call to marker_score().
struct GeneSetDB {
    std::vector<std::string>          set_names;           // n_sets names
    std::vector<std::vector<int>>     member_gene_indices; // [n_sets][variable]
    std::vector<std::vector<float>>   weights;             // [n_sets][variable], may be empty → unit weights
    int                               n_genes = 0;         // total number of genes in the background
};

// ---------------------------------------------------------------------------
// Marker scoring
// ---------------------------------------------------------------------------

enum class MarkerMethod {
    Mlm,   // multivariate linear model (DecoupleR mlm)
    Ulm,   // univariate linear model   (DecoupleR ulm)
    Wsum,  // weighted sum              (DecoupleR wsum)
    UCell, // rank-based AUC            (UCell)
};

struct MarkerScoreConfig {
    MarkerMethod method             = MarkerMethod::Mlm;
    int          min_n_genes_per_set = 5;   // sets with fewer member genes are skipped
    float        lambda_tikhonov    = -1.f; // <0 → auto = 1e-6 * trace(G^T G); Mlm only
    int          ucell_n_bins       = 4096; // histogram bins for UCell ranking trick
};

struct MarkerScoreResult {
    singlet_gpu::core::DeviceMemory<float> scores; // n_sets × n_cells, column-major
    int n_sets;
    int n_cells;
};

// ---------------------------------------------------------------------------
// CellTypist reference model
// ---------------------------------------------------------------------------

// Loaded once from a .npz file via load_celltypist_model().
// Host-side file I/O only (not a hot path).  After loading, weights and
// intercepts live on device; class_names remain on host.
struct CelltypistModel {
    singlet_gpu::core::DeviceMemory<float> weights;    // n_classes × n_pcs, row-major
    singlet_gpu::core::DeviceMemory<float> intercepts; // n_classes
    std::vector<std::string>               class_names;
    int n_classes = 0;
    int n_pcs     = 0;
};

struct RefMapResult {
    singlet_gpu::core::DeviceMemory<float> probabilities; // n_classes × n_cells, column-major
    singlet_gpu::core::DeviceMemory<int>   labels;        // n_cells (argmax class index)
};

} // namespace anno
} // namespace singlet_gpu
