// SPDX-License-Identifier: MIT
// integrates: original (first GPU GSEA implementation)
//
// gsea/types.h — Shared structs and config for the gsea module.
//
// This header is the only dependency shared between fgsea.h and aucell.h.
// No CUDA runtime includes needed here.

#pragma once

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/anno/types.h>

#include <string>
#include <vector>
#include <cstdint>

namespace singlet::gpu {
namespace gsea {

// ---------------------------------------------------------------------------
// Preranked GSEA (fgsea-style) config + result
// ---------------------------------------------------------------------------

struct FgseaConfig {
    int      min_perm              = 1000;   // minimum permutations per pathway
    int      max_perm              = 10000;  // maximum permutations (extreme pathways)
    float    adaptive_target_pvalue = 0.05f; // p-value threshold for early stopping
    int      min_set_size          = 15;     // pathways smaller than this are skipped
    int      max_set_size          = 500;    // pathways larger than this are skipped
    uint64_t seed                  = 0;      // cuRAND Philox4x32 seed; same seed → same perms
};

struct PathwayResult {
    std::string name;            // pathway name (from GeneSetDB::set_names)
    float       es          = 0; // enrichment score (raw running-sum extremum)
    float       nes         = 0; // normalized ES = ES / mean(|ES_perm|)
    float       p_value     = 1; // empirical permutation p-value
    float       q_value     = 1; // BH-adjusted p-value (FDR q-value)
    int         n_member_genes = 0; // members in [min_set_size, max_set_size] range
    int         n_perms_used   = 0; // actual perms run (adaptive stopping)
};

struct FgseaResult {
    std::vector<PathwayResult> pathways; // one entry per pathway that passed size filter
};

// ---------------------------------------------------------------------------
// AUCell config + result
// ---------------------------------------------------------------------------

struct AUCellConfig {
    int      top_k_genes = 500;   // top-K genes per cell (recovery curve truncation)
    int      n_bins      = 4096;  // B: histogram bins for log1p-rank approximation
    int      cell_tile   = 65536; // cells processed per tile (controls GPU workspace)
    uint64_t seed        = 0;     // reserved; AUCell is deterministic by construction
};

struct AUCellResult {
    singlet::gpu::core::DeviceMemory<float> scores; // [n_pathways × n_cells], column-major
    int n_pathways = 0;
    int n_cells    = 0;
};

} // namespace gsea
} // namespace singlet::gpu
