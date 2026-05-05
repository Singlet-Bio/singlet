// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: original (Wilcoxon binned + Welch t)
//
// de/types.h — Shared result types for differential-expression kernels.
//
// Both wilcoxon.h and ttest.h return a per-cluster result containing the top-N
// marker genes (device-resident indices, scores, fold-changes, p-values, BH-adjusted
// p-values). Placed in a shared header so that callers can hold both result types
// without including either DE implementation.

#pragma once

#include <vector>
#include <singlet-gpu/core/types.h>

namespace singlet_gpu {
namespace de {

// Per-cluster top-N marker gene result.
// All device buffers have length == min(top_n, n_genes).
// gene_indices: gene row-indices in the input matrix, sorted by |z_score| descending.
// z_scores: the test statistic (Wilcoxon z or Welch t) per marker gene.
// log2_fc: log2((mean_in + eps) / (mean_out + eps)), eps = 1e-9.
// p_values: two-sided p-value via erfcf(|z| / sqrt(2)).
// p_adj: Benjamini–Hochberg FDR (over all n_genes for this cluster).
struct ClusterMarkers {
    int                              cluster_id;
    core::DeviceMemory<int>          gene_indices;
    core::DeviceMemory<float>        z_scores;
    core::DeviceMemory<float>        log2_fc;
    core::DeviceMemory<float>        p_values;
    core::DeviceMemory<float>        p_adj;
};

struct WilcoxonResult {
    std::vector<ClusterMarkers> per_cluster;  // length == n_clusters
};

struct TtestResult {
    std::vector<ClusterMarkers> per_cluster;  // length == n_clusters
};

}  // namespace de
}  // namespace singlet_gpu
