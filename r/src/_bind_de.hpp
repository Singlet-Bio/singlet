// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_de.hpp
//
// Rcpp bindings for cycle 11 differential expression kernels:
//   wilcoxon_de_cpp  — wraps singlet::gpu::de::wilcoxon_de
//   ttest_de_cpp     — wraps singlet::gpu::de::ttest_de
//
// Both functions take:
//   dgc     — Matrix::dgCMatrix [genes × cells] (logcounts layer)
//   labels  — IntegerVector [n_cells], 0-based cluster IDs
//   top_n   — max marker genes to return per cluster
//   deterministic — bool, true → segmented scan instead of atomicAdd
//
// Both return:
//   Rcpp::List of per-cluster results, each a List with:
//     "cluster_id"  — int
//     "gene_indices"— IntegerVector (0-based gene row indices)
//     "z_scores"    — NumericVector
//     "log2_fc"     — NumericVector
//     "p_values"    — NumericVector
//     "p_adj"       — NumericVector (BH-corrected)
//
// R side (rank_genes_groups.R) stacks these into a DataFrame stored in
// metadata(sce)$rank_genes_groups.
//
// Host-copy overhead:
//   Input: O(nnz) host→device (matrix) + O(n_cells) host→device (labels).
//   Output: O(n_clusters × top_n × 5) device→host (gene indices + 4 float arrays).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/de/wilcoxon.h>
#include <singlet/gpu/de/ttest.h>
#include <singlet/gpu/de/types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// upload_labels_to_device
//
// Copy an R IntegerVector of cell cluster labels to device memory.
// Validates that all labels are in [0, n_clusters).
// ---------------------------------------------------------------------------
inline singlet::gpu::core::DeviceMemory<int>
upload_labels_to_device(const Rcpp::IntegerVector& r_labels,
                        int n_cells,
                        int& n_clusters_out,
                        cudaStream_t stream)
{
    if (r_labels.size() != n_cells)
        throw std::runtime_error("upload_labels: labels length != n_cells");

    int max_label = -1;
    for (int i = 0; i < n_cells; ++i) {
        int l = r_labels[i];
        if (l < 0) throw std::runtime_error("upload_labels: negative label found");
        if (l > max_label) max_label = l;
    }
    n_clusters_out = max_label + 1;

    std::vector<int> h_labels(n_cells);
    for (int i = 0; i < n_cells; ++i) h_labels[i] = r_labels[i];

    singlet::gpu::core::DeviceMemory<int> d_labels(n_cells);
    cudaError_t err = cudaMemcpyAsync(d_labels.get(), h_labels.data(),
                                      n_cells * sizeof(int),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("upload_labels memcpy: ") +
                                 cudaGetErrorString(err));
    cudaStreamSynchronize(stream);
    return d_labels;
}

// ---------------------------------------------------------------------------
// cluster_markers_to_r_list
//
// Convert a singlet::gpu::de::ClusterMarkers result (device-resident) into an
// R List.  O(top_n) device→host copies per cluster.
// ---------------------------------------------------------------------------
inline Rcpp::List
cluster_markers_to_r_list(const singlet::gpu::de::ClusterMarkers& cm, int top_n)
{
    if (top_n <= 0) {
        return Rcpp::List::create(
            Rcpp::Named("cluster_id")   = cm.cluster_id,
            Rcpp::Named("gene_indices") = Rcpp::IntegerVector(0),
            Rcpp::Named("z_scores")     = Rcpp::NumericVector(0),
            Rcpp::Named("log2_fc")      = Rcpp::NumericVector(0),
            Rcpp::Named("p_values")     = Rcpp::NumericVector(0),
            Rcpp::Named("p_adj")        = Rcpp::NumericVector(0)
        );
    }

    std::vector<int>   h_idx (top_n);
    std::vector<float> h_z   (top_n);
    std::vector<float> h_lfc (top_n);
    std::vector<float> h_p   (top_n);
    std::vector<float> h_padj(top_n);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };

    chk(cudaMemcpy(h_idx.data(),  cm.gene_indices.data(),
                   top_n * sizeof(int),   cudaMemcpyDeviceToHost), "gene_indices");
    chk(cudaMemcpy(h_z.data(),    cm.z_scores.data(),
                   top_n * sizeof(float), cudaMemcpyDeviceToHost), "z_scores");
    chk(cudaMemcpy(h_lfc.data(),  cm.log2_fc.data(),
                   top_n * sizeof(float), cudaMemcpyDeviceToHost), "log2_fc");
    chk(cudaMemcpy(h_p.data(),    cm.p_values.data(),
                   top_n * sizeof(float), cudaMemcpyDeviceToHost), "p_values");
    chk(cudaMemcpy(h_padj.data(), cm.p_adj.data(),
                   top_n * sizeof(float), cudaMemcpyDeviceToHost), "p_adj");

    // Convert to R vectors
    Rcpp::IntegerVector r_idx (top_n);
    Rcpp::NumericVector r_z   (top_n);
    Rcpp::NumericVector r_lfc (top_n);
    Rcpp::NumericVector r_p   (top_n);
    Rcpp::NumericVector r_padj(top_n);

    for (int i = 0; i < top_n; ++i) {
        r_idx [i] = h_idx [i];  // 0-based
        r_z   [i] = static_cast<double>(h_z   [i]);
        r_lfc [i] = static_cast<double>(h_lfc [i]);
        r_p   [i] = static_cast<double>(h_p   [i]);
        r_padj[i] = static_cast<double>(h_padj[i]);
    }

    return Rcpp::List::create(
        Rcpp::Named("cluster_id")   = cm.cluster_id,
        Rcpp::Named("gene_indices") = r_idx,
        Rcpp::Named("z_scores")     = r_z,
        Rcpp::Named("log2_fc")      = r_lfc,
        Rcpp::Named("p_values")     = r_p,
        Rcpp::Named("p_adj")        = r_padj
    );
}

// ---------------------------------------------------------------------------
// wilcoxon_de_cpp
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List wilcoxon_de_cpp(
        Rcpp::S4           dgc,
        Rcpp::IntegerVector labels,
        int                 top_n,
        bool                deterministic)
{
    // Marshal R → host CSC
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    const int n_cells = h.n_cols;

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload matrix
    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("wilcoxon_de_cpp upload: %s", e.what()); }

    // Upload labels
    int n_clusters = 0;
    singlet::gpu::core::DeviceMemory<int> d_labels;
    try { d_labels = upload_labels_to_device(labels, n_cells, n_clusters, stream); }
    catch (const std::exception& e) { Rcpp::stop("wilcoxon_de_cpp labels: %s", e.what()); }

    // Configure
    singlet::gpu::de::WilcoxonConfig cfg;
    cfg.top_n         = top_n;
    cfg.deterministic = deterministic;

    // Run kernel
    singlet::gpu::de::WilcoxonResult res;
    try {
        res = singlet::gpu::de::wilcoxon_de(d, d_labels, n_clusters, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("wilcoxon_de_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Marshal results
    Rcpp::List out(n_clusters);
    for (int c = 0; c < n_clusters; ++c) {
        const auto& cm = res.per_cluster[c];
        const int actual_top = static_cast<int>(cm.gene_indices.size());
        try {
            out[c] = cluster_markers_to_r_list(cm, actual_top);
        } catch (const std::exception& e) {
            Rcpp::stop("wilcoxon_de_cpp marshal cluster %d: %s", c, e.what());
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// ttest_de_cpp
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List ttest_de_cpp(
        Rcpp::S4           dgc,
        Rcpp::IntegerVector labels,
        int                 top_n,
        bool                deterministic)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    const int n_cells = h.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("ttest_de_cpp upload: %s", e.what()); }

    int n_clusters = 0;
    singlet::gpu::core::DeviceMemory<int> d_labels;
    try { d_labels = upload_labels_to_device(labels, n_cells, n_clusters, stream); }
    catch (const std::exception& e) { Rcpp::stop("ttest_de_cpp labels: %s", e.what()); }

    singlet::gpu::de::TtestConfig cfg;
    cfg.top_n         = top_n;
    cfg.deterministic = deterministic;

    singlet::gpu::de::TtestResult res;
    try {
        res = singlet::gpu::de::ttest_de(d, d_labels, n_clusters, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("ttest_de_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    Rcpp::List out(n_clusters);
    for (int c = 0; c < n_clusters; ++c) {
        const auto& cm = res.per_cluster[c];
        const int actual_top = static_cast<int>(cm.gene_indices.size());
        try {
            out[c] = cluster_markers_to_r_list(cm, actual_top);
        } catch (const std::exception& e) {
            Rcpp::stop("ttest_de_cpp marshal cluster %d: %s", c, e.what());
        }
    }
    return out;
}

}  // namespace singlet_gpu_r
