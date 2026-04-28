// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_integrate.hpp
//
// Rcpp bindings for cycle 14 batch integration kernels:
//   harmony_cpp  — wraps singlet_gpu::integrate::harmony_gpu
//   bbknn_cpp    — wraps singlet_gpu::integrate::bbknn_gpu
//
// Input format:
//   harmony_cpp:
//     embedding  — NumericMatrix [cells × n_pcs] (reducedDim convention)
//     batch_ids  — IntegerVector [n_cells], 0-based batch labels
//     n_clusters — int, Harmony K parameter
//     max_iter   — int, outer iterations
//     tol        — double, convergence tolerance
//     seed       — int
//
//   bbknn_cpp:
//     embedding  — NumericMatrix [cells × n_pcs]
//     batch_ids  — IntegerVector [n_cells], 0-based batch labels
//     neighbors_within_batch — int, k per batch
//     n_pcs      — int (may be < embedding ncol; 0 = use all)
//     metric     — "euclidean" | "cosine"
//
// Host-copy overhead:
//   harmony_cpp: O(n_cells × n_pcs) host→device (embedding) +
//                O(n_cells) host→device (batch_ids);
//                O(n_cells × n_pcs) device→host (corrected embedding).
//                WARNING: per-iteration device→host for scalar convergence
//                check only (≤4 bytes per iteration, within the allowed exception).
//   bbknn_cpp:   O(n_cells × n_pcs) host→device;
//                O(n_cells × k_total) device→host (graph indices + distances).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/integrate/harmony.h>
#include <singlet-gpu/integrate/bbknn.h>
#include <singlet-gpu/integrate/types.h>

#include "_r_to_eigen.hpp"
#include "_bind_graph.hpp"   // r_matrix_to_device_dense

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// upload_batch_ids
//
// Convert an R integer/factor/character vector of batch labels to a 0-based
// integer device buffer.  Returns the number of distinct batches detected.
// ---------------------------------------------------------------------------
inline singlet_gpu::core::DeviceMemory<int>
upload_batch_ids(const Rcpp::RObject& r_batch,
                 int                  n_cells,
                 int&                 n_batches_out,
                 cudaStream_t         stream)
{
    std::vector<int> h_ids(n_cells);
    if (Rf_isFactor(r_batch)) {
        Rcpp::IntegerVector fct(r_batch);
        for (int i = 0; i < n_cells; ++i) h_ids[i] = fct[i] - 1;  // 1-based → 0-based
        n_batches_out = Rf_nlevels(r_batch);
    } else if (TYPEOF(r_batch) == INTSXP) {
        Rcpp::IntegerVector iv(r_batch);
        for (int i = 0; i < n_cells; ++i) h_ids[i] = iv[i];
        int max_id = *std::max_element(h_ids.begin(), h_ids.end());
        n_batches_out = max_id + 1;
    } else if (TYPEOF(r_batch) == STRSXP) {
        Rcpp::CharacterVector cv(r_batch);
        std::vector<std::string> lvls;
        std::unordered_map<std::string, int> lmap;
        for (int i = 0; i < n_cells; ++i) {
            std::string s = Rcpp::as<std::string>(cv[i]);
            auto it = lmap.find(s);
            if (it == lmap.end()) {
                lmap[s] = static_cast<int>(lvls.size());
                lvls.push_back(s);
            }
            h_ids[i] = lmap[s];
        }
        n_batches_out = static_cast<int>(lvls.size());
    } else {
        throw std::runtime_error("upload_batch_ids: unsupported type for batch_ids");
    }

    singlet_gpu::core::DeviceMemory<int> d_ids(n_cells);
    cudaError_t err = cudaMemcpyAsync(d_ids.get(), h_ids.data(),
                                      n_cells * sizeof(int),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("upload_batch_ids memcpy: ") +
                                 cudaGetErrorString(err));
    cudaStreamSynchronize(stream);
    return d_ids;
}

// ---------------------------------------------------------------------------
// harmony_cpp
//
// R-callable binding for GPU Harmony batch correction.
// Mirrors harmony::RunHarmony():
//   - takes a cell × dim embedding + batch key
//   - returns a corrected cell × dim embedding stored as a new reducedDim
//
// Parameters:
//   embedding   — NumericMatrix [n_cells × n_pcs]
//   batch_ids   — IntegerVector [n_cells] (0-based batch IDs)
//   n_clusters  — int, number of Harmony clusters K (default 20)
//   max_iter    — int, maximum Harmony outer iterations (default 10)
//   tol         — double, convergence tolerance on cluster assignments
//   seed        — int, RNG seed for cluster init
//
// Returns:
//   NumericMatrix [n_cells × n_pcs] (corrected embedding)
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix harmony_cpp(
        Rcpp::NumericMatrix embedding,
        Rcpp::IntegerVector batch_ids,
        int                 n_clusters,
        int                 max_iter,
        double              tol,
        int                 seed)
{
    const int n_cells = embedding.nrow();
    const int n_pcs   = embedding.ncol();

    if (n_cells <= 0 || n_pcs <= 0)
        Rcpp::stop("harmony_cpp: empty embedding matrix.");
    if (batch_ids.size() != n_cells)
        Rcpp::stop("harmony_cpp: batch_ids length (%d) != n_cells (%d).",
                   (int)batch_ids.size(), n_cells);
    if (n_clusters < 1)
        Rcpp::stop("harmony_cpp: n_clusters must be >= 1.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload embedding host → device
    singlet_gpu::core::DeviceDense d_emb;
    try { d_emb = r_matrix_to_device_dense(embedding, stream); }
    catch (const std::exception& e) { Rcpp::stop("harmony_cpp upload embedding: %s", e.what()); }

    // Upload batch IDs
    int n_batches = 0;
    singlet_gpu::core::DeviceMemory<int> d_batch;
    try { d_batch = upload_batch_ids(Rcpp::RObject(batch_ids), n_cells, n_batches, stream); }
    catch (const std::exception& e) { Rcpp::stop("harmony_cpp upload batch_ids: %s", e.what()); }

    // Configure Harmony
    singlet_gpu::integrate::HarmonyConfig cfg;
    cfg.n_clusters = n_clusters;
    cfg.max_iter   = max_iter;
    cfg.tol        = static_cast<float>(tol);
    cfg.seed       = static_cast<uint64_t>(seed);

    // Run Harmony kernel
    singlet_gpu::integrate::HarmonyResult res;
    try {
        res = singlet_gpu::integrate::harmony_gpu(d_emb, d_batch, n_batches, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("harmony_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy corrected embedding device → host
    const std::size_t sz = static_cast<std::size_t>(n_cells) * n_pcs;
    std::vector<float> h_corrected(sz);
    {
        cudaError_t err = cudaMemcpy(h_corrected.data(), res.corrected.data(),
                                     sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("harmony_cpp copyback: %s", cudaGetErrorString(err));
    }

    // Build R NumericMatrix [n_cells × n_pcs]
    Rcpp::NumericMatrix r_corrected(n_cells, n_pcs);
    for (int c = 0; c < n_cells; ++c) {
        for (int p = 0; p < n_pcs; ++p) {
            r_corrected(c, p) = static_cast<double>(
                h_corrected[static_cast<std::size_t>(c) * n_pcs + p]);
        }
    }

    // Preserve column names from input embedding if present
    Rcpp::CharacterVector col_nms = Rcpp::colnames(embedding);
    if (col_nms.size() == n_pcs) Rcpp::colnames(r_corrected) = col_nms;

    return r_corrected;
}

// ---------------------------------------------------------------------------
// bbknn_cpp
//
// R-callable binding for GPU batch-balanced kNN graph construction.
// Mirrors bbknnR::bbknn():
//   - computes k nearest neighbours within each batch, then merges
//   - returns indices + distances for use in Leiden/UMAP
//
// Parameters:
//   embedding              — NumericMatrix [n_cells × n_dims]
//   batch_ids              — IntegerVector [n_cells] (0-based)
//   neighbors_within_batch — int, k per batch (default 3)
//   n_pcs                  — int, number of dims to use (0 = use all)
//   metric                 — "euclidean" | "cosine"
//
// Returns:
//   Rcpp::List{
//     "indices"       — IntegerMatrix [n_cells × k_total] (0-based)
//     "distances"     — NumericMatrix [n_cells × k_total]
//     "connectivities"— NumericMatrix [n_cells × k_total] (Gaussian weights)
//     "n_batches"     — int
//     "k_per_batch"   — int
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List bbknn_cpp(
        Rcpp::NumericMatrix embedding,
        Rcpp::IntegerVector batch_ids,
        int                 neighbors_within_batch,
        int                 n_pcs,
        std::string         metric)
{
    const int n_cells = embedding.nrow();
    const int n_dims  = embedding.ncol();

    if (n_cells <= 0 || n_dims <= 0)
        Rcpp::stop("bbknn_cpp: empty embedding matrix.");
    if (batch_ids.size() != n_cells)
        Rcpp::stop("bbknn_cpp: batch_ids length (%d) != n_cells (%d).",
                   (int)batch_ids.size(), n_cells);
    if (neighbors_within_batch < 1)
        Rcpp::stop("bbknn_cpp: neighbors_within_batch must be >= 1.");

    // n_pcs = 0 means use all dims
    const int use_dims = (n_pcs <= 0 || n_pcs > n_dims) ? n_dims : n_pcs;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Slice embedding to use_dims columns if needed
    Rcpp::NumericMatrix emb_slice = embedding;
    if (use_dims < n_dims) {
        emb_slice = Rcpp::NumericMatrix(n_cells, use_dims);
        for (int c = 0; c < n_cells; ++c)
            for (int p = 0; p < use_dims; ++p)
                emb_slice(c, p) = embedding(c, p);
    }

    singlet_gpu::core::DeviceDense d_emb;
    try { d_emb = r_matrix_to_device_dense(emb_slice, stream); }
    catch (const std::exception& e) { Rcpp::stop("bbknn_cpp upload embedding: %s", e.what()); }

    int n_batches = 0;
    singlet_gpu::core::DeviceMemory<int> d_batch;
    try { d_batch = upload_batch_ids(Rcpp::RObject(batch_ids), n_cells, n_batches, stream); }
    catch (const std::exception& e) { Rcpp::stop("bbknn_cpp upload batch_ids: %s", e.what()); }

    singlet_gpu::integrate::BBKNNConfig cfg;
    cfg.neighbors_within_batch = neighbors_within_batch;
    if      (metric == "euclidean") cfg.metric = singlet_gpu::integrate::BBKNNMetric::Euclidean;
    else if (metric == "cosine")    cfg.metric = singlet_gpu::integrate::BBKNNMetric::Cosine;
    else Rcpp::stop("bbknn_cpp: unknown metric '%s'. Use 'euclidean' or 'cosine'.", metric.c_str());

    singlet_gpu::integrate::BBKNNResult res;
    try {
        res = singlet_gpu::integrate::bbknn_gpu(d_emb, d_batch, n_batches, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("bbknn_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int k_total = res.k_total;
    const std::size_t nk = static_cast<std::size_t>(n_cells) * k_total;

    std::vector<int>   h_indices(nk);
    std::vector<float> h_dists  (nk);
    std::vector<float> h_conn   (nk);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_indices.data(), res.indices.data(),       nk * sizeof(int),   cudaMemcpyDeviceToHost), "indices");
        chk(cudaMemcpy(h_dists.data(),   res.distances.data(),     nk * sizeof(float), cudaMemcpyDeviceToHost), "distances");
        chk(cudaMemcpy(h_conn.data(),    res.connectivities.data(), nk * sizeof(float), cudaMemcpyDeviceToHost), "connectivities");
    } catch (const std::exception& e) {
        Rcpp::stop("bbknn_cpp copyback: %s", e.what());
    }

    Rcpp::IntegerMatrix r_idx  (n_cells, k_total);
    Rcpp::NumericMatrix r_dist (n_cells, k_total);
    Rcpp::NumericMatrix r_conn (n_cells, k_total);

    for (int i = 0; i < n_cells; ++i) {
        for (int j = 0; j < k_total; ++j) {
            const std::size_t off = static_cast<std::size_t>(i) * k_total + j;
            r_idx (i, j) = h_indices[off];
            r_dist(i, j) = static_cast<double>(h_dists[off]);
            r_conn(i, j) = static_cast<double>(h_conn[off]);
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("indices")        = r_idx,
        Rcpp::Named("distances")      = r_dist,
        Rcpp::Named("connectivities") = r_conn,
        Rcpp::Named("n_batches")      = n_batches,
        Rcpp::Named("k_per_batch")    = neighbors_within_batch
    );
}

}  // namespace singlet_gpu_r
