// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_graph.hpp
//
// Rcpp bindings for cycle 8 (kNN) and cycle 9 (Leiden) kernels:
//   neighbors_cpp  — wraps singlet_gpu::graph::compute_knn
//   leiden_cpp     — wraps singlet_gpu::graph::leiden
//
// Input format for both:
//   neighbors_cpp: R NumericMatrix (cells × dims), dimensions stored as
//     columns (SCE reducedDim convention — cells are rows).
//   leiden_cpp: the Rcpp::List returned by neighbors_cpp (indices + distances,
//     both IntegerMatrix / NumericMatrix, cells × k).
//
// Host-copy overhead:
//   neighbors_cpp: O(n_cells × n_dims) host→device + O(n_cells × k) device→host.
//   leiden_cpp:    O(n_cells × k) host→device (graph) + O(n_cells) device→host (labels).
//   R has no native GPU memory; every call incurs two O-copies.
//
// Streams: one temporary CUDA stream per call, destroyed after synchronization.
//   Pattern is identical to cycle 24 bindings.
//
// Result storage (R side):
//   neighbors_cpp → named List { indices[n×k], distances[n×k] }
//   leiden_cpp    → named List { labels[n], n_clusters, modularity, iterations }

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/graph/leiden.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device, StreamGuard pattern

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// r_matrix_to_device_dense
//
// Copy an R NumericMatrix (rows × cols, col-major) to a DeviceDense
// (factornet::gpu::DenseMatrixGPU<float>).
//
// SCE reducedDim convention: matrix is cells × dims (rows=cells, cols=dims).
// compute_knn expects rows=cells, cols=dims — same layout. No transpose needed.
// ---------------------------------------------------------------------------
inline singlet_gpu::core::DeviceDense
r_matrix_to_device_dense(const Rcpp::NumericMatrix& mat,
                          cudaStream_t stream)
{
    const int n_rows = mat.nrow();
    const int n_cols = mat.ncol();
    if (n_rows <= 0 || n_cols <= 0)
        throw std::runtime_error("r_matrix_to_device_dense: empty matrix");

    // Allocate host float buffer (double → float cast)
    std::vector<float> h_buf(static_cast<std::size_t>(n_rows) * n_cols);
    for (int j = 0; j < n_cols; ++j) {
        for (int i = 0; i < n_rows; ++i) {
            h_buf[static_cast<std::size_t>(i) * n_cols + j] =
                static_cast<float>(mat(i, j));
        }
    }

    // Allocate DeviceDense and upload
    singlet_gpu::core::DeviceMemory<float> d_data(
        static_cast<std::size_t>(n_rows) * n_cols);

    cudaError_t err = cudaMemcpyAsync(
        d_data.get(), h_buf.data(),
        h_buf.size() * sizeof(float),
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("r_matrix_to_device_dense: ") +
                                 cudaGetErrorString(err));

    cudaStreamSynchronize(stream);

    // Wrap into DeviceDense (factornet::gpu::DenseMatrixGPU<float> is row-major;
    // compute_knn reads embedding[i*d + j] which matches our layout above).
    singlet_gpu::core::DeviceDense dense(n_rows, n_cols);
    cudaMemcpy(dense.data.get(), d_data.get(),
               h_buf.size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    return dense;
}

// ---------------------------------------------------------------------------
// neighbors_cpp
//
// R-callable binding for GPU kNN graph construction.
//
// Parameters:
//   embedding   — NumericMatrix [cells × dims] (from reducedDim, e.g. PCA)
//   n_neighbors — int k
//   metric      — "euclidean" | "cosine" | "inner"
//   seed        — int, forwarded to HNSW builder
//
// Returns:
//   Rcpp::List{
//     "indices"   — IntegerMatrix [cells × k] (0-based neighbor cell indices)
//     "distances" — NumericMatrix [cells × k]
//     "n"         — int (number of cells)
//     "k"         — int (actual k used)
//   }
//
// NOTE: indices are 0-based to match Python/scanpy convention.
//   R side (neighbors.R) converts to 1-based if needed for R indexing.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List neighbors_cpp(
        Rcpp::NumericMatrix embedding,
        int                 n_neighbors,
        std::string         metric,
        int                 seed)
{
    const int n = embedding.nrow();
    const int d = embedding.ncol();

    if (n <= 0 || d <= 0) Rcpp::stop("neighbors_cpp: empty embedding matrix");
    if (n_neighbors < 1)  Rcpp::stop("neighbors_cpp: n_neighbors must be >= 1");

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload embedding host → device
    singlet_gpu::core::DeviceDense d_emb;
    try { d_emb = r_matrix_to_device_dense(embedding, stream); }
    catch (const std::exception& e) { Rcpp::stop("neighbors_cpp upload: %s", e.what()); }

    // Configure kNN
    singlet_gpu::graph::KnnConfig cfg;
    cfg.k    = n_neighbors;
    cfg.seed = static_cast<uint64_t>(seed);

    if      (metric == "euclidean") cfg.metric = singlet_gpu::graph::DistanceMetric::L2;
    else if (metric == "cosine")    cfg.metric = singlet_gpu::graph::DistanceMetric::Cosine;
    else if (metric == "inner")     cfg.metric = singlet_gpu::graph::DistanceMetric::Inner;
    else Rcpp::stop("neighbors_cpp: unknown metric '%s'. Use 'euclidean', 'cosine', or 'inner'.",
                    metric.c_str());

    // Run kNN
    singlet_gpu::graph::KnnResult knn;
    try {
        knn = singlet_gpu::graph::compute_knn(d_emb, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("neighbors_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int actual_k = knn.k;
    const std::size_t nk = static_cast<std::size_t>(n) * actual_k;

    // Copy results device → host
    std::vector<int>   h_neighbors(nk);
    std::vector<float> h_distances(nk);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_neighbors.data(), knn.neighbors.data(),
                       nk * sizeof(int),   cudaMemcpyDeviceToHost), "knn neighbors");
        chk(cudaMemcpy(h_distances.data(), knn.distances.data(),
                       nk * sizeof(float), cudaMemcpyDeviceToHost), "knn distances");
    } catch (const std::exception& e) {
        Rcpp::stop("neighbors_cpp copyback: %s", e.what());
    }

    // Build R matrices [n × k]
    Rcpp::IntegerMatrix r_idx(n, actual_k);
    Rcpp::NumericMatrix r_dist(n, actual_k);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < actual_k; ++j) {
            const std::size_t off = static_cast<std::size_t>(i) * actual_k + j;
            r_idx (i, j) = h_neighbors[off];  // 0-based
            r_dist(i, j) = static_cast<double>(h_distances[off]);
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("indices")   = r_idx,
        Rcpp::Named("distances") = r_dist,
        Rcpp::Named("n")         = n,
        Rcpp::Named("k")         = actual_k
    );
}

// ---------------------------------------------------------------------------
// leiden_cpp
//
// R-callable binding for GPU Leiden community detection.
//
// Parameters:
//   indices         — IntegerMatrix [cells × k] (0-based, from neighbors_cpp)
//   distances       — NumericMatrix [cells × k]
//   resolution      — float modularity resolution
//   seed            — int
//   weight_function — "connectivity" | "inverse" | "gaussian"
//
// Returns:
//   Rcpp::List{
//     "labels"     — IntegerVector [n_cells] (0-based community IDs)
//     "n_clusters" — int
//     "modularity" — double
//     "iterations" — int
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List leiden_cpp(
        Rcpp::IntegerMatrix indices,
        Rcpp::NumericMatrix distances,
        double              resolution,
        int                 seed,
        std::string         weight_function)
{
    const int n = indices.nrow();
    const int k = indices.ncol();

    if (n <= 0 || k <= 0)
        Rcpp::stop("leiden_cpp: empty indices matrix");
    if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol())
        Rcpp::stop("leiden_cpp: indices and distances dimensions must match");

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload kNN graph to device — reconstruct a KnnResult
    const std::size_t nk = static_cast<std::size_t>(n) * k;

    singlet_gpu::graph::KnnResult knn;
    knn.n = n;
    knn.k = k;
    knn.backend_used = singlet_gpu::graph::KnnBackend::Exact;  // provenance only

    knn.row_offsets = singlet_gpu::core::DeviceMemory<int>(n + 1);
    knn.neighbors   = singlet_gpu::core::DeviceMemory<int>(nk);
    knn.distances   = singlet_gpu::core::DeviceMemory<float>(nk);

    // Copy row_offsets (uniform stride k)
    std::vector<int> h_offsets(n + 1);
    for (int i = 0; i <= n; ++i) h_offsets[i] = i * k;
    cudaMemcpyAsync(knn.row_offsets.data(), h_offsets.data(),
                    (n + 1) * sizeof(int), cudaMemcpyHostToDevice, stream);

    // Flatten indices (int) and distances (float)
    std::vector<int>   h_neighbors(nk);
    std::vector<float> h_distances(nk);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            const std::size_t off = static_cast<std::size_t>(i) * k + j;
            h_neighbors[off] = indices(i, j);
            h_distances[off] = static_cast<float>(distances(i, j));
        }
    }

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpyAsync(knn.neighbors.data(), h_neighbors.data(),
                            nk * sizeof(int),   cudaMemcpyHostToDevice, stream), "leiden upload neighbors");
        chk(cudaMemcpyAsync(knn.distances.data(), h_distances.data(),
                            nk * sizeof(float), cudaMemcpyHostToDevice, stream), "leiden upload distances");
        chk(cudaStreamSynchronize(stream), "leiden upload sync");
    } catch (const std::exception& e) {
        Rcpp::stop("leiden_cpp upload: %s", e.what());
    }

    // Configure Leiden
    singlet_gpu::graph::LeidenConfig cfg;
    cfg.resolution = static_cast<float>(resolution);
    cfg.seed       = static_cast<uint64_t>(seed);

    if      (weight_function == "connectivity") cfg.weight_function = singlet_gpu::graph::LeidenWeight::Connectivity;
    else if (weight_function == "inverse")      cfg.weight_function = singlet_gpu::graph::LeidenWeight::Inverse;
    else if (weight_function == "gaussian")     cfg.weight_function = singlet_gpu::graph::LeidenWeight::Gaussian;
    else Rcpp::stop("leiden_cpp: unknown weight_function '%s'. Use 'connectivity', 'inverse', or 'gaussian'.",
                    weight_function.c_str());

    // Run Leiden
    singlet_gpu::graph::LeidenResult res;
    try {
        res = singlet_gpu::graph::leiden(knn, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("leiden_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy labels device → host
    std::vector<int> h_labels(n);
    try {
        chk(cudaMemcpy(h_labels.data(), res.labels.data(),
                       n * sizeof(int), cudaMemcpyDeviceToHost), "leiden labels copyback");
    } catch (const std::exception& e) {
        Rcpp::stop("leiden_cpp copyback: %s", e.what());
    }

    Rcpp::IntegerVector r_labels(n);
    for (int i = 0; i < n; ++i) r_labels[i] = h_labels[i];

    return Rcpp::List::create(
        Rcpp::Named("labels")      = r_labels,
        Rcpp::Named("n_clusters")  = res.n_clusters,
        Rcpp::Named("modularity")  = static_cast<double>(res.modularity),
        Rcpp::Named("iterations")  = res.iterations
    );
}

}  // namespace singlet_gpu_r
