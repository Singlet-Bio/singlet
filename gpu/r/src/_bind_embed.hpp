// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_embed.hpp
//
// Rcpp binding for the cycle 10 UMAP kernel:
//   umap_cpp  — wraps singlet_gpu::embed::umap
//
// Input: kNN graph as IntegerMatrix (indices) + NumericMatrix (distances)
//   from neighbors_cpp.  The precomputed-kNN path avoids recomputing the graph
//   inside cuML; see embed/umap.h for the API contract.
//
// Output: NumericMatrix [n_cells × n_components] written to reducedDim(sce, "UMAP").
//
// Determinism: Random init + fixed seed → deterministic across runs.
//   Spectral init (init="spectral") is NOT deterministic (cuML GitHub #6696).
//   Document in R wrapper.
//
// Host-copy overhead:
//   Input: O(n*k) host→device (indices + distances).
//   Output: O(n*n_components) device→host.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/graph/knn.h>
#include <singlet-gpu/embed/umap.h>

#include "_r_to_eigen.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// umap_cpp
//
// R-callable binding for GPU UMAP.
//
// Parameters:
//   indices     — IntegerMatrix [cells × k], 0-based (from neighbors_cpp)
//   distances   — NumericMatrix [cells × k]
//   n_components — int (typically 2 or 3)
//   min_dist    — float
//   spread      — float
//   init        — "random" | "spectral"
//   seed        — int
//
// Returns:
//   Rcpp::NumericMatrix [n_cells × n_components] — UMAP embedding.
//   R side (umap.R) stores in reducedDim(sce, "UMAP").
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix umap_cpp(
        Rcpp::IntegerMatrix indices,
        Rcpp::NumericMatrix distances,
        int                 n_components,
        double              min_dist,
        double              spread,
        std::string         init,
        int                 seed)
{
    const int n = indices.nrow();
    const int k = indices.ncol();

    if (n <= 0 || k <= 0)
        Rcpp::stop("umap_cpp: empty indices matrix");
    if (n_components < 1)
        Rcpp::stop("umap_cpp: n_components must be >= 1");
    if (indices.nrow() != distances.nrow() || indices.ncol() != distances.ncol())
        Rcpp::stop("umap_cpp: indices and distances dimensions must match");

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Reconstruct KnnResult on device
    const std::size_t nk = static_cast<std::size_t>(n) * k;

    singlet_gpu::graph::KnnResult knn;
    knn.n = n;
    knn.k = k;
    knn.backend_used = singlet_gpu::graph::KnnBackend::Exact;

    knn.row_offsets = singlet_gpu::core::DeviceMemory<int>(n + 1);
    knn.neighbors   = singlet_gpu::core::DeviceMemory<int>(nk);
    knn.distances   = singlet_gpu::core::DeviceMemory<float>(nk);

    // Uniform row offsets
    std::vector<int> h_offsets(n + 1);
    for (int i = 0; i <= n; ++i) h_offsets[i] = i * k;

    std::vector<int>   h_nbrs(nk);
    std::vector<float> h_dist(nk);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < k; ++j) {
            const std::size_t off = static_cast<std::size_t>(i) * k + j;
            h_nbrs[off] = indices(i, j);
            h_dist[off] = static_cast<float>(distances(i, j));
        }
    }

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpyAsync(knn.row_offsets.data(), h_offsets.data(),
                            (n + 1) * sizeof(int),  cudaMemcpyHostToDevice, stream), "umap offsets");
        chk(cudaMemcpyAsync(knn.neighbors.data(),   h_nbrs.data(),
                            nk       * sizeof(int),  cudaMemcpyHostToDevice, stream), "umap neighbors");
        chk(cudaMemcpyAsync(knn.distances.data(),   h_dist.data(),
                            nk       * sizeof(float),cudaMemcpyHostToDevice, stream), "umap distances");
        chk(cudaStreamSynchronize(stream), "umap upload sync");
    } catch (const std::exception& e) {
        Rcpp::stop("umap_cpp upload: %s", e.what());
    }

    // Configure UMAP
    singlet_gpu::embed::UmapConfig cfg;
    cfg.n_components = n_components;
    cfg.min_dist     = static_cast<float>(min_dist);
    cfg.spread       = static_cast<float>(spread);
    cfg.seed         = static_cast<uint64_t>(seed);

    if (init == "random")   cfg.init = singlet_gpu::embed::UmapInit::Random;
    else if (init == "spectral") cfg.init = singlet_gpu::embed::UmapInit::Spectral;
    else Rcpp::stop("umap_cpp: unknown init '%s'. Use 'random' or 'spectral'.",
                    init.c_str());

    // Run UMAP
    singlet_gpu::embed::UmapResult res;
    try {
        res = singlet_gpu::embed::umap(knn, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("umap_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy embedding device → host
    const std::size_t n_emb = static_cast<std::size_t>(n) * n_components;
    std::vector<float> h_emb(n_emb);
    try {
        chk(cudaMemcpy(h_emb.data(), res.embedding.data(),
                       n_emb * sizeof(float), cudaMemcpyDeviceToHost), "umap copyback");
    } catch (const std::exception& e) {
        Rcpp::stop("umap_cpp copyback: %s", e.what());
    }

    // Build R matrix [n × n_components] (row-major from device, col-major in R)
    Rcpp::NumericMatrix r_emb(n, n_components);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n_components; ++j) {
            r_emb(i, j) = static_cast<double>(
                h_emb[static_cast<std::size_t>(i) * n_components + j]);
        }
    }

    return r_emb;
}

}  // namespace singlet_gpu_r
