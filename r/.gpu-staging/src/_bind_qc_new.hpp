// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_qc_new.hpp
//
// Rcpp bindings for advanced QC kernels (cycle 52b):
//   omni_doublet_score_cpp — wraps singlet::gpu::qc::omni_doublet_score
//   doublet_score_cpp      — wraps singlet::gpu::qc::doublet_score (cycle 31 kernel)
//
// OmniDoublet:
//   Ensemble doublet detection combining:
//   - Simulated-doublet local density (Scrublet-style)
//   - k-NN co-embedding score
//   - PCA-based classifier
//   SOTA analog: Scrublet, DoubletFinder, scDblFinder
//   R surface: detect_doublets(sce) → colData(sce)$doublet_score + $is_doublet
//
// DoubletScore (cycle 31):
//   GPU-accelerated Scrublet-style doublet detection (simulated doublet approach).
//   R surface: scrublet_score(sce) → colData(sce)$scrublet_score + $predicted_doublet
//
// Host-copy overhead:
//   O(nnz) host→device; O(n_cells × 2 fields) device→host (score + binary flag).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/qc/omni_doublet.h>
#include <singlet/gpu/qc/omni_doublet_types.h>
#include <singlet/gpu/qc/doublet_score.h>
#include <singlet/gpu/qc/doublet_score_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// omni_doublet_score_cpp
//
// Ensemble GPU doublet scoring (OmniDoublet).
// Analog of scDblFinder::scDblFinder() or DoubletFinder::doubletFinder().
//
// Parameters:
//   dgc              — dgCMatrix [genes × cells], logcounts
//   n_pcs            — int, number of PCs used in kNN / classifier (default 20)
//   n_neighbors      — int, kNN k for density estimation (default 30)
//   sim_doublet_ratio — double, ratio of simulated doublets to cells (default 2.0)
//   expected_rate    — double, prior expected doublet rate (default 0.05)
//   seed             — int
//
// Returns:
//   List{
//     "doublet_score"    — NumericVector [n_cells], continuous score [0, 1]
//     "is_doublet"       — LogicalVector [n_cells], threshold at expected_rate
//     "threshold"        — double, computed threshold
//     "n_doublets_pred"  — int, number of predicted doublets
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List omni_doublet_score_cpp(
        Rcpp::S4 dgc,
        int      n_pcs,
        int      n_neighbors,
        double   sim_doublet_ratio,
        double   expected_rate,
        int      seed)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("omni_doublet_score_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("omni_doublet_score_cpp upload: %s", e.what()); }

    singlet::gpu::qc::OmniDoubletConfig cfg;
    cfg.n_pcs              = n_pcs;
    cfg.n_neighbors        = n_neighbors;
    cfg.sim_doublet_ratio  = static_cast<float>(sim_doublet_ratio);
    cfg.expected_rate      = static_cast<float>(expected_rate);
    cfg.seed               = static_cast<uint64_t>(seed);

    singlet::gpu::qc::OmniDoubletResult res;
    try {
        res = singlet::gpu::qc::omni_doublet_score(d, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("omni_doublet_score_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    std::vector<float> h_scores(n_cells);
    std::vector<int>   h_flags (n_cells);
    float h_threshold = 0.0f;
    int   h_n_doublets = 0;

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_scores.data(), res.doublet_score.data(),
                       n_cells * sizeof(float), cudaMemcpyDeviceToHost), "scores");
        chk(cudaMemcpy(h_flags.data(), res.is_doublet.data(),
                       n_cells * sizeof(int),   cudaMemcpyDeviceToHost), "flags");
        chk(cudaMemcpy(&h_threshold,  res.threshold.data(),
                       sizeof(float), cudaMemcpyDeviceToHost), "threshold");
        chk(cudaMemcpy(&h_n_doublets, res.n_doublets.data(),
                       sizeof(int),   cudaMemcpyDeviceToHost), "n_doublets");
    } catch (const std::exception& e) {
        Rcpp::stop("omni_doublet_score_cpp copyback: %s", e.what());
    }

    Rcpp::NumericVector r_scores(n_cells);
    Rcpp::LogicalVector r_flags (n_cells);
    for (int i = 0; i < n_cells; ++i) {
        r_scores[i] = static_cast<double>(h_scores[i]);
        r_flags [i] = (h_flags[i] != 0);
    }

    return Rcpp::List::create(
        Rcpp::Named("doublet_score")   = r_scores,
        Rcpp::Named("is_doublet")      = r_flags,
        Rcpp::Named("threshold")       = static_cast<double>(h_threshold),
        Rcpp::Named("n_doublets_pred") = h_n_doublets
    );
}

// ---------------------------------------------------------------------------
// doublet_score_cpp
//
// GPU Scrublet-style doublet detection (cycle 31 kernel).
// Analog of Scrublet::scrub_doublets() / scDblFinder::scDblFinder(method="scran").
//
// Parameters:
//   dgc              — dgCMatrix [genes × cells], raw counts
//   n_pcs            — int, PCA dimensions for kNN (default 30)
//   n_neighbors      — int, kNN k (default 30)
//   sim_doublet_ratio — double (default 2.0)
//   threshold        — double or NULL (NA), if NA: automatically set by bimodality
//   seed             — int
//
// Returns:
//   List{
//     "doublet_score"      — NumericVector [n_cells]
//     "predicted_doublet"  — LogicalVector [n_cells]
//     "threshold"          — double
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List doublet_score_cpp(
        Rcpp::S4 dgc,
        int      n_pcs,
        int      n_neighbors,
        double   sim_doublet_ratio,
        double   threshold,
        int      seed)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("doublet_score_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("doublet_score_cpp upload: %s", e.what()); }

    singlet::gpu::qc::DoubletScoreConfig cfg;
    cfg.n_pcs              = n_pcs;
    cfg.n_neighbors        = n_neighbors;
    cfg.sim_doublet_ratio  = static_cast<float>(sim_doublet_ratio);
    // threshold < 0 → auto-detect via bimodality test
    cfg.threshold          = (threshold < 0.0) ? -1.0f : static_cast<float>(threshold);
    cfg.seed               = static_cast<uint64_t>(seed);

    singlet::gpu::qc::DoubletScoreResult res;
    try {
        res = singlet::gpu::qc::doublet_score(d, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("doublet_score_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    std::vector<float> h_scores(n_cells);
    std::vector<int>   h_flags (n_cells);
    float h_thresh = 0.0f;

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_scores.data(), res.scores.data(),
                       n_cells * sizeof(float), cudaMemcpyDeviceToHost), "scores");
        chk(cudaMemcpy(h_flags.data(), res.is_doublet.data(),
                       n_cells * sizeof(int),   cudaMemcpyDeviceToHost), "flags");
        chk(cudaMemcpy(&h_thresh, res.threshold.data(),
                       sizeof(float), cudaMemcpyDeviceToHost), "threshold");
    } catch (const std::exception& e) {
        Rcpp::stop("doublet_score_cpp copyback: %s", e.what());
    }

    Rcpp::NumericVector r_scores(n_cells);
    Rcpp::LogicalVector r_flags (n_cells);
    for (int i = 0; i < n_cells; ++i) {
        r_scores[i] = static_cast<double>(h_scores[i]);
        r_flags [i] = (h_flags[i] != 0);
    }

    return Rcpp::List::create(
        Rcpp::Named("doublet_score")     = r_scores,
        Rcpp::Named("predicted_doublet") = r_flags,
        Rcpp::Named("threshold")         = static_cast<double>(h_thresh)
    );
}

}  // namespace singlet_gpu_r
