// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_nmf_new.hpp
//
// Rcpp bindings for advanced NMF kernel (cycle 52b):
//   csi_gep_cpp — wraps singlet::gpu::nmf::csi_gep (cycle 28 kernel)
//
// CSI-GEP (Consensus Sparse Identification — Gene Expression Programs):
//   Multi-run NMF with consensus cluster assignment for robust GEP discovery.
//   Combines L1-regularized NMF runs with cosine-similarity consensus clustering
//   to extract stable, biologically reproducible gene expression programs.
//   SOTA analog: cNMF (Python), RcppML::nmf() with n_runs, SignatureAnalyzer
//
// API:
//   csi_gep(sce, k, n_runs, ...) → sce with rowData + reducedDim + metadata
//
// Host-copy overhead:
//   O(nnz) host→device; O(n_genes × k + n_cells × k) device→host per run;
//   consensus step: O(k × k × n_runs) on device, O(k) device→host (stability).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/nmf/csi_gep.h>
#include <singlet/gpu/nmf/csi_gep_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// csi_gep_cpp
//
// Consensus Sparse NMF for gene expression program discovery.
// Analog of cNMF.run() + cNMF.consensus() / RcppML::nmf(n_runs=N, consensus=T).
//
// Parameters:
//   dgc         — dgCMatrix [genes × cells], tpm-normalized counts
//   k           — int, number of programs (factors)
//   n_runs      — int, independent NMF runs for consensus (default 20)
//   l1          — double, L1 regularization on H (usage matrix, default 0.05)
//   tol         — double, convergence tolerance (default 1e-4)
//   max_iter    — int, max iterations per run (default 1000)
//   min_stability — double, minimum cosine similarity for a stable program (0–1)
//   seed        — int
//
// Returns:
//   List{
//     "W"           — NumericMatrix [n_genes × k], consensus gene loadings
//     "H"           — NumericMatrix [k × n_cells], consensus cell usage
//     "stability"   — NumericVector [k], per-program cosine stability score
//     "n_stable"    — int, programs meeting min_stability threshold
//     "usage"       — NumericMatrix [n_cells × k], H transposed (R convention)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List csi_gep_cpp(
        Rcpp::S4 dgc,
        int      k,
        int      n_runs,
        double   l1,
        double   tol,
        int      max_iter,
        double   min_stability,
        int      seed)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("csi_gep_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if (k < 1) Rcpp::stop("csi_gep_cpp: k must be >= 1.");
    if (n_runs < 1) Rcpp::stop("csi_gep_cpp: n_runs must be >= 1.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("csi_gep_cpp upload: %s", e.what()); }

    singlet::gpu::nmf::CsiGepConfig cfg;
    cfg.k             = k;
    cfg.n_runs        = n_runs;
    cfg.l1            = static_cast<float>(l1);
    cfg.tol           = static_cast<float>(tol);
    cfg.max_iter      = max_iter;
    cfg.min_stability = static_cast<float>(min_stability);
    cfg.seed          = static_cast<uint64_t>(seed);

    singlet::gpu::nmf::CsiGepResult res;
    try {
        res = singlet::gpu::nmf::csi_gep(d, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("csi_gep_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // W [n_genes × k] col-major on device
    const std::size_t w_sz = static_cast<std::size_t>(n_genes) * k;
    std::vector<float> h_W(w_sz);
    cudaMemcpy(h_W.data(), res.W.data(), w_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // H [k × n_cells] col-major on device
    const std::size_t h_sz = static_cast<std::size_t>(k) * n_cells;
    std::vector<float> h_H(h_sz);
    cudaMemcpy(h_H.data(), res.H.data(), h_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // stability [k]
    std::vector<float> h_stab(k);
    cudaMemcpy(h_stab.data(), res.stability.data(), k * sizeof(float), cudaMemcpyDeviceToHost);

    int h_n_stable = 0;
    cudaMemcpy(&h_n_stable, res.n_stable.data(), sizeof(int), cudaMemcpyDeviceToHost);

    // Build R matrices
    Rcpp::NumericMatrix r_W(n_genes, k);
    for (int f = 0; f < k; ++f)
        for (int g = 0; g < n_genes; ++g)
            r_W(g, f) = static_cast<double>(h_W[static_cast<std::size_t>(f) * n_genes + g]);

    Rcpp::NumericMatrix r_H(k, n_cells);
    for (int c = 0; c < n_cells; ++c)
        for (int f = 0; f < k; ++f)
            r_H(f, c) = static_cast<double>(h_H[static_cast<std::size_t>(f) * n_cells + c]);

    // H transposed [n_cells × k] for R reducedDim convention
    Rcpp::NumericMatrix r_usage(n_cells, k);
    for (int f = 0; f < k; ++f)
        for (int c = 0; c < n_cells; ++c)
            r_usage(c, f) = r_H(f, c);

    Rcpp::NumericVector r_stab(k);
    for (int f = 0; f < k; ++f) r_stab[f] = static_cast<double>(h_stab[f]);

    return Rcpp::List::create(
        Rcpp::Named("W")           = r_W,
        Rcpp::Named("H")           = r_H,
        Rcpp::Named("stability")   = r_stab,
        Rcpp::Named("n_stable")    = h_n_stable,
        Rcpp::Named("usage")       = r_usage
    );
}

}  // namespace singlet_gpu_r
