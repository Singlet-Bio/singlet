// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_lineage.hpp
//
// Rcpp bindings for cycle 16 MT lineage tracing kernel:
//   mt_lineage_call_clones_cpp — wraps singlet_gpu::anno::mt_lineage_call_clones
//
// Input format:
//   alt_matrix    — dgCMatrix [MT sites × cells], assays(sce)$mt_alt
//   depth_matrix  — dgCMatrix [MT sites × cells], assays(sce)$mt_depth
//   min_depth     — int, minimum depth at a site to call a variant
//   min_cells_alt — int, minimum cells with alt allele to declare informative site
//   min_vaf       — float, minimum variant allele frequency
//   min_K         — int, minimum number of clones to test
//   max_K         — int, maximum number of clones to test (auto-select by BIC)
//   seed          — int, RNG seed for EM init
//
// Host-copy overhead:
//   Input: 2 × O(nnz) host→device (sparse matrices).
//   Output: O(n_cells) device→host (clone IDs) +
//           O(n_sites) device→host (informative site mask) +
//           O(n_sites × n_clones) device→host (heteroplasmy matrix).
//
// Design note: alt and depth come from singlify's mt_alleles.1pz (which stores
// alt allele counts) and an associated depth matrix.  The R wrapper (lineage.R)
// provides read_pz_mt_sce() to load these into an SCE from a singlify output
// directory before calling detect_clones().
//
// Closest CPU reference: MitoTrace (R) and mgatk / MQuad (Python).
// The singlet-gpu kernel uses GPU-accelerated EM clustering on the VAF matrix
// restricted to informative sites (VAF > min_vaf, depth >= min_depth).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/anno/mt_lineage.h>
#include <singlet-gpu/anno/mt_lineage_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// mt_lineage_call_clones_cpp
//
// R-callable binding for GPU MT lineage tracing + clone calling.
// Closest published analog: MitoTrace (R) / MQuad (Python).
//
// Parameters:
//   alt_matrix    — dgCMatrix [MT sites × cells]
//   depth_matrix  — dgCMatrix [MT sites × cells]
//   min_depth     — int, minimum depth at site to include in VAF computation
//   min_cells_alt — int, minimum cells with alt allele (≥1 count) at a site
//   min_vaf       — double, minimum mean VAF across cells to call site informative
//   min_K         — int, minimum number of clones to evaluate
//   max_K         — int, maximum number of clones to evaluate (BIC-selected)
//   seed          — int, EM initialisation seed
//
// Returns:
//   Rcpp::List{
//     "clone_ids"           — IntegerVector [n_cells] (0-based clone label)
//     "n_clones"            — int (BIC-optimal K)
//     "informative_sites"   — IntegerVector (0-based site indices that passed filter)
//     "vaf_matrix"          — NumericMatrix [n_informative_sites × n_cells]
//     "clone_probs"         — NumericMatrix [n_cells × n_clones] (posterior prob)
//     "bic"                 — NumericVector [max_K - min_K + 1] (BIC per K)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List mt_lineage_call_clones_cpp(
        Rcpp::S4 alt_matrix,
        Rcpp::S4 depth_matrix,
        int      min_depth,
        int      min_cells_alt,
        double   min_vaf,
        int      min_K,
        int      max_K,
        int      seed)
{
    HostCsc h_alt, h_dep;
    try { h_alt = r_sparse_to_host_csc(alt_matrix);   }
    catch (const std::exception& e) { Rcpp::stop("mt_lineage_call_clones_cpp alt: %s", e.what()); }
    try { h_dep = r_sparse_to_host_csc(depth_matrix); }
    catch (const std::exception& e) { Rcpp::stop("mt_lineage_call_clones_cpp depth: %s", e.what()); }

    const int n_sites = h_alt.n_rows;
    const int n_cells = h_alt.n_cols;

    if (h_dep.n_rows != n_sites || h_dep.n_cols != n_cells)
        Rcpp::stop("mt_lineage_call_clones_cpp: alt/depth dimension mismatch.");
    if (min_K < 1 || max_K < min_K)
        Rcpp::stop("mt_lineage_call_clones_cpp: require 1 <= min_K (%d) <= max_K (%d).",
                   min_K, max_K);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_alt, d_dep;
    try { d_alt = upload_host_csc_to_device(h_alt, stream); }
    catch (const std::exception& e) { Rcpp::stop("mt_lineage_call_clones_cpp upload alt: %s", e.what()); }
    try { d_dep = upload_host_csc_to_device(h_dep, stream); }
    catch (const std::exception& e) { Rcpp::stop("mt_lineage_call_clones_cpp upload depth: %s", e.what()); }

    singlet_gpu::anno::MtLineageConfig cfg;
    cfg.min_depth     = min_depth;
    cfg.min_cells_alt = min_cells_alt;
    cfg.min_vaf       = static_cast<float>(min_vaf);
    cfg.min_K         = min_K;
    cfg.max_K         = max_K;
    cfg.seed          = static_cast<uint64_t>(seed);

    singlet_gpu::anno::MtLineageResult res;
    try {
        res = singlet_gpu::anno::mt_lineage_call_clones(d_alt, d_dep, n_sites, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("mt_lineage_call_clones_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Clone IDs [n_cells]
    std::vector<int> h_clone_ids(n_cells);
    {
        cudaError_t err = cudaMemcpy(h_clone_ids.data(), res.clone_ids.data(),
                                     n_cells * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("mt_lineage_call_clones_cpp clone_ids copyback: %s", cudaGetErrorString(err));
    }

    // Informative site indices [n_informative]
    const int n_inf = static_cast<int>(res.informative_sites_count);
    std::vector<int> h_inf_sites(n_inf > 0 ? n_inf : 1);
    if (n_inf > 0) {
        cudaError_t err = cudaMemcpy(h_inf_sites.data(), res.informative_site_indices.data(),
                                     n_inf * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("mt_lineage_call_clones_cpp inf_sites copyback: %s", cudaGetErrorString(err));
    }

    // VAF matrix [n_inf × n_cells]
    const std::size_t vaf_sz = static_cast<std::size_t>(n_inf) * n_cells;
    std::vector<float> h_vaf(vaf_sz > 0 ? vaf_sz : 1);
    if (vaf_sz > 0) {
        cudaError_t err = cudaMemcpy(h_vaf.data(), res.vaf_matrix.data(),
                                     vaf_sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("mt_lineage_call_clones_cpp vaf copyback: %s", cudaGetErrorString(err));
    }

    // Clone probabilities [n_cells × n_clones]
    const int n_clones = res.n_clones;
    const std::size_t prob_sz = static_cast<std::size_t>(n_cells) * n_clones;
    std::vector<float> h_probs(prob_sz > 0 ? prob_sz : 1);
    if (prob_sz > 0) {
        cudaError_t err = cudaMemcpy(h_probs.data(), res.clone_probs.data(),
                                     prob_sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("mt_lineage_call_clones_cpp clone_probs copyback: %s", cudaGetErrorString(err));
    }

    // BIC scores [max_K - min_K + 1]
    const int n_bic = max_K - min_K + 1;
    std::vector<float> h_bic(n_bic);
    {
        cudaError_t err = cudaMemcpy(h_bic.data(), res.bic.data(),
                                     n_bic * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("mt_lineage_call_clones_cpp bic copyback: %s", cudaGetErrorString(err));
    }

    // Build R output
    Rcpp::IntegerVector r_clone_ids(n_cells);
    for (int i = 0; i < n_cells; ++i) r_clone_ids[i] = h_clone_ids[i];

    Rcpp::IntegerVector r_inf_sites(n_inf);
    for (int i = 0; i < n_inf; ++i) r_inf_sites[i] = h_inf_sites[i];

    Rcpp::NumericMatrix r_vaf(n_inf, n_cells);
    for (int s = 0; s < n_inf; ++s)
        for (int c = 0; c < n_cells; ++c)
            r_vaf(s, c) = static_cast<double>(
                h_vaf[static_cast<std::size_t>(s) * n_cells + c]);

    Rcpp::NumericMatrix r_probs(n_cells, n_clones);
    for (int c = 0; c < n_cells; ++c)
        for (int k = 0; k < n_clones; ++k)
            r_probs(c, k) = static_cast<double>(
                h_probs[static_cast<std::size_t>(c) * n_clones + k]);

    Rcpp::NumericVector r_bic(n_bic);
    for (int b = 0; b < n_bic; ++b) r_bic[b] = static_cast<double>(h_bic[b]);

    return Rcpp::List::create(
        Rcpp::Named("clone_ids")         = r_clone_ids,
        Rcpp::Named("n_clones")          = n_clones,
        Rcpp::Named("informative_sites") = r_inf_sites,
        Rcpp::Named("vaf_matrix")        = r_vaf,
        Rcpp::Named("clone_probs")       = r_probs,
        Rcpp::Named("bic")               = r_bic
    );
}

}  // namespace singlet_gpu_r
