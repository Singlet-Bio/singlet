// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_streaming.hpp
//
// Rcpp binding for the cycle 7 streaming pipeline:
//   streaming_pipeline_run_cpp  — wraps singlet_gpu::streaming::run_pipeline
//
// This binding is deliberately NOT SCE-returning: the input is a vector of
// file paths whose total cell count may exceed in-memory capacity.  The result
// is a plain R list with scalar summaries and optional factor matrices.
//
// WHY a plain list rather than SCE:
//   SCE requires all cells in memory simultaneously (colData + assay).
//   A streaming pipeline across 200+ .1pz files may span 10M+ cells.
//   The returned list contains only per-gene and per-factor arrays; per-cell
//   arrays (PCA cell embeddings, NMF H) are returned ONLY when pca_ran / nmf_ran
//   is TRUE and n_cells ≤ in_memory_pca_threshold (2M by default).
//   Callers who want an SCE should pass a single file or a small batch.
//
// Host-copy:
//   All copies are SETUP (file path list, config scalars) or TEARDOWN (reading
//   PipelineResult fields from host-resident std::vector).  No hot-loop copies.
//   PCA/NMF results are Eigen matrices already downloaded by factornet before
//   run_pipeline returns — no additional device→host copy needed here.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/streaming/streamed_pipeline.h>

#include "_r_to_eigen.hpp"
#include "_bind_reduce.hpp"   // eigen_dense_to_r_matrix

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// streaming_pipeline_run_cpp
//
// R-callable binding for the streaming end-to-end pipeline.
//
// Parameters:
//   input_paths    — CharacterVector of .1pz directory or file paths
//   chunk_cols     — int, cells per chunk (memory tunable)
//   run_lognorm    — bool
//   run_hvg        — bool
//   run_pca        — bool (skipped silently if n_cells > 2M)
//   pca_k          — int, number of PCA components
//   run_nmf        — bool
//   nmf_factors    — int, number of NMF factors
//
// Returns: named R list (NOT an SCE):
//   $n_cells            — integer
//   $n_genes            — integer
//   $n_nnz              — double (may exceed int32 range)
//   $size_factors       — NumericVector [n_cells] or NULL
//   $lognorm_target     — double
//   $hvg_indices        — IntegerVector (0-based) or NULL
//   $hvg_scores         — NumericVector or NULL
//   $gene_means         — NumericVector [n_genes] or NULL
//   $gene_vars          — NumericVector [n_genes] or NULL
//   $pca_ran            — logical
//   $pca_V              — NumericMatrix [n_cells × pca_k] or NULL
//   $pca_U              — NumericMatrix [n_genes × pca_k] or NULL (may be empty in streaming)
//   $pca_d              — NumericVector [pca_k] or NULL
//   $nmf_ran            — logical
//   $nmf_H              — NumericMatrix [n_cells × nmf_factors] or NULL
//   $nmf_W              — NumericMatrix [n_genes × nmf_factors] or NULL
//   $nmf_d              — NumericVector [nmf_factors] or NULL
//   $nmf_converged      — logical or NULL
//   $nmf_iterations     — integer or NULL
//   $wall_lognorm_s     — double (wall-clock seconds)
//   $wall_hvg_s         — double
//   $wall_pca_s         — double
//   $wall_nmf_s         — double
//   $n_chunks_processed — integer
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List streaming_pipeline_run_cpp(
        Rcpp::CharacterVector input_paths,
        int                   chunk_cols,
        bool                  run_lognorm,
        bool                  run_hvg,
        bool                  run_pca,
        int                   pca_k,
        bool                  run_nmf,
        int                   nmf_factors)
{
    if (input_paths.size() == 0)
        Rcpp::stop("streaming_pipeline_run_cpp: input_paths is empty");
    if (chunk_cols < 1)
        Rcpp::stop("streaming_pipeline_run_cpp: chunk_cols must be >= 1");

    // Build PipelineConfig
    singlet_gpu::streaming::PipelineConfig cfg;
    cfg.chunk_cols  = chunk_cols;
    cfg.run_lognorm = run_lognorm;
    cfg.run_hvg     = run_hvg;
    cfg.run_pca     = run_pca;
    cfg.run_nmf     = run_nmf;

    cfg.input_paths.reserve(static_cast<std::size_t>(input_paths.size()));
    for (int i = 0; i < input_paths.size(); ++i) {
        cfg.input_paths.push_back(Rcpp::as<std::string>(input_paths[i]));
    }

    if (run_pca) {
        cfg.pca_k            = pca_k;
        cfg.pca_cfg.k_max    = pca_k;
    }
    if (run_nmf) {
        cfg.nmf_cfg.k        = nmf_factors;
    }

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Run pipeline
    singlet_gpu::streaming::PipelineResult res;
    try {
        res = singlet_gpu::streaming::run_pipeline(cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("streaming_pipeline_run_cpp: %s", e.what());
    }
    // All PipelineResult fields are host-resident std::vector or Eigen matrices;
    // no additional device→host copies needed.

    // ---- Size factors ----
    Rcpp::RObject r_sf = R_NilValue;
    if (!res.size_factors.empty()) {
        Rcpp::NumericVector v(res.size_factors.size());
        for (std::size_t i = 0; i < res.size_factors.size(); ++i)
            v[static_cast<R_xlen_t>(i)] = static_cast<double>(res.size_factors[i]);
        r_sf = v;
    }

    // ---- HVG ----
    Rcpp::RObject r_hvg_idx    = R_NilValue;
    Rcpp::RObject r_hvg_scores = R_NilValue;
    Rcpp::RObject r_gene_means = R_NilValue;
    Rcpp::RObject r_gene_vars  = R_NilValue;
    if (run_hvg && !res.hvg_indices.empty()) {
        const int nh = static_cast<int>(res.hvg_indices.size());
        Rcpp::IntegerVector ri(nh);
        Rcpp::NumericVector rs(nh);
        for (int i = 0; i < nh; ++i) {
            ri[i] = res.hvg_indices[i];   // 0-based
            rs[i] = static_cast<double>(res.hvg_scores[i]);
        }
        r_hvg_idx    = ri;
        r_hvg_scores = rs;
    }
    if (!res.gene_means.empty()) {
        const int ng = static_cast<int>(res.gene_means.size());
        Rcpp::NumericVector rm(ng), rv(ng);
        for (int i = 0; i < ng; ++i) {
            rm[i] = static_cast<double>(res.gene_means[i]);
            rv[i] = static_cast<double>(res.gene_vars [i]);
        }
        r_gene_means = rm;
        r_gene_vars  = rv;
    }

    // ---- PCA ----
    Rcpp::RObject r_pca_V = R_NilValue;
    Rcpp::RObject r_pca_U = R_NilValue;
    Rcpp::RObject r_pca_d = R_NilValue;
    if (res.pca_ran) {
        const int k = static_cast<int>(res.pca.k_selected);
        if (k > 0) {
            r_pca_V = eigen_dense_to_r_matrix(res.pca.V);   // cells × k
            r_pca_U = eigen_dense_to_r_matrix(res.pca.U);   // genes × k
            Rcpp::NumericVector rd(k);
            for (int i = 0; i < k; ++i)
                rd[i] = static_cast<double>(res.pca.d(i));
            r_pca_d = rd;
        }
    }

    // ---- NMF ----
    Rcpp::RObject r_nmf_H         = R_NilValue;
    Rcpp::RObject r_nmf_W         = R_NilValue;
    Rcpp::RObject r_nmf_d         = R_NilValue;
    Rcpp::RObject r_nmf_converged = R_NilValue;
    Rcpp::RObject r_nmf_iters     = R_NilValue;
    if (res.nmf_ran) {
        const int k = static_cast<int>(res.nmf.d.size());
        if (k > 0) {
            // H: k × cells → transpose to cells × k for R
            r_nmf_H = eigen_dense_to_r_matrix(res.nmf.H.transpose());
            r_nmf_W = eigen_dense_to_r_matrix(res.nmf.W);   // genes × k
            Rcpp::NumericVector rd(k);
            for (int i = 0; i < k; ++i)
                rd[i] = static_cast<double>(res.nmf.d(i));
            r_nmf_d         = rd;
            r_nmf_converged = Rcpp::wrap(static_cast<bool>(res.nmf.converged));
            r_nmf_iters     = Rcpp::wrap(static_cast<int>(res.nmf.iterations));
        }
    }

    return Rcpp::List::create(
        Rcpp::Named("n_cells")            = static_cast<double>(res.n_cells),
        Rcpp::Named("n_genes")            = static_cast<double>(res.n_genes_total),
        Rcpp::Named("n_nnz")              = static_cast<double>(res.n_nnz_total),
        Rcpp::Named("size_factors")       = r_sf,
        Rcpp::Named("lognorm_target")     = static_cast<double>(res.lognorm_target_used),
        Rcpp::Named("hvg_indices")        = r_hvg_idx,
        Rcpp::Named("hvg_scores")         = r_hvg_scores,
        Rcpp::Named("gene_means")         = r_gene_means,
        Rcpp::Named("gene_vars")          = r_gene_vars,
        Rcpp::Named("pca_ran")            = res.pca_ran,
        Rcpp::Named("pca_V")              = r_pca_V,
        Rcpp::Named("pca_U")              = r_pca_U,
        Rcpp::Named("pca_d")              = r_pca_d,
        Rcpp::Named("nmf_ran")            = res.nmf_ran,
        Rcpp::Named("nmf_H")              = r_nmf_H,
        Rcpp::Named("nmf_W")              = r_nmf_W,
        Rcpp::Named("nmf_d")              = r_nmf_d,
        Rcpp::Named("nmf_converged")      = r_nmf_converged,
        Rcpp::Named("nmf_iterations")     = r_nmf_iters,
        Rcpp::Named("wall_lognorm_s")     = res.wall_lognorm_s,
        Rcpp::Named("wall_hvg_s")         = res.wall_hvg_s,
        Rcpp::Named("wall_pca_s")         = res.wall_pca_s,
        Rcpp::Named("wall_nmf_s")         = res.wall_nmf_s,
        Rcpp::Named("n_chunks_processed") = res.n_chunks_processed
    );
}

}  // namespace singlet_gpu_r
