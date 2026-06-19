// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_disease.hpp
//
// Rcpp binding for disease relevance scoring kernel (cycle 48):
//   scdrs_cpp — wraps singlet::gpu::disease::run_scdrs
//
// Input format:
//   expression        — dgCMatrix [genes × cells], log-normalized GEX
//   gene_set_names    — CharacterVector, disease names (length n_diseases)
//   gene_set_indices  — IntegerVector, gene indices (0-based); concatenated across diseases
//   gene_set_weights  — NumericVector, gene weights; same length as gene_set_indices
//   gene_set_offsets  — IntegerVector [n_diseases + 1], CSR-style offsets into gene_set_indices
//   n_ctrl_sets       — int, control gene sets per disease (default 1000)
//   seed              — int
//   resource          — "gpu" | "cpu" | "auto"
//
// Host-copy overhead:
//   Input: O(nnz) host→device (expression) + O(n_genes_total) H→D (gene set tables).
//   Output: O(n_cells × n_diseases × 4) device→host (raw_score, norm_score, p_value, fdr).
//
// GPU-stub fallback:
//   Returns list(error="GPU not available") when FACTORNET_HAS_GPU not defined.

#pragma once

#include <Rcpp.h>
#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FACTORNET_HAS_GPU

namespace singlet_gpu_r {

// [[Rcpp::export]]
inline Rcpp::List scdrs_cpp(
        Rcpp::S4            /*expression*/,
        Rcpp::CharacterVector /*gene_set_names*/,
        Rcpp::IntegerVector /*gene_set_indices*/,
        Rcpp::NumericVector /*gene_set_weights*/,
        Rcpp::IntegerVector /*gene_set_offsets*/,
        int    /*n_ctrl_sets*/,
        int    /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

}  // namespace singlet_gpu_r

#else  // FACTORNET_HAS_GPU defined

#define FACTORNET_HAS_GPU 1
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/disease/scdrs.h>

#include <cuda_runtime.h>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// scdrs_cpp
//
// GPU per-cell disease relevance scoring from GWAS gene sets.
// Wraps singlet::gpu::disease::run_scdrs.
//
// Returns: Rcpp::List{
//   "raw_score"   — NumericMatrix [n_cells × n_diseases]
//   "norm_score"  — NumericMatrix [n_cells × n_diseases]
//   "p_value"     — NumericMatrix [n_cells × n_diseases]
//   "fdr"         — NumericMatrix [n_cells × n_diseases]
//   "disease_names" — CharacterVector [n_diseases]
//   "n_diseases"  — int
// }
// [[Rcpp::export]]
inline Rcpp::List scdrs_cpp(
        Rcpp::S4              expression,
        Rcpp::CharacterVector gene_set_names,
        Rcpp::IntegerVector   gene_set_indices,
        Rcpp::NumericVector   gene_set_weights,
        Rcpp::IntegerVector   gene_set_offsets,
        int         n_ctrl_sets,
        int         seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("scdrs_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;
    const int n_diseases = static_cast<int>(gene_set_names.size());

    if ((int)gene_set_offsets.size() != n_diseases + 1)
        Rcpp::stop("scdrs_cpp: gene_set_offsets length (%d) != n_diseases+1 (%d).",
                   (int)gene_set_offsets.size(), n_diseases + 1);
    if (gene_set_indices.size() != gene_set_weights.size())
        Rcpp::stop("scdrs_cpp: gene_set_indices and gene_set_weights length mismatch.");

    const int n_gs_total = static_cast<int>(gene_set_indices.size());

    // Validate gene indices are in range
    for (int i = 0; i < n_gs_total; ++i)
        if (gene_set_indices[i] < 0 || gene_set_indices[i] >= n_genes)
            Rcpp::stop("scdrs_cpp: gene_set_indices[%d]=%d out of range [0, %d).",
                       i, gene_set_indices[i], n_genes);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("scdrs_cpp upload: %s", e.what()); }

    // Build host gene-set tables
    std::vector<int>   h_gs_idx(n_gs_total);
    std::vector<float> h_gs_wt (n_gs_total);
    std::vector<int>   h_gs_off(n_diseases + 1);

    for (int i = 0; i < n_gs_total; ++i) {
        h_gs_idx[i] = gene_set_indices[i];
        h_gs_wt [i] = static_cast<float>(gene_set_weights[i]);
    }
    for (int d = 0; d <= n_diseases; ++d) h_gs_off[d] = gene_set_offsets[d];

    singlet::gpu::disease::ScDRSConfig cfg;
    cfg.n_diseases    = n_diseases;
    cfg.n_ctrl_sets   = n_ctrl_sets;
    cfg.seed          = static_cast<uint64_t>(seed);
    cfg.use_gpu       = (resource != "cpu");

    singlet::gpu::disease::ScDRSResult res;
    try {
        res = singlet::gpu::disease::run_scdrs(
            d_expr, h_gs_idx, h_gs_wt, h_gs_off,
            n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("scdrs_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const std::size_t mat_sz = static_cast<std::size_t>(n_cells) * n_diseases;

    std::vector<float> h_raw(mat_sz), h_norm(mat_sz), h_pv(mat_sz), h_fdr(mat_sz);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpy(h_raw .data(), res.raw_score.data(),  mat_sz * sizeof(float), cudaMemcpyDeviceToHost), "raw_score");
            chk(cudaMemcpy(h_norm.data(), res.norm_score.data(), mat_sz * sizeof(float), cudaMemcpyDeviceToHost), "norm_score");
            chk(cudaMemcpy(h_pv  .data(), res.p_value.data(),    mat_sz * sizeof(float), cudaMemcpyDeviceToHost), "p_value");
            chk(cudaMemcpy(h_fdr .data(), res.fdr.data(),        mat_sz * sizeof(float), cudaMemcpyDeviceToHost), "fdr");
        } catch (const std::exception& e) {
            Rcpp::stop("scdrs_cpp copyback: %s", e.what());
        }
    }

    Rcpp::NumericMatrix r_raw (n_cells, n_diseases);
    Rcpp::NumericMatrix r_norm(n_cells, n_diseases);
    Rcpp::NumericMatrix r_pv  (n_cells, n_diseases);
    Rcpp::NumericMatrix r_fdr (n_cells, n_diseases);

    for (int c = 0; c < n_cells; ++c)
        for (int d = 0; d < n_diseases; ++d) {
            const std::size_t idx = static_cast<std::size_t>(c) * n_diseases + d;
            r_raw (c, d) = static_cast<double>(h_raw [idx]);
            r_norm(c, d) = static_cast<double>(h_norm[idx]);
            r_pv  (c, d) = static_cast<double>(h_pv  [idx]);
            r_fdr (c, d) = static_cast<double>(h_fdr [idx]);
        }

    // Attach disease names as column names
    Rcpp::CharacterVector r_dnames(n_diseases);
    for (int d = 0; d < n_diseases; ++d)
        r_dnames[d] = Rcpp::as<std::string>(gene_set_names[d]);

    return Rcpp::List::create(
        Rcpp::Named("raw_score")     = r_raw,
        Rcpp::Named("norm_score")    = r_norm,
        Rcpp::Named("p_value")       = r_pv,
        Rcpp::Named("fdr")           = r_fdr,
        Rcpp::Named("disease_names") = r_dnames,
        Rcpp::Named("n_diseases")    = n_diseases
    );
}

}  // namespace singlet_gpu_r

#endif  // FACTORNET_HAS_GPU
