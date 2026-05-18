// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_ase.hpp
//
// Rcpp bindings for DAESC-style GPU allele-specific expression (ASE) scanning:
//   daesc_cpp — wraps singlet::gpu::ase::run_daesc
//
// Input marshalling decisions:
//   snp_ad:       dgCMatrix [snps × cells]   → DeviceCSC (upload_host_csc_to_device)
//   snp_dp:       dgCMatrix [snps × cells]   → DeviceCSC (upload_host_csc_to_device)
//   cell_type:    IntegerVector/factor or NULL → host int[] or nullptr
//
// Note: snp_ad and snp_dp come from singlet's snp_ad.1pz / snp_dp.1pz.
// The CSC layout has snps as rows (= genes dim in the .1pz schema), cells as cols.
//
// Host-copy overhead:
//   Upload: O(nnz_snp_ad + nnz_snp_dp + n_cells) H→D
//   Download: O(n_snps × 7 fields) D→H (result)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/ase/daesc.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// daesc_cpp
//
// R-callable binding for GPU DAESC-style allele-specific expression scanning.
// Mirrors DAESC R package output: per-SNP beta (allelic log-odds), se, phi,
// p_value, p_adj, converged, n_cells_used.
//
// Parameters:
//   snp_ad      — dgCMatrix [snps × cells] (alt-allele counts from snp_ad.1pz)
//   snp_dp      — dgCMatrix [snps × cells] (total depth from snp_dp.1pz)
//   cell_type   — IntegerVector / factor [n_cells] or NULL (for stratified ASE)
//   chunk_snps  — int, SNPs per device chunk (0 = use default 10000)
//   run_bh_fdr  — bool, apply BH FDR correction
//   fdr_alpha   — double, BH FDR threshold
//   min_depth   — int, minimum depth to include a cell for a SNP
//   min_cells   — int, minimum qualifying cells per SNP
//   seed        — int (not used; kept for API symmetry with other wrappers)
//   resource    — string "auto"/"gpu"/"cpu"
//
// Returns:
//   Rcpp::List{
//     "beta"      — NumericVector [n_snps]
//     "se"        — NumericVector [n_snps]
//     "phi"       — NumericVector [n_snps] (beta-binomial overdispersion)
//     "p_value"   — NumericVector [n_snps]
//     "p_adj"     — NumericVector [n_snps] (NaN if !run_bh_fdr)
//     "converged" — LogicalVector [n_snps]
//     "n_cells"   — IntegerVector [n_snps] (cells used per SNP)
//     "strat_beta"— NumericMatrix [n_snps × n_types] (empty if no cell_type)
//     "strat_se"  — NumericMatrix [n_snps × n_types]
//     "strat_phi" — NumericMatrix [n_snps × n_types]
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List daesc_cpp(
        Rcpp::S4      snp_ad,
        Rcpp::S4      snp_dp,
        Rcpp::RObject cell_type,
        int           chunk_snps,
        bool          run_bh_fdr,
        double        fdr_alpha,
        int           min_depth,
        int           min_cells,
        int           seed,
        std::string   resource)
{
    // ── Marshal sparse inputs ─────────────────────────────────────────────────
    HostCsc h_ad, h_dp;
    try { h_ad = r_sparse_to_host_csc(snp_ad); }
    catch (const std::exception& e) { Rcpp::stop("daesc_cpp snp_ad: %s", e.what()); }
    try { h_dp = r_sparse_to_host_csc(snp_dp); }
    catch (const std::exception& e) { Rcpp::stop("daesc_cpp snp_dp: %s", e.what()); }

    const int n_snps  = h_ad.n_rows;
    const int n_cells = h_ad.n_cols;

    if (h_dp.n_rows != n_snps || h_dp.n_cols != n_cells)
        Rcpp::stop("daesc_cpp: snp_ad and snp_dp must have identical dimensions.");

    // ── Cell type labels (optional) ───────────────────────────────────────────
    std::vector<int> h_cell_type;
    int n_types = 0;
    if (!Rf_isNull(cell_type)) {
        std::vector<std::string> lvls;
        h_cell_type = encode_labels_0based(cell_type, n_cells, lvls);
        n_types     = static_cast<int>(lvls.size());
    }

    // ── GPUContext + second stream ────────────────────────────────────────────
    // GPUContext() creates all handles (cuBLAS/cuSPARSE/cuSOLVER) and owns a
    // primary stream (ctx.stream).  We use ctx.stream as stream1 for compute and
    // a separately-created stream2 for prefetch overlap.
    singlet::gpu::core::GPUContext ctx;
    cudaStream_t stream1 = ctx.stream;  // alias; ctx owns lifetime
    cudaStream_t stream2 = nullptr;
    cudaStreamCreate(&stream2);
    struct Stream2Guard {
        cudaStream_t s;
        ~Stream2Guard() { if (s) cudaStreamDestroy(s); }
    } sg2{stream2};

    // ── Upload matrices ───────────────────────────────────────────────────────
    singlet::gpu::core::DeviceCSC d_ad, d_dp;
    try { d_ad = upload_host_csc_to_device(h_ad, stream1); }
    catch (const std::exception& e) { Rcpp::stop("daesc_cpp upload snp_ad: %s", e.what()); }
    try { d_dp = upload_host_csc_to_device(h_dp, stream1); }
    catch (const std::exception& e) { Rcpp::stop("daesc_cpp upload snp_dp: %s", e.what()); }

    // ── Configure ─────────────────────────────────────────────────────────────
    singlet::gpu::ase::DaescConfig cfg;
    cfg.chunk_snps  = chunk_snps > 0 ? chunk_snps : singlet::gpu::ase::DEFAULT_CHUNK_SNPS;
    cfg.run_bh_fdr  = run_bh_fdr;
    cfg.fdr_alpha   = static_cast<float>(fdr_alpha);
    cfg.min_depth   = min_depth;
    cfg.min_cells   = min_cells;
    cfg.n_types     = n_types;

    // ── Run kernel ─────────────────────────────────────────────────────────────
    singlet::gpu::ase::DaescResult res;
    try {
        res = singlet::gpu::ase::run_daesc(
            d_ad, d_dp,
            h_cell_type.empty() ? nullptr : h_cell_type.data(),
            cfg, ctx, stream2);
    } catch (const std::exception& e) {
        Rcpp::stop("daesc_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ── Marshal per-SNP result vectors ────────────────────────────────────────
    const int ns = res.n_snps;
    Rcpp::NumericVector r_beta    (ns);
    Rcpp::NumericVector r_se      (ns);
    Rcpp::NumericVector r_phi     (ns);
    Rcpp::NumericVector r_pvalue  (ns);
    Rcpp::NumericVector r_padj    (ns);
    Rcpp::LogicalVector r_converged(ns);
    Rcpp::IntegerVector r_ncells  (ns);

    for (int i = 0; i < ns; ++i) {
        r_beta    [i] = static_cast<double>(res.beta     [i]);
        r_se      [i] = static_cast<double>(res.se       [i]);
        r_phi     [i] = static_cast<double>(res.phi      [i]);
        r_pvalue  [i] = static_cast<double>(res.p_value  [i]);
        r_padj    [i] = static_cast<double>(res.p_adj    [i]);
        r_converged[i] = (res.converged[i] != 0);
        r_ncells  [i] = res.n_cells[i];
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("beta")      = r_beta,
        Rcpp::Named("se")        = r_se,
        Rcpp::Named("phi")       = r_phi,
        Rcpp::Named("p_value")   = r_pvalue,
        Rcpp::Named("p_adj")     = r_padj,
        Rcpp::Named("converged") = r_converged,
        Rcpp::Named("n_cells")   = r_ncells
    );

    // ── Optional stratified results ───────────────────────────────────────────
    if (n_types > 0 && !res.strat_beta.empty()) {
        const std::size_t total_st = static_cast<std::size_t>(ns) * n_types;
        Rcpp::NumericMatrix r_st_beta(ns, n_types);
        Rcpp::NumericMatrix r_st_se  (ns, n_types);
        Rcpp::NumericMatrix r_st_phi (ns, n_types);

        for (int i = 0; i < ns; ++i) {
            for (int t = 0; t < n_types; ++t) {
                const std::size_t off = static_cast<std::size_t>(i) * n_types + t;
                r_st_beta(i, t) = static_cast<double>(res.strat_beta[off]);
                r_st_se  (i, t) = static_cast<double>(res.strat_se  [off]);
                r_st_phi (i, t) = static_cast<double>(res.strat_phi [off]);
            }
        }
        out["strat_beta"] = r_st_beta;
        out["strat_se"]   = r_st_se;
        out["strat_phi"]  = r_st_phi;
    } else {
        out["strat_beta"] = Rcpp::NumericMatrix(0, 0);
        out["strat_se"]   = Rcpp::NumericMatrix(0, 0);
        out["strat_phi"]  = Rcpp::NumericMatrix(0, 0);
    }

    return out;
}

}  // namespace singlet_gpu_r
