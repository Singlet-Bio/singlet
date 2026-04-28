// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_eqtl.hpp
//
// Rcpp bindings for NEBULA-style GPU single-cell eQTL mapping:
//   nebula_cpp — wraps singlet_gpu::eqtl::run_nebula
//
// Input marshalling decisions:
//   counts:         dgCMatrix [genes × cells]  → DeviceCSC (upload_host_csc_to_device)
//   snp_ad:         dgCMatrix [snps × cells]   → DeviceCSC (upload_host_csc_to_device)
//   snp_dp:         dgCMatrix [snps × cells]   → DeviceCSC (upload_host_csc_to_device)
//   donor_id:       IntegerVector/factor [n_cells] → host int[] passed directly
//   covariates:     NumericMatrix [n_donors × n_cov] or NULL → float[] or nullptr
//
// WHY multi-sparse-matrix design: eQTL maps expression (counts CSC) vs genotype
//   (snp_ad/snp_dp CSC).  All three come from singlify outputs.  Uploading them
//   separately matches how they are loaded from different .1pz files.
//
// Output: the R side receives results aggregated in-memory.  For production-scale
//   (1M SNPs × 20k genes), users should implement the callback path by writing a
//   custom R loop — this binding uses the in-memory path for small/medium scale.
//   A warning is emitted when n_snps × n_genes > 1e8.
//
// Host-copy overhead:
//   Upload: O(nnz_counts + nnz_snp_ad + nnz_snp_dp + n_cells + n_donors×n_cov) H→D
//   Download: O(n_snps × n_genes × 6 fields) D→H (result)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/eqtl/nebula.h>

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
// nebula_cpp
//
// R-callable binding for GPU NEBULA-style single-cell eQTL mapping.
// Mirrors nebula::nebula() output shape:
//   a data.frame of (snp_idx, gene_idx, beta, se, sigma_sq, p_value, p_adj, converged)
//
// Parameters:
//   counts      — dgCMatrix [genes × cells] (UMI counts)
//   snp_ad      — dgCMatrix [snps × cells]  (alt-allele counts from snp_ad.1pz)
//   snp_dp      — dgCMatrix [snps × cells]  (total depth from snp_dp.1pz)
//   donor_id    — IntegerVector or factor [n_cells], 0-based donor labels
//   covariates  — NumericMatrix [n_donors × n_cov] or NULL
//   chunk_snps  — int, SNPs per device chunk
//   chunk_genes — int, genes per device chunk
//   run_bh_fdr  — bool, apply BH FDR correction per gene
//   fdr_alpha   — double, BH FDR threshold
//   seed        — int (no-op; algorithm is deterministic, kept for API symmetry)
//   resource    — string "auto"/"gpu"/"cpu"
//
// Returns:
//   Rcpp::List{
//     "beta"      — NumericVector [n_snps × n_genes] (eQTL effect size)
//     "se"        — NumericVector [n_snps × n_genes] (standard error)
//     "sigma_sq"  — NumericVector [n_snps × n_genes] (random-effect variance)
//     "p_value"   — NumericVector [n_snps × n_genes] (uncorrected Wald p)
//     "p_adj"     — NumericVector [n_snps × n_genes] (BH q, NaN if !run_bh_fdr)
//     "converged" — LogicalVector [n_snps × n_genes]
//     "n_snps"    — int
//     "n_genes"   — int
//   }
//   All flat vectors are in [snp × gene] row-major order.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List nebula_cpp(
        Rcpp::S4            counts,
        Rcpp::S4            snp_ad,
        Rcpp::S4            snp_dp,
        Rcpp::RObject       donor_id,
        Rcpp::RObject       covariates,
        int                 chunk_snps,
        int                 chunk_genes,
        bool                run_bh_fdr,
        double              fdr_alpha,
        int                 seed,
        std::string         resource)
{
    // ── Marshal sparse inputs ─────────────────────────────────────────────────
    HostCsc h_counts, h_snp_ad, h_snp_dp;
    try { h_counts = r_sparse_to_host_csc(counts); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp counts: %s", e.what()); }
    try { h_snp_ad = r_sparse_to_host_csc(snp_ad); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp snp_ad: %s", e.what()); }
    try { h_snp_dp = r_sparse_to_host_csc(snp_dp); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp snp_dp: %s", e.what()); }

    const int n_cells = h_counts.n_cols;
    const int n_genes = h_counts.n_rows;
    const int n_snps  = h_snp_ad.n_rows;  // snp_ad is [snps × cells]; CSC cols = cells

    if (h_snp_ad.n_cols != n_cells || h_snp_dp.n_cols != n_cells)
        Rcpp::stop("nebula_cpp: counts, snp_ad, snp_dp must have the same number of cells.");
    if (h_snp_dp.n_rows != n_snps)
        Rcpp::stop("nebula_cpp: snp_ad and snp_dp must have the same number of SNPs.");

    // ── Encode donor IDs → 0-based host int[] ────────────────────────────────
    std::vector<int> h_donor_id(n_cells);
    int n_donors = 0;
    {
        std::vector<std::string> lvls;
        std::vector<int> raw = encode_labels_0based(donor_id, n_cells, lvls);
        h_donor_id = raw;
        n_donors   = static_cast<int>(lvls.size());
    }

    if (n_donors > singlet_gpu::eqtl::MAX_DONORS_SHMEM)
        Rcpp::stop("nebula_cpp: n_donors (%d) exceeds MAX_DONORS_SHMEM (%d). "
                   "Downsample donors.", n_donors, singlet_gpu::eqtl::MAX_DONORS_SHMEM);

    // ── Encode covariates → flat float[] or nullptr ───────────────────────────
    std::vector<float> h_cov;
    int n_cov = 0;
    if (!Rf_isNull(covariates)) {
        Rcpp::NumericMatrix cov_mat(covariates);
        if (cov_mat.nrow() != n_donors)
            Rcpp::stop("nebula_cpp: covariates nrow (%d) != n_donors (%d).",
                       cov_mat.nrow(), n_donors);
        n_cov = cov_mat.ncol();
        if (n_cov > singlet_gpu::eqtl::MAX_COV)
            Rcpp::stop("nebula_cpp: n_cov (%d) exceeds MAX_COV (%d).",
                       n_cov, singlet_gpu::eqtl::MAX_COV);
        h_cov.resize(static_cast<std::size_t>(n_donors) * n_cov);
        for (int d = 0; d < n_donors; ++d)
            for (int c = 0; c < n_cov; ++c)
                h_cov[static_cast<std::size_t>(d) * n_cov + c] =
                    static_cast<float>(cov_mat(d, c));
    }

    // Scale warning for large result sets
    if (static_cast<int64_t>(n_snps) * n_genes > 100000000LL)
        Rcpp::warning("nebula_cpp: n_snps (%d) × n_genes (%d) = %lld entries. "
                      "In-memory result requires ~%.1f GB. "
                      "Consider using callback path for production scale.",
                      n_snps, n_genes,
                      static_cast<long long>(n_snps) * n_genes,
                      static_cast<double>(n_snps) * n_genes * 6 * sizeof(float) / 1e9);

    // ── GPUContext + second stream ────────────────────────────────────────────
    // GPUContext() creates all handles (cuBLAS/cuSPARSE/cuSOLVER) and owns a
    // primary stream (ctx.stream).  We use ctx.stream as stream1 for compute and
    // a separately-created stream2 for prefetch overlap.
    singlet_gpu::core::GPUContext ctx;
    cudaStream_t stream1 = ctx.stream;  // alias; ctx owns lifetime
    cudaStream_t stream2 = nullptr;
    cudaStreamCreate(&stream2);
    struct Stream2Guard {
        cudaStream_t s;
        ~Stream2Guard() { if (s) cudaStreamDestroy(s); }
    } sg2{stream2};

    // ── Upload sparse matrices ────────────────────────────────────────────────
    singlet_gpu::core::DeviceCSC d_counts, d_snp_ad, d_snp_dp;
    try { d_counts = upload_host_csc_to_device(h_counts, stream1); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp upload counts: %s", e.what()); }
    try { d_snp_ad = upload_host_csc_to_device(h_snp_ad, stream1); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp upload snp_ad: %s", e.what()); }
    try { d_snp_dp = upload_host_csc_to_device(h_snp_dp, stream1); }
    catch (const std::exception& e) { Rcpp::stop("nebula_cpp upload snp_dp: %s", e.what()); }

    // ── Configure ─────────────────────────────────────────────────────────────
    singlet_gpu::eqtl::NebulaConfig cfg;
    cfg.chunk_snps  = chunk_snps  > 0 ? chunk_snps  : singlet_gpu::eqtl::DEFAULT_CHUNK_SNPS;
    cfg.chunk_genes = chunk_genes > 0 ? chunk_genes : singlet_gpu::eqtl::DEFAULT_CHUNK_GENES;
    cfg.run_bh_fdr  = run_bh_fdr;
    cfg.fdr_alpha   = static_cast<float>(fdr_alpha);

    // ── Run kernel ─────────────────────────────────────────────────────────────
    singlet_gpu::eqtl::NebulaResult res;
    try {
        res = singlet_gpu::eqtl::run_nebula(
            d_counts, d_snp_ad, d_snp_dp,
            h_donor_id.data(), n_donors,
            h_cov.empty() ? nullptr : h_cov.data(), n_cov,
            cfg, ctx, stream2, nullptr);
    } catch (const std::exception& e) {
        Rcpp::stop("nebula_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ── Marshal result ────────────────────────────────────────────────────────
    const std::size_t total = static_cast<std::size_t>(res.n_snps) * res.n_genes;

    Rcpp::NumericVector r_beta    (total);
    Rcpp::NumericVector r_se      (total);
    Rcpp::NumericVector r_sigma_sq(total);
    Rcpp::NumericVector r_pvalue  (total);
    Rcpp::NumericVector r_padj    (total);
    Rcpp::LogicalVector r_converged(total);

    for (std::size_t k = 0; k < total; ++k) {
        r_beta    [k] = static_cast<double>(res.beta    [k]);
        r_se      [k] = static_cast<double>(res.se      [k]);
        r_sigma_sq[k] = static_cast<double>(res.sigma_sq[k]);
        r_pvalue  [k] = static_cast<double>(res.p_value [k]);
        r_padj    [k] = static_cast<double>(res.p_adj   [k]);
        r_converged[k] = (res.converged[k] != 0);
    }

    return Rcpp::List::create(
        Rcpp::Named("beta")      = r_beta,
        Rcpp::Named("se")        = r_se,
        Rcpp::Named("sigma_sq")  = r_sigma_sq,
        Rcpp::Named("p_value")   = r_pvalue,
        Rcpp::Named("p_adj")     = r_padj,
        Rcpp::Named("converged") = r_converged,
        Rcpp::Named("n_snps")    = res.n_snps,
        Rcpp::Named("n_genes")   = res.n_genes
    );
}

}  // namespace singlet_gpu_r
