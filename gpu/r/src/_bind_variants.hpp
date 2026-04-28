// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_variants.hpp
//
// Rcpp bindings for Monopogen-style GPU somatic variant calling:
//   call_variants_cpp — wraps singlet_gpu::variants::call_variants
//
// Input marshalling decisions:
//   snp_ad:      dgCMatrix [snps × cells] → DeviceCSC (upload_host_csc_to_device)
//   snp_dp:      dgCMatrix [snps × cells] → DeviceCSC (upload_host_csc_to_device)
//   chromosome:  IntegerVector [n_snps], 0-based chromosome index
//   ld_panels:   R list of per-chromosome LD panels, each a named list{
//                  n_snps, indptr, col_idx, val}
//                (1-based R indices → 0-based in LdPanelCsr)
//
// WHY ld_panels as list of lists: LD panels are per-chromosome and together may
//   exceed memory.  Passing them as a list keeps each panel in R's GC-managed heap
//   until it is needed.  The kernel consumes them one chromosome at a time via
//   run_chromosome_tile(), which is the natural match.
//
// Host-copy overhead:
//   Upload: O(nnz_snp_ad + nnz_snp_dp) H→D (one-time)
//   Per-chromosome: O(nnz_ld_panel) H→D (inside C++ tile loop)
//   Download: O(n_snps × 3 + n_somatic_candidates) D→H (result)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/variants/monopogen.h>

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
// r_list_to_ld_panel
//
// Convert a named R list {n_snps, indptr, col_idx, val} to a LdPanelCsr.
// indptr and col_idx are 1-based R vectors → converted to 0-based here.
// ---------------------------------------------------------------------------
inline singlet_gpu::variants::LdPanelCsr
r_list_to_ld_panel(const Rcpp::List& panel, int expected_n_snps, int chr_idx)
{
    for (const char* slot : {"n_snps", "indptr", "col_idx", "val"}) {
        if (!panel.containsElementNamed(slot))
            throw std::runtime_error(std::string("ld_panels[[") + std::to_string(chr_idx + 1) +
                                     "]]: missing slot '" + slot + "'");
    }

    singlet_gpu::variants::LdPanelCsr lp;
    lp.n_snps = Rcpp::as<int>(panel["n_snps"]);

    if (lp.n_snps != expected_n_snps)
        throw std::runtime_error("ld_panels[[" + std::to_string(chr_idx + 1) +
                                 "]]: n_snps (" + std::to_string(lp.n_snps) +
                                 ") != expected (" + std::to_string(expected_n_snps) + ")");

    Rcpp::IntegerVector rp  = panel["indptr"];
    Rcpp::IntegerVector ci  = panel["col_idx"];
    Rcpp::NumericVector val = panel["val"];

    const int nnz = static_cast<int>(ci.size());
    lp.indptr .resize(static_cast<std::size_t>(lp.n_snps + 1));
    lp.col_idx.resize(static_cast<std::size_t>(nnz));
    lp.val    .resize(static_cast<std::size_t>(nnz));

    // 1-based R → 0-based C++
    for (int i = 0; i <= lp.n_snps; ++i) lp.indptr [i] = rp[i] - 1;
    for (int k = 0; k < nnz;        ++k) {
        lp.col_idx[k] = ci [k] - 1;
        lp.val    [k] = static_cast<float>(val[k]);
    }

    return lp;
}

// ---------------------------------------------------------------------------
// call_variants_cpp
//
// R-callable binding for GPU Monopogen-style somatic variant calling.
// Mirrors Monopogen Python output: per-SNP germline genotypes, quality scores,
// and a sparse list of somatic candidate SNPs with LRT stats.
//
// Parameters:
//   snp_ad      — dgCMatrix [snps × cells] (alt-allele counts from snp_ad.1pz)
//   snp_dp      — dgCMatrix [snps × cells] (total depth from snp_dp.1pz)
//   chromosome  — IntegerVector [n_snps], 0-based chromosome index
//   ld_panels   — R list of per-chromosome LD panels (named list{n_snps,indptr,col_idx,val})
//   somatic_af_min  — double, minimum AF for somatic candidate (default 0.05)
//   somatic_af_max  — double, maximum AF for somatic candidate (default 0.35)
//   fdr_alpha       — double, BH FDR threshold for somatic calls
//   run_bh_fdr      — bool
//   min_total_dp    — int, minimum total depth across cells
//   seed            — int (not used; deterministic algorithm; kept for API symmetry)
//   resource        — string "auto"/"gpu"/"cpu"
//
// Returns:
//   Rcpp::List{
//     "germline_genotypes" — NumericMatrix [n_snps × 3] P(geno | data)
//     "germline_calls"     — IntegerVector [n_snps] (0=0/0, 1=0/1, 2=1/1, -1=filtered)
//     "qual_scores"        — NumericVector [n_snps] (Phred-scaled posterior quality)
//     "somatic_candidates" — data.frame(snp_idx, alt_freq, ld_score, lrt_stat, p_value, q_value)
//     "n_somatic"          — int
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List call_variants_cpp(
        Rcpp::S4            snp_ad,
        Rcpp::S4            snp_dp,
        Rcpp::IntegerVector chromosome,
        Rcpp::List          ld_panels,
        double              somatic_af_min,
        double              somatic_af_max,
        double              fdr_alpha,
        bool                run_bh_fdr,
        int                 min_total_dp,
        int                 seed,
        std::string         resource)
{
    // ── Marshal sparse inputs ─────────────────────────────────────────────────
    HostCsc h_ad, h_dp;
    try { h_ad = r_sparse_to_host_csc(snp_ad); }
    catch (const std::exception& e) { Rcpp::stop("call_variants_cpp snp_ad: %s", e.what()); }
    try { h_dp = r_sparse_to_host_csc(snp_dp); }
    catch (const std::exception& e) { Rcpp::stop("call_variants_cpp snp_dp: %s", e.what()); }

    const int n_snps  = h_ad.n_rows;
    const int n_cells = h_ad.n_cols;

    if (h_dp.n_rows != n_snps || h_dp.n_cols != n_cells)
        Rcpp::stop("call_variants_cpp: snp_ad and snp_dp must have identical dimensions.");
    if ((int)chromosome.size() != n_snps)
        Rcpp::stop("call_variants_cpp: chromosome length (%d) != n_snps (%d).",
                   (int)chromosome.size(), n_snps);

    // ── Build chromosome vector (host std::vector<int>) ───────────────────────
    std::vector<int> h_chr(static_cast<std::size_t>(n_snps));
    for (int i = 0; i < n_snps; ++i) h_chr[i] = chromosome[i];

    // Determine per-chromosome SNP counts from sorted chromosome array.
    // call_variants requires contiguous chromosomes — validate this.
    int max_chr = *std::max_element(h_chr.begin(), h_chr.end());
    if (max_chr >= singlet_gpu::variants::MAX_CHROMOSOMES)
        Rcpp::stop("call_variants_cpp: chromosome index %d exceeds MAX_CHROMOSOMES (%d).",
                   max_chr, singlet_gpu::variants::MAX_CHROMOSOMES);
    if ((int)ld_panels.size() <= max_chr)
        Rcpp::stop("call_variants_cpp: ld_panels has %d elements but max chr index is %d.",
                   (int)ld_panels.size(), max_chr);

    // Build per-chromosome expected SNP counts for LD panel validation
    std::vector<int> chr_start(singlet_gpu::variants::MAX_CHROMOSOMES, n_snps);
    std::vector<int> chr_end  (singlet_gpu::variants::MAX_CHROMOSOMES, 0);
    for (int s = 0; s < n_snps; ++s) {
        int c = h_chr[s];
        chr_start[c] = std::min(chr_start[c], s);
        chr_end  [c] = std::max(chr_end[c],   s + 1);
    }

    // Build LD panel vector (indexed 0..max_chr)
    std::vector<singlet_gpu::variants::LdPanelCsr> ld_vec;
    ld_vec.resize(static_cast<std::size_t>(max_chr + 1));
    for (int c = 0; c <= max_chr; ++c) {
        if (chr_start[c] >= chr_end[c]) {
            // No SNPs on chromosome c — build an empty panel
            ld_vec[c].n_snps = 0;
            continue;
        }
        const int expected = chr_end[c] - chr_start[c];
        try {
            ld_vec[c] = r_list_to_ld_panel(
                Rcpp::as<Rcpp::List>(ld_panels[c]), expected, c);
        } catch (const std::exception& e) {
            Rcpp::stop("call_variants_cpp ld_panels[[%d]]: %s", c + 1, e.what());
        }
    }

    // ── GPUContext (owns cuBLAS + cuSPARSE + cuSOLVER handles + stream) ──────
    // GPUContext() creates all handles and a dedicated stream internally, then
    // binds all handles to that stream.  We use ctx.stream and ctx.cusparse.
    singlet_gpu::core::GPUContext ctx;
    cudaStream_t     stream    = ctx.stream;   // alias; ctx owns lifetime
    cusparseHandle_t sp_handle = ctx.cusparse; // alias; ctx owns lifetime

    // ── Upload matrices ───────────────────────────────────────────────────────
    singlet_gpu::core::DeviceCSC d_ad, d_dp;
    try { d_ad = upload_host_csc_to_device(h_ad, stream); }
    catch (const std::exception& e) { Rcpp::stop("call_variants_cpp upload snp_ad: %s", e.what()); }
    try { d_dp = upload_host_csc_to_device(h_dp, stream); }
    catch (const std::exception& e) { Rcpp::stop("call_variants_cpp upload snp_dp: %s", e.what()); }

    // ── Configure ─────────────────────────────────────────────────────────────
    singlet_gpu::variants::MonopogenConfig cfg;
    cfg.somatic_af_min = static_cast<float>(somatic_af_min);
    cfg.somatic_af_max = static_cast<float>(somatic_af_max);
    cfg.fdr_alpha      = static_cast<float>(fdr_alpha);
    cfg.run_bh_fdr     = run_bh_fdr;
    cfg.min_total_dp   = min_total_dp;
    cfg.deterministic  = true;

    // ── Run kernel ─────────────────────────────────────────────────────────────
    singlet_gpu::variants::MonopogenResult res;
    try {
        res = singlet_gpu::variants::call_variants(
            d_ad, d_dp, h_chr, ld_vec, cfg, stream, sp_handle);
    } catch (const std::exception& e) {
        Rcpp::stop("call_variants_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // ── Marshal germline results ──────────────────────────────────────────────
    Rcpp::NumericMatrix r_geno(n_snps, 3);
    Rcpp::IntegerVector r_calls(n_snps);
    Rcpp::NumericVector r_qual (n_snps);

    for (int i = 0; i < n_snps; ++i) {
        r_geno(i, 0) = static_cast<double>(res.germline_genotypes[i * 3    ]);
        r_geno(i, 1) = static_cast<double>(res.germline_genotypes[i * 3 + 1]);
        r_geno(i, 2) = static_cast<double>(res.germline_genotypes[i * 3 + 2]);
        r_calls[i]   = static_cast<int>(res.germline_calls[i]);
        r_qual [i]   = static_cast<double>(res.qual_scores[i]);
    }
    Rcpp::colnames(r_geno) = Rcpp::CharacterVector::create("P_hom_ref", "P_het", "P_hom_alt");

    // ── Marshal somatic candidates ────────────────────────────────────────────
    const int n_som = static_cast<int>(res.somatic_candidates.size());
    Rcpp::IntegerVector r_som_idx  (n_som);
    Rcpp::NumericVector r_som_af   (n_som);
    Rcpp::NumericVector r_som_ld   (n_som);
    Rcpp::NumericVector r_som_lrt  (n_som);
    Rcpp::NumericVector r_som_pval (n_som);
    Rcpp::NumericVector r_som_qval (n_som);

    for (int k = 0; k < n_som; ++k) {
        const auto& sc = res.somatic_candidates[k];
        r_som_idx [k] = sc.snp_idx + 1;  // 0-based → 1-based for R
        r_som_af  [k] = static_cast<double>(sc.alt_freq);
        r_som_ld  [k] = static_cast<double>(sc.ld_score);
        r_som_lrt [k] = static_cast<double>(sc.lrt_stat);
        r_som_pval[k] = static_cast<double>(sc.p_value);
        r_som_qval[k] = static_cast<double>(sc.q_value);
    }

    return Rcpp::List::create(
        Rcpp::Named("germline_genotypes") = r_geno,
        Rcpp::Named("germline_calls")     = r_calls,
        Rcpp::Named("qual_scores")        = r_qual,
        Rcpp::Named("somatic_candidates") = Rcpp::DataFrame::create(
            Rcpp::Named("snp_idx")   = r_som_idx,
            Rcpp::Named("alt_freq")  = r_som_af,
            Rcpp::Named("ld_score")  = r_som_ld,
            Rcpp::Named("lrt_stat")  = r_som_lrt,
            Rcpp::Named("p_value")   = r_som_pval,
            Rcpp::Named("q_value")   = r_som_qval,
            Rcpp::Named("stringsAsFactors") = false
        ),
        Rcpp::Named("n_somatic") = n_som
    );
}

}  // namespace singlet_gpu_r
