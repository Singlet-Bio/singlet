// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_cna.hpp
//
// Rcpp bindings for Numbat-style GPU CNA detection:
//   detect_cna_cpp — wraps singlet::gpu::cna::detect_cna
//
// Input marshalling decisions:
//   expr:       dgCMatrix [genes × cells]   → DeviceCSC
//   gene_chr:   IntegerVector [n_genes], 1-based chromosome (1–22, 23=X, 24=Y, 25=MT)
//   gene_start: IntegerVector [n_genes], genomic start position (bp)
//   ref:        NumericVector [n_genes] or NULL → float* or nullptr
//   cfg params: scalar R types → CnaConfig fields
//
// WHY GeneAnnotation via two vectors: R does not have a natural struct type.
//   Passing two parallel vectors (chr + start_bp) maps cleanly to GeneAnnotation.
//   Callers typically build these from a rowData(sce)$chr and rowData(sce)$start_bp
//   column annotation.
//
// Result marshalling:
//   cna_states:    device uint8 [n_cells × n_segments] → R IntegerMatrix
//   hmm_posteriors: device float [n_cells × n_segments × 3] → R array [n_cells, n_segs, 3]
//   clone_labels:  device int [n_cells] → R IntegerVector
//   segments:      host vector<SegmentBoundary> → R data.frame
//
// Host-copy overhead:
//   Upload: O(nnz_expr + n_genes) H→D
//   Download: O(n_cells × n_segs × 4) D→H (states + posteriors + clones)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/cna/numbat.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// detect_cna_cpp
//
// R-callable binding for GPU Numbat-style CNA detection.
// Mirrors numbat::numbat() output: per-cell per-segment CNA state calls,
// HMM posteriors, and clone clusters.
//
// Parameters:
//   expr            — dgCMatrix [genes × cells] (log-normalized expression)
//   gene_chr        — IntegerVector [n_genes], chromosome (1-based, 23=X, 24=Y, 25=MT)
//   gene_start      — IntegerVector [n_genes], genomic start position (bp)
//   ref             — NumericVector [n_genes] or NULL (per-gene reference mean)
//   smooth_half_win — int, rolling window half-width (full = 2*h+1)
//   trans_prob      — double, HMM off-diagonal transition prob per segment
//   segments_per_chr — int, HMM segments per chromosome
//   leiden_resolution — double, Leiden resolution for clone clustering
//   knn_k           — int, kNN k for clone clustering
//   include_sex_chrs — bool, include chrX/Y
//   seed            — int, RNG seed for Leiden (0L per cycle-24 pattern)
//   resource        — string "auto"/"gpu"/"cpu"
//
// Returns:
//   Rcpp::List{
//     "cna_states"     — IntegerMatrix [n_cells × n_total_segments] (0=loss,1=neutral,2=gain)
//     "hmm_posteriors" — array [n_cells, n_segs, 3] (HMM posterior probs)
//     "clone_labels"   — IntegerVector [n_cells] (0-based Leiden clone ID)
//     "n_clones"       — int
//     "leiden_modularity" — double
//     "segments"       — data.frame(seg_idx, chr, gene_start, gene_end)
//     "n_total_segments" — int
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List detect_cna_cpp(
        Rcpp::S4             expr,
        Rcpp::IntegerVector  gene_chr,
        Rcpp::IntegerVector  gene_start,
        Rcpp::RObject        ref,
        int                  smooth_half_win,
        double               trans_prob,
        int                  segments_per_chr,
        double               leiden_resolution,
        int                  knn_k,
        bool                 include_sex_chrs,
        int                  seed,
        std::string          resource)
{
    // ── Marshal input matrix ──────────────────────────────────────────────────
    HostCsc h_expr;
    try { h_expr = r_sparse_to_host_csc(expr); }
    catch (const std::exception& e) { Rcpp::stop("detect_cna_cpp expr: %s", e.what()); }

    const int n_genes = h_expr.n_rows;
    const int n_cells = h_expr.n_cols;

    if (gene_chr.size()   != n_genes)
        Rcpp::stop("detect_cna_cpp: gene_chr length (%d) != n_genes (%d).",
                   (int)gene_chr.size(), n_genes);
    if (gene_start.size() != n_genes)
        Rcpp::stop("detect_cna_cpp: gene_start length (%d) != n_genes (%d).",
                   (int)gene_start.size(), n_genes);

    // ── Build GeneAnnotation ──────────────────────────────────────────────────
    singlet::gpu::cna::GeneAnnotation annot;
    annot.chr     .resize(static_cast<std::size_t>(n_genes));
    annot.start_bp.resize(static_cast<std::size_t>(n_genes));
    for (int g = 0; g < n_genes; ++g) {
        annot.chr     [g] = gene_chr  [g];
        annot.start_bp[g] = static_cast<uint32_t>(std::max(0, gene_start[g]));
    }

    // ── GPUContext (owns cuBLAS + cuSPARSE + cuSOLVER handles + stream) ──────
    // GPUContext() creates all handles and a dedicated stream internally.
    singlet::gpu::core::GPUContext ctx;
    cudaStream_t stream = ctx.stream;  // alias; ctx owns lifetime

    // ── Reference normal expression (optional) ─────────────────────────────────
    std::vector<float> h_ref;
    const float* ref_ptr = nullptr;
    if (!Rf_isNull(ref)) {
        Rcpp::NumericVector rv(ref);
        if (rv.size() != n_genes)
            Rcpp::stop("detect_cna_cpp: ref length (%d) != n_genes (%d).",
                       (int)rv.size(), n_genes);
        h_ref.resize(static_cast<std::size_t>(n_genes));
        for (int g = 0; g < n_genes; ++g) h_ref[g] = static_cast<float>(rv[g]);
    }

    // Upload expression
    singlet::gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h_expr, stream); }
    catch (const std::exception& e) { Rcpp::stop("detect_cna_cpp upload expr: %s", e.what()); }

    // Upload ref if provided (one-time; outside any loop)
    singlet::gpu::core::DeviceMemory<float> d_ref;
    if (!h_ref.empty()) {
        d_ref = singlet::gpu::core::DeviceMemory<float>(static_cast<std::size_t>(n_genes));
        cudaError_t err = cudaMemcpyAsync(d_ref.get(), h_ref.data(),
                                          n_genes * sizeof(float), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            Rcpp::stop("detect_cna_cpp upload ref: %s", cudaGetErrorString(err));
        cudaStreamSynchronize(stream);
        ref_ptr = d_ref.get();
    }

    // ── Configure ─────────────────────────────────────────────────────────────
    singlet::gpu::cna::CnaConfig cfg;
    cfg.smooth_half_win     = smooth_half_win;
    cfg.trans_prob          = static_cast<float>(trans_prob);
    cfg.segments_per_chr    = segments_per_chr;
    cfg.include_sex_chrs    = include_sex_chrs;
    cfg.seed                = static_cast<uint64_t>(seed);
    cfg.knn_k               = knn_k;
    cfg.leiden_cfg.resolution = static_cast<float>(leiden_resolution);
    cfg.leiden_cfg.seed     = static_cast<uint64_t>(seed);

    // ── Run kernel ─────────────────────────────────────────────────────────────
    singlet::gpu::cna::CnaResult res;
    try {
        res = singlet::gpu::cna::detect_cna(d_expr, annot, ref_ptr, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("detect_cna_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_segs = res.n_total_segments;

    // ── Copy cna_states device → host ─────────────────────────────────────────
    const std::size_t states_sz = static_cast<std::size_t>(n_cells) * n_segs;
    std::vector<uint8_t> h_states(states_sz > 0 ? states_sz : 1);
    if (states_sz > 0) {
        cudaError_t err = cudaMemcpy(h_states.data(), res.cna_states.data(),
                                     states_sz * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("detect_cna_cpp states copyback: %s", cudaGetErrorString(err));
    }

    Rcpp::IntegerMatrix r_states(n_cells, n_segs);
    for (int c = 0; c < n_cells; ++c)
        for (int s = 0; s < n_segs; ++s)
            r_states(c, s) = static_cast<int>(
                h_states[static_cast<std::size_t>(c) * n_segs + s]);

    // ── Copy hmm_posteriors device → host ─────────────────────────────────────
    const int n_states = singlet::gpu::cna::HMM_NSTATES;
    const std::size_t post_sz = static_cast<std::size_t>(n_cells) * n_segs * n_states;
    std::vector<float> h_post(post_sz > 0 ? post_sz : 1);
    if (post_sz > 0) {
        cudaError_t err = cudaMemcpy(h_post.data(), res.hmm_posteriors.data(),
                                     post_sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("detect_cna_cpp posteriors copyback: %s", cudaGetErrorString(err));
    }

    // Build R array [n_cells, n_segs, 3]
    Rcpp::NumericVector r_post_flat(static_cast<int>(post_sz));
    for (std::size_t k = 0; k < post_sz; ++k)
        r_post_flat[static_cast<int>(k)] = static_cast<double>(h_post[k]);
    r_post_flat.attr("dim") = Rcpp::IntegerVector::create(n_cells, n_segs, n_states);

    // ── Copy clone_labels device → host ───────────────────────────────────────
    std::vector<int> h_clones(static_cast<std::size_t>(n_cells));
    {
        cudaError_t err = cudaMemcpy(h_clones.data(), res.clone_labels.data(),
                                     n_cells * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("detect_cna_cpp clones copyback: %s", cudaGetErrorString(err));
    }
    Rcpp::IntegerVector r_clones(n_cells);
    for (int i = 0; i < n_cells; ++i) r_clones[i] = h_clones[i];

    // ── Build segments data.frame ──────────────────────────────────────────────
    const int n_seg_meta = static_cast<int>(res.segments.size());
    Rcpp::IntegerVector r_seg_idx   (n_seg_meta);
    Rcpp::IntegerVector r_seg_chr   (n_seg_meta);
    Rcpp::IntegerVector r_seg_gstart(n_seg_meta);
    Rcpp::IntegerVector r_seg_gend  (n_seg_meta);
    for (int i = 0; i < n_seg_meta; ++i) {
        r_seg_idx   [i] = res.segments[i].seg_idx;
        r_seg_chr   [i] = res.segments[i].chr;
        r_seg_gstart[i] = res.segments[i].gene_start;
        r_seg_gend  [i] = res.segments[i].gene_end;
    }

    return Rcpp::List::create(
        Rcpp::Named("cna_states")        = r_states,
        Rcpp::Named("hmm_posteriors")    = r_post_flat,
        Rcpp::Named("clone_labels")      = r_clones,
        Rcpp::Named("n_clones")          = res.n_clones,
        Rcpp::Named("leiden_modularity") = static_cast<double>(res.leiden_modularity),
        Rcpp::Named("segments")          = Rcpp::DataFrame::create(
            Rcpp::Named("seg_idx")    = r_seg_idx,
            Rcpp::Named("chr")        = r_seg_chr,
            Rcpp::Named("gene_start") = r_seg_gstart,
            Rcpp::Named("gene_end")   = r_seg_gend,
            Rcpp::Named("stringsAsFactors") = false
        ),
        Rcpp::Named("n_total_segments") = n_segs
    );
}

}  // namespace singlet_gpu_r
