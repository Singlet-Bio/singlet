// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_gsea.hpp
//
// Rcpp bindings for cycle 13 GSEA + AUCell kernels:
//   fgsea_cpp    — wraps singlet::gpu::gsea::fgsea_gpu
//   aucell_cpp   — wraps singlet::gpu::gsea::aucell_gpu
//
// Input format:
//   fgsea_cpp:
//     stats    — NumericVector [n_genes], named (gene names as names())
//     pathways — named R List of character vectors (gene set members)
//     eps, min_size, max_size, n_perm, seed — fgsea::fgsea equivalents
//
//   aucell_cpp:
//     dgc      — Matrix::dgCMatrix [genes × cells]
//     gene_sets  — named R List of character vectors
//     gene_names — CharacterVector (rownames of the matrix)
//     auc_max_rank — int, top fraction of genes to use in AUC
//     norm_auc   — bool, normalize AUC scores by auc_max_rank
//
// Host-copy overhead:
//   fgsea_cpp:  O(n_genes) stats + O(total pathway members) host→device;
//               O(n_pathways × 6 fields) device→host.
//   aucell_cpp: O(nnz) matrix host→device; O(n_cells × n_sets) device→host.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/gsea/fgsea.h>
#include <singlet/gpu/gsea/aucell.h>
#include <singlet/gpu/gsea/types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// build_pathway_db
//
// Convert a named R list of character vectors (pathway members) + a named
// numeric vector (gene rank statistics) into a PathwayDB.
//
// Genes not present in stats_names are silently skipped (fgsea behavior).
// Pathways outside [min_size, max_size] are filtered here before upload to
// avoid wasting device memory on pathways the kernel will skip anyway.
// ---------------------------------------------------------------------------
inline singlet::gpu::gsea::PathwayDB
build_pathway_db(const Rcpp::List&            pathways,
                 const Rcpp::CharacterVector& stats_names,
                 int                          min_size,
                 int                          max_size)
{
    // Build gene name → 0-based index map from stats vector names
    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(static_cast<std::size_t>(stats_names.size()));
    for (int i = 0; i < stats_names.size(); ++i) {
        name_to_idx[Rcpp::as<std::string>(stats_names[i])] = i;
    }

    Rcpp::CharacterVector path_names = pathways.names();
    if (path_names.size() == 0)
        throw std::runtime_error("build_pathway_db: pathways must be a named list");

    singlet::gpu::gsea::PathwayDB db;
    const int n_paths = static_cast<int>(pathways.size());
    db.pathway_names.reserve(n_paths);
    db.member_gene_indices.reserve(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        Rcpp::CharacterVector members = pathways[p];
        std::vector<int> indices;
        indices.reserve(static_cast<std::size_t>(members.size()));

        for (int g = 0; g < members.size(); ++g) {
            auto it = name_to_idx.find(Rcpp::as<std::string>(members[g]));
            if (it != name_to_idx.end()) {
                indices.push_back(it->second);
            }
        }

        const int sz = static_cast<int>(indices.size());
        if (sz < min_size || sz > max_size) continue;

        db.pathway_names.push_back(Rcpp::as<std::string>(path_names[p]));
        db.member_gene_indices.push_back(std::move(indices));
    }

    return db;
}

// ---------------------------------------------------------------------------
// fgsea_cpp
//
// R-callable binding for GPU fast gene-set enrichment analysis.
// Mirrors fgsea::fgsea() output: data.frame with columns
//   pathway, ES, NES, pval, padj, leadingEdge
//
// Parameters:
//   stats     — named NumericVector [n_genes], gene-level rank statistics
//   pathways  — named List of character vectors (gene set members)
//   eps       — float, boundary for p-value calculation (fgsea eps)
//   min_size  — int, minimum gene set size after filtering (default 15)
//   max_size  — int, maximum gene set size (default 500)
//   n_perm    — int, number of permutations (default 1000)
//   seed      — int, RNG seed for permutations
//
// Returns:
//   data.frame(pathway, ES, NES, pval, padj, leadingEdge)
//   leadingEdge is a character vector per row stored as R list column
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::DataFrame fgsea_cpp(
        Rcpp::NumericVector  stats,
        Rcpp::List           pathways,
        double               eps,
        int                  min_size,
        int                  max_size,
        int                  n_perm,
        int                  seed)
{
    // Validate stats has names
    Rcpp::CharacterVector stats_names = stats.names();
    if (stats_names.size() == 0 || stats_names.size() != stats.size())
        Rcpp::stop("fgsea_cpp: stats must be a named numeric vector (gene names as names()).");

    const int n_genes = static_cast<int>(stats.size());

    // Build pathway DB (filtered to [min_size, max_size])
    singlet::gpu::gsea::PathwayDB db;
    try {
        db = build_pathway_db(pathways, stats_names, min_size, max_size);
    } catch (const std::exception& e) {
        Rcpp::stop("fgsea_cpp pathway DB: %s", e.what());
    }

    const int n_paths = static_cast<int>(db.pathway_names.size());
    if (n_paths == 0)
        Rcpp::stop("fgsea_cpp: no pathways passed size filters [%d, %d].", min_size, max_size);

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload stats vector host → device
    std::vector<float> h_stats(n_genes);
    for (int i = 0; i < n_genes; ++i) h_stats[i] = static_cast<float>(stats[i]);

    singlet::gpu::core::DeviceMemory<float> d_stats(n_genes);
    {
        cudaError_t err = cudaMemcpyAsync(d_stats.get(), h_stats.data(),
                                          n_genes * sizeof(float),
                                          cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            Rcpp::stop("fgsea_cpp upload stats: %s", cudaGetErrorString(err));
        cudaStreamSynchronize(stream);
    }

    // Configure and run fgsea kernel
    singlet::gpu::gsea::FgseaConfig cfg;
    cfg.eps      = static_cast<float>(eps);
    cfg.n_perm   = n_perm;
    cfg.seed     = static_cast<uint64_t>(seed);

    singlet::gpu::gsea::FgseaResult res;
    try {
        res = singlet::gpu::gsea::fgsea_gpu(d_stats, n_genes, db, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("fgsea_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy result arrays device → host
    std::vector<float> h_es   (n_paths);
    std::vector<float> h_nes  (n_paths);
    std::vector<float> h_pval (n_paths);
    std::vector<float> h_padj (n_paths);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_es.data(),   res.es.data(),   n_paths * sizeof(float), cudaMemcpyDeviceToHost), "es");
        chk(cudaMemcpy(h_nes.data(),  res.nes.data(),  n_paths * sizeof(float), cudaMemcpyDeviceToHost), "nes");
        chk(cudaMemcpy(h_pval.data(), res.pval.data(), n_paths * sizeof(float), cudaMemcpyDeviceToHost), "pval");
        chk(cudaMemcpy(h_padj.data(), res.padj.data(), n_paths * sizeof(float), cudaMemcpyDeviceToHost), "padj");
    } catch (const std::exception& e) {
        Rcpp::stop("fgsea_cpp copyback: %s", e.what());
    }

    // Build output data.frame matching fgsea::fgsea() column layout
    Rcpp::CharacterVector r_pathway(n_paths);
    Rcpp::NumericVector   r_es     (n_paths);
    Rcpp::NumericVector   r_nes    (n_paths);
    Rcpp::NumericVector   r_pval   (n_paths);
    Rcpp::NumericVector   r_padj   (n_paths);
    Rcpp::List            r_leading(n_paths);   // list column: CharacterVector per row

    // leading-edge indices come back as a flattened vector + per-pathway offsets
    std::vector<int> h_le_offsets(n_paths + 1);
    chk(cudaMemcpy(h_le_offsets.data(), res.leading_edge_offsets.data(),
                   (n_paths + 1) * sizeof(int), cudaMemcpyDeviceToHost), "le_offsets");

    const int total_le = h_le_offsets[n_paths];
    std::vector<int> h_le_idx(total_le > 0 ? total_le : 1);
    if (total_le > 0) {
        chk(cudaMemcpy(h_le_idx.data(), res.leading_edge_indices.data(),
                       total_le * sizeof(int), cudaMemcpyDeviceToHost), "le_indices");
    }

    for (int p = 0; p < n_paths; ++p) {
        r_pathway[p] = db.pathway_names[p];
        r_es     [p] = static_cast<double>(h_es  [p]);
        r_nes    [p] = static_cast<double>(h_nes [p]);
        r_pval   [p] = static_cast<double>(h_pval[p]);
        r_padj   [p] = static_cast<double>(h_padj[p]);

        const int le_start = h_le_offsets[p];
        const int le_end   = h_le_offsets[p + 1];
        const int le_n     = le_end - le_start;
        Rcpp::CharacterVector le_genes(le_n);
        for (int g = 0; g < le_n; ++g) {
            int gene_idx = h_le_idx[le_start + g];
            le_genes[g]  = (gene_idx >= 0 && gene_idx < n_genes)
                               ? Rcpp::as<std::string>(stats_names[gene_idx])
                               : "";
        }
        r_leading[p] = le_genes;
    }

    Rcpp::DataFrame out = Rcpp::DataFrame::create(
        Rcpp::Named("pathway")      = r_pathway,
        Rcpp::Named("ES")           = r_es,
        Rcpp::Named("NES")          = r_nes,
        Rcpp::Named("pval")         = r_pval,
        Rcpp::Named("padj")         = r_padj,
        Rcpp::Named("leadingEdge")  = r_leading,
        Rcpp::Named("stringsAsFactors") = false
    );
    return out;
}

// ---------------------------------------------------------------------------
// aucell_cpp
//
// R-callable binding for GPU AUCell scoring.
// Mirrors AUCell::AUCell_run() result: per-gene-set AUC scores per cell.
//
// Parameters:
//   dgc          — Matrix::dgCMatrix [genes × cells]
//   gene_sets    — named List of character vectors
//   gene_names   — CharacterVector (rownames of the matrix)
//   auc_max_rank — int, top-N genes in per-cell ranking to compute AUC over
//   norm_auc     — bool, normalize AUC by auc_max_rank (0–1 scale)
//
// Returns:
//   NumericMatrix [n_cells × n_sets] (matches AUCell aucMaxRank convention)
//   with dimnames[[2]] = set names
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix aucell_cpp(
        Rcpp::S4             dgc,
        Rcpp::List           gene_sets,
        Rcpp::CharacterVector gene_names,
        int                  auc_max_rank,
        bool                 norm_auc)
{
    // Marshal R sparse matrix → host CSC
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if (auc_max_rank <= 0 || auc_max_rank > n_genes)
        Rcpp::stop("aucell_cpp: auc_max_rank must be in [1, n_genes=%d].", n_genes);

    // Build gene-set DB (reuse helper from _bind_anno.hpp namespace via same TU)
    singlet::gpu::anno::GeneSetDB db;
    try { db = build_gene_set_db(gene_sets, gene_names); }
    catch (const std::exception& e) { Rcpp::stop("aucell_cpp gene sets: %s", e.what()); }

    const int n_sets = static_cast<int>(db.set_names.size());
    if (n_sets == 0)
        Rcpp::stop("aucell_cpp: no gene sets remain after filtering.");

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload matrix
    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("aucell_cpp upload: %s", e.what()); }

    // Configure AUCell kernel
    singlet::gpu::gsea::AUCellConfig cfg;
    cfg.auc_max_rank = auc_max_rank;
    cfg.norm_auc     = norm_auc;

    singlet::gpu::gsea::AUCellResult res;
    try {
        res = singlet::gpu::gsea::aucell_gpu(d, db, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("aucell_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy scores device → host [n_sets × n_cells column-major]
    const std::size_t total = static_cast<std::size_t>(n_sets) * n_cells;
    std::vector<float> h_scores(total);
    {
        cudaError_t err = cudaMemcpy(h_scores.data(), res.scores.data(),
                                     total * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("aucell_cpp copyback: %s", cudaGetErrorString(err));
    }

    // Build R NumericMatrix [n_cells × n_sets] (transpose device layout)
    Rcpp::NumericMatrix r_auc(n_cells, n_sets);
    for (int s = 0; s < n_sets; ++s) {
        for (int c = 0; c < n_cells; ++c) {
            r_auc(c, s) = static_cast<double>(
                h_scores[static_cast<std::size_t>(s) * n_cells + c]);
        }
    }

    // Column names = set names
    Rcpp::CharacterVector r_set_names(n_sets);
    for (int s = 0; s < n_sets; ++s) r_set_names[s] = db.set_names[s];
    Rcpp::colnames(r_auc) = r_set_names;

    return r_auc;
}

}  // namespace singlet_gpu_r
