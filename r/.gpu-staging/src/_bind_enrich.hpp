// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_enrich.hpp
//
// Rcpp bindings for advanced enrichment kernels (cycle 52b):
//   ssgsea_cpp   — wraps singlet::gpu::enrich::ssgsea_gpu
//   progeny_cpp  — wraps singlet::gpu::enrich::progeny_gpu
//
// ssGSEA:
//   Single-sample GSEA.  Extends run_aucell() with rank-normalized enrichment
//   scores per cell per pathway.  Analog of gsva::gsva(method="ssgsea").
//
// PROGENy:
//   Pathway RespOnsive GENe activity: linear model scores for 14 cancer pathways.
//   Analog of progeny::progeny() from Bioconductor.
//
// Host-copy overhead:
//   ssgsea_cpp: O(nnz) host→device; O(n_cells × n_sets) device→host.
//   progeny_cpp: O(nnz) host→device; O(n_cells × n_pathways) device→host.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/enrich/ssgsea.h>
#include <singlet/gpu/enrich/ssgsea_types.h>
#include <singlet/gpu/enrich/progeny.h>
#include <singlet/gpu/enrich/progeny_types.h>
#include <singlet/gpu/gsea/types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device
#include "_bind_anno.hpp"         // build_gene_set_db

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// ssgsea_cpp
//
// Single-sample GSEA scores for every cell × gene set combination.
// Analog of gsva::gsva(method="ssgsea") / escape::enrichIt(method="ssGSEA").
//
// Parameters:
//   dgc          — dgCMatrix [genes × cells]
//   gene_sets    — named List of character vectors (gene set members)
//   gene_names   — CharacterVector [n_genes] (rownames of matrix)
//   alpha        — double, rank-normalization exponent (ssGSEA alpha, default 0.25)
//   normalize    — bool, divide scores by |set| to get 0–1 range
//
// Returns:
//   NumericMatrix [n_cells × n_sets], ssGSEA scores per cell per pathway
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix ssgsea_cpp(
        Rcpp::S4              dgc,
        Rcpp::List            gene_sets,
        Rcpp::CharacterVector gene_names,
        double                alpha,
        bool                  normalize)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("ssgsea_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if (static_cast<int>(gene_names.size()) != n_genes)
        Rcpp::stop("ssgsea_cpp: gene_names length (%d) != n_genes (%d).",
                   (int)gene_names.size(), n_genes);

    singlet::gpu::anno::GeneSetDB db;
    try { db = build_gene_set_db(gene_sets, gene_names); }
    catch (const std::exception& e) { Rcpp::stop("ssgsea_cpp gene sets: %s", e.what()); }

    const int n_sets = static_cast<int>(db.set_names.size());
    if (n_sets == 0)
        Rcpp::stop("ssgsea_cpp: no gene sets mapped to matrix genes.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("ssgsea_cpp upload: %s", e.what()); }

    singlet::gpu::enrich::SsGseaConfig cfg;
    cfg.alpha     = static_cast<float>(alpha);
    cfg.normalize = normalize;

    singlet::gpu::enrich::SsGseaResult res;
    try {
        res = singlet::gpu::enrich::ssgsea_gpu(d, db, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("ssgsea_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const std::size_t total = static_cast<std::size_t>(n_sets) * n_cells;
    std::vector<float> h_scores(total);
    cudaMemcpy(h_scores.data(), res.scores.data(),
               total * sizeof(float), cudaMemcpyDeviceToHost);

    // Result layout on device: [n_sets × n_cells] col-major
    // Return [n_cells × n_sets] for R convention
    Rcpp::NumericMatrix r_mat(n_cells, n_sets);
    for (int s = 0; s < n_sets; ++s)
        for (int c = 0; c < n_cells; ++c)
            r_mat(c, s) = static_cast<double>(
                h_scores[static_cast<std::size_t>(s) * n_cells + c]);

    Rcpp::CharacterVector r_set_names(n_sets);
    for (int s = 0; s < n_sets; ++s) r_set_names[s] = db.set_names[s];
    Rcpp::colnames(r_mat) = r_set_names;

    return r_mat;
}

// ---------------------------------------------------------------------------
// progeny_cpp
//
// PROGENy pathway activity scores from a pre-trained weight matrix.
// Analog of progeny::progeny(expr, organism="Human", top=500).
//
// Parameters:
//   dgc            — dgCMatrix [genes × cells], logcounts
//   gene_names     — CharacterVector [n_genes], matrix rownames
//   weight_matrix  — NumericMatrix [n_pathway_genes × n_pathways], PROGENy weights
//   weight_genes   — CharacterVector [n_pathway_genes], rownames of weight_matrix
//   pathway_names  — CharacterVector [n_pathways], colnames of weight_matrix
//   top_n          — int, top N genes per pathway to use (0 = all)
//   scale          — bool, z-score scale output across cells
//
// Returns:
//   NumericMatrix [n_cells × n_pathways], pathway activity scores
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix progeny_cpp(
        Rcpp::S4              dgc,
        Rcpp::CharacterVector gene_names,
        Rcpp::NumericMatrix   weight_matrix,
        Rcpp::CharacterVector weight_genes,
        Rcpp::CharacterVector pathway_names,
        int                   top_n,
        bool                  scale)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("progeny_cpp: %s", e.what()); }

    const int n_genes     = h.n_rows;
    const int n_cells     = h.n_cols;
    const int n_pw_genes  = static_cast<int>(weight_genes.size());
    const int n_pathways  = static_cast<int>(pathway_names.size());

    if (weight_matrix.nrow() != n_pw_genes || weight_matrix.ncol() != n_pathways)
        Rcpp::stop("progeny_cpp: weight_matrix dims (%d x %d) inconsistent with "
                   "weight_genes (%d) / pathway_names (%d).",
                   weight_matrix.nrow(), weight_matrix.ncol(), n_pw_genes, n_pathways);
    if (static_cast<int>(gene_names.size()) != n_genes)
        Rcpp::stop("progeny_cpp: gene_names length (%d) != n_genes (%d).",
                   (int)gene_names.size(), n_genes);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("progeny_cpp upload: %s", e.what()); }

    // Build gene-name → matrix-row-index lookup
    std::unordered_map<std::string, int> gene_idx;
    gene_idx.reserve(static_cast<std::size_t>(n_genes));
    for (int i = 0; i < n_genes; ++i)
        gene_idx[Rcpp::as<std::string>(gene_names[i])] = i;

    // Build PROGENy weight table: for each (gene, pathway) entry in the weight
    // matrix, record (gene_row_index, pathway_idx, weight) and upload to device.
    std::vector<int>   h_prog_gene_idx;
    std::vector<int>   h_prog_path_idx;
    std::vector<float> h_prog_weight;

    for (int g = 0; g < n_pw_genes; ++g) {
        auto it = gene_idx.find(Rcpp::as<std::string>(weight_genes[g]));
        if (it == gene_idx.end()) continue;
        const int row_idx = it->second;
        for (int p = 0; p < n_pathways; ++p) {
            float w = static_cast<float>(weight_matrix(g, p));
            if (w == 0.0f) continue;
            h_prog_gene_idx.push_back(row_idx);
            h_prog_path_idx.push_back(p);
            h_prog_weight  .push_back(w);
        }
    }

    const int n_entries = static_cast<int>(h_prog_gene_idx.size());
    if (n_entries == 0)
        Rcpp::stop("progeny_cpp: no weight_genes found in matrix gene_names.");

    singlet::gpu::core::DeviceMemory<int>   d_pg(n_entries);
    singlet::gpu::core::DeviceMemory<int>   d_pp(n_entries);
    singlet::gpu::core::DeviceMemory<float> d_pw(n_entries);
    cudaMemcpyAsync(d_pg.get(), h_prog_gene_idx.data(), n_entries * sizeof(int),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pp.get(), h_prog_path_idx.data(), n_entries * sizeof(int),   cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_pw.get(), h_prog_weight.data(),   n_entries * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    singlet::gpu::enrich::ProgenyConfig cfg;
    cfg.n_pathways  = n_pathways;
    cfg.top_n       = top_n;
    cfg.scale       = scale;

    singlet::gpu::enrich::ProgenyResult res;
    try {
        res = singlet::gpu::enrich::progeny_gpu(
            d, d_pg, d_pp, d_pw, n_entries, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("progeny_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const std::size_t out_sz = static_cast<std::size_t>(n_cells) * n_pathways;
    std::vector<float> h_act(out_sz);
    cudaMemcpy(h_act.data(), res.activity.data(),
               out_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // Device layout: [n_pathways × n_cells] col-major → [n_cells × n_pathways] R
    Rcpp::NumericMatrix r_act(n_cells, n_pathways);
    for (int p = 0; p < n_pathways; ++p)
        for (int c = 0; c < n_cells; ++c)
            r_act(c, p) = static_cast<double>(
                h_act[static_cast<std::size_t>(p) * n_cells + c]);

    Rcpp::CharacterVector r_pw_names(n_pathways);
    for (int p = 0; p < n_pathways; ++p)
        r_pw_names[p] = Rcpp::as<std::string>(pathway_names[p]);
    Rcpp::colnames(r_act) = r_pw_names;

    return r_act;
}

}  // namespace singlet_gpu_r
