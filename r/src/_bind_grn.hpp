// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_grn.hpp
//
// Rcpp bindings for GRaNIE-style GPU GRN inference:
//   granie_cpp — wraps singlet::gpu::grn::run_granie
//
// Input marshalling decisions:
//   gex:              dgCMatrix [genes × cells]  → DeviceCSC (upload_host_csc_to_device)
//   peaks:            dgCMatrix [peaks × cells]  → DeviceCSC (upload_host_csc_to_device)
//   tf_motif:         R list with {row_ptr, col_idx, values, n_tfs, n_peaks} → HostCSR
//   peak_gene_pairs:  IntegerMatrix [n_pairs × 2] (peak_idx, gene_idx) → std::vector<int2>
//   cfg params:       scalar R numerics/logicals → GranieConfig
//
// WHY tf_motif as R list: the caller typically constructs this from JASPAR / HOMER
// motif annotations in R before calling granie_cpp.  A named list is the natural
// R representation of a sparse CSR matrix (row_ptr, col_idx, values).
//
// WHY peak_gene_pairs as IntegerMatrix: simpler than asking the user to construct
// a list of pairs.  The caller uses GenomicRanges::findOverlaps() to build cis-window
// pairs in R, then passes the two-column index matrix here.
//
// Host-copy overhead (one-time, at function entry):
//   gex upload:      O(nnz_gex)   host→device
//   peaks upload:    O(nnz_peaks) host→device
//   tf_motif upload: O(nnz_motif) host→device (inside run_granie)
//   pairs upload:    O(n_pairs)   host→device (inside run_granie)
//   result:          O(n_edges + n_tfs × n_cells) device→host (at exit)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/handles.h>
#include <singlet/gpu/grn/granie.h>

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
// r_list_to_host_csr
//
// Convert a named R list with slots {row_ptr, col_idx, values, n_rows, n_cols}
// into a singlet::gpu::grn::HostCSR.
//
// row_ptr must be 1-based (R convention); we convert to 0-based here.
// col_idx must be 1-based (R convention); we convert to 0-based here.
// ---------------------------------------------------------------------------
inline singlet::gpu::grn::HostCSR
r_list_to_host_csr(const Rcpp::List& tf_motif)
{
    // Validate required slots
    for (const char* slot : {"row_ptr", "col_idx", "values", "n_rows", "n_cols"}) {
        if (!tf_motif.containsElementNamed(slot))
            throw std::runtime_error(std::string("r_list_to_host_csr: missing slot '") +
                                     slot + "'");
    }

    singlet::gpu::grn::HostCSR csr;
    csr.n_rows = Rcpp::as<int>(tf_motif["n_rows"]);
    csr.n_cols = Rcpp::as<int>(tf_motif["n_cols"]);

    Rcpp::IntegerVector rp  = tf_motif["row_ptr"];
    Rcpp::IntegerVector ci  = tf_motif["col_idx"];
    Rcpp::NumericVector val = tf_motif["values"];

    csr.nnz = static_cast<int>(ci.size());

    csr.row_ptr.resize(static_cast<std::size_t>(csr.n_rows + 1));
    csr.col_indices.resize(static_cast<std::size_t>(csr.nnz));
    csr.values.resize(static_cast<std::size_t>(csr.nnz));

    // row_ptr: R is 1-based (length n_rows+1)
    for (int i = 0; i <= csr.n_rows; ++i) csr.row_ptr[i] = rp[i] - 1;
    // col_idx: R is 1-based
    for (int k = 0; k < csr.nnz; ++k) {
        csr.col_indices[k] = ci[k] - 1;
        csr.values[k]      = static_cast<float>(val[k]);
    }

    return csr;
}

// ---------------------------------------------------------------------------
// r_matrix_to_int2_pairs
//
// Convert an IntegerMatrix [n_pairs × 2] (peak_idx, gene_idx) to a host
// std::vector<int2>.  Indices are converted from 1-based R to 0-based C++.
// ---------------------------------------------------------------------------
inline std::vector<int2>
r_matrix_to_int2_pairs(const Rcpp::IntegerMatrix& mat)
{
    const int n = mat.nrow();
    if (mat.ncol() < 2)
        throw std::runtime_error("r_matrix_to_int2_pairs: matrix must have 2 columns "
                                 "(peak_idx, gene_idx)");
    std::vector<int2> pairs(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        pairs[i].x = mat(i, 0) - 1;  // peak_idx, 1→0-based
        pairs[i].y = mat(i, 1) - 1;  // gene_idx, 1→0-based
    }
    return pairs;
}

// ---------------------------------------------------------------------------
// granie_cpp
//
// R-callable binding for GPU GRaNIE-style GRN inference.
// Mirrors the GRaNIE R package (Gaumondo et al. 2024) output:
//   a data.frame of TF→target edges with combined_score, p_value, fdr.
//
// Parameters:
//   gex            — dgCMatrix [genes × cells] (log-normalized expression)
//   peaks          — dgCMatrix [peaks × cells] (raw ATAC accessibility)
//   tf_motif       — named list{row_ptr, col_idx, values, n_rows=n_tfs, n_cols=n_peaks}
//                    CSR of TF×peak binary motif hits (1-based R indices)
//   peak_gene_pairs — IntegerMatrix [n_pairs × 2] col(0)=peak_idx, col(1)=gene_idx (1-based)
//   tf_names       — CharacterVector [n_tfs] (names for result labeling)
//   gene_names     — CharacterVector [n_genes] (names for result labeling)
//   min_abs_r      — double, |r| threshold for peak–gene links
//   peak_gene_fdr  — double, BH FDR threshold for peak–gene links
//   tf_target_fdr  — double, BH FDR threshold for TF–target edges
//   min_peak_mean  — double
//   min_peak_var   — double
//   run_leiden     — bool, run Leiden community detection on GRN
//   leiden_resolution — double
//   chunk_size_tfs — int, TFs per chunk in step 4
//   seed           — int
//   resource       — string "auto"/"gpu"/"cpu" (controls GPU usage)
//
// Returns:
//   Rcpp::List{
//     "edges"           — data.frame(tf, gene, combined_score, pvalue, fdr, community)
//     "tf_activity"     — NumericMatrix [n_tfs × n_cells]
//     "n_edges"         — int
//     "n_peak_gene_links" — int
//     "n_communities"   — int (0 if run_leiden=FALSE)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List granie_cpp(
        Rcpp::S4             gex,
        Rcpp::S4             peaks,
        Rcpp::List           tf_motif,
        Rcpp::IntegerMatrix  peak_gene_pairs,
        Rcpp::CharacterVector tf_names,
        Rcpp::CharacterVector gene_names,
        double               min_abs_r,
        double               peak_gene_fdr,
        double               tf_target_fdr,
        double               min_peak_mean,
        double               min_peak_var,
        bool                 run_leiden,
        double               leiden_resolution,
        int                  chunk_size_tfs,
        int                  seed,
        std::string          resource)
{
    // ── Input validation ─────────────────────────────────────────────────────
    HostCsc h_gex, h_peaks;
    try { h_gex   = r_sparse_to_host_csc(gex);   }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp gex: %s",   e.what()); }
    try { h_peaks = r_sparse_to_host_csc(peaks);  }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp peaks: %s", e.what()); }

    const int n_cells = h_gex.n_cols;
    const int n_genes = h_gex.n_rows;
    const int n_peaks_count = h_peaks.n_rows;

    if (h_peaks.n_cols != n_cells)
        Rcpp::stop("granie_cpp: gex and peaks must have the same number of cells "
                   "(gex: %d cells, peaks: %d cells).", n_cells, h_peaks.n_cols);
    if (peak_gene_pairs.nrow() == 0)
        Rcpp::stop("granie_cpp: peak_gene_pairs is empty.");

    // Build host data structures
    singlet::gpu::grn::HostCSR csr_motif;
    try { csr_motif = r_list_to_host_csr(tf_motif); }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp tf_motif: %s", e.what()); }

    if (csr_motif.n_cols != n_peaks_count)
        Rcpp::stop("granie_cpp: tf_motif n_cols (%d) != n_peaks in peaks matrix (%d).",
                   csr_motif.n_cols, n_peaks_count);

    std::vector<int2> pairs_h;
    try { pairs_h = r_matrix_to_int2_pairs(peak_gene_pairs); }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp pairs: %s", e.what()); }

    // ── GPUContext (owns cuBLAS + cuSPARSE + cuSOLVER handles + stream) ──────
    // GPUContext() creates all handles and a dedicated stream internally, then
    // binds all handles to that stream.  We use ctx.stream for all operations.
    singlet::gpu::core::GPUContext ctx;
    cudaStream_t stream = ctx.stream;  // alias for readability; ctx owns lifetime

    // ── Upload matrices ──────────────────────────────────────────────────────
    singlet::gpu::core::DeviceCSC d_gex, d_peaks;
    try { d_gex   = upload_host_csc_to_device(h_gex,   stream); }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp upload gex: %s",   e.what()); }
    try { d_peaks = upload_host_csc_to_device(h_peaks, stream); }
    catch (const std::exception& e) { Rcpp::stop("granie_cpp upload peaks: %s", e.what()); }

    // ── Configure GraNIE ─────────────────────────────────────────────────────
    singlet::gpu::grn::GranieConfig cfg;
    cfg.min_peak_mean      = static_cast<float>(min_peak_mean);
    cfg.min_peak_var       = static_cast<float>(min_peak_var);
    cfg.min_abs_r          = static_cast<float>(min_abs_r);
    cfg.peak_gene_fdr      = static_cast<float>(peak_gene_fdr);
    cfg.tf_target_fdr      = static_cast<float>(tf_target_fdr);
    cfg.run_leiden         = run_leiden;
    cfg.leiden_resolution  = static_cast<float>(leiden_resolution);
    cfg.chunk_size_tfs     = chunk_size_tfs;
    cfg.seed               = static_cast<uint64_t>(seed);

    // ── Run kernel ────────────────────────────────────────────────────────────
    singlet::gpu::grn::GranieResult res;
    try {
        res = singlet::gpu::grn::run_granie(d_gex, d_peaks, csr_motif, pairs_h, ctx, cfg);
    } catch (const std::exception& e) {
        Rcpp::stop("granie_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // ── Marshal edges data.frame ─────────────────────────────────────────────
    const int n_edges = res.n_edges;
    Rcpp::CharacterVector r_tf   (n_edges);
    Rcpp::CharacterVector r_gene (n_edges);
    Rcpp::NumericVector   r_score(n_edges);
    Rcpp::NumericVector   r_pval (n_edges);
    Rcpp::NumericVector   r_fdr  (n_edges);
    Rcpp::IntegerVector   r_comm (n_edges);

    const bool have_names_tf   = (tf_names.size()   == csr_motif.n_rows);
    const bool have_names_gene = (gene_names.size() == n_genes);
    const bool have_comm       = ((int)res.community_labels.size() == n_edges);

    for (int i = 0; i < n_edges; ++i) {
        const auto& e = res.edges[i];
        r_tf  [i] = have_names_tf
                        ? Rcpp::as<std::string>(tf_names  [e.tf_idx])
                        : std::to_string(e.tf_idx);
        r_gene[i] = have_names_gene
                        ? Rcpp::as<std::string>(gene_names[e.gene_idx])
                        : std::to_string(e.gene_idx);
        r_score[i] = static_cast<double>(e.combined_score);
        r_pval [i] = static_cast<double>(e.p_value);
        r_fdr  [i] = static_cast<double>(e.fdr);
        r_comm [i] = have_comm ? res.community_labels[i] : -1;
    }

    Rcpp::DataFrame r_edges = Rcpp::DataFrame::create(
        Rcpp::Named("tf")             = r_tf,
        Rcpp::Named("gene")           = r_gene,
        Rcpp::Named("combined_score") = r_score,
        Rcpp::Named("pvalue")         = r_pval,
        Rcpp::Named("fdr")            = r_fdr,
        Rcpp::Named("community")      = r_comm,
        Rcpp::Named("stringsAsFactors") = false
    );

    // ── Marshal TF activity matrix ───────────────────────────────────────────
    const int n_tfs_out = res.n_tfs;
    const int n_cells_out = res.n_cells;
    Rcpp::NumericMatrix r_tf_act(n_tfs_out, n_cells_out);
    for (int t = 0; t < n_tfs_out; ++t) {
        for (int c = 0; c < n_cells_out; ++c) {
            r_tf_act(t, c) = static_cast<double>(
                res.tf_activity_host[static_cast<std::size_t>(t) * n_cells_out + c]);
        }
    }
    if (have_names_tf) {
        Rcpp::CharacterVector rownames_tf(n_tfs_out);
        for (int t = 0; t < n_tfs_out && t < tf_names.size(); ++t)
            rownames_tf[t] = tf_names[t];
        Rcpp::rownames(r_tf_act) = rownames_tf;
    }

    return Rcpp::List::create(
        Rcpp::Named("edges")              = r_edges,
        Rcpp::Named("tf_activity")        = r_tf_act,
        Rcpp::Named("n_edges")            = n_edges,
        Rcpp::Named("n_peak_gene_links")  = res.n_peak_gene_links,
        Rcpp::Named("n_communities")      = res.n_communities
    );
}

}  // namespace singlet_gpu_r
