// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_comm.hpp
//
// Rcpp binding for cell-communication kernel (cycle 37):
//   cellchat_cpp — wraps singlet_gpu::comm::run_cellchat
//
// Input format:
//   expression    — dgCMatrix [genes × cells], log-normalized GEX
//   cell_type_ids — IntegerVector [n_cells], 0-based cell-type labels
//   ligand_ids    — IntegerMatrix [n_lr × max_set_size] (gene indices, -1 = pad)
//   receptor_ids  — IntegerMatrix [n_lr × max_set_size] (gene indices, -1 = pad)
//   pathway_ids   — IntegerVector [n_lr], 0-based pathway for each L-R pair
//   n_pathways    — int
//   n_permutations — int (default 1000)
//   seed          — int
//   resource      — "gpu" | "cpu" | "auto"
//
// Host-copy overhead:
//   Input: O(nnz) host→device (expression) + O(n_lr × max_set) H→D (LR tables).
//   Output: O(n_lr × n_types²) D→H (comm_prob, p_values) +
//           O(n_pathways × n_types²) D→H (pathway_activity).
//
// GPU-stub fallback:
//   Returns list(error="GPU not available") when FACTORNET_HAS_GPU not defined.

#pragma once

#include <Rcpp.h>
#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FACTORNET_HAS_GPU

namespace singlet_gpu_r {

// [[Rcpp::export]]
inline Rcpp::List cellchat_cpp(
        Rcpp::S4            /*expression*/,
        Rcpp::IntegerVector /*cell_type_ids*/,
        Rcpp::IntegerMatrix /*ligand_ids*/,
        Rcpp::IntegerMatrix /*receptor_ids*/,
        Rcpp::IntegerVector /*pathway_ids*/,
        int    /*n_pathways*/,
        int    /*n_permutations*/,
        int    /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

}  // namespace singlet_gpu_r

#else  // FACTORNET_HAS_GPU defined

#define FACTORNET_HAS_GPU 1
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/comm/cellchat.h>

#include <cuda_runtime.h>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// cellchat_cpp
//
// GPU cell-cell communication inference via permutation test.
// Wraps singlet_gpu::comm::run_cellchat.
//
// Returns: Rcpp::List{
//   "comm_prob"        — NumericArray [n_types × n_types × n_lr] (row-major, 3D)
//                        stored as NumericMatrix [n_types² × n_lr] for R convenience
//   "p_values"         — NumericMatrix [n_types² × n_lr]
//   "pathway_activity" — NumericMatrix [n_types² × n_pathways]
//   "n_types"          — int
//   "n_lr"             — int
//   "n_pathways"       — int
// }
// [[Rcpp::export]]
inline Rcpp::List cellchat_cpp(
        Rcpp::S4            expression,
        Rcpp::IntegerVector cell_type_ids,
        Rcpp::IntegerMatrix ligand_ids,
        Rcpp::IntegerMatrix receptor_ids,
        Rcpp::IntegerVector pathway_ids,
        int         n_pathways,
        int         n_permutations,
        int         seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("cellchat_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;
    const int n_genes = h.n_rows;
    (void)n_genes;

    if ((int)cell_type_ids.size() != n_cells)
        Rcpp::stop("cellchat_cpp: cell_type_ids length (%d) != n_cells (%d).",
                   (int)cell_type_ids.size(), n_cells);

    const int n_lr       = ligand_ids.nrow();
    const int max_set_sz = ligand_ids.ncol();

    if (receptor_ids.nrow() != n_lr || receptor_ids.ncol() != max_set_sz)
        Rcpp::stop("cellchat_cpp: ligand_ids and receptor_ids must have same dimensions.");
    if ((int)pathway_ids.size() != n_lr)
        Rcpp::stop("cellchat_cpp: pathway_ids length (%d) != n_lr (%d).",
                   (int)pathway_ids.size(), n_lr);

    int max_ct = *std::max_element(cell_type_ids.begin(), cell_type_ids.end());
    const int n_types = max_ct + 1;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("cellchat_cpp upload: %s", e.what()); }

    // Upload cell type labels
    std::vector<int> h_ct(cell_type_ids.begin(), cell_type_ids.end());
    singlet_gpu::core::DeviceMemory<int> d_ct(n_cells);
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_ct.get(), h_ct.data(),
                                n_cells * sizeof(int), cudaMemcpyHostToDevice, stream), "ct");
            chk(cudaStreamSynchronize(stream), "sync");
        } catch (const std::exception& e) {
            Rcpp::stop("cellchat_cpp label upload: %s", e.what());
        }
    }

    // Flatten L-R index matrices → host vectors for kernel
    std::vector<int> h_lig(static_cast<std::size_t>(n_lr) * max_set_sz);
    std::vector<int> h_rec(static_cast<std::size_t>(n_lr) * max_set_sz);
    std::vector<int> h_pw(n_lr);
    for (int i = 0; i < n_lr; ++i) {
        for (int j = 0; j < max_set_sz; ++j) {
            h_lig[static_cast<std::size_t>(i) * max_set_sz + j] = ligand_ids(i, j);
            h_rec[static_cast<std::size_t>(i) * max_set_sz + j] = receptor_ids(i, j);
        }
        h_pw[i] = pathway_ids[i];
    }

    singlet_gpu::comm::CellChatConfig cfg;
    cfg.n_types        = n_types;
    cfg.n_lr           = n_lr;
    cfg.n_pathways     = n_pathways;
    cfg.max_set_size   = max_set_sz;
    cfg.n_permutations = n_permutations;
    cfg.seed           = static_cast<uint64_t>(seed);
    cfg.use_gpu        = (resource != "cpu");

    singlet_gpu::comm::CellChatResult res;
    try {
        res = singlet_gpu::comm::run_cellchat(
            d_expr, d_ct, h_lig, h_rec, h_pw, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("cellchat_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_pairs   = n_types * n_types;
    const std::size_t prob_sz = static_cast<std::size_t>(n_lr) * n_pairs;
    const std::size_t pw_sz   = static_cast<std::size_t>(n_pathways) * n_pairs;

    std::vector<float> h_prob(prob_sz > 0 ? prob_sz : 1);
    std::vector<float> h_pval(prob_sz > 0 ? prob_sz : 1);
    std::vector<float> h_pact(pw_sz   > 0 ? pw_sz   : 1);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            if (prob_sz > 0) {
                chk(cudaMemcpy(h_prob.data(), res.comm_prob.data(),
                               prob_sz * sizeof(float), cudaMemcpyDeviceToHost), "comm_prob");
                chk(cudaMemcpy(h_pval.data(), res.p_values.data(),
                               prob_sz * sizeof(float), cudaMemcpyDeviceToHost), "p_values");
            }
            if (pw_sz > 0)
                chk(cudaMemcpy(h_pact.data(), res.pathway_activity.data(),
                               pw_sz * sizeof(float), cudaMemcpyDeviceToHost), "pathway_activity");
        } catch (const std::exception& e) {
            Rcpp::stop("cellchat_cpp copyback: %s", e.what());
        }
    }

    // Return as [n_pairs × n_lr] NumericMatrix (columns = LR pairs, rows = type-type combos)
    Rcpp::NumericMatrix r_prob(n_pairs, n_lr);
    Rcpp::NumericMatrix r_pval(n_pairs, n_lr);
    Rcpp::NumericMatrix r_pact(n_pairs, n_pathways);

    for (int lr = 0; lr < n_lr; ++lr)
        for (int p = 0; p < n_pairs; ++p) {
            r_prob(p, lr) = static_cast<double>(
                h_prob[static_cast<std::size_t>(lr) * n_pairs + p]);
            r_pval(p, lr) = static_cast<double>(
                h_pval[static_cast<std::size_t>(lr) * n_pairs + p]);
        }

    for (int pw = 0; pw < n_pathways; ++pw)
        for (int p = 0; p < n_pairs; ++p)
            r_pact(p, pw) = static_cast<double>(
                h_pact[static_cast<std::size_t>(pw) * n_pairs + p]);

    return Rcpp::List::create(
        Rcpp::Named("comm_prob")        = r_prob,
        Rcpp::Named("p_values")         = r_pval,
        Rcpp::Named("pathway_activity") = r_pact,
        Rcpp::Named("n_types")          = n_types,
        Rcpp::Named("n_lr")             = n_lr,
        Rcpp::Named("n_pathways")       = n_pathways
    );
}

}  // namespace singlet_gpu_r

#endif  // FACTORNET_HAS_GPU
