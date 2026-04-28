// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_fate.hpp
//
// Rcpp bindings for fate trajectory kernels (cycles 41, 43, 45):
//   cospar_cpp      — wraps singlet_gpu::fate::run_cospar
//   cellrank2_cpp   — wraps singlet_gpu::fate::run_cellrank2
//   palantir_cpp    — wraps singlet_gpu::fate::run_palantir
//
// Shared file for all three fate features — matches design doc note that
// Cospar, CellRank 2, and Palantir form a complementary fate-inference triad.
//
// Input/output conventions:
//   All three functions take a dgCMatrix [genes × cells] (log-normalized GEX)
//   as the primary expression input plus method-specific ancillary arguments.
//   All return a named Rcpp::List that can be enriched by the R-side wrappers
//   (fate.R) before being stored into colData / reducedDim / metadata of an SCE.
//
// Host-copy overhead (documented per cycle 24 pattern):
//   cospar_cpp:    O(nnz) host→device (expression) + O(n_cells) H→D (ancillaries)
//                  O(n_cells × n_fates + n_cells) device→host (fate_bias, potency)
//   cellrank2_cpp: O(nnz_T) host→device (transition matrix) +
//                  O(n_cells × n_terminals) device→host (absorption_prob)
//   palantir_cpp:  O(nnz) host→device (expression) +
//                  O(n_cells × (1 + n_terminals)) device→host (pseudotime + branch_probs)
//
// GPU-stub behaviour (no CUDA):
//   When FACTORNET_HAS_GPU is not defined at compile time the three functions
//   return list(error="GPU not available") — identical to the gpu_stubs pattern.
//   The R wrappers (fate.R) check for the presence of $error before proceeding.

#pragma once

#include <Rcpp.h>
#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// GPU stub path — compiled when CUDA / GPU headers are absent.
// Returns an error list so the package still loads on CRAN build servers.
// ---------------------------------------------------------------------------
#ifndef FACTORNET_HAS_GPU

namespace singlet_gpu_r {

// [[Rcpp::export]]
inline Rcpp::List cospar_cpp(
        Rcpp::S4            /*expression*/,
        Rcpp::IntegerVector /*time_points*/,
        Rcpp::IntegerVector /*clone_ids*/,
        int    /*n_fates*/,
        double /*lambda1*/,
        double /*lambda2*/,
        int    /*max_iter*/,
        double /*tol*/,
        int    /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

// [[Rcpp::export]]
inline Rcpp::List cellrank2_cpp(
        Rcpp::S4            /*transition_matrix*/,
        Rcpp::IntegerVector /*terminal_state_ids*/,
        int    /*max_iter*/,
        double /*tol*/,
        int    /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

// [[Rcpp::export]]
inline Rcpp::List palantir_cpp(
        Rcpp::S4 /*expression*/,
        int      /*start_cell*/,
        int      /*n_terminals*/,
        int      /*n_neighbors*/,
        int      /*n_diffusion_components*/,
        int      /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

}  // namespace singlet_gpu_r

#else  // FACTORNET_HAS_GPU defined

// ---------------------------------------------------------------------------
// Full GPU path
// ---------------------------------------------------------------------------

#define FACTORNET_HAS_GPU 1
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/fate/cospar.h>
#include <singlet-gpu/fate/cellrank2.h>
#include <singlet-gpu/fate/palantir.h>

#include <cuda_runtime.h>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// cospar_cpp
//
// GPU cell fate transition mapping via constrained coordinate descent.
// Wraps singlet_gpu::fate::run_cospar.
//
// Parameters:
//   expression    — dgCMatrix [genes × cells], log-normalized GEX
//   time_points   — IntegerVector [n_cells], 0-based time-point index per cell
//   clone_ids     — IntegerVector [n_cells], shared clone label (-1 = unassigned)
//   n_fates       — int, number of terminal fates (0 = auto from time_points)
//   lambda1       — double, L1 sparsity weight on transition map
//   lambda2       — double, state-similarity regularisation weight
//   max_iter      — int, coordinate-descent outer iterations (default 50)
//   tol           — double, relative Frobenius convergence threshold (1e-4)
//   seed          — int, RNG seed (deterministic — no stochasticity in core)
//   resource      — "gpu" | "cpu" | "auto"
//
// Returns: Rcpp::List{
//   "fate_bias"       — NumericMatrix [n_cells × n_fates]
//   "potency_score"   — NumericVector [n_cells]
//   "driver_genes"    — NumericMatrix [n_fates × n_genes]
//   "iters_run"       — int
//   "converged"       — bool
//   "state_only_mode" — bool (TRUE when no clone IDs provided)
// }
// [[Rcpp::export]]
inline Rcpp::List cospar_cpp(
        Rcpp::S4            expression,
        Rcpp::IntegerVector time_points,
        Rcpp::IntegerVector clone_ids,
        int         n_fates,
        double      lambda1,
        double      lambda2,
        int         max_iter,
        double      tol,
        int         seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("cospar_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if ((int)time_points.size() != n_cells)
        Rcpp::stop("cospar_cpp: time_points length (%d) != n_cells (%d).",
                   (int)time_points.size(), n_cells);
    if ((int)clone_ids.size() != n_cells && (int)clone_ids.size() != 0)
        Rcpp::stop("cospar_cpp: clone_ids must be length n_cells (%d) or 0.", n_cells);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("cospar_cpp upload: %s", e.what()); }

    // Upload ancillary integer vectors
    std::vector<int> h_tp(time_points.begin(), time_points.end());
    std::vector<int> h_cl(clone_ids.begin(),   clone_ids.end());

    singlet_gpu::core::DeviceMemory<int> d_tp(n_cells), d_cl;
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_tp.get(), h_tp.data(),
                                n_cells * sizeof(int), cudaMemcpyHostToDevice, stream), "tp upload");
            if (!h_cl.empty()) {
                d_cl = singlet_gpu::core::DeviceMemory<int>(n_cells);
                chk(cudaMemcpyAsync(d_cl.data(), h_cl.data(),
                                    n_cells * sizeof(int), cudaMemcpyHostToDevice, stream), "cl upload");
            }
            chk(cudaStreamSynchronize(stream), "sync");
        } catch (const std::exception& e) {
            Rcpp::stop("cospar_cpp ancillary upload: %s", e.what());
        }
    }

    singlet_gpu::fate::CosparConfig cfg;
    cfg.n_fates    = n_fates;
    cfg.lambda1    = static_cast<float>(lambda1);
    cfg.lambda2    = static_cast<float>(lambda2);
    cfg.max_iter   = max_iter;
    cfg.tol        = static_cast<float>(tol);
    cfg.seed       = static_cast<uint64_t>(seed);
    cfg.use_gpu    = (resource != "cpu");

    singlet_gpu::fate::CosparResult res;
    try {
        res = singlet_gpu::fate::run_cospar(
            d_expr, d_tp, d_cl, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("cospar_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int nf = res.n_terminal_fates;
    const std::size_t bias_sz = static_cast<std::size_t>(n_cells) * nf;
    const std::size_t drv_sz  = static_cast<std::size_t>(nf) * n_genes;

    std::vector<float> h_bias(bias_sz > 0 ? bias_sz : 1);
    std::vector<float> h_pot(n_cells);
    std::vector<float> h_drv(drv_sz > 0 ? drv_sz : 1);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            if (bias_sz > 0)
                chk(cudaMemcpy(h_bias.data(), res.fate_bias.data(),
                               bias_sz * sizeof(float), cudaMemcpyDeviceToHost), "fate_bias");
            chk(cudaMemcpy(h_pot.data(), res.potency_score.data(),
                           n_cells * sizeof(float), cudaMemcpyDeviceToHost), "potency");
            if (drv_sz > 0)
                chk(cudaMemcpy(h_drv.data(), res.driver_genes.data(),
                               drv_sz * sizeof(float), cudaMemcpyDeviceToHost), "driver_genes");
        } catch (const std::exception& e) {
            Rcpp::stop("cospar_cpp copyback: %s", e.what());
        }
    }

    Rcpp::NumericMatrix r_bias(n_cells, nf);
    for (int c = 0; c < n_cells; ++c)
        for (int f = 0; f < nf; ++f)
            r_bias(c, f) = static_cast<double>(h_bias[static_cast<std::size_t>(c) * nf + f]);

    Rcpp::NumericVector r_pot(n_cells);
    for (int c = 0; c < n_cells; ++c) r_pot[c] = static_cast<double>(h_pot[c]);

    Rcpp::NumericMatrix r_drv(nf, n_genes);
    for (int f = 0; f < nf; ++f)
        for (int g = 0; g < n_genes; ++g)
            r_drv(f, g) = static_cast<double>(
                h_drv[static_cast<std::size_t>(f) * n_genes + g]);

    return Rcpp::List::create(
        Rcpp::Named("fate_bias")       = r_bias,
        Rcpp::Named("potency_score")   = r_pot,
        Rcpp::Named("driver_genes")    = r_drv,
        Rcpp::Named("iters_run")       = res.iters_run,
        Rcpp::Named("converged")       = res.converged,
        Rcpp::Named("state_only_mode") = res.state_only_mode
    );
}

// ---------------------------------------------------------------------------
// cellrank2_cpp
//
// GPU absorption probability computation via batched GMRES.
// Wraps singlet_gpu::fate::run_cellrank2.
//
// Parameters:
//   transition_matrix  — dgCMatrix [n_cells × n_cells], row-stochastic
//   terminal_state_ids — IntegerVector, 0-based indices of absorbing states
//   max_iter           — int, GMRES restart dimension (default 50)
//   tol                — double, relative residual tolerance (1e-6)
//   seed               — int, (unused; kept for API consistency)
//   resource           — "gpu" | "cpu" | "auto"
//
// Returns: Rcpp::List{
//   "absorption_prob"  — NumericMatrix [n_cells × n_terminals]
//   "driver_genes"     — NumericMatrix [n_terminals × n_genes] (NULL if no expression given)
//   "converged_all"    — bool
//   "n_terminals"      — int
// }
// [[Rcpp::export]]
inline Rcpp::List cellrank2_cpp(
        Rcpp::S4            transition_matrix,
        Rcpp::IntegerVector terminal_state_ids,
        int         max_iter,
        double      tol,
        int         seed,
        std::string resource)
{
    HostCsc h_T;
    try { h_T = r_sparse_to_host_csc(transition_matrix); }
    catch (const std::exception& e) { Rcpp::stop("cellrank2_cpp: %s", e.what()); }

    const int n_cells    = h_T.n_cols;
    const int n_term     = static_cast<int>(terminal_state_ids.size());

    if (h_T.n_rows != n_cells)
        Rcpp::stop("cellrank2_cpp: transition_matrix must be square [%d × %d].",
                   h_T.n_rows, n_cells);
    if (n_term < 1)
        Rcpp::stop("cellrank2_cpp: need at least 1 terminal state.");

    std::vector<int> h_term(terminal_state_ids.begin(), terminal_state_ids.end());
    for (int t : h_term)
        if (t < 0 || t >= n_cells)
            Rcpp::stop("cellrank2_cpp: terminal_state_id %d out of range [0, %d).", t, n_cells);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_T;
    try { d_T = upload_host_csc_to_device(h_T, stream); }
    catch (const std::exception& e) { Rcpp::stop("cellrank2_cpp upload: %s", e.what()); }

    singlet_gpu::core::DeviceMemory<int> d_term(n_term);
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_term.get(), h_term.data(),
                                n_term * sizeof(int), cudaMemcpyHostToDevice, stream), "term upload");
            chk(cudaStreamSynchronize(stream), "sync");
        } catch (const std::exception& e) {
            Rcpp::stop("cellrank2_cpp terminal upload: %s", e.what());
        }
    }

    singlet_gpu::fate::CellRank2Config cfg;
    cfg.max_iter = max_iter;
    cfg.tol      = static_cast<float>(tol);
    cfg.seed     = static_cast<uint64_t>(seed);
    cfg.use_gpu  = (resource != "cpu");

    singlet_gpu::fate::CellRank2Result res;
    try {
        res = singlet_gpu::fate::run_cellrank2(d_T, d_term, n_cells, n_term, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("cellrank2_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const std::size_t abs_sz = static_cast<std::size_t>(n_cells) * n_term;
    std::vector<float> h_abs(abs_sz > 0 ? abs_sz : 1);
    if (abs_sz > 0) {
        cudaError_t err = cudaMemcpy(h_abs.data(), res.absorption_prob.data(),
                                     abs_sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("cellrank2_cpp absorption copyback: %s", cudaGetErrorString(err));
    }

    Rcpp::NumericMatrix r_abs(n_cells, n_term);
    for (int c = 0; c < n_cells; ++c)
        for (int t = 0; t < n_term; ++t)
            r_abs(c, t) = static_cast<double>(
                h_abs[static_cast<std::size_t>(c) * n_term + t]);

    return Rcpp::List::create(
        Rcpp::Named("absorption_prob") = r_abs,
        Rcpp::Named("converged_all")   = res.converged_all,
        Rcpp::Named("n_terminals")     = n_term
    );
}

// ---------------------------------------------------------------------------
// palantir_cpp
//
// GPU diffusion pseudotime + branch probabilities via spectral decomposition.
// Wraps singlet_gpu::fate::run_palantir.
//
// Parameters:
//   expression              — dgCMatrix [genes × cells], log-normalized GEX
//   start_cell              — int, 0-based index of the root cell
//   n_terminals             — int, number of terminal states (0 = auto-detect)
//   n_neighbors             — int, kNN graph k (default 30)
//   n_diffusion_components  — int, spectral dimensions (default 10)
//   seed                    — int
//   resource                — "gpu" | "cpu" | "auto"
//
// Returns: Rcpp::List{
//   "pseudotime"     — NumericVector [n_cells]
//   "branch_probs"   — NumericMatrix [n_cells × n_terminals]
//   "terminal_cells" — IntegerVector (0-based indices detected/used)
//   "n_terminals"    — int
// }
// [[Rcpp::export]]
inline Rcpp::List palantir_cpp(
        Rcpp::S4 expression,
        int      start_cell,
        int      n_terminals,
        int      n_neighbors,
        int      n_diffusion_components,
        int      seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("palantir_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;
    (void)n_genes;  // used only for dim checking

    if (start_cell < 0 || start_cell >= n_cells)
        Rcpp::stop("palantir_cpp: start_cell %d out of range [0, %d).", start_cell, n_cells);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("palantir_cpp upload: %s", e.what()); }

    singlet_gpu::fate::PalantirConfig cfg;
    cfg.start_cell             = start_cell;
    cfg.n_terminals            = n_terminals;
    cfg.n_neighbors            = n_neighbors;
    cfg.n_diffusion_components = n_diffusion_components;
    cfg.seed                   = static_cast<uint64_t>(seed);
    cfg.use_gpu                = (resource != "cpu");

    singlet_gpu::fate::PalantirResult res;
    try {
        res = singlet_gpu::fate::run_palantir(d_expr, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("palantir_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int nterm = res.n_terminals;
    const std::size_t bp_sz = static_cast<std::size_t>(n_cells) * nterm;

    std::vector<float> h_pt(n_cells), h_bp(bp_sz > 0 ? bp_sz : 1);
    std::vector<int>   h_tc(nterm);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpy(h_pt.data(), res.pseudotime.data(),
                           n_cells * sizeof(float), cudaMemcpyDeviceToHost), "pseudotime");
            if (bp_sz > 0)
                chk(cudaMemcpy(h_bp.data(), res.branch_probs.data(),
                               bp_sz * sizeof(float), cudaMemcpyDeviceToHost), "branch_probs");
            if (nterm > 0)
                chk(cudaMemcpy(h_tc.data(), res.terminal_cells.data(),
                               nterm * sizeof(int), cudaMemcpyDeviceToHost), "terminal_cells");
        } catch (const std::exception& e) {
            Rcpp::stop("palantir_cpp copyback: %s", e.what());
        }
    }

    Rcpp::NumericVector r_pt(n_cells);
    for (int c = 0; c < n_cells; ++c) r_pt[c] = static_cast<double>(h_pt[c]);

    Rcpp::NumericMatrix r_bp(n_cells, nterm);
    for (int c = 0; c < n_cells; ++c)
        for (int t = 0; t < nterm; ++t)
            r_bp(c, t) = static_cast<double>(
                h_bp[static_cast<std::size_t>(c) * nterm + t]);

    Rcpp::IntegerVector r_tc(nterm);
    for (int t = 0; t < nterm; ++t) r_tc[t] = h_tc[t];

    return Rcpp::List::create(
        Rcpp::Named("pseudotime")     = r_pt,
        Rcpp::Named("branch_probs")   = r_bp,
        Rcpp::Named("terminal_cells") = r_tc,
        Rcpp::Named("n_terminals")    = nterm
    );
}

}  // namespace singlet_gpu_r

#endif  // FACTORNET_HAS_GPU
