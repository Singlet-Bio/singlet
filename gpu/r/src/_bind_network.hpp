// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_network.hpp
//
// Rcpp binding for gene co-expression network kernel (cycle 46):
//   hdwgcna_cpp — wraps singlet_gpu::network::run_hdwgcna
//
// Input format:
//   expression   — dgCMatrix [genes × cells], log-normalized GEX (HVG subset recommended)
//   soft_power   — int, soft-thresholding power beta (default 6)
//   min_module_size — int, minimum genes per module (default 30)
//   n_top_hub    — int, top hub genes per module to return (default 20)
//   seed         — int
//   resource     — "gpu" | "cpu" | "auto"
//
// Host-copy overhead:
//   Input: O(nnz) host→device (expression).
//   Output: O(n_genes) D→H (module_assignment) +
//           O(n_cells × n_modules) D→H (eigengenes) +
//           O(n_modules × n_top_hub) D→H (hub gene indices and kME scores).
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
inline Rcpp::List hdwgcna_cpp(
        Rcpp::S4 /*expression*/,
        int      /*soft_power*/,
        int      /*min_module_size*/,
        int      /*n_top_hub*/,
        int      /*seed*/,
        std::string /*resource*/)
{
    return Rcpp::List::create(Rcpp::Named("error") = "GPU not available");
}

}  // namespace singlet_gpu_r

#else  // FACTORNET_HAS_GPU defined

#define FACTORNET_HAS_GPU 1
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/network/hdwgcna.h>

#include <cuda_runtime.h>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// hdwgcna_cpp
//
// GPU weighted gene co-expression network analysis.
// Wraps singlet_gpu::network::run_hdwgcna.
//
// Returns: Rcpp::List{
//   "module_assignment" — IntegerVector [n_genes], 0-based module label (0 = unassigned)
//   "eigengenes"        — NumericMatrix [n_cells × n_modules]
//   "hub_genes"         — IntegerMatrix [n_modules × n_top_hub] (0-based gene indices)
//   "hub_kme"           — NumericMatrix  [n_modules × n_top_hub] (kME scores)
//   "n_modules"         — int
// }
// [[Rcpp::export]]
inline Rcpp::List hdwgcna_cpp(
        Rcpp::S4    expression,
        int         soft_power,
        int         min_module_size,
        int         n_top_hub,
        int         seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("hdwgcna_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if (n_genes < 2)
        Rcpp::stop("hdwgcna_cpp: need at least 2 genes.");
    if (soft_power < 1)
        Rcpp::stop("hdwgcna_cpp: soft_power must be >= 1.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("hdwgcna_cpp upload: %s", e.what()); }

    singlet_gpu::network::HdWGCNAConfig cfg;
    cfg.soft_power      = soft_power;
    cfg.min_module_size = min_module_size;
    cfg.n_top_hub       = n_top_hub;
    cfg.seed            = static_cast<uint64_t>(seed);
    cfg.use_gpu         = (resource != "cpu");

    singlet_gpu::network::HdWGCNAResult res;
    try {
        res = singlet_gpu::network::run_hdwgcna(d_expr, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("hdwgcna_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_modules = res.n_modules;
    const int ntop      = std::min(n_top_hub, res.max_hub_per_module);

    // Copy results device → host
    std::vector<int>   h_mod(n_genes);
    std::vector<float> h_eig(static_cast<std::size_t>(n_cells) * n_modules);
    std::vector<int>   h_hub(static_cast<std::size_t>(n_modules) * ntop);
    std::vector<float> h_kme(static_cast<std::size_t>(n_modules) * ntop);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpy(h_mod.data(), res.module_assignment.data(),
                           n_genes * sizeof(int), cudaMemcpyDeviceToHost), "module_assignment");
            if (!h_eig.empty())
                chk(cudaMemcpy(h_eig.data(), res.eigengenes.data(),
                               h_eig.size() * sizeof(float), cudaMemcpyDeviceToHost), "eigengenes");
            if (!h_hub.empty()) {
                chk(cudaMemcpy(h_hub.data(), res.hub_gene_indices.data(),
                               h_hub.size() * sizeof(int),   cudaMemcpyDeviceToHost), "hub_genes");
                chk(cudaMemcpy(h_kme.data(), res.hub_kme.data(),
                               h_kme.size() * sizeof(float), cudaMemcpyDeviceToHost), "hub_kme");
            }
        } catch (const std::exception& e) {
            Rcpp::stop("hdwgcna_cpp copyback: %s", e.what());
        }
    }

    Rcpp::IntegerVector r_mod(n_genes);
    for (int g = 0; g < n_genes; ++g) r_mod[g] = h_mod[g];

    Rcpp::NumericMatrix r_eig(n_cells, n_modules);
    for (int c = 0; c < n_cells; ++c)
        for (int m = 0; m < n_modules; ++m)
            r_eig(c, m) = static_cast<double>(
                h_eig[static_cast<std::size_t>(c) * n_modules + m]);

    Rcpp::IntegerMatrix r_hub(n_modules, ntop);
    Rcpp::NumericMatrix r_kme(n_modules, ntop);
    for (int m = 0; m < n_modules; ++m)
        for (int k = 0; k < ntop; ++k) {
            r_hub(m, k) = h_hub[static_cast<std::size_t>(m) * ntop + k];
            r_kme(m, k) = static_cast<double>(
                h_kme[static_cast<std::size_t>(m) * ntop + k]);
        }

    return Rcpp::List::create(
        Rcpp::Named("module_assignment") = r_mod,
        Rcpp::Named("eigengenes")        = r_eig,
        Rcpp::Named("hub_genes")         = r_hub,
        Rcpp::Named("hub_kme")           = r_kme,
        Rcpp::Named("n_modules")         = n_modules
    );
}

}  // namespace singlet_gpu_r

#endif  // FACTORNET_HAS_GPU
