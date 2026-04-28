// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_abundance.hpp
//
// Rcpp binding for differential abundance kernel (cycle 47):
//   milo_cpp — wraps singlet_gpu::abundance::run_milo
//
// Input format:
//   expression   — dgCMatrix [genes × cells], log-normalized GEX
//   donor_ids    — IntegerVector [n_cells], 0-based donor label (from donor_assignments.tsv)
//   conditions   — IntegerVector [n_donors], 0-based condition per donor (binary)
//   n_neighbors  — int, kNN graph k (default 30)
//   n_nh_cells   — int, neighborhood sample size (default = round(n_cells * 0.1))
//   seed         — int
//   resource     — "gpu" | "cpu" | "auto"
//
// Host-copy overhead:
//   Input: O(nnz) host→device (expression).
//   Output: O(n_neighborhoods) D→H (log_fc, se, p_value, fdr) +
//           O(n_neighborhoods × n_donors) D→H (nh_counts).
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
inline Rcpp::List milo_cpp(
        Rcpp::S4            /*expression*/,
        Rcpp::IntegerVector /*donor_ids*/,
        Rcpp::IntegerVector /*conditions*/,
        int    /*n_neighbors*/,
        int    /*n_nh_cells*/,
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
#include <singlet-gpu/abundance/milo.h>

#include <cuda_runtime.h>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// milo_cpp
//
// GPU differential abundance testing via kNN neighborhoods.
// Wraps singlet_gpu::abundance::run_milo.
//
// Returns: Rcpp::List{
//   "log_fc"      — NumericVector [n_neighborhoods]
//   "se"          — NumericVector [n_neighborhoods]
//   "p_value"     — NumericVector [n_neighborhoods]
//   "fdr"         — NumericVector [n_neighborhoods] (graph-weighted BH)
//   "nh_counts"   — NumericMatrix [n_neighborhoods × n_donors]
//   "n_neighborhoods" — int
// }
// [[Rcpp::export]]
inline Rcpp::List milo_cpp(
        Rcpp::S4            expression,
        Rcpp::IntegerVector donor_ids,
        Rcpp::IntegerVector conditions,
        int         n_neighbors,
        int         n_nh_cells,
        int         seed,
        std::string resource)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(expression); }
    catch (const std::exception& e) { Rcpp::stop("milo_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;
    (void)n_genes;

    if ((int)donor_ids.size() != n_cells)
        Rcpp::stop("milo_cpp: donor_ids length (%d) != n_cells (%d).",
                   (int)donor_ids.size(), n_cells);

    int max_donor = *std::max_element(donor_ids.begin(), donor_ids.end());
    const int n_donors = max_donor + 1;

    if ((int)conditions.size() != n_donors)
        Rcpp::stop("milo_cpp: conditions length (%d) != n_donors (%d).",
                   (int)conditions.size(), n_donors);

    if (n_nh_cells <= 0) n_nh_cells = std::max(10, n_cells / 10);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_expr;
    try { d_expr = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("milo_cpp upload: %s", e.what()); }

    // Upload donor_ids and conditions
    std::vector<int> h_don(donor_ids.begin(),  donor_ids.end());
    std::vector<int> h_cnd(conditions.begin(), conditions.end());

    singlet_gpu::core::DeviceMemory<int> d_don(n_cells), d_cnd(n_donors);
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_don.get(), h_don.data(),
                                n_cells  * sizeof(int), cudaMemcpyHostToDevice, stream), "donor_ids");
            chk(cudaMemcpyAsync(d_cnd.data(), h_cnd.data(),
                                n_donors * sizeof(int), cudaMemcpyHostToDevice, stream), "conditions");
            chk(cudaStreamSynchronize(stream), "sync");
        } catch (const std::exception& e) {
            Rcpp::stop("milo_cpp ancillary upload: %s", e.what());
        }
    }

    singlet_gpu::abundance::MiloConfig cfg;
    cfg.n_neighbors = n_neighbors;
    cfg.n_nh_cells  = n_nh_cells;
    cfg.n_donors    = n_donors;
    cfg.seed        = static_cast<uint64_t>(seed);
    cfg.use_gpu     = (resource != "cpu");

    singlet_gpu::abundance::MiloResult res;
    try {
        res = singlet_gpu::abundance::run_milo(
            d_expr, d_don, d_cnd, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("milo_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_nh = res.n_neighborhoods;
    const std::size_t cnt_sz = static_cast<std::size_t>(n_nh) * n_donors;

    std::vector<float> h_lfc(n_nh), h_se(n_nh), h_pv(n_nh), h_fdr(n_nh);
    std::vector<float> h_cnt(cnt_sz > 0 ? cnt_sz : 1);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpy(h_lfc.data(), res.log_fc.data(),  n_nh * sizeof(float), cudaMemcpyDeviceToHost), "log_fc");
            chk(cudaMemcpy(h_se .data(), res.se.data(),      n_nh * sizeof(float), cudaMemcpyDeviceToHost), "se");
            chk(cudaMemcpy(h_pv .data(), res.p_value.data(), n_nh * sizeof(float), cudaMemcpyDeviceToHost), "p_value");
            chk(cudaMemcpy(h_fdr.data(), res.fdr.data(),     n_nh * sizeof(float), cudaMemcpyDeviceToHost), "fdr");
            if (cnt_sz > 0)
                chk(cudaMemcpy(h_cnt.data(), res.nh_counts.data(),
                               cnt_sz * sizeof(float), cudaMemcpyDeviceToHost), "nh_counts");
        } catch (const std::exception& e) {
            Rcpp::stop("milo_cpp copyback: %s", e.what());
        }
    }

    Rcpp::NumericVector r_lfc(n_nh), r_se(n_nh), r_pv(n_nh), r_fdr(n_nh);
    for (int i = 0; i < n_nh; ++i) {
        r_lfc[i] = static_cast<double>(h_lfc[i]);
        r_se [i] = static_cast<double>(h_se [i]);
        r_pv [i] = static_cast<double>(h_pv [i]);
        r_fdr[i] = static_cast<double>(h_fdr[i]);
    }

    Rcpp::NumericMatrix r_cnt(n_nh, n_donors);
    for (int n = 0; n < n_nh; ++n)
        for (int d = 0; d < n_donors; ++d)
            r_cnt(n, d) = static_cast<double>(
                h_cnt[static_cast<std::size_t>(n) * n_donors + d]);

    return Rcpp::List::create(
        Rcpp::Named("log_fc")          = r_lfc,
        Rcpp::Named("se")              = r_se,
        Rcpp::Named("p_value")         = r_pv,
        Rcpp::Named("fdr")             = r_fdr,
        Rcpp::Named("nh_counts")       = r_cnt,
        Rcpp::Named("n_neighborhoods") = n_nh
    );
}

}  // namespace singlet_gpu_r

#endif  // FACTORNET_HAS_GPU
