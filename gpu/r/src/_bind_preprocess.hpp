// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_preprocess.hpp
//
// Rcpp bindings for preprocessing kernels:
//   lognorm_cpp  — wraps singlet_gpu::preprocess::log_normalize
//   hvg_cpp      — wraps singlet_gpu::preprocess::select_hvg
//
// Both follow the same pattern:
//   1. Validate the S4 dgCMatrix input.
//   2. Marshal R → HostCsc (pageable host copy).
//   3. Upload HostCsc to device via cudaMemcpyAsync + synchronize.
//   4. Call the singlet-gpu kernel on device.
//   5. Copy result back to host (O(nnz) for modified matrix, O(genes) for scores).
//   6. Build and return the R result list.
//
// Host-copy overhead:
//   For a 10k-cell 30M-nnz sample: ~100–200 ms per call (PCIe 16x Gen3).
//   Document this in R-side wrappers so users know the cost.
//
// Streams: we create a temporary CUDA stream per call and destroy it after
//   synchronization.  This matches factornet's GPUContext pattern and avoids
//   global stream state that would be unsafe across R GC boundaries.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/preprocess/lognorm.h>
#include <singlet-gpu/preprocess/hvg.h>

#include "_r_to_eigen.hpp"

#include <cuda_runtime.h>
#include <factornet/gpu/types.cuh>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// upload_host_csc_to_device
//
// Allocate device CSC (via factornet DeviceMemory RAII), copy host→device
// async on `stream`, synchronize, return DeviceCSC.
//
// WHY synchronize here: the kernel call immediately follows.  The stream is
// also brand-new so no other work is in flight.
// ---------------------------------------------------------------------------
inline singlet_gpu::core::DeviceCSC upload_host_csc_to_device(
        const HostCsc& h,
        cudaStream_t   stream)
{
    using singlet_gpu::core::DeviceMemory;

    const std::size_t nnz_sz     = static_cast<std::size_t>(h.nnz);
    const std::size_t indptr_sz  = static_cast<std::size_t>(h.n_cols + 1);

    singlet_gpu::core::DeviceCSC d;
    d.rows = static_cast<int64_t>(h.n_rows);
    d.cols = static_cast<int64_t>(h.n_cols);
    d.nnz  = static_cast<int64_t>(h.nnz);

    // Allocate device buffers via RAII
    d.col_ptr    = DeviceMemory<int32_t>(indptr_sz);
    d.row_indices= DeviceMemory<int32_t>(nnz_sz);
    d.values     = DeviceMemory<float>  (nnz_sz);

    auto chk = [](cudaError_t e, const char* tag) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(tag) + ": " + cudaGetErrorString(e));
    };

    chk(cudaMemcpyAsync(d.col_ptr.data(),     h.indptr.data(),
                        indptr_sz * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
        "upload indptr");
    chk(cudaMemcpyAsync(d.row_indices.data(), h.indices.data(),
                        nnz_sz    * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
        "upload indices");
    chk(cudaMemcpyAsync(d.values.data(),      h.values.data(),
                        nnz_sz    * sizeof(float),   cudaMemcpyHostToDevice, stream),
        "upload values");

    chk(cudaStreamSynchronize(stream), "upload sync");
    return d;
}

// ---------------------------------------------------------------------------
// lognorm_cpp
//
// R-callable binding for log_normalize.
//
// Parameters:
//   dgc         — Matrix::dgCMatrix (counts layer)
//   target_sum  — library-size target; 0 = use median (recommended)
//
// Returns:
//   Rcpp::S4 dgCMatrix with log1p-normalized values.
//
// The input matrix is NOT modified in-place; a new S4 is returned.
// R/preprocess.R stores the result in assays(sce)$logcounts.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::S4 lognorm_cpp(Rcpp::S4 dgc, double target_sum) {
    // Marshal R → host CSC
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    // Create temporary stream
    cudaStream_t stream = nullptr;
    cudaError_t serr = cudaStreamCreate(&stream);
    if (serr != cudaSuccess)
        Rcpp::stop("lognorm_cpp: cudaStreamCreate: %s", cudaGetErrorString(serr));

    // RAII stream cleanup
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard() { if (s) cudaStreamDestroy(s); }
    } guard{stream};

    // Upload host → device
    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("lognorm_cpp upload: %s", e.what()); }

    // Configure and run kernel
    singlet_gpu::preprocess::LogNormConfig cfg;
    cfg.method        = singlet_gpu::preprocess::LogNormMethod::TotalCount;
    cfg.target_count  = static_cast<float>(target_sum);  // 0 → use median

    singlet_gpu::preprocess::LogNormResult res;
    try {
        res = singlet_gpu::preprocess::log_normalize(d, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("lognorm_cpp kernel: %s", e.what());
    }

    cudaStreamSynchronize(stream);

    // log_normalize modifies d.values in-place; copy back to host
    std::vector<float> h_vals(static_cast<std::size_t>(h.nnz));
    cudaError_t ce = cudaMemcpy(h_vals.data(), d.values.data(),
                                static_cast<std::size_t>(h.nnz) * sizeof(float),
                                cudaMemcpyDeviceToHost);
    if (ce != cudaSuccess)
        Rcpp::stop("lognorm_cpp copyback: %s", cudaGetErrorString(ce));

    // Build output dgCMatrix with updated values (same structure as input)
    return host_csc_to_r_sparse(
        h.indptr.data(), h.indices.data(), h_vals.data(),
        h.n_rows, h.n_cols);
}

// ---------------------------------------------------------------------------
// hvg_cpp
//
// R-callable binding for select_hvg.
//
// Parameters:
//   dgc    — Matrix::dgCMatrix (counts or logcounts layer)
//   n_top  — number of top HVGs to select
//   flavor — "seurat_v3" | "pearson_residuals"
//
// Returns:
//   Rcpp::List{
//     "indices"  — IntegerVector top_n gene indices (1-based for R)
//     "scores"   — NumericVector top_n variances
//     "mean"     — NumericVector per-gene mean (length n_genes)
//     "var"      — NumericVector per-gene variance (length n_genes)
//   }
//
// R/preprocess.R converts indices to row names and writes rowData columns.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List hvg_cpp(Rcpp::S4 dgc, int n_top, std::string flavor) {
    // Marshal R → host CSC
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload
    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("hvg_cpp upload: %s", e.what()); }

    // Configure flavor
    singlet_gpu::preprocess::HvgConfig cfg;
    cfg.top_n = n_top;
    if (flavor == "seurat_v3") {
        cfg.flavor = singlet_gpu::preprocess::HvgFlavor::SeuratV3;
    } else if (flavor == "pearson_residuals") {
        cfg.flavor = singlet_gpu::preprocess::HvgFlavor::PearsonResiduals;
    } else {
        Rcpp::stop("hvg_cpp: unknown flavor '%s'. Use 'seurat_v3' or "
                   "'pearson_residuals'.", flavor.c_str());
    }

    // Run kernel
    singlet_gpu::preprocess::HvgResult res;
    try {
        res = singlet_gpu::preprocess::select_hvg(d, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("hvg_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy results to host
    const int32_t n_genes  = h.n_rows;
    const int32_t top_n    = std::min(n_top, n_genes);

    std::vector<int32_t> h_idx  (static_cast<std::size_t>(top_n));
    std::vector<float>   h_scores(static_cast<std::size_t>(top_n));
    std::vector<float>   h_mean  (static_cast<std::size_t>(n_genes));
    std::vector<float>   h_var   (static_cast<std::size_t>(n_genes));

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_idx.data(),    res.indices.data(),
                       top_n * sizeof(int32_t), cudaMemcpyDeviceToHost), "hvg idx");
        chk(cudaMemcpy(h_scores.data(), res.scores.data(),
                       top_n * sizeof(float),   cudaMemcpyDeviceToHost), "hvg scores");
        chk(cudaMemcpy(h_mean.data(),   res.mean.data(),
                       n_genes * sizeof(float), cudaMemcpyDeviceToHost), "hvg mean");
        chk(cudaMemcpy(h_var.data(),    res.var.data(),
                       n_genes * sizeof(float), cudaMemcpyDeviceToHost), "hvg var");
    } catch (const std::exception& e) {
        Rcpp::stop("hvg_cpp copyback: %s", e.what());
    }

    // Convert 0-based C++ indices to 1-based R indices
    Rcpp::IntegerVector r_idx(top_n);
    for (int32_t k = 0; k < top_n; ++k) r_idx[k] = h_idx[k] + 1;

    // Convert float arrays to R numeric
    Rcpp::NumericVector r_scores(top_n), r_mean(n_genes), r_var(n_genes);
    for (int32_t k = 0; k < top_n;   ++k) r_scores[k] = static_cast<double>(h_scores[k]);
    for (int32_t k = 0; k < n_genes; ++k) r_mean[k]   = static_cast<double>(h_mean[k]);
    for (int32_t k = 0; k < n_genes; ++k) r_var[k]    = static_cast<double>(h_var[k]);

    return Rcpp::List::create(
        Rcpp::Named("indices") = r_idx,
        Rcpp::Named("scores")  = r_scores,
        Rcpp::Named("mean")    = r_mean,
        Rcpp::Named("var")     = r_var
    );
}

}  // namespace singlet_gpu_r
