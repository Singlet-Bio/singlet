// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_reduce.hpp
//
// Rcpp bindings for dimensionality reduction kernels:
//   pca_cpp  — wraps singlet_gpu::reduce::svd::auto_select
//   nmf_cpp  — wraps singlet_gpu::reduce::nmf::fit
//
// Both kernels take device CSC as input and return factornet CPU-resident
// Eigen DenseMatrix results (factornet downloads results from GPU before
// returning the result struct).
//
// PCA: factornet::SVDResult<float> fields:
//   .U  — Eigen col-major DenseMatrix (genes × k) left singular vectors
//   .V  — Eigen col-major DenseMatrix (cells × k) right singular vectors
//   .d  — Eigen DenseVector (k)        singular values
//   .k_selected — int                  components actually computed
//
// NMF: factornet::NMFResult<float> fields:
//   .W  — Eigen DenseMatrix (genes × k)
//   .H  — Eigen DenseMatrix (k × cells)
//   .d  — Eigen DenseVector (k)
//   .iterations — int
//   .converged  — bool
//
// No additional device→host copy is needed: factornet GPU adapters already
// download results into these Eigen structs.  The only copies here are the
// initial host→device upload (HostCsc → DeviceCSC) and the final
// Eigen→NumericMatrix conversion for R.
//
// Streams: one temporary stream per call for the host→device CSC upload.
// factornet's GPUContext creates its own internal stream for the kernel.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/auto_select.h>
#include <singlet-gpu/reduce/svd/types.h>
#include <singlet-gpu/reduce/nmf/fit.h>
#include <singlet-gpu/reduce/nmf/types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// eigen_dense_to_r_matrix
//
// Copy an Eigen col-major DenseMatrix<float> (m × k) into an R NumericMatrix.
// Eigen stores col-major (same as R), so we can memcpy each column in order.
// ---------------------------------------------------------------------------
template<typename EigenMat>
inline Rcpp::NumericMatrix eigen_dense_to_r_matrix(const EigenMat& M) {
    const int32_t n_rows = static_cast<int32_t>(M.rows());
    const int32_t n_cols = static_cast<int32_t>(M.cols());
    Rcpp::NumericMatrix out(n_rows, n_cols);
    for (int32_t j = 0; j < n_cols; ++j) {
        for (int32_t i = 0; i < n_rows; ++i) {
            out(i, j) = static_cast<double>(M(i, j));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// pca_cpp
//
// R-callable binding for SVD-based PCA.
//
// Parameters:
//   dgc          — Matrix::dgCMatrix (logcounts layer, genes × cells)
//   n_pcs        — number of principal components to compute
//   zero_center  — if TRUE, implicit mean-centering (A(v) = Xv − μ(1ᵀv))
//   seed         — RNG seed for factornet stochastic SVD backends
//
// Returns:
//   Rcpp::List{
//     "U"  — NumericMatrix [genes × n_pcs]  left singular vectors (gene loadings)
//     "V"  — NumericMatrix [cells × n_pcs]  right singular vectors (cell embeddings)
//     "d"  — NumericVector [n_pcs]          singular values
//     "k"  — int                            components actually computed
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List pca_cpp(Rcpp::S4 dgc, int n_pcs, bool zero_center, int seed) {
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    // Temporary stream for host→device upload
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard() { if (s) cudaStreamDestroy(s); }
    } guard{stream};

    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("pca_cpp upload: %s", e.what()); }

    // Build PzDeviceMatrix wrapper required by auto_select adapter
    singlet_gpu::io::PzDeviceMatrix pzdm;
    pzdm.mat             = std::move(d);
    pzdm.producer_stream = stream;
    // Metadata is empty for computation-only calls; that is fine.

    singlet_gpu::reduce::svd::SvdConfig cfg;
    cfg.k_max     = n_pcs;
    cfg.centering = zero_center;
    cfg.seed      = static_cast<uint64_t>(seed);

    singlet_gpu::reduce::svd::SvdResult res;
    try {
        res = singlet_gpu::reduce::svd::auto_select(pzdm, n_pcs, cfg);
    } catch (const std::exception& e) {
        Rcpp::stop("pca_cpp kernel: %s", e.what());
    }
    // factornet downloads results into res.U / res.V / res.d before returning.
    // No extra cudaMemcpy needed here.

    const int32_t k = static_cast<int32_t>(res.k_selected);

    // res.U: genes × k  (left singular vectors)
    // res.V: cells × k  (right singular vectors / cell embeddings)
    // res.d: k           (singular values)
    Rcpp::NumericMatrix r_U = eigen_dense_to_r_matrix(res.U);
    Rcpp::NumericMatrix r_V = eigen_dense_to_r_matrix(res.V);

    const int32_t d_size = static_cast<int32_t>(res.d.size());
    Rcpp::NumericVector r_d(d_size);
    for (int32_t i = 0; i < d_size; ++i) {
        r_d[i] = static_cast<double>(res.d(i));
    }

    return Rcpp::List::create(
        Rcpp::Named("U")  = r_U,
        Rcpp::Named("V")  = r_V,
        Rcpp::Named("d")  = r_d,
        Rcpp::Named("k")  = k
    );
}

// ---------------------------------------------------------------------------
// nmf_cpp
//
// R-callable binding for GPU NMF.
//
// Parameters:
//   dgc       — Matrix::dgCMatrix (logcounts layer, genes × cells)
//   n_factors — number of NMF factors k
//   loss      — "MSE" | "KL" | "NB" (Negative Binomial)
//   seed      — RNG seed for init
//
// Returns:
//   Rcpp::List{
//     "W"          — NumericMatrix [genes × k]     gene loadings
//     "H"          — NumericMatrix [k × cells]     cell usage
//     "d"          — NumericVector [k]              scale diagonal
//     "k"          — int                            factors computed
//     "iterations" — int                            iters until convergence
//     "converged"  — logical
//   }
//
// factornet::nmf::nmf_fit_gpu requires host-pinned CSC pointers.
// We store them in pzdm.h_indptr / h_indices / h_values (non-pinned vectors
// here — factornet internally allocates pinned staging if needed, or accepts
// pageable host memory with an implicit sync).
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List nmf_cpp(Rcpp::S4 dgc, int n_factors, std::string loss, int seed) {
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard {
        cudaStream_t s;
        ~StreamGuard() { if (s) cudaStreamDestroy(s); }
    } guard{stream};

    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("nmf_cpp upload: %s", e.what()); }

    // Build PzDeviceMatrix — the nmf::fit adapter reads host pinned buffers.
    // Allocate pinned host staging for indptr, indices, values and populate.
    // factornet::nmf::nmf_fit_gpu takes host pointers and stages to device
    // internally.  We use singlet_gpu::core::PinnedPool to get page-locked
    // memory so the async DMA inside factornet does not stall.
    const std::size_t indptr_bytes  = static_cast<std::size_t>(h.n_cols + 1)
                                      * sizeof(int32_t);
    const std::size_t indices_bytes = static_cast<std::size_t>(h.nnz) * sizeof(int32_t);
    const std::size_t values_bytes  = static_cast<std::size_t>(h.nnz) * sizeof(float);

    singlet_gpu::core::PinnedBuffer pb_indptr  = singlet_gpu::core::PinnedPool::acquire(indptr_bytes);
    singlet_gpu::core::PinnedBuffer pb_indices = singlet_gpu::core::PinnedPool::acquire(indices_bytes);
    singlet_gpu::core::PinnedBuffer pb_values  = singlet_gpu::core::PinnedPool::acquire(values_bytes);

    std::memcpy(pb_indptr.get(),  h.indptr.data(),  indptr_bytes);
    std::memcpy(pb_indices.get(), h.indices.data(), indices_bytes);
    std::memcpy(pb_values.get(),  h.values.data(),  values_bytes);

    singlet_gpu::io::PzDeviceMatrix pzdm;
    pzdm.mat             = std::move(d);
    pzdm.producer_stream = stream;
    pzdm.pinned_indptr   = std::move(pb_indptr);
    pzdm.pinned_indices  = std::move(pb_indices);
    pzdm.pinned_values   = std::move(pb_values);
    pzdm.host_retained   = true;

    // Expose raw pinned pointers to factornet via the shared_ptr host_* fields.
    // PzDeviceMatrix declares these as shared_ptr<int> / shared_ptr<float>.
    // The PinnedBuffers above keep the backing memory alive; we alias with
    // no-op deleters so the shared_ptr does not double-free.
    pzdm.host_indptr  = std::shared_ptr<int>(
        reinterpret_cast<int*>(pb_indptr.as<int32_t>()),  [](int*){});
    pzdm.host_indices = std::shared_ptr<int>(
        reinterpret_cast<int*>(pb_indices.as<int32_t>()), [](int*){});
    pzdm.host_values  = std::shared_ptr<float>(
        pb_values.as<float>(),    [](float*){});

    // Configure NMF
    singlet_gpu::reduce::nmf::NmfConfig cfg;
    cfg.k    = n_factors;
    cfg.seed = static_cast<uint64_t>(seed);

    // Map loss string to factornet LossType
    if (loss == "MSE" || loss == "mse") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::MSE;
    } else if (loss == "KL" || loss == "kl") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::KL;
    } else if (loss == "NB" || loss == "nb") {
        cfg.loss.type = singlet_gpu::reduce::nmf::LossType::NB;
    } else {
        Rcpp::stop("nmf_cpp: unknown loss '%s'. Use 'MSE', 'KL', or 'NB'.",
                   loss.c_str());
    }

    singlet_gpu::reduce::nmf::NmfResult res;
    try {
        res = singlet_gpu::reduce::nmf::fit(pzdm, cfg,
                                            /*W_init=*/nullptr,
                                            /*H_init=*/nullptr);
    } catch (const std::exception& e) {
        Rcpp::stop("nmf_cpp kernel: %s", e.what());
    }
    // factornet downloads W, H, d into res before returning.

    const int32_t k = static_cast<int32_t>(res.d.size());

    Rcpp::NumericMatrix r_W = eigen_dense_to_r_matrix(res.W);  // genes × k
    Rcpp::NumericMatrix r_H = eigen_dense_to_r_matrix(res.H);  // k × cells

    Rcpp::NumericVector r_d(k);
    for (int32_t i = 0; i < k; ++i) {
        r_d[i] = static_cast<double>(res.d(i));
    }

    return Rcpp::List::create(
        Rcpp::Named("W")          = r_W,
        Rcpp::Named("H")          = r_H,
        Rcpp::Named("d")          = r_d,
        Rcpp::Named("k")          = k,
        Rcpp::Named("iterations") = static_cast<int>(res.iterations),
        Rcpp::Named("converged")  = static_cast<bool>(res.converged)
    );
}

}  // namespace singlet_gpu_r
