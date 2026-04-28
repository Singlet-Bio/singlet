// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_velocity.hpp
//
// Rcpp bindings for cycle 15 RNA-velocity kernels:
//   velocity_moments_cpp       — wraps singlet_gpu::velocity::compute_moments
//   velocity_prep_compute_cpp  — wraps singlet_gpu::velocity::velocity_prep_compute
//
// Input format:
//   velocity_moments_cpp:
//     spliced    — dgCMatrix [genes × cells], assays(sce)$spliced
//     unspliced  — dgCMatrix [genes × cells], assays(sce)$unspliced
//     knn_indices — IntegerMatrix [cells × k], from neighbors_cpp
//     n_neighbors — int, k to use (may be <= knn_indices ncol)
//
//   velocity_prep_compute_cpp:
//     spliced_moments   — NumericMatrix [genes × cells] (Ms from moments)
//     unspliced_moments — NumericMatrix [genes × cells] (Mu from moments)
//     min_S_count       — int, minimum total spliced count to keep gene
//     min_U_count       — int, minimum total unspliced count to keep gene
//     top_n_quantile    — int, top N% quantile of velocity genes to retain
//     compute_velocity  — bool, also compute velocity (S - gamma*U)
//
// Host-copy overhead:
//   velocity_moments_cpp:  2 × O(nnz) host→device (sparse matrices) +
//                          O(n_cells × k) host→device (kNN) +
//                          2 × O(n_genes × n_cells) device→host (moments).
//   velocity_prep_compute_cpp:
//                          2 × O(n_genes × n_cells) host→device (moments) +
//                          O(n_genes × n_cells) device→host (velocity) +
//                          2 × O(n_genes) device→host (gamma, filter).
//
// Design note: spliced/unspliced come from singlify's exon_counts.1pz /
// intron_counts.1pz respectively.  The R wrapper (velocity.R) provides a
// read_pz_velocity_sce() helper that loads these assays from a singlify
// output directory.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/preprocess/velocity_prep.h>
#include <singlet-gpu/preprocess/velocity_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device
#include "_bind_graph.hpp"        // r_matrix_to_device_dense

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// dense_device_to_r_matrix
//
// Copy a device-resident flat float buffer (n_rows × n_cols, row-major) to
// an R NumericMatrix [n_rows × n_cols] (col-major).
// ---------------------------------------------------------------------------
inline Rcpp::NumericMatrix
dense_device_to_r_matrix(const singlet_gpu::core::DeviceMemory<float>& d_buf,
                          int n_rows, int n_cols)
{
    const std::size_t sz = static_cast<std::size_t>(n_rows) * n_cols;
    std::vector<float> h(sz);
    cudaError_t err = cudaMemcpy(h.data(), d_buf.data(),
                                 sz * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(std::string("dense_device_to_r_matrix: ") +
                                 cudaGetErrorString(err));

    Rcpp::NumericMatrix out(n_rows, n_cols);
    for (int r = 0; r < n_rows; ++r)
        for (int c = 0; c < n_cols; ++c)
            out(r, c) = static_cast<double>(h[static_cast<std::size_t>(r) * n_cols + c]);
    return out;
}

// ---------------------------------------------------------------------------
// velocity_moments_cpp
//
// Computes first-order moments (smoothed spliced Ms + smoothed unspliced Mu)
// by averaging each cell's counts over its k nearest neighbors.
// Analog of scVelo's compute_moments().
//
// Parameters:
//   spliced      — dgCMatrix [genes × cells]
//   unspliced    — dgCMatrix [genes × cells]
//   knn_indices  — IntegerMatrix [cells × k] (0-based, from neighbors_cpp)
//   n_neighbors  — int, k to use for smoothing
//
// Returns:
//   Rcpp::List{
//     "Ms" — NumericMatrix [genes × cells] (smoothed spliced)
//     "Mu" — NumericMatrix [genes × cells] (smoothed unspliced)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List velocity_moments_cpp(
        Rcpp::S4            spliced,
        Rcpp::S4            unspliced,
        Rcpp::IntegerMatrix knn_indices,
        int                 n_neighbors)
{
    HostCsc h_s, h_u;
    try { h_s = r_sparse_to_host_csc(spliced);   }
    catch (const std::exception& e) { Rcpp::stop("velocity_moments_cpp spliced: %s", e.what()); }
    try { h_u = r_sparse_to_host_csc(unspliced); }
    catch (const std::exception& e) { Rcpp::stop("velocity_moments_cpp unspliced: %s", e.what()); }

    const int n_genes = h_s.n_rows;
    const int n_cells = h_s.n_cols;

    if (h_u.n_rows != n_genes || h_u.n_cols != n_cells)
        Rcpp::stop("velocity_moments_cpp: spliced/unspliced dimensions do not match.");
    if (knn_indices.nrow() != n_cells)
        Rcpp::stop("velocity_moments_cpp: knn_indices rows (%d) != n_cells (%d).",
                   knn_indices.nrow(), n_cells);
    if (n_neighbors < 1 || n_neighbors > knn_indices.ncol())
        Rcpp::stop("velocity_moments_cpp: n_neighbors must be in [1, %d].", knn_indices.ncol());

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_s, d_u;
    try { d_s = upload_host_csc_to_device(h_s, stream); }
    catch (const std::exception& e) { Rcpp::stop("velocity_moments_cpp upload spliced: %s", e.what()); }
    try { d_u = upload_host_csc_to_device(h_u, stream); }
    catch (const std::exception& e) { Rcpp::stop("velocity_moments_cpp upload unspliced: %s", e.what()); }

    // Upload kNN indices (truncate to n_neighbors columns)
    const std::size_t nk = static_cast<std::size_t>(n_cells) * n_neighbors;
    std::vector<int> h_knn(nk);
    for (int i = 0; i < n_cells; ++i)
        for (int j = 0; j < n_neighbors; ++j)
            h_knn[static_cast<std::size_t>(i) * n_neighbors + j] = knn_indices(i, j);

    singlet_gpu::core::DeviceMemory<int> d_knn(nk);
    {
        cudaError_t err = cudaMemcpyAsync(d_knn.get(), h_knn.data(),
                                          nk * sizeof(int), cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess)
            Rcpp::stop("velocity_moments_cpp upload knn: %s", cudaGetErrorString(err));
        cudaStreamSynchronize(stream);
    }

    // Run moments kernel
    singlet_gpu::velocity::MomentsConfig cfg;
    cfg.n_neighbors = n_neighbors;

    singlet_gpu::velocity::MomentsResult res;
    try {
        res = singlet_gpu::velocity::compute_moments(d_s, d_u, d_knn, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("velocity_moments_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Copy Ms and Mu device → host
    Rcpp::NumericMatrix r_Ms, r_Mu;
    try {
        r_Ms = dense_device_to_r_matrix(res.Ms, n_genes, n_cells);
        r_Mu = dense_device_to_r_matrix(res.Mu, n_genes, n_cells);
    } catch (const std::exception& e) {
        Rcpp::stop("velocity_moments_cpp copyback: %s", e.what());
    }

    return Rcpp::List::create(
        Rcpp::Named("Ms") = r_Ms,
        Rcpp::Named("Mu") = r_Mu
    );
}

// ---------------------------------------------------------------------------
// velocity_prep_compute_cpp
//
// Estimates per-gene velocity parameters (gamma, offset) via linear regression
// of unspliced on spliced across cells, then optionally computes the velocity
// vector v = dS/dt ≈ S - gamma * U.
// Analog of scVelo velocity() + get_velocity().
//
// Parameters:
//   spliced_moments   — NumericMatrix [genes × cells] (Ms)
//   unspliced_moments — NumericMatrix [genes × cells] (Mu)
//   min_S_count       — int, minimum total spliced count to retain gene
//   min_U_count       — int, minimum total unspliced count to retain gene
//   top_n_quantile    — int, top N% velocity-likelihood quantile to keep
//   compute_velocity  — bool, compute velocity matrix (S - gamma*U)
//
// Returns:
//   Rcpp::List{
//     "velocity"         — NumericMatrix [genes × cells] (if compute_velocity)
//     "velocity_gamma"   — NumericVector [genes]
//     "velocity_filter"  — LogicalVector [genes] (TRUE = passes QC)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List velocity_prep_compute_cpp(
        Rcpp::NumericMatrix spliced_moments,
        Rcpp::NumericMatrix unspliced_moments,
        int                 min_S_count,
        int                 min_U_count,
        int                 top_n_quantile,
        bool                compute_velocity)
{
    const int n_genes = spliced_moments.nrow();
    const int n_cells = spliced_moments.ncol();

    if (n_genes <= 0 || n_cells <= 0)
        Rcpp::stop("velocity_prep_compute_cpp: empty moments matrices.");
    if (unspliced_moments.nrow() != n_genes || unspliced_moments.ncol() != n_cells)
        Rcpp::stop("velocity_prep_compute_cpp: Ms/Mu dimension mismatch.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceDense d_Ms, d_Mu;
    try { d_Ms = r_matrix_to_device_dense(spliced_moments,   stream); }
    catch (const std::exception& e) { Rcpp::stop("velocity_prep_compute_cpp upload Ms: %s", e.what()); }
    try { d_Mu = r_matrix_to_device_dense(unspliced_moments, stream); }
    catch (const std::exception& e) { Rcpp::stop("velocity_prep_compute_cpp upload Mu: %s", e.what()); }

    singlet_gpu::velocity::VelocityPrepConfig cfg;
    cfg.min_S_count      = min_S_count;
    cfg.min_U_count      = min_U_count;
    cfg.top_n_quantile   = top_n_quantile;
    cfg.compute_velocity = compute_velocity;

    singlet_gpu::velocity::VelocityPrepResult res;
    try {
        res = singlet_gpu::velocity::velocity_prep_compute(d_Ms, d_Mu, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("velocity_prep_compute_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // gamma [n_genes]
    std::vector<float> h_gamma(n_genes);
    {
        cudaError_t err = cudaMemcpy(h_gamma.data(), res.gamma.data(),
                                     n_genes * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("velocity_prep_compute_cpp gamma copyback: %s", cudaGetErrorString(err));
    }

    // filter [n_genes] stored as uint8 device buffer (1=pass)
    std::vector<uint8_t> h_filter_raw(n_genes);
    {
        cudaError_t err = cudaMemcpy(h_filter_raw.data(), res.velocity_filter.data(),
                                     n_genes * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("velocity_prep_compute_cpp filter copyback: %s", cudaGetErrorString(err));
    }

    Rcpp::NumericVector r_gamma(n_genes);
    Rcpp::LogicalVector r_filter(n_genes);
    for (int g = 0; g < n_genes; ++g) {
        r_gamma [g] = static_cast<double>(h_gamma[g]);
        r_filter[g] = (h_filter_raw[g] != 0);
    }

    Rcpp::List out = Rcpp::List::create(
        Rcpp::Named("velocity_gamma")  = r_gamma,
        Rcpp::Named("velocity_filter") = r_filter
    );

    if (compute_velocity) {
        Rcpp::NumericMatrix r_vel;
        try { r_vel = dense_device_to_r_matrix(res.velocity, n_genes, n_cells); }
        catch (const std::exception& e) { Rcpp::stop("velocity_prep_compute_cpp vel copyback: %s", e.what()); }
        out["velocity"] = r_vel;
    }

    return out;
}

}  // namespace singlet_gpu_r
