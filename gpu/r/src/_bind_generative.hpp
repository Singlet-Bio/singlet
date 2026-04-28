// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_generative.hpp
//
// Rcpp bindings for generative model kernels (cycle 52b):
//   discrete_diffusion_train_cpp  — wraps singlet_gpu::generative::discrete_diffusion_train
//   discrete_diffusion_sample_cpp — wraps singlet_gpu::generative::discrete_diffusion_sample
//
// Discrete Diffusion:
//   Discrete-state diffusion model for scRNA-seq count generation.
//   API surface: train() + sample(n) — matches the DiffusionModel convention.
//   SOTA analog: scDiffCom (Python), scDiff, DiffVAE
//   R surface: train_discrete_diffusion() → DiscreteDiffusionModel externalptr
//              sample_discrete_diffusion(model, n) → dgCMatrix
//
// Host-copy overhead:
//   train:  O(nnz) host→device; O(model_params) device→host (stored in ptr).
//   sample: O(n × n_genes) device→host (sampled count matrix).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/generative/discrete_diffusion.h>
#include <singlet-gpu/generative/discrete_diffusion_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// discrete_diffusion_train_cpp
//
// Train a discrete diffusion generative model on count data.
// Analog of scDiff.train() / DiffVAE.fit().
//
// Parameters:
//   dgc       — dgCMatrix [genes × cells], training counts
//   n_latent  — int, latent dimension
//   n_steps   — int, number of diffusion steps (noise schedule length)
//   n_epochs  — int, training epochs
//   lr        — double, Adam learning rate
//   resource  — character "auto" (kept for API symmetry; always GPU)
//   seed      — int
//
// Returns: externalptr (DiscreteDiffusionModel*) — pass to sample_cpp()
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::XPtr<singlet_gpu::generative::DiscreteDiffusionModel>
discrete_diffusion_train_cpp(
        Rcpp::S4    dgc,
        int         n_latent,
        int         n_steps,
        int         n_epochs,
        double      lr,
        std::string resource,
        int         seed)
{
    (void)resource;  // always GPU; parameter kept for API symmetry with cycle 24 pattern

    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("discrete_diffusion_train_cpp: %s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("discrete_diffusion_train_cpp upload: %s", e.what()); }

    singlet_gpu::generative::DiscreteDiffusionConfig cfg;
    cfg.n_genes   = n_genes;
    cfg.n_cells   = n_cells;
    cfg.n_latent  = n_latent;
    cfg.n_steps   = n_steps;
    cfg.n_epochs  = n_epochs;
    cfg.lr        = static_cast<float>(lr);
    cfg.seed      = static_cast<uint64_t>(seed);

    auto* model = new singlet_gpu::generative::DiscreteDiffusionModel();
    try {
        *model = singlet_gpu::generative::discrete_diffusion_train(d, n_genes, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        delete model;
        Rcpp::stop("discrete_diffusion_train_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    return Rcpp::XPtr<singlet_gpu::generative::DiscreteDiffusionModel>(model, true);
}

// ---------------------------------------------------------------------------
// discrete_diffusion_sample_cpp
//
// Sample new cells from a trained discrete diffusion model.
// Analog of scDiff.generate(n) / DiffVAE.sample(n).
//
// Parameters:
//   model_ptr — externalptr from discrete_diffusion_train_cpp
//   n         — int, number of synthetic cells to generate
//   seed      — int
//
// Returns:
//   List{
//     "counts"     — Matrix::dgCMatrix [genes × n] (synthetic count matrix)
//     "gene_names" — CharacterVector [n_genes] (empty if model has no gene names)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List discrete_diffusion_sample_cpp(
        SEXP model_ptr,
        int  n,
        int  seed)
{
    Rcpp::XPtr<singlet_gpu::generative::DiscreteDiffusionModel> model(model_ptr);
    if (!model)
        Rcpp::stop("discrete_diffusion_sample_cpp: model_ptr is NULL.");
    if (n < 1)
        Rcpp::stop("discrete_diffusion_sample_cpp: n must be >= 1.");

    const int n_genes = model->n_genes;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::generative::DiscreteDiffusionSampleConfig scfg;
    scfg.n_samples = n;
    scfg.seed      = static_cast<uint64_t>(seed);

    singlet_gpu::generative::DiscreteDiffusionSampleResult res;
    try {
        res = singlet_gpu::generative::discrete_diffusion_sample(*model, scfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("discrete_diffusion_sample_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Sampled counts layout: CSC (indptr[n+1], indices[], data[] on device)
    // Copy back three arrays then build R dgCMatrix via sparseMatrix()
    const int nnz = res.nnz;
    std::vector<int>   h_indptr(n + 1);
    std::vector<int>   h_indices(nnz > 0 ? nnz : 1);
    std::vector<float> h_data  (nnz > 0 ? nnz : 1);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_indptr.data(),  res.indptr.data(),
                       (n + 1) * sizeof(int), cudaMemcpyDeviceToHost), "indptr");
        if (nnz > 0) {
            chk(cudaMemcpy(h_indices.data(), res.indices.data(),
                           nnz * sizeof(int),   cudaMemcpyDeviceToHost), "indices");
            chk(cudaMemcpy(h_data.data(),    res.data.data(),
                           nnz * sizeof(float), cudaMemcpyDeviceToHost), "data");
        }
    } catch (const std::exception& e) {
        Rcpp::stop("discrete_diffusion_sample_cpp copyback: %s", e.what());
    }

    // Build R dgCMatrix [n_genes × n]
    // We use sparseMatrix(i, j, x, dims) triplet form for clarity
    Rcpp::IntegerVector r_i(nnz), r_j(nnz);
    Rcpp::NumericVector r_x(nnz);
    for (int col = 0; col < n; ++col) {
        for (int pos = h_indptr[col]; pos < h_indptr[col + 1]; ++pos) {
            r_i[pos] = h_indices[pos] + 1;   // 1-based row index
            r_j[pos] = col + 1;              // 1-based col index
            r_x[pos] = static_cast<double>(h_data[pos]);
        }
    }
    // Call Matrix::sparseMatrix via Rcpp
    Rcpp::Function sparseMatrix = Rcpp::Environment::namespace_env("Matrix")["sparseMatrix"];
    Rcpp::S4 r_mat = sparseMatrix(
        Rcpp::Named("i")    = r_i,
        Rcpp::Named("j")    = r_j,
        Rcpp::Named("x")    = r_x,
        Rcpp::Named("dims") = Rcpp::IntegerVector::create(n_genes, n)
    );

    Rcpp::CharacterVector r_gene_names(n_genes);  // empty placeholder

    return Rcpp::List::create(
        Rcpp::Named("counts")     = r_mat,
        Rcpp::Named("gene_names") = r_gene_names
    );
}

}  // namespace singlet_gpu_r
