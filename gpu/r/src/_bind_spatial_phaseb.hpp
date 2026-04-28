// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_spatial_phaseb.hpp
//
// Rcpp bindings for spatial Phase B kernels (cycle 52b):
//   flash_deconv_cpp — wraps singlet_gpu::spatial::flash_deconv
//   stagate_fit_cpp  — wraps singlet_gpu::spatial::stagate_fit (model stub)
//   cell2fate_fit_cpp — wraps singlet_gpu::spatial::cell2fate_fit
//   cell2fate_predict_cpp — wraps singlet_gpu::spatial::cell2fate_predict
//
// FlashDeconv:
//   Input: expression dgCMatrix [genes × cells/spots] + spatial coords matrix
//   Output: deconvolved cell-type proportion matrix [spots × n_types]
//   SOTA analog: RCTD, SPOTlight, cell2location
//
// STAGATE:
//   Graph attention autoencoder for spatial domain detection.
//   fit() returns a model externalptr; predict() applies it.
//   SOTA analog: STAGATE (Python/PyTorch), BayesSpace
//
// Cell2fate:
//   Neural ODE for fate prediction from velocity.
//   S4 model class: Cell2fateModel; fit() / predict() generics.
//   SOTA analog: CellRank, cell2fate (Python/PyTorch)
//
// Host-copy overhead per function documented inline.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/spatial/flash_deconv.h>
#include <singlet-gpu/spatial/flash_deconv_types.h>
#include <singlet-gpu/spatial/stagate.h>
#include <singlet-gpu/spatial/stagate_types.h>
#include <singlet-gpu/spatial/cell2fate.h>
#include <singlet-gpu/spatial/cell2fate_types.h>

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
// flash_deconv_cpp
//
// GPU FlashDeconv spatial deconvolution.
// Analog of RCTD::create_RCTD() + run.RCTD() or SPOTlight::SPOTlight().
//
// Parameters:
//   dgc          — dgCMatrix [genes × spots], raw spot counts
//   ref_dgc      — dgCMatrix [genes × ref_cells], reference cell-type matrix
//   ref_labels   — IntegerVector [n_ref_cells], 0-based cell-type IDs
//   coords       — NumericMatrix [spots × 2], spatial x/y coordinates
//   n_types      — int, number of cell types (max(ref_labels) + 1 if 0)
//   n_iter       — int, EM iterations (default 100)
//   tol          — double, convergence tolerance
//   seed         — int
//
// Returns:
//   List{
//     "props"         — NumericMatrix [spots × n_types] proportion matrix
//     "converged"     — LogicalVector [spots], per-spot convergence flag
//     "n_iter_actual" — int, median iterations to convergence
//   }
//
// Host-copy: O(nnz_spot + nnz_ref) host→device; O(spots × n_types) device→host.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List flash_deconv_cpp(
        Rcpp::S4             dgc,
        Rcpp::S4             ref_dgc,
        Rcpp::IntegerVector  ref_labels,
        Rcpp::NumericMatrix  coords,
        int                  n_types,
        int                  n_iter,
        double               tol,
        int                  seed)
{
    HostCsc h_spot, h_ref;
    try { h_spot = r_sparse_to_host_csc(dgc);     }
    catch (const std::exception& e) { Rcpp::stop("flash_deconv_cpp spot: %s", e.what()); }
    try { h_ref  = r_sparse_to_host_csc(ref_dgc); }
    catch (const std::exception& e) { Rcpp::stop("flash_deconv_cpp ref: %s", e.what()); }

    if (h_spot.n_rows != h_ref.n_rows)
        Rcpp::stop("flash_deconv_cpp: spot and ref gene dimensions must match (%d vs %d).",
                   h_spot.n_rows, h_ref.n_rows);

    const int n_spots   = h_spot.n_cols;
    const int n_ref     = h_ref.n_cols;
    const int n_types_  = (n_types <= 0)
                              ? (*std::max_element(ref_labels.begin(), ref_labels.end()) + 1)
                              : n_types;

    if (ref_labels.size() != n_ref)
        Rcpp::stop("flash_deconv_cpp: ref_labels length (%d) != n_ref_cells (%d).",
                   (int)ref_labels.size(), n_ref);
    if (coords.nrow() != n_spots || coords.ncol() != 2)
        Rcpp::stop("flash_deconv_cpp: coords must be [n_spots x 2].");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_spot, d_ref;
    try { d_spot = upload_host_csc_to_device(h_spot, stream); }
    catch (const std::exception& e) { Rcpp::stop("flash_deconv_cpp upload spot: %s", e.what()); }
    try { d_ref  = upload_host_csc_to_device(h_ref, stream); }
    catch (const std::exception& e) { Rcpp::stop("flash_deconv_cpp upload ref: %s", e.what()); }

    // Upload reference labels and spatial coords
    std::vector<int>   h_labels(n_ref);
    for (int i = 0; i < n_ref; ++i) h_labels[i] = ref_labels[i];
    singlet_gpu::core::DeviceMemory<int> d_labels(n_ref);
    cudaMemcpyAsync(d_labels.get(), h_labels.data(), n_ref * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    std::vector<float> h_coords(n_spots * 2);
    for (int i = 0; i < n_spots; ++i) {
        h_coords[i]           = static_cast<float>(coords(i, 0));
        h_coords[i + n_spots] = static_cast<float>(coords(i, 1));
    }
    singlet_gpu::core::DeviceMemory<float> d_coords(n_spots * 2);
    cudaMemcpyAsync(d_coords.get(), h_coords.data(), n_spots * 2 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    singlet_gpu::spatial::FlashDeconvConfig cfg;
    cfg.n_types = n_types_;
    cfg.n_iter  = n_iter;
    cfg.tol     = static_cast<float>(tol);
    cfg.seed    = static_cast<uint64_t>(seed);

    singlet_gpu::spatial::FlashDeconvResult res;
    try {
        res = singlet_gpu::spatial::flash_deconv(
            d_spot, d_ref, d_labels, d_coords, n_spots, n_ref, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("flash_deconv_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // props [spots × n_types], col-major on device → copy row-major for R matrix
    const std::size_t prop_sz = static_cast<std::size_t>(n_spots) * n_types_;
    std::vector<float> h_props(prop_sz);
    cudaMemcpy(h_props.data(), res.props.data(),
               prop_sz * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int> h_conv(n_spots);
    cudaMemcpy(h_conv.data(), res.converged.data(),
               n_spots * sizeof(int), cudaMemcpyDeviceToHost);

    int h_n_iter_actual = 0;
    cudaMemcpy(&h_n_iter_actual, res.n_iter_actual.data(),
               sizeof(int), cudaMemcpyDeviceToHost);

    Rcpp::NumericMatrix r_props(n_spots, n_types_);
    for (int t = 0; t < n_types_; ++t)
        for (int s = 0; s < n_spots; ++s)
            r_props(s, t) = static_cast<double>(
                h_props[static_cast<std::size_t>(t) * n_spots + s]);

    Rcpp::LogicalVector r_conv(n_spots);
    for (int i = 0; i < n_spots; ++i) r_conv[i] = (h_conv[i] != 0);

    return Rcpp::List::create(
        Rcpp::Named("props")         = r_props,
        Rcpp::Named("converged")     = r_conv,
        Rcpp::Named("n_iter_actual") = h_n_iter_actual
    );
}

// ---------------------------------------------------------------------------
// stagate_fit_cpp
//
// GPU STAGATE graph-attention autoencoder: fit phase.
// Returns an externalptr wrapping a StagateModel (device-resident weights).
// Analog of STAGATE.run_PCA() + STAGATE.run_STAGATE() in Python.
//
// Parameters:
//   dgc        — dgCMatrix [genes × cells], expression input
//   coords     — NumericMatrix [cells × 2], spatial coords for graph construction
//   n_latent   — int, latent dimension (default 50)
//   n_heads    — int, attention heads (default 1)
//   n_epochs   — int
//   lr         — double, Adam learning rate
//   seed       — int
//
// Returns: externalptr (StagateModel*) — pass to stagate_predict_cpp()
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::XPtr<singlet_gpu::spatial::StagateModel>
stagate_fit_cpp(
        Rcpp::S4            dgc,
        Rcpp::NumericMatrix coords,
        int                 n_latent,
        int                 n_heads,
        int                 n_epochs,
        double              lr,
        int                 seed)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("stagate_fit_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;
    if (coords.nrow() != n_cells || coords.ncol() != 2)
        Rcpp::stop("stagate_fit_cpp: coords must be [n_cells x 2].");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("stagate_fit_cpp upload: %s", e.what()); }

    std::vector<float> h_coords(n_cells * 2);
    for (int i = 0; i < n_cells; ++i) {
        h_coords[i]           = static_cast<float>(coords(i, 0));
        h_coords[i + n_cells] = static_cast<float>(coords(i, 1));
    }
    singlet_gpu::core::DeviceMemory<float> d_coords(n_cells * 2);
    cudaMemcpyAsync(d_coords.get(), h_coords.data(), n_cells * 2 * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    singlet_gpu::spatial::StagateConfig cfg;
    cfg.n_latent = n_latent;
    cfg.n_heads  = n_heads;
    cfg.n_epochs = n_epochs;
    cfg.lr       = static_cast<float>(lr);
    cfg.seed     = static_cast<uint64_t>(seed);

    auto* model = new singlet_gpu::spatial::StagateModel();
    try {
        *model = singlet_gpu::spatial::stagate_fit(d, d_coords, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        delete model;
        Rcpp::stop("stagate_fit_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // XPtr with finalizer to free device-resident model weights
    return Rcpp::XPtr<singlet_gpu::spatial::StagateModel>(model, true);
}

// ---------------------------------------------------------------------------
// stagate_predict_cpp
//
// Apply a fitted STAGATE model to produce a latent embedding.
//
// Parameters:
//   dgc        — dgCMatrix [genes × cells]
//   model_ptr  — externalptr from stagate_fit_cpp
//
// Returns:
//   NumericMatrix [cells × n_latent] — STAGATE latent embedding
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::NumericMatrix stagate_predict_cpp(
        Rcpp::S4 dgc,
        SEXP     model_ptr)
{
    Rcpp::XPtr<singlet_gpu::spatial::StagateModel> model(model_ptr);
    if (!model)
        Rcpp::stop("stagate_predict_cpp: model_ptr is NULL (model not fitted?).");

    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("stagate_predict_cpp: %s", e.what()); }

    const int n_cells   = h.n_cols;
    const int n_latent  = model->n_latent;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("stagate_predict_cpp upload: %s", e.what()); }

    singlet_gpu::spatial::StagatePredictResult res;
    try {
        res = singlet_gpu::spatial::stagate_predict(d, *model, n_cells, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("stagate_predict_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const std::size_t emb_sz = static_cast<std::size_t>(n_cells) * n_latent;
    std::vector<float> h_emb(emb_sz);
    cudaMemcpy(h_emb.data(), res.embedding.data(),
               emb_sz * sizeof(float), cudaMemcpyDeviceToHost);

    Rcpp::NumericMatrix r_emb(n_cells, n_latent);
    for (int d_idx = 0; d_idx < n_latent; ++d_idx)
        for (int c = 0; c < n_cells; ++c)
            r_emb(c, d_idx) = static_cast<double>(
                h_emb[static_cast<std::size_t>(d_idx) * n_cells + c]);

    return r_emb;
}

// ---------------------------------------------------------------------------
// cell2fate_fit_cpp
//
// GPU Cell2fate neural ODE: training phase.
// Analog of cell2fate.models.Cell2fate_DynamicalModel.train().
//
// Parameters:
//   spliced_dgc   — dgCMatrix [genes × cells]
//   unspliced_dgc — dgCMatrix [genes × cells]
//   n_latent      — int, latent state dimension
//   n_epochs      — int
//   lr            — double
//   resource      — character, "auto" | "cpu" (ignored; always GPU)
//   seed          — int
//
// Returns: externalptr (Cell2fateModel*) — pass to cell2fate_predict_cpp()
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::XPtr<singlet_gpu::spatial::Cell2fateModel>
cell2fate_fit_cpp(
        Rcpp::S4            spliced_dgc,
        Rcpp::S4            unspliced_dgc,
        int                 n_latent,
        int                 n_epochs,
        double              lr,
        std::string         resource,
        int                 seed)
{
    (void)resource;  // always GPU; parameter kept for API symmetry with cycle 24 pattern

    HostCsc h_s, h_u;
    try { h_s = r_sparse_to_host_csc(spliced_dgc);   }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_fit_cpp spliced: %s", e.what()); }
    try { h_u = r_sparse_to_host_csc(unspliced_dgc); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_fit_cpp unspliced: %s", e.what()); }

    if (h_s.n_rows != h_u.n_rows || h_s.n_cols != h_u.n_cols)
        Rcpp::stop("cell2fate_fit_cpp: spliced/unspliced dimension mismatch.");

    const int n_cells = h_s.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_s, d_u;
    try { d_s = upload_host_csc_to_device(h_s, stream); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_fit_cpp upload spliced: %s", e.what()); }
    try { d_u = upload_host_csc_to_device(h_u, stream); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_fit_cpp upload unspliced: %s", e.what()); }

    singlet_gpu::spatial::Cell2fateConfig cfg;
    cfg.n_latent = n_latent;
    cfg.n_epochs = n_epochs;
    cfg.lr       = static_cast<float>(lr);
    cfg.seed     = static_cast<uint64_t>(seed);

    auto* model = new singlet_gpu::spatial::Cell2fateModel();
    try {
        *model = singlet_gpu::spatial::cell2fate_fit(d_s, d_u, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        delete model;
        Rcpp::stop("cell2fate_fit_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    return Rcpp::XPtr<singlet_gpu::spatial::Cell2fateModel>(model, true);
}

// ---------------------------------------------------------------------------
// cell2fate_predict_cpp
//
// Apply a fitted Cell2fate model to predict cell fate probabilities.
//
// Parameters:
//   spliced_dgc   — dgCMatrix [genes × cells]
//   unspliced_dgc — dgCMatrix [genes × cells]
//   model_ptr     — externalptr from cell2fate_fit_cpp
//   n_states      — int, number of fate states to return
//
// Returns:
//   List{
//     "fate_probs"   — NumericMatrix [cells × n_states], fate probability per state
//     "latent_time"  — NumericVector [cells], pseudotime
//     "state_names"  — CharacterVector [n_states]
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List cell2fate_predict_cpp(
        Rcpp::S4 spliced_dgc,
        Rcpp::S4 unspliced_dgc,
        SEXP     model_ptr,
        int      n_states)
{
    Rcpp::XPtr<singlet_gpu::spatial::Cell2fateModel> model(model_ptr);
    if (!model)
        Rcpp::stop("cell2fate_predict_cpp: model_ptr is NULL.");

    HostCsc h_s, h_u;
    try { h_s = r_sparse_to_host_csc(spliced_dgc);   }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_predict_cpp spliced: %s", e.what()); }
    try { h_u = r_sparse_to_host_csc(unspliced_dgc); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_predict_cpp unspliced: %s", e.what()); }

    const int n_cells = h_s.n_cols;

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet_gpu::core::DeviceCSC d_s, d_u;
    try { d_s = upload_host_csc_to_device(h_s, stream); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_predict_cpp upload spliced: %s", e.what()); }
    try { d_u = upload_host_csc_to_device(h_u, stream); }
    catch (const std::exception& e) { Rcpp::stop("cell2fate_predict_cpp upload unspliced: %s", e.what()); }

    singlet_gpu::spatial::Cell2fatePredictConfig pcfg;
    pcfg.n_states = n_states;

    singlet_gpu::spatial::Cell2fatePredictResult res;
    try {
        res = singlet_gpu::spatial::cell2fate_predict(d_s, d_u, *model, n_cells, pcfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("cell2fate_predict_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int ns = res.n_states;
    const std::size_t fp_sz = static_cast<std::size_t>(n_cells) * ns;
    std::vector<float> h_fp(fp_sz);
    std::vector<float> h_lt(n_cells);

    cudaMemcpy(h_fp.data(), res.fate_probs.data(),
               fp_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_lt.data(), res.latent_time.data(),
               n_cells * sizeof(float), cudaMemcpyDeviceToHost);

    Rcpp::NumericMatrix r_fp(n_cells, ns);
    for (int st = 0; st < ns; ++st)
        for (int c = 0; c < n_cells; ++c)
            r_fp(c, st) = static_cast<double>(
                h_fp[static_cast<std::size_t>(st) * n_cells + c]);

    Rcpp::NumericVector r_lt(n_cells);
    for (int c = 0; c < n_cells; ++c) r_lt[c] = static_cast<double>(h_lt[c]);

    Rcpp::CharacterVector r_names(ns);
    for (int s = 0; s < ns; ++s)
        r_names[s] = "State" + std::to_string(s + 1);

    return Rcpp::List::create(
        Rcpp::Named("fate_probs")  = r_fp,
        Rcpp::Named("latent_time") = r_lt,
        Rcpp::Named("state_names") = r_names
    );
}

}  // namespace singlet_gpu_r
