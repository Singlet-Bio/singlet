// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_perturbation.hpp
//
// Rcpp bindings for perturbation-response kernel (cycle 52b):
//   perturb_graph_fit_cpp     — wraps singlet::gpu::perturb::perturb_graph_fit
//   perturb_graph_predict_cpp — wraps singlet::gpu::perturb::perturb_graph_predict
//
// PerturbGraph:
//   Neural perturbation graph model for in-silico gene knockdown prediction.
//   S4 model class: PerturbGraphModel; fit() / predict() generics.
//   SOTA analog: scGen (Python/scVI), CellOracle (Python), GEARS (PyTorch)
//
// API summary:
//   model <- perturb_graph_fit(sce, perturb_key, seed = 0L, resource = "auto")
//   pred  <- perturb_graph_predict(model, sce, target_genes)
//
// Host-copy overhead:
//   fit:     O(nnz) host→device (expression) + O(n_cells) labels; model stored on device.
//   predict: O(nnz) host→device; O(n_cells × n_genes) device→host (predicted expression).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/perturb/perturb_graph.h>
#include <singlet/gpu/perturb/perturb_graph_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// perturb_graph_fit_cpp
//
// Train a PerturbGraph neural model on paired (control, perturbed) expression.
// Analog of scGen.train() / GEARS.train().
//
// Parameters:
//   dgc            — dgCMatrix [genes × cells], combined control + perturbed counts
//   perturb_labels — IntegerVector [n_cells], 0 = control, 1..K = perturbation IDs
//   n_latent       — int, encoder latent dimension
//   n_epochs       — int
//   lr             — double
//   resource       — character "auto" (API symmetry; always GPU)
//   seed           — int
//
// Returns: externalptr (PerturbGraphModel*)
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::XPtr<singlet::gpu::perturb::PerturbGraphModel>
perturb_graph_fit_cpp(
        Rcpp::S4            dgc,
        Rcpp::IntegerVector perturb_labels,
        int                 n_latent,
        int                 n_epochs,
        double              lr,
        std::string         resource,
        int                 seed)
{
    (void)resource;  // always GPU

    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("perturb_graph_fit_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;
    if (perturb_labels.size() != n_cells)
        Rcpp::stop("perturb_graph_fit_cpp: perturb_labels length (%d) != n_cells (%d).",
                   (int)perturb_labels.size(), n_cells);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("perturb_graph_fit_cpp upload: %s", e.what()); }

    std::vector<int> h_labels(n_cells);
    for (int i = 0; i < n_cells; ++i) h_labels[i] = perturb_labels[i];
    singlet::gpu::core::DeviceMemory<int> d_labels(n_cells);
    cudaMemcpyAsync(d_labels.get(), h_labels.data(), n_cells * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    int n_perturb = *std::max_element(h_labels.begin(), h_labels.end());  // max label = n_perturbations

    singlet::gpu::perturb::PerturbGraphConfig cfg;
    cfg.n_latent  = n_latent;
    cfg.n_perturb = n_perturb;
    cfg.n_epochs  = n_epochs;
    cfg.lr        = static_cast<float>(lr);
    cfg.seed      = static_cast<uint64_t>(seed);

    auto* model = new singlet::gpu::perturb::PerturbGraphModel();
    try {
        *model = singlet::gpu::perturb::perturb_graph_fit(d, d_labels, n_cells, cfg, stream);
    } catch (const std::exception& e) {
        delete model;
        Rcpp::stop("perturb_graph_fit_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    return Rcpp::XPtr<singlet::gpu::perturb::PerturbGraphModel>(model, true);
}

// ---------------------------------------------------------------------------
// perturb_graph_predict_cpp
//
// Predict the transcriptional response to in-silico perturbations.
// Analog of scGen.predict() / GEARS.predict().
//
// Parameters:
//   dgc           — dgCMatrix [genes × cells], control expression (no perturbation)
//   model_ptr     — externalptr from perturb_graph_fit_cpp
//   target_labels — IntegerVector, perturbation IDs to simulate (1-based, matching
//                   the perturb_labels used at fit time)
//
// Returns:
//   List{
//     "predicted"  — NumericMatrix [cells × n_genes] per-perturbation mean prediction
//     "delta"      — NumericMatrix [cells × n_genes] predicted delta (pred - ctrl)
//     "perturb_ids"— IntegerVector, the queried perturbation IDs
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List perturb_graph_predict_cpp(
        Rcpp::S4            dgc,
        SEXP                model_ptr,
        Rcpp::IntegerVector target_labels)
{
    Rcpp::XPtr<singlet::gpu::perturb::PerturbGraphModel> model(model_ptr);
    if (!model)
        Rcpp::stop("perturb_graph_predict_cpp: model_ptr is NULL.");

    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("perturb_graph_predict_cpp: %s", e.what()); }

    const int n_cells = h.n_cols;
    const int n_genes = h.n_rows;
    const int n_targets = static_cast<int>(target_labels.size());
    if (n_targets < 1)
        Rcpp::stop("perturb_graph_predict_cpp: target_labels must be non-empty.");

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("perturb_graph_predict_cpp upload: %s", e.what()); }

    std::vector<int> h_targets(n_targets);
    for (int i = 0; i < n_targets; ++i) h_targets[i] = target_labels[i];
    singlet::gpu::core::DeviceMemory<int> d_targets(n_targets);
    cudaMemcpyAsync(d_targets.get(), h_targets.data(), n_targets * sizeof(int),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    singlet::gpu::perturb::PerturbGraphPredictResult res;
    try {
        res = singlet::gpu::perturb::perturb_graph_predict(
            d, *model, d_targets, n_cells, n_targets, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("perturb_graph_predict_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // predicted and delta are both [n_genes × n_cells] col-major on device
    const std::size_t mat_sz = static_cast<std::size_t>(n_genes) * n_cells;
    std::vector<float> h_pred(mat_sz), h_delta(mat_sz);
    cudaMemcpy(h_pred.data(),  res.predicted.data(), mat_sz * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_delta.data(), res.delta.data(),     mat_sz * sizeof(float), cudaMemcpyDeviceToHost);

    // Return as [cells × genes] for R convention
    Rcpp::NumericMatrix r_pred (n_cells, n_genes);
    Rcpp::NumericMatrix r_delta(n_cells, n_genes);
    for (int g = 0; g < n_genes; ++g) {
        for (int c = 0; c < n_cells; ++c) {
            r_pred (c, g) = static_cast<double>(h_pred [static_cast<std::size_t>(g) * n_cells + c]);
            r_delta(c, g) = static_cast<double>(h_delta[static_cast<std::size_t>(g) * n_cells + c]);
        }
    }

    Rcpp::IntegerVector r_ids(n_targets);
    for (int i = 0; i < n_targets; ++i) r_ids[i] = h_targets[i];

    return Rcpp::List::create(
        Rcpp::Named("predicted")   = r_pred,
        Rcpp::Named("delta")       = r_delta,
        Rcpp::Named("perturb_ids") = r_ids
    );
}

}  // namespace singlet_gpu_r
