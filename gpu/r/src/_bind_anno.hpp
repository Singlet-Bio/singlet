// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_anno.hpp
//
// Rcpp bindings for cycle 12 annotation kernels:
//   marker_score_cpp        — wraps singlet_gpu::anno::marker_score
//   celltypist_project_cpp  — wraps singlet_gpu::anno::project_to_reference
//   load_celltypist_model_cpp — wraps singlet_gpu::anno::load_celltypist_model
//                              (returns a model handle stored in R as an external pointer)
//
// Gene-set input format (R side):
//   gene_sets — named R List of character vectors: list(set_name = c("GENE1", ...), ...)
//   gene_names — character vector of all gene names in the matrix (rownames(sce))
//   The binding resolves names → 0-based indices to build GeneSetDB.
//
// CellTypist input:
//   embedding — NumericMatrix [cells × n_pcs] (from reducedDim(sce, use_dimred))
//   model_ptr — externalptr to CelltypistModel (from load_celltypist_model_cpp)
//
// Host-copy overhead:
//   marker_score_cpp:  O(nnz) matrix + O(n_sets × total_members) gene-set data.
//   celltypist_project_cpp: O(n_cells × n_pcs) embedding + O(n_cells) labels.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/anno/types.h>
#include <singlet-gpu/anno/marker_score.h>
#include <singlet-gpu/anno/reference_map.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device
#include "_bind_graph.hpp"        // r_matrix_to_device_dense

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// build_gene_set_db
//
// Convert an R named List of character vectors (gene set members) + a
// character vector of all gene names into a GeneSetDB.
//
// Gene names not found in gene_name_map are silently skipped (same behavior
// as DecoupleR / scran::fgsea when a gene is not in the expression matrix).
// ---------------------------------------------------------------------------
inline singlet_gpu::anno::GeneSetDB
build_gene_set_db(const Rcpp::List&            gene_sets,
                  const Rcpp::CharacterVector& gene_names)
{
    // Build name → 0-based index map
    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(static_cast<std::size_t>(gene_names.size()));
    for (int i = 0; i < gene_names.size(); ++i) {
        name_to_idx[Rcpp::as<std::string>(gene_names[i])] = i;
    }

    Rcpp::CharacterVector set_names = gene_sets.names();
    if (set_names.size() == 0)
        throw std::runtime_error("build_gene_set_db: gene_sets must be a named list");

    singlet_gpu::anno::GeneSetDB db;
    const int n_sets = static_cast<int>(gene_sets.size());
    db.set_names.resize(n_sets);
    db.member_gene_indices.resize(n_sets);
    db.weights.resize(n_sets);  // empty → unit weights (marker_score.h handles this)

    for (int s = 0; s < n_sets; ++s) {
        db.set_names[s] = Rcpp::as<std::string>(set_names[s]);
        Rcpp::CharacterVector members = gene_sets[s];
        for (int g = 0; g < members.size(); ++g) {
            auto it = name_to_idx.find(Rcpp::as<std::string>(members[g]));
            if (it != name_to_idx.end()) {
                db.member_gene_indices[s].push_back(it->second);
            }
        }
    }
    return db;
}

// ---------------------------------------------------------------------------
// CelltypistModelHandle
//
// RAII wrapper around CelltypistModel stored as an R external pointer.
// The R-visible type is "CelltypistModel" (externalptr tag).
// ---------------------------------------------------------------------------
struct CelltypistModelHandle {
    singlet_gpu::anno::CelltypistModel model;
    explicit CelltypistModelHandle(singlet_gpu::anno::CelltypistModel m)
        : model(std::move(m)) {}
};

inline void celltypist_finalizer(SEXP ptr) {
    CelltypistModelHandle* h = reinterpret_cast<CelltypistModelHandle*>(
        R_ExternalPtrAddr(ptr));
    if (h) {
        delete h;
        R_SetExternalPtrAddr(ptr, nullptr);
    }
}

// ---------------------------------------------------------------------------
// load_celltypist_model_cpp
//
// Loads a CellTypist .npz model from disk and stores it in device memory.
// Returns an R externalptr (class "CelltypistModel") that celltypist_project_cpp
// accepts.  The model is freed when R GC collects the external pointer.
//
// Parameters:
//   npz_path — character, path to the .npz model file
//
// Returns:
//   R externalptr with class attribute "CelltypistModel"
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline SEXP load_celltypist_model_cpp(std::string npz_path)
{
    singlet_gpu::anno::CelltypistModel model;
    try {
        model = singlet_gpu::anno::load_celltypist_model(npz_path);
    } catch (const std::exception& e) {
        Rcpp::stop("load_celltypist_model_cpp: %s", e.what());
    }

    CelltypistModelHandle* h = new CelltypistModelHandle(std::move(model));
    SEXP ptr = R_MakeExternalPtr(reinterpret_cast<void*>(h),
                                 Rf_install("CelltypistModel"), R_NilValue);
    R_RegisterCFinalizerEx(ptr, celltypist_finalizer, TRUE);

    // Set class attribute so R can dispatch methods
    Rcpp::Shield<SEXP> cls(Rf_allocVector(STRSXP, 1));
    SET_STRING_ELT(cls, 0, Rf_mkChar("CelltypistModel"));
    Rf_classgets(ptr, cls);

    return ptr;
}

// ---------------------------------------------------------------------------
// marker_score_cpp
//
// R-callable binding for GPU marker scoring.
//
// Parameters:
//   dgc        — Matrix::dgCMatrix [genes × cells] (logcounts)
//   gene_sets  — named R List of character vectors
//   gene_names — CharacterVector (rownames of the matrix)
//   method     — "mlm" | "ulm" | "wsum" | "ucell"
//
// Returns:
//   Rcpp::List{
//     "scores"    — NumericMatrix [n_cells × n_sets] (transposed for R convention)
//     "set_names" — CharacterVector
//   }
//
// R side (markers.R) writes one colData column per set.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List marker_score_cpp(
        Rcpp::S4           dgc,
        Rcpp::List         gene_sets,
        Rcpp::CharacterVector gene_names,
        std::string        method)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    // Build GeneSetDB
    singlet_gpu::anno::GeneSetDB db;
    try { db = build_gene_set_db(gene_sets, gene_names); }
    catch (const std::exception& e) { Rcpp::stop("marker_score_cpp gene sets: %s", e.what()); }

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload matrix
    singlet_gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("marker_score_cpp upload: %s", e.what()); }

    // Configure method
    singlet_gpu::anno::MarkerScoreConfig cfg;
    if      (method == "mlm")   cfg.method = singlet_gpu::anno::MarkerMethod::Mlm;
    else if (method == "ulm")   cfg.method = singlet_gpu::anno::MarkerMethod::Ulm;
    else if (method == "wsum")  cfg.method = singlet_gpu::anno::MarkerMethod::Wsum;
    else if (method == "ucell") cfg.method = singlet_gpu::anno::MarkerMethod::UCell;
    else Rcpp::stop("marker_score_cpp: unknown method '%s'. Use 'mlm', 'ulm', 'wsum', or 'ucell'.",
                    method.c_str());

    // Run kernel
    singlet_gpu::anno::MarkerScoreResult res;
    try {
        res = singlet_gpu::anno::marker_score(d, db, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("marker_score_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_sets  = res.n_sets;
    const int n_cells = res.n_cells;
    const std::size_t total = static_cast<std::size_t>(n_sets) * n_cells;

    // Copy scores device → host (layout: n_sets × n_cells, column-major)
    std::vector<float> h_scores(total);
    try {
        cudaError_t err = cudaMemcpy(h_scores.data(), res.scores.data(),
                                     total * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("scores copyback: ") +
                                     cudaGetErrorString(err));
    } catch (const std::exception& e) {
        Rcpp::stop("marker_score_cpp copyback: %s", e.what());
    }

    // Transpose to R convention: [n_cells × n_sets]
    // Device layout: column c, set s → h_scores[s * n_cells + c]
    Rcpp::NumericMatrix r_scores(n_cells, n_sets);
    for (int s = 0; s < n_sets; ++s) {
        for (int c = 0; c < n_cells; ++c) {
            r_scores(c, s) = static_cast<double>(
                h_scores[static_cast<std::size_t>(s) * n_cells + c]);
        }
    }

    // Set column names to gene-set names
    Rcpp::CharacterVector r_set_names(n_sets);
    for (int s = 0; s < n_sets; ++s) r_set_names[s] = db.set_names[s];
    Rcpp::colnames(r_scores) = r_set_names;

    return Rcpp::List::create(
        Rcpp::Named("scores")    = r_scores,
        Rcpp::Named("set_names") = r_set_names
    );
}

// ---------------------------------------------------------------------------
// celltypist_project_cpp
//
// R-callable binding for CellTypist-style reference projection.
//
// Parameters:
//   embedding  — NumericMatrix [cells × n_pcs] (from reducedDim(sce, use_dimred))
//   model_ptr  — externalptr (from load_celltypist_model_cpp)
//
// Returns:
//   Rcpp::List{
//     "labels"      — IntegerVector [n_cells] (0-based class index)
//     "class_names" — CharacterVector (model class names)
//     "probs"       — NumericMatrix [n_cells × n_classes] (softmax probabilities)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List celltypist_project_cpp(
        Rcpp::NumericMatrix embedding,
        SEXP                model_ptr)
{
    // Validate external pointer
    if (TYPEOF(model_ptr) != EXTPTRSXP)
        Rcpp::stop("celltypist_project_cpp: model_ptr must be an externalptr from load_celltypist_model_cpp");

    CelltypistModelHandle* h = reinterpret_cast<CelltypistModelHandle*>(
        R_ExternalPtrAddr(model_ptr));
    if (!h)
        Rcpp::stop("celltypist_project_cpp: model has been freed (external pointer is NULL)");

    const int n_cells = embedding.nrow();
    const int n_pcs   = embedding.ncol();

    if (n_cells <= 0 || n_pcs <= 0)
        Rcpp::stop("celltypist_project_cpp: empty embedding");
    if (n_pcs != h->model.n_pcs)
        Rcpp::stop("celltypist_project_cpp: embedding has %d dims but model expects %d.",
                   n_pcs, h->model.n_pcs);

    // Stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    // Upload embedding host → device as a DeviceDense
    singlet_gpu::core::DeviceDense d_emb;
    try { d_emb = r_matrix_to_device_dense(embedding, stream); }
    catch (const std::exception& e) { Rcpp::stop("celltypist_project_cpp upload: %s", e.what()); }

    // Run projection
    singlet_gpu::anno::RefMapResult res;
    try {
        res = singlet_gpu::anno::project_to_reference(d_emb, h->model, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("celltypist_project_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_classes = h->model.n_classes;
    const std::size_t total_probs = static_cast<std::size_t>(n_classes) * n_cells;

    // Copy probabilities and labels device → host
    std::vector<float> h_probs(total_probs);
    std::vector<int>   h_labels(n_cells);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };
    try {
        chk(cudaMemcpy(h_probs.data(),  res.probabilities.data(),
                       total_probs * sizeof(float), cudaMemcpyDeviceToHost), "probs copyback");
        chk(cudaMemcpy(h_labels.data(), res.labels.data(),
                       n_cells     * sizeof(int),   cudaMemcpyDeviceToHost), "labels copyback");
    } catch (const std::exception& e) {
        Rcpp::stop("celltypist_project_cpp copyback: %s", e.what());
    }

    // Build R probs matrix [n_cells × n_classes]
    // Device layout: n_classes × n_cells column-major → row i, class c = h_probs[c * n_cells + i]
    Rcpp::NumericMatrix r_probs(n_cells, n_classes);
    for (int c = 0; c < n_classes; ++c) {
        for (int i = 0; i < n_cells; ++i) {
            r_probs(i, c) = static_cast<double>(
                h_probs[static_cast<std::size_t>(c) * n_cells + i]);
        }
    }

    // Class name columns on probs matrix
    Rcpp::CharacterVector r_class_names(n_classes);
    for (int c = 0; c < n_classes; ++c) r_class_names[c] = h->model.class_names[c];
    Rcpp::colnames(r_probs) = r_class_names;

    Rcpp::IntegerVector r_labels(n_cells);
    for (int i = 0; i < n_cells; ++i) r_labels[i] = h_labels[i];

    return Rcpp::List::create(
        Rcpp::Named("labels")      = r_labels,
        Rcpp::Named("class_names") = r_class_names,
        Rcpp::Named("probs")       = r_probs
    );
}

}  // namespace singlet_gpu_r
