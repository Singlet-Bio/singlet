// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_load.hpp
//
// Rcpp implementation of load_pz_to_sce_cpp.
//
// Entry point:
//   Rcpp::List load_pz_to_sce_cpp(std::string pz_dir,
//                                  std::string modality,
//                                  bool keep_host_pinned)
//
// What this does:
//   1. Resolve the .1pz file path: pz_dir / {modality}_counts.1pz
//      (falls back to pz_dir / counts.1pz for "exon").
//   2. Call singlet::gpu::io::load_pz(path, /*stream=*/nullptr, keep_host_pinned).
//   3. Synchronize the CUDA stream so device data is ready.
//   4. Copy device CSC back to host (float*) — R cannot consume GPU memory.
//   5. Build a Matrix::dgCMatrix from the host data via host_csc_to_r_sparse.
//   6. Package metadata into a named R list.
//   7. Return Rcpp::List{ "counts"=dgCMatrix, "metadata"=list,
//                         "cells"=CharacterVector, "genes"=CharacterVector }.
//
// Cost: one O(nnz) device→host copy for the matrix values + two O(nnz) for
//   indices/indptr.  Pinned host memory makes these DMA transfers async-capable.
//   The synchronization before returning ensures R sees complete data.
//
// Caller (R side): R/load_pz.R wraps this into a full SCE via
//   SingleCellExperiment::SingleCellExperiment(...).

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/io/pz_device_loader.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/core/types.h>

#include "_r_to_eigen.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// modality_to_filename
//
// Map the user-facing modality string to the .1pz file name inside pz_dir.
// Mirrors the singlet output naming convention from CLAUDE.md §Full Output.
// ---------------------------------------------------------------------------
inline std::string modality_to_filename(const std::string& modality) {
    if (modality == "exon"    || modality == "counts") return "counts.1pz";
    if (modality == "intron")                          return "intron_counts.1pz";
    if (modality == "gene")                            return "gene_counts.1pz";
    if (modality == "adt")                             return "adt.1pz";
    if (modality == "atac")                            return "fragments.1pz";
    if (modality == "snp_ad")                          return "snp_ad.1pz";
    if (modality == "snp_dp")                          return "snp_dp.1pz";
    // Default: try the modality string verbatim + .1pz
    return modality + ".1pz";
}

// ---------------------------------------------------------------------------
// load_pz_to_sce_cpp
//
// Core binding function.  Called from R/load_pz.R via .Call.
// Returns a named Rcpp::List that the R wrapper converts into a SCE.
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List load_pz_to_sce_cpp(
        std::string pz_dir,
        std::string modality,
        bool        keep_host_pinned)
{
    // Normalise directory path (remove trailing slash, add separator)
    if (!pz_dir.empty() && pz_dir.back() != '/') pz_dir += '/';
    const std::string pz_path = pz_dir + modality_to_filename(modality);

    // Validate file exists before calling into CUDA
    {
        std::ifstream test(pz_path);
        if (!test.good()) {
            Rcpp::stop("load_pz_to_sce_cpp: cannot open '%s'. "
                       "Check pz_dir and modality.", pz_path.c_str());
        }
    }

    // Load .1pz → device CSC + metadata (CPU decompress + async GPU upload)
    singlet::gpu::io::PzDeviceMatrix dev_mat;
    try {
        dev_mat = singlet::gpu::io::load_pz(pz_path,
                                           /*stream=*/nullptr,
                                           keep_host_pinned);
    } catch (const std::exception& e) {
        Rcpp::stop("load_pz_to_sce_cpp: load_pz failed: %s", e.what());
    }

    // Synchronize so device data is ready before device→host copy
    if (dev_mat.producer_stream) {
        cudaError_t err = cudaStreamSynchronize(dev_mat.producer_stream);
        if (err != cudaSuccess) {
            Rcpp::stop("load_pz_to_sce_cpp: cudaStreamSynchronize failed: %s",
                       cudaGetErrorString(err));
        }
    }

    const auto& csc   = dev_mat.mat;
    const auto& meta  = dev_mat.meta;
    const int64_t nnz    = csc.nnz;
    const int32_t n_rows = static_cast<int32_t>(csc.rows);
    const int32_t n_cols = static_cast<int32_t>(csc.cols);

    // Allocate host buffers and copy device → host
    std::vector<int32_t> h_indptr (static_cast<std::size_t>(n_cols + 1));
    std::vector<int32_t> h_indices(static_cast<std::size_t>(nnz));
    std::vector<float>   h_values (static_cast<std::size_t>(nnz));

    auto chk = [](cudaError_t e, const char* tag) {
        if (e != cudaSuccess) {
            throw std::runtime_error(std::string(tag)
                + ": " + cudaGetErrorString(e));
        }
    };

    try {
        chk(cudaMemcpy(h_indptr.data(), csc.col_ptr.data(),
                       static_cast<std::size_t>(n_cols + 1) * sizeof(int32_t),
                       cudaMemcpyDeviceToHost), "indptr");
        chk(cudaMemcpy(h_indices.data(), csc.row_indices.data(),
                       static_cast<std::size_t>(nnz) * sizeof(int32_t),
                       cudaMemcpyDeviceToHost), "indices");
        chk(cudaMemcpy(h_values.data(), csc.values.data(),
                       static_cast<std::size_t>(nnz) * sizeof(float),
                       cudaMemcpyDeviceToHost), "values");
    } catch (const std::exception& e) {
        Rcpp::stop("load_pz_to_sce_cpp: device→host copy failed: %s", e.what());
    }

    // Build Matrix::dgCMatrix
    Rcpp::S4 dgc = host_csc_to_r_sparse(
        h_indptr, h_indices, h_values, n_rows, n_cols);

    // Build metadata list (mirrors GEO JSON schema from CLAUDE.md)
    Rcpp::List r_meta = Rcpp::List::create(
        Rcpp::Named("gsm_id")           = meta.gsm_id,
        Rcpp::Named("gse_id")           = meta.gse_id,
        Rcpp::Named("organism")         = meta.organism,
        Rcpp::Named("taxon_id")         = static_cast<int>(meta.taxon_id),
        Rcpp::Named("protocol")         = meta.protocol,
        Rcpp::Named("modality")         = meta.modality,
        Rcpp::Named("srr_ids")          = Rcpp::wrap(meta.srr_ids),
        Rcpp::Named("read_count")       = static_cast<double>(meta.read_count),
        Rcpp::Named("geo_title")        = meta.geo_title,
        Rcpp::Named("geo_source_name")  = meta.geo_source_name,
        Rcpp::Named("singlet_version") = meta.singlet_version,
        Rcpp::Named("pipeline_date")    = meta.pipeline_date
    );

    // Gene and cell name vectors from metadata
    Rcpp::CharacterVector genes(meta.rownames.begin(), meta.rownames.end());
    Rcpp::CharacterVector cells(meta.colnames.begin(), meta.colnames.end());

    return Rcpp::List::create(
        Rcpp::Named("counts")   = dgc,
        Rcpp::Named("metadata") = r_meta,
        Rcpp::Named("cells")    = cells,
        Rcpp::Named("genes")    = genes
    );
}

}  // namespace singlet_gpu_r
