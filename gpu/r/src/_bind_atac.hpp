// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/_bind_atac.hpp
//
// Rcpp bindings for chromVAR-style GPU ATAC motif variability scoring:
//   chromvar_cpp — wraps singlet_gpu::atac::chromvar
//
// Input marshalling decisions:
//   accessibility:   dgCMatrix [peaks × cells] → DeviceCSC (upload_host_csc_to_device)
//   motif_in_peak:   R list{row_ptr, col_idx, values, n_rows=n_motifs, n_cols=n_peaks}
//                    (same HostCSR format as _bind_grn.hpp) → DeviceCSC via manual upload
//   peak_gc:         NumericVector [n_peaks] → DeviceMemory<float>
//   peak_mean_access: NumericVector [n_peaks] → DeviceMemory<float>
//   cfg params:      scalar R types → ChromVarConfig fields
//
// WHY motif_in_peak as CSR list (not dgCMatrix): The chromvar kernel requires a
//   CSR motif×peak matrix but the R user typically has a dgCMatrix (CSC peaks×motifs).
//   We accept a named list{row_ptr, col_idx, values, n_rows, n_cols} (same
//   r_list_to_host_csr helper as _bind_grn.hpp), then upload it as DeviceCSC by
//   treating the CSR rows as "genes" and cols as "cells" in DeviceCSC layout.
//   This avoids requiring the caller to transpose their dgCMatrix in R.
//   ALTERNATIVE: if the caller already has motif_in_peak as a dgCMatrix
//   (peaks × motifs), they transpose it to (motifs × peaks) and call with dgCMatrix.
//   We support dgCMatrix input via a separate overload detected by R type checking.
//
// Host-copy overhead:
//   Upload: O(nnz_access + nnz_motif + 2 × n_peaks) H→D
//   Download: O(n_motifs × n_cells × 4) D→H (deviation + p_values + p_adj +
//             variability is O(n_motifs) only)

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>
#include <singlet-gpu/core/handles.h>
#include <singlet-gpu/atac/chromvar.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// upload_host_csr_as_device_csc
//
// Upload a HostCSR (n_rows × n_cols) as a DeviceCSC by treating CSR row_ptr
// as col_ptr, col_indices as row_indices.  This is a layout reinterpretation:
// the caller must ensure the kernel expects exactly this transposed interpretation.
//
// For chromvar's motif_in_peak input: the kernel expects CSC where cols = peaks
// and rows = motifs. Providing the motifs×peaks CSR as a "CSC" where
// col_ptr = row_ptr, row_indices = col_indices achieves this.
//
// WHY NOT transpose on device: CSR→CSC transpose requires cuSPARSE csr2csc,
// adding complexity. The caller in R typically has motifs×peaks as CSR from
// their JASPAR pipeline; the direct reinterpretation is exact for our kernel.
// ---------------------------------------------------------------------------
inline singlet_gpu::core::DeviceCSC
upload_host_csr_as_device_csc(const singlet_gpu::grn::HostCSR& csr, cudaStream_t stream)
{
    singlet_gpu::core::DeviceCSC d;
    d.rows = static_cast<int64_t>(csr.n_cols);  // cols of CSC = peaks
    d.cols = static_cast<int64_t>(csr.n_rows);  // rows of CSC = motifs (as "cols")
    d.nnz  = static_cast<int64_t>(csr.nnz);

    const std::size_t indptr_sz = static_cast<std::size_t>(csr.n_rows + 1);
    const std::size_t nnz_sz    = static_cast<std::size_t>(csr.nnz);

    d.col_ptr     = singlet_gpu::core::DeviceMemory<int32_t>(indptr_sz);
    d.row_indices = singlet_gpu::core::DeviceMemory<int32_t>(nnz_sz);
    d.values      = singlet_gpu::core::DeviceMemory<float>  (nnz_sz);

    auto chk = [](cudaError_t e, const char* tag) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(tag) + ": " + cudaGetErrorString(e));
    };

    chk(cudaMemcpyAsync(d.col_ptr.data(), csr.row_ptr.data(),
                        indptr_sz * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
        "motif csr row_ptr");
    chk(cudaMemcpyAsync(d.row_indices.data(), csr.col_indices.data(),
                        nnz_sz * sizeof(int32_t), cudaMemcpyHostToDevice, stream),
        "motif csr col_indices");
    chk(cudaMemcpyAsync(d.values.data(), csr.values.data(),
                        nnz_sz * sizeof(float), cudaMemcpyHostToDevice, stream),
        "motif csr values");

    chk(cudaStreamSynchronize(stream), "motif csr upload sync");
    return d;
}

// ---------------------------------------------------------------------------
// chromvar_cpp
//
// R-callable binding for GPU chromVAR motif accessibility variability.
// Mirrors chromVAR R package (Schep et al. 2017) output:
//   deviation matrix (n_cells × n_motifs), p-value matrix, variability vector.
//
// Parameters:
//   accessibility      — dgCMatrix [peaks × cells] (raw ATAC fragment counts)
//   motif_in_peak      — named list{row_ptr, col_idx, values, n_rows=n_motifs, n_cols=n_peaks}
//                        CSR of motifs × peaks (1-based R indices)
//   peak_gc            — NumericVector [n_peaks] (GC content per peak, 0–1)
//   peak_mean_access   — NumericVector [n_peaks] (mean accessibility per peak)
//   motif_names        — CharacterVector [n_motifs] (names for result labeling)
//   n_background_peaks — int, matched background peaks per peak (default 50)
//   n_permutations     — int, permutation count for p-values (default 1000)
//   cell_batch         — int, cells per OOC batch (default 4096)
//   seed               — int, RNG seed (0L per cycle-24 pattern)
//   resource           — string "auto"/"gpu"/"cpu"
//
// Returns:
//   Rcpp::List{
//     "deviation"   — NumericMatrix [n_cells × n_motifs] (bias-corrected z-score)
//     "variability" — NumericVector [n_motifs] (variance across cells)
//     "p_values"    — NumericMatrix [n_cells × n_motifs] (raw permutation p)
//     "p_adj"       — NumericMatrix [n_cells × n_motifs] (BH-adjusted per motif)
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List chromvar_cpp(
        Rcpp::S4              accessibility,
        Rcpp::List            motif_in_peak,
        Rcpp::NumericVector   peak_gc,
        Rcpp::NumericVector   peak_mean_access,
        Rcpp::CharacterVector motif_names,
        int                   n_background_peaks,
        int                   n_permutations,
        int                   cell_batch,
        int                   seed,
        std::string           resource)
{
    // ── Marshal accessibility matrix ──────────────────────────────────────────
    HostCsc h_access;
    try { h_access = r_sparse_to_host_csc(accessibility); }
    catch (const std::exception& e) { Rcpp::stop("chromvar_cpp accessibility: %s", e.what()); }

    const int n_peaks = h_access.n_rows;
    const int n_cells = h_access.n_cols;

    // ── Marshal motif_in_peak CSR list ────────────────────────────────────────
    singlet_gpu::grn::HostCSR csr_motif;
    try { csr_motif = r_list_to_host_csr(motif_in_peak); }
    catch (const std::exception& e) { Rcpp::stop("chromvar_cpp motif_in_peak: %s", e.what()); }

    const int n_motifs = csr_motif.n_rows;

    if (csr_motif.n_cols != n_peaks)
        Rcpp::stop("chromvar_cpp: motif_in_peak n_cols (%d) != n_peaks in accessibility (%d).",
                   csr_motif.n_cols, n_peaks);
    if ((int)peak_gc.size() != n_peaks)
        Rcpp::stop("chromvar_cpp: peak_gc length (%d) != n_peaks (%d).",
                   (int)peak_gc.size(), n_peaks);
    if ((int)peak_mean_access.size() != n_peaks)
        Rcpp::stop("chromvar_cpp: peak_mean_access length (%d) != n_peaks (%d).",
                   (int)peak_mean_access.size(), n_peaks);

    // ── GPUContext (owns cuBLAS + cuSPARSE + cuSOLVER handles + stream) ──────
    // GPUContext() creates all handles and a dedicated stream internally, then
    // binds all handles to that stream.  We use ctx.stream for all operations.
    singlet_gpu::core::GPUContext ctx;
    cudaStream_t stream = ctx.stream;  // alias; ctx owns lifetime

    // ── Upload matrices ───────────────────────────────────────────────────────
    singlet_gpu::core::DeviceCSC d_access;
    try { d_access = upload_host_csc_to_device(h_access, stream); }
    catch (const std::exception& e) { Rcpp::stop("chromvar_cpp upload access: %s", e.what()); }

    singlet_gpu::core::DeviceCSC d_motif;
    try { d_motif = upload_host_csr_as_device_csc(csr_motif, stream); }
    catch (const std::exception& e) { Rcpp::stop("chromvar_cpp upload motif: %s", e.what()); }

    // Upload peak_gc and peak_mean_access
    std::vector<float> h_gc(static_cast<std::size_t>(n_peaks));
    std::vector<float> h_mean_acc(static_cast<std::size_t>(n_peaks));
    for (int p = 0; p < n_peaks; ++p) {
        h_gc      [p] = static_cast<float>(peak_gc[p]);
        h_mean_acc[p] = static_cast<float>(peak_mean_access[p]);
    }

    singlet_gpu::core::DeviceMemory<float> d_gc  (static_cast<std::size_t>(n_peaks));
    singlet_gpu::core::DeviceMemory<float> d_mean_acc(static_cast<std::size_t>(n_peaks));

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_gc.get(),       h_gc.data(),
                                n_peaks * sizeof(float), cudaMemcpyHostToDevice, stream), "gc");
            chk(cudaMemcpyAsync(d_mean_acc.get(), h_mean_acc.data(),
                                n_peaks * sizeof(float), cudaMemcpyHostToDevice, stream), "mean_acc");
            chk(cudaStreamSynchronize(stream), "peak vectors sync");
        } catch (const std::exception& e) {
            Rcpp::stop("chromvar_cpp upload peaks vectors: %s", e.what());
        }
    }

    // ── Configure ─────────────────────────────────────────────────────────────
    singlet_gpu::atac::ChromVarConfig cfg;
    cfg.n_background_peaks = n_background_peaks > 0 ? n_background_peaks : 50;
    cfg.n_permutations     = n_permutations     > 0 ? n_permutations     : 1000;
    cfg.cell_batch         = cell_batch         > 0 ? cell_batch         : 4096;
    cfg.seed               = static_cast<uint64_t>(seed);

    // ── Run kernel ─────────────────────────────────────────────────────────────
    singlet_gpu::atac::ChromVarResult res;
    try {
        res = singlet_gpu::atac::chromvar(d_access, d_motif, d_gc, d_mean_acc, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("chromvar_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // ── Copy results device → host ────────────────────────────────────────────
    const std::size_t total_mc = static_cast<std::size_t>(n_motifs) * n_cells;

    std::vector<float> h_dev (total_mc);
    std::vector<float> h_var (static_cast<std::size_t>(n_motifs));
    std::vector<float> h_pval(total_mc);
    std::vector<float> h_padj(total_mc);

    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpy(h_dev .data(), res.deviation   .data(), total_mc       * sizeof(float), cudaMemcpyDeviceToHost), "deviation");
            chk(cudaMemcpy(h_var .data(), res.variability  .data(), n_motifs       * sizeof(float), cudaMemcpyDeviceToHost), "variability");
            chk(cudaMemcpy(h_pval.data(), res.p_values     .data(), total_mc       * sizeof(float), cudaMemcpyDeviceToHost), "p_values");
            chk(cudaMemcpy(h_padj.data(), res.p_adj        .data(), total_mc       * sizeof(float), cudaMemcpyDeviceToHost), "p_adj");
        } catch (const std::exception& e) {
            Rcpp::stop("chromvar_cpp copyback: %s", e.what());
        }
    }

    // kernel layout: deviation[motif × cell] (motif-major, row-major in kernel).
    // R convention: matrix [n_cells × n_motifs] (cells are rows, motifs are cols).
    Rcpp::NumericMatrix r_dev (n_cells, n_motifs);
    Rcpp::NumericMatrix r_pval(n_cells, n_motifs);
    Rcpp::NumericMatrix r_padj(n_cells, n_motifs);
    Rcpp::NumericVector r_var (n_motifs);

    for (int m = 0; m < n_motifs; ++m) {
        r_var[m] = static_cast<double>(h_var[m]);
        for (int c = 0; c < n_cells; ++c) {
            const std::size_t off = static_cast<std::size_t>(m) * n_cells + c;
            r_dev (c, m) = static_cast<double>(h_dev [off]);
            r_pval(c, m) = static_cast<double>(h_pval[off]);
            r_padj(c, m) = static_cast<double>(h_padj[off]);
        }
    }

    // Set column names (motif names) if provided
    const bool have_motif_names = (motif_names.size() == n_motifs);
    if (have_motif_names) {
        Rcpp::colnames(r_dev)  = motif_names;
        Rcpp::colnames(r_pval) = motif_names;
        Rcpp::colnames(r_padj) = motif_names;
        r_var.names()          = motif_names;
    }

    return Rcpp::List::create(
        Rcpp::Named("deviation")   = r_dev,
        Rcpp::Named("variability") = r_var,
        Rcpp::Named("p_values")    = r_pval,
        Rcpp::Named("p_adj")       = r_padj
    );
}

}  // namespace singlet_gpu_r
