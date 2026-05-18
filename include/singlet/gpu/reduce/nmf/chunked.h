// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/nmf/chunked.h
//
// Chunked (out-of-core) NMF — thin wrapper looping over .1pz shards.
// CYCLE-105: native implementation via PzChunkIterator, no factornet backend.
//
// Algorithm: iterate PzChunkIterator column panels; for each panel, call
//   nmf::fit with warm-start H slice and running-average W.
//
// The factornet::io::DataLoader interface is no longer used. Callers that
// previously used chunked_fit(PzDataLoader&, ...) can instead open
// a PzChunkIterator from io/pz_device_loader.h.
//
// WHY PzChunkIterator not PzDataLoader: PzDataLoader still depends on
//   factornet::io::DataLoader<float> for its virtual interface. CYCLE-105
//   removes factornet as a dependency; PzChunkIterator is the native iterator
//   and does not require factornet headers.
//
// OOC: PzChunkIterator holds the full file in RAM but processes one column
//   panel at a time on device. For files > host RAM, a mmap variant is needed
//   (tracked in pz_device_loader.h TODO(OOC)).
// Streams: each panel fit uses its own GPUContext.
// Precision: fp32.

#pragma once

#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/reduce/nmf/fit.h>
#include <singlet/gpu/io/pz_device_loader.h>
// PzDataLoader lives in streaming/pz_data_loader.h (not io/pz_device_loader.h).
// chunked_fit(PzDataLoader& loader, ...) needs this include to resolve
// singlet::gpu::io::PzDataLoader — the CYCLE-105/106 split header caused the gap.
#include <singlet/gpu/streaming/pz_data_loader.h>

#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace reduce {
namespace nmf {

// ---------------------------------------------------------------------------
// chunked_fit — streaming NMF adapter.
//
// path:   .1pz file path.
// cfg:    NmfConfig. solver_mode=2 (MU) is recommended; forced if mode=3.
// chunk_cols: column panel width (default 100k).
// W_init: optional warm-start W (m × k). nullptr = random init on first panel.
// H_init: optional warm-start H (k × n). nullptr = random init per panel.
//
// Returns NmfResult with W (m × k), H (k × n), convergence history.
// ---------------------------------------------------------------------------
inline NmfResult chunked_fit(const std::string& path,
                              const NmfConfig& cfg,
                              uint32_t chunk_cols = 100000u,
                              const DenseMatrix* W_init = nullptr,
                              const DenseMatrix* H_init = nullptr)
{
    // Force MU — Cholesky requires full Gram, incompatible with chunked.
    NmfConfig effective_cfg = cfg;
    if (effective_cfg.solver_mode != 2) effective_cfg.solver_mode = 2;

    // Open the file.
    io::PzChunkIterator iter(path, /*stream=*/nullptr, chunk_cols);
    const int mat_m = static_cast<int>(iter.nrows());
    const int mat_n = static_cast<int>(iter.ncols());
    const int k     = effective_cfg.rank;

    if (mat_m <= 0 || mat_n <= 0 || k <= 0)
        throw std::runtime_error("chunked_fit: invalid matrix dimensions or rank");

    DenseMatrix W_accum(mat_m, k);
    DenseMatrix H_full(k, mat_n);

    if (W_init && !W_init->empty())
        W_accum = *W_init;

    int n_panels = 0;
    int col_start = 0;

    while (iter.has_next()) {
        auto chunk_mat = iter.next();  // PzDeviceMatrix for this column panel
        const int panel_cols = chunk_mat.mat.cols;
        if (panel_cols == 0) { col_start += panel_cols; continue; }

        // Enable host retention for the SVD/NMF path.
        // chunk_mat doesn't have host buffers — build a thin view for fit().
        // fit() uses device pointers directly via the cuSPARSE CSC descriptor.

        // Extract H_init slice for this column range.
        DenseMatrix H_slice(k, panel_cols);
        if (H_init && !H_init->empty()) {
            for (int r = 0; r < k; ++r)
                for (int jj = 0; jj < panel_cols; ++jj)
                    H_slice(r, jj) = (*H_init)(r, col_start + jj);
        }

        const DenseMatrix* w_in = (n_panels == 0 && W_init) ? W_init : &W_accum;
        const DenseMatrix* h_in = (H_init && !H_init->empty()) ? &H_slice : nullptr;

        NmfResult r = fit(chunk_mat, effective_cfg, FitConfig{}, w_in, h_in);

        // Running-average W accumulation.
        if (n_panels == 0) {
            W_accum = r.W;
        } else {
            float alpha = static_cast<float>(n_panels) / (n_panels + 1.f);
            float beta  = 1.f / (n_panels + 1.f);
            for (int i = 0; i < mat_m * k; ++i)
                W_accum.data[i] = alpha * W_accum.data[i] + beta * r.W.data[i];
        }

        // Store H slice.
        for (int r2 = 0; r2 < k; ++r2)
            for (int jj = 0; jj < panel_cols; ++jj)
                H_full(r2, col_start + jj) = r.H(r2, jj);

        col_start += panel_cols;
        ++n_panels;
    }

    NmfResult result;
    result.m = mat_m;
    result.n = mat_n;
    result.k = k;
    result.W = std::move(W_accum);
    result.H = std::move(H_full);
    result.d = DenseVector(k);
    for (int i = 0; i < k; ++i) result.d(i) = 1.f;
    result.converged = (n_panels > 0);
    result.iterations = n_panels;
    return result;
}

// ---------------------------------------------------------------------------
// Backward-compatible overload: accepts a PzDeviceMatrix directly.
// Used by _bind_kernels.hpp's nmf_chunked binding which passes a pre-loaded matrix.
// ---------------------------------------------------------------------------
inline NmfResult chunked_fit(io::PzDataLoader& loader,
                              const NmfConfig& cfg,
                              const DenseMatrix* W_init = nullptr,
                              const DenseMatrix* H_init = nullptr)
{
    // PzDataLoader exposes rows() / cols() and the path it was opened on.
    // Delegate to the PzChunkIterator-based path using the file.
    return chunked_fit(loader.path(), cfg, io::PzChunkIterator::DEFAULT_CHUNK_COLS, W_init, H_init);
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet::gpu
