// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/nmf/fit_cv_gpu.cuh — factornet::nmf::nmf_cv_fit_gpu<float>
//
// Adapter: singlet_gpu::reduce::nmf::cv_fit
//
// Algorithm: speckled cross-validation NMF on GPU.
//   Holds out a random fraction of entries (config.holdout_fraction or config.cv_seed).
//   Trains on the remaining entries; evaluates reconstruction loss on held-out set.
//   Repeats for each candidate rank (orchestrator calls cv_fit multiple times with
//   increasing config.rank, then reads result.test_loss to select the best rank).
//
//   GPU path:
//     Cholesky + Gram trick for MSE (default); GPU-native MU for non-MSE losses.
//     A stays on device.  G (k×k) host-round-trips for feature application only.
//
//   CV mask: LazySpeckledMask — O(1) query per entry, O(1) memory, no materialization.
//     The mask is seeded by config.seed (or config.cv_seed when nonzero).
//
//   Reference: factornet/nmf/fit_cv_gpu.cuh §"Public API".
//
// Time complexity: same as nmf_fit_gpu plus O(n_test) test-loss evaluation per iter.
// Workspace: same as fit.h plus delta_G / delta_B correction buffers for held-out cols.
//   Hold-out correction: O(k² * n_test) additional device memory.
//
// Streams: factornet creates GPUContext per call (no external stream).
// Precision: fp32 data; fp64 for convergence residual in Gram.
// Determinism: seeded via config.seed + config.cv_seed.
//
// OOC plan: for matrices exceeding device memory, streaming CV is not yet supported
//   in factornet.  Use rank selection on a representative chunk, then fit full matrix.
//   Tracked as CYCLE-6-FOLLOWUP-STREAMING-CV.
//
// Auto-rank usage (orchestrator pattern):
//   for (int k : {5, 10, 15, 20, 30, 40}) {
//     auto cfg = base_cfg;
//     cfg.rank = k;
//     auto r = singlet_gpu::reduce::nmf::cv_fit(m, cfg);
//     if (r.best_test_loss < best) { best = r.best_test_loss; best_k = k; }
//   }

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/nmf/fit_cv_gpu.cuh>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/svd/deflation.h>  // require_host_retained (canonical owner)

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// cv_fit — speckled cross-validation NMF adapter.
//
// Same precondition as fit(): m must have been loaded with keep_host_pinned=true.
//
// Result fields used for rank selection:
//   result.best_test_loss  — minimum held-out MSE across all iterations
//   result.test_loss       — final held-out MSE (may be higher than best)
//   result.best_iter       — iteration that achieved best_test_loss
//   result.W / .d / .H    — factors at early-stopping iteration
//
// Enable CV by setting cfg.holdout_fraction > 0 (e.g., 0.05 = 5% held out).
// Set cfg.cv_patience to control early-stopping (default: NMF_PATIENCE).
inline NmfResult cv_fit(const io::PzDeviceMatrix& m,
                        const NmfConfig& cfg,
                        const DenseMatrix* W_init = nullptr,
                        const DenseMatrix* H_init = nullptr)
{
    ::singlet_gpu::reduce::svd::require_host_retained(m, "nmf::cv_fit");

    return ::factornet::nmf::nmf_cv_fit_gpu<float>(
        m.host_indptr.get(),
        m.host_indices.get(),
        m.host_values.get(),
        static_cast<int>(m.mat.rows),
        static_cast<int>(m.mat.cols),
        static_cast<int>(m.mat.nnz),
        cfg,
        W_init,
        H_init);
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
