// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/deflation_gpu.cuh — factornet::svd::deflation_svd_gpu<float>
//
// Adapter: singlet_gpu::reduce::svd::deflation  [PRIMARY BACKEND — Rule 32 adopt-winner]
//
// Benchmark result (Rule 32 consolidation, 2026-04-16):
//   Deflation: 28 ms, k-independent wall time, dominates all other backends across
//   k ∈ {10, 30, 50, 100} and scales 10k/100k/1M. Randomized kept as fallback only.
//   Lanczos, IRLBA, Krylov removed.
//
// Algorithm: Rank-1 ALS with explicit deflation (sequentially extracts each
//   singular triplet via alternating least squares, then deflates the residual).
//   Reference: factornet/svd/deflation_gpu.cuh.
//
// Features supported via SvdConfig:
//   Robust SVD (Huber IRLS):
//     Set cfg.robust_delta > 0 to activate Huber-type IRLS reweighting.
//     Each ALS step re-weights entries by the Huber weight of the residual.
//     cfg.irls_max_iter (default 5) controls outer IRLS iterations per factor.
//     cfg.irls_tol controls convergence of IRLS weights.
//     1.345 × median_abs_residual is a typical robust_delta choice.
//
//   Graph Laplacian regularization:
//     cfg.graph_u / cfg.graph_v: CPU-side factornet::SparseMatrix<float>.
//     cfg.graph_u_lambda / cfg.graph_v_lambda: strength (λ > 0 penalizes
//     high-frequency components of U/V relative to the graph structure).
//
//   L1, L2, non-negativity, upper bound, L21, angular constraints (all SvdConfig fields).
//
//   Post-factor reorthogonalization: cfg.reorth = true (default).
//
// Time complexity: O(nnz * k_max * iter) sequential rank-1 steps.
// Workspace: factornet allocates device buffers internally per rank.
//   Host-retention cost: (n+1)*4 + nnz*4 + nnz*4 bytes.
//
// Streams: factornet creates GPUContext internally; does NOT accept external stream.
// Precision: fp32; fp64 for IRLS weight accumulation when robust_delta > 0.
// Determinism: deterministic for fixed cfg.seed.
//
// OOC plan: factornet deflation allocates the full sparse matrix on device.
//   For matrices exceeding device memory, use PzChunkIterator + incremental SVD
//   via factornet::svd::streaming_matvec (deferred to cycle 7).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/svd/deflation_gpu.cuh>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/types.h>

#include <stdexcept>

namespace singlet_gpu {
namespace reduce {
namespace svd {

// require_host_retained — shared precondition guard for all SVD adapters.
// Throws if m was loaded without keep_host_pinned=true; the factornet GPU SVD
// functions take host pointers, so host buffers must be retained alongside the
// device CSC.  Call load_pz(..., keep_host_pinned=true) before calling SVD.
// NOTE: previously defined in lanczos.h (removed by Rule 32 consolidation).
//   deflation.h is now the canonical owner — any header needing this guard
//   should include <singlet-gpu/reduce/svd/deflation.h>.
inline void require_host_retained(const io::PzDeviceMatrix& m,
                                   const char* adapter_name) {
    if (!m.host_retained) {
        throw std::runtime_error(
            std::string("SVD adapter '") + adapter_name
            + "' requires PzDeviceMatrix loaded with keep_host_pinned=true");
    }
}

// deflation — rank-1 ALS deflation SVD adapter (PRIMARY backend).
// Dominates all k values and scales. Use auto_select to route here automatically.
inline SvdResult deflation(const io::PzDeviceMatrix& m, const SvdConfig& cfg) {
    require_host_retained(m, "deflation");
    return ::factornet::svd::deflation_svd_gpu<float>(
        m.host_indptr.get(),
        m.host_indices.get(),
        m.host_values.get(),
        m.mat.rows,
        m.mat.cols,
        m.mat.nnz,
        cfg);
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
