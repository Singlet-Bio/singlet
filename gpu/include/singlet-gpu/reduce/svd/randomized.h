// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/randomized_gpu.cuh — factornet::svd::randomized_svd_gpu<float>
//
// Adapter: singlet_gpu::reduce::svd::randomized
//
// Algorithm: Halko–Martinsson–Tropp randomized SVD with power iterations.
//   Sketch dimension l = k_max + oversampling (max(10, k_max/5)).
//   Power iteration count q = max_iter (default 3 per factornet; q=3 reduces
//   tail SV error to <1% vs q=2 at ~5%).
//   cuSPARSE SpMM (batched) for A*Ω and A'*(A*Ω) passes;
//   cuSOLVER gesvd for the small l×l QR factorization.
//   Reference: Halko, Martinsson & Tropp 2011, §4 (Algorithm 4.4).
//   factornet doc: svd/randomized_gpu.cuh.
//
// Time complexity: O(nnz * l * q) where l ≈ k_max + 10.
//   Fixed cost per call: independent of convergence (no restarts).
// Workspace: factornet allocates Ω(n×l), Y(m×l), Q(m×l) on device.
//   Host-retention cost: (n+1)*4 + nnz*4 + nnz*4 bytes.
//
// Streams: factornet creates GPUContext internally; does NOT accept external stream.
//   CYCLE-5-FOLLOWUP-FACTORNET-STREAM-OVERLOAD tracks the upstream request.
//
// Precision: fp32 for sketching and matvecs; factornet uses fp64 for QR/SVD
//   of the small l×l factor (Eigen JacobiSVD).
// Determinism: deterministic for fixed cfg.seed (random sketch seeded explicitly).
// Optimal range: 32 ≤ k_max < 64 on GPU; lowest memory footprint of all backends.
//
// OOC plan: lowest memory footprint of all backends due to fixed sketch size.
//   Preferred backend for >1M-cell chunks; streaming deferred to cycle 7.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/svd/randomized_gpu.cuh>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/svd/types.h>
#include <singlet-gpu/reduce/svd/deflation.h>  // require_host_retained (canonical owner)

namespace singlet_gpu {
namespace reduce {
namespace svd {

// randomized — HMT randomized SVD adapter.
// Optimal for 32 ≤ k_max < 64 on GPU.
inline SvdResult randomized(const io::PzDeviceMatrix& m, const SvdConfig& cfg) {
    require_host_retained(m, "randomized");
    return ::factornet::svd::randomized_svd_gpu<float>(
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
