// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/nmf/nmf_init.hpp — factornet::nmf::detail::initialize_{factors,lanczos,irlba}
//             factornet/svd/deflation_gpu.cuh (via reduce/svd/deflation.h)
//
// NMF factor initialization helpers.
//
// Two strategies after Rule 32 consolidation (lanczos + irlba removed):
//   mode 0 — Random uniform [0,1) seeded by `seed`.
//   mode 1 — Deflation SVD seed: W = |U|·√Σ, H = |V|·√Σ (primary backend, all k).
//
// Note: factornet::nmf::detail::initialize_* functions operate on CPU Eigen matrices
//   and use CPU SVD backends.  For GPU-seeded init (better performance at large k),
//   this adapter calls our GPU deflation adapter instead, then converts the result
//   to DenseMatrix (host Eigen) for hand-off to nmf_fit_gpu via W_init / H_init.
//
// WHY deflation: k-independent 28ms wall time; winner of Rule 32 benchmark across
//   k ∈ {10, 30, 50, 100}.  Randomized available as fallback (see auto_select.h).
//
// Time complexity: O(nnz * k) for SVD init; O(m * k + n * k) for random.
// Workspace: SVD init allocates the same workspace as the SVD adapter.
//   Random init is O(1) device (SplitMix64 runs on host).
//
// Streams: inherited from SVD adapter (factornet GPUContext, no external stream).
// Precision: fp32 throughout.  sqrt(singular_values) computed in float.
// Determinism: seeded via `seed` argument.
//
// OOC plan: init acts on the full matrix (or a representative chunk for very large data).
//   For OOC NMF, pass the init result as W_init/H_init to chunked_fit.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/svd/deflation.h>
#include <singlet-gpu/io/pz_device_loader.h>

#include <cmath>
#include <cstdint>
#include <utility>

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// init_random — uniform [0,1) initialization seeded by `seed`.
// Returns W_init (m×k).  H_init can be constructed identically by passing
// a different seed, or by calling this function twice.
// The factornet nmf_fit_gpu init_mode=0 path uses the same SplitMix64 RNG,
// so passing nullptr for W_init/H_init and config.seed achieves the same result.
// This helper is provided for callers that want to inspect or modify the init.
inline DenseMatrix init_random(int m, int n, int k, uint64_t seed)
{
    (void)n;  // n not needed for W; H has same shape pattern
    DenseMatrix W(m, k);
    // SplitMix64 inline: deterministic, same as factornet rng::SplitMix64.
    uint64_t s = seed;
    auto next = [&]() -> float {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        // Map to [0, 1)
        return static_cast<float>(z >> 11) * (1.0f / static_cast<float>(1ULL << 53));
    };
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i)
            W(i, j) = next();
    return W;
}

// init_deflation — SVD-seeded initialization using the GPU deflation adapter (primary).
// Returns (W_init [m×k], H_init [k×n]).
// Computes rank-k deflation SVD on GPU, then sets:
//   W_init(i, :) = |U(:, i)| * sqrt(sigma_i)   → absolute value ensures non-negativity.
//   H_init(:, j) = sqrt(sigma_i) * |V(:, j)|ᵀ
// Rule 32 winner: k-independent 28ms wall time across all k and scales.
// Previously named init_lanczos (lanczos removed) and init_irlba (irlba removed).
inline std::pair<DenseMatrix, DenseMatrix>
init_deflation(const io::PzDeviceMatrix& mat, int k, const NmfConfig& cfg)
{
    svd::SvdConfig svd_cfg;
    svd_cfg.k_max   = k;
    svd_cfg.center  = false;  // NMF operates on non-negative data; centering breaks non-negativity.
    svd_cfg.verbose = false;
    svd_cfg.seed    = cfg.seed;

    auto svd_result = ::singlet_gpu::reduce::svd::deflation(mat, svd_cfg);
    const int k_actual = svd_result.k_selected;
    const int m = static_cast<int>(mat.mat.rows);
    const int n = static_cast<int>(mat.mat.cols);

    DenseMatrix W(m, k);
    DenseMatrix H(k, n);
    for (int i = 0; i < k_actual && i < k; ++i) {
        float sqrtd = std::sqrt(std::max(0.0f, svd_result.d(i)));
        W.col(i) = svd_result.U.col(i).cwiseAbs() * sqrtd;
        H.row(i) = svd_result.V.col(i).cwiseAbs().transpose() * sqrtd;
    }
    // Fill any remaining ranks (if SVD returned fewer than k) with random.
    for (int i = k_actual; i < k; ++i) {
        auto Wcol = init_random(m, 1, 1, cfg.seed + static_cast<uint64_t>(i) + 1000ULL);
        auto Hrow = init_random(1, n, n, cfg.seed + static_cast<uint64_t>(i) + 2000ULL);
        W.col(i) = Wcol.col(0);
        H.row(i) = Hrow.row(0);
    }
    return {W, H};
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
