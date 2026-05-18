// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/nmf/init.h
//
// NMF factor initialization helpers.
// CYCLE-105: updated to use native DenseMatrix (no factornet/Eigen dep).
//
// Two strategies:
//   mode 0 — Random uniform [0,1) seeded by `seed`.
//   mode 1 — Deflation SVD seed: W = |U|·√Σ, H = |V|·√Σ (primary backend, all k).
//
// WHY deflation: k-independent 28ms wall time; winner of Rule 32 benchmark.

#pragma once

#include <singlet/gpu/reduce/nmf/types.h>
#include <singlet/gpu/reduce/svd/deflation.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <cmath>
#include <cstdint>
#include <utility>

namespace singlet::gpu {
namespace reduce {
namespace nmf {

// init_random — uniform [0,1) initialization seeded by `seed`.
// Returns W_init (m × k). H_init can be constructed identically with different seed.
inline DenseMatrix init_random(int m, int /*n*/, int k, uint64_t seed)
{
    DenseMatrix W(m, k);
    uint64_t s = seed;
    auto next = [&]() -> float {
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
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
//   W_init(i, r) = |U(i, r)| * sqrt(sigma_r)
//   H_init(r, j) = sqrt(sigma_r) * |V(j, r)|
inline std::pair<DenseMatrix, DenseMatrix>
init_deflation(const io::PzDeviceMatrix& mat, int k, const NmfConfig& cfg)
{
    svd::SvdConfig svd_cfg;
    svd_cfg.k_max   = k;
    svd_cfg.center  = false;
    svd_cfg.verbose = false;
    svd_cfg.seed    = cfg.seed;

    auto svd_result = svd::deflation(mat, svd_cfg);
    const int k_actual = svd_result.k_selected;
    const int m = static_cast<int>(mat.mat.rows);
    const int n = static_cast<int>(mat.mat.cols);

    DenseMatrix W(m, k);
    DenseMatrix H(k, n);

    for (int r = 0; r < k_actual && r < k; ++r) {
        float sqrtd = std::sqrt(std::max(0.0f, svd_result.d(r)));
        for (int i = 0; i < m; ++i)
            W(i, r) = std::abs(svd_result.U_data[r * m + i]) * sqrtd;
        for (int j = 0; j < n; ++j)
            H(r, j) = sqrtd * std::abs(svd_result.V_data[r * n + j]);
    }

    // Fill any remaining ranks with random.
    for (int r = k_actual; r < k; ++r) {
        auto Wcol = init_random(m, 1, 1, cfg.seed + static_cast<uint64_t>(r) + 1000ULL);
        auto Hrow = init_random(1, n, n, cfg.seed + static_cast<uint64_t>(r) + 2000ULL);
        for (int i = 0; i < m; ++i) W(i, r) = Wcol(i, 0);
        for (int j = 0; j < n; ++j) H(r, j) = Hrow(0, j);
    }
    return {W, H};
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet::gpu
