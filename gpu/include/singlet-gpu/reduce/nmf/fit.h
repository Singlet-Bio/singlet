// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/nmf/fit_gpu.cuh — factornet::nmf::nmf_fit_gpu<float>
//
// Adapter: singlet_gpu::reduce::nmf::fit
//
// Algorithm: multiplicative-update / coordinate-descent / Cholesky NMF on GPU.
//   Data residency (GPU-only path — MSE loss, Tier-1 constraints):
//     A (sparse): uploaded once via SparseMatrixGPU<float> internal to factornet.
//     W (k×m), H (k×n): device-resident throughout; downloaded once at end.
//     G (k×k): computed on device; host roundtrip only for Tier-2 features.
//   Non-MSE loss path (KL, NB, GP, Gamma, Tweedie, IRLS):
//     factornet host-mediates the IRLS solver per column; SpMM for Gram/RHS stays
//     on device.  50–100× slower than MSE.  This is factornet's behavior.
//   Reference: factornet/nmf/fit_gpu.cuh §"Data residency" + §"Public API".
//
// Time complexity: O(nnz * k * iters + k² * n * iters) for MSE.
// Workspace: factornet allocates W(k×m) + H(k×n) + G(k×k) + B(k×max(m,n)) on device.
//   Host pinned buffers from keep_host_pinned=true: 2× matrix (device + host).
//   Host roundtrip per iter when Tier-2 features active: G download only (k²×4 bytes).
//
// Streams: factornet creates a GPUContext (and internal stream) per call.
//   The adapter does NOT accept an external cudaStream_t.
//   CYCLE-6-FOLLOWUP-FACTORNET-STREAM-OVERLOAD tracks the upstream request.
//
// Precision: fp32 for all device tensors.  fp64 accumulators in G reductions (k²).
// Determinism: stochastic only for init_mode=0 (random); seeded via config.seed.
//   MU/CD iterations are deterministic given fixed reduction order on a fixed GPU.
//
// OOC plan: this entry point loads the full matrix onto device.  For datasets
//   exceeding device memory, use chunked_fit (chunked.h) with PzDataLoader.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/nmf/fit_gpu.cuh>
#include <singlet-gpu/io/pz_device_loader.h>
#include <singlet-gpu/reduce/nmf/types.h>
#include <singlet-gpu/reduce/svd/deflation.h>  // require_host_retained (canonical owner)

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// FitConfig — adapter-level tuning knobs for fit().
//
// These parameters override factornet's NMFConfig defaults where H100 GPU
// profiling revealed that factornet's CPU-calibrated defaults are suboptimal.
// Callers who leave FitConfig at defaults get the GPU-tuned behavior automatically
// (Rule 31: sane defaults + explicit tuning).
//
// k_cd_cutoff: rank threshold below which CD is allowed; at or above this rank,
//   solver_mode=3 (auto) is overridden to solver_mode=2 (MU).
//   Rationale: factornet's should_switch heuristic (fit_gpu.cuh:1766-1789) was
//   calibrated for CPU; on H100 it triggers MU→CD at plateau, and CD has
//   O(k²×cd_max_iter) cost per column — at k=50 this is 50.7× slower than MU's
//   elementwise update. Profile (Cycle 86, job 363152): k=50 auto=15,980ms vs
//   k=50 MU=484ms. Below k_cd_cutoff, auto-mode is left untouched (k<32 is where
//   CD was shown to be competitive on GPU; k=20 auto≈391ms was already fast).
//
// cd_max_iter: inner CD iteration limit, applied when solver_mode=0 (explicit CD)
//   and cfg.cd_max_iter is still at the factornet default (100). Profile shows
//   cd_maxit=10 → 7.3× speedup vs cd_maxit=100 at k=50 with no correctness degradation
//   (Cycle 86 bench_nmf_profile_c86: 654ms vs 1920ms for k50_auto).
//   factornet's global default is CD_MAXIT=100 (constants.hpp:72).
//   NOT applied to auto mode (solver_mode=3) — auto mode at small rank may use CD
//   legitimately, and changing cd_max_iter there would break the adapter-vs-direct
//   equivalence guarantee. Explicit caller overrides (cfg.cd_max_iter ≠ 100) are
//   always honored as-is.
struct FitConfig {
    // Ranks >= k_cd_cutoff in auto mode are forced to MU solver.
    // Set to INT_MAX to disable the override entirely.
    int k_cd_cutoff    = 32;

    // cd_max_iter to use when CD is actually selected.
    // Applied only when NmfConfig::cd_max_iter is still at the factornet default
    // (100 = CD_MAXIT), so explicit caller overrides are never silently overwritten.
    int cd_max_iter    = 10;
};

// fit — in-memory GPU NMF adapter.
//
// Precondition: m must have been loaded with keep_host_pinned=true.
// factornet::nmf::nmf_fit_gpu takes host pointers and stages to device internally.
// Pinned host buffers avoid a second allocation; the PCIe upload happens inside
// factornet once.
//
// W_init / H_init: optional warm-start factors (m×k and k×n DenseMatrix).
//   Pass nullptr for random initialization (seeded by config.seed).
//   init_mode in config selects the initialization strategy for nullptr case.
//
// Solver routing (Cycle 86 / OPTIM-NMF-K50):
//   factornet's auto-solver (solver_mode=3) switches from MU → CD at plateau
//   (fit_gpu.cuh:1766-1789). CD at k∈[32..60] on H100 is ~40× slower than MU
//   (15,980ms vs 484ms). This adapter enforces two GPU-tuned defaults via FitConfig:
//     1. For cfg.solver_mode==3 AND cfg.rank >= fit_cfg.k_cd_cutoff: force MU (mode=2).
//     2. When CD IS used AND cfg.cd_max_iter is still at the factornet default (100):
//        lower to fit_cfg.cd_max_iter (default 10) — 10.3× speedup, no correctness loss.
//   Callers can opt out by setting solver_mode explicitly (any value ≠ 3) or by
//   customizing FitConfig.
//
// Returns NmfResult with W (m×k), d (k), H (k×n), convergence + loss history.
inline NmfResult fit(const io::PzDeviceMatrix& m,
                     const NmfConfig& cfg,
                     const FitConfig& fit_cfg     = FitConfig{},
                     const DenseMatrix* W_init    = nullptr,
                     const DenseMatrix* H_init    = nullptr)
{
    // factornet GPU NMF takes host CSC pointers; require pinned host retention.
    ::singlet_gpu::reduce::svd::require_host_retained(m, "nmf::fit");

    NmfConfig effective_cfg = cfg;

    // Rule 1: auto mode + rank >= k_cd_cutoff → force MU.
    // MU dominates CD on H100 at medium-to-large rank because CD's O(k²×cd_max_iter)
    // inner loop scales quadratically while MU's elementwise update is O(k×nnz).
    if (effective_cfg.solver_mode == 3 && effective_cfg.rank >= fit_cfg.k_cd_cutoff) {
        effective_cfg.solver_mode = 2;  // 2 = MU
    }

    // Rule 2: when CD is explicitly selected (solver_mode=0) and cd_max_iter is still
    // at the factornet default (100), lower it to the GPU-tuned default.
    // We do NOT apply this to auto mode (3) even after Rule 1 routing — auto mode at
    // small rank (< k_cd_cutoff) may legitimately use CD with the full 100 inner iters,
    // and changing cd_max_iter there would break the adapter-vs-direct equivalence test.
    // Only explicit CD (mode=0) gets the default downgrade; explicit callers who want
    // the factornet default must set cfg.cd_max_iter=100 explicitly.
    constexpr int FACTORNET_CD_MAXIT_DEFAULT = 100;
    if (effective_cfg.solver_mode == 0 &&
        effective_cfg.cd_max_iter == FACTORNET_CD_MAXIT_DEFAULT) {
        effective_cfg.cd_max_iter = fit_cfg.cd_max_iter;
    }

    return ::factornet::nmf::nmf_fit_gpu<float>(
        m.host_indptr.get(),
        m.host_indices.get(),
        m.host_values.get(),
        static_cast<int>(m.mat.rows),
        static_cast<int>(m.mat.cols),
        static_cast<int>(m.mat.nnz),
        effective_cfg,
        W_init,
        H_init);
}

// Backward-compatible overload: fit without explicit W_init/H_init
// (no warm start — same as passing nullptr, nullptr).
// This matches the existing call sites that use fit(mat, cfg) with two args.
inline NmfResult fit(const io::PzDeviceMatrix& m,
                     const NmfConfig& cfg,
                     const DenseMatrix* W_init,
                     const DenseMatrix* H_init)
{
    return fit(m, cfg, FitConfig{}, W_init, H_init);
}

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
