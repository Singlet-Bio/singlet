// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: singlet_gpu::reduce::svd::deflation (primary)
//             singlet_gpu::reduce::svd::randomized (fallback)
//
// Adapter: singlet_gpu::reduce::svd::auto_select
//
// Rule 32 adopt-winner result (2026-04-16 benchmark):
//   Deflation:   28 ms, k-independent wall time — winner across all k and scales.
//   Randomized:  kept as fallback; lowest memory footprint for >1M-cell OOC chunks.
//   Lanczos:     REMOVED (dominated by deflation at k < 32).
//   IRLBA:       REMOVED (dominated by deflation at k ≥ 64).
//   Krylov:      REMOVED (dominated by deflation for constrained SVD).
//
// Routing table (singlet-gpu owned — no longer delegates to factornet::auto_select_svd_method):
//   DEFAULT:                    → DEFLATION (primary, all k, all scales)
//   DEFLATION returns empty d:  → RANDOMIZED (automatic fallback on degenerate input)
//
// There is no user-visible backend selection via Config.  To use randomized
// directly, call singlet_gpu::reduce::svd::randomized() explicitly.
//
// has_constraints is preserved for caller convenience; no longer changes routing.
// Deflation handles all constraint types (L1, L2, non-neg, upper bound, L21, angular, graph).
//
// Streams: each backend creates a GPUContext internally; no stream overlap.
// Precision: delegated to the selected backend (all fp32 default).
// Determinism: delegated to the selected backend (cfg.seed forwarded).
//
// OOC plan: delegated to the selected backend; streaming deferred to cycle 7.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <singlet-gpu/reduce/svd/deflation.h>
#include <singlet-gpu/reduce/svd/randomized.h>

namespace singlet_gpu {
namespace reduce {
namespace svd {

// has_constraints — inspect SvdConfig for any active constraint.
// Preserved for caller convenience; no longer changes routing (deflation
// handles all constraint types that krylov_constrained previously handled).
inline bool has_constraints(const SvdConfig& cfg) noexcept {
    return cfg.nonneg_u || cfg.nonneg_v
        || cfg.L1_u != 0.f  || cfg.L1_v != 0.f
        || cfg.L2_u != 0.f  || cfg.L2_v != 0.f
        || cfg.upper_bound_u > 0.f || cfg.upper_bound_v > 0.f
        || cfg.L21_u != 0.f || cfg.L21_v != 0.f
        || cfg.angular_u != 0.f || cfg.angular_v != 0.f
        || cfg.graph_u != nullptr || cfg.graph_v != nullptr;
}

// auto_select — routes to deflation (primary) or randomized (automatic fallback).
//
// k: desired rank.  cfg.k_max is overwritten with k in the forwarded copy so
//    callers do not need to set it separately.
//
// Fallback to randomized occurs ONLY when deflation returns an empty result
// (d.size() == 0), which guards against factornet deflation failure on
// degenerate inputs.  This is NOT a user-configurable choice — call
// singlet_gpu::reduce::svd::randomized() directly to use that backend.
inline SvdResult auto_select(const io::PzDeviceMatrix& m, int k, const SvdConfig& cfg) {
    require_host_retained(m, "auto_select");

    // Copy config so we can set k_max without mutating caller's struct.
    SvdConfig cfg_k = cfg;
    cfg_k.k_max = k;

    // Primary path: deflation (winner across all k and scales per Rule 32).
    SvdResult result = deflation(m, cfg_k);

    // Automatic fallback: if deflation produces no singular values on a degenerate
    // input, fall through to randomized rather than returning an empty result.
    if (result.d.size() == 0) {
        result = randomized(m, cfg_k);
    }

    return result;
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
