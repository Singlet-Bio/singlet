// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/reduce/svd::auto_select
//
// Routes to deflation (primary) or randomized (automatic fallback).
// CYCLE-105: both backends are now native — no factornet dependency.
//
// Routing table (Rule 32 adopt-winner, 2026-04-16 benchmark):
//   DEFAULT:                    → DEFLATION (winner at all k, all scales)
//   DEFLATION returns empty d:  → RANDOMIZED (fallback on degenerate input)
//
// cfg.k_max is overwritten with k in the forwarded copy.
// has_constraints is preserved for caller convenience.

#pragma once

#include <singlet-gpu/reduce/svd/deflation.h>
#include <singlet-gpu/reduce/svd/randomized.h>

namespace singlet_gpu {
namespace reduce {
namespace svd {

// has_constraints — inspect SvdConfig for any active constraint.
// No longer changes routing (deflation handles all constraint types).
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
// k: desired rank. cfg.k_max is overwritten with k in the forwarded copy so
//    callers do not need to set it separately.
//
// Fallback to randomized occurs ONLY when deflation returns an empty result
// (k_selected == 0), which guards against degenerate inputs.
inline SvdResult auto_select(const io::PzDeviceMatrix& m, int k, const SvdConfig& cfg) {
    SvdConfig cfg_k = cfg;
    cfg_k.k_max = k;

    SvdResult result = deflation(m, cfg_k);

    if (result.k_selected == 0) {
        result = randomized(m, cfg_k);
    }

    return result;
}

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
