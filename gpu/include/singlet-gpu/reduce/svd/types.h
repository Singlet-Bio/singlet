// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/svd/auto_select.hpp, factornet/core/svd_config.hpp,
//             factornet/core/svd_result.hpp — type re-exports only
//
// Provides singlet_gpu::reduce::svd::{SvdConfig, SvdResult, SvdMethod} as
// thin aliases for the factornet types.  All algorithm configuration (k_max,
// tol, constraints, robust_delta, centering) lives on factornet::SVDConfig<float>.
//
// NOTE: the rank field is k_max, not k.  Set cfg.k_max = desired_rank before
// passing to any adapter.  Design doc §"Adapter headers" says "cfg.k" but
// the factornet struct uses k_max; adapters use k_max throughout.
//
// OOC plan: factornet SVD does not stream; for >1M-cell batches the caller
// must chunk via PzChunkIterator and run incremental SVD.  Deferred to cycle 7.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/svd/auto_select.hpp>   // SVDMethod + auto_select_svd_method
#include <factornet/core/svd_config.hpp>   // SVDConfig<Scalar>
#include <factornet/core/svd_result.hpp>   // SVDResult<Scalar>
// NOTE: auto_select.hpp does NOT include svd_config/svd_result transitively,
// so both core headers are listed explicitly.

namespace singlet_gpu {
namespace reduce {
namespace svd {

// Re-export factornet types under the singlet_gpu namespace.
// All fields, helpers, and defaults are inherited verbatim — we do not extend.
using SvdConfig = ::factornet::SVDConfig<float>;
using SvdResult = ::factornet::SVDResult<float>;
using SvdMethod = ::factornet::svd::SVDMethod;

}  // namespace svd
}  // namespace reduce
}  // namespace singlet_gpu
