// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/core/config.hpp, factornet/core/result.hpp,
//             factornet/core/factor_config.hpp, factornet/core/types.hpp,
//             factornet/math/loss.hpp, factornet/nmf/speckled_cv.hpp
//
// Type re-exports for the singlet_gpu::reduce::nmf namespace.
// Every field, default, and helper lives in factornet — we expose them under
// a short singlet_gpu alias so adapter headers stay clean.
//
// Correction vs design doc:
//   - SolverMode / InitMode are NOT enum types in factornet; they are int fields
//     (NMFConfig::solver_mode, NMFConfig::init_mode).  No type alias for them.
//   - LossType lives in factornet:: (math/loss.hpp), NOT factornet::nmf::.
//   - FactorConfig lives in factornet:: (core/factor_config.hpp), NOT factornet::nmf::.
//   - DenseMatrix lives in factornet:: (core/types.hpp) as a template alias.
//
// OOC plan: chunked NMF via singlet_gpu::io::PzDataLoader (streaming/pz_data_loader.h).

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/core/config.hpp>       // NMFConfig<Scalar>
#include <factornet/core/result.hpp>       // NMFResult<Scalar>
#include <factornet/core/factor_config.hpp>// FactorConfig<Scalar>
#include <factornet/core/types.hpp>        // DenseMatrix<Scalar>, DenseVector<Scalar>
#include <factornet/math/loss.hpp>         // LossType, LossConfig<Scalar>
#include <factornet/nmf/speckled_cv.hpp>   // LazySpeckledMask<Scalar>

namespace singlet_gpu {
namespace reduce {
namespace nmf {

// Core config / result — passed by callers for every NMF entry point.
using NmfConfig    = ::factornet::NMFConfig<float>;
using NmfResult    = ::factornet::NMFResult<float>;

// Per-factor regularization and constraints (embedded in NmfConfig as .W / .H).
using FactorConfig = ::factornet::FactorConfig<float>;

// Dense matrix type: Eigen::Matrix<float, Dynamic, Dynamic> (col-major).
// W_init / H_init arguments use this type.
using DenseMatrix  = ::factornet::DenseMatrix<float>;
using DenseVector  = ::factornet::DenseVector<float>;

// Loss function selector — set via config.loss.type.
// Values: MSE, MAE, HUBER, KL, GP, NB, GAMMA, INVGAUSS, TWEEDIE.
// Non-MSE losses use host-mediated IRLS per column (factornet behavior).
using LossType     = ::factornet::LossType;
using LossConfig   = ::factornet::LossConfig<float>;

// Speckled cross-validation mask for auto-rank determination.
// Used by orchestrator to run nmf_cv_fit_gpu with held-out entries.
using LazySpeckledMask = ::factornet::nmf::LazySpeckledMask<float>;

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet_gpu
