// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/nmf/types.h
//
// Native NMF config and result types — replaces factornet re-exports.
// CYCLE-105: fully internal; no factornet dependency.
//
// algorithm derived from factornet/core/config.hpp and result.hpp

#pragma once

#include <cstdint>
#include <vector>
#include <cmath>

namespace singlet::gpu {
namespace reduce {
namespace nmf {

// ---------------------------------------------------------------------------
// Dense matrix type — host-side, col-major (m × n), fp32.
//
// Replaces Eigen::Matrix<float,Dynamic,Dynamic> used as W_init / H_init arg.
// Intentionally minimal: only what nmf::fit callers need (element access,
// col/row indexing, construction). No BLAS integration.
// algorithm derived from factornet/core/types.hpp DenseMatrix<float>
// ---------------------------------------------------------------------------
struct DenseMatrix {
    int m = 0, n = 0;
    std::vector<float> data;  // col-major: data[i + j*m]

    DenseMatrix() = default;
    DenseMatrix(int m_, int n_) : m(m_), n(n_), data(static_cast<std::size_t>(m_) * n_, 0.f) {}

    float& operator()(int i, int j)       { return data[i + j * m]; }
    float  operator()(int i, int j) const { return data[i + j * m]; }

    // col(j) returns a proxy that supports element access and cwiseAbs().
    struct ColProxy {
        float* ptr; int m_;
        float& operator()(int i)       { return ptr[i]; }
        float  operator()(int i) const { return ptr[i]; }
        const float* begin() const { return ptr; }
        const float* end()   const { return ptr + m_; }

        // cwiseAbs() * scale — used in nmf/init.h
        struct AbsScaled { const float* ptr; int m; float s;
            float operator()(int i) const { float v=ptr[i]; return (v<0?-v:v)*s; }
        };
        AbsScaled cwiseAbs_scaled(float s) const { return {ptr, m_, s}; }
    };

    struct ConstColProxy {
        const float* ptr; int m_;
        float operator()(int i) const { return ptr[i]; }
        const float* begin() const { return ptr; }
        const float* end()   const { return ptr + m_; }
    };

    ColProxy      col(int j)       { return ColProxy{data.data() + j*m, m}; }
    ConstColProxy col(int j) const { return ConstColProxy{data.data() + j*m, m}; }

    struct RowProxy {
        DenseMatrix* mat; int i;
        float& operator()(int j)       { return (*mat)(i, j); }
        float  operator()(int j) const { return (*mat)(i, j); }
        struct AbsScaled { DenseMatrix* mat; int i; float s;
            float operator()(int j) const { float v=(*mat)(i,j); return (v<0?-v:v)*s; }
        };
        AbsScaled cwiseAbs_scaled(float s) { return {mat, i, s}; }
        // transpose() for H.row(i) = val.transpose() pattern → no-op
        RowProxy& transpose() { return *this; }
    };

    RowProxy row(int i) { return RowProxy{this, i}; }

    bool empty() const { return m == 0 || n == 0; }
    int rows() const { return m; }
    int cols() const { return n; }

    // ptr() — returns raw float* to the underlying host data.
    // WHY not data(): the field 'data' is a std::vector<float>; a method named
    // data() would conflict with the field name in C++ lookup. Use ptr() instead.
    // _bind_kernels.hpp calls r.W.ptr() / r.H.ptr() to pass to upload_eigen_matrix.
    const float* ptr() const noexcept { return data.data(); }
          float* ptr()       noexcept { return data.data(); }
};

// DenseVector — 1-D host vector (size m).
struct DenseVector {
    int m = 0;
    std::vector<float> data;

    DenseVector() = default;
    explicit DenseVector(int m_) : m(m_), data(m_, 0.f) {}

    float& operator()(int i)       { return data[i]; }
    float  operator()(int i) const { return data[i]; }
    int size() const { return m; }

    // Accessors for _bind_kernels.hpp compat (r.d.get(), r.d.data()).
    const float* get()  const noexcept { return data.data(); }
    const float* data_ptr() const noexcept { return data.data(); }
    bool empty() const noexcept { return data.empty(); }
};

// ---------------------------------------------------------------------------
// LossType, LossConfig, FactorConfig — Frobenius (MSE) is the ONLY supported
// loss family. Per orchestrator Rule 20 (revised 2026-04-29): KL / IS / NB-GLM
// / β-divergence and other non-Frobenius losses are explicitly out of scope.
// Frobenius covers ≥95% of single-cell NMF use; obscure losses are deferred.
//
// `LossType` enum kept for ABI back-compat with `_bind_kernels.hpp`. KL and
// MAE are silently mapped to MSE; new code should not request them.
// ---------------------------------------------------------------------------
enum class LossType {
    MSE = 0,    // Frobenius — the canonical NMF loss for single-cell.
    KL  = 1,    // [DEPRECATED ABI shim] silently mapped to MSE per Rule 20.
    MAE = 2,    // [DEPRECATED ABI shim] silently mapped to MSE per Rule 20.
};

struct LossConfig {
    LossType type = LossType::MSE;   // any non-MSE value is silently downgraded.
};

// FactorConfig — per-factor regularization. Frobenius NMF supports the four
// regularizations that matter in practice (L1, L2, non-negativity, and an
// orthogonality penalty). Anything else (graph regularization, sparsity
// indicator masks, group lasso) is future work.
struct FactorConfig {
    float L1     = 0.f;    // L1 penalty weight (sparsity).
    float L2     = 0.f;    // L2 penalty weight (ridge / Tikhonov).
    float ortho  = 0.f;    // off-diagonal Gram penalty (factor decorrelation).
    bool  nonneg = true;   // non-negativity (always TRUE for NMF).
};

// ---------------------------------------------------------------------------
// NmfConfig — algorithm parameters for fit().
//
// Field names follow the long-standing NmfConfig layout for call-site stability.
// Sane GPU-tuned defaults (Rule 31).
// algorithm derived from factornet/core/config.hpp
// ---------------------------------------------------------------------------
struct NmfConfig {
    int rank        = 10;      // k (number of factors)
    int max_iter    = 200;     // outer iteration limit
    float tol       = 1e-4f;  // convergence tolerance (rel. change in loss)

    // solver_mode: 0=CD, 1=Cholesky, 2=MU, 3=auto
    // auto (3): uses MU for rank >= k_cd_cutoff (FitConfig), CD otherwise.
    int solver_mode = 3;

    // init_mode: 0=random, 1=deflation-SVD-seed
    int init_mode   = 1;

    // CD inner iteration limit
    int cd_max_iter = 100;

    // Holdout fraction for CV (used by cv_fit only).
    float holdout_fraction = 0.05f;
    int cv_patience        = 10;   // early-stopping patience (CV iters without improvement)

    // Loss type: 0=MSE (only MSE implemented natively)
    int loss_type = 0;

    // fp32 epsilon for MU denominator (prevents division by zero)
    float eps = 1e-9f;

    uint64_t seed = 0;

    // Per-factor regularization — CYCLE-114: wired into MU update kernels.
    // Flat scalar fields kept for ABI back-compat with older callers.
    // Preferred access is the sub-struct form: cfg.W.L1, cfg.H.L2, etc.
    float L1_W = 0.f, L1_H = 0.f;
    float L2_W = 0.f, L2_H = 0.f;
    bool  nonneg_W = true, nonneg_H = true;  // NMF is always non-negative

    // FactorConfig sub-structs — primary regularization interface (CYCLE-114).
    // When set, these override the flat scalars above in run_mu_nmf.
    // ortho_W / ortho_H are NOT present as flat scalars, only here.
    FactorConfig W;   // regularization applied to the W factor
    FactorConfig H;   // regularization applied to the H factor

    bool verbose = false;

    // Loss function — _bind_kernels.hpp sets cfg.loss.type.
    // KL/MAE are accepted but silently routed to MSE (native MU is MSE only).
    LossConfig loss;

    // Convenience: copy flat reg scalars into sub-structs before passing to
    // run_mu_nmf.  Called by fit() after applying solver routing.  Idempotent
    // when flat scalars are zero (the default).
    void sync_factor_configs() noexcept {
        // flat scalars win when non-zero; sub-struct fields otherwise kept.
        if (L1_W != 0.f) W.L1 = L1_W;
        if (L2_W != 0.f) W.L2 = L2_W;
        if (L1_H != 0.f) H.L1 = L1_H;
        if (L2_H != 0.f) H.L2 = L2_H;
        W.nonneg = nonneg_W;
        H.nonneg = nonneg_H;
    }
};

// ---------------------------------------------------------------------------
// NmfResult — factors + convergence history.
//
// W: m × k (host), H: k × n (host), d: k (scale factors).
// algorithm derived from factornet/core/result.hpp
// ---------------------------------------------------------------------------
struct NmfResult {
    int m = 0, n = 0, k = 0;

    DenseMatrix W;              // m × k
    DenseMatrix H;              // k × n
    DenseVector d;              // k (scale; all-ones for MU; may differ for CD)

    std::vector<float> train_loss;   // per-iter training MSE
    float best_test_loss = 1e30f;   // minimum held-out MSE (CV only)
    float test_loss      = 1e30f;   // final held-out MSE (CV only)
    int   best_iter      = 0;       // iteration that achieved best_test_loss (CV only)
    bool  converged      = false;
    int   iterations     = 0;       // count of outer MU/CD iterations executed
};

// ---------------------------------------------------------------------------
// CVNMFResult — rank-sweep CV output (speckled_cv).
// ---------------------------------------------------------------------------
struct CVNMFResult {
    std::vector<int>   k_values;
    std::vector<float> test_mse;
    std::vector<float> train_mse;
    int                chosen_k = 0;
};

}  // namespace nmf
}  // namespace reduce
}  // namespace singlet::gpu
