// SPDX-License-Identifier: MIT
// singlet/gpu/reduce/svd/types.h
//
// Native SvdConfig and SvdResult — replaces factornet type re-exports.
// CYCLE-105: fully internal; no factornet dependency.
//
// algorithm derived from factornet/core/svd_config.hpp and svd_result.hpp

#pragma once

#include <cstdint>
#include <vector>

namespace singlet::gpu {
namespace reduce {
namespace svd {

// ---------------------------------------------------------------------------
// SvdConfig — tuning knobs for all SVD backends.
//
// Sane defaults auto-route via auto_select (Rule 31).
// field names preserved for compatibility with existing call sites.
// ---------------------------------------------------------------------------
struct SvdConfig {
    int      k_max      = 10;      // desired rank
    int      max_iter   = 100;     // power-iteration / randomized-pass iters per factor
    float    tol        = 1e-5f;   // convergence tolerance on singular vector change
    bool     center     = false;   // center columns before SVD (not used for NMF init)
    bool     scale      = false;   // scale columns to unit variance before SVD
    bool     reorth     = true;    // reorthogonalize after each rank-1 extraction
    bool     verbose    = false;
    uint64_t seed       = 0;       // RNG seed for randomized SVD Gaussian sketch

    // Randomized SVD specific
    int      oversample = 10;      // extra sketch columns: sketch dim = k_max + oversample
    int      n_power_iter = 3;     // number of power iterations (q) for randomized SVD

    // Robust SVD (Huber IRLS) — set robust_delta > 0 to activate.
    float    robust_delta = 0.f;
    int      irls_max_iter = 5;
    float    irls_tol      = 1e-4f;

    // Constraint flags (passed to deflation ALS inner loop)
    bool  nonneg_u = false;
    bool  nonneg_v = false;
    float L1_u = 0.f, L1_v = 0.f;
    float L2_u = 0.f, L2_v = 0.f;
    float upper_bound_u = 0.f, upper_bound_v = 0.f;
    float L21_u = 0.f, L21_v = 0.f;
    float angular_u = 0.f, angular_v = 0.f;

    // Unused graph-regularization placeholders (kept for source compat).
    const void* graph_u = nullptr;
    const void* graph_v = nullptr;
    float       graph_u_lambda = 0.f;
    float       graph_v_lambda = 0.f;

    // Cross-validation (CYCLE-115): fraction of nonzero entries to hold out.
    float holdout_fraction = 0.05f;
};

// ---------------------------------------------------------------------------
// SvdResult — singular values + left/right singular vectors.
//
// U: m × k_selected   (host, col-major float).
// d: k_selected        (singular values, descending).
// V: n × k_selected   (host, col-major float).
//
// Stored as flat std::vector<float> for portability; the pybind11 result
// wrappers in _bind_kernels.hpp call:
//   r.d.size()   → count of singular values
//   r.d.get()    → const float* to singular value array
//   r.U.data()   → const float* to U data (m × k, col-major)
//   r.V.data()   → const float* to V data (n × k, col-major)
// and nmf/init.h calls:
//   r.U.col(j)   → column proxy with operator()(i) and cwiseAbs_scaled(s)
//   r.V.col(j)   → same for V
// ---------------------------------------------------------------------------

// DVec — proxy for the d vector satisfying both .size()/.get() (binding compat)
// and float d(i) element access (init.h compat).
struct DVec {
    std::vector<float> data_;
    std::size_t size()          const noexcept { return data_.size(); }
    const float* get()          const noexcept { return data_.data(); }
    const float* data()         const noexcept { return data_.data(); }
    float operator()(int i)     const          { return data_[static_cast<std::size_t>(i)]; }
    float operator[](int i)     const          { return data_[static_cast<std::size_t>(i)]; }
    void resize(std::size_t n)  { data_.resize(n); }
    void push_back(float v)     { data_.push_back(v); }
    float& at(int i)            { return data_[static_cast<std::size_t>(i)]; }
    bool empty()                const noexcept { return data_.empty(); }
};

struct SvdResult {
    int k_selected = 0;            // actual rank returned (≤ k_max)
    int m = 0, n = 0;

    std::vector<float> U_data;    // m × k_selected, col-major
    DVec               d;         // k_selected singular values (descending)
    std::vector<float> V_data;    // n × k_selected, col-major

    // Lightweight matrix views — nmf/init.h calls col(j).operator()(i) and
    // col(j).cwiseAbs_scaled(s); _bind_kernels.hpp calls .data() for raw ptr.
    struct UView {
        const std::vector<float>* storage = nullptr;
        int m_ = 0, k_ = 0;

        const float* data() const noexcept {
            return storage ? storage->data() : nullptr;
        }
        struct Col {
            const float* ptr; int m;
            float operator()(int i) const { return ptr[i]; }
            struct AbsScaled { const float* ptr; int m; float scale;
                float operator()(int i) const {
                    float v = ptr[i]; return (v < 0.f ? -v : v) * scale;
                }
            };
            AbsScaled cwiseAbs_scaled(float s) const { return {ptr, m, s}; }
        };
        Col col(int j) const {
            return Col{storage->data() + static_cast<std::size_t>(j) * m_, m_};
        }
    } U;

    struct VView {
        const std::vector<float>* storage = nullptr;
        int n_ = 0, k_ = 0;

        const float* data() const noexcept {
            return storage ? storage->data() : nullptr;
        }
        struct Col {
            const float* ptr; int n;
            float operator()(int i) const { return ptr[i]; }
            struct AbsScaled { const float* ptr; int n; float scale;
                float operator()(int i) const {
                    float v = ptr[i]; return (v < 0.f ? -v : v) * scale;
                }
            };
            AbsScaled cwiseAbs_scaled(float s) const { return {ptr, n, s}; }
        };
        Col col(int j) const {
            return Col{storage->data() + static_cast<std::size_t>(j) * n_, n_};
        }
    } V;

    // CV loss fields — populated only when the result comes from cv_fit.
    // test_loss: MSE on held-out (masked) entries.
    // train_loss: Frobenius residual on unmasked entries (approximation via
    //   ||A_train - U diag(d) V^T||^2_F / nnz_train, computed post-fit on device).
    float test_loss  = 1e30f;
    float train_loss = 1e30f;

    // Initialize view pointers after populating U_data/d/V_data.
    // Must be called after any resize/move of the underlying vectors.
    void finalize(int rows, int cols, int k) {
        m = rows; n = cols; k_selected = k;
        U.storage = &U_data; U.m_ = rows; U.k_ = k;
        V.storage = &V_data; V.n_ = cols; V.k_ = k;
    }
};

// ---------------------------------------------------------------------------
// CVSVDResult — rank-sweep CV output (speckled_cv).
// Mirrors nmf::CVNMFResult; chosen_k is argmin(test_mse).
// ---------------------------------------------------------------------------
struct CVSVDResult {
    std::vector<int>   k_values;
    std::vector<float> train_mse;   // residual on unmasked entries
    std::vector<float> test_mse;    // residual on held-out (masked) entries
    int                chosen_k = 0;
};

// Compat alias — existing code may reference SvdMethod.
enum class SvdMethod { DEFLATION, RANDOMIZED };

}  // namespace svd
}  // namespace reduce
}  // namespace singlet::gpu
