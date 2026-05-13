#pragma once
// singlet-pileup: donor_demux.h
// Variational Bayes binomial mixture model for donor demultiplexing.
// C++ replacement for vireoSNP — works directly on in-memory AD/DP CSC matrices.
//
// Algorithm: Two-phase K-selection + VB coordinate ascent on Beta-Binomial mixture.
//   Phase 1: Fast EM+BIC on a cell subsample to narrow K to 2-3 candidates.
//   Phase 2: Full VB (flat buffers, parallel E/M-step) only on candidate K values.
//
// Performance optimisations vs. original:
//   1. EM+BIC Phase 1 reduces K sweep from 1..8 full VB runs to 2-3 (3-4x speedup).
//   2. Flat s1/s2/log_lik buffers eliminate O(ns) heap allocs per iteration.
//   3. Parallel M-step and E-step across cfg.threads threads (final refinement pass).
//   4. K-level early stopping when ELBO degrades for 2 consecutive K values.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace singlet {

// ============================================================================
// Configuration
// ============================================================================
struct DonorDemuxConfig {
    int n_donors = -1;          // -1 = auto-detect (try 1..max_donors)
    int max_donors = 8;
    int max_iter = 200;
    int max_iter_pre = 100;     // pre-iterations per random restart
    int n_inits = 10;           // random restarts (vireo default)
    int min_iter = 20;          // minimum iterations before convergence check
    float eps_conv = 1e-2f;     // absolute ELBO convergence threshold
    int threads = 4;
    uint64_t seed = 42;
    float min_prob = 0.9f;       // min posterior for confident assignment
    // ── Phase 1 EM+BIC K-selection ─────────────────────────────────────────
    int subsample_cells = 500;   // cells used in Phase 1 EM (0 = use all)
    int em_max_iter = 50;        // EM iterations per K candidate in Phase 1
    bool fast_k_select = true;   // true = EM+BIC Phase 1; false = old full sweep
};

// ============================================================================
// Per-cell output
// ============================================================================
struct DonorAssignment {
    int donor_id;           // 0..K-1, or K=doublet, K+1=unassigned
    float prob_max;         // posterior P(z_c = donor_id)
    float prob_doublet;     // posterior P(doublet)
    std::string label;      // "donor0", "donor1", ..., "doublet", "unassigned"
    std::string cell_status = "cell"; // "cell" (EmptyDrops-called) or "ambient" (back-assigned)
};

// ============================================================================
// Digamma approximation — asymptotic series, accurate for x > 0.5
// ============================================================================
static inline float digamma(float x) {
    // Boost x to > 6 using unrolled if-chain (eliminates branch misprediction
    // from while loop in tight VB inner loop)
    float result = 0.0f;
    if (x < 1.0f) { result -= 1.0f / x; x += 1.0f; }
    if (x < 2.0f) { result -= 1.0f / x; x += 1.0f; }
    if (x < 3.0f) { result -= 1.0f / x; x += 1.0f; }
    if (x < 4.0f) { result -= 1.0f / x; x += 1.0f; }
    if (x < 5.0f) { result -= 1.0f / x; x += 1.0f; }
    if (x < 6.0f) { result -= 1.0f / x; x += 1.0f; }
    // Asymptotic: psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - ...
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    result += std::log(x) - 0.5f * inv
              - inv2 * (1.0f/12.0f - inv2 * (1.0f/120.0f - inv2 / 252.0f));
    return result;
}

// ============================================================================
// Internal: sparse AD/DP representation for covered SNPs only
// ============================================================================
struct CoveredSNPData {
    // For each cell c, for each covered SNP j (in reduced index):
    // ad[c][k] and dp[c][k] where k is position in cell's sparse list
    struct CellSNPs {
        std::vector<int32_t> snp_idx;  // reduced SNP index (into covered_snps)
        std::vector<uint8_t> ad;       // alt depth
        std::vector<uint8_t> dp;       // total depth
    };
    std::vector<CellSNPs> cells;
    int n_covered_snps;
    int n_cells;
    std::vector<int32_t> covered_to_original; // [n_covered_snps] → original SNP index
};

// Extract covered SNPs from CSC AD/DP matrices
static CoveredSNPData extract_covered_snps(
    uint32_t n_features, uint32_t n_barcodes,
    const int32_t* ad_indptr, const int32_t* ad_indices, const uint8_t* ad_data,
    const int32_t* dp_indptr, const int32_t* dp_indices, const uint8_t* dp_data)
{
    // Find SNPs with any coverage
    std::vector<bool> has_coverage(n_features, false);
    for (uint32_t j = 0; j < n_barcodes; ++j) {
        for (int32_t k = dp_indptr[j]; k < dp_indptr[j+1]; ++k) {
            if (dp_data[k] > 0)
                has_coverage[dp_indices[k]] = true;
        }
    }

    // Build covered SNP index mapping
    std::vector<int32_t> snp_to_reduced(n_features, -1);
    int n_covered = 0;
    for (uint32_t i = 0; i < n_features; ++i)
        if (has_coverage[i]) snp_to_reduced[i] = n_covered++;

    CoveredSNPData result;
    result.n_covered_snps = n_covered;
    result.n_cells = n_barcodes;
    result.cells.resize(n_barcodes);
    result.covered_to_original.reserve(n_covered);
    for (uint32_t i = 0; i < n_features; ++i)
        if (has_coverage[i]) result.covered_to_original.push_back(static_cast<int32_t>(i));

    // Build per-cell sparse representation
    for (uint32_t j = 0; j < n_barcodes; ++j) {
        auto& cell = result.cells[j];

        // Merge AD and DP for this cell
        // DP drives the loop (DP >= AD at every position)
        for (int32_t k = dp_indptr[j]; k < dp_indptr[j+1]; ++k) {
            int32_t snp = dp_indices[k];
            uint8_t dp_val = dp_data[k];
            if (dp_val == 0) continue;
            int32_t ridx = snp_to_reduced[snp];
            if (ridx < 0) continue;
            cell.snp_idx.push_back(ridx);
            cell.dp.push_back(dp_val);
            cell.ad.push_back(0);  // fill AD below
        }

        // Fill AD values
        if (!cell.snp_idx.empty()) {
            // Build quick lookup for this cell's snp positions
            std::vector<int32_t> pos_map(cell.snp_idx.size());
            std::iota(pos_map.begin(), pos_map.end(), 0);

            // Scan AD entries for this cell
            for (int32_t k = ad_indptr[j]; k < ad_indptr[j+1]; ++k) {
                int32_t snp = ad_indices[k];
                int32_t ridx = snp_to_reduced[snp];
                if (ridx < 0) continue;
                // Find ridx in cell.snp_idx (sorted, so binary search)
                auto it = std::lower_bound(cell.snp_idx.begin(), cell.snp_idx.end(), ridx);
                if (it != cell.snp_idx.end() && *it == ridx) {
                    size_t pos = it - cell.snp_idx.begin();
                    cell.ad[pos] = ad_data[k];
                }
            }
        }
    }

    return result;
}

// ============================================================================
// Phase 1: Fast K-selection via EM+BIC on a cell subsample
// ============================================================================

// EMResult only carries BIC + log-likelihood (no allele freq needed externally)
struct EMResult {
    double log_likelihood = 0.0;
    double bic = 0.0;
};

// Run plain maximum-likelihood EM for K donors on a cell subset.
// Uses Beta(0.5, 0.5) conjugate prior on allele frequencies for stability.
// No digamma — only log/exp; much cheaper than VB.
static EMResult run_em(
    const CoveredSNPData& data,
    int K,
    const std::vector<int>& cell_subset,
    int max_iter,
    uint64_t seed)
{
    const int N  = static_cast<int>(cell_subset.size());
    const int ns = data.n_covered_snps;
    if (N == 0 || ns == 0 || K <= 0) { return {}; }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(0.1f, 0.9f);

    // Allele frequencies: flat [K × ns], indexed [k * ns + j]
    std::vector<float> f(K * ns);
    for (auto& v : f) v = unif(rng);

    // Mixing proportions
    std::vector<float> pi(K, 1.0f / K);

    // Responsibilities: [N × K], and M-step buffers (allocated once)
    std::vector<float> r(N * K);
    std::vector<float> f_num(K * ns);
    std::vector<float> f_den(K * ns);
    // Precomputed log(f) and log(1-f): updated after each M-step
    // so E-step inner loop avoids O(N × snps/cell × K) log() calls
    std::vector<float> log_f(K * ns);
    std::vector<float> log_1mf(K * ns);

    // Initial log_f computation
    auto update_logf = [&]() {
        for (int k = 0; k < K; ++k) {
            const float* fk  = f.data()      + k * ns;
            float*       lfk = log_f.data()  + k * ns;
            float*       lmk = log_1mf.data() + k * ns;
            for (int j = 0; j < ns; ++j) {
                const float fv = std::max(1e-6f, std::min(1.0f - 1e-6f, fk[j]));
                lfk[j] = std::log(fv);
                lmk[j] = std::log(1.0f - fv);
            }
        }
    };
    update_logf();

    double prev_ll = -std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < max_iter; ++iter) {
        // ── E-step ─────────────────────────────────────────────────────────
        double ll = 0.0;
        for (int n = 0; n < N; ++n) {
            const auto& cell = data.cells[cell_subset[n]];
            float* rn = r.data() + n * K;

            float max_log = -1e30f;
            for (int k = 0; k < K; ++k) {
                float lp = std::log(pi[k] + 1e-30f);
                const float* lfk  = log_f.data()   + k * ns;
                const float* lmfk = log_1mf.data() + k * ns;
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    const int   j  = cell.snp_idx[si];
                    const float ad = static_cast<float>(cell.ad[si]);
                    const float bd = static_cast<float>(cell.dp[si]) - ad;
                    lp += ad * lfk[j] + bd * lmfk[j];  // table lookup, no log()
                }
                rn[k] = lp;
                if (lp > max_log) max_log = lp;
            }

            float sm = 0.0f;
            for (int k = 0; k < K; ++k) { rn[k] = std::exp(rn[k] - max_log); sm += rn[k]; }
            ll += std::log(sm) + static_cast<double>(max_log);
            const float inv = 1.0f / sm;
            for (int k = 0; k < K; ++k) rn[k] *= inv;
        }

        // ── M-step ─────────────────────────────────────────────────────────
        std::fill(f_num.begin(), f_num.end(), 0.5f);  // Beta(0.5, 0.5) prior
        std::fill(f_den.begin(), f_den.end(), 1.0f);
        std::fill(pi.begin(),    pi.end(),    0.0f);

        for (int n = 0; n < N; ++n) {
            const auto& cell = data.cells[cell_subset[n]];
            const float* rn  = r.data() + n * K;
            for (int k = 0; k < K; ++k) {
                pi[k] += rn[k];
                const float rk = rn[k];
                if (rk < 1e-10f) continue;
                float* fn = f_num.data() + k * ns;
                float* fd = f_den.data() + k * ns;
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    const int j = cell.snp_idx[si];
                    fn[j] += rk * static_cast<float>(cell.ad[si]);
                    fd[j] += rk * static_cast<float>(cell.dp[si]);
                }
            }
        }
        const float invN = 1.0f / N;
        for (int k = 0; k < K; ++k) {
            pi[k] *= invN;
            float*       fk = f.data()     + k * ns;
            const float* fn = f_num.data() + k * ns;
            const float* fd = f_den.data() + k * ns;
            for (int j = 0; j < ns; ++j) fk[j] = fn[j] / fd[j];
        }
        update_logf();  // refresh log(f) cache after M-step

        if (iter > 2 && std::abs(ll - prev_ll) < 1e-4 * N) break;
        prev_ll = ll;
    }

    // BIC = -2*LL + n_params * log(N)
    const int n_params = K * ns + (K - 1);
    EMResult res;
    res.log_likelihood = prev_ll;
    res.bic = -2.0 * prev_ll + static_cast<double>(n_params) * std::log(static_cast<double>(N));
    return res;
}

// Select K candidates via EM+BIC on a random subsample.
// Returns {K*-1, K*, K*+1} ∩ [k_min, k_max] — typically 2-3 values.
static std::vector<int> select_k_bic(
    const CoveredSNPData& data,
    int k_min, int k_max,
    const DonorDemuxConfig& cfg)
{
    const int n_cells = data.n_cells;
    const int sub_n   = (cfg.subsample_cells > 0)
                        ? std::min(cfg.subsample_cells, n_cells)
                        : n_cells;

    std::vector<int> idx(n_cells);
    std::iota(idx.begin(), idx.end(), 0);
    if (sub_n < n_cells) {
        std::mt19937 rng(cfg.seed + 99999ULL);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    const std::vector<int> subset(idx.begin(), idx.begin() + sub_n);

    const int n_inits = std::min(3, cfg.n_inits);
    double best_bic = std::numeric_limits<double>::infinity();
    int    best_k   = k_min;

    for (int K = k_min; K <= k_max; ++K) {
        double k_best_bic = std::numeric_limits<double>::infinity();
        for (int init = 0; init < n_inits; ++init) {
            const uint64_t s = cfg.seed
                             + static_cast<uint64_t>(init) * 7777ULL
                             + static_cast<uint64_t>(K)    * 12345ULL;
            const EMResult em = run_em(data, K, subset, cfg.em_max_iter, s);
            if (em.bic < k_best_bic) k_best_bic = em.bic;
        }
        std::cerr << "[demux] EM K=" << K << " BIC=" << k_best_bic << "\n";
        if (k_best_bic < best_bic) { best_bic = k_best_bic; best_k = K; }
    }

    // Return K*±1 safety window (protects against small BIC estimation errors)
    std::vector<int> cands;
    for (int K = std::max(k_min, best_k - 1); K <= std::min(k_max, best_k + 1); ++K)
        cands.push_back(K);
    return cands;
}

// ============================================================================
// VB inference for a fixed K — faithful port of vireoSNP BinomMixtureVB
// ============================================================================
struct VBResult {
    double elbo;
    std::vector<std::vector<float>> id_prob;      // [n_cells][K]
    std::vector<std::vector<float>> beta_mu;       // [n_covered_snps][K]
    std::vector<std::vector<float>> beta_sum;      // [n_covered_snps][K]
    int n_iter;
};

// theta_s1 = beta_mu * beta_sum (= alpha of Beta)
// theta_s2 = (1 - beta_mu) * beta_sum (= beta of Beta)
// Prior: beta_mu_prior = 0.5, beta_sum_prior = 2.0 → s1_prior = 1.0, s2_prior = 1.0

// ── Core VB iteration on a pre-initialised VBResult (in-place) ──
// Flat s1[j*K+k] / s2[j*K+k] buffers — no per-iteration heap allocation.
// Parallel M-step and E-step when nthreads > 1.
static void run_vb_inplace(
    const CoveredSNPData& data, int K,
    const DonorDemuxConfig& cfg, VBResult& result,
    int max_iter, int nthreads)
{
    const int nc = data.n_cells;
    const int ns = data.n_covered_snps;
    const float s1_prior = 1.0f, s2_prior = 1.0f;
    nthreads = std::max(1, nthreads);

    // Pre-allocate flat working buffers (one allocation, reused every iter)
    std::vector<float> s1(ns * K);
    std::vector<float> s2(ns * K);
    // Digamma cache: precomputed once per M-step so the cell loop uses table lookups
    std::vector<float> d1(ns * K);   // digamma(s1[j*K+k])
    std::vector<float> d2(ns * K);   // digamma(s2[j*K+k])
    std::vector<float> d12(ns * K);  // digamma(s1[j*K+k] + s2[j*K+k])

    // Thread-local M-step accumulators (allocated once, cleared per iter)
    std::vector<std::vector<float>> s1_tl(nthreads, std::vector<float>(ns * K));
    std::vector<std::vector<float>> s2_tl(nthreads, std::vector<float>(ns * K));

    float prev_elbo = static_cast<float>(result.elbo);
    const float log_inv_k  = -std::log(static_cast<float>(K));
    const float lgamma_prior = std::lgamma(s1_prior + s2_prior)
                             - std::lgamma(s1_prior) - std::lgamma(s2_prior);

    for (int iter = 0; iter < max_iter; ++iter) {

        // ── M-step ─────────────────────────────────────────────────────────
        if (nthreads == 1) {
            std::fill(s1.begin(), s1.end(), s1_prior);
            std::fill(s2.begin(), s2.end(), s2_prior);
            for (int c = 0; c < nc; ++c) {
                const auto& cell = data.cells[c];
                const float* qc  = result.id_prob[c].data();
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    const int   j  = cell.snp_idx[si];
                    const float ad = static_cast<float>(cell.ad[si]);
                    const float bd = static_cast<float>(cell.dp[si]) - ad;
                    float* s1j = s1.data() + j * K;
                    float* s2j = s2.data() + j * K;
                    for (int k = 0; k < K; ++k) {
                        s1j[k] += qc[k] * ad;
                        s2j[k] += qc[k] * bd;
                    }
                }
            }
        } else {
            // Each thread accumulates into its own s1_tl/s2_tl; no mutex needed.
            auto m_worker = [&](int t, int c_start, int c_end) {
                float* s1t = s1_tl[t].data();
                float* s2t = s2_tl[t].data();
                std::fill(s1t, s1t + ns * K, 0.0f);
                std::fill(s2t, s2t + ns * K, 0.0f);
                for (int c = c_start; c < c_end; ++c) {
                    const auto& cell = data.cells[c];
                    const float* qc  = result.id_prob[c].data();
                    for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                        const int   j  = cell.snp_idx[si];
                        const float ad = static_cast<float>(cell.ad[si]);
                        const float bd = static_cast<float>(cell.dp[si]) - ad;
                        float* s1j = s1t + j * K;
                        float* s2j = s2t + j * K;
                        for (int k = 0; k < K; ++k) {
                            s1j[k] += qc[k] * ad;
                            s2j[k] += qc[k] * bd;
                        }
                    }
                }
            };
            {
                std::vector<std::thread> ts;
                ts.reserve(nthreads);
                for (int t = 0; t < nthreads; ++t) {
                    const int cs = t * nc / nthreads;
                    const int ce = (t + 1) * nc / nthreads;
                    ts.emplace_back(m_worker, t, cs, ce);
                }
                for (auto& t : ts) t.join();
            }
            // Reduce
            for (int i = 0; i < ns * K; ++i) {
                float a = s1_prior, b = s2_prior;
                for (int t = 0; t < nthreads; ++t) { a += s1_tl[t][i]; b += s2_tl[t][i]; }
                s1[i] = a;  s2[i] = b;
            }
        }

        // Update beta_mu / beta_sum (stored for output)
        for (int j = 0; j < ns; ++j) {
            const float* s1j = s1.data() + j * K;
            const float* s2j = s2.data() + j * K;
            for (int k = 0; k < K; ++k) {
                result.beta_mu[j][k]  = s1j[k] / (s1j[k] + s2j[k]);
                result.beta_sum[j][k] = s1j[k] + s2j[k];
            }
        }

        // ── Precompute digamma cache (KEY OPTIMISATION) ─────────────────────
        // Costs O(ns × K) digamma calls once per iter instead of O(nc × snps/cell × K).
        // For ns=5000 K=5: 75K calls vs 67.5M calls in the old inner loop. ~900× speedup.
        for (int j = 0; j < ns; ++j) {
            const float* s1j = s1.data() + j * K;
            const float* s2j = s2.data() + j * K;
            float* d1j  = d1.data()  + j * K;
            float* d2j  = d2.data()  + j * K;
            float* d12j = d12.data() + j * K;
            for (int k = 0; k < K; ++k) {
                d1j[k]  = digamma(s1j[k]);
                d2j[k]  = digamma(s2j[k]);
                d12j[k] = digamma(s1j[k] + s2j[k]);
            }
        }

        // ── E-step ─────────────────────────────────────────────────────────
        auto e_worker = [&](int c_start, int c_end) {
            std::vector<float> ll(K);  // thread-local log-lik buffer; no alloc per cell
            const float inv_k = 1.0f / K;
            for (int c = c_start; c < c_end; ++c) {
                const auto& cell = data.cells[c];
                if (cell.snp_idx.empty()) {
                    for (int k = 0; k < K; ++k) result.id_prob[c][k] = inv_k;
                    continue;
                }
                std::fill(ll.begin(), ll.end(), 0.0f);
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    const int   j  = cell.snp_idx[si];
                    const float ad = static_cast<float>(cell.ad[si]);
                    const float dp = static_cast<float>(cell.dp[si]);
                    const float bd = dp - ad;
                    const float* d1j  = d1.data()  + j * K;  // precomputed digamma(s1)
                    const float* d2j  = d2.data()  + j * K;
                    const float* d12j = d12.data() + j * K;
                    for (int k = 0; k < K; ++k) {
                        ll[k] += ad * d1j[k] + bd * d2j[k] - dp * d12j[k];
                    }
                }
                float mx = ll[0];
                for (int k = 1; k < K; ++k) if (ll[k] > mx) mx = ll[k];
                float sm = 0.0f;
                float* idp = result.id_prob[c].data();
                for (int k = 0; k < K; ++k) { idp[k] = std::exp(ll[k] - mx); sm += idp[k]; }
                const float inv = 1.0f / sm;
                for (int k = 0; k < K; ++k) idp[k] *= inv;
            }
        };

        if (nthreads == 1) {
            e_worker(0, nc);
        } else {
            std::vector<std::thread> ts;
            ts.reserve(nthreads);
            for (int t = 0; t < nthreads; ++t) {
                const int cs = t * nc / nthreads;
                const int ce = (t + 1) * nc / nthreads;
                ts.emplace_back(e_worker, cs, ce);
            }
            for (auto& t : ts) t.join();
        }

        // ── ELBO ───────────────────────────────────────────────────────────
        double elbo = 0.0;

        // E[log p(x | z, theta)] — uses precomputed d1/d2/d12 (no digamma calls here)
        for (int c = 0; c < nc; ++c) {
            const auto& cell = data.cells[c];
            for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                const int   j  = cell.snp_idx[si];
                const float ad = static_cast<float>(cell.ad[si]);
                const float dp = static_cast<float>(cell.dp[si]);
                const float bd = dp - ad;
                const float* d1j  = d1.data()  + j * K;
                const float* d2j  = d2.data()  + j * K;
                const float* d12j = d12.data() + j * K;
                for (int k = 0; k < K; ++k) {
                    const float qk = result.id_prob[c][k];
                    if (qk < 1e-10f) continue;
                    elbo += qk * (ad * d1j[k] + bd * d2j[k] - dp * d12j[k]);
                }
            }
        }
        // -KL[q(z) || p(z)]
        for (int c = 0; c < nc; ++c) {
            for (int k = 0; k < K; ++k) {
                const float qk = result.id_prob[c][k];
                if (qk > 1e-10f)
                    elbo -= qk * (std::log(qk) + log_inv_k);
            }
        }
        // -KL[q(theta) || p(theta)] — uses precomputed d1/d2 (3 lgamma + d1 + d2 + d12 per j,k)
        for (int j = 0; j < ns; ++j) {
            const float* s1j  = s1.data()  + j * K;
            const float* s2j  = s2.data()  + j * K;
            const float* d1j  = d1.data()  + j * K;
            const float* d2j  = d2.data()  + j * K;
            const float* d12j = d12.data() + j * K;
            for (int k = 0; k < K; ++k) {
                const float a = s1j[k], b = s2j[k];
                elbo -= std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                       - lgamma_prior
                       + (a - s1_prior) * d1j[k] + (b - s2_prior) * d2j[k]
                       - (a + b - s1_prior - s2_prior) * d12j[k];
            }
        }

        result.elbo = elbo;
        result.n_iter++;

        if (iter >= cfg.min_iter) {
            if (static_cast<float>(elbo - prev_elbo) < cfg.eps_conv) break;
        }
        prev_elbo = static_cast<float>(elbo);
    }
}

// ============================================================================
// PCA-seeded VB initialization — fixes symmetric-init cluster collapse
//
// Algorithm (mirrors vireoSNP's K-means seeding):
//   Step 1: find informative SNPs (≥20 cells with DP≥2, pop AF ∈ [0.1, 0.9])
//   Step 2: power-iteration PCA on centered per-cell allele-frequency profiles
//   Step 3: K-means++ on K-1 PCA scores → K cluster assignments
//   Step 4: seed beta_mu from cluster-mean AFs; id_prob from soft membership
// ============================================================================

struct InformativeSNPs {
    std::vector<int>   indices;   // indices into CoveredSNPData covered-SNP space
    std::vector<float> mean_af;   // population-level mean ALT frequency
};

// Returns SNPs with ≥ min_cell_dp cells having DP ≥ 2 and pop AF ∈ [min_af, max_af].
static InformativeSNPs find_informative_snps(
    const CoveredSNPData& data,
    int   min_cell_dp = 20,
    float min_af      = 0.1f,
    float max_af      = 0.9f)
{
    const int ns = data.n_covered_snps;
    const int nc = data.n_cells;
    std::vector<float>   total_ad(ns, 0.0f), total_dp(ns, 0.0f);
    std::vector<int32_t> n_cov(ns, 0);
    for (int c = 0; c < nc; ++c) {
        const auto& cell = data.cells[c];
        for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
            int j = cell.snp_idx[si];
            if (cell.dp[si] >= 2) {
                ++n_cov[j];
                total_ad[j] += cell.ad[si];
                total_dp[j] += cell.dp[si];
            }
        }
    }
    InformativeSNPs res;
    for (int j = 0; j < ns; ++j) {
        if (n_cov[j] < min_cell_dp) continue;
        float af = total_dp[j] > 0.0f ? total_ad[j] / total_dp[j] : 0.0f;
        if (af < min_af || af > max_af) continue;
        res.indices.push_back(j);
        res.mean_af.push_back(af);
    }
    return res;
}

// Power-iteration PCA on mean-centered cell allele-frequency profiles.
// X_centered[c][j] = AF[c][j] - mean_af[j] for covered (c,j), 0 otherwise.
// Uses Gram-Schmidt deflation for multiple PCs.
// Returns cell scores [nc × n_pcs] in cell-major layout.
static std::vector<float> power_iteration_pca(
    const CoveredSNPData& data,
    const InformativeSNPs& inf,
    int         n_pcs,
    int         n_iter = 30,
    uint64_t    seed   = 0)
{
    const int nc = data.n_cells;
    const int ni = static_cast<int>(inf.indices.size());
    const int n_pcs_actual = std::min(n_pcs, std::min(nc - 1, ni));
    std::vector<float> result(static_cast<size_t>(nc) * n_pcs, 0.0f);
    if (n_pcs_actual <= 0 || ni == 0 || nc < 2) return result;

    // Build covered_snp → informative-index lookup (-1 if not informative)
    std::vector<int> snp_to_inf(data.n_covered_snps, -1);
    for (int i = 0; i < ni; ++i) snp_to_inf[inf.indices[i]] = i;

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<std::vector<float>> evecs(n_pcs_actual, std::vector<float>(ni, 0.0f));
    std::vector<float> v(ni), w(nc), v_new(ni);

    for (int pc = 0; pc < n_pcs_actual; ++pc) {
        // Random unit-vector init in SNP space
        float vn = 0.0f;
        for (int i = 0; i < ni; ++i) { v[i] = nd(rng); vn += v[i] * v[i]; }
        vn = std::sqrt(vn) + 1e-12f;
        for (int i = 0; i < ni; ++i) v[i] /= vn;

        for (int it = 0; it < n_iter; ++it) {
            // w = X_centered * v  (n_cells)
            for (int c = 0; c < nc; ++c) {
                const auto& cell = data.cells[c];
                float val = 0.0f;
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    int ii = snp_to_inf[cell.snp_idx[si]];
                    if (ii < 0 || cell.dp[si] == 0) continue;
                    float af = static_cast<float>(cell.ad[si]) / static_cast<float>(cell.dp[si]);
                    val += (af - inf.mean_af[ii]) * v[ii];
                }
                w[c] = val;
            }
            // v_new = X_centered^T * w  (n_snps)
            std::fill(v_new.begin(), v_new.end(), 0.0f);
            for (int c = 0; c < nc; ++c) {
                if (w[c] == 0.0f) continue;
                const auto& cell = data.cells[c];
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    int ii = snp_to_inf[cell.snp_idx[si]];
                    if (ii < 0 || cell.dp[si] == 0) continue;
                    float af = static_cast<float>(cell.ad[si]) / static_cast<float>(cell.dp[si]);
                    v_new[ii] += (af - inf.mean_af[ii]) * w[c];
                }
            }
            // Gram-Schmidt deflation against previous eigenvectors
            for (int prev = 0; prev < pc; ++prev) {
                float dot = 0.0f;
                for (int i = 0; i < ni; ++i) dot += v_new[i] * evecs[prev][i];
                for (int i = 0; i < ni; ++i) v_new[i] -= dot * evecs[prev][i];
            }
            // Normalize
            float vnorm = 0.0f;
            for (int i = 0; i < ni; ++i) vnorm += v_new[i] * v_new[i];
            vnorm = std::sqrt(vnorm) + 1e-12f;
            for (int i = 0; i < ni; ++i) v[i] = v_new[i] / vnorm;
        }
        evecs[pc] = v;

        // Cell PC scores: w = X_centered * v (final result for this PC)
        for (int c = 0; c < nc; ++c) {
            const auto& cell = data.cells[c];
            float val = 0.0f;
            for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                int ii = snp_to_inf[cell.snp_idx[si]];
                if (ii < 0 || cell.dp[si] == 0) continue;
                float af = static_cast<float>(cell.ad[si]) / static_cast<float>(cell.dp[si]);
                val += (af - inf.mean_af[ii]) * v[ii];
            }
            result[static_cast<size_t>(c) * n_pcs + pc] = val;
        }
    }
    return result;
}

// K-means++ clustering on n_pcs-dimensional cell PCA scores.
// Returns cluster assignments [nc].
static std::vector<int> kmeans_plusplus(
    const std::vector<float>& scores,   // [nc × n_pcs], cell-major
    int nc, int n_pcs, int K,
    int      kmeans_iter = 20,
    uint64_t seed        = 0)
{
    if (K <= 1 || nc < K) {
        std::vector<int> a(nc);
        for (int c = 0; c < nc; ++c) a[c] = c % K;
        return a;
    }

    std::mt19937 rng(seed);
    std::vector<std::vector<float>> centroids(K, std::vector<float>(n_pcs, 0.0f));
    std::vector<float> min_d2(nc, std::numeric_limits<float>::infinity());

    auto dist2 = [&](int c, const std::vector<float>& cen) {
        float d = 0.0f;
        for (int p = 0; p < n_pcs; ++p) {
            float diff = scores[static_cast<size_t>(c) * n_pcs + p] - cen[p];
            d += diff * diff;
        }
        return d;
    };

    // Pick first center uniformly at random
    std::uniform_int_distribution<int> rpick(0, nc - 1);
    {
        int first = rpick(rng);
        for (int p = 0; p < n_pcs; ++p)
            centroids[0][p] = scores[static_cast<size_t>(first) * n_pcs + p];
    }
    // K-means++ distance-proportional center selection
    for (int ki = 1; ki < K; ++ki) {
        for (int c = 0; c < nc; ++c) {
            float d2 = dist2(c, centroids[ki - 1]);
            if (d2 < min_d2[c]) min_d2[c] = d2;
        }
        float total = 0.0f;
        for (float d : min_d2) total += d;
        if (total < 1e-30f) {
            int next = rpick(rng);
            for (int p = 0; p < n_pcs; ++p)
                centroids[ki][p] = scores[static_cast<size_t>(next) * n_pcs + p];
        } else {
            std::discrete_distribution<int> dd(min_d2.begin(), min_d2.end());
            int next = dd(rng);
            for (int p = 0; p < n_pcs; ++p)
                centroids[ki][p] = scores[static_cast<size_t>(next) * n_pcs + p];
        }
    }

    // K-means Lloyd iterations
    std::vector<int> assign(nc, 0);
    std::vector<int> counts(K, 0);
    for (int iter = 0; iter < kmeans_iter; ++iter) {
        bool changed = false;
        for (int c = 0; c < nc; ++c) {
            int bk = 0; float bd = dist2(c, centroids[0]);
            for (int k = 1; k < K; ++k) {
                float d = dist2(c, centroids[k]);
                if (d < bd) { bd = d; bk = k; }
            }
            if (assign[c] != bk) { assign[c] = bk; changed = true; }
        }
        if (!changed) break;
        for (auto& cv : centroids) std::fill(cv.begin(), cv.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        for (int c = 0; c < nc; ++c) {
            int k = assign[c]; ++counts[k];
            for (int p = 0; p < n_pcs; ++p)
                centroids[k][p] += scores[static_cast<size_t>(c) * n_pcs + p];
        }
        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) continue;
            float inv = 1.0f / counts[k];
            for (int p = 0; p < n_pcs; ++p) centroids[k][p] *= inv;
        }
    }
    return assign;
}

// Build a PCA-seeded VBResult: beta_mu seeded from cluster-mean AFs,
// id_prob from soft (0.9 / 0.1/(K-1)) cluster membership.
// Falls back to uniform random id_prob when too few informative SNPs exist.
static VBResult init_vb_pca_seeded(
    const CoveredSNPData& data, int K,
    const DonorDemuxConfig& cfg, uint64_t seed)
{
    const int nc = data.n_cells;
    const int ns = data.n_covered_snps;

    VBResult result;
    result.elbo   = -std::numeric_limits<double>::infinity();
    result.n_iter = 0;
    result.beta_mu.assign(ns,  std::vector<float>(K, 0.5f));
    result.beta_sum.assign(ns, std::vector<float>(K, 2.0f));
    result.id_prob.assign(nc,  std::vector<float>(K, 1.0f / K));

    if (K <= 1) return result;

    const auto inf = find_informative_snps(data);
    const int  ni  = static_cast<int>(inf.indices.size());

    if (ni < K * 10) {
        // Not enough informative SNPs — fall back to random init
        std::cerr << "[demux] PCA init: only " << ni << " informative SNPs (< "
                  << K * 10 << " threshold), using random init\n";
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unif(0.0f, 1.0f);
        for (int c = 0; c < nc; ++c) {
            float sm = 0.0f;
            for (int k = 0; k < K; ++k) { result.id_prob[c][k] = unif(rng); sm += result.id_prob[c][k]; }
            const float inv = 1.0f / sm;
            for (int k = 0; k < K; ++k) result.id_prob[c][k] *= inv;
        }
        return result;
    }

    // PCA: top K-1 principal components of cell AF profiles
    const int n_pcs = K - 1;
    const auto pc_scores = power_iteration_pca(data, inf, n_pcs, /*n_iter=*/30, seed);
    const auto assign    = kmeans_plusplus(pc_scores, nc, n_pcs, K, /*kmeans_iter=*/20, seed + 1ULL);

    std::cerr << "[demux] PCA init: " << ni << " informative SNPs, "
              << n_pcs << " PC(s), k-means:";
    {
        std::vector<int> cnts(K, 0);
        for (int a : assign) ++cnts[a];
        for (int k = 0; k < K; ++k) std::cerr << " k" << k << "=" << cnts[k];
    }
    std::cerr << "\n";

    // Cluster-mean AFs per covered SNP → seed beta_mu (add-1 pseudo-count)
    const size_t ns_K = static_cast<size_t>(ns) * K;
    std::vector<float> ad_sum(ns_K, 1.0f);   // pseudo-count: 1 alt read per cluster
    std::vector<float> dp_sum(ns_K, 2.0f);   // pseudo-count: 1 alt + 1 ref per cluster
    for (int c = 0; c < nc; ++c) {
        int k = assign[c];
        const auto& cell = data.cells[c];
        for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
            size_t jK = static_cast<size_t>(cell.snp_idx[si]) * K + k;
            ad_sum[jK] += static_cast<float>(cell.ad[si]);
            dp_sum[jK] += static_cast<float>(cell.dp[si]);
        }
    }
    for (int j = 0; j < ns; ++j) {
        for (int k = 0; k < K; ++k) {
            size_t jK = static_cast<size_t>(j) * K + k;
            float mu = ad_sum[jK] / dp_sum[jK];
            result.beta_mu[j][k]  = std::max(0.05f, std::min(0.95f, mu));
            result.beta_sum[j][k] = dp_sum[jK];
        }
    }

    // Soft cell assignments: 0.9 for own cluster, 0.1/(K-1) for others
    const float p_self  = 0.9f;
    const float p_other = (K > 1) ? (1.0f - p_self) / static_cast<float>(K - 1) : 0.0f;
    for (int c = 0; c < nc; ++c) {
        for (int k = 0; k < K; ++k)
            result.id_prob[c][k] = (k == assign[c]) ? p_self : p_other;
    }
    return result;
}

// Run VB from scratch (random initialisation), single-threaded.
// Called from run_batch() which runs restarts in parallel.
static VBResult run_vb(
    const CoveredSNPData& data, int K,
    const DonorDemuxConfig& cfg, uint64_t seed, int max_iter)
{
    const int nc = data.n_cells;
    const int ns = data.n_covered_snps;

    VBResult result;
    result.elbo   = -std::numeric_limits<double>::infinity();
    result.n_iter = 0;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);

    result.beta_mu.assign(ns,  std::vector<float>(K, 0.5f));
    result.beta_sum.assign(ns, std::vector<float>(K, 30.0f));
    result.id_prob.assign(nc,  std::vector<float>(K));

    for (int c = 0; c < nc; ++c) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) { result.id_prob[c][k] = unif(rng); sum += result.id_prob[c][k]; }
        const float inv = 1.0f / sum;
        for (int k = 0; k < K; ++k) result.id_prob[c][k] *= inv;
    }

    run_vb_inplace(data, K, cfg, result, max_iter, /*nthreads=*/1);
    return result;
}

// ============================================================================
// Extended demux result — carries assignments + VB posteriors for genotyping
// ============================================================================
struct DemuxResult {
    std::vector<DonorAssignment> assignments;
    std::vector<std::vector<float>> beta_mu;       // [n_covered_snps][K] posterior AF
    std::vector<std::vector<float>> beta_sum;      // [n_covered_snps][K] posterior precision
    std::vector<int32_t> covered_to_original;      // [n_covered_snps] → original SNP index
    int n_donors_k = 0;                            // final K chosen by model
};

// ============================================================================
// Main entry point — run donor demultiplexing
// ============================================================================
inline DemuxResult run_demux(
    uint32_t n_snps, uint32_t n_cells,
    const int32_t* ad_indptr, const int32_t* ad_indices, const uint8_t* ad_data,
    uint64_t /*ad_nnz*/,
    const int32_t* dp_indptr, const int32_t* dp_indices, const uint8_t* dp_data,
    uint64_t /*dp_nnz*/,
    const DonorDemuxConfig& cfg,
    const std::vector<uint32_t>& cell_indices = {}) // empty = use all barcodes (no filter)
{
    std::cerr << "[demux] Starting donor demultiplexing: "
              << n_snps << " SNPs, " << n_cells << " cells\n";

    auto cov = extract_covered_snps(n_snps, n_cells,
        ad_indptr, ad_indices, ad_data,
        dp_indptr, dp_indices, dp_data);

    std::cerr << "[demux] Covered SNPs: " << cov.n_covered_snps
              << " (from " << n_snps << " total)\n";

    if (cov.n_covered_snps == 0) {
        std::cerr << "[demux] WARNING: No covered SNPs found. All cells unassigned.\n";
        DemuxResult empty;
        empty.assignments.resize(n_cells);
        for (uint32_t c = 0; c < n_cells; ++c)
            empty.assignments[c] = {-1, 0.0f, 0.0f, "unassigned", "cell"};
        empty.n_donors_k = 0;
        return empty;
    }

    // ── Optionally filter to EmptyDrops-called cells only ────────────────────
    // When cell_indices is provided, VB runs only on those barcodes, avoiding
    // dilution by N_ambient barcodes with near-zero SNP coverage.
    // After convergence, ambient barcodes are back-assigned via a single E-step.
    const bool use_cell_filter = !cell_indices.empty() &&
        cell_indices.size() < static_cast<size_t>(n_cells);

    CoveredSNPData cov_filt;
    if (use_cell_filter) {
        // Share the same SNP index space as cov_all; only subset the cell list.
        cov_filt.n_covered_snps    = cov.n_covered_snps;
        cov_filt.n_cells           = static_cast<int>(cell_indices.size());
        cov_filt.covered_to_original = cov.covered_to_original;
        cov_filt.cells.reserve(cell_indices.size());
        for (uint32_t idx : cell_indices)
            cov_filt.cells.push_back(cov.cells[idx]);
        std::cerr << "[demux] Cell filter: running VB on "
                  << cov_filt.n_cells << " called cells (of " << n_cells
                  << " total barcodes); "
                  << (n_cells - static_cast<uint32_t>(cov_filt.n_cells))
                  << " ambient barcodes will be back-assigned post-convergence\n";
    }
    // VB model operates on this (filtered or full)
    const CoveredSNPData& cov_use = use_cell_filter ? cov_filt : cov;

    // ── Determine K range ────────────────────────────────────────────────────
    int k_min = 1, k_max = cfg.max_donors;
    if (cfg.n_donors > 0) k_min = k_max = cfg.n_donors;

    // ── Phase 1: EM+BIC — narrow K to 2-3 candidates ─────────────────────────
    std::vector<int> k_candidates;
    if (cfg.fast_k_select && k_min < k_max) {
        k_candidates = select_k_bic(cov_use, k_min, k_max, cfg);
        std::cerr << "[demux] EM+BIC K candidates:";
        for (int K : k_candidates) std::cerr << " " << K;
        std::cerr << "\n";
    } else {
        for (int K = k_min; K <= k_max; ++K) k_candidates.push_back(K);
    }

    // ── Phase 2: Full VB only for candidate K values ─────────────────────────
    double   best_elbo   = -std::numeric_limits<double>::infinity();
    int      best_k      = k_candidates[0];
    VBResult best_result;

    double prev2_k_elbo = -std::numeric_limits<double>::infinity();
    double prev1_k_elbo = -std::numeric_limits<double>::infinity();

    for (int K : k_candidates) {
        double   k_best_elbo = -std::numeric_limits<double>::infinity();
        VBResult k_best_result;

        int n_inits = cfg.n_inits;
        if (K <= 3 && n_inits > 3) n_inits = 3;

        // ── Parallel random restarts (serial VB per restart) ─────────────────
        std::vector<VBResult> init_results(n_inits);
        const int nw = std::min(cfg.threads, n_inits);

        auto run_batch = [&](int start, int end) {
            for (int i = start; i < end; ++i) {
                const uint64_t init_seed = cfg.seed
                    + static_cast<uint64_t>(i) * 1000ULL
                    + static_cast<uint64_t>(K) * 100000ULL;
                if (i == 0 && K > 1) {
                    // PCA-seeded first restart — prevents symmetric-init cluster collapse
                    init_results[i] = init_vb_pca_seeded(cov_use, K, cfg, init_seed);
                    run_vb_inplace(cov_use, K, cfg, init_results[i], cfg.max_iter_pre, /*nthreads=*/1);
                } else {
                    init_results[i] = run_vb(cov_use, K, cfg, init_seed, cfg.max_iter_pre);
                }
            }
        };

        if (nw <= 1) {
            run_batch(0, n_inits);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nw);
            const int per = n_inits / nw, rem = n_inits % nw;
            int start = 0;
            for (int w = 0; w < nw; ++w) {
                const int cnt = per + (w < rem ? 1 : 0);
                threads.emplace_back(run_batch, start, start + cnt);
                start += cnt;
            }
            for (auto& t : threads) t.join();
        }

        int best_init_idx = 0;
        for (int i = 0; i < n_inits; ++i)
            if (init_results[i].elbo > k_best_elbo) { k_best_elbo = init_results[i].elbo; best_init_idx = i; }
        k_best_result = std::move(init_results[best_init_idx]);

        // ── Continue from best init with all threads ─────────────────────────
        run_vb_inplace(cov_use, K, cfg, k_best_result, cfg.max_iter, cfg.threads);

        std::cerr << "[demux] K=" << K << ": ELBO=" << k_best_result.elbo
                  << " (" << k_best_result.n_iter << " iters)\n";

        if (k_best_result.elbo > best_elbo) {
            best_elbo   = k_best_result.elbo;
            best_k      = K;
            best_result = std::move(k_best_result);
        }

        // ── K-level early stopping ────────────────────────────────────────────
        if (prev2_k_elbo > -1e300 &&
            k_best_result.elbo < prev1_k_elbo &&
            prev1_k_elbo       < prev2_k_elbo) {
            std::cerr << "[demux] K-level early stop at K=" << K << "\n";
            break;
        }
        prev2_k_elbo = prev1_k_elbo;
        prev1_k_elbo = best_elbo;
    }

    std::cerr << "[demux] Best K=" << best_k << " (ELBO=" << best_elbo << ")\n";

    // ── Assign cells and back-assign ambient barcodes ─────────────────────────
    // Default sentinel: unassigned ambient (overwritten below for called cells)
    std::vector<DonorAssignment> assignments(n_cells,
        {best_k, 0.0f, 0.0f, "unassigned", "ambient"});

    int n_assigned_cells = 0,   n_unassigned_cells = 0;
    int n_assigned_ambient = 0, n_unassigned_ambient = 0;

    // Helper: pick best donor from a posterior probability vector
    auto assign_from_probs = [&](const std::vector<float>& probs,
                                 const std::string& status) -> DonorAssignment {
        int best_donor = 0;
        float best_prob = probs[0];
        for (int k = 1; k < best_k; ++k)
            if (probs[k] > best_prob) { best_prob = probs[k]; best_donor = k; }
        if (best_prob >= cfg.min_prob)
            return {best_donor, best_prob, 0.0f,
                    "donor" + std::to_string(best_donor), status};
        return {best_k, best_prob, 0.0f, "unassigned", status};
    };

    if (!use_cell_filter) {
        // ── No filter: all barcodes are treated as cells (original path) ──────
        for (int c = 0; c < static_cast<int>(n_cells); ++c) {
            assignments[c] = assign_from_probs(best_result.id_prob[c], "cell");
            if (assignments[c].label == "unassigned") ++n_unassigned_cells;
            else ++n_assigned_cells;
        }
    } else {
        // ── 1. Assign called cells from VB posteriors ─────────────────────────
        for (size_t i = 0; i < cell_indices.size(); ++i) {
            const uint32_t orig_c = cell_indices[i];
            assignments[orig_c] = assign_from_probs(best_result.id_prob[i], "cell");
            if (assignments[orig_c].label == "unassigned") ++n_unassigned_cells;
            else ++n_assigned_cells;
        }

        // ── 2. Back-assign ambient barcodes via single E-step ─────────────────
        // Precompute digamma cache from converged Beta posterior (ns × K lookups,
        // avoids N_ambient × snps/cell × K digamma calls in the inner loop).
        const int ns = cov_use.n_covered_snps;
        std::vector<float> d1a(ns * best_k), d2a(ns * best_k), d12a(ns * best_k);
        for (int j = 0; j < ns; ++j) {
            for (int k = 0; k < best_k; ++k) {
                const float mu  = best_result.beta_mu[j][k];
                const float sum = best_result.beta_sum[j][k];
                const float s1  = mu * sum;
                const float s2  = (1.0f - mu) * sum;
                d1a [j * best_k + k] = digamma(s1);
                d2a [j * best_k + k] = digamma(s2);
                d12a[j * best_k + k] = digamma(sum);
            }
        }

        // Build called-cell set for O(1) skip
        std::unordered_set<uint32_t> cell_set(cell_indices.begin(), cell_indices.end());

        std::vector<float> ll(best_k);
        for (uint32_t c = 0; c < n_cells; ++c) {
            if (cell_set.count(c)) continue;  // already assigned above

            // `cov` (full-barcode CoveredSNPData) uses the same SNP index space
            // as `cov_use` (filtered), so snp_idx values are directly compatible.
            const auto& cell = cov.cells[c];
            if (cell.snp_idx.empty()) {
                assignments[c] = {best_k, 0.0f, 0.0f, "unassigned", "ambient"};
                ++n_unassigned_ambient;
                continue;
            }

            std::fill(ll.begin(), ll.end(), 0.0f);
            for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                const int   j  = cell.snp_idx[si];
                const float ad = static_cast<float>(cell.ad[si]);
                const float dp = static_cast<float>(cell.dp[si]);
                const float bd = dp - ad;
                for (int k = 0; k < best_k; ++k) {
                    ll[k] += ad * d1a [j * best_k + k]
                           + bd * d2a [j * best_k + k]
                           - dp * d12a[j * best_k + k];
                }
            }

            float mx = ll[0];
            for (int k = 1; k < best_k; ++k) if (ll[k] > mx) mx = ll[k];
            float sm = 0.0f;
            std::vector<float> probs(best_k);
            for (int k = 0; k < best_k; ++k) { probs[k] = std::exp(ll[k] - mx); sm += probs[k]; }
            const float inv = 1.0f / sm;
            for (int k = 0; k < best_k; ++k) probs[k] *= inv;

            assignments[c] = assign_from_probs(probs, "ambient");
            if (assignments[c].label == "unassigned") ++n_unassigned_ambient;
            else ++n_assigned_ambient;
        }
    }

    std::cerr << "[demux] Called cells: " << n_assigned_cells << " assigned, "
              << n_unassigned_cells << " unassigned\n";
    if (use_cell_filter) {
        std::cerr << "[demux] Ambient barcodes: " << n_assigned_ambient << " assigned, "
                  << n_unassigned_ambient << " unassigned\n";
    }

    DemuxResult demux_result;
    demux_result.assignments = std::move(assignments);
    // Move beta arrays after back-assignment is complete (d1a/d2a/d12a hold copies)
    demux_result.beta_mu             = std::move(best_result.beta_mu);
    demux_result.beta_sum            = std::move(best_result.beta_sum);
    demux_result.covered_to_original = use_cell_filter
        ? cov_filt.covered_to_original           // copy (already a copy of cov's)
        : std::move(cov.covered_to_original);    // move for no-filter case
    demux_result.n_donors_k = best_k;
    return demux_result;
}

// ============================================================================
// Aggregate AD/DP per SNP per donor from the full CSC matrices.
// Only accumulates reads from confidently assigned barcodes (prob >= min_prob).
// Output is indexed by COVERED SNP (reduced index, same as beta_mu).
// ============================================================================
struct DonorDepths {
    std::vector<std::vector<uint32_t>> ad;  // [n_covered_snps][K]
    std::vector<std::vector<uint32_t>> dp;  // [n_covered_snps][K]
    int n_donors = 0;
    int n_covered = 0;
};

inline DonorDepths aggregate_donor_depths(
    const DemuxResult& demux,
    uint32_t n_snps, uint32_t n_barcodes,
    const int32_t* ad_indptr, const int32_t* ad_indices, const uint8_t* ad_data,
    const int32_t* dp_indptr, const int32_t* dp_indices, const uint8_t* dp_data,
    float min_prob = 0.9f)
{
    DonorDepths depths;
    depths.n_donors = demux.n_donors_k;
    depths.n_covered = static_cast<int>(demux.covered_to_original.size());

    depths.ad.assign(depths.n_covered, std::vector<uint32_t>(depths.n_donors, 0));
    depths.dp.assign(depths.n_covered, std::vector<uint32_t>(depths.n_donors, 0));

    // Build original → covered index mapping
    std::vector<int32_t> orig_to_cov(n_snps, -1);
    for (int j = 0; j < depths.n_covered; ++j)
        orig_to_cov[demux.covered_to_original[j]] = j;

    for (uint32_t c = 0; c < n_barcodes; ++c) {
        const auto& a = demux.assignments[c];
        if (a.donor_id < 0 || a.donor_id >= depths.n_donors) continue; // unassigned
        if (a.prob_max < min_prob) continue;
        int k = a.donor_id;

        for (int32_t i = dp_indptr[c]; i < dp_indptr[c + 1]; ++i) {
            int32_t j = orig_to_cov[dp_indices[i]];
            if (j >= 0) depths.dp[j][k] += dp_data[i];
        }
        for (int32_t i = ad_indptr[c]; i < ad_indptr[c + 1]; ++i) {
            int32_t j = orig_to_cov[ad_indices[i]];
            if (j >= 0) depths.ad[j][k] += ad_data[i];
        }
    }
    return depths;
}

// ============================================================================
// Parse a singlet SNP name "chr:pos:R>A" into component fields.
// pos is 1-based in the name (as stored) and returned 1-based.
// ============================================================================
static void parse_snp_name(const std::string& name,
                            std::string& chrom, int32_t& pos,
                            char& ref, char& alt)
{
    auto c1 = name.find(':');
    if (c1 == std::string::npos) { chrom = name; pos = 0; ref = alt = 'N'; return; }
    chrom = name.substr(0, c1);
    auto c2 = name.find(':', c1 + 1);
    if (c2 == std::string::npos) { pos = 0; ref = alt = 'N'; return; }
    pos = std::atoi(name.c_str() + c1 + 1);
    // After c2: "R>A"
    ref = (c2 + 1 < name.size()) ? name[c2 + 1] : 'N';
    alt = (c2 + 3 < name.size()) ? name[c2 + 3] : 'N';
}

// ============================================================================
// Write per-donor VCF files.
// Produces: out_prefix/donor{k}.vcf for k = 0..K-1
// FORMAT: GT:AF:AD:DP per site where the donor has any read depth.
// Only SNPs with coverage in at least one donor are emitted.
// ============================================================================
inline void write_donor_vcfs(
    const std::string& out_prefix,
    const DemuxResult& demux,
    const DonorDepths& depths,
    const std::vector<std::string>& snp_names,
    const std::string& singlet_version = "0.2.0")
{
    const int K = demux.n_donors_k;
    const int nc = depths.n_covered;

    for (int k = 0; k < K; ++k) {
        std::string path = out_prefix + "/donor" + std::to_string(k) + ".vcf";
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[demux] WARNING: cannot write " << path << "\n";
            continue;
        }

        f << "##fileformat=VCFv4.2\n";
        f << "##source=singlet v" << singlet_version << "\n";
        f << "##donorID=donor" << k << "\n";
        f << "##FILTER=<ID=PASS,Description=\"All filters passed\">\n";
        f << "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of donors with any coverage at site\">\n";
        f << "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype inferred from VB posterior allele frequency\">\n";
        f << "##FORMAT=<ID=AF,Number=A,Type=Float,Description=\"VB posterior allele frequency (beta_mu)\">\n";
        f << "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Aggregated ref,alt allele depths from assigned cells\">\n";
        f << "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Aggregated total depth from assigned cells\">\n";
        f << "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tdonor" << k << "\n";

        for (int j = 0; j < nc; ++j) {
            int32_t orig = demux.covered_to_original[j];
            if (orig >= static_cast<int32_t>(snp_names.size())) continue;

            uint32_t dp_val = depths.dp[j][k];
            uint32_t ad_val = depths.ad[j][k];
            float af = (j < static_cast<int>(demux.beta_mu.size()) &&
                        k < static_cast<int>(demux.beta_mu[j].size()))
                       ? demux.beta_mu[j][k] : 0.5f;

            // Skip sites with no depth for this donor
            if (dp_val == 0) continue;

            std::string chrom; int32_t pos; char ref, alt;
            parse_snp_name(snp_names[orig], chrom, pos, ref, alt);

            // Genotype call from VB posterior AF
            const char* gt;
            if      (af < 0.1f) gt = "0/0";
            else if (af > 0.9f) gt = "1/1";
            else                gt = "0/1";

            uint32_t ref_val = (dp_val >= ad_val) ? dp_val - ad_val : 0;

            // Count donors with any depth at this site for NS
            int ns = 0;
            for (int kk = 0; kk < K; ++kk)
                if (depths.dp[j][kk] > 0) ++ns;

            char buf[256];
            int n = std::snprintf(buf, sizeof(buf),
                "%s\t%d\t%s\t%c\t%c\t.\tPASS\tNS=%d\tGT:AF:AD:DP\t"
                "%s:%.3f:%u,%u:%u\n",
                chrom.c_str(), pos, snp_names[orig].c_str(), ref, alt,
                ns, gt, af, ref_val, ad_val, dp_val);
            if (n > 0) f.write(buf, n);
        }
        std::cerr << "[demux] Wrote " << path << "\n";
    }
}

// ============================================================================
// Write per-donor coverage maps.
// Produces: out_prefix/donor{k}_coverage.tsv for k = 0..K-1
// Columns: snp_id  chrom  pos  ref  alt  dp  ad  af_vb  gt
// Only rows where dp > 0 for this donor are written.
// ============================================================================
inline void write_donor_coverages(
    const std::string& out_prefix,
    const DemuxResult& demux,
    const DonorDepths& depths,
    const std::vector<std::string>& snp_names)
{
    const int K = demux.n_donors_k;
    const int nc = depths.n_covered;

    for (int k = 0; k < K; ++k) {
        std::string path = out_prefix + "/donor" + std::to_string(k) + "_coverage.tsv";
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[demux] WARNING: cannot write " << path << "\n";
            continue;
        }

        f << "snp_id\tchrom\tpos\tref\talt\tdp\tad\taf_vb\tgt\n";

        for (int j = 0; j < nc; ++j) {
            int32_t orig = demux.covered_to_original[j];
            if (orig >= static_cast<int32_t>(snp_names.size())) continue;

            uint32_t dp_val = depths.dp[j][k];
            if (dp_val == 0) continue;

            uint32_t ad_val = depths.ad[j][k];
            float af = (j < static_cast<int>(demux.beta_mu.size()) &&
                        k < static_cast<int>(demux.beta_mu[j].size()))
                       ? demux.beta_mu[j][k] : 0.5f;

            std::string chrom; int32_t pos; char ref, alt;
            parse_snp_name(snp_names[orig], chrom, pos, ref, alt);

            const char* gt;
            if      (af < 0.1f) gt = "0/0";
            else if (af > 0.9f) gt = "1/1";
            else                gt = "0/1";

            char buf[256];
            int n = std::snprintf(buf, sizeof(buf),
                "%s\t%s\t%d\t%c\t%c\t%u\t%u\t%.4f\t%s\n",
                snp_names[orig].c_str(), chrom.c_str(), pos,
                ref, alt, dp_val, ad_val, af, gt);
            if (n > 0) f.write(buf, n);
        }
        std::cerr << "[demux] Wrote " << path << "\n";
    }
}

// ============================================================================
// Write donor assignments TSV
// ============================================================================
inline bool write_donor_assignments(
    const std::string& path,
    const std::vector<DonorAssignment>& assignments,
    const std::vector<std::string>& barcodes)
{
    std::ofstream f(path);
    if (!f) return false;
    f << "cell\tdonor_id\tprob_max\tprob_doublet\tbest_singlet\tbest_doublet\tcell_status\n";
    for (size_t i = 0; i < assignments.size(); ++i) {
        const auto& a = assignments[i];
        f << barcodes[i] << "\t" << a.label << "\t"
          << a.prob_max << "\t" << a.prob_doublet << "\t"
          << a.label << "\t"
          << (a.prob_doublet > 0.5f ? "doublet" : a.label) << "\t"
          << a.cell_status << "\n";
    }
    return true;
}


// ============================================================================
// B-G5-3: K=1 single-donor workaround
// When --n-donors 1 is requested, skip VB entirely (it is trivially correct that
// all cells belong to donor0) and synthesize a DemuxResult directly.
// covered_to_original is still populated so per-donor VCF/coverage writes work.
// ============================================================================
inline DemuxResult make_single_donor_result(
    uint32_t n_snps, uint32_t n_cells,
    const int32_t* ad_indptr, const int32_t* ad_indices, const uint8_t* ad_data,
    const int32_t* dp_indptr, const int32_t* dp_indices, const uint8_t* dp_data)
{
    std::cerr << "[demux] K=1 workaround: skipping VB inference, assigning all "
              << n_cells << " cells to donor0\n";
    DemuxResult result;
    result.n_donors_k = 1;
    result.assignments.resize(n_cells);
    for (uint32_t c = 0; c < n_cells; ++c)
        result.assignments[c] = {0, 1.0f, 0.0f, "donor0", "cell"};
    // Populate covered_to_original so aggregate_donor_depths / write_donor_vcfs work.
    auto cov = extract_covered_snps(n_snps, n_cells,
        ad_indptr, ad_indices, ad_data,
        dp_indptr, dp_indices, dp_data);
    result.covered_to_original = std::move(cov.covered_to_original);
    std::cerr << "[demux] K=1 single-donor: " << result.covered_to_original.size()
              << " covered SNPs indexed for donor outputs\n";
    return result;
}

}  // namespace singlet
