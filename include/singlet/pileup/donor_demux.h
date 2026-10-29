// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: donor_demux.h
// Variational Bayes binomial mixture model for donor demultiplexing.
// C++ replacement for vireoSNP — works directly on in-memory AD/DP CSC matrices.
//
// Algorithm: VB coordinate ascent on a Beta-Binomial mixture, plus
//   - auto-K by cross-fitted held-out likelihood (select_k_heldout): fit K donors on
//     one half of the cells, score the other half with the singlet+doublet mixture;
//   - Vireo-style genotype doublets (doublet_stage / score_cells): per-cell likelihood
//     under each donor and each donor pair (allele probability = mean of the two),
//     posterior prob_doublet, genotypes refit without likely doublets.
//   (The older EM+BIC search, select_k_bic, is kept for reference but no longer used.)
//
// Performance optimisations vs. original:
//   1. EM+BIC Phase 1 reduces K sweep from 1..8 full VB runs to 2-3 (3-4x speedup).
//   2. Flat s1/s2/log_lik buffers eliminate O(ns) heap allocs per iteration.
//   3. Parallel M-step and E-step across cfg.threads threads (final refinement pass).
//   4. K-level early stopping when ELBO degrades for 2 consecutive K values.

#include <algorithm>
#include <atomic>
#include <chrono>
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
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {

// ============================================================================
// Configuration
// ============================================================================
struct DonorDemuxConfig {
    int n_donors = -1;          // -1 = auto-detect (try 1..max_donors)
    int max_donors = 16;
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
    bool fast_k_select = true;   // auto-K only: true = cross-fitted held-out likelihood search
                                 // (select_k_heldout); false = full ELBO sweep over 1..max_donors
    // ── Auto-K (held-out likelihood, see select_k_heldout) ─────────────────
    int   k_select_max_cells = 20000;  // cap on cells used for the K search
    int   k_select_min_snp_cells = 8;  // SNP must be seen in this many cells to enter the K search
    int   k_select_inits = 3;          // PCA/k-means restarts per K and fold
    int   k_select_max_iter = 60;      // VB iterations per fit in the K search
    double k_select_sep_z = 5.0;       // z for "components i and j are different donors"
    double k_select_sep_min = 1.0;     // ... and the mean log-lik ratio per cell must exceed this (nats)
    double k_select_z = 4.0;           // more donors are accepted only if held-out log-lik improves, paired z above this
    double k_select_min_gain = 200.0;  // ... and by at least this many nats in total (about 10 cells' worth of a donor)
    int   k_select_patience = 3;       // stop after this many consecutive K without more distinct donors
    int   min_donor_cells = 15;        // a donor owns at least max(this, min_donor_frac * cells) cells
    float min_donor_frac = 0.004f;
    int   n_ward_inits = 4;            // extra Ward-seeded restarts in the final fit (K >= 2)
    double split_merge_min_gain_rel = 1e-4; // candidate must beat the current ELBO by this fraction of |ELBO|
    int   split_merge_rounds = 0;      // split-merge repair rounds after the final fit (0 = off)
    int   split_merge_iters = 30;      // VB iterations per split-merge candidate
    int   split_merge_merge_cands = 2; // closest component pairs tried as merge + split
    int   split_merge_resplit_pairs = 10; // closest pairs whose pooled cells are re-split in two
    // ── Genotype doublet model (Vireo-style, see score_cells) ──────────────
    bool  detect_doublets = true;      // fill prob_doublet / "doublet" label (K >= 2)
    float doublet_threshold = 0.9f;    // prob_doublet above this -> label "doublet"
    float doublet_rate_per_1000 = 0.008f; // expected multiplet rate per 1000 cells (10x)
    float doublet_prior = -1.0f;       // >= 0 overrides the total multiplet prior
    int   doublet_rounds = 2;          // scoring rounds; round 2 refits genotypes without doublets
    bool  staged_fit = true;           // final fit: many cheap restarts on a reduced SNP set, best few refined
    int   explore_ward = 16;           // ... Ward-seeded restarts
    int   explore_legacy = 6;          // ... k-means++-seeded restarts
    int   explore_random = 2;          // ... random restarts
    int   explore_iters = 60;          // ... VB iterations per exploratory restart
    int   explore_top = 2;             // ... restarts continued on the full SNP set
    int   explore_full_iters = 40;     // ... VB iterations of that continuation
    int   explore_min_snp_cells = 8;   // ... SNP must be seen in this many cells to enter the exploration
    float core_min_rel = 0.35f;        // donors are fit on called cells with >= this x median SNP observations
                                       // (0 = fit on all called cells); the rest are assigned afterwards
    float doublet_temper = 0.2f;      // exponent on the read likelihood in the final doublet/donor posteriors:
                                       // UMIs of one cell are not independent draws (monoallelic bursts, PCR), so
                                       // the raw binomial is overconfident; 0.2 was fit on the Kang pool (see bench)
    int   doublet_min_snp_cells = 3;   // SNP must be seen in this many cells to be scored
    bool  doublet_crossfit = true;     // score each half of the cells with genotypes fit on the other half
    float geno_error = 0.03f;          // initial per-read error of a genotype call (re-estimated)
    float geno_conc_hom = 100.0f;      // beta concentration of the homozygous allele-probability states
    float geno_conc_het = 50.0f;       // ... and of the heterozygous state (allows ~0.07 sd of allelic bias)
};

// ============================================================================
// Per-cell output
// ============================================================================
struct DonorAssignment {
    int donor_id;           // 0..K-1; K for "doublet" and "unassigned" (see label); -1 = no SNPs
    float prob_max;         // posterior P(z_c = donor_id)
    float prob_doublet;     // posterior P(doublet)
    std::string label;      // "donor0", "donor1", ..., "doublet", "unassigned"
    std::string cell_status = "cell"; // "cell" (EmptyDrops-called) or "ambient" (back-assigned)
    int best_singlet = -1;          // most likely single donor (also when labelled doublet/unassigned)
    int doublet_a = -1;             // most likely donor pair (a < b); -1 if not scored
    int doublet_b = -1;
    float doublet_llr = 0.0f;       // log-lik(best pair) - log-lik(best single donor), reads only
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
// Threading helper: run fn(thread_index, begin, end) over [0, n) in contiguous chunks.
// ============================================================================
template <class Fn>
static void parallel_chunks(int n, int nthreads, Fn&& fn) {
    nthreads = std::max(1, std::min(nthreads, std::max(1, n)));
    if (nthreads == 1) { fn(0, 0, n); return; }
    std::vector<std::thread> ts;
    ts.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        const int b = static_cast<int>(static_cast<int64_t>(t) * n / nthreads);
        const int e = static_cast<int>(static_cast<int64_t>(t + 1) * n / nthreads);
        ts.emplace_back([&fn, t, b, e]() { fn(t, b, e); });
    }
    for (auto& th : ts) th.join();
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
[[maybe_unused]] static std::vector<int> select_k_bic(
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
            // Reduce (parallel over the flat ns*K index)
            parallel_chunks(ns * K, nthreads, [&](int, int i0, int i1) {
                for (int i = i0; i < i1; ++i) {
                    float a = s1_prior, b = s2_prior;
                    for (int t = 0; t < nthreads; ++t) { a += s1_tl[t][i]; b += s2_tl[t][i]; }
                    s1[i] = a;  s2[i] = b;
                }
            });
        }

        // Update beta_mu / beta_sum (stored for output)
        parallel_chunks(ns, nthreads, [&](int, int j0, int j1) {
            for (int j = j0; j < j1; ++j) {
                const float* s1j = s1.data() + j * K;
                const float* s2j = s2.data() + j * K;
                for (int k = 0; k < K; ++k) {
                    result.beta_mu[j][k]  = s1j[k] / (s1j[k] + s2j[k]);
                    result.beta_sum[j][k] = s1j[k] + s2j[k];
                }
            }
        });

        // ── Precompute digamma cache (KEY OPTIMISATION) ─────────────────────
        // Costs O(ns × K) digamma calls once per iter instead of O(nc × snps/cell × K).
        // For ns=5000 K=5: 75K calls vs 67.5M calls in the old inner loop. ~900× speedup.
        parallel_chunks(ns, nthreads, [&](int, int j0, int j1) {
            for (int j = j0; j < j1; ++j) {
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
        });

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
        // Partial sums per thread chunk (parallel; summed in a fixed order).
        std::vector<double> e_part(nthreads, 0.0);

        // E[log p(x | z, theta)] — uses precomputed d1/d2/d12 (no digamma calls here)
        // and -KL[q(z) || p(z)]
        parallel_chunks(nc, nthreads, [&](int t, int c0, int c1) {
            double e = 0.0;
            for (int c = c0; c < c1; ++c) {
                const auto& cell = data.cells[c];
                const float* qc = result.id_prob[c].data();
                for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                    const int   j  = cell.snp_idx[si];
                    const float ad = static_cast<float>(cell.ad[si]);
                    const float dp = static_cast<float>(cell.dp[si]);
                    const float bd = dp - ad;
                    const float* d1j  = d1.data()  + j * K;
                    const float* d2j  = d2.data()  + j * K;
                    const float* d12j = d12.data() + j * K;
                    for (int k = 0; k < K; ++k) {
                        const float qk = qc[k];
                        if (qk < 1e-10f) continue;
                        e += qk * (ad * d1j[k] + bd * d2j[k] - dp * d12j[k]);
                    }
                }
                for (int k = 0; k < K; ++k) {
                    const float qk = qc[k];
                    if (qk > 1e-10f) e -= qk * (std::log(qk) + log_inv_k);
                }
            }
            e_part[t] += e;
        });
        // -KL[q(theta) || p(theta)] — uses precomputed d1/d2 (3 lgamma + d1 + d2 + d12 per j,k)
        parallel_chunks(ns, nthreads, [&](int t, int j0, int j1) {
            double e = 0.0;
            for (int j = j0; j < j1; ++j) {
                const float* s1j  = s1.data()  + j * K;
                const float* s2j  = s2.data()  + j * K;
                const float* d1j  = d1.data()  + j * K;
                const float* d2j  = d2.data()  + j * K;
                const float* d12j = d12.data() + j * K;
                for (int k = 0; k < K; ++k) {
                    const float a = s1j[k], b = s2j[k];
                    e -= std::lgamma(a + b) - std::lgamma(a) - std::lgamma(b)
                         - lgamma_prior
                         + (a - s1_prior) * d1j[k] + (b - s2_prior) * d2j[k]
                         - (a + b - s1_prior - s2_prior) * d12j[k];
                }
            }
            e_part[t] += e;
        });
        double elbo = 0.0;
        for (double v : e_part) elbo += v;

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
// Uses Gram-Schmidt deflation for multiple PCs. The PCs are computed one after
// the other from one RNG stream, so the first m PCs do not depend on n_pcs.
// Returns cell scores [nc × n_pcs] in cell-major layout.
static std::vector<float> power_iteration_pca(
    const CoveredSNPData& data,
    const InformativeSNPs& inf,
    int         n_pcs,
    int         n_iter = 30,
    uint64_t    seed   = 0,
    int         nthreads = 1)
{
    const int nc = data.n_cells;
    const int ni = static_cast<int>(inf.indices.size());
    const int n_pcs_actual = std::min(n_pcs, std::min(nc - 1, ni));
    std::vector<float> result(static_cast<size_t>(nc) * n_pcs, 0.0f);
    if (n_pcs_actual <= 0 || ni == 0 || nc < 2) return result;
    nthreads = std::max(1, nthreads);

    // Build covered_snp → informative-index lookup (-1 if not informative)
    std::vector<int> snp_to_inf(data.n_covered_snps, -1);
    for (int i = 0; i < ni; ++i) snp_to_inf[inf.indices[i]] = i;

    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<std::vector<float>> evecs(n_pcs_actual, std::vector<float>(ni, 0.0f));
    std::vector<float> v(ni), w(nc), v_new(ni);
    std::vector<std::vector<float>> vtl(nthreads, std::vector<float>(ni));

    // w = X_centered * v   (n_cells)
    auto matvec = [&]() {
        parallel_chunks(nc, nthreads, [&](int, int c0, int c1) {
            for (int c = c0; c < c1; ++c) {
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
        });
    };

    for (int pc = 0; pc < n_pcs_actual; ++pc) {
        // Random unit-vector init in SNP space
        float vn = 0.0f;
        for (int i = 0; i < ni; ++i) { v[i] = nd(rng); vn += v[i] * v[i]; }
        vn = std::sqrt(vn) + 1e-12f;
        for (int i = 0; i < ni; ++i) v[i] /= vn;

        for (int it = 0; it < n_iter; ++it) {
            matvec();
            // v_new = X_centered^T * w  (n_snps), thread-local accumulators
            parallel_chunks(nc, nthreads, [&](int t, int c0, int c1) {
                std::vector<float>& acc = vtl[t];
                std::fill(acc.begin(), acc.end(), 0.0f);
                for (int c = c0; c < c1; ++c) {
                    if (w[c] == 0.0f) continue;
                    const auto& cell = data.cells[c];
                    for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                        int ii = snp_to_inf[cell.snp_idx[si]];
                        if (ii < 0 || cell.dp[si] == 0) continue;
                        float af = static_cast<float>(cell.ad[si]) / static_cast<float>(cell.dp[si]);
                        acc[ii] += (af - inf.mean_af[ii]) * w[c];
                    }
                }
            });
            std::fill(v_new.begin(), v_new.end(), 0.0f);
            for (int t = 0; t < nthreads; ++t)
                for (int i = 0; i < ni; ++i) v_new[i] += vtl[t][i];
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
        matvec();
        for (int c = 0; c < nc; ++c)
            result[static_cast<size_t>(c) * n_pcs + pc] = w[c];
    }
    return result;
}

// K-means++ clustering on n_pcs-dimensional cell PCA scores.
// Returns cluster assignments [nc].
static std::vector<int> kmeans_plusplus(
    const std::vector<float>& scores,   // [nc × stride], cell-major; first n_pcs columns used
    int nc, int n_pcs, int K,
    int      kmeans_iter = 20,
    uint64_t seed        = 0,
    int      stride      = -1)
{
    if (stride < 0) stride = n_pcs;
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
            float diff = scores[static_cast<size_t>(c) * stride + p] - cen[p];
            d += diff * diff;
        }
        return d;
    };

    // Pick first center uniformly at random
    std::uniform_int_distribution<int> rpick(0, nc - 1);
    {
        int first = rpick(rng);
        for (int p = 0; p < n_pcs; ++p)
            centroids[0][p] = scores[static_cast<size_t>(first) * stride + p];
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
                centroids[ki][p] = scores[static_cast<size_t>(next) * stride + p];
        } else {
            std::discrete_distribution<int> dd(min_d2.begin(), min_d2.end());
            int next = dd(rng);
            for (int p = 0; p < n_pcs; ++p)
                centroids[ki][p] = scores[static_cast<size_t>(next) * stride + p];
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
                centroids[k][p] += scores[static_cast<size_t>(c) * stride + p];
        }
        for (int k = 0; k < K; ++k) {
            if (counts[k] == 0) continue;
            float inv = 1.0f / counts[k];
            for (int p = 0; p < n_pcs; ++p) centroids[k][p] *= inv;
        }
    }
    return assign;
}

// Outlier-robust seeding: k-means++ with `over`*K centres, then Ward agglomeration down
// to K clusters. Plain k-means++ spends centres on far-away outliers (doublets look
// like that in PC space) and merges real donors; with surplus centres the small outlier
// clusters are the cheapest merges under Ward's criterion and the donors survive.
static std::vector<int> kmeans_ward(
    const std::vector<float>& scores, int nc, int n_pcs, int K,
    uint64_t seed, int stride, int over = 3)
{
    if (K <= 1 || nc <= K) return kmeans_plusplus(scores, nc, n_pcs, K, 20, seed, stride);
    const int K2 = std::max(K, std::min(over * K, nc / 2));
    std::vector<int> a2 = kmeans_plusplus(scores, nc, n_pcs, K2, 20, seed, stride);
    std::vector<std::vector<double>> cen(K2, std::vector<double>(n_pcs, 0.0));
    std::vector<double> cnt(K2, 0.0);
    for (int c = 0; c < nc; ++c) {
        const int k = a2[c];
        cnt[k] += 1.0;
        for (int p = 0; p < n_pcs; ++p) cen[k][p] += scores[static_cast<size_t>(c) * stride + p];
    }
    std::vector<int> parent(K2);
    std::iota(parent.begin(), parent.end(), 0);
    std::vector<int> active;
    for (int k = 0; k < K2; ++k) {
        if (cnt[k] <= 0.0) continue;
        for (int p = 0; p < n_pcs; ++p) cen[k][p] /= cnt[k];
        active.push_back(k);
    }
    while (static_cast<int>(active.size()) > K) {
        double best = std::numeric_limits<double>::infinity();
        size_t bi = 0, bj = 1;
        for (size_t i = 0; i < active.size(); ++i)
            for (size_t j = i + 1; j < active.size(); ++j) {
                const int a = active[i], b = active[j];
                double d2 = 0.0;
                for (int p = 0; p < n_pcs; ++p) {
                    const double df = cen[a][p] - cen[b][p];
                    d2 += df * df;
                }
                const double cost = cnt[a] * cnt[b] / (cnt[a] + cnt[b]) * d2;
                if (cost < best) { best = cost; bi = i; bj = j; }
            }
        const int a = active[bi], b = active[bj];
        for (int p = 0; p < n_pcs; ++p)
            cen[a][p] = (cen[a][p] * cnt[a] + cen[b][p] * cnt[b]) / (cnt[a] + cnt[b]);
        cnt[a] += cnt[b];
        parent[b] = a;
        active.erase(active.begin() + bj);
    }
    // relabel 0..K-1 (a cluster whose parent chain ends at a surviving centre)
    std::vector<int> label(K2, -1);
    for (size_t i = 0; i < active.size(); ++i) label[active[i]] = static_cast<int>(i);
    auto root = [&](int k) { while (parent[k] != k) k = parent[k]; return k; };
    std::vector<int> out(nc);
    for (int c = 0; c < nc; ++c) out[c] = label[root(a2[c])];
    // pad if fewer than K non-empty clusters came out
    return out;
}

// Number of PCs used to seed K clusters: K-1 span K well-separated clusters, but PCs
// absorbed by doublets / depth leave too few, so a few extra are used.
static inline int pca_dims_for_k(int K) { return K - 1 + 3; }

// PCA of a dataset computed once and shared by every K / restart that seeds from it.
struct PcaCache {
    InformativeSNPs inf;
    std::vector<float> scores;   // [nc × n_pcs], cell-major
    int n_pcs = 0;
};

static PcaCache compute_pca_cache(const CoveredSNPData& data, int n_pcs,
                                  int nthreads, uint64_t seed)
{
    PcaCache pc;
    // A SNP must be informative in ~1% of the cells (20 for large inputs, >= 5 for small ones).
    pc.inf = find_informative_snps(data, std::max(5, std::min(20, data.n_cells / 100)));
    pc.n_pcs = std::max(0, n_pcs);
    if (pc.n_pcs > 0)
        pc.scores = power_iteration_pca(data, pc.inf, pc.n_pcs, 30, seed, nthreads);
    return pc;
}

// Build a PCA-seeded VBResult: beta_mu seeded from cluster-mean AFs,
// id_prob from soft (0.9 / 0.1/(K-1)) cluster membership.
// Falls back to uniform random id_prob when too few informative SNPs exist.
static VBResult init_vb_pca_seeded(
    const CoveredSNPData& data, int K,
    const DonorDemuxConfig& cfg, uint64_t seed,
    const PcaCache* cache = nullptr, bool quiet = false,
    int mode = 2)   // 1 = legacy (k-means++ on K-1 PCs), 2 = Ward-merged k-means on K+2 PCs
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

    PcaCache local;
    const int n_pcs = (mode == 1) ? K - 1 : pca_dims_for_k(K);
    if (!cache || cache->n_pcs < n_pcs) {
        local = compute_pca_cache(data, n_pcs, 1, seed);
        cache = &local;
    }
    const InformativeSNPs& inf = cache->inf;
    const int  ni  = static_cast<int>(inf.indices.size());

    if (ni < K * 10) {
        // Not enough informative SNPs — fall back to random init
        if (!quiet)
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
    const auto assign = (mode == 1)
        ? kmeans_plusplus(cache->scores, nc, n_pcs, K, /*kmeans_iter=*/20, seed + 1ULL, cache->n_pcs)
        : kmeans_ward(cache->scores, nc, n_pcs, K, seed + 1ULL, cache->n_pcs);

    if (!quiet) {
        std::cerr << "[demux] PCA init: " << ni << " informative SNPs, "
                  << n_pcs << " PC(s), k-means:";
        std::vector<int> cnts(K, 0);
        for (int a : assign) ++cnts[a];
        for (int k = 0; k < K; ++k) std::cerr << " k" << k << "=" << cnts[k];
        std::cerr << "\n";
    }

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
// Restarted VB fit for one K: PCA/k-means seeded restarts + random restarts run
// in parallel (serial VB each), best ELBO continued with all threads.
// ============================================================================
// Restart kinds: 0 = random responsibilities, 1 = PCA + k-means++ (K-1 PCs, the original
// seeding), 2 = PCA + Ward-merged k-means (K+2 PCs, robust to doublet outliers).
static VBResult fit_vb_k(
    const CoveredSNPData& cov, int K, const DonorDemuxConfig& cfg,
    const std::vector<int>& kinds, int max_iter_pre, int max_iter,
    int threads, const PcaCache* pca, uint64_t seed_base, bool quiet,
    std::vector<VBResult>* all_out = nullptr)
{
    const int n_inits = std::max(1, static_cast<int>(kinds.size()));
    std::vector<VBResult> init_results(n_inits);
    const int nw = std::max(1, std::min(threads, n_inits));

    auto run_batch = [&](int start, int end) {
        for (int i = start; i < end; ++i) {
            const uint64_t init_seed = seed_base
                + static_cast<uint64_t>(i) * 1000ULL
                + static_cast<uint64_t>(K) * 100000ULL;
            const int kind = kinds.empty() ? 0 : kinds[i];
            if (K > 1 && kind > 0) {
                // PCA-seeded restart: prevents symmetric-init cluster collapse
                init_results[i] = init_vb_pca_seeded(cov, K, cfg, init_seed, pca, quiet || i > 0, kind);
                run_vb_inplace(cov, K, cfg, init_results[i], max_iter_pre, /*nthreads=*/1);
            } else {
                init_results[i] = run_vb(cov, K, cfg, init_seed, max_iter_pre);
            }
        }
    };

    if (nw <= 1) {
        run_batch(0, n_inits);
    } else {
        std::vector<std::thread> workers;
        workers.reserve(nw);
        const int per = n_inits / nw, rem = n_inits % nw;
        int start = 0;
        for (int w = 0; w < nw; ++w) {
            const int cnt = per + (w < rem ? 1 : 0);
            workers.emplace_back(run_batch, start, start + cnt);
            start += cnt;
        }
        for (auto& t : workers) t.join();
    }

    int best = 0;
    for (int i = 1; i < n_inits; ++i)
        if (init_results[i].elbo > init_results[best].elbo) best = i;
    if (!quiet) {
        std::cerr << "[demux] K=" << K << " restarts (kind:ELBO after " << max_iter_pre << " it):";
        for (int i = 0; i < n_inits; ++i)
            std::cerr << " " << (kinds.empty() ? 0 : kinds[i]) << ":" << init_results[i].elbo
                      << (i == best ? "*" : "");
        std::cerr << "\n";
    }
    if (all_out) {                       // caller wants every restart (staged search)
        *all_out = std::move(init_results);
        return VBResult();
    }
    VBResult res = std::move(init_results[best]);
    if (max_iter > 0) run_vb_inplace(cov, K, cfg, res, max_iter, threads);
    return res;
}

// ============================================================================
// SNP reduction: keep SNPs seen in >= min_cells of the listed cells, remap indices.
// Cell i of the result is cells[i] of the source; SNP order is preserved.
// ============================================================================
struct ReducedSNPs {
    CoveredSNPData data;
    std::vector<int32_t> to_src;   // reduced SNP -> source covered-SNP index
};

static ReducedSNPs reduce_snps(const CoveredSNPData& src,
                               const std::vector<int>& cells, int min_cells)
{
    std::vector<int32_t> cnt(src.n_covered_snps, 0);
    for (int c : cells)
        for (int32_t j : src.cells[c].snp_idx) ++cnt[j];
    std::vector<int32_t> map(src.n_covered_snps, -1);
    ReducedSNPs r;
    int n = 0;
    for (int j = 0; j < src.n_covered_snps; ++j)
        if (cnt[j] >= min_cells) { map[j] = n++; r.to_src.push_back(j); }
    r.data.n_covered_snps = n;
    r.data.n_cells = static_cast<int>(cells.size());
    r.data.cells.resize(cells.size());
    for (size_t i = 0; i < cells.size(); ++i) {
        const auto& sc = src.cells[cells[i]];
        auto& dc = r.data.cells[i];
        dc.snp_idx.reserve(sc.snp_idx.size());
        dc.ad.reserve(sc.snp_idx.size());
        dc.dp.reserve(sc.snp_idx.size());
        for (size_t si = 0; si < sc.snp_idx.size(); ++si) {
            const int32_t m = map[sc.snp_idx[si]];
            if (m < 0) continue;
            dc.snp_idx.push_back(m);
            dc.ad.push_back(sc.ad[si]);
            dc.dp.push_back(sc.dp[si]);
        }
    }
    return r;
}

// Staged restart search for one K. Exploration is cheap: many restarts on a reduced SNP set
// (SNPs seen in >= cfg.explore_min_snp_cells called cells, a small fraction of the covered SNPs).
// The best cfg.explore_top restarts by reduced-data ELBO are then continued on the full SNP set
// from their cell posteriors, and the best of those is converged. VB has many poor local optima
// (a donor split in two while two others share a component); the ELBO tells good from bad
// reliably (the truth partition's ELBO is ~1.4% higher), but only if enough starts are tried.
static VBResult fit_vb_k_staged(
    const CoveredSNPData& cov, int K, const DonorDemuxConfig& cfg,
    const PcaCache* pca, bool quiet)
{
    std::vector<int> all(cov.n_cells);
    std::iota(all.begin(), all.end(), 0);
    ReducedSNPs red = reduce_snps(cov, all, cfg.explore_min_snp_cells);
    if (red.data.n_covered_snps < std::max(200, 20 * K)) {      // sparse input: no reduction
        red.data = cov;
        red.data.covered_to_original.clear();
    }
    std::vector<int> kinds;
    for (int i = 0; i < cfg.explore_ward; ++i) kinds.push_back(2);
    for (int i = 0; i < cfg.explore_legacy; ++i) kinds.push_back(1);
    for (int i = 0; i < cfg.explore_random; ++i) kinds.push_back(0);
    std::vector<VBResult> all_res;
    const auto t0 = std::chrono::steady_clock::now();
    fit_vb_k(red.data, K, cfg, kinds, cfg.explore_iters, 0, cfg.threads, pca, cfg.seed + 4242ULL, true, &all_res);
    std::vector<int> order(all_res.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return all_res[a].elbo > all_res[b].elbo; });
    const int top = std::max(1, std::min<int>(cfg.explore_top, static_cast<int>(order.size())));
    if (!quiet)
        std::cerr << "[demux] K=" << K << " explored " << kinds.size() << " restarts on " << red.data.n_covered_snps
                  << " SNPs in " << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
                  << " s; reduced ELBO best " << all_res[order[0]].elbo << " (kind " << kinds[order[0]]
                  << "), median " << all_res[order[order.size() / 2]].elbo << "\n";
    VBResult best;
    best.elbo = -std::numeric_limits<double>::infinity();
    for (int t = 0; t < top; ++t) {
        VBResult r;
        r.elbo = -std::numeric_limits<double>::infinity();
        r.n_iter = 0;
        r.beta_mu.assign(cov.n_covered_snps, std::vector<float>(K, 0.5f));
        r.beta_sum.assign(cov.n_covered_snps, std::vector<float>(K, 2.0f));
        r.id_prob = std::move(all_res[order[t]].id_prob);
        run_vb_inplace(cov, K, cfg, r, cfg.explore_full_iters, cfg.threads);
        if (!quiet)
            std::cerr << "[demux]   candidate " << t << " (kind " << kinds[order[t]] << "): full ELBO " << r.elbo << "\n";
        if (r.elbo > best.elbo) best = std::move(r);
    }
    run_vb_inplace(cov, K, cfg, best, cfg.max_iter, cfg.threads);
    return best;
}

// ============================================================================
// Genotype-regularised allele probabilities per donor and SNP.
//
// The VB posterior mean (Beta(1,1) prior) is too diffuse for doublet scoring: a
// homozygous donor with three ref reads gets f = 0.2, not ~0. Here every donor's
// weighted read counts (A alt, B ref) at a SNP are scored under three genotype states
// with beta-binomial likelihoods,
//     0/0: theta ~ Beta(eps*c_hom, (1-eps)*c_hom)    0/1: theta ~ Beta(c_het/2, c_het/2)
//     1/1: theta ~ Beta((1-eps)*c_hom, eps*c_hom)
// (HWE prior from the pooled alt fraction of the SNP; eps = per-read error, re-estimated
// by EM from the data) and F = sum_g P(g | A, B) * E[theta | g, A, B]. With few reads F
// is pulled to the genotype means, with many reads it follows the data, so allele-specific
// bias or contamination cannot make a donor impossible to fit.
// ============================================================================
struct AlleleTables {
    int K = 0, ns = 0;
    float eps = 0.0f;                 // fitted per-read error rate
    std::vector<float> F, G, lf, lg;  // F, 1-F, log F, log(1-F); index j*K + k
};

static AlleleTables build_allele_tables(
    const CoveredSNPData& d,
    const std::vector<std::vector<float>>& q,   // [nc][K] donor weights per cell
    const std::vector<float>* cell_w,           // optional extra weight per cell
    const std::vector<uint8_t>* use,            // optional mask: cells that contribute counts
    int K, const DonorDemuxConfig& cfg, float eps_fixed = -1.0f)
{
    const int nc = d.n_cells, ns = d.n_covered_snps;
    const int nthreads = std::max(1, cfg.threads);
    AlleleTables T;
    T.K = K; T.ns = ns;
    const size_t nk = static_cast<size_t>(ns) * K;
    std::vector<float> A(nk, 0.0f), Dp(nk, 0.0f);
    for (int c = 0; c < nc; ++c) {
        if (use && !(*use)[c]) continue;
        const float w = cell_w ? (*cell_w)[c] : 1.0f;
        if (w <= 1e-4f) continue;
        const auto& cell = d.cells[c];
        const float* qc = q[c].data();
        for (int k = 0; k < K; ++k) {
            const float qk = qc[k] * w;
            if (qk < 1e-3f) continue;
            for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                const size_t o = static_cast<size_t>(cell.snp_idx[si]) * K + k;
                A[o]  += qk * static_cast<float>(cell.ad[si]);
                Dp[o] += qk * static_cast<float>(cell.dp[si]);
            }
        }
    }

    // Pooled alt fraction per SNP -> HWE genotype prior
    double gA = 0.0, gD = 0.0;
    for (size_t i = 0; i < nk; ++i) { gA += A[i]; gD += Dp[i]; }
    const float m_glob = static_cast<float>(
        std::max(0.1, std::min(0.9, (gA + 1.0) / (gD + 2.0))));
    std::vector<float> lpr0(ns), lpr1(ns), lpr2(ns);
    for (int j = 0; j < ns; ++j) {
        float sA = 0.0f, sD = 0.0f;
        for (int k = 0; k < K; ++k) { sA += A[static_cast<size_t>(j) * K + k]; sD += Dp[static_cast<size_t>(j) * K + k]; }
        const float p = std::max(0.01f, std::min(0.99f, (sA + 2.0f * m_glob) / (sD + 2.0f)));
        lpr0[j] = 2.0f * std::log(1.0f - p);
        lpr1[j] = std::log(2.0f * p * (1.0f - p));
        lpr2[j] = 2.0f * std::log(p);
    }

    const double c_hom = std::max(4.0f, cfg.geno_conc_hom), c_het = std::max(4.0f, cfg.geno_conc_het);
    // Per eps: state Beta parameters and normalisers
    struct States {
        double a[3], b[3], c[3], ln[3];
        void set(double eps, double c_hom, double c_het) {
            a[0] = eps * c_hom;         b[0] = (1.0 - eps) * c_hom;
            a[1] = 0.5 * c_het;         b[1] = 0.5 * c_het;
            a[2] = (1.0 - eps) * c_hom; b[2] = eps * c_hom;
            for (int g = 0; g < 3; ++g) {
                c[g] = a[g] + b[g];
                ln[g] = std::lgamma(a[g]) + std::lgamma(b[g]) - std::lgamma(c[g]);
            }
        }
    };
    // posterior over genotype states and their conditional allele probabilities
    auto geno = [&](const States& st, size_t o, int j, double* pg, double* eg) {
        const double a = A[o], dpt = Dp[o], bb = std::max(0.0, dpt - a);
        double l[3] = {lpr0[j], lpr1[j], lpr2[j]};
        if (dpt > 0.0)
            for (int g = 0; g < 3; ++g)
                l[g] += std::lgamma(st.a[g] + a) + std::lgamma(st.b[g] + bb)
                      - std::lgamma(st.c[g] + dpt) - st.ln[g];
        const double mx = std::max(l[0], std::max(l[1], l[2]));
        double sum = 0.0;
        for (int g = 0; g < 3; ++g) { pg[g] = std::exp(l[g] - mx); sum += pg[g]; }
        for (int g = 0; g < 3; ++g) {
            pg[g] /= sum;
            eg[g] = (st.a[g] + a) / (st.c[g] + dpt);
        }
    };

    // EM for the per-read error: confident homozygous sites explain alt reads at 0/0
    // (and ref reads at 1/1) only through eps (sequencing error, ambient RNA).
    float eps = std::max(0.003f, std::min(0.15f, cfg.geno_error));
    if (eps_fixed > 0.0f) {
        eps = eps_fixed;
    } else {
        for (int it = 0; it < 3; ++it) {
            States st;
            st.set(eps, c_hom, c_het);
            std::vector<double> num(nthreads, 0.0), den(nthreads, 0.0);
            parallel_chunks(ns, nthreads, [&](int t, int j0, int j1) {
                double nu = 0.0, de = 0.0, pg[3], eg[3];
                for (int j = j0; j < j1; ++j)
                    for (int k = 0; k < K; ++k) {
                        const size_t o = static_cast<size_t>(j) * K + k;
                        if (Dp[o] <= 0.0f) continue;
                        geno(st, o, j, pg, eg);
                        const double b = std::max(0.0f, Dp[o] - A[o]);
                        nu += pg[0] * A[o] + pg[2] * b;
                        de += (pg[0] + pg[2]) * Dp[o];
                    }
                num[t] += nu; den[t] += de;
            });
            double nu = 0.0, de = 0.0;
            for (int t = 0; t < nthreads; ++t) { nu += num[t]; de += den[t]; }
            if (de > 0.0)
                eps = std::max(0.003f, std::min(0.15f, static_cast<float>(nu / de)));
        }
    }
    T.eps = eps;

    T.F.resize(nk); T.G.resize(nk); T.lf.resize(nk); T.lg.resize(nk);
    States st;
    st.set(eps, c_hom, c_het);
    parallel_chunks(ns, nthreads, [&](int, int j0, int j1) {
        double pg[3], eg[3];
        for (int j = j0; j < j1; ++j)
            for (int k = 0; k < K; ++k) {
                const size_t o = static_cast<size_t>(j) * K + k;
                geno(st, o, j, pg, eg);
                double f = pg[0] * eg[0] + pg[1] * eg[1] + pg[2] * eg[2];
                f = std::max(1e-4, std::min(1.0 - 1e-4, f));
                T.F[o] = static_cast<float>(f);
                T.G[o] = static_cast<float>(1.0 - f);
                T.lf[o] = static_cast<float>(std::log(f));
                T.lg[o] = static_cast<float>(std::log(1.0 - f));
            }
    });
    return T;
}

// ============================================================================
// Vireo-style singlet / doublet model.
//   P(cell | donor k)            = prod_snps Binom(ad | dp, F_jk)
//   P(cell | doublet of (a, b))  = prod_snps Binom(ad | dp, (F_ja + F_jb) / 2)
// Priors: singlet k pi_k * (1 - d_het); doublet {a,b} d * 2 pi_a pi_b, where d is
// the total multiplet rate and d_het = d * (1 - sum pi^2) the detectable share
// (homotypic doublets look like singlets and stay in the singlet mass).
// ============================================================================
struct CellScore {
    float  p_doublet = 0.0f;   // posterior P(heterotypic doublet)
    int    best_single = -1;
    float  p_single = 0.0f;    // posterior of best_single given a singlet
    int    pa = -1, pb = -1;   // most likely donor pair (pa < pb)
    float  llr = 0.0f;         // best pair log-lik - best single log-lik (reads only)
    double ll_marg = 0.0;      // log marginal likelihood of the cell under the mixture
};

static void make_log_priors(const std::vector<double>& pi, double delta,
                            std::vector<double>& lps, std::vector<double>& lpp)
{
    const int K = static_cast<int>(pi.size());
    double s2 = 0.0;
    for (double v : pi) s2 += v * v;
    delta = std::max(0.0, std::min(0.9, delta));
    if (K < 2) delta = 0.0;
    const double d_het = delta * (1.0 - s2);
    lps.assign(K, 0.0);
    lpp.assign(static_cast<size_t>(K) * (K - 1) / 2, 0.0);
    for (int k = 0; k < K; ++k)
        lps[k] = std::log(std::max(1e-12, pi[k] * (1.0 - d_het)));
    size_t idx = 0;
    for (int a = 0; a < K; ++a)
        for (int b = a + 1; b < K; ++b)
            lpp[idx++] = std::log(std::max(1e-300, delta * 2.0 * pi[a] * pi[b]));
}

// Score cells[i] (indices into d) under tables T. out[i] aligned with cells.
// single_post (optional): out[i] = posterior over the K singlet hypotheses given "singlet".
static void score_cells(
    const CoveredSNPData& d, const std::vector<int>& cells, const AlleleTables& T,
    const std::vector<double>& lps, const std::vector<double>& lpp,
    int nthreads, std::vector<CellScore>& out,
    std::vector<std::vector<float>>* single_post = nullptr,
    std::vector<float>* half_ll = nullptr,   // optional [n][2][K + K(K-1)/2]: singlet then pair log-lik
                                             // from the even / odd observations of each cell
    float temper = 1.0f)                     // < 1 flattens the read likelihood (reads of one cell are
                                             // not independent: PCR, bursting) so posteriors are calibrated
{
    const int K = T.K;
    const int npair = K * (K - 1) / 2;
    const int W = K + npair;
    const int n = static_cast<int>(cells.size());
    out.assign(n, CellScore{});
    if (single_post) single_post->assign(n, std::vector<float>(K, 1.0f / std::max(1, K)));
    if (half_ll) half_ll->assign(static_cast<size_t>(n) * 2 * W, 0.0f);
    parallel_chunks(n, nthreads, [&](int, int i0, int i1) {
        std::vector<float> ls(K), lse(K), lso(K), lp(std::max(1, npair)),
                           lpe(std::max(1, npair)), lpo(std::max(1, npair));
        std::vector<double> js(K), jp(std::max(1, npair));
        for (int i = i0; i < i1; ++i) {
            const auto& cell = d.cells[cells[i]];
            std::fill(lse.begin(), lse.end(), 0.0f);
            std::fill(lso.begin(), lso.end(), 0.0f);
            std::fill(lpe.begin(), lpe.end(), 0.0f);
            std::fill(lpo.begin(), lpo.end(), 0.0f);
            for (size_t si = 0; si < cell.snp_idx.size(); ++si) {
                const size_t o = static_cast<size_t>(cell.snp_idx[si]) * K;
                const float ad = static_cast<float>(cell.ad[si]);
                const float bd = static_cast<float>(cell.dp[si]) - ad;
                const float* lfj = T.lf.data() + o;
                const float* lgj = T.lg.data() + o;
                float* dst = (si & 1) ? lso.data() : lse.data();
                for (int k = 0; k < K; ++k) dst[k] += ad * lfj[k] + bd * lgj[k];
                if (npair > 0) {
                    const float* Fj = T.F.data() + o;
                    const float* Gj = T.G.data() + o;
                    float* pd = (si & 1) ? lpo.data() : lpe.data();
                    int idx = 0;
                    if (ad > 0.0f && bd > 0.0f) {
                        for (int a = 0; a < K; ++a)
                            for (int b = a + 1; b < K; ++b)
                                pd[idx++] += ad * std::log(0.5f * (Fj[a] + Fj[b]))
                                           + bd * std::log(0.5f * (Gj[a] + Gj[b]));
                    } else if (ad > 0.0f) {
                        for (int a = 0; a < K; ++a)
                            for (int b = a + 1; b < K; ++b)
                                pd[idx++] += ad * std::log(0.5f * (Fj[a] + Fj[b]));
                    } else {
                        for (int a = 0; a < K; ++a)
                            for (int b = a + 1; b < K; ++b)
                                pd[idx++] += bd * std::log(0.5f * (Gj[a] + Gj[b]));
                    }
                }
            }
            for (int k = 0; k < K; ++k) ls[k] = lse[k] + lso[k];
            for (int p = 0; p < npair; ++p) lp[p] = lpe[p] + lpo[p];
            if (half_ll) {
                float* he = half_ll->data() + (static_cast<size_t>(i) * 2) * W;
                float* ho = he + W;
                for (int k = 0; k < K; ++k) { he[k] = lse[k]; ho[k] = lso[k]; }
                for (int p = 0; p < npair; ++p) { he[K + p] = lpe[p]; ho[K + p] = lpo[p]; }
            }
            CellScore sc;
            int bs = 0;
            double mx = -std::numeric_limits<double>::infinity();
            for (int k = 0; k < K; ++k) {
                js[k] = static_cast<double>(temper) * ls[k] + lps[k];
                if (js[k] > js[bs]) bs = k;
                mx = std::max(mx, js[k]);
            }
            int bp = -1;
            for (int p = 0; p < npair; ++p) {
                jp[p] = static_cast<double>(temper) * lp[p] + lpp[p];
                if (bp < 0 || jp[p] > jp[bp]) bp = p;
                mx = std::max(mx, jp[p]);
            }
            double sum_s = 0.0, sum_p = 0.0;
            for (int k = 0; k < K; ++k) sum_s += std::exp(js[k] - mx);
            for (int p = 0; p < npair; ++p) sum_p += std::exp(jp[p] - mx);
            const double tot = sum_s + sum_p;
            sc.ll_marg = mx + std::log(tot);
            sc.p_doublet = static_cast<float>(sum_p / tot);
            sc.best_single = bs;
            sc.p_single = static_cast<float>(std::exp(js[bs] - mx) / sum_s);
            if (bp >= 0) {
                int idx = 0;
                for (int a = 0; a < K && sc.pa < 0; ++a)
                    for (int b = a + 1; b < K; ++b, ++idx)
                        if (idx == bp) { sc.pa = a; sc.pb = b; break; }
                float best_ls = ls[0];
                for (int k = 1; k < K; ++k) best_ls = std::max(best_ls, ls[k]);
                sc.llr = lp[bp] - best_ls;
            }
            if (single_post) {
                auto& sp = (*single_post)[i];
                for (int k = 0; k < K; ++k)
                    sp[k] = static_cast<float>(std::exp(js[k] - mx) / sum_s);
            }
            out[i] = sc;
        }
    });
}

static inline int cell_fold(int c) {
    uint32_t x = static_cast<uint32_t>(c) * 2654435761u;
    x ^= x >> 15;
    return static_cast<int>((x >> 7) & 1u);
}

static inline double multiplet_prior(const DonorDemuxConfig& cfg, int n_cells) {
    if (cfg.doublet_prior >= 0.0f) return cfg.doublet_prior;
    return std::min(0.4, static_cast<double>(cfg.doublet_rate_per_1000) * n_cells / 1000.0);
}

static std::vector<double> mean_donor_weights(const std::vector<std::vector<float>>& q, int K) {
    std::vector<double> pi(K, 0.0);
    for (const auto& r : q) for (int k = 0; k < K; ++k) pi[k] += r[k];
    double tot = 0.0;
    for (double v : pi) tot += v;
    for (double& v : pi) v = std::max(1e-4, v / std::max(1e-9, tot));
    double s = 0.0;
    for (double v : pi) s += v;
    for (double& v : pi) v /= s;
    return pi;
}

// ============================================================================
// Doublet stage on the cells the VB model was fit on.
// Round 1 builds genotypes from the VB posteriors; round 2 rebuilds them with each
// cell down-weighted by (1 - P(doublet)), so doublets stop contaminating donors.
// With cfg.doublet_crossfit each half of the cells is scored with genotypes built
// from the other half (a cell's own reads never vouch for its own call).
// ============================================================================
static void doublet_stage(
    const CoveredSNPData& cov, int K, const DonorDemuxConfig& cfg,
    const std::vector<std::vector<float>>& q,
    std::vector<CellScore>& score, float* eps_out = nullptr,
    const CoveredSNPData* extra = nullptr, std::vector<CellScore>* extra_score = nullptr)
{
    // `extra`: more cells (same SNP index space) that were not used to fit the donors; they are
    // scored at the end with genotypes built from all fitted cells.
    const int nc = cov.n_cells;
    const int ne = extra ? extra->n_cells : 0;
    CoveredSNPData both;
    if (ne > 0) {
        both.n_covered_snps = cov.n_covered_snps;
        both.n_cells = nc + ne;
        both.cells = cov.cells;
        both.cells.insert(both.cells.end(), extra->cells.begin(), extra->cells.end());
    }
    std::vector<int> all(nc + ne);
    std::iota(all.begin(), all.end(), 0);
    ReducedSNPs red = reduce_snps(ne > 0 ? both : cov, all, cfg.doublet_min_snp_cells);
    const CoveredSNPData& d = red.data;

    const std::vector<double> pi = mean_donor_weights(q, K);
    std::vector<double> lps, lpp;
    make_log_priors(pi, multiplet_prior(cfg, nc), lps, lpp);

    std::vector<int> fold_cells[2];
    for (int c = 0; c < nc; ++c) fold_cells[cfg.doublet_crossfit ? cell_fold(c) : 0].push_back(c);

    score.assign(nc, CellScore{});
    std::vector<float> w;     // (1 - P(doublet)) after the first round
    std::vector<float>* wp = nullptr;
    const int rounds = std::max(1, cfg.doublet_rounds);
    for (int r = 0; r < rounds; ++r) {
        for (int f = 0; f < 2; ++f) {
            if (fold_cells[f].empty()) continue;
            std::vector<uint8_t> use(nc + ne, 1);
            for (int c = nc; c < nc + ne; ++c) use[c] = 0;
            if (cfg.doublet_crossfit)
                for (int c : fold_cells[f]) use[c] = 0;   // genotypes from the other half
            AlleleTables T = build_allele_tables(d, q, wp, &use, K, cfg);
            if (eps_out) *eps_out = T.eps;
            std::vector<CellScore> out;
            score_cells(d, fold_cells[f], T, lps, lpp, cfg.threads, out, nullptr, nullptr, cfg.doublet_temper);
            for (size_t i = 0; i < fold_cells[f].size(); ++i) score[fold_cells[f][i]] = out[i];
        }
        if (r + 1 < rounds) {
            w.assign(nc, 1.0f);
            for (int c = 0; c < nc; ++c) w[c] = 1.0f - score[c].p_doublet;
            wp = &w;
        }
    }
    if (ne > 0 && extra_score) {
        std::vector<float> wf(nc + ne, 0.0f);
        for (int c = 0; c < nc; ++c) wf[c] = 1.0f - score[c].p_doublet;
        std::vector<uint8_t> use(nc + ne, 0);
        for (int c = 0; c < nc; ++c) use[c] = 1;
        AlleleTables T = build_allele_tables(d, q, &wf, &use, K, cfg);
        std::vector<int> ids(ne);
        std::iota(ids.begin(), ids.end(), nc);
        score_cells(d, ids, T, lps, lpp, cfg.threads, *extra_score, nullptr, nullptr, cfg.doublet_temper);
    }
}

// ============================================================================
// Auto-K: cross-fitted held-out model selection with a donor-distinctness check.
//
// The cells are split in two random halves. For K = 1, 2, ... a K-component VB model
// is fit on one half, its genotypes (and mixing proportions) are frozen and the other
// half is scored with the full singlet + doublet mixture, so every cell is scored by a
// model that never saw it. The held-out log-likelihood is reported per K, but the
// decision is made on how many of the K components are real donors:
//   * a donor owns at least max(cfg.min_donor_cells, cfg.min_donor_frac) of the cells;
//   * two components are the same donor unless held-out cells of each are better
//     explained by their own component than by the other one, judged on observations
//     that were not used to assign the cell (assignment on the even half of a cell's
//     SNP observations, evidence on the odd half and vice versa; mean log-likelihood
//     ratio > cfg.k_select_sep_min and mean / s.e. > cfg.k_select_sep_z). Splits of one
//     donor fail this test, and so do tiny outlier clusters (doublets, debris).
// The estimate is the largest number of valid donors found for any K; the search stops
// after cfg.k_select_patience consecutive K without a new maximum. Both folds must
// agree (the smaller count of the two is used).
// ============================================================================
struct KSelectRow {
    int    K = 0;
    double ll_per_cell = 0.0;   // mean held-out log-likelihood per cell
    double gain = 0.0;          // mean paired gain over the previous K
    double z = 0.0;
    int    valid = 0;           // distinct donors among the K components (min over folds)
    bool   accepted = false;    // this K raised the best count
    double seconds = 0.0;
};

// Number of components of a K-model that are real, distinct donors, judged on held-out cells.
// half_ll[((2i + h) * W) + x] (W = K + K(K-1)/2): log-lik of cell i from its even (h=0) / odd
// (h=1) observations under hypothesis x = single donor k, or donor pair (a<b) in row-major order.
//
// Component m is a donor if its own cells (assigned to "single m" with one half of their
// observations) are better explained by "single m" than by the best hypothesis that does
// not involve m (another donor, or a doublet of two other donors), judged on the other half.
// Backward elimination: the weakest failing component is dropped and the test repeated,
// so two copies of the same donor leave one survivor.
static int count_valid_components(const std::vector<float>& half_ll, int n, int K,
                                  const std::vector<double>& lps,
                                  const std::vector<double>& lpp,
                                  const DonorDemuxConfig& cfg,
                                  std::vector<char>* alive_out = nullptr)
{
    const int npair = K * (K - 1) / 2;
    const int W = K + npair;
    std::vector<int> pa(npair), pb(npair);
    {
        int idx = 0;
        for (int a = 0; a < K; ++a)
            for (int b = a + 1; b < K; ++b) { pa[idx] = a; pb[idx] = b; ++idx; }
    }
    auto term = [&](int i, int h, int x) -> double {
        return static_cast<double>(half_ll[(static_cast<size_t>(i) * 2 + h) * W + x])
             + (x < K ? lps[x] : lpp[x - K]);
    };
    // sizes from the hard singlet assignment on all observations
    std::vector<int> cnt(K, 0);
    for (int i = 0; i < n; ++i) {
        int b = 0;
        double vb = -1e300;
        for (int k = 0; k < K; ++k) {
            const double a = term(i, 0, k) + term(i, 1, k) - lps[k];
            if (a > vb) { vb = a; b = k; }
        }
        ++cnt[b];
    }
    const int min_n = std::max(cfg.min_donor_cells,
                               static_cast<int>(std::ceil(cfg.min_donor_frac * n)));
    std::vector<char> alive(K, 1);
    for (int k = 0; k < K; ++k) alive[k] = cnt[k] >= min_n;

    for (int round = 0; round < K; ++round) {
        int n_alive = 0;
        for (int k = 0; k < K; ++k) n_alive += alive[k] ? 1 : 0;
        if (n_alive <= 1) break;
        // hypotheses available while the dead components are ignored
        std::vector<char> hyp(W, 0);
        for (int k = 0; k < K; ++k) hyp[k] = alive[k];
        for (int p = 0; p < npair; ++p) hyp[K + p] = alive[pa[p]] && alive[pb[p]];

        int worst = -1;
        double worst_z = 1e300;
        for (int m = 0; m < K; ++m) {
            if (!alive[m]) continue;
            double s = 0.0, s2 = 0.0;
            int cnt_m = 0;
            for (int h = 0; h < 2; ++h) {
                const int g = 1 - h;     // assign on half h, evidence on half g
                for (int i = 0; i < n; ++i) {
                    int bx = -1;
                    double bv = -1e300;
                    for (int x = 0; x < W; ++x) {
                        if (!hyp[x]) continue;
                        const double v = term(i, h, x);
                        if (v > bv) { bv = v; bx = x; }
                    }
                    if (bx != m) continue;
                    double alt = -1e300;
                    for (int x = 0; x < W; ++x) {
                        if (!hyp[x]) continue;
                        if (x < K ? (x == m) : (pa[x - K] == m || pb[x - K] == m)) continue;
                        alt = std::max(alt, term(i, g, x));
                    }
                    const double dlt = term(i, g, m) - alt;
                    s += dlt; s2 += dlt * dlt; ++cnt_m;
                }
            }
            bool pass = false;
            double z = -1e300;
            if (cnt_m >= 10) {
                const double mean = s / cnt_m;
                const double var = std::max(1e-12, (s2 / cnt_m - mean * mean) * cnt_m / (cnt_m - 1.0));
                z = mean / std::sqrt(var / (0.5 * cnt_m));   // cells are counted once per half
                pass = mean > cfg.k_select_sep_min && z > cfg.k_select_sep_z;
            }
            if (!pass && z < worst_z) { worst_z = z; worst = m; }
        }
        if (worst < 0) break;
        alive[worst] = 0;
    }
    int v = 0;
    for (int k = 0; k < K; ++k) v += alive[k] ? 1 : 0;
    if (alive_out) *alive_out = alive;
    return std::max(1, v);
}

static int select_k_heldout(const CoveredSNPData& cov, const DonorDemuxConfig& cfg,
                            std::vector<KSelectRow>* trace = nullptr)
{
    const auto t0 = std::chrono::steady_clock::now();
    // Cells with any data, shuffled, capped
    std::vector<int> idx;
    for (int c = 0; c < cov.n_cells; ++c)
        if (!cov.cells[c].snp_idx.empty()) idx.push_back(c);
    {
        std::mt19937 rng(cfg.seed + 99999ULL);
        std::shuffle(idx.begin(), idx.end(), rng);
    }
    if (cfg.k_select_max_cells > 0 && static_cast<int>(idx.size()) > cfg.k_select_max_cells)
        idx.resize(cfg.k_select_max_cells);
    const int n = static_cast<int>(idx.size());
    if (n < 40) {
        std::cerr << "[demux] auto-K: only " << n << " cells with data, K=1\n";
        return 1;
    }

    ReducedSNPs red = reduce_snps(cov, idx, cfg.k_select_min_snp_cells);
    const int ns = red.data.n_covered_snps;
    // Two halves sharing the reduced SNP space; global position i = 2*j + f
    CoveredSNPData D[2];
    for (int f = 0; f < 2; ++f) {
        D[f].n_covered_snps = ns;
        D[f].n_cells = 0;
    }
    for (int i = 0; i < n; ++i) {
        const int f = i & 1;
        D[f].cells.push_back(std::move(red.data.cells[i]));
        ++D[f].n_cells;
    }
    const int kmax = std::max(1, std::min(cfg.max_donors, n / 25));
    std::cerr << "[demux] auto-K: " << n << " cells, " << ns << " SNPs (seen in >= "
              << cfg.k_select_min_snp_cells << " cells), K in 1.." << kmax << "\n";

    PcaCache pca[2];
    const int n_pcs = pca_dims_for_k(kmax);
    for (int f = 0; f < 2; ++f)
        pca[f] = compute_pca_cache(D[f], n_pcs, cfg.threads, cfg.seed + 17ULL + f);

    DonorDemuxConfig fcfg = cfg;                 // per-fold thread budget (folds run side by side)
    fcfg.threads = std::max(1, cfg.threads / 2);
    const double delta = multiplet_prior(cfg, cov.n_cells);
    std::vector<double> prev_ll;
    int best_v = 1, stall = 0;

    for (int K = 1; K <= kmax; ++K) {
        const auto tk = std::chrono::steady_clock::now();
        std::vector<double> ll(n, 0.0);
        int valid_f[2] = {1, 1};

        auto do_fold = [&](int f) {
            const std::vector<int> kinds = (K == 1) ? std::vector<int>{0}
                                                    : std::vector<int>(cfg.k_select_inits, 2);
            VBResult fit = fit_vb_k(D[f], K, fcfg, kinds, std::min(cfg.max_iter_pre, 30),
                                    cfg.k_select_max_iter, fcfg.threads, &pca[f],
                                    cfg.seed + 31ULL * (f + 1), true);
            std::vector<double> lps, lpp;
            make_log_priors(mean_donor_weights(fit.id_prob, K), delta, lps, lpp);
            // Genotypes from the training half; the second round drops likely doublets
            // (scored in-sample) so they do not contaminate the donors.
            AlleleTables T = build_allele_tables(D[f], fit.id_prob, nullptr, nullptr, K, fcfg);
            if (K >= 2 && cfg.detect_doublets) {
                std::vector<int> train(D[f].n_cells);
                std::iota(train.begin(), train.end(), 0);
                std::vector<CellScore> tsc;
                score_cells(D[f], train, T, lps, lpp, fcfg.threads, tsc);
                std::vector<float> w(D[f].n_cells);
                for (int j = 0; j < D[f].n_cells; ++j) w[j] = 1.0f - tsc[j].p_doublet;
                T = build_allele_tables(D[f], fit.id_prob, &w, nullptr, K, fcfg, T.eps);
            }
            std::vector<int> test(D[1 - f].n_cells);
            std::iota(test.begin(), test.end(), 0);
            std::vector<CellScore> sc;
            std::vector<float> half;
            score_cells(D[1 - f], test, T, lps, lpp, fcfg.threads, sc, nullptr, &half);
            for (int j = 0; j < D[1 - f].n_cells; ++j) ll[2 * j + (1 - f)] = sc[j].ll_marg;
            valid_f[f] = (K == 1) ? 1 : count_valid_components(half, D[1 - f].n_cells, K, lps, lpp, cfg);
        };
        if (K >= 2 && cfg.threads >= 2) {
            std::thread t1(do_fold, 1);
            do_fold(0);
            t1.join();
        } else {
            do_fold(0);
            do_fold(1);
        }

        double mean = 0.0;
        for (double v : ll) mean += v;
        mean /= n;
        KSelectRow row;
        row.K = K; row.ll_per_cell = mean;
        row.valid = std::min(valid_f[0], valid_f[1]);
        if (!prev_ll.empty()) {   // paired gain over the best accepted model so far
            double md = 0.0, m2 = 0.0;
            for (int i = 0; i < n; ++i) {
                const double dd = ll[i] - prev_ll[i];
                md += dd; m2 += dd * dd;
            }
            md /= n;
            const double var = std::max(1e-12, (m2 / n - md * md) * n / (n - 1.0));
            row.gain = md;
            row.z = md / std::sqrt(var / n);
        }
        if (K == 1) {
            prev_ll = ll;
        } else if (row.valid > best_v && row.z > cfg.k_select_z &&
                   row.gain * n >= cfg.k_select_min_gain) {
            // more distinct donors AND the held-out likelihood really improved: a component that
            // only fragments a donor (e.g. by cell type, whose SNP coverage differs) passes the
            // distinctness test but does not predict held-out cells better.
            best_v = row.valid; prev_ll = ll; stall = 0; row.accepted = true;
        } else {
            ++stall;
        }
        row.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tk).count();
        std::cerr << "[demux] auto-K K=" << K << ": held-out ll/cell=" << mean
                  << " gain=" << row.gain << " z=" << row.z << " distinct donors=" << valid_f[0]
                  << "/" << valid_f[1] << (row.accepted ? " *" : "") << " (" << row.seconds << " s)\n";
        if (trace) trace->push_back(row);
        if (stall >= cfg.k_select_patience) break;
    }
    std::cerr << "[demux] auto-K: selected K=" << best_v << " in "
              << std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()
              << " s\n";
    return best_v;
}

// ============================================================================
// Split-merge repair of a fitted K-donor VB model.
//
// VB from a PCA / random start often stops in a local optimum where two donors share one
// component while a third donor is split over two (a "merge + split" away from the optimum).
// Each round tries: merge the genotype-closest pair of components and split another
// component in two (2-means on the PC scores of its cells), runs VB briefly from that
// state, and keeps the best candidate if its ELBO beats the current converged ELBO.
// ============================================================================
static void split_merge_refine(const CoveredSNPData& cov, int K, const DonorDemuxConfig& cfg,
                               VBResult& res, const PcaCache* pca)
{
    if (K < 3 || cfg.split_merge_rounds <= 0 || !pca || pca->scores.empty()) return;
    const int nc = cov.n_cells, ns = cov.n_covered_snps;
    const int dims = std::min(pca->n_pcs, pca_dims_for_k(K));
    for (int round = 0; round < cfg.split_merge_rounds; ++round) {
        // hard assignment and sizes
        std::vector<int> hard(nc, 0), cnt(K, 0);
        for (int c = 0; c < nc; ++c) {
            int b = 0;
            for (int k = 1; k < K; ++k) if (res.id_prob[c][k] > res.id_prob[c][b]) b = k;
            hard[c] = b; ++cnt[b];
        }
        // genotype distance between components over SNPs both have evidence for
        std::vector<double> num(static_cast<size_t>(K) * K, 0.0), den(static_cast<size_t>(K) * K, 0.0);
        for (int j = 0; j < ns; ++j) {
            const auto& mu = res.beta_mu[j];
            const auto& sm = res.beta_sum[j];
            for (int a = 0; a < K; ++a) {
                if (sm[a] < 6.0f) continue;
                for (int b = a + 1; b < K; ++b) {
                    if (sm[b] < 6.0f) continue;
                    const double w = std::min(sm[a], sm[b]);
                    num[a * K + b] += w * std::fabs(mu[a] - mu[b]);
                    den[a * K + b] += w;
                }
            }
        }
        std::vector<std::pair<double, std::pair<int, int>>> pairs;
        for (int a = 0; a < K; ++a)
            for (int b = a + 1; b < K; ++b)
                if (den[a * K + b] > 0.0) pairs.push_back({num[a * K + b] / den[a * K + b], {a, b}});
        std::sort(pairs.begin(), pairs.end());

        // split labels per component (2-means on PC scores of its cells)
        std::vector<std::vector<int>> members(K);
        for (int c = 0; c < nc; ++c) members[hard[c]].push_back(c);
        std::vector<std::vector<char>> split_lab(K);   // 0/1 per member
        for (int m = 0; m < K; ++m) {
            if (static_cast<int>(members[m].size()) < std::max(2 * cfg.min_donor_cells, 40)) continue;
            std::vector<float> sc(members[m].size() * dims);
            for (size_t i = 0; i < members[m].size(); ++i)
                for (int p = 0; p < dims; ++p)
                    sc[i * dims + p] = pca->scores[static_cast<size_t>(members[m][i]) * pca->n_pcs + p];
            const auto lab = kmeans_plusplus(sc, static_cast<int>(members[m].size()), dims, 2, 30,
                                             cfg.seed + 777ULL + m, dims);
            split_lab[m].resize(lab.size());
            int n1 = 0;
            for (size_t i = 0; i < lab.size(); ++i) { split_lab[m][i] = static_cast<char>(lab[i]); n1 += lab[i]; }
            if (n1 < cfg.min_donor_cells || static_cast<int>(lab.size()) - n1 < cfg.min_donor_cells)
                split_lab[m].clear();
        }

        struct Cand { int a, b, m; VBResult r; };
        std::vector<Cand> cands;
        // (b) re-split: pool the cells of two components and cut them in two again by 2-means
        // (fixes "three donors in two components": two comps hold {A1, A2+B} instead of {A, B})
        for (size_t pi = 0; pi < pairs.size() && pi < static_cast<size_t>(cfg.split_merge_resplit_pairs); ++pi) {
            const int a = pairs[pi].second.first, b = pairs[pi].second.second;
            std::vector<int> mem;
            for (int c = 0; c < nc; ++c) if (hard[c] == a || hard[c] == b) mem.push_back(c);
            if (static_cast<int>(mem.size()) < 4 * cfg.min_donor_cells) continue;
            std::vector<float> sc(mem.size() * dims);
            for (size_t i = 0; i < mem.size(); ++i)
                for (int p = 0; p < dims; ++p)
                    sc[i * dims + p] = pca->scores[static_cast<size_t>(mem[i]) * pca->n_pcs + p];
            const auto lab = kmeans_ward(sc, static_cast<int>(mem.size()), dims, 2,
                                         cfg.seed + 991ULL + a * 31 + b, dims, 4);
            Cand cd;
            cd.a = a; cd.b = b; cd.m = -1;
            cd.r.elbo = -std::numeric_limits<double>::infinity();
            cd.r.n_iter = 0;
            cd.r.beta_mu.assign(ns, std::vector<float>(K, 0.5f));
            cd.r.beta_sum.assign(ns, std::vector<float>(K, 2.0f));
            cd.r.id_prob = res.id_prob;
            for (size_t i = 0; i < mem.size(); ++i) {
                auto& q = cd.r.id_prob[mem[i]];
                const float mass = q[a] + q[b];
                q[a] = lab[i] == 0 ? mass : 0.0f;
                q[b] = lab[i] == 0 ? 0.0f : mass;
            }
            cands.push_back(std::move(cd));
        }
        // (a) merge the closest pair and split another component
        for (size_t pi = 0; pi < pairs.size() && pi < static_cast<size_t>(cfg.split_merge_merge_cands); ++pi) {
            const auto& pr = pairs[pi];
            for (int m = 0; m < K; ++m) {
                if (split_lab[m].empty() || m == pr.second.first || m == pr.second.second) continue;
                Cand cd;
                cd.a = pr.second.first; cd.b = pr.second.second; cd.m = m;
                cd.r.elbo = -std::numeric_limits<double>::infinity();
                cd.r.n_iter = 0;
                cd.r.beta_mu.assign(ns, std::vector<float>(K, 0.5f));
                cd.r.beta_sum.assign(ns, std::vector<float>(K, 2.0f));
                cd.r.id_prob = res.id_prob;
                for (int c = 0; c < nc; ++c) {              // merge b into a
                    cd.r.id_prob[c][cd.a] += cd.r.id_prob[c][cd.b];
                    cd.r.id_prob[c][cd.b] = 0.0f;
                }
                for (size_t i = 0; i < members[m].size(); ++i)   // split m into m and b
                    if (split_lab[m][i]) {
                        auto& q = cd.r.id_prob[members[m][i]];
                        q[cd.b] = q[m]; q[m] = 0.0f;
                    }
                cands.push_back(std::move(cd));
            }
        }
        if (cands.empty()) break;

        const int nw = std::max(1, std::min(cfg.threads, static_cast<int>(cands.size())));
        std::vector<std::thread> ws;
        std::atomic<int> next{0};
        for (int w = 0; w < nw; ++w)
            ws.emplace_back([&]() {
                for (int i = next++; i < static_cast<int>(cands.size()); i = next++)
                    run_vb_inplace(cov, K, cfg, cands[i].r, cfg.split_merge_iters, 1);
            });
        for (auto& t : ws) t.join();
        int bi = 0;
        for (size_t i = 1; i < cands.size(); ++i) if (cands[i].r.elbo > cands[bi].r.elbo) bi = static_cast<int>(i);
        std::cerr << "[demux] split-merge round " << round + 1 << ": " << cands.size()
                  << " candidates, best (merge " << cands[bi].a << "+" << cands[bi].b << ", split "
                  << cands[bi].m << ") ELBO " << cands[bi].r.elbo << " vs current " << res.elbo << "\n";
        if (!(cands[bi].r.elbo > res.elbo + cfg.split_merge_min_gain_rel * std::fabs(res.elbo))) break;
        res = std::move(cands[bi].r);
        run_vb_inplace(cov, K, cfg, res, cfg.max_iter, cfg.threads);
        std::cerr << "[demux] split-merge accepted: ELBO " << res.elbo << "\n";
    }
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
    std::vector<KSelectRow> k_selection;           // auto-K trace (empty when K was given)
    int n_doublets = 0;                            // called cells labelled "doublet"
    float geno_error = 0.0f;                       // fitted per-read genotype error (doublet model)
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

    // Core / extra split of the called cells: low-depth called cells (few SNP observations) carry
    // little genotype information, blur the donor components and push VB into local optima
    // (two donors in one component, a third split in two). The donors are fit on the core cells
    // (>= cfg.core_min_rel x the median number of SNP observations) and the remaining called
    // cells are assigned afterwards like ambient barcodes, but keep status "cell".
    std::vector<uint32_t> fit_idx, extra_idx;
    if (use_cell_filter) {
        fit_idx = cell_indices;
        if (cfg.core_min_rel > 0.0f) {
            std::vector<size_t> nobs;
            nobs.reserve(cell_indices.size());
            for (uint32_t idx : cell_indices) nobs.push_back(cov.cells[idx].snp_idx.size());
            std::nth_element(nobs.begin(), nobs.begin() + nobs.size() / 2, nobs.end());
            const double thr = cfg.core_min_rel * static_cast<double>(nobs[nobs.size() / 2]);
            std::vector<uint32_t> core;
            for (uint32_t idx : cell_indices) {
                if (static_cast<double>(cov.cells[idx].snp_idx.size()) >= thr) core.push_back(idx);
                else extra_idx.push_back(idx);
            }
            if (extra_idx.size() * 50 < cell_indices.size() || core.size() < 200) {
                extra_idx.clear();                 // nothing worth separating
            } else {
                fit_idx = std::move(core);
                std::cerr << "[demux] Core fit: " << fit_idx.size() << " cells with >= " << thr
                          << " SNP observations; " << extra_idx.size()
                          << " low-depth called cells are assigned afterwards\n";
            }
        }
    }

    CoveredSNPData cov_filt;
    if (use_cell_filter) {
        // Share the same SNP index space as cov_all; only subset the cell list.
        cov_filt.n_covered_snps    = cov.n_covered_snps;
        cov_filt.n_cells           = static_cast<int>(fit_idx.size());
        cov_filt.covered_to_original = cov.covered_to_original;
        cov_filt.cells.reserve(fit_idx.size());
        for (uint32_t idx : fit_idx)
            cov_filt.cells.push_back(cov.cells[idx]);
        std::cerr << "[demux] Cell filter: running VB on "
                  << cov_filt.n_cells << " called cells (of " << n_cells
                  << " total barcodes); "
                  << (n_cells - static_cast<uint32_t>(cov_filt.n_cells))
                  << " other barcodes will be back-assigned post-convergence\n";
    }
    // VB model operates on this (filtered or full)
    const CoveredSNPData& cov_use = use_cell_filter ? cov_filt : cov;

    // ── Determine K candidates ────────────────────────────────────────────────
    DemuxResult demux_result;
    std::vector<int> k_candidates;
    if (cfg.n_donors > 0) {
        k_candidates.push_back(cfg.n_donors);
    } else if (cfg.fast_k_select) {
        // Held-out likelihood search (see select_k_heldout)
        k_candidates.push_back(select_k_heldout(cov_use, cfg, &demux_result.k_selection));
    } else {
        for (int K = 1; K <= cfg.max_donors; ++K) k_candidates.push_back(K);
    }

    // ── Full VB for the candidate K values ───────────────────────────────────
    double   best_elbo   = -std::numeric_limits<double>::infinity();
    int      best_k      = k_candidates[0];
    VBResult best_result;

    double prev2_k_elbo = -std::numeric_limits<double>::infinity();
    double prev1_k_elbo = -std::numeric_limits<double>::infinity();

    const int max_cand = *std::max_element(k_candidates.begin(), k_candidates.end());
    PcaCache pca_full;
    if (max_cand > 1)
        pca_full = compute_pca_cache(cov_use, pca_dims_for_k(max_cand), cfg.threads,
                                     cfg.seed + static_cast<uint64_t>(max_cand) * 100000ULL);

    for (int K : k_candidates) {
        int n_inits = cfg.n_inits;
        if (K <= 3 && n_inits > 3) n_inits = 3;
        // restart 0: PCA + k-means++ (original seeding); then random restarts; then Ward-seeded ones
        std::vector<int> kinds;
        if (K > 1) {
            kinds.push_back(1);
            for (int i = 1; i < n_inits; ++i) kinds.push_back(0);
            for (int i = 0; i < cfg.n_ward_inits; ++i) kinds.push_back(2);
        } else {
            kinds.push_back(0);
        }

        VBResult k_best_result = (cfg.staged_fit && K > 1 && cov_use.n_cells >= 200)
            ? fit_vb_k_staged(cov_use, K, cfg, &pca_full, false)
            : fit_vb_k(cov_use, K, cfg, kinds, cfg.max_iter_pre, cfg.max_iter, cfg.threads,
                       &pca_full, cfg.seed, false);

        std::cerr << "[demux] K=" << K << ": ELBO=" << k_best_result.elbo
                  << " (" << k_best_result.n_iter << " iters)\n";

        const double k_elbo = k_best_result.elbo;
        if (k_elbo > best_elbo) {
            best_elbo   = k_elbo;
            best_k      = K;
            best_result = std::move(k_best_result);
        }

        // ── K-level early stopping (full sweep only) ──────────────────────────
        if (prev2_k_elbo > -1e300 &&
            k_elbo < prev1_k_elbo &&
            prev1_k_elbo       < prev2_k_elbo) {
            std::cerr << "[demux] K-level early stop at K=" << K << "\n";
            break;
        }
        prev2_k_elbo = prev1_k_elbo;
        prev1_k_elbo = best_elbo;
    }

    if (best_k >= 3) {
        split_merge_refine(cov_use, best_k, cfg, best_result, &pca_full);
        best_elbo = best_result.elbo;
    }
    std::cerr << "[demux] Best K=" << best_k << " (ELBO=" << best_elbo << ")\n";

    // ── Doublet model on the called cells ────────────────────────────────────
    std::vector<CellScore> dbl, dbl_extra;
    const bool have_dbl = cfg.detect_doublets && best_k >= 2;
    if (have_dbl) {
        const auto td = std::chrono::steady_clock::now();
        CoveredSNPData extra_cov;
        if (!extra_idx.empty()) {
            extra_cov.n_covered_snps = cov.n_covered_snps;
            extra_cov.n_cells = static_cast<int>(extra_idx.size());
            for (uint32_t idx : extra_idx) extra_cov.cells.push_back(cov.cells[idx]);
        }
        doublet_stage(cov_use, best_k, cfg, best_result.id_prob, dbl, &demux_result.geno_error,
                      extra_idx.empty() ? nullptr : &extra_cov, &dbl_extra);
        // Re-fit the VB donor genotypes without the likely doublets (one extra round):
        // scale each cell's posterior by (1 - P(doublet)) for one M-step, then E-step.
        for (int c = 0; c < cov_use.n_cells; ++c) {
            const float w = 1.0f - dbl[c].p_doublet;
            for (float& v : best_result.id_prob[c]) v *= w;
        }
        run_vb_inplace(cov_use, best_k, cfg, best_result, 1, cfg.threads);
        double exp_dbl = 0.0;
        for (const auto& s : dbl) exp_dbl += s.p_doublet;
        std::cerr << "[demux] Doublet model: expected doublets=" << exp_dbl
                  << " (prior " << multiplet_prior(cfg, cov_use.n_cells)
                  << "), geno error=" << demux_result.geno_error << ", "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - td).count()
                  << " s\n";
    }

    // ── Assign cells and back-assign ambient barcodes ─────────────────────────
    // Default sentinel: unassigned ambient (overwritten below for called cells)
    std::vector<DonorAssignment> assignments(n_cells,
        {best_k, 0.0f, 0.0f, "unassigned", "ambient"});

    int n_assigned_cells = 0,   n_unassigned_cells = 0, n_doublet_cells = 0;
    int n_assigned_ambient = 0, n_unassigned_ambient = 0;

    // Helper: pick best donor from a posterior probability vector; `sc` (optional)
    // carries the doublet model's verdict for this cell.
    auto assign_from_probs = [&](const std::vector<float>& probs,
                                 const std::string& status,
                                 const CellScore* sc) -> DonorAssignment {
        int best_donor = 0;
        float best_prob = probs[0];
        for (int k = 1; k < best_k; ++k)
            if (probs[k] > best_prob) { best_prob = probs[k]; best_donor = k; }
        DonorAssignment a{best_k, best_prob, 0.0f, "unassigned", status};
        a.best_singlet = best_donor;
        if (sc) {
            a.prob_doublet = sc->p_doublet;
            a.doublet_a = sc->pa;
            a.doublet_b = sc->pb;
            a.doublet_llr = sc->llr;
            if (sc->p_doublet > cfg.doublet_threshold) {
                a.label = "doublet";
                return a;
            }
        }
        if (best_prob >= cfg.min_prob) {
            a.donor_id = best_donor;
            a.label = "donor" + std::to_string(best_donor);
        }
        return a;
    };

    if (!use_cell_filter) {
        // ── No filter: all barcodes are treated as cells (original path) ──────
        for (int c = 0; c < static_cast<int>(n_cells); ++c) {
            assignments[c] = assign_from_probs(best_result.id_prob[c], "cell",
                                               have_dbl ? &dbl[c] : nullptr);
            if (assignments[c].label == "unassigned") ++n_unassigned_cells;
            else if (assignments[c].label == "doublet") ++n_doublet_cells;
            else ++n_assigned_cells;
        }
    } else {
        // ── 1. Assign called cells from VB posteriors ─────────────────────────
        for (size_t i = 0; i < fit_idx.size(); ++i) {
            const uint32_t orig_c = fit_idx[i];
            assignments[orig_c] = assign_from_probs(best_result.id_prob[i], "cell",
                                                    have_dbl ? &dbl[i] : nullptr);
            if (assignments[orig_c].label == "unassigned") ++n_unassigned_cells;
            else if (assignments[orig_c].label == "doublet") ++n_doublet_cells;
            else ++n_assigned_cells;
        }

        // ── 2. Back-assign ambient barcodes via single E-step ─────────────────
        // Precompute digamma cache from converged Beta posterior (ns × K lookups,
        // avoids N_ambient × snps/cell × K digamma calls in the inner loop).
        const int ns = cov_use.n_covered_snps;
        std::vector<float> d1a(static_cast<size_t>(ns) * best_k), d2a(static_cast<size_t>(ns) * best_k),
                           d12a(static_cast<size_t>(ns) * best_k);
        for (int j = 0; j < ns; ++j) {
            for (int k = 0; k < best_k; ++k) {
                const float mu  = best_result.beta_mu[j][k];
                const float sum = best_result.beta_sum[j][k];
                const float s1  = mu * sum;
                const float s2  = (1.0f - mu) * sum;
                d1a [static_cast<size_t>(j) * best_k + k] = digamma(s1);
                d2a [static_cast<size_t>(j) * best_k + k] = digamma(s2);
                d12a[static_cast<size_t>(j) * best_k + k] = digamma(sum);
            }
        }

        // Build called-cell set for O(1) skip
        std::unordered_set<uint32_t> cell_set(fit_idx.begin(), fit_idx.end());
        std::unordered_map<uint32_t, int> extra_pos;     // low-depth called cells (status "cell")
        for (size_t i = 0; i < extra_idx.size(); ++i) extra_pos[extra_idx[i]] = static_cast<int>(i);

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
                const size_t j  = static_cast<size_t>(cell.snp_idx[si]);
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

            auto ep = extra_pos.find(c);
            if (ep != extra_pos.end()) {
                assignments[c] = assign_from_probs(probs, "cell",
                    (have_dbl && ep->second < static_cast<int>(dbl_extra.size())) ? &dbl_extra[ep->second] : nullptr);
                if (assignments[c].label == "unassigned") ++n_unassigned_cells;
                else if (assignments[c].label == "doublet") ++n_doublet_cells;
                else ++n_assigned_cells;
                continue;
            }
            assignments[c] = assign_from_probs(probs, "ambient", nullptr);
            if (assignments[c].label == "unassigned") ++n_unassigned_ambient;
            else ++n_assigned_ambient;
        }
    }

    std::cerr << "[demux] Called cells: " << n_assigned_cells << " assigned, "
              << n_doublet_cells << " doublet, "
              << n_unassigned_cells << " unassigned\n";
    if (use_cell_filter) {
        std::cerr << "[demux] Ambient barcodes: " << n_assigned_ambient << " assigned, "
                  << n_unassigned_ambient << " unassigned\n";
    }

    demux_result.assignments = std::move(assignments);
    // Move beta arrays after back-assignment is complete (d1a/d2a/d12a hold copies)
    demux_result.beta_mu             = std::move(best_result.beta_mu);
    demux_result.beta_sum            = std::move(best_result.beta_sum);
    demux_result.covered_to_original = use_cell_filter
        ? cov_filt.covered_to_original           // copy (already a copy of cov's)
        : std::move(cov.covered_to_original);    // move for no-filter case
    demux_result.n_donors_k = best_k;
    demux_result.n_doublets = n_doublet_cells;
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
        // best_singlet: most likely single donor (even when the call is doublet/unassigned);
        // best_doublet: most likely donor pair "donorA,donorB" (Vireo convention).
        const std::string best_singlet = a.best_singlet >= 0
            ? "donor" + std::to_string(a.best_singlet) : a.label;
        const std::string best_doublet = a.doublet_a >= 0
            ? "donor" + std::to_string(a.doublet_a) + ",donor" + std::to_string(a.doublet_b)
            : (a.prob_doublet > 0.5f ? "doublet" : a.label);
        f << barcodes[i] << "\t" << a.label << "\t"
          << a.prob_max << "\t" << a.prob_doublet << "\t"
          << best_singlet << "\t" << best_doublet << "\t"
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
