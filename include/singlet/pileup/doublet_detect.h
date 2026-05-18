// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: doublet_detect.h
// N12: Doublet detection using a simulation-based kNN approach (v6 — GMM threshold).
//
// Algorithm (Wolock et al. 2019, Scrublet):
//   1. Select top K = min(n_cells, 200) highly variable genes (by dispersion = var/mean).
//   2. Extract HVG submatrix: raw_real (n_cells × K) dense float, raw counts.
//   3. Log-normalize each cell (CP10K log1p): norm_real (n_cells × K).
//   4. Generate n_sim = min(2*n_cells, 20000) synthetic doublets:
//      each = sum of two randomly chosen raw cells, then log-normalized.
//   5. Power-iteration PCA on centred combined matrix (n_total = n_cells + n_sim) × K:
//      - Covariance C = X^T X (K × K); 25 iterations for top 20 PCs.
//   6. k-NN: k = max(5, round(0.5 * sqrt(n_cells))), brute-force in PCA space.
//   7. raw_score[i] = fraction of k-NN that are synthetic doublets.
//      Background level for random singlets ≈ sim_frac = n_sim / n_total.
//      Normalize to background-subtracted [0,1] score:
//        norm[i] = clamp((raw[i] - sim_frac) / (1 - sim_frac), 0, 1)
//      Random singlets → norm ≈ 0; perfect doublets → norm ≈ 1.
//   8. GMM threshold (v6 — GMM threshold):
//      Fit 2-component Gaussian mixture to norm scores via EM (100 iter).
//      If bimodal (μ₁-μ₀ > 0.15 AND π₁ ∈ [0.005, 0.80]):
//        threshold = Bayes valley between the two components (binary search).
//      Else (unimodal/degenerate):
//        threshold = expected-rate quantile (sort descending, take top fraction).
//      Always clamped to [0.15, 0.90].
//
//   Fallback (n_cells < 50 or n_genes < 10): UMI-ratio heuristic with [0,1] clamping.
//
// Fix history:
//   v2: divided raw_score by sim_frac (inflating scores above 1.0), renormalized by
//       max_raw → 42.9% false positive rate on gold cells (Jaccard=0.0074 vs Scrublet).
//   v3: background-subtract (raw - sim_frac) then normalize by max_adj — still over-
//       flagged because max normalization collapses the natural [0,1] range.
//   v4: raw kNN fraction directly as score, threshold=0.5 — still wrong: threshold=0.5
//       is BELOW sim_frac≈0.667, so random-background singlets are flagged as doublets.
//       Root cause: with n_sim=2*n_cells, sim_frac=0.667 > threshold=0.5, so any cell
//       with random kNN composition (raw_score≈0.667) exceeds the threshold.
//   v5: adaptive threshold = (sim_frac + 1.0) / 2.0 ≈ 0.833.  Always above sim_frac;
//       places the cutoff midway between background and perfect-doublet scores.
//       Reduces FPR from 42.9% → ≤9% (even for sparse/unclustered data).
//       Flaw: static worst-case bound; ~8.8% FPR on small datasets (k=6, n≈100).
//   v6: background-subtracted score normalization + 2-component GMM threshold.
//       Adapts to the actual bimodal distribution; <2% FPR on structured data.
//
// API:
//   struct DoubletResult { double score; bool is_doublet; }
//   detect_doublets(counts, cell_indices, expected_doublet_rate=0.08) -> vector<DoubletResult>
//   write_doublet_tsv(path, results, cell_indices, barcodes, counts)

#include "sparse_accumulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace singlet {

// -- Result type -------------------------------------------------------------

struct DoubletResult {
    double score;       ///< doublet_score in [0, 1]
    bool   is_doublet;  ///< true if score > threshold (default 0.5)
};

// -- Internal helpers --------------------------------------------------------

namespace doublet_detail {

/// Modified Gram-Schmidt orthonormalization of columns of mat (K × N_PCS, row-major).
static void gram_schmidt(std::vector<float>& mat, int K, int N_PCS)
{
    for (int j = 0; j < N_PCS; ++j) {
        // Normalise column j
        double norm2 = 0.0;
        for (int i = 0; i < K; ++i) {
            double v = mat[static_cast<size_t>(i) * N_PCS + j];
            norm2 += v * v;
        }
        float inv = (norm2 > 1e-10) ? 1.0f / static_cast<float>(std::sqrt(norm2)) : 0.0f;
        for (int i = 0; i < K; ++i)
            mat[static_cast<size_t>(i) * N_PCS + j] *= inv;

        // Subtract projection from remaining columns
        for (int jj = j + 1; jj < N_PCS; ++jj) {
            double dot = 0.0;
            for (int i = 0; i < K; ++i)
                dot += static_cast<double>(mat[static_cast<size_t>(i) * N_PCS + j])
                     * static_cast<double>(mat[static_cast<size_t>(i) * N_PCS + jj]);
            float f = static_cast<float>(dot);
            for (int i = 0; i < K; ++i)
                mat[static_cast<size_t>(i) * N_PCS + jj] -=
                    f * mat[static_cast<size_t>(i) * N_PCS + j];
        }
    }
}

/// Compute symmetric covariance C = X^T X (K × K, row-major).
/// X is (n × K) row-major.
static void compute_cov(const std::vector<float>& X, int n, int K,
                         std::vector<float>& C)
{
    C.assign(static_cast<size_t>(K) * K, 0.0f);
    for (int i = 0; i < n; ++i) {
        const float* row = X.data() + static_cast<size_t>(i) * K;
        for (int a = 0; a < K; ++a) {
            float ra = row[a];
            if (ra == 0.0f) continue;
            for (int b = a; b < K; ++b)
                C[static_cast<size_t>(a) * K + b] += ra * row[b];
        }
    }
    // Symmetrise upper triangle to lower
    for (int a = 0; a < K; ++a)
        for (int b = a + 1; b < K; ++b)
            C[static_cast<size_t>(b) * K + a] = C[static_cast<size_t>(a) * K + b];
}

/// Multiply C (K × K) @ Q (K × N_PCS) → R (K × N_PCS), all row-major.
static void matmul_sq(const std::vector<float>& C, const std::vector<float>& Q,
                       std::vector<float>& R, int K, int N_PCS)
{
    R.assign(static_cast<size_t>(K) * N_PCS, 0.0f);
    for (int i = 0; i < K; ++i)
        for (int p = 0; p < K; ++p) {
            float cip = C[static_cast<size_t>(i) * K + p];
            if (cip == 0.0f) continue;
            for (int j = 0; j < N_PCS; ++j)
                R[static_cast<size_t>(i) * N_PCS + j] +=
                    cip * Q[static_cast<size_t>(p) * N_PCS + j];
        }
}

/// Project X (n × K) @ Q (K × N_PCS) → PCA (n × N_PCS), all row-major.
static void project(const std::vector<float>& X, const std::vector<float>& Q,
                    std::vector<float>& PCA, int n, int K, int N_PCS)
{
    PCA.assign(static_cast<size_t>(n) * N_PCS, 0.0f);
    for (int i = 0; i < n; ++i)
        for (int p = 0; p < K; ++p) {
            float xip = X[static_cast<size_t>(i) * K + p];
            if (xip == 0.0f) continue;
            for (int j = 0; j < N_PCS; ++j)
                PCA[static_cast<size_t>(i) * N_PCS + j] +=
                    xip * Q[static_cast<size_t>(p) * N_PCS + j];
        }
}

/// In-place CP10K log1p normalisation of a single row.
static void log_normalize_row(float* row, int K)
{
    float sum = 0.0f;
    for (int j = 0; j < K; ++j) sum += row[j];
    if (sum <= 0.0f) return;
    float scale = 10000.0f / sum;
    for (int j = 0; j < K; ++j)
        row[j] = std::log1p(row[j] * scale);
}

/// Fit a 2-component GMM to normalized doublet scores and return a threshold.
/// Uses 100 EM iterations initialized at μ₀=0.15 (singlets), μ₁=0.75 (doublets).
/// Falls back to expected-rate quantile when the distribution is unimodal or
/// the dataset is too small.
/// Result is clamped to [0.15, 0.90].
static double gmm_threshold(const std::vector<float>& norm_scores,
                             double expected_doublet_rate)
{
    const int n = static_cast<int>(norm_scores.size());

    // Rate-based fallback: sort descending, take the score at top-fraction index
    auto rate_fallback = [&]() -> double {
        std::vector<float> tmp(norm_scores);
        std::sort(tmp.begin(), tmp.end(), std::greater<float>());
        int idx = static_cast<int>(std::round(expected_doublet_rate * n));
        idx = std::max(0, std::min(idx, n - 1));
        double val = static_cast<double>(tmp[idx]);
        return std::max(0.15, std::min(0.90, val));
    };

    if (n < 20) return rate_fallback();

    auto gauss = [](double x, double mu, double sig) -> double {
        const double inv_sqrt2pi = 0.3989422804014327;
        double z = (x - mu) / sig;
        return (inv_sqrt2pi / sig) * std::exp(-0.5 * z * z);
    };

    // Initialize GMM parameters
    double mu0  = 0.15,  mu1  = 0.75;
    double sig0 = 0.12,  sig1 = 0.12;
    double pi1  = expected_doublet_rate;
    double pi0  = 1.0 - pi1;

    // EM iterations
    std::vector<double> r1(n);  // responsibility of component 1
    for (int iter = 0; iter < 100; ++iter) {
        // E-step: compute responsibilities
        for (int i = 0; i < n; ++i) {
            double x  = static_cast<double>(norm_scores[i]);
            double p0 = pi0 * gauss(x, mu0, sig0);
            double p1 = pi1 * gauss(x, mu1, sig1);
            double denom = p0 + p1;
            r1[i] = (denom > 1e-300) ? p1 / denom : 0.5;
        }

        // M-step
        double sum_r1 = 0.0, sum_r0 = 0.0;
        for (int i = 0; i < n; ++i) {
            sum_r1 += r1[i];
            sum_r0 += 1.0 - r1[i];
        }
        if (sum_r0 < 1e-10 || sum_r1 < 1e-10) break;

        double new_mu0 = 0.0, new_mu1 = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = static_cast<double>(norm_scores[i]);
            new_mu0 += (1.0 - r1[i]) * x;
            new_mu1 += r1[i] * x;
        }
        new_mu0 /= sum_r0;
        new_mu1 /= sum_r1;

        double new_sig0 = 0.0, new_sig1 = 0.0;
        for (int i = 0; i < n; ++i) {
            double x = static_cast<double>(norm_scores[i]);
            new_sig0 += (1.0 - r1[i]) * (x - new_mu0) * (x - new_mu0);
            new_sig1 += r1[i] * (x - new_mu1) * (x - new_mu1);
        }
        new_sig0 = std::sqrt(std::max(1e-6, new_sig0 / sum_r0));
        new_sig1 = std::sqrt(std::max(1e-6, new_sig1 / sum_r1));

        double new_pi1 = sum_r1 / n;
        double new_pi0 = 1.0 - new_pi1;

        mu0 = new_mu0; mu1 = new_mu1;
        sig0 = new_sig0; sig1 = new_sig1;
        pi0 = new_pi0; pi1 = new_pi1;

        // Keep mu0 < mu1 by swapping if needed
        if (mu0 > mu1) {
            std::swap(mu0, mu1);
            std::swap(sig0, sig1);
            std::swap(pi0, pi1);
            for (int i = 0; i < n; ++i) r1[i] = 1.0 - r1[i];
        }
    }

    // Check bimodality: separation > 0.15 AND pi1 in [0.005, 0.80]
    const bool bimodal = (mu1 - mu0 > 0.15) && (pi1 >= 0.005) && (pi1 <= 0.80);
    if (!bimodal) return rate_fallback();

    // Binary search for Bayes valley: π₀·N(x;μ₀,σ₀) = π₁·N(x;μ₁,σ₁) in [mu0, mu1]
    double lo = mu0, hi = mu1;
    for (int i = 0; i < 60; ++i) {
        double mid = 0.5 * (lo + hi);
        double diff = pi0 * gauss(mid, mu0, sig0) - pi1 * gauss(mid, mu1, sig1);
        if (diff > 0.0) lo = mid; else hi = mid;
    }
    double valley = 0.5 * (lo + hi);
    return std::max(0.15, std::min(0.90, valley));
}

} // namespace doublet_detail

// -- Core API ----------------------------------------------------------------

/// Detect doublets using a Scrublet-style simulation kNN scorer (v6 — GMM threshold).
///
/// @param counts                Full CSC exon count matrix (nrows=genes, ncols=all barcodes)
/// @param cell_indices          Indices of called cells within counts
/// @param expected_doublet_rate Expected fraction of doublets; used to seed GMM and fallback
/// @return                      Per-cell DoubletResult, one entry per cell_index (same order)
inline std::vector<DoubletResult> detect_doublets(
    const SparseAccumulator<uint16_t>::CSCMatrix& counts,
    const std::vector<uint32_t>& cell_indices,
    double expected_doublet_rate = 0.08)
{
    const uint32_t n_cells = static_cast<uint32_t>(cell_indices.size());
    std::vector<DoubletResult> results(n_cells);

    if (n_cells == 0) return results;

    const uint32_t n_genes = counts.nrows;

    // -------------------------------------------------------------------------
    // Fallback: too few cells or genes → UMI-ratio heuristic, clamped to [0,1]
    // -------------------------------------------------------------------------
    if (n_cells < 50 || n_genes < 10) {
        std::vector<uint32_t> cell_umis(n_cells, 0u);
        for (uint32_t i = 0; i < n_cells; ++i) {
            const uint32_t j = cell_indices[i];
            for (int32_t k = counts.indptr[j]; k < counts.indptr[j + 1]; ++k)
                cell_umis[i] += counts.data[k];
        }
        std::vector<uint32_t> sorted = cell_umis;
        std::sort(sorted.begin(), sorted.end());
        const double median = 0.5 * (sorted[(n_cells - 1) / 2] + sorted[n_cells / 2]);
        const double denom  = (median > 0.0) ? 3.0 * median : 1.0;
        for (uint32_t i = 0; i < n_cells; ++i) {
            const double s    = (median > 0.0) ? std::min(1.0, cell_umis[i] / denom) : 0.0;
            results[i].score      = s;
            results[i].is_doublet = (s > expected_doublet_rate);
        }
        return results;
    }

    // -------------------------------------------------------------------------
    // Step 1: Select top K highly variable genes (dispersion = variance / mean)
    // -------------------------------------------------------------------------
    const int K_max = 200;
    const int K = std::min({ static_cast<int>(n_genes),
                              static_cast<int>(n_cells),
                              K_max });
    if (K < 10) {
        for (auto& r : results) { r.score = 0.0; r.is_doublet = false; }
        return results;
    }

    std::vector<double> gene_sum(n_genes, 0.0);
    std::vector<double> gene_sum2(n_genes, 0.0);
    for (uint32_t i = 0; i < n_cells; ++i) {
        const uint32_t j = cell_indices[i];
        for (int32_t k = counts.indptr[j]; k < counts.indptr[j + 1]; ++k) {
            const uint32_t g = static_cast<uint32_t>(counts.indices[k]);
            const double   v = counts.data[k];
            gene_sum[g]  += v;
            gene_sum2[g] += v * v;
        }
    }

    // Dispersion = variance / mean
    std::vector<std::pair<double, uint32_t>> disp(n_genes);
    const double inv_nc = 1.0 / n_cells;
    for (uint32_t g = 0; g < n_genes; ++g) {
        const double mean = gene_sum[g] * inv_nc;
        const double var  = gene_sum2[g] * inv_nc - mean * mean;
        disp[g] = { (mean > 0.0) ? var / mean : 0.0, g };
    }
    std::partial_sort(disp.begin(), disp.begin() + K, disp.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    // gene_index → HVG position (UINT32_MAX = not selected)
    std::vector<uint32_t> gene_to_hvg(n_genes, UINT32_MAX);
    for (int ki = 0; ki < K; ++ki)
        gene_to_hvg[disp[ki].second] = static_cast<uint32_t>(ki);

    // -------------------------------------------------------------------------
    // Step 2: Build raw HVG submatrix for real cells — raw_real (n_cells × K)
    // -------------------------------------------------------------------------
    std::vector<float> raw_real(static_cast<size_t>(n_cells) * K, 0.0f);
    for (uint32_t i = 0; i < n_cells; ++i) {
        const uint32_t j = cell_indices[i];
        float* row = raw_real.data() + static_cast<size_t>(i) * K;
        for (int32_t k = counts.indptr[j]; k < counts.indptr[j + 1]; ++k) {
            const uint32_t hvg = gene_to_hvg[static_cast<uint32_t>(counts.indices[k])];
            if (hvg != UINT32_MAX)
                row[hvg] = static_cast<float>(counts.data[k]);
        }
    }

    // -------------------------------------------------------------------------
    // Step 3: Log-normalise real cells → norm_real (n_cells × K)
    // -------------------------------------------------------------------------
    std::vector<float> norm_real = raw_real;
    for (uint32_t i = 0; i < n_cells; ++i)
        doublet_detail::log_normalize_row(
            norm_real.data() + static_cast<size_t>(i) * K, K);

    // -------------------------------------------------------------------------
    // Step 4: Generate synthetic doublets — norm_sim (n_sim × K)
    // n_sim = min(2*n_cells, 20000) — larger cap for better kNN coverage.
    // Each synthetic doublet = sum of two randomly chosen raw cells, then normalised.
    // -------------------------------------------------------------------------
    const uint32_t n_sim = std::min(2u * n_cells, 20000u);
    std::vector<float> norm_sim(static_cast<size_t>(n_sim) * K, 0.0f);
    {
        std::mt19937 rng_sim(static_cast<uint32_t>(n_cells) * 2654435761u ^ 42u);
        std::uniform_int_distribution<uint32_t> cell_dist(0, n_cells - 1);
        for (uint32_t s = 0; s < n_sim; ++s) {
            const uint32_t ci = cell_dist(rng_sim);
            const uint32_t cj = cell_dist(rng_sim);
            const float* ri = raw_real.data() + static_cast<size_t>(ci) * K;
            const float* rj = raw_real.data() + static_cast<size_t>(cj) * K;
            float* out_row  = norm_sim.data()  + static_cast<size_t>(s)  * K;
            for (int p = 0; p < K; ++p) out_row[p] = ri[p] + rj[p];
            doublet_detail::log_normalize_row(out_row, K);
        }
    }

    // -------------------------------------------------------------------------
    // Step 5: Build centred combined matrix full_mat (n_total × K)
    // -------------------------------------------------------------------------
    const uint32_t n_total = n_cells + n_sim;
    std::vector<float> full_mat(static_cast<size_t>(n_total) * K);
    std::copy(norm_real.begin(), norm_real.end(), full_mat.begin());
    std::copy(norm_sim.begin(),  norm_sim.end(),
              full_mat.begin() + static_cast<size_t>(n_cells) * K);

    // Centre columns (subtract per-column mean)
    std::vector<float> col_mean(K, 0.0f);
    for (uint32_t i = 0; i < n_total; ++i)
        for (int p = 0; p < K; ++p)
            col_mean[p] += full_mat[static_cast<size_t>(i) * K + p];
    const float inv_nt = 1.0f / static_cast<float>(n_total);
    for (int p = 0; p < K; ++p) col_mean[p] *= inv_nt;
    for (uint32_t i = 0; i < n_total; ++i)
        for (int p = 0; p < K; ++p)
            full_mat[static_cast<size_t>(i) * K + p] -= col_mean[p];

    // -------------------------------------------------------------------------
    // Step 5b: Power-iteration PCA — top N_PCS = 20 principal components
    //   C = full_mat^T full_mat (K × K covariance)
    //   Q = random (K × N_PCS), iterate: Q = normalise(C @ Q)
    //   PCA_coords = full_mat @ Q  (n_total × N_PCS)
    // -------------------------------------------------------------------------
    const int N_PCS  = 20;
    const int n_iter = 25;

    std::vector<float> C;
    doublet_detail::compute_cov(full_mat, static_cast<int>(n_total), K, C);

    // Initialise Q with reproducible Gaussian noise
    std::mt19937 rng_q(42u);
    std::normal_distribution<float> ndist(0.0f, 1.0f);
    std::vector<float> Q(static_cast<size_t>(K) * N_PCS);
    for (auto& v : Q) v = ndist(rng_q);
    doublet_detail::gram_schmidt(Q, K, N_PCS);

    std::vector<float> Qtmp;
    for (int it = 0; it < n_iter; ++it) {
        doublet_detail::matmul_sq(C, Q, Qtmp, K, N_PCS);
        std::swap(Q, Qtmp);
        doublet_detail::gram_schmidt(Q, K, N_PCS);
    }

    std::vector<float> pca_coords;
    doublet_detail::project(full_mat, Q, pca_coords,
                             static_cast<int>(n_total), K, N_PCS);

    // -------------------------------------------------------------------------
    // Step 6: k-NN in PCA space.
    // k = max(5, round(0.5 * sqrt(n_cells))) — scales with library size, like Scrublet.
    // Brute-force O(n_total × N_PCS) per cell: trivial for n_total < 40K.
    // -------------------------------------------------------------------------
    const int k_nn = std::max(5, static_cast<int>(std::round(
        0.5 * std::sqrt(static_cast<double>(n_cells)))));

    using Pair = std::pair<float, uint32_t>;  // (squared_dist, index)
    std::vector<Pair> pair_buf(n_total);

    // -------------------------------------------------------------------------
    // Step 7: kNN doublet fraction per real cell, then background-normalize.
    //
    // raw_score[i] = sim_count / k_nn           — fraction of kNN that are sims
    // background   = sim_frac = n_sim / n_total  — expected fraction for any cell
    //
    // Normalize to background-subtracted [0,1] score (v6):
    //   norm[i] = clamp((raw[i] - sim_frac) / (1 - sim_frac), 0, 1)
    // Random singlets (raw ≈ sim_frac)  → norm ≈ 0
    // Perfect doublets (raw ≈ 1.0)      → norm ≈ 1
    // -------------------------------------------------------------------------
    const float sim_frac = static_cast<float>(n_sim) / static_cast<float>(n_total);
    std::vector<float> norm_scores_v(n_cells, 0.0f);

    for (uint32_t i = 0; i < n_cells; ++i) {
        const float* pi = pca_coords.data() + static_cast<size_t>(i) * N_PCS;

        // Compute squared Euclidean distance to all n_total points
        for (uint32_t j = 0; j < n_total; ++j) {
            const float* pj = pca_coords.data() + static_cast<size_t>(j) * N_PCS;
            float d = 0.0f;
            for (int p = 0; p < N_PCS; ++p) {
                float diff = pi[p] - pj[p];
                d += diff * diff;
            }
            pair_buf[j] = { d, j };
        }
        pair_buf[i].first = 1e30f;  // exclude self

        // O(n_total) average: bring k nearest to the front
        std::nth_element(pair_buf.begin(), pair_buf.begin() + k_nn, pair_buf.end(),
                         [](const Pair& a, const Pair& b) { return a.first < b.first; });

        int sim_count = 0;
        for (int ki = 0; ki < k_nn; ++ki)
            if (pair_buf[ki].second >= n_cells) ++sim_count;

        const float raw  = static_cast<float>(sim_count) / static_cast<float>(k_nn);
        const float denom_norm = 1.0f - sim_frac;
        const float norm = (denom_norm > 1e-6f)
            ? std::max(0.0f, std::min(1.0f, (raw - sim_frac) / denom_norm))
            : 0.0f;
        norm_scores_v[i] = norm;
    }

    // -------------------------------------------------------------------------
    // Step 8: GMM threshold (v6 — GMM threshold)
    //
    // Fit a 2-component Gaussian mixture to the normalized scores.
    // If bimodal: threshold at the Bayes valley.
    // Else: threshold at the expected-rate quantile.
    // Always clamped to [0.15, 0.90].
    // -------------------------------------------------------------------------
    const double gmm_thresh = doublet_detail::gmm_threshold(norm_scores_v, expected_doublet_rate);
    for (uint32_t i = 0; i < n_cells; ++i) {
        results[i].score      = static_cast<double>(norm_scores_v[i]);
        results[i].is_doublet = (norm_scores_v[i] > static_cast<float>(gmm_thresh));
    }

    return results;
}

// -- Output ------------------------------------------------------------------

/// Write doublet_scores.tsv:
///   barcode  total_umis  doublet_score  is_doublet
inline void write_doublet_tsv(
    const std::string& path,
    const std::vector<DoubletResult>& results,
    const std::vector<uint32_t>& cell_indices,
    const std::vector<std::string>& barcodes,
    const SparseAccumulator<uint16_t>::CSCMatrix& counts)
{
    std::ofstream out(path);
    if (!out) {
        std::cerr << "[doublet] WARNING: cannot write " << path << "\n";
        return;
    }
    out << "barcode\ttotal_umis\tdoublet_score\tis_doublet\n";
    const uint32_t n = static_cast<uint32_t>(results.size());
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t j = cell_indices[i];
        uint32_t total = 0;
        for (int32_t k = counts.indptr[j]; k < counts.indptr[j + 1]; ++k)
            total += counts.data[k];
        out << barcodes[j] << '\t'
            << total << '\t'
            << results[i].score << '\t'
            << (results[i].is_doublet ? "True" : "False") << '\n';
    }
}

} // namespace singlet
