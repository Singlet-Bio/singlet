// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: ambient_correction.h
// N11: ambient RNA contamination estimation and correction.
//
// Two layers live here.
//
// (A) Gene-level cluster EM estimator + corrected matrix  (current; used by export.h)
//     correct_ambient_gene_level()  (core: decontx_ambient_core)
//
//   Model (DecontX, Yang et al. 2020, with the background fixed to the empty-droplet profile):
//     x_c ~ Multinomial(N_c, (1 - rho_c) * phi_{z(c)} + rho_c * a)
//   where a is the ambient (empty-droplet) gene profile, phi_k the native profile of cell cluster k and
//   rho_c ~ Beta(kappa * mu_k, kappa * (1 - mu_k)) the cell's contamination fraction. It runs on GENE counts
//   (exon, plus intron counts when present): at exon-interval level the profile is spread over ~7e5 features and
//   a per-feature subtraction rounds to nothing.
//
//   Steps (all deterministic: fixed seed, fixed block partitioning, no thread-count dependence):
//     1. ambient profile a from empty droplets (1..lower_umi UMIs, top n_ambient_max by UMIs, pseudocount 0.5);
//     2. clusters: log1p(CP10k) -> 2000 highest-variance genes -> 20 PCs (subspace iteration on the implicitly
//        centred sparse matrix) -> k-means++ / Lloyd (K = clamp(round(sqrt(cells / 20)), 2, 24));
//     3. per cluster, SoupX-style: the cluster's contamination c_k is the Poisson ratio estimate over the genes the
//        cluster does not express above the ambient expectation (iterated trimmed MLE) and that are markers of
//        another cluster (so weakly expressed native genes cannot masquerade as soup); the native profile is
//        phi_k = (n_k - c_k N_k a) for the genes clearly above ambient (+ z sd), 0 elsewhere, normalised;
//        c_k is then shrunk toward the sample rho c_s (cell-weighted median of the c_k) as a Gamma-Poisson posterior with
//        cluster_prior_reads pseudo off-gene reads: a cluster keeps its own c_k in proportion to its off-gene information
//        (v2; the raw c_k is unstable in tissue, docs/ambient-calibration.md), and the native profiles use the shrunk c_k;
//     4. per cell: rho_c from the cell's reads in the genes its cluster does not express (all ambient there:
//        X_O ~ Poisson(rho N A_O), Gamma prior with mean c_k and off_prior_reads pseudo-reads, so the cell's own weight is
//        its expected off-gene UMIs N A_O c_k over that + off_prior_reads), or, when that set is too light, the mixture EM
//        with the Beta prior rho <- (E[ambient UMIs] + kappa c_k) / (N + kappa);
//     5. outlier round: cells whose rho is far above their cluster's (doublets, damaged cells) are left out of the
//        cluster statistics and the clusters are refitted; they are still corrected;
//     6. corrected counts: x_g * (1 - r_g) with r_g = rho a_g / ((1 - rho) phi_g + rho a_g), rounded to integers by
//        the largest-remainder method so every cell keeps exactly round(sum_g x_g (1 - r_g)) UMIs and no gene
//        exceeds its observed count.
//   Only called cells get a non-empty column; the raw matrix is never modified.
//
// (B) Legacy exon-level depth estimator (estimate_contamination / correct_counts), kept for the older tests and
//     as a fallback: rho_c = S / N_c with S the mean UMIs per empty droplet. It carries no gene information.
//
// References: Young & Bhatt 2020 (SoupX, GigaScience 9(12)); Yang et al. 2020 (DecontX, Genome Biol 21:57).
// Outputs written by export.h: ambient_profile.tsv, ambient_contamination.tsv/.json, ambient_corrected_counts.

#include "sparse_accumulator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace singlet {

using CSCu16 = SparseAccumulator<uint16_t>::CSCMatrix;
using CSCu32 = SparseAccumulator<uint32_t>::CSCMatrix;

// ── Data structures ──────────────────────────────────────────────────────────

struct AmbientProfile {
    std::vector<double> gene_fractions;    ///< per-gene ambient fraction (sums to 1)
    double   total_ambient_umi    = 0.0;   ///< total UMI across ambient barcodes
    uint32_t n_ambient_barcodes   = 0;     ///< number of empty droplets used
};

struct CellContamination {
    double   rho          = 0.0;  ///< estimated contamination fraction [0, 0.95]
    double   log_likelihood = 0.0; ///< reserved
    uint32_t n_genes_used = 0;    ///< genes with a non-zero count in the cell (legacy estimator: 0)
    // Gene-level estimator extras (appended so older aggregate users keep compiling)
    uint32_t cluster      = 0;    ///< cluster of the cell
    uint32_t n_umi        = 0;    ///< observed UMIs (gene level)
    uint32_t n_removed    = 0;    ///< integer ambient UMIs removed in the corrected matrix
    double   rho_total    = 0.0;  ///< ambient fraction of ALL the cell's UMIs (== rho unless the split exon/intron API ran)
    double   off_reads    = 0.0;  ///< the cell's UMIs in its cluster's "not expressed" genes (off-gene estimator; else 0)
    double   off_exposure = 0.0;  ///< N x ambient mass of those genes: expected off-gene UMIs at rho = 1
    double   prior_rho    = 0.0;  ///< prior mean of rho (the cell's cluster c_k after cluster -> sample shrinkage)
};

// ── Ambient profile estimation ───────────────────────────────────────────────

/// Estimate ambient RNA profile from empty droplets (barcodes <= lower_umi).
///
/// @param counts        Full CSC feature x barcode matrix (all barcodes); uint16 or uint32 data.
/// @param bc_totals     Per-barcode total UMI (size = counts.ncols).
/// @param lower_umi     Threshold: barcodes with count <= this define ambient pool.
/// @param n_ambient_max Maximum empty droplets to use (default 50000; highest-count ones).
/// @return              AmbientProfile with per-gene fractions summing to 1.
template <class CSC>
inline AmbientProfile estimate_ambient(
    const CSC&                   counts,
    const std::vector<uint64_t>& bc_totals,
    uint64_t                     lower_umi,
    int                          n_ambient_max = 50000)
{
    AmbientProfile profile;
    const uint32_t n_genes    = counts.nrows;
    const uint32_t n_barcodes = counts.ncols;
    if (n_barcodes == 0 || n_genes == 0) return profile;

    profile.gene_fractions.assign(n_genes, 0.0);

    // Non-zero empty barcodes, descending by count (ties by index: deterministic), capped.
    std::vector<uint32_t> empty_bcs;
    empty_bcs.reserve(n_barcodes);
    for (uint32_t b = 0; b < n_barcodes; ++b) {
        if (bc_totals[b] > 0 && bc_totals[b] <= lower_umi)
            empty_bcs.push_back(b);
    }
    std::sort(empty_bcs.begin(), empty_bcs.end(),
              [&](uint32_t a, uint32_t b2) {
                  return bc_totals[a] != bc_totals[b2] ? bc_totals[a] > bc_totals[b2] : a < b2;
              });
    if (static_cast<int>(empty_bcs.size()) > n_ambient_max)
        empty_bcs.resize(static_cast<size_t>(n_ambient_max));

    for (uint32_t b : empty_bcs) {
        profile.total_ambient_umi += static_cast<double>(bc_totals[b]);
        const int32_t start = counts.indptr[b];
        const int32_t end   = counts.indptr[b + 1];
        for (int32_t k = start; k < end; ++k)
            profile.gene_fractions[counts.indices[k]] +=
                static_cast<double>(counts.data[k]);
    }
    profile.n_ambient_barcodes = static_cast<uint32_t>(empty_bcs.size());
    if (profile.total_ambient_umi < 1.0) return profile;

    // Normalize with pseudocount 0.5 per gene to avoid zero fractions.
    double sum = 0.0;
    for (uint32_t g = 0; g < n_genes; ++g) {
        profile.gene_fractions[g] += 0.5;
        sum += profile.gene_fractions[g];
    }
    const double inv_sum = 1.0 / sum;
    for (uint32_t g = 0; g < n_genes; ++g)
        profile.gene_fractions[g] *= inv_sum;

    return profile;
}

// ══════════════════════════════════════════════════════════════════════════════
// (A) Gene-level cluster EM estimator
// ══════════════════════════════════════════════════════════════════════════════

struct AmbientDecontConfig {
    uint64_t lower_umi        = 100;    ///< barcodes with 1..lower_umi UMIs form the ambient pool
    int      n_ambient_max    = 50000;  ///< pool size cap (highest-count empty droplets)
    uint32_t n_clusters       = 0;      ///< 0 = auto: clamp(round(sqrt(n_cells / 20)), 2, 24); 1 when < 40 cells
    uint32_t n_hvg            = 2000;   ///< genes used for clustering
    uint32_t n_pcs            = 20;     ///< principal components used for clustering
    uint32_t pca_power_iters  = 5;
    uint32_t kmeans_iters     = 30;
    uint64_t seed             = 42;
    double   rho_min          = 1e-6;
    double   rho_max          = 0.5;
    double   prior_strength   = 20.0;   ///< kappa: Beta prior pseudo-UMIs, prior mean = the cluster's c_k
    double   offgene_z        = 3.0;    ///< Poisson tolerance (sd) for "not expressed above ambient" genes
    double   phi_pseudocount  = 1e-4;   ///< per-gene pseudocount of the native profile
    bool     equal_cell_weight = false; ///< cluster profiles average cells with equal weight instead of by UMIs
    bool     outlier_round = true;      ///< refit the clusters without cells whose rho is far above their cluster's
    bool     use_off_estimator = true;  ///< rho from the cluster's marker-elsewhere "not expressed" genes (else full EM)
    double   marker_fold   = 1.25;      ///< a gene is a marker elsewhere if another cluster has it >= fold x ambient (+ z sd)
    double   min_off_mass  = 0.01;      ///< min ambient mass of the off-gene set for the off-gene estimator
    double   off_prior_reads = 5.0;     ///< Gamma-prior pseudo-reads of the off-gene estimator (mean = c_k)
    double   outlier_z     = 5.0;       ///< robust z (MAD) of that cut; also requires rho > 2x the cluster median
    std::vector<char> estimate_features;///< optional: features allowed to inform c_k / rho (empty = all); clustering, the
                                        ///< native profiles and the correction always use every feature
    double   low_depth_frac = 0.1;      ///< cells below this fraction of the median depth (the ambient-like barcodes a
                                        ///< permissive cell caller keeps) are kept out of clustering and cluster statistics
    double   cluster_prior_cells = 0.0; ///< pseudo-cells shrinking each cluster's c_k toward the sample rho (cell-weighted
                                        ///< median of the c_k): c_k <- (n_k c_k + n0 c_s) / (n_k + n0); 0 = no shrinkage
    double   cluster_prior_reads = 1e4; ///< Gamma prior of each cluster's c_k (mean c_s) in pseudo off-gene reads: weight
                                        ///< of the cluster's own data = its off-gene reads / (those + this); 0 = no shrinkage
                                        ///< (chosen on Kang 2.1 genotype truth + barnyard, docs/ambient-calibration.md)
    uint32_t min_cluster_cells = 0;     ///< k-means clusters with fewer core cells are dissolved into the nearest remaining
                                        ///< centroid (their native profile is borrowed from the neighbour); 0 = keep all
    uint32_t cell_iters       = 30;     ///< per-cell EM iterations
    uint32_t threads          = 0;      ///< 0 = hardware_concurrency (results do not depend on it)
};

struct AmbientCorrectionResult {
    AmbientProfile                 profile;         ///< gene-level ambient profile
    std::vector<CellContamination> contam;          ///< one per called cell, in cell_indices order
    CSCu32                         corrected;       ///< genes x all barcodes; non-empty columns only for called cells
    std::vector<double>            cluster_rho;     ///< c_k per cluster (after cluster -> sample shrinkage)
    std::vector<double>            cluster_rho_raw; ///< c_k per cluster before shrinkage
    std::vector<uint32_t>          cluster_cells;   ///< core cells per cluster that entered the cluster statistics
    std::vector<double>            cluster_off_mass;///< ambient mass of each cluster's "not expressed" gene set
    double   sample_rho            = 0.0;           ///< shrinkage target: cell-weighted median of the raw c_k
    uint32_t n_cells_input         = 0;             ///< cell_indices.size() as passed (before the guard)
    uint32_t n_cells_dropped       = 0;             ///< duplicate or out-of-range cell indices dropped by the guard
    uint32_t n_uncalled_above_pool = 0;             ///< non-cell barcodes above lower_umi: in neither the pool nor the fit
    uint32_t n_clusters            = 0;
    double   exon_profile_share    = 0.0;           ///< split API: share of the soup that is exon-derived
    uint32_t n_low_depth_cells     = 0;             ///< cells below low_depth_frac x median depth (kept out of the fits)
    uint32_t n_outlier_cells       = 0;             ///< cells left out of the cluster statistics (doublet-like rho)
    uint32_t n_genes_detected_pool = 0;             ///< genes with >= 1 UMI in the ambient pool
    double   mean_rho              = 0.0;
    double   median_rho            = 0.0;
    double   umi_weighted_rho      = 0.0;           ///< removed UMIs / observed UMIs over called cells
    uint64_t umis_observed         = 0;
    uint64_t umis_removed          = 0;
    double   seconds               = 0.0;
    bool     ok                    = false;         ///< false: no ambient pool / no cells (corrected == identity)
    std::string method = "decontx_cluster_offgene_v2";   ///< v2: cluster -> sample shrinkage (v1: cluster_prior_reads = 0)
};

namespace ambient_detail {

struct SplitMix64 {
    uint64_t s;
    explicit SplitMix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    double uniform() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }
};

/// Dynamic-chunk parallel loop. f(i) must write only slot i, so the result does not depend on scheduling.
template <class F>
inline void parallel_for(size_t n, unsigned threads, F&& f) {
    if (n == 0) return;
    if (threads <= 1 || n == 1) { for (size_t i = 0; i < n; ++i) f(i); return; }
    threads = static_cast<unsigned>(std::min<size_t>(threads, n));
    const size_t chunk = std::max<size_t>(1, n / (static_cast<size_t>(threads) * 8));
    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (;;) {
            size_t s = next.fetch_add(chunk);
            if (s >= n) break;
            size_t e = std::min(n, s + chunk);
            for (size_t i = s; i < e; ++i) f(i);
        }
    };
    std::vector<std::thread> pool;
    for (unsigned t = 1; t < threads; ++t) pool.emplace_back(work);
    work();
    for (auto& th : pool) th.join();
}

/// Called cells only: feature x cell, CSC with 64-bit offsets (counts as double-exact integers in uint32).
struct CellMat {
    uint32_t n_genes = 0, n_cells = 0;
    std::vector<size_t>   ptr;   // n_cells + 1
    std::vector<uint32_t> gene;
    std::vector<uint32_t> val;
    std::vector<double>   tot;   // per-cell UMIs
};

/// Modified Gram-Schmidt on the columns of a row-major rows x k matrix.
inline void orthonormalize(std::vector<double>& M, size_t rows, size_t k) {
    for (size_t c = 0; c < k; ++c) {
        for (size_t p = 0; p < c; ++p) {
            double d = 0.0;
            for (size_t r = 0; r < rows; ++r) d += M[r * k + p] * M[r * k + c];
            for (size_t r = 0; r < rows; ++r) M[r * k + c] -= d * M[r * k + p];
        }
        double nn = 0.0;
        for (size_t r = 0; r < rows; ++r) nn += M[r * k + c] * M[r * k + c];
        nn = std::sqrt(nn);
        const double inv = nn > 1e-12 ? 1.0 / nn : 0.0;
        for (size_t r = 0; r < rows; ++r) M[r * k + c] *= inv;
    }
}

/// Deterministic quick clustering of the cells: log1p(CP10k), top-variance genes, PCA (subspace iteration on the
/// implicitly centred sparse matrix), k-means++ and Lloyd. Returns cluster ids in [0, K); sets K.
inline std::vector<uint32_t> quick_cluster(const CellMat& X, const AmbientDecontConfig& cfg,
                                           unsigned threads, uint32_t& K_out)
{
    const size_t n = X.n_cells, G = X.n_genes;
    uint32_t K = cfg.n_clusters;
    if (K == 0) {
        if (n < 40) K = 1;
        else K = static_cast<uint32_t>(std::clamp<long>(std::lround(std::sqrt(static_cast<double>(n) / 20.0)), 2, 24));
    }
    if (K > n) K = static_cast<uint32_t>(n);
    K_out = std::max<uint32_t>(K, 1);
    std::vector<uint32_t> cl(n, 0);
    if (K <= 1 || n < 4 || G == 0) { K_out = 1; return cl; }

    // log-normalised values per nnz, gene mean / variance (serial: fixed summation order)
    std::vector<double> s1(G, 0.0), s2(G, 0.0);
    std::vector<uint32_t> ndet(G, 0);
    for (size_t j = 0; j < n; ++j) {
        const double scale = X.tot[j] > 0 ? 1e4 / X.tot[j] : 0.0;
        for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) {
            const double v = std::log1p(static_cast<double>(X.val[e]) * scale);
            s1[X.gene[e]] += v; s2[X.gene[e]] += v * v; ++ndet[X.gene[e]];
        }
    }
    std::vector<uint32_t> cand;
    cand.reserve(G);
    std::vector<double> var(G, 0.0), mean(G, 0.0);
    for (size_t g = 0; g < G; ++g) {
        mean[g] = s1[g] / static_cast<double>(n);
        var[g]  = s2[g] / static_cast<double>(n) - mean[g] * mean[g];
        if (ndet[g] >= 3) cand.push_back(static_cast<uint32_t>(g));
    }
    const size_t H = std::min<size_t>(cfg.n_hvg, cand.size());
    if (H < 2) { K_out = 1; return cl; }
    std::stable_sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return var[a] > var[b]; });
    cand.resize(H);
    std::vector<int32_t> hmap(G, -1);
    for (size_t h = 0; h < H; ++h) hmap[cand[h]] = static_cast<int32_t>(h);

    // sparse HVG matrix S (cells x entries), centred implicitly by mu
    std::vector<size_t> hptr(n + 1, 0);
    std::vector<uint32_t> hidx; std::vector<float> hval;
    for (size_t j = 0; j < n; ++j) {
        const double scale = X.tot[j] > 0 ? 1e4 / X.tot[j] : 0.0;
        for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) {
            const int32_t h = hmap[X.gene[e]];
            if (h < 0) continue;
            hidx.push_back(static_cast<uint32_t>(h));
            hval.push_back(static_cast<float>(std::log1p(static_cast<double>(X.val[e]) * scale)));
        }
        hptr[j + 1] = hidx.size();
    }
    std::vector<double> mu(H);
    for (size_t h = 0; h < H; ++h) mu[h] = mean[cand[h]];

    const size_t kp = std::min<size_t>({cfg.n_pcs, H, n > 1 ? n - 1 : 1});
    SplitMix64 rng(cfg.seed);
    std::vector<double> Q(n * kp);
    for (auto& q : Q) q = rng.uniform() - 0.5;

    constexpr size_t NB = 32;   // fixed scatter blocks -> thread-count-independent sums
    std::vector<std::vector<double>> part(NB);
    auto Y_times = [&](const std::vector<double>& W, std::vector<double>& Z) {   // Z (H x kp) = Y W (n x kp)
        std::vector<double> wsum(kp, 0.0);
        for (size_t j = 0; j < n; ++j) for (size_t c = 0; c < kp; ++c) wsum[c] += W[j * kp + c];
        parallel_for(NB, threads, [&](size_t b) {
            auto& P = part[b];
            P.assign(H * kp, 0.0);
            const size_t j0 = n * b / NB, j1 = n * (b + 1) / NB;
            for (size_t j = j0; j < j1; ++j)
                for (size_t e = hptr[j]; e < hptr[j + 1]; ++e) {
                    const double v = hval[e];
                    double* z = &P[static_cast<size_t>(hidx[e]) * kp];
                    const double* w = &W[j * kp];
                    for (size_t c = 0; c < kp; ++c) z[c] += v * w[c];
                }
        });
        Z.assign(H * kp, 0.0);
        for (size_t b = 0; b < NB; ++b)
            for (size_t i = 0; i < H * kp; ++i) Z[i] += part[b][i];
        for (size_t h = 0; h < H; ++h)
            for (size_t c = 0; c < kp; ++c) Z[h * kp + c] -= mu[h] * wsum[c];
    };
    auto Yt_times = [&](const std::vector<double>& U, std::vector<double>& W) {   // W (n x kp) = Y^T U
        std::vector<double> muU(kp, 0.0);
        for (size_t h = 0; h < H; ++h) for (size_t c = 0; c < kp; ++c) muU[c] += mu[h] * U[h * kp + c];
        W.assign(n * kp, 0.0);
        parallel_for(n, threads, [&](size_t j) {
            double* w = &W[j * kp];
            for (size_t c = 0; c < kp; ++c) w[c] = -muU[c];
            for (size_t e = hptr[j]; e < hptr[j + 1]; ++e) {
                const double v = hval[e];
                const double* u = &U[static_cast<size_t>(hidx[e]) * kp];
                for (size_t c = 0; c < kp; ++c) w[c] += v * u[c];
            }
        });
    };
    std::vector<double> Z, E;
    for (uint32_t it = 0; it < cfg.pca_power_iters; ++it) {
        Y_times(Q, Z); orthonormalize(Z, H, kp);
        Yt_times(Z, Q); orthonormalize(Q, n, kp);
    }
    Y_times(Q, Z); orthonormalize(Z, H, kp);
    Yt_times(Z, E);   // n x kp embedding (rotation of the PC scores: distances are what k-means uses)

    // k-means++ and Lloyd
    std::vector<double> C(static_cast<size_t>(K) * kp);
    std::vector<double> d2(n, std::numeric_limits<double>::max());
    auto dist2 = [&](size_t j, size_t k) {
        double s = 0.0;
        for (size_t c = 0; c < kp; ++c) { const double d = E[j * kp + c] - C[k * kp + c]; s += d * d; }
        return s;
    };
    size_t first = std::min<size_t>(n - 1, static_cast<size_t>(rng.uniform() * static_cast<double>(n)));
    for (size_t c = 0; c < kp; ++c) C[c] = E[first * kp + c];
    for (uint32_t k = 1; k < K; ++k) {
        double tot = 0.0;
        for (size_t j = 0; j < n; ++j) { d2[j] = std::min(d2[j], dist2(j, k - 1)); tot += d2[j]; }
        double target = rng.uniform() * tot, acc = 0.0;
        size_t pick = n - 1;
        for (size_t j = 0; j < n; ++j) { acc += d2[j]; if (acc >= target && d2[j] > 0) { pick = j; break; } }
        for (size_t c = 0; c < kp; ++c) C[static_cast<size_t>(k) * kp + c] = E[pick * kp + c];
    }
    std::vector<uint32_t> prev(n, std::numeric_limits<uint32_t>::max());
    for (uint32_t it = 0; it < cfg.kmeans_iters; ++it) {
        parallel_for(n, threads, [&](size_t j) {
            double best = std::numeric_limits<double>::max(); uint32_t bk = 0;
            for (uint32_t k = 0; k < K; ++k) { const double d = dist2(j, k); if (d < best) { best = d; bk = k; } }
            cl[j] = bk;
        });
        if (cl == prev) break;
        prev = cl;
        std::vector<double> sum(static_cast<size_t>(K) * kp, 0.0);
        std::vector<size_t> cnt(K, 0);
        for (size_t j = 0; j < n; ++j) {
            ++cnt[cl[j]];
            for (size_t c = 0; c < kp; ++c) sum[static_cast<size_t>(cl[j]) * kp + c] += E[j * kp + c];
        }
        for (uint32_t k = 0; k < K; ++k)
            if (cnt[k] > 0)
                for (size_t c = 0; c < kp; ++c) C[static_cast<size_t>(k) * kp + c] = sum[static_cast<size_t>(k) * kp + c] / static_cast<double>(cnt[k]);
    }
    // Small clusters: a native profile from a few dozen cells cannot tell weakly expressed native genes from soup (their
    // c_k comes out high), so clusters below min_cluster_cells are dissolved, smallest first, into the nearest remaining
    // centroid; the cells then borrow that neighbour's native profile and c_k.
    if (cfg.min_cluster_cells > 0) {
        std::vector<char> alive(K, 1);
        for (;;) {
            std::vector<size_t> cnt(K, 0);
            for (size_t j = 0; j < n; ++j) ++cnt[cl[j]];
            uint32_t n_alive = 0, s = K;
            for (uint32_t k = 0; k < K; ++k) {
                if (cnt[k] == 0) alive[k] = 0;
                if (!alive[k]) continue;
                ++n_alive;
                if (cnt[k] < cfg.min_cluster_cells && (s == K || cnt[k] < cnt[s])) s = k;
            }
            if (s == K || n_alive <= 2) break;   // keep >= 2 clusters: the off-gene estimator needs markers elsewhere
            alive[s] = 0;
            for (size_t j = 0; j < n; ++j) {
                if (cl[j] != s) continue;
                double best = std::numeric_limits<double>::max(); uint32_t bk = cl[j];
                for (uint32_t k = 0; k < K; ++k) {
                    if (!alive[k]) continue;
                    const double d = dist2(j, k);
                    if (d < best) { best = d; bk = k; }
                }
                cl[j] = bk;
            }
        }
    }
    // relabel to drop empty clusters
    std::vector<int32_t> remap(K, -1);
    uint32_t nk = 0;
    for (size_t j = 0; j < n; ++j) if (remap[cl[j]] < 0) remap[cl[j]] = -2;
    for (uint32_t k = 0; k < K; ++k) if (remap[k] == -2) remap[k] = static_cast<int32_t>(nk++);
    for (size_t j = 0; j < n; ++j) cl[j] = static_cast<uint32_t>(remap[cl[j]]);
    K_out = nk;
    return cl;
}

/// Cluster contamination c_k: Poisson ratio estimate over the genes the cluster does not express above the ambient
/// expectation (trimmed MLE iterated from c = rho_max). n = cluster gene totals (cell-weighted), Nk its UMIs, infl the
/// variance inflation of the weighting (1 for plain sums).
inline double cluster_contamination(const std::vector<double>& n, double Nk, double infl, const std::vector<double>& a,
                                    const AmbientDecontConfig& cfg, const std::vector<char>* mask = nullptr,
                                    double* exposure = nullptr)
{
    const size_t G = a.size();
    if (exposure) *exposure = 0.0;
    if (Nk <= 0.0) return 0.0;
    // Iterate from above (c = rho_max): the first self-consistent value below the start. Iterating from a low
    // start gets stuck at a downward-biased fixed point (the upper cut then discards the informative genes).
    double c = cfg.rho_max;
    for (int it = 0; it < 200; ++it) {
        double num = 0.0, den = 0.0;
        for (size_t g = 0; g < G; ++g) {
            if (mask && !(*mask)[g]) continue;
            const double e1 = Nk * a[g];
            const double thr = c * e1;
            if (n[g] <= thr + cfg.offgene_z * std::sqrt(thr * infl) + 1.0) { num += n[g]; den += e1; }
        }
        const double cn = den > 0 ? std::min(num / den, cfg.rho_max) : 0.0;
        const bool done = std::fabs(cn - c) <= 1e-4 * std::max(c, 1e-5);
        c = std::max(cn, 1e-6);
        if (exposure) *exposure = den;
        if (done) break;
    }
    return std::min(c, cfg.rho_max);
}

}  // namespace ambient_detail

/// Core: estimate rho and produce integer corrected counts for the called cells.
/// @param X   called cells, gene x cell (ambient_detail::CellMat)
/// @param a   ambient gene fractions (sum 1, all > 0)
inline void decontx_ambient_core(const ambient_detail::CellMat& X, const std::vector<double>& a,
                                 const AmbientDecontConfig& cfg, AmbientCorrectionResult& R,
                                 std::vector<uint32_t>& corrected_vals)
{
    using namespace ambient_detail;
    const size_t n = X.n_cells, G = X.n_genes;
    unsigned thr = cfg.threads ? cfg.threads : std::max(1u, std::thread::hardware_concurrency());

    // Core / low-depth split: barcodes far below the typical cell depth look like soup and would form their own
    // "native" profile, so clusters and cluster statistics come from the core cells only; low-depth cells are attached
    // to the most likely cluster afterwards and get their own rho.
    std::vector<char> low(n, 0);
    size_t n_low = 0;
    if (cfg.low_depth_frac > 0.0 && n >= 80) {
        std::vector<double> t(X.tot);
        std::nth_element(t.begin(), t.begin() + static_cast<std::ptrdiff_t>(n / 2), t.end());
        const double floor_depth = cfg.low_depth_frac * t[n / 2];
        for (size_t j = 0; j < n; ++j) if (X.tot[j] < floor_depth) { low[j] = 1; ++n_low; }
        if (n - n_low < 40) { std::fill(low.begin(), low.end(), 0); n_low = 0; }
    }
    R.n_low_depth_cells = static_cast<uint32_t>(n_low);

    uint32_t K = 1;
    std::vector<uint32_t> cl(n, 0);
    if (n_low == 0) {
        cl = quick_cluster(X, cfg, thr, K);
    } else {
        CellMat Xc;
        Xc.n_genes = X.n_genes;
        std::vector<uint32_t> core;
        Xc.ptr.push_back(0);
        for (size_t j = 0; j < n; ++j) {
            if (low[j]) continue;
            core.push_back(static_cast<uint32_t>(j));
            Xc.gene.insert(Xc.gene.end(), X.gene.begin() + static_cast<std::ptrdiff_t>(X.ptr[j]), X.gene.begin() + static_cast<std::ptrdiff_t>(X.ptr[j + 1]));
            Xc.val.insert(Xc.val.end(), X.val.begin() + static_cast<std::ptrdiff_t>(X.ptr[j]), X.val.begin() + static_cast<std::ptrdiff_t>(X.ptr[j + 1]));
            Xc.ptr.push_back(Xc.gene.size());
            Xc.tot.push_back(X.tot[j]);
        }
        Xc.n_cells = static_cast<uint32_t>(core.size());
        const std::vector<uint32_t> clc = quick_cluster(Xc, cfg, thr, K);
        for (size_t i = 0; i < core.size(); ++i) cl[core[i]] = clc[i];
    }
    R.n_clusters = K;

    // per-cluster membership of the core cells (ascending order -> fixed summation order)
    std::vector<std::vector<uint32_t>> members(K);
    for (size_t j = 0; j < n; ++j) if (!low[j]) members[cl[j]].push_back(static_cast<uint32_t>(j));

    // cluster contamination c_k and native profile phi (gene-major: phi[g * K + k]); cells with use[j] == 0 are
    // left out of the cluster statistics (outlier round) but are still corrected with them.
    std::vector<double> ck(K, 0.0), ck_raw(K, 0.0), ncells_k(K, 0.0), phi(G * K, 0.0), rho(n, 0.0), offA(K, 0.0);
    std::vector<double> xo_v(n, 0.0), ex_v(n, 0.0), expo_k(K, 0.0);
    double c_sample = 0.0;
    std::vector<char> use(n, 1), offmask(G * K, 0);
    for (size_t j = 0; j < n; ++j) if (low[j]) use[j] = 0;   // offmask[g * K + k]: gene g is "not expressed" in cluster k
    std::vector<std::vector<double>> nkv(K);
    std::vector<double> Nkv(K, 0.0), inflv(K, 1.0);
    auto fit_clusters = [&]() {
        // A. cell-weighted gene totals per cluster
        parallel_for(K, thr, [&](size_t k) {
            std::vector<uint32_t> mem;
            for (uint32_t j : members[k]) if (use[j]) mem.push_back(j);
            if (mem.size() < 3) mem = members[k];   // too few left: use everyone
            auto& nk = nkv[k];
            nk.assign(G, 0.0);
            double Nk = 0.0, infl = 1.0;
            for (uint32_t j : mem) Nk += X.tot[j];
            if (cfg.equal_cell_weight && !mem.empty() && Nk > 0) {
                // w_j = mean depth / N_j: every cell contributes the same total mass; variance inflation = mean(w_j)
                const double Nbar = Nk / static_cast<double>(mem.size());
                double sw = 0.0;
                for (uint32_t j : mem) {
                    if (X.tot[j] <= 0) continue;
                    const double w = Nbar / X.tot[j];
                    sw += w;
                    for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) nk[X.gene[e]] += w * X.val[e];
                }
                infl = std::max(1.0, sw / static_cast<double>(mem.size()));
            } else {
                for (uint32_t j : mem)
                    for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) nk[X.gene[e]] += X.val[e];
            }
            Nkv[k] = Nk; inflv[k] = infl;
        });
        // B. marker genes: a gene clearly above its ambient share in some cluster
        std::vector<char> strong(G * K, 0);
        std::vector<uint16_t> n_strong(G, 0);
        for (uint32_t k = 0; k < K; ++k)
            for (size_t g = 0; g < G; ++g) {
                const double e = cfg.marker_fold * Nkv[k] * a[g];
                if (nkv[k][g] >= e + cfg.offgene_z * std::sqrt(e * inflv[k]) + 1.0) { strong[g * K + k] = 1; ++n_strong[g]; }
            }
        // C1. raw c_k over the marker-elsewhere genes
        std::vector<std::vector<char>> masks(K);
        std::vector<char> have_maskv(K, 0);
        parallel_for(K, thr, [&](size_t k) {
            auto& nk = nkv[k];
            const double Nk = Nkv[k], infl = inflv[k];
            auto& mask = masks[k];
            mask.assign(G, 0);
            double Aeff = 0.0;
            const bool est_all = cfg.estimate_features.size() != G;
            for (size_t g = 0; g < G; ++g)
                if (n_strong[g] - strong[g * K + k] > 0 && (est_all || cfg.estimate_features[g])) { mask[g] = 1; Aeff += a[g]; }
            const bool have_mask = K > 1 && cfg.use_off_estimator && Aeff >= cfg.min_off_mass;
            have_maskv[k] = have_mask ? 1 : 0;
            ck_raw[k] = cluster_contamination(nk, Nk, infl, a, cfg,
                                              have_mask ? &mask : (est_all ? nullptr : &cfg.estimate_features), &expo_k[k]);
        });
        // C2. cluster -> sample shrinkage. The raw c_k is unstable in tissue: the trimmed fixed point jumps between
        // solutions (Kang 2.1: 0.6% to 50% across clusters whose genotype truth is 1.0-3.6%), and a cluster's c_k is the
        // prior mean of all its cells. Sample rho c_s = cell-weighted median of the raw c_k; each c_k becomes a
        // Gamma-Poisson posterior with prior mean c_s and cluster_prior_reads pseudo off-gene reads, so a cluster keeps its
        // own value in proportion to its off-gene information (exposure x c_k reads: ~1e2-1e3 in a PBMC cluster, ~1e4-1e5 in
        // a barnyard cluster); cluster_prior_cells adds a pseudo-cell average. docs/ambient-calibration.md has the sweep.
        for (uint32_t k = 0; k < K; ++k) {
            double nc = 0.0;
            for (uint32_t j : members[k]) if (use[j]) nc += 1.0;
            ncells_k[k] = nc;
        }
        {
            std::vector<std::pair<double, double>> cw;   // (c_k, cells)
            double wt = 0.0;
            for (uint32_t k = 0; k < K; ++k)
                if (ncells_k[k] > 0) { cw.emplace_back(ck_raw[k], ncells_k[k]); wt += ncells_k[k]; }
            std::sort(cw.begin(), cw.end());
            double acc = 0.0, cs = cw.empty() ? 0.0 : cw.back().first;
            for (const auto& p : cw) { acc += p.second; if (acc >= 0.5 * wt) { cs = p.first; break; } }
            c_sample = std::max(cs, cfg.rho_min);
        }
        for (uint32_t k = 0; k < K; ++k) {
            double c = ck_raw[k];
            if (cfg.cluster_prior_reads > 0.0)
                c = (c * expo_k[k] + cfg.cluster_prior_reads) / (expo_k[k] + cfg.cluster_prior_reads / c_sample);
            if (cfg.cluster_prior_cells > 0.0)
                c = (ncells_k[k] * c + cfg.cluster_prior_cells * c_sample) / (ncells_k[k] + cfg.cluster_prior_cells);
            ck[k] = std::min(std::max(c, cfg.rho_min), cfg.rho_max);
        }
        // C3. native profile and off-gene set given the shrunk c_k
        parallel_for(K, thr, [&](size_t k) {
            const auto& nk = nkv[k];
            const double Nk = Nkv[k], infl = inflv[k];
            const auto& mask = masks[k];
            const bool have_mask = have_maskv[k] != 0;
            const double c = ck[k];
            double tot = 0.0, A = 0.0;
            for (size_t g = 0; g < G; ++g) {
                // native evidence only above the ambient expectation + z sd (same cut as the "not expressed" genes of
                // cluster_contamination): the positive part of Poisson noise on off genes would otherwise leave a
                // native profile that keeps ~half of the ambient counts of every gene the cluster does not express.
                const double e = c * Nk * a[g];
                const bool native = nk[g] > e + cfg.offgene_z * std::sqrt(e * infl) + 1.0;
                const double nat = (native ? nk[g] - e : 0.0) + cfg.phi_pseudocount;
                phi[g * K + k] = nat; tot += nat;
                const bool off = have_mask && mask[g] && !native;
                offmask[g * K + k] = off ? 1 : 0;
                if (off) A += a[g];
            }
            offA[k] = A;
            for (size_t g = 0; g < G; ++g) phi[g * K + k] /= tot;
        });
    };
    // per-cell rho given its cluster's phi and the ambient profile. Off-gene estimator: the cell's reads in the genes
    // its cluster does not express are all ambient, X_O ~ Poisson(rho N A_O); Gamma prior (mean c_k) gives
    // rho = (X_O + s) / (N A_O + s / c_k). When the off set is too light, the mixture EM with the Beta prior.
    auto fit_cells = [&]() {
        parallel_for(n, thr, [&](size_t j) {
            const uint32_t k = cl[j];
            const double N = X.tot[j];
            const double mu = std::max(ck[k], cfg.rho_min);
            double r = std::min(std::max(mu, cfg.rho_min), cfg.rho_max);
            if (N > 0 && offA[k] >= cfg.min_off_mass) {
                double xo = 0.0;
                for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e)
                    if (offmask[static_cast<size_t>(X.gene[e]) * K + k]) xo += X.val[e];
                r = (xo + cfg.off_prior_reads) / (N * offA[k] + cfg.off_prior_reads / mu);
                r = std::min(std::max(r, cfg.rho_min), cfg.rho_max);
                xo_v[j] = xo; ex_v[j] = N * offA[k];
            } else if (N > 0) {
                for (uint32_t it = 0; it < cfg.cell_iters; ++it) {
                    double S = 0.0;
                    for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) {
                        const uint32_t g = X.gene[e];
                        const double pa = r * a[g], pn = (1.0 - r) * phi[static_cast<size_t>(g) * K + k];
                        S += X.val[e] * (pa / (pa + pn));
                    }
                    const double rn = std::min(std::max((S + cfg.prior_strength * mu) / (N + cfg.prior_strength), cfg.rho_min), cfg.rho_max);
                    const bool done = std::fabs(rn - r) < 1e-7;
                    r = rn;
                    if (done) break;
                }
            }
            rho[j] = r;
        });
    };
    fit_clusters();
    if (n_low > 0) {
        // attach each low-depth cell to the cluster whose mixture (1 - c_k) phi_k + c_k a explains its counts best
        parallel_for(n, thr, [&](size_t j) {
            if (!low[j]) return;
            double best = -std::numeric_limits<double>::max();
            uint32_t bk = 0;
            for (uint32_t k = 0; k < K; ++k) {
                double ll = 0.0;
                for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) {
                    const uint32_t g = X.gene[e];
                    ll += X.val[e] * std::log((1.0 - ck[k]) * phi[static_cast<size_t>(g) * K + k] + ck[k] * a[g]);
                }
                if (ll > best) { best = ll; bk = k; }
            }
            cl[j] = bk;
        });
    }
    fit_cells();
    // Outlier round: a doublet or damaged cell whose "ambient" fraction is far above its cluster's would drag the
    // cluster's c_k up and pollute its native profile. Flag cells with rho >> cluster median (robust z), refit without them.
    if (cfg.outlier_round) {
        size_t n_out = 0;
        for (uint32_t k = 0; k < K; ++k) {
            if (members[k].size() < 20) continue;
            std::vector<double> v;
            v.reserve(members[k].size());
            for (uint32_t j : members[k]) v.push_back(rho[j]);
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2];
            std::vector<double> dev;
            dev.reserve(v.size());
            for (double x : v) dev.push_back(std::fabs(x - med));
            std::sort(dev.begin(), dev.end());
            const double scale = std::max(1.4826 * dev[dev.size() / 2], 0.2 * med);
            for (uint32_t j : members[k])
                if (rho[j] > med + cfg.outlier_z * scale && rho[j] > 2.0 * med) { use[j] = 0; ++n_out; }
        }
        R.n_outlier_cells = static_cast<uint32_t>(n_out);
        if (n_out > 0) { fit_clusters(); fit_cells(); }
    }
    if (cfg.cluster_prior_reads <= 0.0 && cfg.cluster_prior_cells <= 0.0) R.method = "decontx_cluster_offgene_v1";
    R.cluster_rho = ck;
    R.cluster_rho_raw = ck_raw;
    R.sample_rho = c_sample;
    R.cluster_off_mass = offA;
    R.cluster_cells.assign(K, 0);
    for (uint32_t k = 0; k < K; ++k) R.cluster_cells[k] = static_cast<uint32_t>(ncells_k[k]);

    // integer correction: expected native counts x_g (1 - r_g), largest-remainder rounding per cell
    corrected_vals.assign(X.gene.size(), 0);
    parallel_for(n, thr, [&](size_t j) {
        const uint32_t k = cl[j];
        const double r = rho[j];
        const size_t b = X.ptr[j], e1 = X.ptr[j + 1];
        std::vector<std::pair<double, uint32_t>> frac;   // (fractional part, nnz offset)
        frac.reserve(e1 - b);
        double sum_nat = 0.0; uint64_t sum_floor = 0;
        for (size_t e = b; e < e1; ++e) {
            const uint32_t g = X.gene[e];
            const double pa = r * a[g], pn = (1.0 - r) * phi[static_cast<size_t>(g) * K + k];
            double nat = X.val[e] * (pn / (pa + pn));
            nat = std::min(nat, static_cast<double>(X.val[e]));
            const double fl = std::floor(nat);
            corrected_vals[e] = static_cast<uint32_t>(fl);
            sum_floor += static_cast<uint64_t>(fl);
            sum_nat += nat;
            frac.emplace_back(nat - fl, static_cast<uint32_t>(e - b));
        }
        const uint64_t target = static_cast<uint64_t>(std::llround(sum_nat));
        if (target > sum_floor && !frac.empty()) {
            size_t need = static_cast<size_t>(target - sum_floor);
            need = std::min(need, frac.size());
            auto cmp = [](const std::pair<double, uint32_t>& x, const std::pair<double, uint32_t>& y) {
                return x.first != y.first ? x.first > y.first : x.second < y.second;
            };
            std::nth_element(frac.begin(), frac.begin() + static_cast<std::ptrdiff_t>(need - 1), frac.end(), cmp);
            for (size_t i = 0; i < need; ++i) {
                const size_t e = b + frac[i].second;
                if (corrected_vals[e] < X.val[e]) ++corrected_vals[e];
            }
        }
    });

    R.contam.assign(n, CellContamination{});
    for (size_t j = 0; j < n; ++j) {
        auto& c = R.contam[j];
        c.rho = rho[j];
        c.cluster = cl[j];
        c.off_reads = xo_v[j];
        c.off_exposure = ex_v[j];
        c.prior_rho = ck[cl[j]];
        c.n_genes_used = static_cast<uint32_t>(X.ptr[j + 1] - X.ptr[j]);
        c.n_umi = static_cast<uint32_t>(std::min<double>(X.tot[j], std::numeric_limits<uint32_t>::max()));
        uint64_t kept = 0;
        for (size_t e = X.ptr[j]; e < X.ptr[j + 1]; ++e) kept += corrected_vals[e];
        c.n_removed = static_cast<uint32_t>(static_cast<uint64_t>(X.tot[j]) - kept);
    }
}

/// End-to-end: gene-level count matrix of ALL barcodes -> ambient profile, per-cell rho, corrected matrix.
///
/// @param gene_counts  genes x barcodes (all barcodes, uint16 or uint32 data); gene = collapse_exon_to_gene()
///                     (+ introns via collapse_to_gene_counts()).
/// @param cell_indices barcode columns of the called cells.
/// @return result.corrected: genes x gene_counts.ncols (uint32), non-empty only for called cells;
///         result.contam[i] belongs to cell_indices[i]. If there is no ambient pool or no cell, ok == false and the
///         corrected matrix is the observed counts of the called cells (identity), rho = 0.
template <class CSC>
inline AmbientCorrectionResult correct_ambient_gene_level(
    const CSC&                   gene_counts,
    const std::vector<uint32_t>& cell_indices,
    const AmbientDecontConfig&   cfg = AmbientDecontConfig())
{
    using namespace ambient_detail;
    const auto t0 = std::chrono::steady_clock::now();
    AmbientCorrectionResult R;
    R.n_cells_input = static_cast<uint32_t>(cell_indices.size());
    const uint32_t G = gene_counts.nrows, B = gene_counts.ncols;
    R.corrected.nrows = G;
    R.corrected.ncols = B;
    R.corrected.indptr.assign(static_cast<size_t>(B) + 1, 0);
    R.contam.assign(cell_indices.size(), CellContamination{});
    if (G == 0 || B == 0 || cell_indices.empty()) return R;

    // barcode totals; ambient pool excludes called cells
    std::vector<uint64_t> tot(B, 0);
    for (uint32_t b = 0; b < B; ++b)
        for (int32_t k = gene_counts.indptr[b]; k < gene_counts.indptr[b + 1]; ++k)
            tot[b] += static_cast<uint64_t>(gene_counts.data[k]);
    // Cell-set guard. The estimator must see the run's final caller cells (cc_result.cell_indices): every called barcode is
    // kept out of the ambient pool, and the clusters / native profiles are fitted on them. Duplicate and out-of-range
    // indices are dropped from the fit (a duplicate gets its first occurrence's record, an out-of-range index rho 0), and
    // barcodes above the pool cap that are not in the cell set are counted: they are in neither the pool nor the fit, which
    // is what a knee-only cell set does (on Kang 2.1 it doubles the estimate, docs/ambient-calibration.md).
    std::vector<int32_t> cell_of(B, -1);     // barcode -> position in the deduplicated cell list
    std::vector<int32_t> first_of(cell_indices.size(), -1);
    std::vector<char> first_occ(cell_indices.size(), 0);
    std::vector<uint32_t> cells_u;
    cells_u.reserve(cell_indices.size());
    for (size_t i = 0; i < cell_indices.size(); ++i) {
        const uint32_t b = cell_indices[i];
        if (b >= B) { ++R.n_cells_dropped; continue; }
        if (cell_of[b] >= 0) { first_of[i] = cell_of[b]; ++R.n_cells_dropped; continue; }
        cell_of[b] = static_cast<int32_t>(cells_u.size());
        first_of[i] = cell_of[b];
        first_occ[i] = 1;
        cells_u.push_back(b);
    }
    std::vector<uint64_t> pool_tot = tot;
    for (uint32_t b = 0; b < B; ++b) {
        if (cell_of[b] >= 0) pool_tot[b] = 0;
        else if (tot[b] > cfg.lower_umi) ++R.n_uncalled_above_pool;
    }
    if (R.n_cells_dropped > 0)
        std::cerr << "[ambient] WARNING: " << R.n_cells_dropped << " duplicate or out-of-range cell indices ignored\n";
    if (!cells_u.empty() && R.n_uncalled_above_pool > cells_u.size() / 10)
        std::cerr << "[ambient] WARNING: " << R.n_uncalled_above_pool << " uncalled barcodes above " << cfg.lower_umi
                  << " UMIs (" << cells_u.size() << " cells): pass the run's final caller cells, not a knee subset\n";
    R.profile = estimate_ambient(gene_counts, pool_tot, cfg.lower_umi, cfg.n_ambient_max);

    // called-cell sub-matrix (deduplicated cells)
    const size_t nu = cells_u.size();
    CellMat X;
    X.n_genes = G; X.n_cells = static_cast<uint32_t>(nu);
    X.ptr.assign(nu + 1, 0);
    X.tot.assign(nu, 0.0);
    for (size_t i = 0; i < nu; ++i) {
        const uint32_t b = cells_u[i];
        for (int32_t k = gene_counts.indptr[b]; k < gene_counts.indptr[b + 1]; ++k) {
            const auto v = static_cast<uint32_t>(gene_counts.data[k]);
            if (v == 0) continue;
            X.gene.push_back(static_cast<uint32_t>(gene_counts.indices[k]));
            X.val.push_back(v);
            X.tot[i] += v;
        }
        X.ptr[i + 1] = X.gene.size();
    }

    std::vector<uint32_t> cvals;
    const bool have_pool = R.profile.n_ambient_barcodes > 0 && R.profile.total_ambient_umi >= 1.0;
    if (have_pool && nu > 0) {
        decontx_ambient_core(X, R.profile.gene_fractions, cfg, R, cvals);
        R.ok = true;
    } else {
        cvals = X.val;   // identity
        R.contam.assign(nu, CellContamination{});
        for (size_t i = 0; i < nu; ++i) {
            R.contam[i].n_genes_used = static_cast<uint32_t>(X.ptr[i + 1] - X.ptr[i]);
            R.contam[i].n_umi = static_cast<uint32_t>(X.tot[i]);
        }
    }
    {
        // back to the caller's order: contam[i] belongs to cell_indices[i]
        std::vector<CellContamination> out(cell_indices.size(), CellContamination{});
        for (size_t i = 0; i < cell_indices.size(); ++i)
            if (first_of[i] >= 0) out[i] = R.contam[static_cast<size_t>(first_of[i])];
        R.contam = std::move(out);
    }
    {
        // genes seen in the pool: recompute cheaply from the fractions (above the pseudocount floor)
        if (!R.profile.gene_fractions.empty()) {
            const double floor_f = 0.5 / (R.profile.total_ambient_umi + 0.5 * G) * 1.0000001;
            uint32_t nz = 0;
            for (double f : R.profile.gene_fractions) if (f > floor_f) ++nz;
            R.n_genes_detected_pool = nz;
        }
    }

    // scatter into the all-barcode CSC (called cells only)
    size_t total_nnz = 0;
    for (size_t e = 0; e < cvals.size(); ++e) if (cvals[e] > 0) ++total_nnz;
    R.corrected.indices.reserve(total_nnz);
    R.corrected.data.reserve(total_nnz);
    for (uint32_t b = 0; b < B; ++b) {
        const int32_t i = cell_of[b];
        if (i >= 0)
            for (size_t e = X.ptr[i]; e < X.ptr[i + 1]; ++e)
                if (cvals[e] > 0) {
                    R.corrected.indices.push_back(static_cast<int32_t>(X.gene[e]));
                    R.corrected.data.push_back(cvals[e]);
                }
        R.corrected.indptr[static_cast<size_t>(b) + 1] = static_cast<int32_t>(R.corrected.indices.size());
    }

    // summary (each distinct valid cell once)
    std::vector<double> rhos;
    rhos.reserve(R.contam.size());
    for (size_t i = 0; i < R.contam.size(); ++i) {
        if (!first_occ[i]) continue;
        const auto& c = R.contam[i];
        rhos.push_back(c.rho); R.mean_rho += c.rho; R.umis_removed += c.n_removed; R.umis_observed += c.n_umi;
    }
    if (rhos.empty()) rhos.push_back(0.0);
    R.mean_rho /= static_cast<double>(rhos.size());
    std::nth_element(rhos.begin(), rhos.begin() + static_cast<std::ptrdiff_t>(rhos.size() / 2), rhos.end());
    R.median_rho = rhos[rhos.size() / 2];
    R.umi_weighted_rho = R.umis_observed ? static_cast<double>(R.umis_removed) / static_cast<double>(R.umis_observed) : 0.0;
    R.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return R;
}

/// Exon and intron gene counts as separate features: the ambient profile, clusters, native profiles and the correction use
/// both (the soup's intron share differs from a cell's), while c_k / rho are estimated from exon features only (cross-species
/// or off-cluster intronic reads track nuclear content, not soup). Returns the result folded back to genes: corrected
/// = exon part + intron part (still integer), profile = exon + intron fractions (sums to 1).
/// @param exon_genes, intron_genes  genes x barcodes (same shape), e.g. collapse_exon_to_gene() / intron_to_gene()
template <class CSC>
inline AmbientCorrectionResult correct_ambient_gene_level_split(
    const CSC& exon_genes, const CSC& intron_genes,
    const std::vector<uint32_t>& cell_indices,
    AmbientDecontConfig cfg = AmbientDecontConfig())
{
    const uint32_t G = exon_genes.nrows, B = exon_genes.ncols;
    if (intron_genes.ncols != B || intron_genes.nrows != G || intron_genes.data.empty())
        return correct_ambient_gene_level(exon_genes, cell_indices, cfg);
    CSCu32 S;
    S.nrows = 2 * G; S.ncols = B;
    S.indptr.assign(static_cast<size_t>(B) + 1, 0);
    S.indices.reserve(exon_genes.indices.size() + intron_genes.indices.size());
    S.data.reserve(exon_genes.data.size() + intron_genes.data.size());
    for (uint32_t b = 0; b < B; ++b) {
        for (int32_t k = exon_genes.indptr[b]; k < exon_genes.indptr[b + 1]; ++k) {
            S.indices.push_back(exon_genes.indices[k]);
            S.data.push_back(static_cast<uint32_t>(exon_genes.data[k]));
        }
        for (int32_t k = intron_genes.indptr[b]; k < intron_genes.indptr[b + 1]; ++k) {
            S.indices.push_back(static_cast<int32_t>(G) + intron_genes.indices[k]);
            S.data.push_back(static_cast<uint32_t>(intron_genes.data[k]));
        }
        S.indptr[static_cast<size_t>(b) + 1] = static_cast<int32_t>(S.indices.size());
    }
    cfg.estimate_features.assign(2 * static_cast<size_t>(G), 0);
    std::fill(cfg.estimate_features.begin(), cfg.estimate_features.begin() + G, 1);
    AmbientCorrectionResult R = correct_ambient_gene_level(S, cell_indices, cfg);
    // fold 2G -> G
    for (uint32_t g = 0; g < G && !R.profile.gene_fractions.empty(); ++g) R.exon_profile_share += R.profile.gene_fractions[g];
    if (!R.profile.gene_fractions.empty()) {
        for (uint32_t g = 0; g < G; ++g) R.profile.gene_fractions[g] += R.profile.gene_fractions[G + g];
        R.profile.gene_fractions.resize(G);
    }
    CSCu32 F;
    F.nrows = G; F.ncols = B;
    F.indptr.assign(static_cast<size_t>(B) + 1, 0);
    std::vector<uint32_t> acc(G, 0);
    std::vector<uint32_t> touched;
    for (uint32_t b = 0; b < B; ++b) {
        for (int32_t k = R.corrected.indptr[b]; k < R.corrected.indptr[b + 1]; ++k) {
            const uint32_t g = static_cast<uint32_t>(R.corrected.indices[k]) % G;
            if (acc[g] == 0) touched.push_back(g);
            acc[g] += R.corrected.data[k];
        }
        std::sort(touched.begin(), touched.end());
        for (uint32_t g : touched) { F.indices.push_back(static_cast<int32_t>(g)); F.data.push_back(acc[g]); acc[g] = 0; }
        touched.clear();
        F.indptr[static_cast<size_t>(b) + 1] = static_cast<int32_t>(F.indices.size());
    }
    R.corrected = std::move(F);
    // Report rho on the exon scale (the scale of gene-count matrices and of cross-species truth): ambient exon UMIs
    // over exon UMIs = rho_total * N_total * (exon share of the soup) / N_exon. rho_total stays in the record.
    if (R.ok) {
        double aex = 0.0;
        // profile was folded (exon + intron); recompute the exon share from the unfolded stack's first half
        aex = R.exon_profile_share;
        std::vector<double> exon_umis(R.contam.size(), 0.0);
        for (size_t i = 0; i < cell_indices.size(); ++i) {
            const uint32_t b = cell_indices[i];
            if (b >= B) continue;
            for (int32_t k = exon_genes.indptr[b]; k < exon_genes.indptr[b + 1]; ++k) exon_umis[i] += static_cast<double>(exon_genes.data[k]);
        }
        double sum = 0.0;
        std::vector<double> rhos;
        for (size_t i = 0; i < R.contam.size(); ++i) {
            auto& c = R.contam[i];
            c.rho_total = c.rho;
            if (exon_umis[i] > 0 && c.n_umi > 0)
                c.rho = std::min(cfg.rho_max, c.rho_total * static_cast<double>(c.n_umi) * aex / exon_umis[i]);
            sum += c.rho; rhos.push_back(c.rho);
        }
        R.mean_rho = sum / static_cast<double>(R.contam.size());
        std::nth_element(rhos.begin(), rhos.begin() + static_cast<std::ptrdiff_t>(rhos.size() / 2), rhos.end());
        R.median_rho = rhos[rhos.size() / 2];
    }
    return R;
}

// ── Legacy exon-level depth estimator ───────────────────────────────────────────

/// Estimate contamination fraction rho per cell using the depth-based model (legacy, no gene information):
/// rho_c = S / N_c, S = mean UMIs per empty droplet. Kept for older callers/tests; export.h uses
/// correct_ambient_gene_level().
inline std::vector<CellContamination> estimate_contamination(
    const CSCu16&                counts,
    const AmbientProfile&        ambient,
    const std::vector<uint32_t>& cell_indices,
    const std::vector<uint64_t>& bc_totals,
    uint32_t                     n_top_genes      = 200,
    double                       rho_min          = 0.001,
    double                       rho_max          = 0.50,
    double                       min_soup_ratio   = 2.0,
    double                       min_ambient_frac = 1e-5,
    uint32_t                     min_markers      = 20,
    double                       pct_estimator    = 0.05)
{
    (void)counts; (void)n_top_genes; (void)min_soup_ratio;
    (void)min_ambient_frac; (void)min_markers; (void)pct_estimator;

    std::vector<CellContamination> result(cell_indices.size());

    if (ambient.total_ambient_umi < 1.0 || ambient.n_ambient_barcodes == 0 ||
        cell_indices.empty())
        return result;

    const double S = ambient.total_ambient_umi /
                     static_cast<double>(ambient.n_ambient_barcodes);

    for (size_t ci = 0; ci < cell_indices.size(); ++ci) {
        const uint32_t bc = cell_indices[ci];
        const double Nc = static_cast<double>(bc_totals[bc]);
        if (Nc <= 0.0) {
            result[ci].rho = rho_max;
            continue;
        }
        double rho = S / Nc;
        rho = std::max(rho_min, std::min(rho, rho_max));
        result[ci].rho          = rho;
        result[ci].n_genes_used = 0;  // depth-based, no marker genes
    }

    return result;
}

// ── Legacy count correction ─────────────────────────────────────────────────

/// Subtract estimated ambient contamination from each called cell's counts (feature level, rounded per entry):
///   corrected[g, c] = max(0, obs[g, c] - round(rho_c * N_c * ambient_fraction[g]))
/// Only meaningful for gene-level matrices (at exon-interval level every subtraction rounds to zero).
/// Non-cell barcodes get rho 0 and are unchanged.
inline CSCu16 correct_counts(
    const CSCu16&                        counts,
    const AmbientProfile&                ambient,
    const std::vector<CellContamination>& contam,
    const std::vector<uint32_t>&         cell_indices,
    const std::vector<uint64_t>&         bc_totals)
{
    CSCu16 out;
    out.nrows   = counts.nrows;
    out.ncols   = counts.ncols;
    out.indptr.resize(static_cast<size_t>(counts.ncols) + 1, 0);

    if (ambient.total_ambient_umi < 1.0 || cell_indices.empty()) {
        out = counts;
        return out;
    }

    std::vector<double> bc_rho(counts.ncols, 0.0);
    for (size_t i = 0; i < cell_indices.size() && i < contam.size(); ++i)
        bc_rho[cell_indices[i]] = contam[i].rho;

    out.indices.reserve(counts.indices.size());
    out.data.reserve(counts.data.size());

    for (uint32_t c = 0; c < counts.ncols; ++c) {
        const double rho = bc_rho[c];
        const double Nc  = static_cast<double>(bc_totals[c]);
        const int32_t st = counts.indptr[c];
        const int32_t en = counts.indptr[c + 1];

        for (int32_t kk = st; kk < en; ++kk) {
            const uint32_t g   = static_cast<uint32_t>(counts.indices[kk]);
            const int32_t  obs = static_cast<int32_t>(counts.data[kk]);
            int32_t corrected  = obs;
            if (rho > 0.0 && Nc > 0.0) {
                const int32_t sub = static_cast<int32_t>(
                    std::round(rho * Nc * ambient.gene_fractions[g]));
                corrected = obs - sub;
            }
            if (corrected <= 0) continue;  // drop zeros and negatives
            out.indices.push_back(static_cast<int32_t>(g));
            out.data.push_back(static_cast<uint16_t>(
                std::min(corrected, static_cast<int32_t>(UINT16_MAX))));
        }
        out.indptr[c + 1] = static_cast<int32_t>(out.indices.size());
    }

    return out;
}

// ── I/O helpers ──────────────────────────────────────────────────────────────

/// Write per-feature ambient fraction profile to TSV.  Columns: feature, ambient_fraction
inline void write_ambient_profile(
    const std::string&              outpath,
    const AmbientProfile&           profile,
    const std::vector<std::string>& feature_names)
{
    std::ofstream out(outpath);
    if (!out.is_open()) {
        std::cerr << "[ambient] ERROR: cannot write " << outpath << "\n";
        return;
    }
    out << "feature\tambient_fraction\n";
    const uint32_t n = static_cast<uint32_t>(
        std::min(profile.gene_fractions.size(), feature_names.size()));
    for (uint32_t g = 0; g < n; ++g)
        out << feature_names[g] << '\t' << profile.gene_fractions[g] << '\n';
}

/// Write per-cell contamination estimates to TSV (called cells only).
/// Columns: barcode, rho, n_genes_used, cluster, n_umi, n_ambient_umi_removed
inline void write_contamination_tsv(
    const std::string&                    outpath,
    const std::vector<CellContamination>& contam,
    const std::vector<uint32_t>&          cell_indices,
    const std::vector<std::string>&       barcodes)
{
    std::ofstream out(outpath);
    if (!out.is_open()) {
        std::cerr << "[ambient] ERROR: cannot write " << outpath << "\n";
        return;
    }
    out << "barcode\trho\tn_genes_used\tcluster\tn_umi\tn_ambient_umi_removed\trho_total\n";
    const size_t n = std::min(contam.size(), cell_indices.size());
    for (size_t i = 0; i < n; ++i) {
        const uint32_t bc = cell_indices[i];
        const std::string& bc_name = (bc < barcodes.size()) ? barcodes[bc] : "?";
        out << bc_name     << '\t'
            << contam[i].rho          << '\t'
            << contam[i].n_genes_used << '\t'
            << contam[i].cluster      << '\t'
            << contam[i].n_umi        << '\t'
            << contam[i].n_removed    << '\t'
            << (contam[i].rho_total > 0.0 ? contam[i].rho_total : contam[i].rho) << '\n';
    }
}

/// Write the run summary (ambient_contamination.json).
inline void write_ambient_json(const std::string& outpath, const AmbientCorrectionResult& R,
                               const AmbientDecontConfig& cfg)
{
    std::ofstream o(outpath);
    if (!o.is_open()) {
        std::cerr << "[ambient] ERROR: cannot write " << outpath << "\n";
        return;
    }
    o.precision(6);
    o << "{\n"
      << "  \"method\": \"" << (R.ok ? R.method : std::string("none_no_ambient_pool")) << "\",\n"
      << "  \"estimated_contamination_fraction\": " << R.mean_rho << ",\n"
      << "  \"mean_rho\": " << R.mean_rho << ",\n"
      << "  \"median_rho\": " << R.median_rho << ",\n"
      << "  \"rho\": " << R.median_rho << ",\n"
      << "  \"umi_weighted_rho\": " << R.umi_weighted_rho << ",\n"
      << "  \"n_cells\": " << R.contam.size() << ",\n"
      << "  \"n_clusters\": " << R.n_clusters << ",\n"
      << "  \"n_outlier_cells\": " << R.n_outlier_cells << ",\n"
      << "  \"n_low_depth_cells\": " << R.n_low_depth_cells << ",\n"
      << "  \"umis_observed\": " << R.umis_observed << ",\n"
      << "  \"umis_removed\": " << R.umis_removed << ",\n"
      << "  \"ambient_profile_n_barcodes\": " << R.profile.n_ambient_barcodes << ",\n"
      << "  \"ambient_profile_total_umis\": " << static_cast<uint64_t>(R.profile.total_ambient_umi) << ",\n"
      << "  \"ambient_profile_n_genes_detected\": " << R.n_genes_detected_pool << ",\n"
      << "  \"ambient_pool_max_umi\": " << cfg.lower_umi << ",\n"
      << "  \"prior_strength\": " << cfg.prior_strength << ",\n"
      << "  \"cell_prior_reads\": " << cfg.off_prior_reads << ",\n"
      << "  \"cluster_prior_reads\": " << cfg.cluster_prior_reads << ",\n"
      << "  \"sample_rho\": " << R.sample_rho << ",\n"
      << "  \"n_cells_dropped\": " << R.n_cells_dropped << ",\n"
      << "  \"n_uncalled_above_pool\": " << R.n_uncalled_above_pool << ",\n"
      << "  \"seconds\": " << R.seconds << "\n"
      << "}\n";
}

} // namespace singlet
