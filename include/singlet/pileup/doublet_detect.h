// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: doublet_detect.h
// N12: expression-based doublet detection (v7: Scrublet-style simulation + classifier, call calibrated to the expected rate).
//
// ---------------------------------------------------------------------------
// Score  (Wolock et al. 2019, Scrublet; classifier stage after Germain et al. 2021, scDblFinder)
// ---------------------------------------------------------------------------
//   1. Features -> genes for the called cells only: optional feature_to_gene map (exon interval -> gene) and an optional
//      second source over the same barcodes (intron counts), so the gene counts are exon + intron.
//   2. Normalise each cell to the mean library size (no log), as Scrublet does.
//   3. Highly variable genes: genes seen in >= 3 cells are binned into 20 equal-count bins by mean; log(variance/mean)
//      is z-scored within the bin (Seurat v1 dispersion) and the top n_hvg (default 8000) are kept.  More genes help
//      (Kang pool AUROC 0.82 / 0.84 / 0.87 with 1,000 / 2,000 / 8,000), so this is far above the old 200.
//   4. Gene-wise z-score (mean/SD of the observed cells); PCA fitted on the OBSERVED cells only (30 PCs, randomized
//      subspace iteration on the sparse matrix, 7 power iterations; no dense matrix is formed).
//   5. Simulate n_sim = min(2 * n_cells, max_sim) doublets as sums of random pairs of observed cells' counts, normalised
//      to the same library size and projected on the observed-cell PCs.  The projection is linear in the raw counts, so
//      a simulated doublet costs O(n_pcs) once every observed cell's raw projection is known.
//   6. kNN over observed + simulated points in PC space (exact, tiled brute force, std::thread): k = round(0.5 *
//      sqrt(n_cells)), k_adj = round(k * (1 + n_sim / n_cells)), self excluded.  Per query point: the simulated-neighbour
//      fraction among the k_adj neighbours (as a logit) and the point's library size, absolute and relative to its
//      observed neighbours (PCA on depth-normalised counts cannot see library size).  Finer neighbourhoods are not
//      used: the observed doublets of one cell-type pair are each other's nearest neighbours.
//   7. Score.  classifier = true (default): logistic regression "simulated doublet vs observed cell" on those three
//      features, trained on the observed cells and up to sim_query_cap simulated doublets, class-weighted to prior rho,
//      (clf_rounds > 1 refits without the top 1.5 * rho * n observed cells, likely doublets, in the negatives).  The two library-size
//      coefficients carry an extra ridge (clf_size_ridge = 0.05 * n_queries): library size is the one cue that also
//      finds same-type doublets and it was worth +0.04 AUROC on the Kang pool, but unregularised it dominates and
//      costs 0.03 AUROC on a barnyard of human + mouse cells with only cross-species doublets.  score = the posterior
//      probability under prior rho, in [0, 1].
//      classifier = false: Scrublet's Bayesian score  q = (n_sim_nb + 1) / (k_adj + 2),
//      ld = q * rho / r / (1 - rho - q * (1 - rho - rho / r)),  r = n_sim / n_cells.
//
// ---------------------------------------------------------------------------
// Call  (calibrated to the expected multiplet rate)
// ---------------------------------------------------------------------------
//   rho = expected_rate, or by default the 10x rule  min(0.008 * n_cells / 1000, 0.25)  (0.8% per 1,000 recovered cells;
//   the cap keeps very large lanes from calling a quarter of the data).
//   h  = detectable fraction = share of the simulated doublets scoring above the 95th percentile of the observed
//        singlets (read at level 0.95 * (1 - rho) of all observed cells, since the top rho * h of them are doublets).
//        Same-type pairs look like singlets to the neighbour features and are the part that cannot be found.
//   is_doublet  <=>  the cell is among the top  round(rho * n_cells * h)  by score (ties broken by position).
//   The call rate therefore never exceeds rho * h, whatever the score distribution looks like (an expected-rate
//   threshold, not a valley in the observed scores: the old Gaussian-mixture threshold flagged 20-25% of cells).
//   Libraries below 50 called cells or 10 usable genes use a UMI-ratio fallback score (>= 3x median UMIs = 1) and
//   are called only above that score, within the same cap.
//
// Everything is deterministic for a fixed seed and independent of the thread count (fixed chunking, ordered reductions).
// Cost: O((n_cells + n_sim) * (n_cells + sim_query_cap)) distance evaluations in 30 dimensions, about 2 s for 14k
// cells and 8 threads.
//
// API:
//   struct DoubletResult { double score; bool is_doublet; }
//   struct DoubletParams { expected_rate (<= 0: auto), n_hvg, n_pcs, sim_ratio, max_sim, classifier, threads, seed, ... }
//   struct DoubletSummary { expected_rate, detectable_fraction, null_score, n_called, seconds, ... }
//   detect_doublets(counts, cell_indices, params = {}, feature_to_gene = nullptr, n_genes = 0, summary = nullptr,
//                   counts2 = nullptr, feature_to_gene2 = nullptr)
//   detect_doublets(counts, cell_indices, double expected_rate)       (compatibility overload)
//   write_doublet_tsv(path, results, cell_indices, barcodes, counts)  (columns unchanged)
//
// History: v2-v6 used 200 HVGs of the exon matrix with a power-iteration PCA on observed + simulated points and a
// Gaussian-mixture threshold clamped to >= 0.15.  The score ranked (AUROC 0.987 on a synthetic barnyard, 0.83 on the
// Kang pool) but the threshold flagged 20-25% of cells, 73% of the scores were exactly 0, and calls had precision 0.2.

#include "sparse_accumulator.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace singlet {

// -- Result types ------------------------------------------------------------

struct DoubletResult {
    double score;       ///< doublet_score in [0, 1]
    bool   is_doublet;  ///< calibrated call (see header)
};

struct DoubletParams {
    double   expected_rate  = -1.0;   ///< multiplet rate rho; <= 0: auto = min(rate_per_1000 * n / 1000, max_rate)
    double   rate_per_1000  = 0.008;  ///< 10x: 0.8% per 1,000 recovered cells
    double   max_rate       = 0.25;
    uint32_t n_hvg          = 8000;
    uint32_t n_pcs          = 30;
    double   sim_ratio      = 2.0;    ///< n_sim / n_cells
    uint32_t max_sim        = 100000;
    uint32_t sim_query_cap  = 20000;  ///< simulated doublets scored (detectable fraction, classifier); all stay kNN references
    uint32_t min_cells_gene = 3;
    bool     classifier     = true;   ///< scDblFinder-style: logistic regression on kNN fractions + library size (see header)
    int      clf_rounds     = 1;      ///< classifier refits; later rounds leave the likely doublets out of the negatives
    double   clf_exclude    = 1.5;    ///< ... namely the top clf_exclude * rho * n observed cells
    double   clf_ridge      = 0.01;   ///< ridge (x n_queries) on all coefficients: keeps separable data from blowing the weights up
    double   clf_size_ridge = 0.05;   ///< extra ridge (x n_queries) on the two library-size coefficients
    uint32_t clf_mask       = 7;     ///< classifier features: bit 0 sim-neighbour logit, 1 log T, 2 log T minus observed neighbours
    double   knn_scale      = 1.0;    ///< k = knn_scale * 0.5 * sqrt(n_cells)
    int      hvg_mode       = 0;      ///< 0: z-scored log dispersion in mean bins; 1: Scrublet v-score (percentile cut)
    double   vscore_pctl    = 85.0;   ///< hvg_mode 1: keep genes above this v-score percentile
    double   vscore_min_count = 2.0;  ///< hvg_mode 1: gene needs a normalised count >= this in >= min_cells_gene cells
    int      threads        = 0;      ///< 0 = hardware_concurrency
    uint64_t seed           = 42;
};

struct DoubletSummary {
    uint32_t n_cells = 0, n_sim = 0, n_hvg = 0, n_pcs = 0, k = 0, n_called = 0;
    double   expected_rate = 0.0;
    double   null_score = -1.0;       ///< score of the 95th-percentile observed singlet (the detectable-fraction threshold)
    double   detectable_fraction = 0.0;
    double   seconds = 0.0;
    bool     used_fallback = false;
    std::vector<uint32_t> sim_hist, obs_hist;  ///< 20-bin histograms of the simulated / observed scores on [0, 1]
};

// -- Internal helpers --------------------------------------------------------

namespace doublet_detail {

using CSC = SparseAccumulator<uint16_t>::CSCMatrix;

/// Run f(chunk) for chunk in [0, n_chunks) on up to n_threads threads (dynamic assignment).
template <class F>
inline void run_chunks(uint32_t n_chunks, int n_threads, F&& f)
{
    n_threads = std::max(1, std::min<int>(n_threads, static_cast<int>(n_chunks)));
    if (n_threads == 1) { for (uint32_t c = 0; c < n_chunks; ++c) f(c); return; }
    std::atomic<uint32_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const uint32_t c = next.fetch_add(1);
            if (c >= n_chunks) break;
            f(c);
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < n_threads; ++t) th.emplace_back(worker);
    worker();
    for (auto& t : th) t.join();
}

/// Per-cell gene accumulator: sums duplicate features and maps feature -> gene, over one or two sources
/// (e.g. exon and intron counts of the same barcodes).
struct Gather {
    const CSC* m[2];
    const std::vector<uint32_t>* f2g[2];
    uint32_t G;
    std::vector<uint32_t> acc, active;
    Gather(const CSC& m0, const std::vector<uint32_t>* f0, const CSC* m1, const std::vector<uint32_t>* f1, uint32_t G_)
        : G(G_), acc(G_, 0u) { m[0] = &m0; m[1] = m1; f2g[0] = f0; f2g[1] = f1; }
    uint32_t gene_of(int src, uint32_t f) const {
        if (!f2g[src]) return f < G ? f : UINT32_MAX;
        return f < f2g[src]->size() ? (*f2g[src])[f] : UINT32_MAX;
    }
    /// total mapped counts of a column
    double total(uint32_t col) const {
        double t = 0.0;
        for (int src = 0; src < 2; ++src) {
            if (!m[src]) continue;
            for (int32_t k = m[src]->indptr[col]; k < m[src]->indptr[col + 1]; ++k)
                if (gene_of(src, static_cast<uint32_t>(m[src]->indices[k])) < G) t += m[src]->data[k];
        }
        return t;
    }
    /// fn(gene, count) in ascending gene order
    template <class F>
    void operator()(uint32_t col, F&& fn) {
        active.clear();
        for (int src = 0; src < 2; ++src) {
            if (!m[src]) continue;
            for (int32_t k = m[src]->indptr[col]; k < m[src]->indptr[col + 1]; ++k) {
                if (m[src]->data[k] == 0) continue;
                const uint32_t g = gene_of(src, static_cast<uint32_t>(m[src]->indices[k]));
                if (g >= G) continue;
                if (acc[g] == 0) active.push_back(g);
                acc[g] += m[src]->data[k];
            }
        }
        std::sort(active.begin(), active.end());
        for (uint32_t g : active) { fn(g, acc[g]); acc[g] = 0; }
    }
};

/// Orthonormalise the r columns of A (H x r, row-major, double) in place (modified Gram-Schmidt, two passes).
inline void orthonormalize(std::vector<double>& A, uint32_t H, uint32_t r)
{
    for (uint32_t j = 0; j < r; ++j) {
        for (int pass = 0; pass < 2; ++pass) {
            for (uint32_t i = 0; i < j; ++i) {
                double dot = 0.0;
                for (uint32_t h = 0; h < H; ++h) dot += A[size_t(h) * r + i] * A[size_t(h) * r + j];
                for (uint32_t h = 0; h < H; ++h) A[size_t(h) * r + j] -= dot * A[size_t(h) * r + i];
            }
        }
        double nrm = 0.0;
        for (uint32_t h = 0; h < H; ++h) nrm += A[size_t(h) * r + j] * A[size_t(h) * r + j];
        nrm = std::sqrt(nrm);
        const double inv = nrm > 1e-300 ? 1.0 / nrm : 0.0;
        for (uint32_t h = 0; h < H; ++h) A[size_t(h) * r + j] *= inv;
    }
}

/// Cyclic Jacobi eigendecomposition of a symmetric r x r matrix (row-major).  vals ascending-unsorted; vecs columns.
inline void sym_eig(std::vector<double> A, uint32_t r, std::vector<double>& vals, std::vector<double>& vecs)
{
    vecs.assign(size_t(r) * r, 0.0);
    for (uint32_t i = 0; i < r; ++i) vecs[size_t(i) * r + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0.0, diag = 0.0;
        for (uint32_t i = 0; i < r; ++i) {
            diag += A[size_t(i) * r + i] * A[size_t(i) * r + i];
            for (uint32_t j = i + 1; j < r; ++j) off += A[size_t(i) * r + j] * A[size_t(i) * r + j];
        }
        if (off <= 1e-22 * (diag + off) || off == 0.0) break;
        for (uint32_t p = 0; p + 1 < r; ++p)
            for (uint32_t q = p + 1; q < r; ++q) {
                const double apq = A[size_t(p) * r + q];
                if (std::fabs(apq) < 1e-300) continue;
                const double theta = (A[size_t(q) * r + q] - A[size_t(p) * r + p]) / (2.0 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (uint32_t k = 0; k < r; ++k) {
                    const double akp = A[size_t(k) * r + p], akq = A[size_t(k) * r + q];
                    A[size_t(k) * r + p] = c * akp - s * akq;
                    A[size_t(k) * r + q] = s * akp + c * akq;
                }
                for (uint32_t k = 0; k < r; ++k) {
                    const double apk = A[size_t(p) * r + k], aqk = A[size_t(q) * r + k];
                    A[size_t(p) * r + k] = c * apk - s * aqk;
                    A[size_t(q) * r + k] = s * apk + c * aqk;
                }
                for (uint32_t k = 0; k < r; ++k) {
                    const double vkp = vecs[size_t(k) * r + p], vkq = vecs[size_t(k) * r + q];
                    vecs[size_t(k) * r + p] = c * vkp - s * vkq;
                    vecs[size_t(k) * r + q] = s * vkp + c * vkq;
                }
            }
    }
    vals.resize(r);
    for (uint32_t i = 0; i < r; ++i) vals[i] = A[size_t(i) * r + i];
}

} // namespace doublet_detail

// -- Core API ----------------------------------------------------------------

/// Detect doublets.
///
/// @param counts          CSC matrix (features x barcodes).  Features are genes, or exon intervals / anything else when
///                        feature_to_gene is given.
/// @param cell_indices    columns of `counts` that are called cells
/// @param prm             see DoubletParams
/// @param feature_to_gene optional map feature -> gene (entries >= n_genes are dropped); null = features are genes
/// @param n_genes         number of genes (ignored when feature_to_gene is null: counts.nrows)
/// @param summary         optional diagnostics
/// @param counts2         optional second matrix over the same barcodes (e.g. intron counts), folded into the same genes
///                        through feature_to_gene2 (null: its features are genes)
/// @return                one DoubletResult per cell_index, same order
inline std::vector<DoubletResult> detect_doublets(
    const SparseAccumulator<uint16_t>::CSCMatrix& counts,
    const std::vector<uint32_t>& cell_indices,
    const DoubletParams& prm = DoubletParams(),
    const std::vector<uint32_t>* feature_to_gene = nullptr,
    uint32_t n_genes = 0,
    DoubletSummary* summary = nullptr,
    const SparseAccumulator<uint16_t>::CSCMatrix* counts2 = nullptr,
    const std::vector<uint32_t>* feature_to_gene2 = nullptr)
{
    using namespace doublet_detail;
    const auto t_start = std::chrono::steady_clock::now();
    const uint32_t n = static_cast<uint32_t>(cell_indices.size());
    std::vector<DoubletResult> results(n, DoubletResult{0.0, false});
    DoubletSummary sm;
    sm.n_cells = n;
    auto lap = [&](const char* what) {
#ifdef SINGLET_DOUBLET_TIMING
        std::cerr << "[doublet-timing] " << what << " "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() << "s\n";
#else
        (void)what;
#endif
    };
    auto finish = [&]() {
        uint32_t nc = 0;
        for (const auto& r : results) nc += r.is_doublet ? 1u : 0u;
        sm.n_called = nc;
        sm.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        if (summary) *summary = sm;
    };
    if (n == 0) { finish(); return results; }

    const uint32_t G = feature_to_gene ? n_genes : counts.nrows;
    const double rho = std::min(prm.max_rate, prm.expected_rate > 0.0
                                    ? prm.expected_rate
                                    : prm.rate_per_1000 * static_cast<double>(n) / 1000.0);
    sm.expected_rate = rho;
    const int nthreads = prm.threads > 0 ? prm.threads
                                         : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    // ---- pass A: library sizes over mapped features ----
    Gather gather(counts, feature_to_gene, counts2, feature_to_gene2, G);
    std::vector<double> T(n, 0.0);
    double T_sum = 0.0; uint32_t T_n = 0;
    for (uint32_t i = 0; i < n; ++i) {
        const double t = gather.total(cell_indices[i]);
        T[i] = t;
        if (t > 0.0) { T_sum += t; ++T_n; }
    }

    // calls: top round(rho * n * h) cells by score, all with score > t (t < 0: no score condition)
    auto call_cells = [&](double t, double h) {
        const size_t cap = static_cast<size_t>(std::llround(rho * n * h));
        std::vector<uint32_t> order(n);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (results[a].score != results[b].score) return results[a].score > results[b].score;
            return a < b;
        });
        for (size_t r = 0; r < std::min<size_t>(cap, n); ++r)
            if (results[order[r]].score > t) results[order[r]].is_doublet = true;
    };

    // ---- fallback: UMI-ratio score ----
    auto umi_fallback = [&]() {
        sm.used_fallback = true;
        std::vector<double> sorted(T);
        std::sort(sorted.begin(), sorted.end());
        const double median = 0.5 * (sorted[(n - 1) / 2] + sorted[n / 2]);
        const double denom = (median > 0.0) ? 3.0 * median : 1.0;
        for (uint32_t i = 0; i < n; ++i)
            results[i].score = (median > 0.0) ? std::min(1.0, T[i] / denom) : 0.0;
        call_cells(1.0 - 1e-9, 0.5);
        finish();
        return results;
    };
    if (n < 50 || G < 10 || T_n < 50) return umi_fallback();

    const double target = T_sum / T_n;

    // ---- pass B: per-gene statistics of the normalised counts ----
    std::vector<double> s1(G, 0.0), s2(G, 0.0);
    std::vector<uint32_t> ncell(G, 0u), ncell_hi(G, 0u);
    for (uint32_t i = 0; i < n; ++i) {
        if (T[i] <= 0.0) continue;
        const double s = target / T[i];
        gather(cell_indices[i], [&](uint32_t g, uint32_t c) {
            const double v = c * s;
            s1[g] += v; s2[g] += v * v; ncell[g] += 1;
            if (v >= prm.vscore_min_count) ncell_hi[g] += 1;
        });
    }
    std::vector<double> gmean(G, 0.0), gvar(G, 0.0);
    std::vector<uint32_t> elig;
    for (uint32_t g = 0; g < G; ++g) {
        if (ncell[g] < prm.min_cells_gene) continue;
        const double mu = s1[g] / n;
        const double var = std::max(0.0, s2[g] / n - mu * mu) * n / std::max(1u, n - 1);
        if (mu <= 0.0 || var <= 0.0) continue;
        gmean[g] = mu; gvar[g] = var;
        elig.push_back(g);
    }
    if (elig.size() < 10) return umi_fallback();

    // ---- HVG ----
    std::vector<uint32_t> hvg_genes;
    if (prm.hvg_mode == 1) {
        // Scrublet's v-score: Fano factor over a noise model (1+a)(1+b*mu) fitted to the 10th-percentile envelope of
        // log(Fano / mean) against log(mean); keep genes above the vscore_pctl percentile that pass the count filter.
        std::vector<uint32_t> pos;
        for (uint32_t g = 0; g < G; ++g) if (s1[g] > 0.0 && ncell[g] >= 1) pos.push_back(g);
        std::vector<double> FF(G, 0.0), lm(G, 0.0), ly(G, 0.0);
        double xlo = 1e300, xhi = -1e300;
        std::vector<double> lff;
        for (uint32_t g : pos) {
            const double mu_g = s1[g] / n;
            const double var = std::max(1e-300, s2[g] / n - mu_g * mu_g);
            FF[g] = var / mu_g; lm[g] = std::log(mu_g); ly[g] = std::log(FF[g] / mu_g);
            xlo = std::min(xlo, lm[g]); xhi = std::max(xhi, lm[g]);
            lff.push_back(std::log(FF[g]));
        }
        if (pos.size() < 10 || !(xhi > xlo)) return umi_fallback();
        const int NB = 50;
        const double dx = (xhi - xlo) / NB;
        std::vector<std::vector<double>> bins(NB);
        for (uint32_t g : pos) bins[std::min(NB - 1, int((lm[g] - xlo) / dx))].push_back(ly[g]);
        std::vector<double> bx, by;
        for (int bi = 0; bi < NB; ++bi) {
            if (bins[bi].empty()) continue;
            std::sort(bins[bi].begin(), bins[bi].end());
            const double pos_ix = 0.1 * (bins[bi].size() - 1);
            const size_t i0 = static_cast<size_t>(pos_ix);
            const double fr = pos_ix - i0;
            const double q10 = bins[bi][i0] * (1.0 - fr) + bins[bi][std::min(bins[bi].size() - 1, i0 + 1)] * fr;
            bx.push_back(xlo + (bi + 0.5) * dx); by.push_back(q10);
        }
        // c: exp(mode of the log-Fano histogram, 200 bins), at least 1
        const double fl = *std::min_element(lff.begin(), lff.end()), fh = *std::max_element(lff.begin(), lff.end());
        std::vector<int> hh(200, 0);
        for (double v : lff) ++hh[std::min(199, int((v - fl) / std::max(1e-12, fh - fl) * 200))];
        const int mx = int(std::max_element(hh.begin(), hh.end()) - hh.begin());
        const double c = std::max(std::exp(fl + (mx + 0.5) * (fh - fl) / 200.0), 1.0);
        auto err = [&](double bb) {
            double e = 0.0;
            for (size_t i = 0; i < bx.size(); ++i) e += std::fabs(std::log(c * std::exp(-bx[i]) + bb) - by[i]);
            return e;
        };
        double bbest = 0.1, ebest = err(0.1);
        for (int i = 0; i <= 400; ++i) {                      // coarse log grid, then local refinement
            const double bb = std::exp(std::log(1e-4) + i * (std::log(50.0) - std::log(1e-4)) / 400.0);
            const double e = err(bb);
            if (e < ebest) { ebest = e; bbest = bb; }
        }
        for (double step = bbest * 0.05; step > bbest * 1e-4; step *= 0.5)
            for (int dir = -1; dir <= 1; dir += 2) {
                const double bb = bbest + dir * step;
                if (bb > 0.0 && err(bb) < ebest) { ebest = err(bb); bbest = bb; }
            }
        const double ca = c / (1.0 + bbest) - 1.0;
        std::vector<std::pair<double, uint32_t>> vs;
        for (uint32_t g : pos) {
            const double v = FF[g] / ((1.0 + ca) * (1.0 + bbest * (s1[g] / n)));
            if (v > 0.0 && v == v) vs.emplace_back(v, g);
        }
        std::vector<double> vv;
        for (auto& e : vs) vv.push_back(e.first);
        std::sort(vv.begin(), vv.end());
        if (vv.empty()) return umi_fallback();
        const double pi = prm.vscore_pctl / 100.0 * (vv.size() - 1);
        const size_t j0 = static_cast<size_t>(pi);
        const double minv = vv[j0] * (1.0 - (pi - j0)) + vv[std::min(vv.size() - 1, j0 + 1)] * (pi - j0);
        std::vector<std::pair<double, uint32_t>> keep;
        for (auto& e : vs)
            if (e.first >= minv && ncell_hi[e.second] >= prm.min_cells_gene) keep.push_back(e);
        std::sort(keep.begin(), keep.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        if (keep.size() > prm.n_hvg) keep.resize(prm.n_hvg);
        for (auto& e : keep) hvg_genes.push_back(e.second);
        std::sort(hvg_genes.begin(), hvg_genes.end());
        if (hvg_genes.size() < 10) return umi_fallback();
    } else {
    // z-scored log dispersion (Seurat v1) within 20 equal-count mean bins
    {
        std::vector<uint32_t> by_mean(elig);
        std::sort(by_mean.begin(), by_mean.end(), [&](uint32_t a, uint32_t b) {
            return gmean[a] != gmean[b] ? gmean[a] < gmean[b] : a < b;
        });
        const size_t ne = by_mean.size();
        const size_t nb = std::max<size_t>(1, std::min<size_t>(20, ne / 50));
        std::vector<std::pair<double, uint32_t>> z(ne);
        for (size_t b = 0; b < nb; ++b) {
            const size_t lo = b * ne / nb, hi = (b + 1) * ne / nb;
            double m = 0.0, m2 = 0.0;
            for (size_t i = lo; i < hi; ++i) {
                const uint32_t g = by_mean[i];
                const double x = std::log(gvar[g] / gmean[g]);
                m += x; m2 += x * x;
            }
            const double cnt = static_cast<double>(hi - lo);
            m /= cnt;
            const double sd = std::sqrt(std::max(0.0, m2 / cnt - m * m));
            for (size_t i = lo; i < hi; ++i) {
                const uint32_t g = by_mean[i];
                const double x = std::log(gvar[g] / gmean[g]);
                z[i] = { sd > 1e-9 ? (x - m) / sd : 0.0, g };
            }
        }
        const size_t H0 = std::min<size_t>(prm.n_hvg, ne);
        std::partial_sort(z.begin(), z.begin() + H0, z.end(), [](const auto& a, const auto& b) {
            return a.first != b.first ? a.first > b.first : a.second < b.second;
        });
        for (size_t i = 0; i < H0; ++i) hvg_genes.push_back(z[i].second);
        std::sort(hvg_genes.begin(), hvg_genes.end());
    }
    }
    lap("hvg");
    const uint32_t H = static_cast<uint32_t>(hvg_genes.size());
    const uint32_t n_pc = std::min<uint32_t>({ prm.n_pcs, H > 1 ? H - 1 : 0u, n - 1 });
    if (n_pc < 2) return umi_fallback();
    std::vector<uint32_t> g2h(G, UINT32_MAX);
    std::vector<double> mu(H), sd(H);
    for (uint32_t h = 0; h < H; ++h) {
        const uint32_t g = hvg_genes[h];
        g2h[g] = h;
        mu[h] = s1[g] / n;
        const double var = std::max(0.0, s2[g] / n - mu[h] * mu[h]);  // population variance, as Scrublet's z-score
        sd[h] = var > 1e-12 ? std::sqrt(var) : 1.0;
    }
    { std::vector<double>().swap(s1); std::vector<double>().swap(s2); }

    // ---- pass C: observed cells x HVG, raw counts (CSR) ----
    std::vector<uint64_t> rptr(n + 1, 0);
    std::vector<uint32_t> ridx;
    std::vector<float> rval;
    for (uint32_t i = 0; i < n; ++i) {
        gather(cell_indices[i], [&](uint32_t g, uint32_t c) {
            const uint32_t h = g2h[g];
            if (h != UINT32_MAX) { ridx.push_back(h); rval.push_back(static_cast<float>(c)); }
        });
        rptr[i + 1] = ridx.size();
    }
    lap("rows");
    std::vector<double> cell_scale(n);  // s_i = target / T_i
    for (uint32_t i = 0; i < n; ++i) cell_scale[i] = T[i] > 0.0 ? target / T[i] : 0.0;

    // ---- PCA of the z-scored normalised matrix, observed cells only (randomized subspace iteration) ----
    const uint32_t r = std::min<uint32_t>(n_pc + 10, H);
    const uint32_t NC = 32;  // fixed cell chunks: results independent of the thread count
    auto chunk_lo = [&](uint32_t c) { return static_cast<uint32_t>(uint64_t(n) * c / NC); };

    // Y = A Q  (n x r),  A = (X / sd) - 1 (mu / sd)^T
    auto apply_A = [&](const std::vector<double>& Q, std::vector<double>& Y) {
        std::vector<double> Qs(size_t(H) * r), offs(r, 0.0);
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t c = 0; c < r; ++c) {
                Qs[size_t(h) * r + c] = Q[size_t(h) * r + c] / sd[h];
                offs[c] += mu[h] * Qs[size_t(h) * r + c];
            }
        Y.assign(size_t(n) * r, 0.0);
        run_chunks(NC, nthreads, [&](uint32_t ch) {
            for (uint32_t i = chunk_lo(ch); i < chunk_lo(ch + 1); ++i) {
                double* y = &Y[size_t(i) * r];
                for (uint64_t e = rptr[i]; e < rptr[i + 1]; ++e) {
                    const double v = rval[e];
                    const double* q = &Qs[size_t(ridx[e]) * r];
                    for (uint32_t c = 0; c < r; ++c) y[c] += v * q[c];
                }
                for (uint32_t c = 0; c < r; ++c) y[c] = cell_scale[i] * y[c] - offs[c];
            }
        });
    };
    // Z = A^T Y  (H x r)
    auto apply_At = [&](const std::vector<double>& Y, std::vector<double>& Z) {
        std::vector<std::vector<double>> Zc(NC), ys(NC);
        run_chunks(NC, nthreads, [&](uint32_t ch) {
            Zc[ch].assign(size_t(H) * r, 0.0);
            ys[ch].assign(r, 0.0);
            for (uint32_t i = chunk_lo(ch); i < chunk_lo(ch + 1); ++i) {
                const double* y = &Y[size_t(i) * r];
                for (uint32_t c = 0; c < r; ++c) ys[ch][c] += y[c];
                for (uint64_t e = rptr[i]; e < rptr[i + 1]; ++e) {
                    const double v = rval[e] * cell_scale[i];
                    double* z = &Zc[ch][size_t(ridx[e]) * r];
                    for (uint32_t c = 0; c < r; ++c) z[c] += v * y[c];
                }
            }
        });
        Z.assign(size_t(H) * r, 0.0);
        std::vector<double> ysum(r, 0.0);
        for (uint32_t ch = 0; ch < NC; ++ch) {
            for (uint32_t c = 0; c < r; ++c) ysum[c] += ys[ch][c];
            for (size_t k = 0; k < Z.size(); ++k) Z[k] += Zc[ch][k];
        }
        for (uint32_t h = 0; h < H; ++h)
            for (uint32_t c = 0; c < r; ++c)
                Z[size_t(h) * r + c] = (Z[size_t(h) * r + c] - mu[h] * ysum[c]) / sd[h];
    };

    std::vector<double> Q(size_t(H) * r), Y, Z;
    {
        std::mt19937_64 rng(prm.seed ^ 0x9e3779b97f4a7c15ULL);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& v : Q) v = nd(rng);
        orthonormalize(Q, H, r);
        for (int it = 0; it < 7; ++it) {
            apply_A(Q, Y);
            apply_At(Y, Z);
            Q.swap(Z);
            orthonormalize(Q, H, r);
        }
        apply_A(Q, Y);
    }
    // top n_pc directions of Y: eigenvectors of Y^T Y
    std::vector<double> W(size_t(H) * n_pc, 0.0);
    {
        std::vector<double> Gm(size_t(r) * r, 0.0);
        for (uint32_t i = 0; i < n; ++i) {
            const double* y = &Y[size_t(i) * r];
            for (uint32_t a = 0; a < r; ++a)
                for (uint32_t b = a; b < r; ++b) Gm[size_t(a) * r + b] += y[a] * y[b];
        }
        for (uint32_t a = 0; a < r; ++a)
            for (uint32_t b = a + 1; b < r; ++b) Gm[size_t(b) * r + a] = Gm[size_t(a) * r + b];
        std::vector<double> ev, U;
        sym_eig(Gm, r, ev, U);
        std::vector<uint32_t> ord(r);
        std::iota(ord.begin(), ord.end(), 0u);
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
            return ev[a] != ev[b] ? ev[a] > ev[b] : a < b;
        });
        for (uint32_t p = 0; p < n_pc; ++p) {
            const uint32_t col = ord[p];
            // fix the sign (largest |loading| positive) so output does not depend on Jacobi details
            double best = 0.0, sgn = 1.0;
            std::vector<double> w(H, 0.0);
            for (uint32_t h = 0; h < H; ++h) {
                double s = 0.0;
                for (uint32_t c = 0; c < r; ++c) s += Q[size_t(h) * r + c] * U[size_t(c) * r + col];
                w[h] = s;
                if (std::fabs(s) > best) { best = std::fabs(s); sgn = s < 0 ? -1.0 : 1.0; }
            }
            for (uint32_t h = 0; h < H; ++h) W[size_t(h) * n_pc + p] = sgn * w[h];
        }
    }
    { std::vector<double>().swap(Y); std::vector<double>().swap(Z); std::vector<double>().swap(Q); }

    lap("pca");
    // ---- raw projections R_i = sum_h c_ih W[h,:] / sd_h ; coordinates are linear in them ----
    std::vector<double> Wsc(size_t(H) * n_pc), offs(n_pc, 0.0);
    for (uint32_t h = 0; h < H; ++h)
        for (uint32_t p = 0; p < n_pc; ++p) {
            Wsc[size_t(h) * n_pc + p] = W[size_t(h) * n_pc + p] / sd[h];
            offs[p] += mu[h] * Wsc[size_t(h) * n_pc + p];
        }
    std::vector<double> R(size_t(n) * n_pc, 0.0);
    run_chunks(NC, nthreads, [&](uint32_t ch) {
        for (uint32_t i = chunk_lo(ch); i < chunk_lo(ch + 1); ++i) {
            double* out = &R[size_t(i) * n_pc];
            for (uint64_t e = rptr[i]; e < rptr[i + 1]; ++e) {
                const double v = rval[e];
                const double* w = &Wsc[size_t(ridx[e]) * n_pc];
                for (uint32_t p = 0; p < n_pc; ++p) out[p] += v * w[p];
            }
        }
    });

    // ---- simulated doublets: sums of random pairs of observed cells ----
    uint32_t n_sim = static_cast<uint32_t>(std::min<double>(std::ceil(prm.sim_ratio * n), prm.max_sim));
    n_sim = std::max(n_sim, 10u);
    std::vector<std::pair<uint32_t, uint32_t>> parents(n_sim);
    {
        std::mt19937_64 rng(prm.seed * 0x2545F4914F6CDD1DULL + 12345u);
        for (uint32_t s = 0; s < n_sim; ++s) {
            const uint32_t a = static_cast<uint32_t>(rng() % n);
            uint32_t b = static_cast<uint32_t>(rng() % (n - 1));
            if (b >= a) ++b;
            parents[s] = { a, b };
        }
    }
    const uint32_t N = n + n_sim;
    std::vector<float> P(size_t(n_pc) * N);   // structure of arrays: P[c * N + point]
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t p = 0; p < n_pc; ++p)
            P[size_t(p) * N + i] = static_cast<float>(cell_scale[i] * R[size_t(i) * n_pc + p] - offs[p]);
    for (uint32_t s = 0; s < n_sim; ++s) {
        const uint32_t a = parents[s].first, b = parents[s].second;
        const double Tab = T[a] + T[b];
        const double sc = Tab > 0.0 ? target / Tab : 0.0;
        for (uint32_t p = 0; p < n_pc; ++p)
            P[size_t(p) * N + n + s] =
                static_cast<float>(sc * (R[size_t(a) * n_pc + p] + R[size_t(b) * n_pc + p]) - offs[p]);
    }

    lap("sims");
    // ---- kNN: observed + the first n_q_sim simulated points are queried against everything ----
    const uint32_t k = std::max<uint32_t>(5, static_cast<uint32_t>(std::lround(prm.knn_scale * 0.5 * std::sqrt(double(n)))));
    const double r_ratio = static_cast<double>(n_sim) / n;
    uint32_t k_adj = static_cast<uint32_t>(std::lround(k * (1.0 + r_ratio)));
    k_adj = std::max<uint32_t>(1, std::min<uint32_t>(k_adj, N - 1));
    const uint32_t n_q_sim = std::min(n_sim, prm.sim_query_cap);
    const uint32_t NQ = n + n_q_sim;
    std::vector<uint32_t> n_sim_nb(NQ, 0u);
    constexpr uint32_t NF = 3;                       // features per query (classifier): sim-neighbour logit, log T, log T - neighbours'
    std::vector<float> feat(size_t(NQ) * NF, 0.0f);
    std::vector<float> logT(N, 0.0f);                // log library size of every point (simulated: parents' sum)
    for (uint32_t i = 0; i < n; ++i) logT[i] = static_cast<float>(std::log(std::max(1.0, T[i])));
    for (uint32_t s2 = 0; s2 < n_sim; ++s2)
        logT[n + s2] = static_cast<float>(std::log(std::max(1.0, T[parents[s2].first] + T[parents[s2].second])));
    {
        const uint32_t QB = 32;                       // queries per work item
        const uint32_t n_items = (NQ + QB - 1) / QB;
        run_chunks(n_items, nthreads, [&](uint32_t item) {
            constexpr uint32_t TILE = 2048;
            std::vector<float> dist(TILE);
            std::vector<std::pair<float, uint32_t>> heap;
            heap.reserve(k_adj + 1);
            std::vector<float> qv(n_pc);
            for (uint32_t q = item * QB; q < std::min(NQ, (item + 1) * QB); ++q) {
                for (uint32_t p = 0; p < n_pc; ++p) qv[p] = P[size_t(p) * N + q];
                heap.clear();
                float thr = 3.4e38f;
                for (uint32_t t0 = 0; t0 < N; t0 += TILE) {
                    const uint32_t len = std::min(TILE, N - t0);
                    std::fill(dist.begin(), dist.begin() + len, 0.0f);
                    for (uint32_t p = 0; p < n_pc; ++p) {
                        const float* col = &P[size_t(p) * N + t0];
                        const float qc = qv[p];
                        for (uint32_t j = 0; j < len; ++j) { const float d = col[j] - qc; dist[j] += d * d; }
                    }
                    for (uint32_t j = 0; j < len; ++j) {
                        const float d = dist[j];
                        if (d >= thr) continue;
                        const uint32_t id = t0 + j;
                        if (id == q) continue;
                        if (heap.size() < k_adj) {
                            heap.emplace_back(d, id);
                            if (heap.size() == k_adj) { std::make_heap(heap.begin(), heap.end()); thr = heap.front().first; }
                        } else {
                            std::pop_heap(heap.begin(), heap.end());
                            heap.back() = { d, id };
                            std::push_heap(heap.begin(), heap.end());
                            thr = heap.front().first;
                        }
                    }
                }
                uint32_t cnt = 0, n_obs_nb = 0;
                double sum_lt = 0.0;
                for (const auto& e : heap) {
                    if (e.second >= n) ++cnt; else { sum_lt += logT[e.second]; ++n_obs_nb; }
                }
                float* fq = &feat[size_t(q) * NF];
                fq[0] = static_cast<float>(std::log((cnt + 0.5) / (heap.size() - cnt + 0.5)));
                fq[1] = logT[q];
                fq[2] = n_obs_nb ? static_cast<float>(logT[q] - sum_lt / n_obs_nb) : 0.0f;
                n_sim_nb[q] = cnt;
            }
        });
    }

    lap("knn");
    // ---- score ----
    // obs_stat / sim_stat: a monotone statistic of the score (neighbour fraction, or the logistic linear predictor) for
    // the observed cells and for the scored simulated doublets; the thresholds below are located on it.
    auto bayes_q = [&](double q) {
        const double rr = r_ratio;
        const double den = 1.0 - rho - q * (1.0 - rho - rho / rr);
        const double ld = den > 1e-12 ? q * rho / rr / den : 1.0;
        return std::min(1.0, std::max(0.0, ld));
    };
    auto bayes = [&](double nsim_nb) { return bayes_q((nsim_nb + 1.0) / (k_adj + 2.0)); };
    std::vector<float> obs_stat(n), sim_stat(n_q_sim), sim_scores(n_q_sim);
    std::function<double(double)> stat_to_score;
    if (!prm.classifier) {
        // Scrublet's Bayesian score from the number of simulated neighbours among the k_adj nearest points.
        for (uint32_t i = 0; i < n; ++i) {
            results[i].score = bayes(n_sim_nb[i]);
            obs_stat[i] = static_cast<float>(n_sim_nb[i]) / static_cast<float>(k_adj);
        }
        for (uint32_t s = 0; s < n_q_sim; ++s) {
            sim_scores[s] = static_cast<float>(bayes(n_sim_nb[n + s]));
            sim_stat[s]   = static_cast<float>(n_sim_nb[n + s]) / static_cast<float>(k_adj);
        }
        stat_to_score = [&](double f) { return bayes(f * k_adj); };
    } else {
        // Logistic regression of "simulated doublet" vs "observed cell" on the kNN features, class-weighted so that the
        // implied prior is rho (scDblFinder's idea).  Library size enters through log T and through log T relative to
        // the observed neighbours, which the normalised PCA space cannot see.
        std::vector<double> fm(NF, 0.0), fs(NF, 0.0);
        for (uint32_t q = 0; q < NQ; ++q) for (uint32_t f = 0; f < NF; ++f) fm[f] += feat[size_t(q) * NF + f];
        for (uint32_t f = 0; f < NF; ++f) fm[f] /= NQ;
        for (uint32_t q = 0; q < NQ; ++q) for (uint32_t f = 0; f < NF; ++f) {
            const double d = feat[size_t(q) * NF + f] - fm[f]; fs[f] += d * d;
        }
        for (uint32_t f = 0; f < NF; ++f) fs[f] = std::max(std::sqrt(fs[f] / NQ), 1e-3);
        constexpr uint32_t D = NF + 1;
        auto xrow = [&](uint32_t q, double* x) {
            x[0] = 1.0;
            for (uint32_t f = 0; f < NF; ++f)
                x[f + 1] = ((prm.clf_mask >> f) & 1u) ? (feat[size_t(q) * NF + f] - fm[f]) / fs[f] : 0.0;
        };
        std::vector<char> excl(n, 0);                // observed cells left out of the negatives (likely doublets)
        double beta[D] = {0};
        const int rounds = std::max(1, prm.clf_rounds);
        for (int round = 0; round < rounds; ++round) {
        uint32_t n_neg = 0;
        for (uint32_t i = 0; i < n; ++i) n_neg += excl[i] ? 0u : 1u;
        const double wo = (1.0 - rho) / std::max(1u, n_neg) * NQ, ws = rho / n_q_sim * NQ;
        for (double& b : beta) b = 0.0;
        for (int it = 0; it < 40; ++it) {
            double g[D] = {0}, Hm[D][D] = {{0}};
            for (uint32_t q = 0; q < NQ; ++q) {
                if (q < n && excl[q]) continue;
                double x[D]; xrow(q, x);
                double eta = 0.0; for (uint32_t d = 0; d < D; ++d) eta += beta[d] * x[d];
                const double pr = 1.0 / (1.0 + std::exp(-std::max(-30.0, std::min(30.0, eta))));
                const double y = q >= n ? 1.0 : 0.0, w = q >= n ? ws : wo;
                for (uint32_t d = 0; d < D; ++d) {
                    g[d] += w * (pr - y) * x[d];
                    for (uint32_t e = 0; e < D; ++e) Hm[d][e] += w * pr * (1.0 - pr) * x[d] * x[e];
                }
            }
            for (uint32_t d = 1; d < D; ++d) {
                const double lam = 1e-2 + prm.clf_ridge * NQ + (d >= 2 ? prm.clf_size_ridge * NQ : 0.0);   // d = 2, 3: the library-size features
                g[d] += lam * beta[d]; Hm[d][d] += lam;
            }
            // solve Hm * delta = g (Gaussian elimination, partial pivoting)
            double A[D][D + 1];
            for (uint32_t d = 0; d < D; ++d) { for (uint32_t e = 0; e < D; ++e) A[d][e] = Hm[d][e]; A[d][D] = g[d]; }
            for (uint32_t c = 0; c < D; ++c) {
                uint32_t pv = c;
                for (uint32_t r2 = c + 1; r2 < D; ++r2) if (std::fabs(A[r2][c]) > std::fabs(A[pv][c])) pv = r2;
                for (uint32_t e = 0; e <= D; ++e) std::swap(A[c][e], A[pv][e]);
                if (std::fabs(A[c][c]) < 1e-12) continue;
                for (uint32_t r2 = c + 1; r2 < D; ++r2) {
                    const double f2 = A[r2][c] / A[c][c];
                    for (uint32_t e = c; e <= D; ++e) A[r2][e] -= f2 * A[c][e];
                }
            }
            double delta[D] = {0};
            for (int c = int(D) - 1; c >= 0; --c) {
                double v = A[c][D];
                for (uint32_t e = c + 1; e < D; ++e) v -= A[c][e] * delta[e];
                delta[c] = std::fabs(A[c][c]) < 1e-12 ? 0.0 : v / A[c][c];
            }
            double step = 0.0;
            for (uint32_t d = 0; d < D; ++d) { beta[d] -= delta[d]; step = std::max(step, std::fabs(delta[d])); }
            if (step < 1e-7) break;
        }
        if (round + 1 < rounds) {
            // leave the highest-scoring observed cells (about 1.5x the expected doublets) out of the negatives and refit
            std::vector<std::pair<double, uint32_t>> ev(n);
            for (uint32_t i = 0; i < n; ++i) {
                double x[D]; xrow(i, x);
                double eta = 0.0; for (uint32_t d = 0; d < D; ++d) eta += beta[d] * x[d];
                ev[i] = { eta, i };
            }
            const size_t kx = std::min<size_t>(n / 2, static_cast<size_t>(std::llround(prm.clf_exclude * rho * n)));
            std::partial_sort(ev.begin(), ev.begin() + kx, ev.end(), [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
            std::fill(excl.begin(), excl.end(), 0);
            for (size_t i = 0; i < kx; ++i) excl[ev[i].second] = 1;
        }
        }  // rounds
        for (uint32_t q = 0; q < NQ; ++q) {
            double x[D]; xrow(q, x);
            double eta = 0.0; for (uint32_t d = 0; d < D; ++d) eta += beta[d] * x[d];
            const double pr = 1.0 / (1.0 + std::exp(-std::max(-30.0, std::min(30.0, eta))));
            if (q < n) { results[q].score = pr; obs_stat[q] = static_cast<float>(eta); }
            else { sim_scores[q - n] = static_cast<float>(pr); sim_stat[q - n] = static_cast<float>(eta); }
        }
        stat_to_score = [&](double eta) { return 1.0 / (1.0 + std::exp(-eta)); };
    }

    // ---- calibrated call ----
    // h = detectable fraction: share of the simulated doublets scoring above the 95th percentile of the observed
    // singlets.  The singlets' quantile is read at level 0.95 * (1 - rho) of ALL observed cells, because the top rho * h
    // of them are the doublets themselves.  Same-type pairs look like singlets and are not counted.
    double h = 0.0, null_score = -1.0;
    {
        std::vector<float> o(obs_stat);
        const size_t qi = static_cast<size_t>(std::floor(0.95 * (1.0 - rho) * (n - 1)));
        std::nth_element(o.begin(), o.begin() + qi, o.end());
        const float thr = o[qi];
        uint32_t above = 0;
        for (float v : sim_stat) if (v > thr) ++above;
        h = static_cast<double>(above) / std::max<size_t>(1, sim_stat.size());
        null_score = stat_to_score(thr);
    }
    sm.sim_hist.assign(20, 0u); sm.obs_hist.assign(20, 0u);
    for (float v : sim_scores) sm.sim_hist[std::min(19, int(v * 20))]++;
    for (const auto& rr : results) sm.obs_hist[std::min(19, int(rr.score * 20))]++;
    call_cells(-1.0, h);
    sm.n_sim = n_sim; sm.n_hvg = H; sm.n_pcs = n_pc; sm.k = k_adj;
    sm.null_score = null_score; sm.detectable_fraction = h;
    finish();
    return results;
}

/// Compatibility overload: explicit expected multiplet rate, everything else default.
inline std::vector<DoubletResult> detect_doublets(
    const SparseAccumulator<uint16_t>::CSCMatrix& counts,
    const std::vector<uint32_t>& cell_indices,
    double expected_doublet_rate)
{
    DoubletParams p;
    p.expected_rate = expected_doublet_rate;
    return detect_doublets(counts, cell_indices, p);
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
