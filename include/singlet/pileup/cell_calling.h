#pragma once
// singlet-pileup: cell_calling.h
// EmptyDrops-style statistical cell calling (N5).
// Lun et al. (2019) "EmptyDrops: distinguishing cells from empty droplets
// in droplet-based single-cell RNA sequencing data", Genome Biology.
//
// Algorithm (Monte Carlo, AUTOFIX-EMPTYDROPS-DEPTH-MC):
//   1. Estimate ambient RNA profile from EMPTY DROPLETS: barcodes with
//      total count <= lower (the true ambient RNA background, not cells).
//   2. For each test barcode (count >= min_umi_test):
//      a. Compute observed multinomial deviance:
//         Deviance = 2 * Σ_{g: n_g > 0} n_g * log(n_g / (N * p_g))
//      b. Monte Carlo p-value: generate n_monte_carlo multinomial draws
//         from the ambient profile at the SAME total UMI depth N as the
//         test barcode; p = (1 + #{sim_dev >= obs_dev}) / (1 + n_mc).
//         Barcodes with the same UMI depth share a pre-computed null.
//   3. Benjamini-Hochberg FDR correction; cells = FDR < fdr_threshold.
//
//   Key difference from chi-squared: the null distribution is generated at
//   matched UMI depth, so deeply-sequenced ambient droplets are NOT called
//   as cells (the chi-squared LRT always rejects when N >> ambient_mean_N).
//
// Input: the UNFILTERED count matrix (including empty droplets with low counts).
//        Barcodes with count <= lower define the ambient pool.
//        Barcodes with count >= min_umi_test are tested.
//
// Integration: call after PileupEngine::run() + COO→CSC conversion on the
// unfiltered exon accumulator. See INTEGRATION_NOTES.md.

#include "sparse_accumulator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {

// ── Result ───────────────────────────────────────────────────────────────────

struct CellCallResult {
    std::vector<uint32_t> cell_indices;   ///< barcode indices of called cells
    std::vector<double>   deviance;       ///< multinomial deviance per tested barcode
    std::vector<double>   fdr;            ///< BH-corrected FDR per tested barcode
    std::vector<uint32_t> tested_indices; ///< barcode indices tested (count > lower)
    uint32_t n_ambient   = 0;             ///< number of ambient barcodes used
    double   ambient_total = 0.0;         ///< total UMI in ambient pool
};

// ── Statistical helpers ───────────────────────────────────────────────────────

namespace detail {

// log Γ(x) via Lanczos approximation (Numerical Recipes, 7th ed.)
inline double lgamma_nr(double x) {
    static const double g   = 7.0;
    static const double c[] = {
        0.99999999999980993,  676.5203681218851,  -1259.1392167224028,
        771.32342877765313,  -176.61502916214059,   12.507343278686905,
         -0.13857109526572012,  9.9843695780195716e-6, 1.5056327351493116e-7
    };
    if (x < 0.5)
        return std::log(M_PI / std::sin(M_PI * x)) - lgamma_nr(1.0 - x);
    x -= 1.0;
    double a = c[0];
    for (int i = 1; i <= 8; ++i) a += c[i] / (x + i);
    double t = x + g + 0.5;
    return 0.5 * std::log(2.0 * M_PI) + (x + 0.5) * std::log(t) - t + std::log(a);
}

// Upper regularized incomplete gamma Q(a, x) = 1 − P(a, x).
// Used for chi-squared survival: P(chi^2(df) > stat) = Q(df/2, stat/2).
inline double gammaq(double a, double x) {
    if (x < 0.0) return 1.0;
    if (x == 0.0) return 1.0;
    if (x < a + 1.0) {
        // Series for P(a, x): return 1 - P
        double ap = a, sum = 1.0 / a, del = sum;
        for (int n = 1; n <= 300; ++n) {
            ap  += 1.0;
            del *= x / ap;
            sum += del;
            if (std::abs(del) < std::abs(sum) * 1e-13) break;
        }
        double P = sum * std::exp(-x + a * std::log(x) - lgamma_nr(a));
        return std::max(0.0, 1.0 - P);
    } else {
        // Lentz continued fraction for Q(a, x)
        double b = x + 1.0 - a, cc = 1e30, d = 1.0 / b, h = d;
        for (int i = 1; i <= 300; ++i) {
            double an = -(double)i * ((double)i - a);
            b  += 2.0;
            d   = an * d + b;
            if (std::abs(d) < 1e-30)  d  = 1e-30;
            cc  = b + an / cc;
            if (std::abs(cc) < 1e-30) cc = 1e-30;
            d    = 1.0 / d;
            h   *= d * cc;
            if (std::abs(d * cc - 1.0) < 1e-13) break;
        }
        return std::min(1.0, h * std::exp(-x + a * std::log(x) - lgamma_nr(a)));
    }
}

/// P(chi^2(df) > stat) — upper-tail chi-squared p-value.
inline double chisq_pvalue(double stat, double df) {
    if (stat <= 0.0 || df <= 0.0) return 1.0;
    return gammaq(df / 2.0, stat / 2.0);
}

/// Benjamini-Hochberg FDR correction (in-place).
/// fdr_out[i] = BH-adjusted p-value (monotone under rank sort).
inline void bh_fdr(const std::vector<double>& pvals, std::vector<double>& fdr_out) {
    const size_t n = pvals.size();
    fdr_out.resize(n, 1.0);
    if (n == 0) return;

    // Rank order
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return pvals[a] < pvals[b]; });

    // Traverse largest rank → smallest, enforce monotonicity
    double running_min = 1.0;
    for (size_t i = n; i-- > 0;) {
        double bh = pvals[order[i]] * static_cast<double>(n) / static_cast<double>(i + 1);
        if (bh < running_min) running_min = bh;
        fdr_out[order[i]] = std::min(running_min, 1.0);
    }
}

// ── Monte Carlo p-value helpers ──────────────────────────────────────────────

/// Vose alias table: O(K) construction, true O(1) per sample (2 RNG calls).
/// Faster than std::discrete_distribution which uses O(log K) binary search on GCC.
/// Table size: K × 12 bytes (4 bytes alias index + 8 bytes threshold probability).
struct AliasTable {
    std::vector<uint32_t> alias;   ///< alias gene index
    std::vector<double>   prob;    ///< acceptance threshold in [0,1]
    uint32_t K = 0;

    /// Build alias table from a probability vector (must sum to 1.0).
    explicit AliasTable(const std::vector<double>& probs) : K(static_cast<uint32_t>(probs.size())) {
        alias.resize(K, 0);
        prob.resize(K, 0.0);
        // Scale probabilities to [0, K]
        std::vector<double> p(probs.begin(), probs.end());
        for (double& v : p) v *= static_cast<double>(K);

        std::vector<uint32_t> small, large;
        small.reserve(K); large.reserve(K);
        for (uint32_t i = 0; i < K; ++i)
            (p[i] < 1.0 ? small : large).push_back(i);

        // Vose alias construction: O(K)
        while (!small.empty() && !large.empty()) {
            uint32_t l = small.back(); small.pop_back();
            uint32_t g = large.back(); large.pop_back();
            prob[l]  = p[l];
            alias[l] = g;
            p[g]     = p[g] + p[l] - 1.0;
            (p[g] < 1.0 ? small : large).push_back(g);
        }
        for (uint32_t i : large) prob[i] = 1.0;
        for (uint32_t i : small) prob[i] = 1.0;  // floating-point residuals
    }

    /// Sample one gene index in O(1): 2 RNG calls, 2 table lookups.
    inline uint32_t sample(std::mt19937& rng) const {
        const uint32_t i = static_cast<uint32_t>(rng()) % K;
        const double u   = static_cast<double>(static_cast<uint32_t>(rng()))
                           * (1.0 / 4294967296.0);
        return (u < prob[i]) ? i : alias[i];
    }
};

/// Compute multinomial deviance for a single barcode against ambient profile.
/// Returns 2 * Σ_{g: x_g > 0} x_g * [log(x_g) - log(N) - log(p_g)]
/// where N = total UMI count for this barcode and p_g = ambient_log_profile[g].
/// Caller must pass log(p_g) in log_ambient (precomputed).
/// CSC column [col_start, col_end) gives the sparse gene indices + counts.
template <typename DataT = uint16_t>
inline double compute_deviance(
    const int32_t* indices, const DataT* data,
    int32_t col_start, int32_t col_end,
    const std::vector<double>& log_ambient,
    uint64_t N)
{
    const double log_N = std::log(static_cast<double>(N));
    double dev = 0.0;
    for (int32_t k = col_start; k < col_end; ++k) {
        double ng = static_cast<double>(data[k]);
        dev += ng * (std::log(ng) - log_N - log_ambient[indices[k]]);
    }
    return (dev > 0.0) ? 2.0 * dev : 0.0;
}

/// Draw n_draw exact multinomial samples from the ambient profile and compute
/// the deviance of each draw.  Returns the vector of simulated deviances.
///
/// Uses an adaptive strategy based on n_total vs K (number of genes):
///
///   N < K  (most production cases):  alias-method multinomial sampling.
///          std::discrete_distribution uses the Vose alias table — O(K) setup,
///          O(1) per sample.  Total cost: O(n_total × n_draw).
///          For n_total=1200, K=38606: ~20× faster than the Poisson loop.
///
///   N ≥ K  (small-K unit tests / degenerate panels):  per-gene Poisson draws.
///          Each gene sampled independently with Poisson(N × p_g).
///          Total cost: O(Ka × n_draw) where Ka = active genes (lambda ≥ 0.01).
///          For K=20, n_total=2000: 20 Poisson calls << 2000 alias samples.
///
/// Both paths produce the same multinomial null distribution to O(1/N) error.
///
/// The nonzero_ambient_genes parameter is retained for API compatibility but
/// is only used by the Poisson path (N ≥ K branch).
inline std::vector<double> mc_null_deviances(
    uint64_t n_total,
    const std::vector<double>& ambient_profile,
    const std::vector<double>& log_ambient,
    const std::vector<uint32_t>& nonzero_ambient_genes,
    int n_draw,
    std::mt19937& rng)
{
    std::vector<double> sim_devs(n_draw, 0.0);
    if (n_total == 0) return sim_devs;

    const double log_N = std::log(static_cast<double>(n_total));
    const uint32_t K   = static_cast<uint32_t>(ambient_profile.size());

    if (n_total < static_cast<uint64_t>(K)) {
        // ── Alias-method path (N < K): O(n_total) per draw ──────────────────
        // Faster when N << K because we sample N times instead of iterating K genes.
        // Production case: K=38606 genes, typical N=500–25000 → always N < K.
        // AliasTable uses Vose algorithm: O(K) build, true O(1) sample (2 RNG calls).
        // Faster than std::discrete_distribution on GCC (which uses O(log K) binary search).
        AliasTable atab(ambient_profile);

        // Per-gene count accumulator, cleared by resetting only non-zero entries.
        std::vector<uint32_t> gene_counts(K, 0);
        std::vector<uint32_t> active_genes;
        active_genes.reserve(std::min(n_total, static_cast<uint64_t>(K)));

        for (int s = 0; s < n_draw; ++s) {
            active_genes.clear();
            for (uint64_t i = 0; i < n_total; ++i) {
                const uint32_t g = atab.sample(rng);
                if (gene_counts[g] == 0) active_genes.push_back(g);
                ++gene_counts[g];
            }
            double dev = 0.0;
            for (uint32_t g : active_genes) {
                const double ng = static_cast<double>(gene_counts[g]);
                dev += ng * (std::log(ng) - log_N - log_ambient[g]);
                gene_counts[g] = 0;  // reset for next draw
            }
            sim_devs[s] = (dev > 0.0) ? 2.0 * dev : 0.0;
        }
    } else {
        // ── Poisson path (N ≥ K): O(Ka) per draw ────────────────────────────
        // Faster when K << N (e.g., small test panels with K=20-50 genes).
        // Approximates Multinomial(N, p) by {Poisson(N*p_g)} independently.
        struct ActiveGene {
            double log_p;
            std::poisson_distribution<int> dist;
        };
        std::vector<ActiveGene> active;
        active.reserve(nonzero_ambient_genes.size());
        const double N_d = static_cast<double>(n_total);
        for (uint32_t g : nonzero_ambient_genes) {
            double lambda = N_d * ambient_profile[g];
            if (lambda < 0.01) continue;
            active.push_back({log_ambient[g],
                              std::poisson_distribution<int>(lambda)});
        }
        const size_t Ka = active.size();
        for (int s = 0; s < n_draw; ++s) {
            double dev = 0.0;
            for (size_t i = 0; i < Ka; ++i) {
                int ng = active[i].dist(rng);
                if (ng == 0) continue;
                double dng = static_cast<double>(ng);
                dev += dng * (std::log(dng) - log_N - active[i].log_p);
            }
            sim_devs[s] = (dev > 0.0) ? 2.0 * dev : 0.0;
        }
    }

    return sim_devs;
}

/// Compute Monte Carlo p-values for a set of test barcodes.
///
/// Barcodes are grouped into UMI-depth bins (10% width in log-space).
/// All barcodes in the same bin share one set of Monte Carlo null draws.
/// This brings the total simulation cost from O(n_tested * n_mc) draws
/// to O(n_bins * n_mc) draws (typically 20–40 bins × 10000 draws).
///
/// Early stopping: if after `early_stop_iter` simulations the running
/// p-value estimate is already > 10 * fdr_alpha * n_tests, skip the rest
/// (that barcode cannot become significant after BH correction).
///
/// @param n_monte_carlo  Simulations per UMI-depth bin (default 10000).
/// @param early_stop_iter  Minimum draws before early stop (default 200).
inline std::vector<double> mc_emptydrops_pvalues(
    const std::vector<uint32_t>& tested_indices,
    const std::vector<double>&   deviances,
    const std::vector<uint64_t>& counts_per_barcode,
    const std::vector<double>&   ambient_profile,
    const std::vector<double>&   log_ambient,
    int n_monte_carlo  = 10000,
    int early_stop_iter = 200)
{
    const size_t n_tested = tested_indices.size();
    std::vector<double> pvals(n_tested, 1.0);
    if (n_tested == 0) return pvals;

    // Build list of nonzero ambient genes for fast multinomial sampling
    std::vector<uint32_t> nonzero_genes;
    nonzero_genes.reserve(ambient_profile.size());
    for (uint32_t g = 0; g < static_cast<uint32_t>(ambient_profile.size()); ++g)
        if (ambient_profile[g] > 0.0)
            nonzero_genes.push_back(g);

    if (nonzero_genes.empty()) {
        // Degenerate ambient profile — all p-values = 1.0 (no cells called)
        return pvals;
    }

    // Bin test barcodes by UMI depth using log10 bucketing (10% width).
    // bin_key = floor(log10(N) * 10)  → bins are ~10% wide in linear UMI space.
    // Barcodes in the same bin share a null distribution.
    struct BinEntry {
        size_t test_idx;     // index into tested_indices / deviances
        uint64_t n_total;    // total UMI for this barcode
        double   obs_dev;    // observed deviance
    };

    // Group by bin_key
    std::unordered_map<int, std::vector<BinEntry>> bins;
    bins.reserve(64);
    for (size_t i = 0; i < n_tested; ++i) {
        uint64_t N = counts_per_barcode[tested_indices[i]];
        if (N == 0) { pvals[i] = 1.0; continue; }
        int bin_key = static_cast<int>(std::floor(std::log10(static_cast<double>(N)) * 10.0));
        bins[bin_key].push_back({i, N, deviances[i]});
    }

    // For each bin: draw null at the median UMI of the bin, assign p-values
    std::mt19937 rng(20240101u);  // fixed seed for reproducibility

    for (auto& [bin_key, entries] : bins) {
        if (entries.empty()) continue;

        // Use median UMI depth for the null draw
        std::sort(entries.begin(), entries.end(),
                  [](const BinEntry& a, const BinEntry& b) {
                      return a.n_total < b.n_total;
                  });
        uint64_t n_null = entries[entries.size() / 2].n_total;

        // Generate null deviances at this depth
        auto null_devs = mc_null_deviances(
            n_null, ambient_profile, log_ambient,
            nonzero_genes, n_monte_carlo, rng);

        // Sort null deviances ascending for binary search
        std::sort(null_devs.begin(), null_devs.end());
        const int n_mc = static_cast<int>(null_devs.size());

        // For each barcode in this bin: p = (1 + #{null >= obs}) / (1 + n_mc)
        for (const auto& e : entries) {
            // Count how many null_devs >= obs_dev via upper_bound
            auto it = std::lower_bound(null_devs.begin(), null_devs.end(), e.obs_dev);
            int n_ge = static_cast<int>(null_devs.end() - it);
            pvals[e.test_idx] = static_cast<double>(1 + n_ge) /
                                  static_cast<double>(1 + n_mc);
        }
    }

    return pvals;
}

} // namespace detail

// ── Knee-point fallback ───────────────────────────────────────────────────────

/// Knee-point (inflection) cell caller. Used as fallback when EmptyDrops has
/// an empty ambient pool (e.g. auto-barcode mode: only barcodes ≥100 reads
/// were discovered, so no barcodes are ≤ lower).
///
/// Algorithm:
///   1. Sort barcodes descending by UMI count.
///   2. Smooth log10(count) with a moving-average window of 5.
///   3. Find the inflection point (most-negative smoothed second derivative).
///   4. Call all barcodes at ranks 0 … knee_idx as cells.
///
/// @param counts_per_barcode  Total UMI per barcode (size = n_barcodes).
/// @param lower               Safety-floor: knee UMI is clamped to ≥ lower.
/// @return                    CellCallResult with cell_indices set; deviance/fdr
///                            are populated with sentinel values (0 / 0 for
///                            called cells, 0 / 1 for non-cells) so that
///                            write_cell_calls() can iterate without UB.
inline CellCallResult call_cells_knee_fallback(
    const std::vector<uint64_t>& counts_per_barcode,
    uint64_t lower = 100,
    uint64_t min_umi_for_knee = 0)  // if > 0, restrict knee detection to barcodes ≥ this UMI
{
    const uint32_t n_barcodes = static_cast<uint32_t>(counts_per_barcode.size());

    // Build rank-sorted index (descending by count, skip zero-count barcodes)
    // When min_umi_for_knee > 0, only include barcodes above that UMI in knee computation.
    // This restricts the inflection search to the tested range, preventing the knee
    // from landing near the very-low-UMI barcodes (near-empties) rather than at the
    // true cell/empty boundary.
    std::vector<uint32_t> ranked;
    ranked.reserve(n_barcodes);
    const uint64_t min_umi_cut = (min_umi_for_knee > lower) ? min_umi_for_knee : lower;
    for (uint32_t i = 0; i < n_barcodes; ++i)
        if (counts_per_barcode[i] >= min_umi_cut)
            ranked.push_back(i);
    std::sort(ranked.begin(), ranked.end(),
              [&](uint32_t a, uint32_t b) {
                  return counts_per_barcode[a] > counts_per_barcode[b];
              });

    const size_t n = ranked.size();

    // Find inflection index via smoothed second derivative
    size_t knee_idx = n > 0 ? n / 2 : 0;  // sensible default
    if (n >= 3) {
        // log10(count) for each rank
        std::vector<double> lc(n);
        for (size_t i = 0; i < n; ++i)
            lc[i] = std::log10(static_cast<double>(counts_per_barcode[ranked[i]]));

        // Moving-average smooth (window = 5)
        const int hw = 2;  // half-window
        std::vector<double> sm(n);
        for (size_t i = 0; i < n; ++i) {
            int lo = static_cast<int>(i) - hw;
            int hi = static_cast<int>(i) + hw;
            if (lo < 0) lo = 0;
            if (hi >= static_cast<int>(n)) hi = static_cast<int>(n) - 1;
            double s = 0.0;
            for (int j = lo; j <= hi; ++j) s += lc[j];
            sm[i] = s / static_cast<double>(hi - lo + 1);
        }

        // Smoothed second derivative; find most-negative (steepest bend)
        double min_d2 = 0.0;
        for (size_t i = 1; i + 1 < n; ++i) {
            double d2 = sm[i + 1] - 2.0 * sm[i] + sm[i - 1];
            if (d2 < min_d2) { min_d2 = d2; knee_idx = i; }
        }
    }

    // Threshold = UMI count at knee, clamped to ≥ lower
    uint64_t knee_umi  = (n > 0) ? counts_per_barcode[ranked[knee_idx]] : lower;
    uint64_t threshold = (knee_umi >= lower) ? knee_umi : lower;

    std::cerr << "[cell-call] EmptyDrops ambient pool empty (all barcodes > lower_umi="
              << lower << "); using knee-point fallback. knee_idx=" << knee_idx
              << " knee_umi=" << knee_umi << "\n";

    CellCallResult result;
    result.n_ambient    = 0;
    result.ambient_total = 0.0;

    for (size_t r = 0; r < n; ++r) {
        uint32_t bc = ranked[r];
        if (counts_per_barcode[bc] < threshold) break;  // sorted descending — done
        result.tested_indices.push_back(bc);
        bool is_cell = (r <= knee_idx);
        result.deviance.push_back(0.0);
        result.fdr.push_back(is_cell ? 0.0 : 1.0);
        if (is_cell) result.cell_indices.push_back(bc);
    }

    std::cerr << "[cell-call] knee-point fallback called " << result.cell_indices.size()
              << " cells from " << n << " barcodes\n";
    return result;
}

// ── Core function ─────────────────────────────────────────────────────────────

/// EmptyDrops statistical cell calling with Monte Carlo p-values (Lun et al. 2019).
///
/// Implements AUTOFIX-EMPTYDROPS-DEPTH-MC: replaces the chi-squared LRT
/// p-value with a depth-matched Monte Carlo p-value, eliminating systematic
/// overcalling of deeply-sequenced ambient droplets.
///
/// Algorithm:
///   1. Ambient profile estimated from barcodes with total count <= lower.
///      Barcodes with UMI dedup count <= lower from the auto-discovery set
///      populate the ambient pool (typically ~50k barcodes with 50–100 dedup
///      UMI that had >= 100 reads before deduplication).
///      When the UMI-threshold ambient pool is empty (all barcodes > lower),
///      a knee-based fallback computes the biological cell/empty boundary
///      and uses barcodes just below the knee as ambient.
///   2. Barcodes with count >= min_umi_test (default 500) are tested.
///      Matches STARsolo EmptyDrops_CR umiMin=500.
///   3. For each test barcode: Monte Carlo p-value from n_monte_carlo draws
///      from the ambient multinomial at the SAME UMI depth as the barcode.
///      Barcodes in the same log10-UMI bin (±10%) share a null distribution.
///      p = (1 + #{sim_dev >= obs_dev}) / (1 + n_monte_carlo).
///   4. Benjamini-Hochberg FDR correction; cells = FDR < fdr_threshold.
///
/// @param counts_per_barcode  Total UMI per barcode (size = n_barcodes).
/// @param gene_counts         CSC gene×barcode matrix (nrows=genes, ncols=barcodes).
/// @param fdr_threshold       FDR cutoff for calling a cell (default 0.01,
///                            matching STARsolo EmptyDrops_CR).
/// @param lower               UMI ceiling: barcodes with count <= lower define the
///                            ambient pool (default 100).
/// @param n_ambient_max       Max empty droplets to use for ambient estimation (default 50000).
/// @param min_umi_test        Minimum UMI count for a barcode to be tested (default 500).
///                            Matches STARsolo EmptyDrops_CR umiMin parameter.
/// @param n_monte_carlo       MC draws per UMI-depth bin (default 10000).
/// @return                    CellCallResult with per-barcode statistics.
template <typename CSCMatrixT = SparseAccumulator<uint16_t>::CSCMatrix>
inline CellCallResult call_cells_emptydrops(
    const std::vector<uint64_t>& counts_per_barcode,
    const CSCMatrixT& gene_counts,
    double   fdr_threshold  = 0.01,
    uint64_t lower          = 100,
    int      n_ambient_max  = 200000,
    uint64_t min_umi_test   = 500,
    int      n_monte_carlo  = 10000,
    // N22: Full-whitelist ambient override (optional)
    const std::vector<uint32_t>* wl_umi_counts      = nullptr,  // per-WL-barcode UMI counts
    const std::vector<uint64_t>* wl_ambient_gene_raw = nullptr, // global gene sums from WL reads
    uint32_t                     wl_ambient_ceil     = 50)      // max UMI for ambient classification
{
    const uint32_t n_barcodes = static_cast<uint32_t>(counts_per_barcode.size());
    const uint32_t n_genes    = gene_counts.nrows;

    // ── 1. Estimate ambient profile from empty droplets (count <= lower) ──────
    // Sort barcodes by count descending to efficiently pick top-n_ambient_max empties
    std::vector<uint32_t> sorted_bc(n_barcodes);
    std::iota(sorted_bc.begin(), sorted_bc.end(), 0u);
    std::sort(sorted_bc.begin(), sorted_bc.end(), [&](uint32_t a, uint32_t b) {
        return counts_per_barcode[a] > counts_per_barcode[b];
    });

    // ── 2. Build ambient profile ──────────────────────────────────────────────
    std::vector<double> ambient_profile(n_genes, 0.0);
    uint32_t n_ambient   = 0;
    double   ambient_tot = 0.0;

    // N22: Use full-whitelist ambient profile when available.
    // WL barcodes are true empty droplets (< discovery_threshold reads), giving a
    // representative ambient pool vs the sparse auto-discovered set (≤100 UMI barcodes).
    bool using_wl_ambient = false;
    if (wl_ambient_gene_raw != nullptr && !wl_ambient_gene_raw->empty()) {
        const auto& wg = *wl_ambient_gene_raw;
        // Check that at least one gene has counts (guard against uninitialized array)
        uint64_t wl_total_reads = 0;
        for (uint64_t v : wg) wl_total_reads += v;
        if (wl_total_reads > 0) {
            using_wl_ambient = true;
            const uint32_t wl_n = std::min(static_cast<uint32_t>(wg.size()), n_genes);
            for (uint32_t g = 0; g < wl_n; ++g) {
                ambient_profile[g] = static_cast<double>(wg[g]);
                ambient_tot       += ambient_profile[g];
            }
            // Count WL barcodes with 1 ≤ UMI ≤ wl_ambient_ceil as the ambient pool size
            if (wl_umi_counts != nullptr) {
                for (uint32_t cnt : *wl_umi_counts) {
                    if (cnt >= 1 && cnt <= wl_ambient_ceil) {
                        ++n_ambient;
                    }
                }
            } else {
                // Estimate from total reads assuming ~25 reads per ambient droplet
                n_ambient = static_cast<uint32_t>(wl_total_reads / 25u);
            }
            std::cerr << "[cell-call] WL ambient pool: n_ambient=" << n_ambient
                      << " total_wl_reads=" << wl_total_reads
                      << " n_genes_nonzero="
                      << std::count_if(ambient_profile.begin(), ambient_profile.end(),
                                       [](double v) { return v > 0.0; }) << "\n";
        }
    }

    // Always scan bc_index_ barcodes with UMI <= lower for the ambient pool.
    // When using_wl_ambient=true, this supplements the WL gene-count profile with
    // per-barcode counts from discovered low-UMI barcodes (typically ~82K barcodes
    // with 5–99 UMI in CB_UMI_Simple mode), dramatically improving gene coverage
    // from ~34% to >80%.  WL barcodes and bc_index_ barcodes are disjoint (WL
    // barcodes never reach discovery_threshold in the pileup), so addition is safe.
    // When using_wl_ambient=false, this is the primary ambient source (original path).
    //
    // Traverse sorted barcodes from LOWEST count (reverse order of sorted_bc)
    // to pick the n_ambient_max highest-count empty droplets (still <= lower).
    // Iterating in reverse gives us the highest-count empties first.
    {
        const uint32_t n_ambient_before = n_ambient;
        for (int64_t i = static_cast<int64_t>(n_barcodes) - 1; i >= 0; --i) {
            uint32_t b = sorted_bc[i];
            uint64_t c = counts_per_barcode[b];
            if (c == 0) continue;        // ignore zero-count barcodes
            if (c >  lower) continue;    // skip cells (count > lower)
            if (n_ambient >= static_cast<uint32_t>(n_ambient_max)) break;
            ++n_ambient;
            ambient_tot += static_cast<double>(c);
            int32_t start = gene_counts.indptr[b];
            int32_t end   = gene_counts.indptr[b + 1];
            for (int32_t k = start; k < end; ++k)
                ambient_profile[gene_counts.indices[k]] +=
                    static_cast<double>(gene_counts.data[k]);
        }
        if (n_ambient > n_ambient_before)
            std::cerr << "[cell-call] bc_index ambient: added "
                      << (n_ambient - n_ambient_before)
                      << " barcodes (UMI <= " << lower
                      << ") n_ambient=" << n_ambient << "\n";
    }

    // ── 2b. Knee-based ambient fallback ──────────────────────────────────────
    // When no barcodes satisfy the UMI threshold (all auto-discovered barcodes
    // are >= lower_umi), compute the biological knee of the barcode-rank curve
    // and use barcodes BELOW the knee as the ambient pool.
    //
    // The knee marks the cell/empty transition.  Barcodes below the knee are
    // the best available proxy for ambient RNA in the absence of truly low-UMI
    // barcodes (which exist in the full whitelist but not in singlify's pileup
    // matrix that only accumulates barcodes with >= discovery_threshold reads).
    //
    // Algorithm identical to call_cells_knee_fallback's inflection finder:
    //   1. log10(UMI) vs rank (sorted_bc is already descending by UMI)
    //   2. Smooth with 5-point moving average
    //   3. Find rank with most-negative smoothed second derivative → knee
    //
    // After finding knee_rank, we:
    //   - Use barcodes at sorted ranks [knee_rank+1 .. knee_rank+n_ambient_max]
    //     as the ambient pool (barcodes just below the cell/empty boundary).
    //   - Limit the test pool to barcodes at ranks [0..knee_rank] with UMI ≥
    //     min_umi_test (so only barcodes above the knee ARE tested).
    if (!using_wl_ambient && (n_ambient == 0 || ambient_tot < 1.0)) {
        // ── Count non-zero barcodes ──
        size_t nv = 0;
        for (; nv < n_barcodes; ++nv)
            if (counts_per_barcode[sorted_bc[nv]] == 0) break;

        size_t knee_rank = (nv > 0) ? nv / 2 : 0;
        uint64_t knee_umi = 0;

        if (nv >= 5) {
            // log10(UMI) for each sorted barcode
            std::vector<double> lc(nv);
            for (size_t i = 0; i < nv; ++i)
                lc[i] = std::log10(static_cast<double>(counts_per_barcode[sorted_bc[i]]));

            // 5-point moving-average smoothing
            const int hw = 2;
            std::vector<double> sm(nv);
            for (size_t i = 0; i < nv; ++i) {
                int lo = std::max(0, static_cast<int>(i) - hw);
                int hi = std::min(static_cast<int>(nv) - 1,
                                  static_cast<int>(i) + hw);
                double s = 0.0;
                for (int j = lo; j <= hi; ++j) s += lc[j];
                sm[i] = s / static_cast<double>(hi - lo + 1);
            }

            // Most-negative smoothed second derivative → inflection (knee)
            double min_d2 = 0.0;
            for (size_t i = 1; i + 1 < nv; ++i) {
                double d2 = sm[i + 1] - 2.0 * sm[i] + sm[i - 1];
                if (d2 < min_d2) { min_d2 = d2; knee_rank = i; }
            }
        }

        knee_umi = counts_per_barcode[sorted_bc[knee_rank]];

        // ── Plate-protocol guard ──────────────────────────────────────────────
        // When all barcodes are above lower (plate-based assays: sci-RNA-seq3,
        // SPLiT-seq, ddSEQ; typically nv ≤ 1000 unique barcodes) and the knee
        // falls in the top 5% of the barcode distribution, the barcodes "below
        // the knee" are NOT empty wells — they are real cells with fewer reads.
        // Using them as ambient and running MC EmptyDrops calls only 1-5 cells
        // from hundreds of real ones.  Correct action: return all barcodes above
        // the original lower_umi threshold as cells immediately.
        //
        // Trigger: nv ≤ 1000 (plate-scale total) AND knee in top 5% of nv.
        {
            const size_t plate_knee_thr = std::max(size_t(5), nv / 20);
            if (nv <= 1000 && knee_rank < plate_knee_thr) {
                CellCallResult pr;
                pr.n_ambient    = 0;
                pr.ambient_total = 0.0;
                for (size_t rank = 0; rank < nv; ++rank) {
                    uint32_t bc = sorted_bc[rank];
                    if (counts_per_barcode[bc] < lower) break;
                    pr.cell_indices.push_back(bc);
                    pr.tested_indices.push_back(bc);
                    pr.deviance.push_back(0.0);
                    pr.fdr.push_back(0.0);
                }
                std::cerr << "[cell-call] Plate-protocol fallback (nv=" << nv
                          << " knee_rank=" << knee_rank << "<" << plate_knee_thr
                          << "): called " << pr.cell_indices.size()
                          << " cells (all barcodes >= lower=" << lower << ")\n";
                return pr;
            }
        }

        // Build ambient pool from barcodes just BELOW the knee
        n_ambient   = 0;
        ambient_tot = 0.0;
        std::fill(ambient_profile.begin(), ambient_profile.end(), 0.0);
        for (size_t rank = knee_rank + 1; rank < nv; ++rank) {
            if (n_ambient >= static_cast<uint32_t>(n_ambient_max)) break;
            uint32_t b = sorted_bc[rank];
            uint64_t c = counts_per_barcode[b];
            if (c == 0) break;
            ++n_ambient;
            ambient_tot += static_cast<double>(c);
            int32_t bstart = gene_counts.indptr[b];
            int32_t bend   = gene_counts.indptr[b + 1];
            for (int32_t k = bstart; k < bend; ++k)
                ambient_profile[gene_counts.indices[k]] +=
                    static_cast<double>(gene_counts.data[k]);
        }

        std::cerr << "[cell-call] EmptyDrops: ambient pool empty (all barcodes > lower="
                  << lower << "); knee-based ambient: knee_rank=" << knee_rank
                  << " knee_umi=" << knee_umi
                  << " n_ambient=" << n_ambient << "\n";

        // Update lower to the knee UMI so only barcodes above the knee are tested
        lower = knee_umi;
    }

    if (!using_wl_ambient && (n_ambient == 0 || ambient_tot < 1.0))
        return call_cells_knee_fallback(counts_per_barcode, lower);

    // ── Ambient pool quality check: supplement with gray-zone barcodes ────────
    //
    // When the ambient pool is too thin the profile is dominated by pseudocounts
    // and the MC null is uncalibrated, causing systematic overcalling.
    //
    // TWO failure modes:
    //   1. WL ambient path: STAR uses auto_barcodes.tsv (not the full WL) as its
    //      whitelist, so WL-ambient barcodes don't receive a CB tag in the BAM.
    //      The pileup never sees them → wl_ambient_gene_counts_ ≈ 0 reads.
    //      using_wl_ambient=true (vector is non-empty) but wl_total_reads ≈ 0-10,
    //      so the profile is pure pseudocount across K=310k genes.
    //
    //   2. Non-WL auto-discovery path: the pileup matrix contains only barcodes
    //      with ≥ discovery_threshold reads.  After UMI dedup, most have ≥ 50 UMI,
    //      so barcodes with UMI ≤ lower=100 are very few (0–50).
    //
    // Fix: supplement with "gray-zone" barcodes (lower < UMI < min_umi_test).
    // These are barcodes discarded by both the ambient pool (UMI > lower) AND the
    // test pool (UMI < min_umi_test), making them currently wasted.  They are
    // empty droplets — not worth testing statistically, but ideal for estimating
    // the ambient RNA profile because they have enough signal to reliably cover
    // the ambient gene set.
    //
    // Minimum ambient pool size for reliable MC calibration:
    //   300 barcodes  (DropletUtils recommends 10 000; 300 is a safe floor for
    //                  singlify's narrower discovered-barcode set)
    //   10 000 UMI    (ensures enough reads to cover ambient genes)
    //
    // sorted_bc is sorted DESCENDING; iterating from i=n_barcodes−1 → 0 visits
    // barcodes in ASCENDING UMI order: lowest first, tested barcodes last.
    // We therefore break as soon as we see c >= gray_zone_ceil (all subsequent
    // barcodes at smaller i are also >= gray_zone_ceil).
    //
    // CRITICAL: the gray-zone upper bound is capped at lower * 2 (not
    // min_umi_test).  Barcodes above lower*2 increasingly overlap the
    // real-cell UMI distribution; adding them inflates the ambient profile
    // with cell-like signal, making the multinomial deviance test trivially
    // significant for every barcode (100% call rate → CR2 fallback).
    // Keeping the gray zone tight (lower..lower*2) ensures only true empty
    // droplets enter the ambient pool.
    static const uint32_t MIN_AMBIENT_BARCODES = 300;
    static const double   MIN_AMBIENT_UMI      = 1e4;
    if (n_ambient < MIN_AMBIENT_BARCODES || ambient_tot < MIN_AMBIENT_UMI) {
        const uint32_t n_before  = n_ambient;
        const double   tot_before = ambient_tot;
        const uint64_t gray_zone_ceil = std::min(
            static_cast<uint64_t>(min_umi_test),
            static_cast<uint64_t>(lower) * 2);
        for (int64_t i = static_cast<int64_t>(n_barcodes) - 1; i >= 0; --i) {
            const uint32_t b = sorted_bc[static_cast<size_t>(i)];
            const uint64_t c = counts_per_barcode[b];
            if (c == 0) continue;
            if (c <= lower) continue;        // already counted (or below threshold)
            if (c >= gray_zone_ceil) break;  // above safe gray zone — stop
            // c is in (lower, gray_zone_ceil): safe gray-zone empty droplet
            if (n_ambient >= static_cast<uint32_t>(n_ambient_max)) break;
            ++n_ambient;
            ambient_tot += static_cast<double>(c);
            const int32_t bstart = gene_counts.indptr[b];
            const int32_t bend   = gene_counts.indptr[b + 1];
            for (int32_t k = bstart; k < bend; ++k)
                ambient_profile[gene_counts.indices[k]] +=
                    static_cast<double>(gene_counts.data[k]);
        }
        if (n_ambient > n_before)
            std::cerr << "[cell-call] Ambient pool supplemented with gray-zone barcodes"
                      << " (lower=" << lower << " < UMI < gray_ceil=" << gray_zone_ceil
                      << "): n_ambient=" << n_before << " → " << n_ambient
                      << "  ambient_tot=" << static_cast<uint64_t>(tot_before)
                      << " → " << static_cast<uint64_t>(ambient_tot) << "\n";
    }

    // ── Final safety: if still no ambient after supplement, hard fallback ─────
    if (n_ambient == 0 || ambient_tot < 1.0)
        return call_cells_knee_fallback(counts_per_barcode, lower);

    // Normalize ambient profile with pseudocount per gene.
    //
    // pseudo = 0.1 (tuned for MC null calibration with sparse ambient).
    //
    // Reasoning: singlify's auto-discovered ambient pool typically covers
    // ~5000/38000 genes (~13%).  With pseudo=1e-4, the remaining 87% of genes
    // have probability ~2e-9 — the alias-method MC null NEVER samples them, so
    // null deviances lack any contribution from unseen genes.  Meanwhile, real
    // barcodes expressing those unseen genes get massive observed deviance
    // → 100% call rate (miscalibrated).
    //
    // With pseudo=0.3, unseen genes have cumulative probability ~0.17
    // (depending on ambient depth), and the MC null samples ~170 unseen-gene
    // hits per 1000-draw null.  This properly calibrates the null distribution:
    //   - Empty droplets (few unseen genes) → observed deviance < null → p=1
    //   - Real cells (many unseen genes) → observed deviance >> null → p≈0
    // Higher pseudo (e.g. 0.5) risks undercalling; ~29% pseudo mass overwhelms
    // the null, suppressing recall to ~3% of STARsolo.
    const double pseudo = 0.3;
    double profile_sum  = 0.0;
    for (uint32_t g = 0; g < n_genes; ++g) {
        ambient_profile[g] += pseudo;
        profile_sum         += ambient_profile[g];
    }
    for (uint32_t g = 0; g < n_genes; ++g)
        ambient_profile[g] /= profile_sum;

    // ── Ambient profile coverage quality gate ───────────────────────────────────
    // When the ambient pool has very few reads, most features have only the
    // pseudocount probability.  Any tested barcode expressing a "dark" feature
    // (p_g ≈ pseudo/profile_sum ≈ 1e-10) will have astronomical deviance → p≈0
    // → FDR≈0 → systematic overcalling.
    //
    // Empirical threshold: require ≥ 5% of features covered (p > 2× pseudocount)
    // before trusting EmptyDrops.  Below this, the MC null is uncalibrated and
    // the knee fallback (which only uses UMI totals) is more reliable.
    {
        const double dark_threshold = 2.0 * pseudo / profile_sum;
        uint32_t n_covered = 0;
        for (uint32_t g = 0; g < n_genes; ++g)
            if (ambient_profile[g] > dark_threshold) ++n_covered;
        const double coverage = static_cast<double>(n_covered) / n_genes;
        std::cerr << "[cell-call] Ambient coverage: " << n_covered << "/" << n_genes
                  << " genes (" << static_cast<int>(coverage * 100) << "%) above pseudocount\n";
        if (coverage < 0.05) {
            std::cerr << "[cell-call] Ambient coverage below 5% — EmptyDrops unreliable;"
                      << " using knee fallback\n";
            return call_cells_knee_fallback(counts_per_barcode, lower);
        }
    }

    // Precompute log(ambient_profile[g]) to avoid repeated log in tight loop
    std::vector<double> log_ambient(n_genes);
    for (uint32_t g = 0; g < n_genes; ++g)
        log_ambient[g] = std::log(ambient_profile[g]);

    // ── 4. Compute observed deviances for all candidate barcodes ────────────────
    std::vector<uint32_t> tested_indices;
    std::vector<double>   deviances;
    tested_indices.reserve(n_barcodes / 2);
    deviances.reserve(n_barcodes / 2);

    for (uint32_t b = 0; b < n_barcodes; ++b) {
        uint64_t N = counts_per_barcode[b];
        if (N <= lower) continue;         // ambient or below threshold — skip
        if (N < min_umi_test) continue;   // below STARsolo umiMin floor — skip

        tested_indices.push_back(b);

        int32_t start = gene_counts.indptr[b];
        int32_t end   = gene_counts.indptr[b + 1];

        // Multinomial deviance = 2 * Σ_{g: n_g>0} n_g * [log(n_g) − log(N) − log(p_g)]
        double dev = detail::compute_deviance(
            gene_counts.indices.data(), gene_counts.data.data(),
            start, end, log_ambient, N);

        deviances.push_back(dev);
    }

    // ── 5. Monte Carlo p-values (depth-matched null distribution) ───────────────
    // Replaces chi-squared LRT: generates null deviances at matched UMI depth
    // for each log10-UMI bin. This prevents overcalling of deep ambient droplets.
    std::vector<double> pvals = detail::mc_emptydrops_pvalues(
        tested_indices, deviances, counts_per_barcode,
        ambient_profile, log_ambient, n_monte_carlo);

    std::cerr << "[cell-call] Monte Carlo EmptyDrops: tested=" << tested_indices.size()
              << " n_mc=" << n_monte_carlo << "\n";

    // ── 6. BH FDR correction ─────────────────────────────────────────────────
    std::vector<double> fdr_vals;
    detail::bh_fdr(pvals, fdr_vals);

    // ── 7. Collect results ───────────────────────────────────────────────────
    CellCallResult result;
    result.n_ambient     = n_ambient;
    result.ambient_total = ambient_tot;
    result.deviance      = std::move(deviances);
    result.fdr           = fdr_vals;
    result.tested_indices = tested_indices;
    for (size_t i = 0; i < tested_indices.size(); ++i) {
        if (fdr_vals[i] < fdr_threshold)
            result.cell_indices.push_back(tested_indices[i]);
    }

    return result;
}

// ── I/O helper ────────────────────────────────────────────────────────────────

/// Write per-barcode cell call statistics to TSV.
/// Columns: barcode, total_umi, deviance, fdr, is_cell
inline void write_cell_calls(
    const std::string&           outpath,
    const CellCallResult&        result,
    const std::vector<std::string>& barcodes,
    const std::vector<uint64_t>& counts_per_barcode,
    double fdr_threshold = 0.01,
    const std::string& call_method = "emptydrops")
{
    // Build a set of called-cell barcode indices so that fallback overrides
    // (CR2, top-N, knee) are correctly reflected in the is_cell column even
    // when EmptyDrops FDR values remain significant for all tested barcodes.
    std::unordered_set<uint32_t> cell_set(
        result.cell_indices.begin(), result.cell_indices.end());

    std::ofstream out(outpath);
    if (!out.is_open()) {
        std::cerr << "[cell_calling] ERROR: cannot write " << outpath << "\n";
        return;
    }
    out << "barcode\tis_cell\tlog10_umi\tempty_drops_pvalue\tcall_method\n";
    for (size_t i = 0; i < result.tested_indices.size(); ++i) {
        uint32_t bc = result.tested_indices[i];
        double log10_umi = counts_per_barcode[bc] > 0
            ? std::log10(static_cast<double>(counts_per_barcode[bc])) : 0.0;
        out << barcodes[bc]            << '\t'
            << (cell_set.count(bc) ? "TRUE" : "FALSE") << '\t'
            << std::fixed << std::setprecision(4) << log10_umi << '\t'
            << std::scientific << std::setprecision(4) << result.fdr[i] << '\t'
            << call_method << '\n';
    }
}

} // namespace singlet
