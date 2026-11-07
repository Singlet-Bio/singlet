// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: hto_demux.h
// T3: HTO (hashtag oligonucleotide) demultiplexing for CITE-seq experiments.
//
// Caller: DropletUtils::hashedDrops-style fold-change rule plus a mixture-model doublet veto, chosen by
// leave-one-dataset-out on the Howitt et al. 2023 genotype-truth benchmark (bench/adt/hto_lodo.py, which holds the numpy
// reference this file reproduces call for call).
//
//   1. Ambient per tag, from the called cells alone: the tag's log1p counts are split by 1-D 2-means and the ambient level is
//      the mean count of the lower group (ambientProfileBimodal). Counts are divided by the ambient proportions.
//   2. Per cell, on the ambient-scaled counts: lfc_top = log2((top + 5) / (second + 5)) and
//      lfc_second = log2((second + 5) / (median of the other tags + 5)).
//        DOUBLET    lfc_second is an upper outlier (> median + doublet_nmads * MAD over cells) and >= doublet_min
//        SINGLET    lfc_top >= confident_min and not a lower outlier (< median - confident_nmads * MAD among non-doublets)
//        NEGATIVE   otherwise (unassigned)
//   3. Veto: each tag's log1p counts are fitted with a two-component Gaussian mixture (EM, deterministic start); a cell with
//      two or more tags whose posterior of the high component is >= mixture_positive is a DOUBLET.
//
// Determinism: every statistic is computed over cells in a canonical order (lexicographic on the cell's count vector), so the
// result does not depend on the barcode order, and the code is single-threaded and free of random numbers.
// Fewer than min_cells cells (or a single tag): the statistics above are undefined; a plain proportion rule is used (a tag is
// positive when its count is >= 5 and >= 30% of the cell's largest).
//
// Input:  SparseAccumulator<uint32_t> in (tag x cell) layout (from AdtCounter).
// Output: vector<HtoDemuxResult>, one entry per cell (same order as barcodes).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "sparse_accumulator.h"

namespace singlet {

// ── Result type ───────────────────────────────────────────────────────────────

struct HtoDemuxResult {
    std::string classification;  ///< "SINGLET", "DOUBLET", or "NEGATIVE"
    std::string assignment;      ///< HTO name for singlets; empty otherwise
    float max_clr;               ///< highest CLR value across all HTOs for this cell
    float margin = 0.0f;         ///< 1 - 2^-lfc_top in [0, 1]: (top - second) / top on ambient-scaled counts
};

/// Settings of the caller. The defaults are the ones chosen by leave-one-dataset-out (see bench/adt/hto_lodo.py).
struct HtoDemuxParams {
    double confident_min = 0.5;     ///< log2 fold between the top two tags for a confident singlet
    double doublet_min = 1.0;       ///< log2 fold of the second tag over ambient for a doublet
    double doublet_nmads = 3.0;
    double confident_nmads = 3.0;
    double pseudo = 5.0;
    double mixture_positive = 0.5;  ///< posterior for a tag to count as positive in the doublet veto
    size_t min_cells = 20;
    int em_iters = 200;
    double em_tol = 1e-9;
};

namespace hto_detail {

inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/// median and 1.4826 * MAD
inline void median_mad(const std::vector<double>& v, double& med, double& mad) {
    med = median_of(v);
    std::vector<double> d(v.size());
    for (size_t i = 0; i < v.size(); ++i) d[i] = std::fabs(v[i] - med);
    mad = 1.4826 * median_of(std::move(d));
}

inline bool allclose2(double a0, double a1, double b0, double b1) {
    return std::fabs(a0 - b0) <= 1e-8 + 1e-5 * std::fabs(b0) && std::fabs(a1 - b1) <= 1e-8 + 1e-5 * std::fabs(b1);
}

/// Ambient level of one tag from the called cells (ambientProfileBimodal).
inline double ambient_bimodal(const std::vector<double>& row) {
    const size_t n = row.size();
    std::vector<double> v(n);
    double lo = std::numeric_limits<double>::infinity(), hi = -lo, mean = 0;
    for (size_t i = 0; i < n; ++i) {
        v[i] = std::log1p(row[i]);
        lo = std::min(lo, v[i]);
        hi = std::max(hi, v[i]);
        mean += row[i];
    }
    mean /= static_cast<double>(n);
    if (hi - lo < 1e-9) return mean;
    double c0 = lo, c1 = hi;
    std::vector<char> lab(n, 0);
    for (int it = 0; it < 100; ++it) {
        double s0 = 0, s1 = 0;
        size_t n0 = 0, n1 = 0;
        for (size_t i = 0; i < n; ++i) {
            lab[i] = std::fabs(v[i] - c1) < std::fabs(v[i] - c0) ? 1 : 0;
            if (lab[i]) { s1 += v[i]; ++n1; } else { s0 += v[i]; ++n0; }
        }
        const double m0 = n0 ? s0 / static_cast<double>(n0) : c0;
        const double m1 = n1 ? s1 / static_cast<double>(n1) : c1;
        if (allclose2(m0, m1, c0, c1)) break;
        c0 = m0;
        c1 = m1;
    }
    const char low = (c1 < c0) ? 1 : 0;  // argmin(c): index 0 on ties
    double s = 0;
    size_t cnt = 0;
    for (size_t i = 0; i < n; ++i)
        if (lab[i] == low) { s += row[i]; ++cnt; }
    return cnt ? s / static_cast<double>(cnt) : mean;
}

/// Posterior of the high component of a two-component Gaussian mixture of y (EM). Cells with equal y get equal posteriors.
inline std::vector<double> mixture_high(const std::vector<double>& y, int iters, double tol) {
    const size_t n = y.size();
    std::vector<double> ys(y);
    std::sort(ys.begin(), ys.end());
    auto quant = [&](double p) {
        const double pos = static_cast<double>(n - 1) * p;
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(lo + 1, n - 1);
        return ys[lo] + (ys[hi] - ys[lo]) * (pos - static_cast<double>(lo));
    };
    double c0 = quant(0.25), c1 = quant(0.95);
    std::vector<char> lab(n, 0);
    for (int it = 0; it < 50; ++it) {
        double s0 = 0, s1 = 0;
        size_t n0 = 0, n1 = 0;
        for (size_t i = 0; i < n; ++i) {
            lab[i] = std::fabs(y[i] - c1) < std::fabs(y[i] - c0) ? 1 : 0;
            if (lab[i]) { s1 += y[i]; ++n1; } else { s0 += y[i]; ++n0; }
        }
        const double m0 = n0 ? s0 / static_cast<double>(n0) : c0;
        const double m1 = n1 ? s1 / static_cast<double>(n1) : c1;
        if (allclose2(m0, m1, c0, c1)) break;
        c0 = m0;
        c1 = m1;
    }
    const char hi_lab = (c1 > c0) ? 1 : 0;  // argmax(c): index 0 on ties
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = std::min(std::max(lab[i] == hi_lab ? 1.0 : 0.0, 0.02), 0.98);
    double ll_old = -std::numeric_limits<double>::infinity();
    double mu[2] = {0, 0};
    const double kLog2Pi = std::log(2.0 * 3.14159265358979323846);
    for (int it = 0; it < iters; ++it) {
        double sw[2], b[2], var[2];
        for (int m = 0; m < 2; ++m) {
            double a = 0, w_sum = 0;
            for (size_t i = 0; i < n; ++i) {
                const double w = (m ? r[i] : 1.0 - r[i]) + 1e-9;
                a += w * y[i];
                w_sum += w;
            }
            b[m] = a / w_sum;
            double ss = 0;
            for (size_t i = 0; i < n; ++i) {
                const double w = (m ? r[i] : 1.0 - r[i]) + 1e-9;
                const double e = y[i] - b[m];
                ss += w * e * e;
            }
            var[m] = std::max(ss / w_sum, 1e-4);
            sw[m] = w_sum;
            mu[m] = b[m];
        }
        const double lp0 = std::log(std::max(sw[0] / static_cast<double>(n), 1e-9)) - 0.5 * (kLog2Pi + std::log(var[0]));
        const double lp1 = std::log(std::max(sw[1] / static_cast<double>(n), 1e-9)) - 0.5 * (kLog2Pi + std::log(var[1]));
        double ll = 0;
        for (size_t i = 0; i < n; ++i) {
            const double e0 = y[i] - b[0], e1 = y[i] - b[1];
            const double d0 = lp0 - 0.5 * e0 * e0 / var[0];
            const double d1 = lp1 - 0.5 * e1 * e1 / var[1];
            const double mx = std::max(d0, d1);
            const double x0 = std::exp(d0 - mx), x1 = std::exp(d1 - mx);
            ll += mx + std::log(x0 + x1);
            r[i] = x1 / (x0 + x1);
        }
        if (std::fabs(ll - ll_old) < tol * std::max(1.0, std::fabs(ll))) break;
        ll_old = ll;
    }
    if (mu[1] < mu[0])
        for (auto& v : r) v = 1.0 - v;
    return r;
}

}  // namespace hto_detail

// ── HtoDemux ──────────────────────────────────────────────────────────────────

class HtoDemux {
   public:
    HtoDemux() = default;
    explicit HtoDemux(const HtoDemuxParams& p) : p_(p) {}

    // ── Demux ──────────────────────────────────────────────────────────────

    /// Demultiplex cells using the HTO count matrix.
    ///
    /// @param hto_counts  SparseAccumulator<uint32_t> (tags x cells).
    ///                    hto_counts.n_features() == tag_names.size().
    /// @param tag_names   HTO tag names (length == number of rows).
    /// @param n_cells     Number of cells (number of columns in matrix).
    /// @returns           One HtoDemuxResult per cell (length == n_cells).
    std::vector<HtoDemuxResult> demux(
        const SparseAccumulator<uint32_t>& hto_counts,
        const std::vector<std::string>& tag_names,
        size_t n_cells) const {
        const size_t k = tag_names.size();
        std::vector<HtoDemuxResult> results(n_cells, HtoDemuxResult{"NEGATIVE", "", 0.0f});
        if (n_cells == 0 || k == 0) return results;

        // dense counts, cell-major: cnt[c * k + t]
        std::vector<uint32_t> cnt(n_cells * k, 0u);
        {
            auto csc = hto_counts.to_csc();
            for (uint32_t col = 0; col < static_cast<uint32_t>(n_cells); ++col)
                for (int32_t j = csc.indptr[col]; j < csc.indptr[col + 1]; ++j) {
                    const uint32_t row = static_cast<uint32_t>(csc.indices[j]);
                    if (row < k) cnt[col * k + row] = static_cast<uint32_t>(csc.data[j]);
                }
        }

        // canonical cell order: lexicographic on the count vector (equal vectors are interchangeable)
        std::vector<uint32_t> order(n_cells);
        std::iota(order.begin(), order.end(), 0u);
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            return std::lexicographical_compare(cnt.begin() + a * k, cnt.begin() + (a + 1) * k, cnt.begin() + b * k,
                                                cnt.begin() + (b + 1) * k);
        });

        // tag-major doubles in canonical order: X[t][i]
        std::vector<std::vector<double>> X(k, std::vector<double>(n_cells));
        for (size_t i = 0; i < n_cells; ++i)
            for (size_t t = 0; t < k; ++t) X[t][i] = static_cast<double>(cnt[order[i] * k + t]);

        std::vector<int> call(n_cells, -1);
        std::vector<double> lfc_top(n_cells, 0.0);
        if (n_cells < p_.min_cells || k < 2)
            proportion_rule(X, call);
        else
            hashed_drops_with_veto(X, call, lfc_top);

        for (size_t i = 0; i < n_cells; ++i) {
            HtoDemuxResult& r = results[order[i]];
            const uint32_t* row = &cnt[order[i] * k];
            double lsum = 0, mx = -1e300;
            for (size_t t = 0; t < k; ++t) lsum += std::log(static_cast<double>(row[t]) + 1.0);
            for (size_t t = 0; t < k; ++t)
                mx = std::max(mx, std::log(static_cast<double>(row[t]) + 1.0) - lsum / static_cast<double>(k));
            r.max_clr = static_cast<float>(mx);
            if (call[i] >= 0) {
                r.classification = "SINGLET";
                r.assignment = tag_names[static_cast<size_t>(call[i])];
                r.margin = static_cast<float>(std::min(std::max(1.0 - std::pow(2.0, -lfc_top[i]), 0.0), 1.0));
            } else if (call[i] == kDoublet) {
                r.classification = "DOUBLET";
            }
        }
        return results;
    }

    // ── Output ────────────────────────────────────────────────────────────

    /// Write a TSV with columns: barcode, classification, assignment, max_clr
    void write_tsv(const std::string& path,
                   const std::vector<std::string>& barcodes,
                   const std::vector<HtoDemuxResult>& results) const {
        std::ofstream ofs(path);
        ofs << "barcode\tclassification\tassignment\tmax_clr\n";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            std::string bc = (i < barcodes.size()) ? barcodes[i] : std::to_string(i);
            ofs << bc << '\t'
                << r.classification << '\t'
                << r.assignment << '\t'
                << r.max_clr << '\n';
        }
    }

   private:
    static constexpr int kDoublet = -2;
    HtoDemuxParams p_;

    static void proportion_rule(const std::vector<std::vector<double>>& X, std::vector<int>& call) {
        const size_t k = X.size(), n = X[0].size();
        for (size_t i = 0; i < n; ++i) {
            double mx = 0;
            for (size_t t = 0; t < k; ++t) mx = std::max(mx, X[t][i]);
            int npos = 0, best = -1;
            for (size_t t = 0; t < k; ++t)
                if (X[t][i] >= 5.0 && X[t][i] >= 0.3 * mx) {
                    ++npos;
                    if (best < 0 || X[t][i] > X[static_cast<size_t>(best)][i]) best = static_cast<int>(t);
                }
            call[i] = npos == 1 ? best : (npos >= 2 ? kDoublet : -1);
        }
    }

    void hashed_drops_with_veto(const std::vector<std::vector<double>>& X, std::vector<int>& call,
                                std::vector<double>& lfc_top_out) const {
        using namespace hto_detail;
        const size_t k = X.size(), n = X[0].size();

        // ambient proportions
        std::vector<double> amb(k);
        double asum = 0;
        for (size_t t = 0; t < k; ++t) {
            amb[t] = ambient_bimodal(X[t]) + 1e-9;
            asum += amb[t];
        }
        for (auto& a : amb) a /= asum;

        std::vector<double> lfc_second(n), adj(k), rest;
        std::vector<int> top_tag(n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t t = 0; t < k; ++t) adj[t] = X[t][i] / (amb[t] * static_cast<double>(k));
            std::vector<size_t> o(k);
            std::iota(o.begin(), o.end(), size_t{0});
            std::stable_sort(o.begin(), o.end(), [&](size_t a, size_t b) { return adj[a] > adj[b]; });
            const double top = adj[o[0]], sec = adj[o[1]];
            double lvl = 0;
            if (k > 2) {
                rest.assign(k - 2, 0.0);
                for (size_t j = 2; j < k; ++j) rest[j - 2] = adj[o[j]];
                lvl = median_of(rest);
            }
            lfc_top_out[i] = std::log2((top + p_.pseudo) / (sec + p_.pseudo));
            lfc_second[i] = std::log2((sec + p_.pseudo) / (lvl + p_.pseudo));
            top_tag[i] = static_cast<int>(o[0]);
        }

        double md, mad;
        median_mad(lfc_second, md, mad);
        std::vector<char> dbl(n);
        std::vector<double> top_nd;
        for (size_t i = 0; i < n; ++i) {
            dbl[i] = (lfc_second[i] > md + p_.doublet_nmads * mad) && (lfc_second[i] >= p_.doublet_min);
            if (!dbl[i]) top_nd.push_back(lfc_top_out[i]);
        }
        double mt = 0, madt = 0;
        if (!top_nd.empty()) median_mad(top_nd, mt, madt);
        for (size_t i = 0; i < n; ++i) {
            if (dbl[i])
                call[i] = kDoublet;
            else if (lfc_top_out[i] >= p_.confident_min && lfc_top_out[i] >= mt - p_.confident_nmads * madt)
                call[i] = top_tag[i];
            else
                call[i] = -1;
        }

        // doublet veto: two or more positive tags in the per-tag mixtures
        std::vector<int> npos(n, 0);
        std::vector<double> y(n);
        for (size_t t = 0; t < k; ++t) {
            double mn = 1e300, mx = -1e300, mean = 0;
            for (size_t i = 0; i < n; ++i) {
                y[i] = std::log1p(X[t][i]);
                mn = std::min(mn, y[i]);
                mx = std::max(mx, y[i]);
            }
            if (!(mx > mn)) continue;
            const std::vector<double> q = mixture_high(y, p_.em_iters, p_.em_tol);
            for (size_t i = 0; i < n; ++i)
                if (q[i] >= p_.mixture_positive) ++npos[i];
            (void)mean;
        }
        for (size_t i = 0; i < n; ++i)
            if (npos[i] >= 2) call[i] = kDoublet;
    }
};

}  // namespace singlet
