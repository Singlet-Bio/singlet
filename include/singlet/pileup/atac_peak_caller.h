// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_peak_caller.h
// Peak calling on Tn5 insertion sites, the MACS2 recipe used for single-cell ATAC by Signac and ArchR
// (`macs2 callpeak -f BED --nomodel --shift -100 --extsize 200 --keep-dup all`, no control):
//
//   * every insertion (both cut sites of every fragment) is a 200 bp tag centred on the cut site;
//   * pileup at x = insertions within [x - 100, x + 100);
//   * local lambda = max(genome background, insertions within +-5 kb scaled to 200 bp) (MACS2 llocal 10 kb,
//     no slocal without control);
//   * Poisson upper-tail p per position, Benjamini-Hochberg q over all evaluated positions (MACS2 p-score -> q-score
//     table), regions with q below the cutoff, merged across gaps <= max_gap, kept if >= min_length;
//   * summit = position of maximum pileup.
//
// Positions are evaluated on a grid (default step 10 bp, centre of each step), which changes region ends by at most
// one step compared with MACS2's per-base segments.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace singlet {

struct AtacPeak {
    int32_t chrom_idx = 0;
    int32_t start = 0;  // 0-based
    int32_t end = 0;    // exclusive
    int32_t summit = 0;  // absolute position
    float pileup = 0;    // insertions in the 200 bp window at the summit
    float neg_log10_p = 0;
    float neg_log10_q = 0;
};

struct InsertionPeakParams {
    int32_t ext = 200;           // tag width (shift -100, extsize 200)
    int32_t step = 10;           // grid step
    int32_t llocal = 10000;      // local lambda window
    double q_cutoff = 0.05;      // BH q-value cutoff (MACS2 default; best peak F1 on Multiome 3k development data)
    int32_t min_length = 200;    // MACS2 default min length = extsize
    int32_t max_gap = 30;        // merge regions closer than this
    double genome_size = 0;      // effective genome size; 0 = 0.87 x sum of chromosome lengths given
};

namespace detail {

// -log10 P(X >= k), X ~ Poisson(lambda), for k > lambda.
inline double poisson_upper_neg_log10(uint32_t k, double lambda) {
    if (k == 0) return 0.0;
    if (lambda <= 0) return 300.0;
    const double log_t0 = -lambda + k * std::log(lambda) - std::lgamma(static_cast<double>(k) + 1.0);
    double sum = 1.0, term = 1.0;
    for (uint32_t i = k + 1; i < k + 100000; ++i) {
        term *= lambda / static_cast<double>(i);
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    const double log_p = log_t0 + std::log(sum);
    return std::max(0.0, -log_p / std::log(10.0));
}

}  // namespace detail

/// insertions[c] = cut-site positions on chromosome c (any order; sorted in place). chrom_len[c] = length.
/// skip[c] = true excludes the chromosome (chrM, decoys). Returns peaks sorted by (chrom, start).
inline std::vector<AtacPeak> call_peaks_insertions(std::vector<std::vector<int32_t>>& insertions,
                                                   const std::vector<int32_t>& chrom_len,
                                                   const std::vector<bool>& skip,
                                                   const InsertionPeakParams& prm = {}) {
    const size_t nc = insertions.size();
    const int32_t half = prm.ext / 2, lhalf = prm.llocal / 2;
    uint64_t total_ins = 0;
    double glen = 0;
    for (size_t c = 0; c < nc; ++c) {
        if (skip[c]) continue;
        std::sort(insertions[c].begin(), insertions[c].end());
        total_ins += insertions[c].size();
        glen += chrom_len[c];
    }
    const double gsize = prm.genome_size > 0 ? prm.genome_size : 0.87 * glen;
    const double lambda_bg = gsize > 0 ? static_cast<double>(total_ins) * prm.ext / gsize : 0;

    // Pass 1: score every grid position, keep positions above a floor, histogram all scores (0.01 bins).
    constexpr double kFloor = 1.0;          // keep -log10 p >= 1 (q cutoffs are always stricter)
    constexpr int kBins = 100000;           // up to -log10 p = 1000
    std::vector<uint64_t> hist(kBins + 1, 0);
    struct Cand { int32_t chrom, grid; float score; uint32_t k; };
    std::vector<Cand> cand;
    uint64_t n_positions = 0;

    for (size_t c = 0; c < nc; ++c) {
        if (skip[c]) continue;
        const auto& v = insertions[c];
        const int32_t L = chrom_len[c];
        const int32_t ng = (L + prm.step - 1) / prm.step;
        n_positions += static_cast<uint64_t>(ng);
        size_t lo = 0, hi = 0, llo = 0, lhi = 0;
        uint64_t zero_bin = 0;
        for (int32_t g = 0; g < ng; ++g) {
            const int32_t x = g * prm.step + prm.step / 2;
            while (hi < v.size() && v[hi] < x + half) ++hi;
            while (lo < hi && v[lo] < x - half) ++lo;
            const uint32_t k = static_cast<uint32_t>(hi - lo);
            if (k == 0) { ++zero_bin; continue; }
            while (lhi < v.size() && v[lhi] < x + lhalf) ++lhi;
            while (llo < lhi && v[llo] < x - lhalf) ++llo;
            const double lam = std::max(lambda_bg, static_cast<double>(lhi - llo) * prm.ext / prm.llocal);
            if (k <= lam) { ++zero_bin; continue; }
            const double s = detail::poisson_upper_neg_log10(k, lam);
            const int b = std::min(kBins, static_cast<int>(s * 100.0));
            ++hist[b];
            if (s >= kFloor) cand.push_back({static_cast<int32_t>(c), g, static_cast<float>(s), k});
        }
        hist[0] += zero_bin;
    }

    // BH: raw -log10 q at p-score bin b = s + log10(n(score >= b) / n_positions). The adjusted q at score s is the
    // minimum q over all scores <= s (larger p), i.e. -log10 q(s) = max of the raw values over bins <= b.
    const double q_need = -std::log10(prm.q_cutoff);
    std::vector<float> qscore_of_bin(kBins + 1, 0.f);
    {
        std::vector<double> raw(kBins + 1, -1.0);
        uint64_t n_ge = 0;
        for (int b = kBins; b >= 0; --b) {
            n_ge += hist[b];
            if (hist[b] == 0) continue;
            raw[b] = b / 100.0 + std::log10(static_cast<double>(n_ge)) - std::log10(static_cast<double>(n_positions));
        }
        double run = 0.0;
        for (int b = 0; b <= kBins; ++b) {
            if (raw[b] > run) run = raw[b];
            qscore_of_bin[b] = static_cast<float>(run);
        }
    }
    double s_thresh = 1e9;
    for (int b = 0; b <= kBins; ++b)
        if (qscore_of_bin[b] >= q_need) { s_thresh = b / 100.0; break; }
    s_thresh = std::max(s_thresh, kFloor);

    // Pass 2: merge passing grid positions into regions.
    std::vector<AtacPeak> peaks;
    auto flush = [&](AtacPeak& p, bool open) {
        if (open && p.end - p.start >= prm.min_length) peaks.push_back(p);
    };
    AtacPeak cur;
    bool open = false;
    int32_t last_grid_end = 0;
    for (const auto& cd : cand) {
        if (cd.score < s_thresh) continue;
        const int32_t gs = cd.grid * prm.step;
        const int32_t ge = std::min(gs + prm.step, chrom_len[static_cast<size_t>(cd.chrom)]);
        const float qs = qscore_of_bin[std::min(kBins, static_cast<int>(cd.score * 100.0f))];
        if (open && cd.chrom == cur.chrom_idx && gs - last_grid_end <= prm.max_gap) {
            cur.end = ge;
            if (cd.k > cur.pileup) {
                cur.pileup = static_cast<float>(cd.k);
                cur.summit = gs + prm.step / 2;
            }
            cur.neg_log10_p = std::max(cur.neg_log10_p, cd.score);
            cur.neg_log10_q = std::max(cur.neg_log10_q, qs);
        } else {
            flush(cur, open);
            cur = AtacPeak{cd.chrom, gs, ge, gs + prm.step / 2, static_cast<float>(cd.k), cd.score, qs};
            open = true;
        }
        last_grid_end = ge;
    }
    flush(cur, open);
    return peaks;
}

/// narrowPeak-like BED: chrom start end name score . pileup -log10p -log10q summit_offset
inline bool write_insertion_peaks_bed(const std::string& path, const std::vector<AtacPeak>& peaks,
                                      const std::vector<std::string>& chrom_names) {
    std::ofstream out(path);
    if (!out) return false;
    for (size_t i = 0; i < peaks.size(); ++i) {
        const auto& p = peaks[i];
        out << chrom_names[static_cast<size_t>(p.chrom_idx)] << '\t' << p.start << '\t' << p.end << "\tpeak_" << (i + 1)
            << '\t' << std::min(1000, static_cast<int>(p.neg_log10_q * 10)) << "\t.\t" << p.pileup << '\t'
            << p.neg_log10_p << '\t' << p.neg_log10_q << '\t' << (p.summit - p.start) << '\n';
    }
    return static_cast<bool>(out);
}

}  // namespace singlet
