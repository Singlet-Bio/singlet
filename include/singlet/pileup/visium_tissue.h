// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: visium_tissue.h
// Visium spot barcodes -> array coordinates, and tissue detection from counts.
//
// Space Ranger decides which spots are under tissue from the H&E / fluorescence image. GEO depositions often lack the
// image or the Space Ranger outputs, so Singlet detects tissue from the library itself: spots under tissue capture far
// more molecules than spots that only see diffused ambient RNA. Model:
//   1. x = log10(UMIs + 1) per spot; a two-component Gaussian mixture (EM) gives each spot a tissue probability p.
//   2. Spatial smoothing on the hexagonal array (neighbours (r, c +- 2), (r +- 1, c +- 1)): mean-field updates
//      logit q = logit p + beta * sum_neighbours (2 q_j - 1), the posterior of an Ising prior that tissue is contiguous.
//   3. in_tissue = q >= 0.5.
// The coordinate list ("barcode<TAB>array_row<TAB>array_col", header optional) is the slide's spot list (visium-v1:
// 4,992 spots). Barcodes are matched exactly, else to the unique whitelist spot at Hamming distance 1.

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

struct VisiumSlide {
    std::vector<std::string> barcode;
    std::vector<int32_t> row, col;
    std::unordered_map<std::string, uint32_t> index;

    bool load(const std::string& path) {
        std::ifstream in(path);
        if (!in) return false;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string b, r, c;
            if (!std::getline(ss, b, '\t') || !std::getline(ss, r, '\t') || !std::getline(ss, c, '\t')) continue;
            if (r.empty() || !(std::isdigit(static_cast<unsigned char>(r[0])) || r[0] == '-')) continue;  // header
            auto dash = b.find('-');
            if (dash != std::string::npos) b.resize(dash);
            index.emplace(b, static_cast<uint32_t>(barcode.size()));
            barcode.push_back(b);
            row.push_back(std::atoi(r.c_str()));
            col.push_back(std::atoi(c.c_str()));
        }
        return !barcode.empty();
    }
    size_t size() const { return barcode.size(); }

    /// Spot index for a barcode (any "-1" suffix ignored): exact, else unique Hamming-1 neighbour, else -1.
    int64_t match(std::string b) const {
        auto dash = b.find('-');
        if (dash != std::string::npos) b.resize(dash);
        auto it = index.find(b);
        if (it != index.end()) return it->second;
        static const char kB[4] = {'A', 'C', 'G', 'T'};
        int64_t hit = -1;
        for (size_t i = 0; i < b.size(); ++i) {
            const char o = b[i];
            for (char x : kB) {
                if (x == o) continue;
                b[i] = x;
                auto j = index.find(b);
                if (j != index.end()) {
                    if (hit >= 0 && hit != j->second) return -1;  // ambiguous
                    hit = j->second;
                }
            }
            b[i] = o;
        }
        return hit;
    }
};

struct TissueCall {
    std::vector<uint8_t> in_tissue;
    std::vector<float> prob;  // smoothed tissue probability
    double mean_low = 0, mean_high = 0, sd_low = 0, sd_high = 0, weight_high = 0;
    uint32_t n_tissue = 0;
    bool separated = false;  // the two components are > 2 pooled SD apart
};

/// umis[i] = UMIs of spot i (slide order). beta = smoothing strength.
inline TissueCall detect_tissue(const VisiumSlide& slide, const std::vector<uint64_t>& umis, double beta = 1.0,
                                int smooth_iter = 20, double logit_cap = 4.0) {
    TissueCall r;
    const size_t n = slide.size();
    r.in_tissue.assign(n, 0);
    r.prob.assign(n, 0.f);
    if (n == 0) return r;
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) x[i] = std::log10(static_cast<double>(umis[i]) + 1.0);
    // Initialise by an Otsu split.
    std::vector<double> s = x;
    std::sort(s.begin(), s.end());
    std::vector<double> pre(n + 1, 0);
    for (size_t i = 0; i < n; ++i) pre[i + 1] = pre[i] + s[i];
    double best = -1, thr = s[n / 2];
    for (size_t i = 1; i < n; ++i) {
        if (s[i] == s[i - 1]) continue;
        const double w0 = double(i) / n, w1 = 1 - w0, m0 = pre[i] / i, m1 = (pre[n] - pre[i]) / (n - i);
        const double v = w0 * w1 * (m0 - m1) * (m0 - m1);
        if (v > best) { best = v; thr = 0.5 * (s[i - 1] + s[i]); }
    }
    std::vector<double> p(n);
    for (size_t i = 0; i < n; ++i) p[i] = x[i] > thr ? 1.0 : 0.0;
    const double kPi = 3.14159265358979323846;
    for (int it = 0; it < 200; ++it) {
        double w1 = 0, s0 = 0, s1 = 0;
        for (size_t i = 0; i < n; ++i) { w1 += p[i]; s1 += p[i] * x[i]; s0 += (1 - p[i]) * x[i]; }
        const double w0 = n - w1;
        if (w1 < 1 || w0 < 1) break;
        const double m0 = s0 / w0, m1 = s1 / w1;
        double v0 = 0, v1 = 0;
        for (size_t i = 0; i < n; ++i) { v1 += p[i] * (x[i] - m1) * (x[i] - m1); v0 += (1 - p[i]) * (x[i] - m0) * (x[i] - m0); }
        const double sd0 = std::sqrt(std::max(v0 / w0, 1e-4)), sd1 = std::sqrt(std::max(v1 / w1, 1e-4));
        const double pi1 = w1 / n;
        double change = 0;
        for (size_t i = 0; i < n; ++i) {
            const double z0 = (x[i] - m0) / sd0, z1 = (x[i] - m1) / sd1;
            const double l0 = std::log(1 - pi1) - 0.5 * z0 * z0 - std::log(sd0 * std::sqrt(2 * kPi));
            const double l1 = std::log(pi1) - 0.5 * z1 * z1 - std::log(sd1 * std::sqrt(2 * kPi));
            const double q = 1.0 / (1.0 + std::exp(l0 - l1));
            change = std::max(change, std::abs(q - p[i]));
            p[i] = q;
        }
        r.mean_low = m0; r.mean_high = m1; r.sd_low = sd0; r.sd_high = sd1; r.weight_high = pi1;
        if (change < 1e-6) break;
    }
    if (r.mean_high < r.mean_low) {
        for (auto& v : p) v = 1 - v;
        std::swap(r.mean_low, r.mean_high);
        std::swap(r.sd_low, r.sd_high);
        r.weight_high = 1 - r.weight_high;
    }
    // Monotone in UMIs: a spot with more UMIs than a confident tissue spot is not "background" by the mixture tail.
    for (size_t i = 0; i < n; ++i) if (x[i] >= r.mean_high) p[i] = std::max(p[i], 0.5 + 0.5 * p[i]);
    const double pooled = std::sqrt(0.5 * (r.sd_low * r.sd_low + r.sd_high * r.sd_high));
    r.separated = (r.mean_high - r.mean_low) > 2 * pooled && r.weight_high > 0 && r.weight_high < 1;
    if (!r.separated) {
        // One population: the slide is either fully covered by tissue or empty. Decide by its level.
        std::vector<double> u(x);
        std::nth_element(u.begin(), u.begin() + static_cast<long>(n / 2), u.end());
        const bool covered = u[n / 2] >= std::log10(501.0);
        for (size_t i = 0; i < n; ++i) {
            r.in_tissue[i] = covered;
            r.prob[i] = covered ? 1.f : 0.f;
        }
        r.n_tissue = covered ? static_cast<uint32_t>(n) : 0;
        return r;
    }

    // Hex neighbours.
    std::unordered_map<int64_t, uint32_t> at;
    auto key = [](int32_t a, int32_t b) { return (static_cast<int64_t>(a) << 32) ^ static_cast<uint32_t>(b); };
    for (size_t i = 0; i < n; ++i) at[key(slide.row[i], slide.col[i])] = static_cast<uint32_t>(i);
    std::vector<std::vector<uint32_t>> nb(n);
    const int dr[6] = {0, 0, -1, -1, 1, 1}, dc[6] = {-2, 2, -1, 1, -1, 1};
    for (size_t i = 0; i < n; ++i)
        for (int k = 0; k < 6; ++k) {
            auto it = at.find(key(slide.row[i] + dr[k], slide.col[i] + dc[k]));
            if (it != at.end()) nb[i].push_back(it->second);
        }
    // The per-spot evidence is capped at +-logit_cap so that six concordant neighbours (field 6 * beta) can overrule an
    // isolated spot however extreme its count: low-count spots inside tissue and specks of ambient RNA outside it.
    auto logit = [&](double v) {
        v = std::min(std::max(v, 1e-6), 1 - 1e-6);
        return std::min(logit_cap, std::max(-logit_cap, std::log(v / (1 - v))));
    };
    std::vector<double> q = p, lp(n);
    for (size_t i = 0; i < n; ++i) lp[i] = logit(p[i]);
    for (int it = 0; it < smooth_iter; ++it) {
        std::vector<double> nq(n);
        for (size_t i = 0; i < n; ++i) {
            double field = 0;
            for (uint32_t j : nb[i]) field += 2 * q[j] - 1;
            nq[i] = 1.0 / (1.0 + std::exp(-(lp[i] + beta * field)));
        }
        q.swap(nq);
    }
    for (size_t i = 0; i < n; ++i) {
        r.prob[i] = static_cast<float>(q[i]);
        r.in_tissue[i] = q[i] >= 0.5;
        r.n_tissue += r.in_tissue[i];
    }
    return r;
}

}  // namespace singlet
