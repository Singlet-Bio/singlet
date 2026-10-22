// SPDX-License-Identifier: MIT
// Idea 2, encoder side only: cell embedding (floating point is fine, the result is transmitted explicitly as
// cluster labels or as a permutation) and k-means labelling of the called cells.
#pragma once
#include "common.h"
#include <cmath>
#include <numeric>
#include <random>

namespace cs { namespace i2 {

struct Embedding {
    std::vector<u32> cells;          // original column ids of called cells
    int D = 0;
    std::vector<float> y;            // cells.size() x D, row-major, PCA scores (centred)
    std::vector<float> depth;        // log total UMI
};

// log-CPM features over the top-F features by nnz count among called cells; randomised subspace-iteration PCA.
inline Embedding embed_cells(const CanonicalMatrix& M, const std::vector<u8>& isc, int D = 24, int F = 2000, int iters = 6,
                             bool l2norm_rows = false) {
    Embedding E;
    for (u32 j = 0; j < M.n; ++j) if (isc[j]) E.cells.push_back(j);
    const size_t nc = E.cells.size();
    E.D = D;
    std::vector<u32> z(M.m, 0);
    for (u32 j : E.cells) for (u32 q = M.indptr[j]; q < M.indptr[j + 1]; ++q) ++z[M.indices[q]];
    std::vector<u32> ord(M.m);
    std::iota(ord.begin(), ord.end(), 0u);
    F = (int)std::min<size_t>((size_t)F, M.m);
    std::partial_sort(ord.begin(), ord.begin() + F, ord.end(), [&](u32 a, u32 b) { return z[a] != z[b] ? z[a] > z[b] : a < b; });
    std::vector<int> fpos(M.m, -1);
    for (int i = 0; i < F; ++i) fpos[ord[i]] = i;
    // sparse rows
    std::vector<size_t> rp(nc + 1, 0);
    std::vector<int> ci; std::vector<float> cv;
    E.depth.resize(nc);
    for (size_t i = 0; i < nc; ++i) {
        u32 j = E.cells[i];
        double k = 0;
        for (u32 q = M.indptr[j]; q < M.indptr[j + 1]; ++q) k += M.data[q];
        E.depth[i] = (float)std::log(k + 1.0);
        for (u32 q = M.indptr[j]; q < M.indptr[j + 1]; ++q) {
            int fp = fpos[M.indices[q]];
            if (fp >= 0) { ci.push_back(fp); cv.push_back((float)std::log1p(1e4 * M.data[q] / std::max(1.0, k))); }
        }
        rp[i + 1] = ci.size();
    }
    // column means
    std::vector<double> mu(F, 0.0);
    for (size_t t = 0; t < ci.size(); ++t) mu[ci[t]] += cv[t];
    for (int f = 0; f < F; ++f) mu[f] /= (double)nc;
    // X_c = X - 1 mu^T ; products implicitly
    auto mulXV = [&](const std::vector<float>& V, int d, std::vector<float>& Y) {      // Y (nc x d) = X_c V (F x d)
        Y.assign(nc * d, 0.f);
        std::vector<double> muV(d, 0.0);
        for (int f = 0; f < F; ++f) for (int c = 0; c < d; ++c) muV[c] += mu[f] * V[(size_t)f * d + c];
        for (size_t i = 0; i < nc; ++i) {
            float* y = &Y[i * d];
            for (size_t t = rp[i]; t < rp[i + 1]; ++t) {
                const float* v = &V[(size_t)ci[t] * d];
                float x = cv[t];
                for (int c = 0; c < d; ++c) y[c] += x * v[c];
            }
            for (int c = 0; c < d; ++c) y[c] -= (float)muV[c];
        }
    };
    auto mulXtY = [&](const std::vector<float>& Y, int d, std::vector<float>& V) {     // V (F x d) = X_c^T Y
        V.assign((size_t)F * d, 0.f);
        std::vector<double> sy(d, 0.0);
        for (size_t i = 0; i < nc; ++i) for (int c = 0; c < d; ++c) sy[c] += Y[i * d + c];
        for (size_t i = 0; i < nc; ++i) {
            const float* y = &Y[i * d];
            for (size_t t = rp[i]; t < rp[i + 1]; ++t) {
                float* v = &V[(size_t)ci[t] * d];
                float x = cv[t];
                for (int c = 0; c < d; ++c) v[c] += x * y[c];
            }
        }
        for (int f = 0; f < F; ++f) for (int c = 0; c < d; ++c) V[(size_t)f * d + c] -= (float)(mu[f] * sy[c]);
    };
    auto orth = [&](std::vector<float>& A, size_t rows, int d) {       // modified Gram-Schmidt on columns
        for (int c = 0; c < d; ++c) {
            for (int c2 = 0; c2 < c; ++c2) {
                double dot = 0;
                for (size_t r = 0; r < rows; ++r) dot += (double)A[r * d + c] * A[r * d + c2];
                for (size_t r = 0; r < rows; ++r) A[r * d + c] -= (float)(dot * A[r * d + c2]);
            }
            double nn = 0;
            for (size_t r = 0; r < rows; ++r) nn += (double)A[r * d + c] * A[r * d + c];
            nn = std::sqrt(nn) + 1e-30;
            for (size_t r = 0; r < rows; ++r) A[r * d + c] = (float)(A[r * d + c] / nn);
        }
    };
    const int Dp = D + 6;
    std::mt19937 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> V((size_t)F * Dp), Y;
    for (auto& v : V) v = nd(rng);
    orth(V, F, Dp);
    for (int it = 0; it < iters; ++it) {
        mulXV(V, Dp, Y);
        orth(Y, nc, Dp);
        mulXtY(Y, Dp, V);
        orth(V, F, Dp);
    }
    mulXV(V, Dp, Y);                  // scores for the Dp-dim subspace (approx PCs, unordered within subspace)
    E.D = Dp; D = Dp;
    E.y = Y;
    if (l2norm_rows) {
        for (size_t i = 0; i < nc; ++i) {
            double nn = 0; for (int c = 0; c < D; ++c) nn += (double)E.y[i * D + c] * E.y[i * D + c];
            nn = std::sqrt(nn) + 1e-12;
            for (int c = 0; c < D; ++c) E.y[i * D + c] = (float)(E.y[i * D + c] / nn);
        }
    }
    return E;
}

// k-means (k-means++ init), returns labels; clusters are renumbered by their mean first coordinate so that
// neighbouring labels are similar.
inline std::vector<u16> kmeans_labels(const Embedding& E, int K, int iters = 30, int use_dims = 0) {
    const size_t nc = E.cells.size();
    const int D = use_dims > 0 ? std::min(use_dims, E.D) : E.D;
    const int ld = E.D;
    std::vector<u16> lab(nc, 0);
    if (K <= 1 || nc == 0) return lab;
    K = (int)std::min<size_t>((size_t)K, nc);
    std::mt19937_64 rng(777);
    std::vector<double> cen((size_t)K * D, 0.0);
    std::vector<double> d2(nc, 1e300);
    size_t first = rng() % nc;
    auto dist = [&](size_t i, const double* c) { double s = 0; for (int d = 0; d < D; ++d) { double t = E.y[i * ld + d] - c[d]; s += t * t; } return s; };
    for (int d = 0; d < D; ++d) cen[d] = E.y[first * ld + d];
    for (int k = 1; k < K; ++k) {
        double tot = 0;
        for (size_t i = 0; i < nc; ++i) { d2[i] = std::min(d2[i], dist(i, &cen[(size_t)(k - 1) * D])); tot += d2[i]; }
        double u = std::uniform_real_distribution<double>(0, tot)(rng), acc = 0;
        size_t pick = nc - 1;
        for (size_t i = 0; i < nc; ++i) { acc += d2[i]; if (acc >= u) { pick = i; break; } }
        for (int d = 0; d < D; ++d) cen[(size_t)k * D + d] = E.y[pick * ld + d];
    }
    std::vector<int> cnt(K);
    for (int it = 0; it < iters; ++it) {
        bool changed = false;
        for (size_t i = 0; i < nc; ++i) {
            int best = 0; double bd = 1e300;
            for (int k = 0; k < K; ++k) { double d_ = dist(i, &cen[(size_t)k * D]); if (d_ < bd) { bd = d_; best = k; } }
            if (lab[i] != best) { lab[i] = (u16)best; changed = true; }
        }
        std::fill(cen.begin(), cen.end(), 0.0); std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < nc; ++i) { ++cnt[lab[i]]; for (int d = 0; d < D; ++d) cen[(size_t)lab[i] * D + d] += E.y[i * ld + d]; }
        for (int k = 0; k < K; ++k) {
            if (cnt[k]) for (int d = 0; d < D; ++d) cen[(size_t)k * D + d] /= cnt[k];
            else { size_t r = rng() % nc; for (int d = 0; d < D; ++d) cen[(size_t)k * D + d] = E.y[r * ld + d]; }
        }
        if (!changed) break;
    }
    std::vector<int> ordk;
    {
        std::vector<char> used(K, 0);
        int cur = 0; used[0] = 1; ordk.push_back(0);
        for (int r = 1; r < K; ++r) {
            int best = -1; double bd = 1e300;
            for (int k = 0; k < K; ++k) if (!used[k]) {
                double s2 = 0; for (int d = 0; d < D; ++d) { double t = cen[(size_t)k * D + d] - cen[(size_t)cur * D + d]; s2 += t * t; }
                if (s2 < bd) { bd = s2; best = k; }
            }
            used[best] = 1; ordk.push_back(best); cur = best;
        }
    }
    std::vector<u16> remap(K);
    for (int r = 0; r < K; ++r) remap[ordk[r]] = (u16)r;
    for (auto& l : lab) l = remap[l];
    return lab;
}

}}  // namespace cs::i2

namespace cs { namespace i2 {

// Greedy nearest-neighbour chain over the cells of one cluster (indices into E), starting from the cell with
// the smallest depth; O(c^2 D) for a cluster of c cells, so clusters are kept to a few thousand cells.
inline std::vector<size_t> nn_chain(const Embedding& E, const std::vector<size_t>& members, size_t cap = 6000) {
    const int D = E.D;
    std::vector<size_t> mem = members, out;
    if (mem.size() > cap) {
        // too big: split by depth into blocks of `cap` and chain each block
        std::stable_sort(mem.begin(), mem.end(), [&](size_t a, size_t b) { return E.depth[a] < E.depth[b]; });
        for (size_t s = 0; s < mem.size(); s += cap) {
            std::vector<size_t> blk(mem.begin() + s, mem.begin() + std::min(mem.size(), s + cap));
            auto o = nn_chain(E, blk, cap);
            out.insert(out.end(), o.begin(), o.end());
        }
        return out;
    }
    size_t c = mem.size();
    if (c == 0) return out;
    std::vector<char> used(c, 0);
    size_t cur = 0;
    for (size_t i = 1; i < c; ++i) if (E.depth[mem[i]] < E.depth[mem[cur]]) cur = i;
    used[cur] = 1; out.push_back(mem[cur]);
    for (size_t step = 1; step < c; ++step) {
        const float* yc = &E.y[mem[cur] * D];
        size_t best = (size_t)-1; float bd = 1e30f;
        for (size_t i = 0; i < c; ++i) {
            if (used[i]) continue;
            const float* yi = &E.y[mem[i] * D];
            float s = 0; for (int d = 0; d < D; ++d) { float t = yi[d] - yc[d]; s += t * t; }
            if (s < bd) { bd = s; best = i; }
        }
        used[best] = 1; out.push_back(mem[best]); cur = best;
    }
    return out;
}

// Fenwick tree with order-statistic select, used to code/decode a permutation as a Lehmer code.
struct Fenwick {
    u32 n; std::vector<u32> t; u32 LOG;
    explicit Fenwick(u32 n_) : n(n_), t(n_ + 1, 0) { LOG = 1; while ((1u << LOG) <= n) ++LOG; for (u32 i = 1; i <= n; ++i) { t[i] += 1; u32 j = i + (i & (0 - i)); if (j <= n) t[j] += t[i]; } }
    void dec(u32 i) { for (++i; i <= n; i += i & (0 - i)) --t[i]; }
    u32 prefix(u32 i) const { u32 s = 0; for (; i; i -= i & (0 - i)) s += t[i]; return s; }   // count of present in [0, i)
    u32 select(u32 k) const {        // index of the k-th (0-based) present element
        u32 pos = 0;
        for (u32 b = LOG; b-- > 0;) { u32 np = pos + (1u << b); if (np <= n && t[np] <= k) { pos = np; k -= t[np]; } }
        return pos;
    }
};

}}  // namespace cs::i2
