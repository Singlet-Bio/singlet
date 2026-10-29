// Track B exploration tool: estimates (bits/nnz) of many index / value coding variants on a .1pz matrix.
// Not a codec: no decoder. Entropy numbers are "ideal static model" sizes (what a static rANS/arith coder
// would reach, excluding tables unless stated). Real codecs are in kernels/track_b*.h.
//   bin/track_b_explore.exe file.1pz [...]
#include "common.h"
#include <zstd.h>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace cs;

struct Space {
    std::vector<uint32_t> ptr, idx, val;
    uint32_t m = 0;
};

static std::vector<uint32_t> freq_rank(const CanonicalMatrix& M) {
    std::vector<uint32_t> rn(M.m, 0);
    for (auto r : M.indices) rn[r]++;
    std::vector<uint32_t> order(M.m);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return rn[a] > rn[b]; });
    std::vector<uint32_t> rank(M.m);
    for (uint32_t i = 0; i < M.m; ++i) rank[order[i]] = i;
    return rank;
}

static Space make_space(const CanonicalMatrix& M, const std::vector<uint32_t>* rank_of) {
    Space S;
    S.m = M.m;
    S.ptr = M.indptr;
    S.idx.resize(M.nnz);
    S.val.resize(M.nnz);
    if (!rank_of) {
        S.idx = M.indices;
        S.val = M.data;
        return S;
    }
    std::vector<std::pair<uint32_t, uint32_t>> tmp;
    for (uint32_t j = 0; j < M.n; ++j) {
        tmp.clear();
        for (uint32_t k = M.indptr[j]; k < M.indptr[j + 1]; ++k) tmp.push_back({(*rank_of)[M.indices[k]], M.data[k]});
        std::sort(tmp.begin(), tmp.end());
        for (size_t t = 0; t < tmp.size(); ++t) {
            S.idx[M.indptr[j] + t] = tmp[t].first;
            S.val[M.indptr[j] + t] = tmp[t].second;
        }
    }
    return S;
}

static inline int ilog2(uint32_t x) { return x ? 31 - __builtin_clz(x) : 0; }
// 2 buckets per octave
static inline int lb2(uint32_t x) {
    if (x < 4) return (int)x;
    int l = ilog2(x);
    return 2 * l + (int)((x >> (l - 1)) & 1);
}

// log-linear symbol for gap g >= 0: g<8 direct; else 4 sub-buckets per octave; returns symbol and extra raw bits.
static inline void gsym(uint32_t g, int& sym, int& extra) {
    if (g < 8) { sym = (int)g; extra = 0; return; }
    int b = ilog2(g);
    sym = 8 + (b - 3) * 4 + (int)((g >> (b - 2)) & 3);
    extra = b - 2;
}

struct Ent {
    std::vector<std::vector<uint64_t>> h;  // ctx -> symbol histogram
    double raw_bits = 0;
    uint64_t n = 0;
    void init(int nctx, int nsym) { h.assign(nctx, std::vector<uint64_t>(nsym, 0)); raw_bits = 0; n = 0; }
    void add(int ctx, int sym, int extra) { h[ctx][sym]++; raw_bits += extra; n++; }
    // returns ideal bits (entropy + raw), and number of nonzero table cells
    double bits(uint64_t* cells = nullptr) const {
        double b = raw_bits;
        uint64_t nz = 0;
        for (auto& v : h) {
            uint64_t tot = 0;
            for (auto c : v) tot += c;
            for (auto c : v)
                if (c) { b -= (double)c * std::log2((double)c / (double)tot); nz++; }
        }
        if (cells) *cells = nz;
        return b;
    }
};

static size_t zsize(const std::vector<uint8_t>& v, int level) {
    if (v.empty()) return 0;
    std::vector<uint8_t> o(ZSTD_compressBound(v.size()));
    size_t z = ZSTD_compress(o.data(), o.size(), v.data(), v.size(), level);
    return z;
}

static void leb(std::vector<uint8_t>& o, uint32_t v) {
    while (v >= 128) { o.push_back((uint8_t)(v | 128)); v >>= 7; }
    o.push_back((uint8_t)v);
}

// per-chunk (1024 cols) streams
static void chunk_stream_cost(const Space& S, const char* name, bool byte_split, double nnz) {
    const uint32_t n = (uint32_t)S.ptr.size() - 1;
    size_t raw = 0, z3 = 0, z19 = 0;
    for (uint32_t c0 = 0; c0 < n; c0 += 1024) {
        uint32_t c1 = std::min(n, c0 + 1024);
        std::vector<uint8_t> st, lo, hi;
        for (uint32_t j = c0; j < c1; ++j) {
            int64_t prev = -1;
            for (uint32_t k = S.ptr[j]; k < S.ptr[j + 1]; ++k) {
                uint32_t g = (uint32_t)((int64_t)S.idx[k] - prev - 1);
                prev = S.idx[k];
                if (byte_split) { lo.push_back((uint8_t)g); hi.push_back((uint8_t)(g >> 8)); if (g >= 65536) { /* 3rd byte ignored: cost only */ hi.push_back((uint8_t)(g >> 16)); } }
                else leb(st, g);
            }
        }
        if (byte_split) { st = lo; st.insert(st.end(), hi.begin(), hi.end()); }
        raw += st.size();
        z3 += zsize(st, 3);
        z19 += zsize(st, 19);
    }
    std::printf("  %-34s raw %6.2f  z3 %6.2f  z19 %6.2f  bits/nnz (idx only)\n", name, raw * 8.0 / nnz, z3 * 8.0 / nnz, z19 * 8.0 / nnz);
}

static double ef_bits(uint32_t k, uint32_t u) {
    if (!k) return 0;
    int l = u / k > 1 ? ilog2(u / k) : 0;
    return (double)k * l + k + (double)((u >> l) + 1);
}

static double tbin(uint32_t v, uint32_t range) {  // truncated binary cost of v in [0,range)
    if (range <= 1) return 0;
    int b = ilog2(range);
    uint32_t u = (1u << (b + 1)) - range;
    if ((1u << b) == range) return b;
    return v < u ? b : b + 1;
}

static double bic_rec(const uint32_t* a, int l, int r, int64_t lo, int64_t hi) {
    // iterative stack
    struct F { int l, r; int64_t lo, hi; };
    std::vector<F> st;
    st.push_back({l, r, lo, hi});
    double cost = 0;
    while (!st.empty()) {
        F f = st.back();
        st.pop_back();
        if (f.r < f.l) continue;
        int mid = (f.l + f.r) / 2;
        int64_t lo2 = f.lo + (mid - f.l), hi2 = f.hi - (f.r - mid);
        int64_t range = hi2 - lo2 + 1;
        cost += tbin((uint32_t)(a[mid] - lo2), (uint32_t)range);
        st.push_back({f.l, mid - 1, f.lo, (int64_t)a[mid] - 1});
        st.push_back({mid + 1, f.r, (int64_t)a[mid] + 1, f.hi});
    }
    return cost;
}

static double roaring_bits(const uint32_t* a, uint32_t k, uint32_t m) {
    double bits = 0;
    uint32_t i = 0;
    while (i < k) {
        uint32_t blk = a[i] >> 16, j = i;
        uint32_t runs = 0;
        uint32_t prev = 0;
        bool first = true;
        while (j < k && (a[j] >> 16) == blk) {
            if (first || a[j] != prev + 1) runs++;
            first = false;
            prev = a[j];
            j++;
        }
        uint32_t c = j - i;
        double arr = 16.0 * c, bm = 8192.0 * 8, run = 32.0 * runs + 16;
        bits += 32 + std::min({arr, bm, run});
        i = j;
    }
    (void)m;
    return bits;
}

static void struct_costs(const Space& S, const char* name, double nnz) {
    const uint32_t n = (uint32_t)S.ptr.size() - 1;
    double ef = 0, bic = 0, roar = 0;
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t k = S.ptr[j + 1] - S.ptr[j];
        if (!k) continue;
        const uint32_t* a = &S.idx[S.ptr[j]];
        ef += ef_bits(k, S.m);
        bic += bic_rec(a, 0, (int)k - 1, 0, (int64_t)S.m - 1);
        roar += roaring_bits(a, k, S.m);
    }
    std::printf("  %-34s EF %6.2f  BIC %6.2f  Roaring %6.2f  bits/nnz (idx only, counts excluded)\n", name, ef / nnz, bic / nnz, roar / nnz);
}

int main(int argc, char** argv) {
    for (int a = 1; a < argc; ++a) {
        CanonicalMatrix M = load_1pz(argv[a]);
        const double nnz = (double)M.nnz;
        std::printf("== %s m=%u n=%u nnz=%llu\n", argv[a], M.m, M.n, (unsigned long long)M.nnz);
        auto rank = freq_rank(M);
        Space So = make_space(M, nullptr), Sr = make_space(M, &rank);

        chunk_stream_cost(Sr, "rank gap LEB128", false, nnz);
        chunk_stream_cost(So, "orig gap LEB128", false, nnz);
        chunk_stream_cost(Sr, "rank gap bytesplit16", true, nnz);
        chunk_stream_cost(So, "orig gap bytesplit16", true, nnz);
        struct_costs(Sr, "rank space", nnz);
        struct_costs(So, "orig space", nnz);

        // counts: entropy of log-linear class + raw
        {
            Ent e;
            e.init(1, 80);
            for (uint32_t j = 0; j < M.n; ++j) {
                int s, x;
                gsym(M.indptr[j + 1] - M.indptr[j], s, x);
                e.add(0, s, x);
            }
            std::printf("  col-count H0 (class+raw)            %6.2f bits/nnz\n", e.bits() / nnz);
        }

        // contexted gap entropies
        for (int space = 0; space < 2; ++space) {
            const Space& S = space ? So : Sr;
            const char* sn = space ? "orig" : "rank";
            const char* cn[] = {"none", "pos", "pos x k", "pos x log2(n_sofar)", "pos x prevsym", "pos x k x prevcls", "pos x k x n_sofar"};
            const int NC = 7;
            for (int cfg = 0; cfg < NC; ++cfg) {
                if (space == 1 && cfg > 2) continue;
                Ent e;
                e.init(40 * 12 * 12 * 3, 80);
                for (uint32_t j = 0; j < M.n; ++j) {
                    uint32_t k = S.ptr[j + 1] - S.ptr[j];
                    int kb = std::min(ilog2(k + 1), 11);
                    int64_t prev = -1;
                    int psym = 0;
                    for (uint32_t t = 0; t < k; ++t) {
                        uint32_t r = S.idx[S.ptr[j] + t];
                        uint32_t g = (uint32_t)(r - prev - 1);
                        int s, x;
                        gsym(g, s, x);
                        int pb = std::min(lb2((uint32_t)(prev + 1)), 39);
                        int nb = std::min(ilog2(t + 1), 11);
                        int pc = std::min(psym / 4, 11);  // coarse prev class
                        int ctx = 0;
                        switch (cfg) {
                            case 0: ctx = 0; break;
                            case 1: ctx = pb; break;
                            case 2: ctx = pb * 12 + kb; break;
                            case 3: ctx = pb * 12 + nb; break;
                            case 4: ctx = pb * 80 + psym; break;
                            case 5: ctx = (pb * 12 + kb) * 12 + pc; break;
                            case 6: ctx = (pb * 12 + kb) * 12 + nb; break;
                        }
                        e.add(ctx, s, x);
                        prev = r;
                        psym = s;
                    }
                }
                uint64_t cells;
                double b = e.bits(&cells);
                std::printf("  %s gap H [%-22s] %6.2f bits/nnz  (+tables ~%.2f)  cells=%llu\n", sn, cn[cfg], b / nnz, cells * 10.0 / nnz, (unsigned long long)cells);
            }
        }

        // values: flag & residual
        {
            for (int cfg = 0; cfg < 4; ++cfg) {
                Ent f;  // flag
                Ent r;  // residual class
                f.init(40 * 12, 2);
                r.init(40, 80);
                for (uint32_t j = 0; j < M.n; ++j) {
                    uint32_t k = Sr.ptr[j + 1] - Sr.ptr[j];
                    int kb = std::min(ilog2(k + 1), 11);
                    for (uint32_t t = 0; t < k; ++t) {
                        uint32_t rk = Sr.idx[Sr.ptr[j] + t];
                        uint32_t v = Sr.val[Sr.ptr[j] + t];
                        int pb = std::min(lb2(rk), 39);
                        int ctx = cfg == 0 ? 0 : cfg == 1 ? pb : cfg == 2 ? kb : pb * 12 + kb;
                        f.add(ctx, v > 1, 0);
                        if (v > 1) {
                            int s, x;
                            gsym(v - 2, s, x);
                            r.add(cfg == 0 ? 0 : pb, s, x);
                        }
                    }
                }
                const char* cn[] = {"none", "rank pos", "k", "rank pos x k"};
                std::printf("  values flag[%-14s] %5.2f + resid %5.2f = %5.2f bits/nnz\n", cn[cfg], f.bits() / nnz, r.bits() / nnz, (f.bits() + r.bits()) / nnz);
            }
        }
    }
    return 0;
}
