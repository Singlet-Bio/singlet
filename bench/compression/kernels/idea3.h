// Idea 3: gene / interval factorisation (and, in idea3_joint.cpp, exon|intron joint coding).
//
// A matrix is coded per cell as   k (genes present)  ->  which genes  ->  UMI count per gene  ->  split of each
// gene's UMIs over its intervals (rows).  Rows are grouped by gene id (the prefix of the row name), which
// the decoder knows from the row names.  Everything on the decode path is integer arithmetic; floating point is
// used (a) to build constant lookup tables once at start-up (log/exp tables, which a production version would bake
// in as constants) and (b) in encoder-side cost accounting only.
//
//  * presence:  independent Bernoulli(p[b][g]) with b = depth bucket of the cell's gene-nnz k (k itself is coded).
//               The k-subset is coded as k "first present gene" events, each by bisection over the gene index range with
//               exact window probabilities from integer prefix sums of -log2(1-p)  (O(k log G) binary decisions per cell).
//  * counts:    gene UMI count c>=1, adaptive multi-symbol model with context (gene rate class, depth bucket).
//  * split:     the c UMIs over the n_g intervals of the gene as an exact multiset: Beta-binomial chain over the
//               intervals (Dirichlet-multinomial with adaptive per-interval counts; static multinomial when frozen).
//  * flat mode: every row is its own "gene" -> the direct interval-level model (same machinery) for comparison.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include <algorithm>
#include <cmath>
#include <memory>
#include <unordered_map>

namespace cs { namespace i3 {

using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using u64 = uint64_t; using i64 = int64_t;
using u128 = unsigned __int128;

constexpr int NB = 24;                 // depth buckets
constexpr int NCLS = 64;               // gene rate classes

inline int bitlen(u32 v) { return v ? 32 - __builtin_clz(v) : 0; }
inline int bucket_of(u32 k) {          // k >= 1
    if (k <= 4) return (int)k - 1;
    u64 kk = (u64)k * k;
    int t = 0;
    while (t < 19 && (16ull << (t + 1)) <= kk) t++;
    int b = 4 + t;
    return b > NB - 1 ? NB - 1 : b;
}

// ------------------------------------------------------------------------------------------------------------
// Range coder (LZMA-style carry handling, 32-bit range, frequency-total up to 2^16).  One class does both
// directions so the model code is written once: every primitive takes the symbol to encode and returns the symbol
// (decoded when decoding).
// ------------------------------------------------------------------------------------------------------------
enum Comp { C_SIDE = 0, C_K, C_PRES, C_CNT, C_SPLIT, C_X0, C_X1, C_X2, NCOMP };

struct Coder {
    bool enc = true;
    // encoder
    u64 low = 0; u32 range = 0xFFFFFFFFu; u8 cache = 0; u64 cache_size = 1; std::vector<u8> out;
    // decoder
    const u8* p = nullptr; const u8* end = nullptr; u32 code = 0; u32 rr = 0;
    // encoder accounting (ideal bits per component)
    double acc[NCOMP] = {0}; int comp = 0; bool account = true;

    void shift_low() {
        if ((u32)low < 0xFF000000u || (low >> 32) != 0) {
            u8 carry = (u8)(low >> 32);
            u8 temp = cache;
            do { out.push_back((u8)(temp + carry)); temp = 0xFF; } while (--cache_size != 0);
            cache = (u8)((low >> 24) & 0xFF);
        }
        cache_size++;
        low = (low & 0x00FFFFFFu) << 8;
    }
    void start_dec(const u8* b, const u8* e) {
        enc = false; p = b; end = e; range = 0xFFFFFFFFu; code = 0;
        for (int i = 0; i < 5; ++i) code = (code << 8) | next();
    }
    inline u8 next() { return p < end ? *p++ : 0; }
    void finish() { for (int i = 0; i < 5; ++i) shift_low(); }

    inline void enc_(u32 cum, u32 freq, u32 tot) {
        u32 r = range / tot;
        low += (u64)cum * r;
        range = freq * r;
        while (range < (1u << 24)) { range <<= 8; shift_low(); }
        if (account) acc[comp] += (double)(l2tab()[tot] - l2tab()[freq]);
    }
    static const float* l2tab() {
        static std::vector<float> t = [] { std::vector<float> v(65537); v[0] = 0; for (int i = 1; i <= 65536; ++i) v[i] = (float)std::log2((double)i); return v; }();
        return t.data();
    }
    inline u32 target(u32 tot) { rr = range / tot; u32 v = code / rr; return v >= tot ? tot - 1 : v; }
    inline void consume(u32 cum, u32 freq) {
        code -= cum * rr; range = freq * rr;
        while (range < (1u << 24)) { range <<= 8; code = (code << 8) | next(); }
    }
    // multi-symbol with explicit frequencies
    inline u32 sym(const u32* freq, u32 n, u32 tot, u32 s) {
        if (enc) {
            u32 cum = 0;
            for (u32 i = 0; i < s; ++i) cum += freq[i];
            enc_(cum, freq[s], tot);
            return s;
        }
        u32 v = target(tot), cum = 0, i = 0;
        while (i + 1 < n && cum + freq[i] <= v) { cum += freq[i]; ++i; }
        consume(cum, freq[i]);
        return i;
    }
    // binary event; p1 = P(bit==1) in 1..65535 (of 65536)
    inline u32 bin(u32 p1, u32 bit) {
        if (enc) {
            u32 r = range >> 16;
            if (bit) range = p1 * r; else { low += (u64)p1 * r; range = (65536 - p1) * r; }
            while (range < (1u << 24)) { range <<= 8; shift_low(); }
            if (account) acc[comp] += (double)(16.0f - l2tab()[bit ? p1 : 65536 - p1]);
            return bit;
        }
        rr = range >> 16;
        u32 v = code / rr; if (v >= 65536) v = 65535;
        if (v < p1) { range = p1 * rr; while (range < (1u << 24)) { range <<= 8; code = (code << 8) | next(); } return 1; }
        code -= p1 * rr; range = (65536 - p1) * rr;
        while (range < (1u << 24)) { range <<= 8; code = (code << 8) | next(); }
        return 0;
    }
    // raw bits (nb up to 31)
    inline u32 bits(u32 v, int nb) {
        u32 res = 0;
        while (nb > 0) {
            int t = nb > 16 ? 16 : nb;
            nb -= t;
            u32 part = enc ? ((v >> nb) & ((1u << t) - 1)) : 0;
            if (enc) enc_(part, 1, 1u << t);
            else { part = target(1u << t); consume(part, 1); }
            res = (res << t) | part;
        }
        return res;
    }
    // Elias-gamma style code for v >= 1 (uniform length prefix over 33 symbols + raw mantissa)
    inline u32 egamma(u32 v) {
        int nb = enc ? bitlen(v) : 0;
        nb = (int)bits((u32)nb, 6);
        u32 m = nb > 1 ? bits(enc ? (v & ((1u << (nb - 1)) - 1)) : 0, nb - 1) : 0;
        return nb == 0 ? 0 : (nb == 1 ? 1u : ((1u << (nb - 1)) | m));
    }
};

// adaptive frequency table (counts, halved when total exceeds lim)
struct AFreq {
    std::vector<u32> f; u32 tot = 0, inc = 32, lim = 65000;
    void init(u32 n, u32 v0 = 1, u32 inc_ = 32) { f.assign(n, v0); tot = n * v0; inc = inc_; }
    inline u32 code(Coder& c, u32 s) {
        s = c.sym(f.data(), (u32)f.size(), tot, s);
        f[s] += inc; tot += inc;
        if (tot > lim) { tot = 0; for (auto& x : f) { x = (x + 1) >> 1; tot += x; } }
        return s;
    }
};

// ------------------------------------------------------------------------------------------------------------
// Constant tables.
// ------------------------------------------------------------------------------------------------------------
struct Tabs {
    std::vector<i64> Lc;       // Lc[code] = -log2(1 - 2^-(code/2048)) in units of 2^-44, code in 1..65535
    u64 T2[4098];              // 2^-(i/4096) in units of 2^-62
    u64 LN2_62;
    Tabs() {
        Lc.assign(65536, 0);
        const double ln2 = 0.69314718055994530942;
        for (int c = 1; c < 65536; ++c) {
            double p = std::exp2(-c / 2048.0);
            double l = -std::log1p(-p) / ln2;
            i64 v = (i64)std::llround(l * 17592186044416.0);
            Lc[c] = v < 1 ? 1 : v;
        }
        Lc[0] = Lc[1];
        for (int i = 0; i <= 4097; ++i) T2[i] = (u64)std::llround(std::exp2(-i / 4096.0) * 4611686018427387904.0);
        LN2_62 = (u64)std::llround(ln2 * 4611686018427387904.0);
    }
};
inline const Tabs& tabs() { static Tabs t; return t; }

// F(D) = 1 - 2^-(D / 2^44), result in units of 2^-48 (< 2^48). Integer only.
inline u64 omexp(i64 D) {
    const Tabs& T = tabs();
    if (D <= 0) return 0;
    if (D >= ((i64)48 << 44)) return (1ull << 48) - 1;
    u64 e = (u64)D >> 44, f = (u64)D & ((1ull << 44) - 1);
    if (e == 0 && f < (1ull << 40)) {
        u64 y = (u64)(((u128)f * T.LN2_62) >> 44);            // y = x ln2 in 2^-62, < 2^57
        u64 y2 = (u64)(((u128)y * y) >> 62), y3 = (u64)(((u128)y2 * y) >> 62), y4 = (u64)(((u128)y3 * y) >> 62);
        u64 F = y - (y2 >> 1) + y3 / 6 - y4 / 24;
        u64 r = F >> 14;
        return r ? r : 1;
    }
    u64 idx = f >> 32, fr = f & 0xFFFFFFFFull;
    u64 t0 = T.T2[idx], t1 = T.T2[idx + 1];
    u64 z = t0 - (u64)(((u128)(t0 - t1) * fr) >> 32);
    z = (e >= 63) ? 0 : (z >> e);
    u64 F62 = (1ull << 62) - z;
    u64 r = F62 >> 14;
    return r >= (1ull << 48) ? (1ull << 48) - 1 : r;
}
inline u32 quant_code(double p) {             // -log2(p)*2048, clamp [1,65535]
    if (!(p > 0)) return 65535;
    double c = -std::log2(p) * 2048.0;
    if (c < 1) return 1;
    if (c > 65535) return 65535;
    return (u32)std::llround(c);
}

// ------------------------------------------------------------------------------------------------------------
// Gene structure from row names.
// ------------------------------------------------------------------------------------------------------------
struct GeneMap {
    u32 G = 0, m = 0;
    std::vector<u32> first;          // G+1
    std::vector<u32> gene_of_row;    // m
    std::vector<std::string> ids;
};
inline GeneMap make_genemap(const std::vector<std::string>& rn, bool flat) {
    GeneMap g;
    g.m = (u32)rn.size();
    g.gene_of_row.resize(g.m);
    if (flat) {
        g.G = g.m; g.first.resize(g.m + 1);
        for (u32 i = 0; i <= g.m; ++i) g.first[i] = i;
        for (u32 i = 0; i < g.m; ++i) g.gene_of_row[i] = i;
        return g;
    }
    std::unordered_map<std::string, u32> seen;
    std::string prev;
    for (u32 i = 0; i < g.m; ++i) {
        size_t u = rn[i].find('_');
        std::string id = u == std::string::npos ? rn[i] : rn[i].substr(0, u);
        if (i == 0 || id != prev) {
            if (seen.count(id)) throw std::runtime_error("non-contiguous gene rows: " + id);
            seen[id] = (u32)g.ids.size();
            g.ids.push_back(id);
            g.first.push_back(i);
            prev = id;
        }
        g.gene_of_row[i] = (u32)g.ids.size() - 1;
    }
    g.G = (u32)g.ids.size();
    g.first.push_back(g.m);
    return g;
}

// ------------------------------------------------------------------------------------------------------------
// Presence model: per (bucket, gene) code of -log2 p, plus gene rate class (for count contexts).
// ------------------------------------------------------------------------------------------------------------
struct GeneTables {
    u32 G = 0;
    std::vector<u8> cls;                    // G
    std::vector<u16> lp;                    // NB * G, b-major
    std::vector<std::vector<i64>> pref;     // lazily built per bucket
    const i64* prefix(int b) {
        if (pref.empty()) pref.resize(NB);
        auto& v = pref[b];
        if (v.empty()) {
            const Tabs& T = tabs();
            v.resize(G + 1);
            v[0] = 0;
            const u16* l = lp.data() + (size_t)b * G;
            for (u32 g = 0; g < G; ++g) v[g + 1] = v[g] + T.Lc[l[g]];
        }
        return v.data();
    }
    size_t bytes() const { return lp.size() * 2 + cls.size(); }
};

inline int rate_class(double rate) {
    if (!(rate > 0)) return NCLS - 1;
    double c = std::floor(3.0 * (-std::log2(rate)));
    if (c < 0) c = 0;
    if (c > NCLS - 1) c = NCLS - 1;
    return (int)c;
}

// encode/decode the presence set of one cell by bisection
inline void code_presence(Coder& c, GeneTables& gt, int b, u32 k, u32* genes) {
    const i64* Lp = gt.prefix(b);
    u32 lo = 0;
    const u32 G = gt.G;
    for (u32 t = 0; t < k; ++t) {
        u32 r = k - t;
        u32 hi = G - (r - 1);
        u32 target = c.enc ? genes[t] : 0;
        u64 Fh = omexp(Lp[hi] - Lp[lo]);
        while (hi - lo > 1) {
            u32 mid = (lo + hi) >> 1;
            u64 Fm = omexp(Lp[mid] - Lp[lo]);
            u64 P = Fh ? (Fm << 16) / Fh : 32768;
            if (P < 1) P = 1;
            if (P > 65535) P = 65535;
            u32 bit = c.bin((u32)P, c.enc ? (target < mid) : 0);
            if (bit) { hi = mid; Fh = Fm; }
            else { lo = mid; Fh = omexp(Lp[hi] - Lp[lo]); }
        }
        genes[t] = lo;
        lo = lo + 1;
    }
}

// ------------------------------------------------------------------------------------------------------------
// Interval split model.
// ------------------------------------------------------------------------------------------------------------
struct SplitModel {
    const GeneMap* gm = nullptr;
    std::vector<u32> f;      // per interval pseudo-counts
    std::vector<u32> F;      // per gene totals
    u32 inc = 32;            // added per UMI observed (0 = frozen)
    u32 spmf = 32;           // UMI unit in the Beta-binomial formulas (concentration); >= 1
    static constexpr u32 LIM = 60000;
    void init_flat(const GeneMap* g, u32 a0, u32 inc_, u32 spmf_) {
        gm = g; inc = inc_; spmf = spmf_;
        f.assign(g->m, a0);
        F.assign(g->G, 0);
        for (u32 i = 0; i < g->G; ++i) F[i] = (g->first[i + 1] - g->first[i]) * a0;
    }
    void init_prior(const GeneMap* g, const std::vector<u32>& f0, u32 inc_, u32 spmf_) {
        gm = g; inc = inc_; spmf = spmf_; f = f0;
        F.assign(g->G, 0);
        for (u32 i = 0; i < g->G; ++i) { u64 s = 0; for (u32 r = g->first[i]; r < g->first[i + 1]; ++r) s += f[r]; F[i] = (u32)s; }
    }
};

// Beta-binomial symbol: x in [0,r] ~ BetaBin(r; a = fi/s, b = (Ft-fi)/s), Ft > fi.  All 64-bit integer arithmetic
// (r <= 2048, fi, Ft < 2^22, s <= 64 keep every product below 2^63).
inline u32 code_bb(Coder& c, u32 r, u64 fi, u64 Ft, u32 s, u32 x_in, std::vector<u64>& up, std::vector<u64>& dn, std::vector<u32>& fr) {
    const u64 ONE = 1ull << 58, CUT = 1ull << 34, CAP = 1ull << 62;
    u64 rest = Ft - fi;
    u32 x0 = (u32)(((u64)r * fi) / Ft);
    up.clear(); dn.clear();
    up.push_back(ONE);
    {
        u32 x = x0; u64 w = ONE;
        while (x < r) {
            u64 num = (u64)(r - x) * (fi + (u64)x * s);
            u64 den = (u64)(x + 1) * (rest + (u64)(r - x - 1) * s);
            u64 q = (num << 24) / den;
            u128 nw128 = ((u128)w * q) >> 24;
            u64 nw = nw128 > CAP ? CAP : (u64)nw128;
            if (nw < CUT && nw <= w) break;
            up.push_back(nw); w = nw; ++x;
        }
    }
    {
        u32 x = x0; u64 w = ONE;
        while (x > 0) {
            u64 num = (u64)x * (rest + (u64)(r - x) * s);
            u64 den = (u64)(r - x + 1) * (fi + (u64)(x - 1) * s);
            u64 q = (num << 24) / den;
            u128 nw128 = ((u128)w * q) >> 24;
            u64 nw = nw128 > CAP ? CAP : (u64)nw128;
            if (nw < CUT && nw <= w) break;
            dn.push_back(nw); w = nw; --x;
        }
    }
    u32 xlo = x0 - (u32)dn.size();
    u32 nw_ = (u32)(dn.size() + up.size());
    u128 sumW = 0;
    for (u64 w : dn) sumW += w;
    for (u64 w : up) sumW += w;
    int sh = 0;
    { u64 hi = (u64)(sumW >> 64), lo = (u64)sumW; int bl = hi ? 64 + (64 - __builtin_clzll(hi)) : (lo ? 64 - __builtin_clzll(lo) : 0); sh = bl > 40 ? bl - 40 : 0; }
    u64 S = (u64)(sumW >> sh);
    u64 avail = 65536 - 32 - nw_;
    fr.resize(nw_ + 1);
    u32 tot = 0;
    for (u32 j = 0; j < nw_; ++j) {
        u64 w = j < dn.size() ? dn[dn.size() - 1 - j] : up[j - dn.size()];
        u32 q = 1 + (u32)(((w >> sh) * avail) / S);
        fr[j] = q; tot += q;
    }
    fr[nw_] = 65536 - tot;
    u32 sidx = nw_;
    if (c.enc && x_in >= xlo && x_in < xlo + nw_) sidx = x_in - xlo;
    sidx = c.sym(fr.data(), nw_ + 1, 65536, sidx);
    if (sidx < nw_) return xlo + sidx;
    return c.egamma(c.enc ? x_in + 1 : 0) - 1;      // escape: x coded raw
}

// Code the split of c UMIs over the intervals of gene g.  xs: dense per-interval counts (size n_g), in (enc) / out (dec).
struct SplitScratch { std::vector<u64> up, dn; std::vector<u32> fr; };
inline void code_split(Coder& c, SplitModel& sm, u32 g, u32 cnt, std::vector<u32>& xs, SplitScratch& sc) {
    const GeneMap& gm = *sm.gm;
    u32 base = gm.first[g], n = gm.first[g + 1] - base;
    if (n == 1) { xs[0] = cnt; return; }
    u32* f = sm.f.data() + base;
    u32 Ftot = sm.F[g];
    if (!c.enc) std::fill(xs.begin(), xs.begin() + n, 0u);
    if (cnt > 2048) {                              // rare huge counts: unmodelled Elias-gamma per interval (still exact)
        u32 r = cnt;
        for (u32 i = 0; i < n && r > 0; ++i) { u32 x = (i == n - 1) ? r : c.egamma(c.enc ? xs[i] + 1 : 0) - 1; xs[i] = x; r -= x; }
    } else if (cnt == 1) {
        u32 s = 0;
        if (c.enc) { while (xs[s] == 0) ++s; }
        s = c.sym(f, n, Ftot, s);
        xs[s] = 1;
    } else {
        u32 r = cnt; u64 Ft = Ftot;
        for (u32 i = 0; i < n && r > 0; ++i) {
            u64 fi = f[i];
            u32 x;
            if (i == n - 1 || Ft == fi) x = r;
            else x = code_bb(c, r, fi, Ft, sm.spmf, c.enc ? xs[i] : 0, sc.up, sc.dn, sc.fr);
            xs[i] = x; r -= x; Ft -= fi;
        }
    }
    if (sm.inc) {
        u64 add = 0;
        for (u32 i = 0; i < n; ++i) if (xs[i]) { u64 d = (u64)xs[i] * sm.inc; f[i] += (u32)std::min<u64>(d, 1u << 30); add += std::min<u64>(d, 1u << 30); }
        u64 tot = (u64)Ftot + add;
        while (tot > SplitModel::LIM) { tot = 0; for (u32 i = 0; i < n; ++i) { f[i] = (f[i] + 1) >> 1; tot += f[i]; } }
        sm.F[g] = (u32)tot;
    }
}

// ------------------------------------------------------------------------------------------------------------
// Cell-level k coder (zigzag delta vs previous cell, adaptive bit-length symbol, raw offset bits).
// ------------------------------------------------------------------------------------------------------------
struct KCoder {
    std::vector<AFreq> m; u32 prev = 0; int psb = 0;
    KCoder() { m.resize(34); for (auto& a : m) a.init(34, 1, 32); }
    u32 code(Coder& c, u32 k) {
        i64 d = (i64)k - (i64)prev;
        u64 z = d >= 0 ? (u64)(2 * d) : (u64)(-2 * d - 1);
        int sb = c.enc ? bitlen((u32)z) : 0;
        sb = (int)m[psb].code(c, (u32)sb);
        u32 zz = 0;
        if (sb >= 1) {
            u32 off = sb > 1 ? c.bits(c.enc ? ((u32)z & ((1u << (sb - 1)) - 1)) : 0, sb - 1) : 0;
            zz = (1u << (sb - 1)) | off;
        }
        i64 dd = (zz & 1) ? -(i64)((zz + 1) >> 1) : (i64)(zz >> 1);
        if (!c.enc) k = (u32)((i64)prev + dd);
        prev = k; psb = sb;
        return k;
    }
};

// count-given-presence coder: symbols 0..15 = c-1, 16 = escape (c>=17 -> egamma(c-16))
struct CountCoder {
    std::vector<AFreq> m;
    void init(u32 nctx) { m.resize(nctx); for (auto& a : m) a.init(17, 1, 32); }
    u32 code(Coder& c, u32 ctx, u32 cnt) {
        u32 s = c.enc ? (cnt - 1 < 16 ? cnt - 1 : 16) : 0;
        s = m[ctx].code(c, s);
        if (s < 16) return s + 1;
        return 15 + c.egamma(c.enc ? cnt - 15 : 0) ;
    }
};

// ------------------------------------------------------------------------------------------------------------
// Statistics gathering (encoder side; also used for training).
// ------------------------------------------------------------------------------------------------------------
struct GeneCells {            // gene-level view of a matrix: for each cell the sorted (gene, count) list
    std::vector<u32> cptr;    // n+1
    std::vector<u32> gene;    // gene-level nnz
    std::vector<u32> cnt;
};
inline GeneCells to_gene_cells(const CanonicalMatrix& M, const GeneMap& gm) {
    GeneCells gc;
    gc.cptr.assign(M.n + 1, 0);
    gc.gene.reserve(M.nnz); gc.cnt.reserve(M.nnz);
    for (u32 c = 0; c < M.n; ++c) {
        u32 a = M.indptr[c], b = M.indptr[c + 1];
        u32 i = a;
        while (i < b) {
            u32 g = gm.gene_of_row[M.indices[i]];
            u64 s = 0;
            while (i < b && gm.gene_of_row[M.indices[i]] == g) { s += M.data[i]; ++i; }
            gc.gene.push_back(g); gc.cnt.push_back((u32)s);
        }
        gc.cptr[c + 1] = (u32)gc.gene.size();
    }
    return gc;
}

// presence counts n[b][g], N[b]
struct PresStats {
    std::vector<u32> n; std::vector<u64> N;
    void init(u32 G) { n.assign((size_t)NB * G, 0); N.assign(NB, 0); }
};
inline void add_pres_stats(PresStats& ps, const GeneCells& gc, u32 ncell, u32 G) {
    for (u32 c = 0; c < ncell; ++c) {
        u32 k = gc.cptr[c + 1] - gc.cptr[c];
        if (!k) continue;
        int b = bucket_of(k);
        ps.N[b]++;
        u32* row = ps.n.data() + (size_t)b * G;
        for (u32 t = gc.cptr[c]; t < gc.cptr[c + 1]; ++t) row[gc.gene[t]]++;
    }
}

// class assignment + class x bucket tables from presence stats
struct ClassTables { std::vector<u8> cls; std::vector<u16> tab; /* NCLS*NB codes */ std::vector<double> p0; };
inline ClassTables make_class_tables(const PresStats& ps, u32 G) {
    ClassTables ct;
    ct.cls.assign(G, NCLS - 1);
    u64 Ntot = 0; for (int b = 0; b < NB; ++b) Ntot += ps.N[b];
    std::vector<double> rate(G, 0.0);
    for (int b = 0; b < NB; ++b) { const u32* row = ps.n.data() + (size_t)b * G; for (u32 g = 0; g < G; ++g) rate[g] += row[g]; }
    std::vector<double> nc(NCLS, 0.0);
    for (u32 g = 0; g < G; ++g) { rate[g] /= (Ntot ? (double)Ntot : 1.0); ct.cls[g] = (u8)rate_class(rate[g]); nc[ct.cls[g]] += 1; }
    ct.p0.assign((size_t)NCLS * NB, 0.0);
    std::vector<double> ncb((size_t)NCLS * NB, 0.0);
    for (int b = 0; b < NB; ++b) { const u32* row = ps.n.data() + (size_t)b * G; for (u32 g = 0; g < G; ++g) ncb[ct.cls[g] * NB + b] += row[g]; }
    for (int cl = 0; cl < NCLS; ++cl)
        for (int b = 0; b < NB; ++b)
            ct.p0[cl * NB + b] = (nc[cl] > 0 && ps.N[b] > 0) ? ncb[cl * NB + b] / (nc[cl] * (double)ps.N[b]) : -1.0;
    // buckets without cells borrow the nearest populated bucket
    for (int cl = 0; cl < NCLS; ++cl)
        for (int b = 0; b < NB; ++b)
            if (ct.p0[cl * NB + b] < 0) {
                int best = -1;
                for (int bb = 0; bb < NB; ++bb) if (ps.N[bb] > 0 && (best < 0 || std::abs(bb - b) < std::abs(best - b))) best = bb;
                ct.p0[cl * NB + b] = best >= 0 && nc[cl] > 0 ? ncb[cl * NB + best] / (nc[cl] * (double)ps.N[best]) : 0.0;
            }
    ct.tab.assign((size_t)NCLS * NB, 65535);
    for (int cl = 0; cl < NCLS; ++cl)
        for (int b = 0; b < NB; ++b) ct.tab[cl * NB + b] = (u16)quant_code(ct.p0[cl * NB + b]);
    return ct;
}

// ------------------------------------------------------------------------------------------------------------
// Out-of-sample model (trained from other samples of the same reference build).
// ------------------------------------------------------------------------------------------------------------
struct OosModel {
    u32 G = 0, m = 0;
    PresStats ps; std::vector<double> T;     // pooled presence stats and per-row UMI totals
    GeneTables gt;                           // finalised
    std::vector<u32> split_f0;               // per-row prior counts for the split
    std::vector<std::string> rownames_hash;  // unused
    size_t model_bytes = 0;
    bool built = false;
    void add(const CanonicalMatrix& Tm, const GeneMap& gm) {
        if (!G) { G = gm.G; m = gm.m; ps.init(G); T.assign(m, 0.0); }
        if (Tm.m != m) throw std::runtime_error("oos: row count mismatch");
        GeneCells gc = to_gene_cells(Tm, gm);
        add_pres_stats(ps, gc, Tm.n, G);
        for (size_t i = 0; i < Tm.nnz; ++i) T[Tm.indices[i]] += Tm.data[i];
    }
    // shrink = pseudo-cells toward the class rate; prior_mass = total pseudo-count per gene for the split prior
    void finalize(const GeneMap& gm, double shrink, double alpha, double prior_mass) {
        ClassTables ct = make_class_tables(ps, G);
        gt.G = G; gt.cls = ct.cls; gt.lp.assign((size_t)NB * G, 65535); gt.pref.clear();
        for (int b = 0; b < NB; ++b) {
            const u32* row = ps.n.data() + (size_t)b * G;
            double Nb = (double)ps.N[b];
            for (u32 g = 0; g < G; ++g) {
                double p0 = ct.p0[ct.cls[g] * NB + b];
                double p = (row[g] + shrink * p0) / (Nb + shrink);
                gt.lp[(size_t)b * G + g] = (u16)quant_code(p);
            }
        }
        split_f0.assign(m, 1);
        for (u32 g = 0; g < G; ++g) {
            u32 a = gm.first[g], e = gm.first[g + 1], n = e - a;
            if (n == 1) continue;
            double tg = 0; for (u32 r = a; r < e; ++r) tg += T[r];
            for (u32 r = a; r < e; ++r) {
                double p = (T[r] + alpha) / (tg + alpha * n);
                double v = std::llround(p * prior_mass);
                split_f0[r] = v < 1 ? 1u : (u32)v;
            }
        }
        model_bytes = gt.bytes() + (size_t)m;      // 16-bit presence codes + class bytes + ~1 byte/interval split prior
        built = true;
    }
};

}}  // namespace cs::i3
