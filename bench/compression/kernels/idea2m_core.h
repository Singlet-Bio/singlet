// SPDX-License-Identifier: MIT
// Idea 2 (the file is its own model): integer-only building blocks.
//   * integer exp (no libm on any decode path), logistic stretch/squash tables, Poisson detection table
//   * 64-bit carry-less range coder (Subbotin), shared Enc/Dec interface (bit with 16-bit prob, symbol with 2^32 total)
//   * adaptive counters, adaptive Elias-gamma number coder, small-alphabet symbol coder
//   * order-statistic coder: min of r iid draws from a (static) integer mass table, exact multinomial / set coding
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif
namespace cs { namespace i2 {

inline unsigned long long thread_cycles() {
#ifdef _WIN32
    ULONG64 c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); return c;
#else
    return 0;
#endif
}
inline double cpu_time_ms() {
#ifdef _WIN32
    FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    return (((unsigned long long)u.dwHighDateTime << 32) | u.dwLowDateTime) / 1e4;
#else
    return 0;
#endif
}


using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using u64 = uint64_t;
using i32 = int32_t; using i64 = int64_t;
using u128 = unsigned __int128;

// ---------------------------------------------------------------------------------------------
// integer exp:  exp(x) * 2^64 for x given in Q32 (|x| <= 40). Pure integer arithmetic.
inline u128 exp_q64(i64 x_q32) {
    constexpr i64 LN2 = 2977044472LL;  // ln 2 * 2^32 (rounded)
    i64 n = x_q32 / LN2;
    i64 rem = x_q32 - n * LN2;
    if (rem < 0) { rem += LN2; --n; }
    u128 term = (u128)1 << 64, sum = term;
    for (int k = 1; k <= 26; ++k) {
        term = (term * (u64)rem) >> 32;
        term /= (u32)k;
        sum += term;
    }
    if (n >= 0) return sum << n;
    if (-n >= 127) return 0;
    return sum >> (-n);
}

// ---------------------------------------------------------------------------------------------
// tables
struct Tabs {
    static constexpr int XS = 3071;           // stretch domain clamp, Q8 (+-12.0)
    static constexpr int OFF = 24 * 1024;     // Poisson table: index = L*1024 + OFF, L in [-24, 8)
    static constexpr int NPT = 32 * 1024;
    u16 squash_t[2 * XS + 1];
    i32 stretch_t_dummy;                      // keep alignment simple
    int16_t stretch_t[65536];
    u32 ptab[NPT];                            // Q.32 : 1 - exp(-exp(L))
    u32 rec[1024];                            // 65536 / (n + 1.5)
    float nl2[65536];                         // -log2(p / 65536), encoder-side cost accounting only
    Tabs() {
        nl2[0] = 16.f;
        for (int p = 1; p < 65536; ++p) nl2[p] = (float)-std::log2(p / 65536.0);
        for (int i = 0; i <= 2 * XS; ++i) {
            int x = i - XS;
            u128 e = exp_q64(-(i64)x * (1 << 24));     // exp(-x/256)
            u128 num = (u128)65536 << 64;
            u128 den = ((u128)1 << 64) + e;
            u64 p = (u64)(num / den);
            if (p < 1) p = 1;
            if (p > 65535) p = 65535;
            squash_t[i] = (u16)p;
        }
        int x = -XS;
        stretch_t[0] = (int16_t)-XS;
        for (int p = 1; p < 65536; ++p) {
            while (x < XS && squash_t[x + XS] < p) ++x;
            stretch_t[p] = (int16_t)x;
        }
        for (int i = 0; i < NPT; ++i) {
            i64 L_q32 = (i64)(i - OFF) * (1 << 22);
            u128 lam = exp_q64(L_q32);                 // Q.64
            u64 p32;
            if (lam > ((u128)40 << 64)) {
                p32 = 0xFFFFFFFFull;
            } else {
                i64 lam_q32 = (i64)(lam >> 32);
                if (lam_q32 < (1ll << 8)) {
                    p32 = (u64)(lam >> 32);            // 1-exp(-l) ~ l for tiny l
                } else {
                    u128 e = exp_q64(-lam_q32);
                    u128 pq = ((u128)1 << 64) - e;
                    p32 = (u64)(pq >> 32);
                    if (p32 > 0xFFFFFFFFull) p32 = 0xFFFFFFFFull;
                }
            }
            ptab[i] = (u32)p32;
        }
        for (int n = 0; n < 1024; ++n) rec[n] = (u32)(131072u / (2u * n + 3u));
    }
};
inline const Tabs& TB() { static Tabs t; return t; }

inline int squash(int x) {   // Q8 stretch -> p16
    if (x > Tabs::XS) x = Tabs::XS;
    if (x < -Tabs::XS) x = -Tabs::XS;
    return TB().squash_t[x + Tabs::XS];
}
inline int stretch(int p16) { return TB().stretch_t[p16 < 1 ? 1 : (p16 > 65535 ? 65535 : p16)]; }

// ---------------------------------------------------------------------------------------------
// range coder (Subbotin carry-less, 64-bit state).  cum/freq with total 2^tb, tb <= 32.
struct RcEnc {
    static constexpr bool ENC = true;
    static constexpr u64 TOP = 1ull << 56, BOT = 1ull << 48;
    std::vector<u8>& out;
    u64 low = 0, range = ~0ull;
    double cost[24] = {};
    int cat = 0;
    explicit RcEnc(std::vector<u8>& o) : out(o) {}
    inline void norm() {
        while (((low ^ (low + range)) < TOP) || (range < BOT && ((range = (0 - low) & (BOT - 1)), true))) {
            out.push_back((u8)(low >> 56));
            low <<= 8; range <<= 8;
        }
    }
    inline void enc(u64 cum, u64 freq, int tb) {
        range >>= tb;
        low += cum * range;
        range *= freq;
        norm();
    }
    inline int bit(int p1_16, int b) {      // p1_16 = P(bit==1)*65536
        if (p1_16 < 1) p1_16 = 1;
        if (p1_16 > 65535) p1_16 = 65535;
        if (b) { enc(0, (u64)p1_16, 16); cost[cat] += TB().nl2[p1_16]; }
        else   { enc((u64)p1_16, (u64)(65536 - p1_16), 16); cost[cat] += TB().nl2[65536 - p1_16]; }
        return b;
    }
    inline void enc_tot(u64 cum, u64 freq, u64 tot) { range /= tot; low += cum * range; range *= freq; norm(); }
    inline u32 uniform(u32 v, u32 tot) { if (tot > 1) { enc_tot(v, 1, tot); cost[cat] += std::log2((double)tot); } return v; }
    inline int rawbit(int b) { enc((u64)b, 1, 1); cost[cat] += 1.0; return b; }
    inline void sym(u64 cum, u64 freq, int tb) { enc(cum, freq, tb); cost[cat] -= std::log2((double)freq / (double)(1ull << tb)); }
    void finish() { for (int i = 0; i < 8; ++i) { out.push_back((u8)(low >> 56)); low <<= 8; } }
};

struct RcDec {
    static constexpr bool ENC = false;
    static constexpr u64 TOP = 1ull << 56, BOT = 1ull << 48;
    const u8* p; const u8* end;
    u64 low = 0, range = ~0ull, code = 0;
    double cost[24] = {};
    int cat = 0;
    RcDec(const u8* b, const u8* e) : p(b), end(e) { for (int i = 0; i < 8; ++i) code = (code << 8) | nextb(); }
    inline u8 nextb() { return p < end ? *p++ : 0; }
    inline void norm() {
        while (((low ^ (low + range)) < TOP) || (range < BOT && ((range = (0 - low) & (BOT - 1)), true))) {
            code = (code << 8) | nextb();
            low <<= 8; range <<= 8;
        }
    }
    inline u64 target(int tb) {
        range >>= tb;
        u64 t = (code - low) / range;
        u64 tot = 1ull << tb;
        return t >= tot ? tot - 1 : t;
    }
    inline void dec(u64 cum, u64 freq) {
        low += cum * range;
        range *= freq;
        norm();
    }
    inline int bit(int p1_16, int) {
        if (p1_16 < 1) p1_16 = 1;
        if (p1_16 > 65535) p1_16 = 65535;
        u64 r = range >> 16;
        u64 bound = r * (u64)p1_16;
        int b;
        if (code - low < bound) { range = bound; b = 1; }
        else { low += bound; range = r * (u64)(65536 - p1_16); b = 0; }
        norm();
        return b;
    }
    inline int rawbit(int) { u64 t = target(1); dec(t, 1); return (int)t; }
    inline u32 uniform(u32, u32 tot) {
        if (tot <= 1) return 0;
        range /= tot;
        u64 t = (code - low) / range;
        if (t >= tot) t = tot - 1;
        dec(t, 1);
        return (u32)t;
    }
};

// ---------------------------------------------------------------------------------------------
// adaptive probability counter: state = p22 << 10 | n ; update rate 1/(n+1.5) until n reaches the limit.
inline u32 ctr_init(int p16, int n0) { if (p16 < 1) p16 = 1; if (p16 > 65535) p16 = 65535; return ((u32)p16 << 16) | (u32)n0; }
inline int ctr_p16(u32 s) { int p = (int)(s >> 16); return p < 1 ? 1 : (p > 65535 ? 65535 : p); }
inline void ctr_upd(u32& s, int bit, int limit) {
    u32 n = s & 1023;
    i32 p22 = (i32)(s >> 10);
    i32 tgt = bit ? 4194303 : 0;
    p22 += (i32)(((i64)(tgt - p22) * TB().rec[n]) >> 16);
    if ((int)n < limit) ++n;
    s = ((u32)p22 << 10) | n;
}

// simple shift-adaptive bit model
struct BitM {
    u16 p = 32768;
};
template <class RC>
inline int code_bitm(RC& rc, BitM& m, int b, int rate = 5) {
    b = rc.bit(m.p, b);
    if (b) m.p += (65535 - m.p) >> rate; else m.p -= m.p >> rate;
    return b;
}

// Adaptive Elias-gamma number coder with nctx contexts. v >= 0.
struct NumCoder {
    int nctx;
    std::vector<BitM> un, mt;
    explicit NumCoder(int nc = 1) : nctx(nc), un((size_t)nc * 66), mt((size_t)nc * 64 * 8) {}
    template <class RC>
    u64 code(RC& rc, int ctx, u64 v) {
        u64 vv = RC::ENC ? v + 1 : 0;
        int nb = RC::ENC ? 63 - __builtin_clzll(vv) : 0;
        BitM* U = &un[(size_t)ctx * 66];
        int i = 0;
        for (;; ++i) {
            int b = code_bitm(rc, U[i], RC::ENC ? (i < nb) : 0, 4);
            if (!b) break;
            if (i >= 62) { ++i; break; }
        }
        int nbd = i;
        u64 r = 1;
        BitM* Mm = &mt[(size_t)ctx * 64 * 8 + (size_t)nbd * 8];
        int node = 1;
        for (int k = nbd - 1; k >= 0; --k) {
            int bb = RC::ENC ? (int)((vv >> k) & 1) : 0;
            if (node < 8 && (nbd - 1 - k) < 3) {
                bb = code_bitm(rc, Mm[node], bb, 5);
                node = node * 2 + bb;
            } else {
                bb = rc.rawbit(bb);
            }
            r = (r << 1) | (u64)bb;
        }
        return r - 1;
    }
};

// symbol coder over alphabet [0, 2^bits) with a binary context tree.
struct SymCoder {
    int bits; std::vector<BitM> t;
    explicit SymCoder(int b = 1, int nctx = 1) : bits(b), t((size_t)nctx << b) {}
    template <class RC>
    u32 code(RC& rc, int ctx, u32 s) {
        size_t base = (size_t)ctx << bits;
        u32 node = 1;
        for (int k = bits - 1; k >= 0; --k) {
            int b = RC::ENC ? (int)((s >> k) & 1) : 0;
            b = code_bitm(rc, t[base + node], b, 4);
            node = node * 2 + (u32)b;
        }
        return node - (1u << bits);
    }
};

// ---------------------------------------------------------------------------------------------
// Order-statistic coder.  Mass table Q[a] = sum_{t>=a} W_t (non-increasing, Q[end] = 0, Q[a] > 0 for a < end).
// Codes x in [lo, end) = the minimum of r iid draws from the distribution proportional to W restricted to [lo, end).
// Applied r times with lo' = x+1 (sets) or lo' = x (multisets) this codes a set / multiset of size r under an iid model.
inline u64 mulq62(u64 a, u64 b) { return (u64)(((u128)a * b) >> 62); }
inline u64 powq62(u64 rho, u32 r) {
    u64 res = 1ull << 62, base = rho;
    while (r) {
        if (r & 1) res = mulq62(res, base);
        r >>= 1;
        if (r) base = mulq62(base, base);
    }
    return res;
}
struct OSCtx {
    const u64* Q; u32 lo, end, r; u128 inv; u64 scale;
    OSCtx(const u64* q, u32 lo_, u32 end_, u32 r_) : Q(q), lo(lo_), end(end_), r(r_) {
        inv = ((u128)1 << 124) / (u128)Q[lo];
        scale = 0x100000000ull - (u64)(end - lo) - 1;
    }
    inline u64 C(u32 a) const {     // cumulative mass strictly below symbol a, total 2^32
        if (a == lo) return 0;
        u64 rho = (u64)((inv * (u128)Q[a]) >> 62);
        if (rho > (1ull << 62)) rho = 1ull << 62;
        u64 pw = powq62(rho, r);
        u64 c = ((1ull << 62) - pw) >> 30;
        c = (u64)(((u128)c * scale) >> 32) + (a - lo);
        return c;
    }
};
template <class RC>
inline u32 code_orderstat(RC& rc, const u64* Q, u32 lo, u32 end, u32 r, u32 x_in) {
    OSCtx cx(Q, lo, end, r);
    if constexpr (RC::ENC) {
        u64 c0 = cx.C(x_in), c1 = cx.C(x_in + 1);
        rc.sym(c0, c1 - c0, 32);
        return x_in;
    } else {
        u64 t = rc.target(32);
        u32 a = lo, b = end;            // find largest a in [lo, end-1] with C(a) <= t
        while (b - a > 1) {
            u32 mid = a + ((b - a) >> 1);
            if (cx.C(mid) <= t) a = mid; else b = mid;
        }
        u64 c0 = cx.C(a), c1 = cx.C(a + 1);
        rc.dec(c0, c1 - c0);
        return a;
    }
}


// ---------------------------------------------------------------------------------------------
// Fast order-statistic coder (log domain).  LQ[a] = log2(Q[a]) in Q.40 (non-increasing in a), Q[a] > 0 for a < end.
// P(min of r iid draws >= a | >= lo) = (Q[a]/Q[lo])^r = 2^-(r (LQ[lo]-LQ[a])).  Decoding inverts the CDF through the
// monotone array LQ (one integer log + galloping search verified with the forward function), not by evaluating
// powers in every probe.  Integer arithmetic only.
struct LogTabs {
    i64 lg[4098];          // log2(1 + i/4096) in Q.40
    u64 e2[258];           // 2^-(i/256) in Q.62
    LogTabs() {
        constexpr u64 INVLN2 = 0x5c551d94ae0bf800ull;       // 1/ln2 in Q.62
        for (int i = 0; i <= 4097; ++i) {
            u64 t = (u64)((((u128)i) << 62) / (u64)(8192 + i));     // x/(2+x), x = i/4096
            u64 t2 = (u64)(((u128)t * t) >> 62), pw = t, sum = 0;
            for (int k = 0; k < 24; ++k) { sum += pw / (u64)(2 * k + 1); pw = (u64)(((u128)pw * t2) >> 62); }
            u64 ln1p = 2 * sum;                                   // ln(1+x) in Q.62
            u64 l2 = (u64)(((u128)ln1p * INVLN2) >> 62);          // log2(1+x) Q.62
            lg[i] = (i64)(l2 >> 22);
        }
        lg[0] = 0; lg[4096] = 1ll << 40; lg[4097] = lg[4096];
        for (int i = 1; i < 4096; ++i) { if (lg[i] > (1ll << 40)) lg[i] = 1ll << 40; if (lg[i] < lg[i - 1]) lg[i] = lg[i - 1]; }
        e2[0] = 1ull << 62;
        u64 R = expm_series(1ull << 32);                      // e^{-ln2/256}
        for (int i = 0; i < 257; ++i) e2[i + 1] = mulq62(e2[i], R);
    }
    // e^{-d ln2 / 2^40} in Q.62 for d in [0, 2^32]
    static u64 expm_series(u64 d) {
        constexpr u64 LN2 = 0x2c5c85fdf473de00ull;
        u64 y = mulq62(d << 22, LN2);
        u64 y2 = mulq62(y, y), y3 = mulq62(y2, y);
        return (1ull << 62) + y2 / 2 - y - y3 / 6;                 // y <= 0.0028: truncation error < 3e-12 (below 2^-32)
    }
};
inline const LogTabs& LT() { static LogTabs t; return t; }

inline i64 ilog2_q40(u64 v) {            // v > 0
    int e = 63 - __builtin_clzll(v);
    u64 m = v << (63 - e);               // [2^63, 2^64)
    u64 f = m - (1ull << 63);            // Q.63 fraction
    u32 idx = (u32)(f >> 51);
    u64 rem = f & ((1ull << 51) - 1);
    const LogTabs& t = LT();
    i64 d = t.lg[idx + 1] - t.lg[idx];
    return ((i64)e << 40) + t.lg[idx] + (i64)(((u128)(u64)d * rem) >> 51);
}
inline i64 ilog2_q40_u128(u128 v) {      // v > 0
    u64 hi = (u64)(v >> 64);
    if (!hi) return ilog2_q40((u64)v);
    int b = 127 - __builtin_clzll(hi);       // msb position of v
    int sh = b - 63;
    return ilog2_q40((u64)(v >> sh)) + ((i64)sh << 40);
}
constexpr i64 LQ_NEG = -(1ll << 60);
// 2^-x in Q.62 for x >= 0 in Q.40
inline u64 exp2m_q62(u64 x) {
    u64 ip = x >> 40;
    if (ip >= 62) return 0;
    u64 frac = x & ((1ull << 40) - 1);
    u32 i = (u32)(frac >> 32);
    u64 d = frac & 0xFFFFFFFFull;                           // residual, Q.40, < 2^32
    u64 E = LogTabs::expm_series(d);
    return mulq62(LT().e2[i], E) >> ip;
}
#ifdef I2_COUNT
inline unsigned long long g_ccalls = 0;
#endif
struct OSFast {
    const i64* LQ; u32 lo, end, r; i64 lql; u64 scale;
    OSFast(const i64* lq, u32 lo_, u32 end_, u32 r_) : LQ(lq), lo(lo_), end(end_), r(r_) {
        lql = LQ[lo];
        scale = 0x100000000ull - 4ull * (u64)(end - lo) - 4;
    }
    inline u64 C(u32 a) const {          // cumulative mass strictly below symbol a (a in [lo, end]), total 2^32
#ifdef I2_COUNT
        ++g_ccalls;
#endif
        if (a == lo) return 0;
        u64 g;
        if (a >= end) g = 1ull << 62;
        else {
            i64 dls = lql - LQ[a]; u64 dl = dls < 0 ? 0 : (u64)dls;
            u128 xx = (u128)dl * r;
            u64 x = xx > ((u128)62 << 40) ? (62ull << 40) : (u64)xx;
            g = (1ull << 62) - exp2m_q62(x);
        }
        u64 c = g >> 30;
        return (u64)(((u128)c * scale) >> 32) + 4ull * (a - lo);
    }
};
struct LqTab {
    std::vector<i64> lq;       // lq[0..n], lq[n] = LQ_NEG
    std::vector<u32> idx;      // idx[k] = #positions a with lq[a] >= top - (k << sh)
    i64 top = 0; int sh = 0; u32 n = 0;
    void build(const u64* Q, u32 n_) {
        n = n_;
        lq.assign((size_t)n + 1, LQ_NEG);
        for (u32 a = 0; a < n; ++a) lq[a] = ilog2_q40(Q[a]);
        if (n == 0) { idx.assign(2, 0); return; }
        top = lq[0];
        i64 span = top - lq[n - 1];
        sh = 0;
        while ((span >> sh) >= 16384) ++sh;
        size_t K = (size_t)(span >> sh) + 3;
        idx.assign(K, 0);
        u32 a = 0;
        for (size_t k = 0; k < K; ++k) {
            i64 lim = top - ((i64)k << sh);
            while (a < n && lq[a] >= lim) ++a;
            idx[k] = a;
        }
    }
};
template <class RC>
inline u32 code_orderstat_fast(RC& rc, const LqTab& tab, u32 lo, u32 end, u32 r, u32 x_in) {
    const i64* LQ = tab.lq.data();
    OSFast cx(LQ, lo, end, r);
    if constexpr (RC::ENC) {
        u64 c0 = cx.C(x_in), c1 = cx.C(x_in + 1);
        rc.sym(c0, c1 - c0, 32);
        return x_in;
    } else {
        u64 t = rc.target(32);
        // approximate inverse of the unguarded model
        u64 u = (1ull << 32) - t;                             // in (0, 2^32]
        i64 xt = (i64)(32ll << 40) - ilog2_q40(u);            // -log2(u / 2^32) >= 0, Q.40
        i64 thr = cx.lql - xt / (i64)r;
        // largest a in [lo, end-1] with LQ[a] >= thr, bracketed by the log-grid index
        u32 a0 = lo, b0 = end;
        {
            i64 d = tab.top - thr;
            if (d < 0) d = 0;
            size_t k = (size_t)(d >> tab.sh);
            if (k + 1 < tab.idx.size()) {
                u32 lo_c = tab.idx[k] ? tab.idx[k] - 1 : 0;
                u32 hi_c = tab.idx[k + 1];                    // exclusive upper bound candidate (a* <= idx[k+1]-1)
                if (lo_c > a0) a0 = lo_c;
                if (hi_c < b0) b0 = hi_c;
                if (b0 <= a0) b0 = a0 + 1;
                if (b0 > end) b0 = end;
            }
        }
        // make sure the bracket is valid: LQ[a0] >= thr and (b0 == end or LQ[b0] < thr); otherwise widen
        if (LQ[a0] < thr) a0 = lo;
        if (b0 < end && LQ[b0] >= thr) b0 = end;
        while (b0 - a0 > 1) {
            u32 mid = a0 + ((b0 - a0) >> 1);
            if (LQ[mid] >= thr) a0 = mid; else b0 = mid;
        }
        u32 a = a0;
        u64 c0 = cx.C(a), c1 = cx.C(a + 1);
        if (c0 > t) {                                          // approximation too high: gallop down
            u32 hi = a, step = 1, lo2 = a;
            while (true) {
                lo2 = hi > lo + step ? hi - step : lo;
                if (lo2 == lo || cx.C(lo2) <= t) break;
                hi = lo2; step <<= 1;
            }
            while (hi - lo2 > 1) { u32 mid = lo2 + ((hi - lo2) >> 1); if (cx.C(mid) <= t) lo2 = mid; else hi = mid; }
            a = lo2; c0 = cx.C(a); c1 = cx.C(a + 1);
        } else if (a + 1 < end && c1 <= t) {                   // too low: gallop up
            u32 lo2 = a + 1, step = 1, hi = end;
            while (true) {
                u32 probe = lo2 + step;
                if (probe >= end) { hi = end; break; }
                if (cx.C(probe) > t) { hi = probe; break; }
                lo2 = probe; step <<= 1;
            }
            while (hi - lo2 > 1) { u32 mid = lo2 + ((hi - lo2) >> 1); if (cx.C(mid) <= t) lo2 = mid; else hi = mid; }
            a = lo2; c0 = cx.C(a); c1 = cx.C(a + 1);
        }
        rc.dec(c0, c1 - c0);
        return a;
    }
}

}}  // namespace cs::i2
