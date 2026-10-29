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

namespace cs { namespace i2 {

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
        u64 t = target(16);
        int b = t < (u64)p1_16;
        if (b) dec(0, (u64)p1_16); else dec((u64)p1_16, (u64)(65536 - p1_16));
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

}}  // namespace cs::i2
