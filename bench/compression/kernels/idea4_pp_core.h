// SPDX-License-Identifier: MIT
// Idea 4 core: deterministic integer exp(-x), a 64-bit rANS coder and small adaptive models.
// Nothing on the decode path uses floating point. Tables are generated at start-up by exact
// 128-bit integer Taylor series, so every machine produces identical tables.
#pragma once
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

namespace cs { namespace ipp {

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

// ---------------------------------------------------------------------------------------------
// exp(-D): D in Q32 (u64). Returns exp(-D) in Q30 (0..2^30).
// ---------------------------------------------------------------------------------------------
struct ExpTab {
    u64 e_int[32];     // exp(-n)        Q62
    u64 e_hi[1024];    // exp(-i/1024)   Q62
    ExpTab() {
        // Taylor series in Q62 with u128 products; arguments x in [0,1].
        auto exp_neg = [](u64 xq /*Q62, <= 2^62*/) -> u64 {
            // sum_{n} (-x)^n / n!  accumulate positive and negative parts separately
            u128 term = (u128)1 << 62;            // x^0/0!
            u128 pos = term, neg = 0;
            for (int n = 1; n < 40; ++n) {
                term = (term * (u128)xq) >> 62;   // x^n/(n-1)! scaled
                term /= (u128)n;
                if (n & 1) neg += term; else pos += term;
                if (term == 0) break;
            }
            return (u64)(pos - neg);
        };
        e_int[0] = (u64)1 << 62;
        u64 e1 = exp_neg((u64)1 << 62);           // exp(-1) in Q62
        for (int n = 1; n < 32; ++n) e_int[n] = (u64)(((u128)e_int[n - 1] * (u128)e1) >> 62);
        for (int i = 0; i < 1024; ++i) e_hi[i] = exp_neg((u64)i << 52);
    }
};
inline const ExpTab& exp_tab() { static const ExpTab t; return t; }

// exp(-D) in Q62 for D in Q32; 0 when D >= 32.
inline u64 exp_neg_q62(u64 D) {
    u64 n = D >> 32;
    if (n >= 32) return 0;
    const ExpTab& T = exp_tab();
    u64 f = D & 0xFFFFFFFFull;
    u64 hi = f >> 22, r = f & 0x3FFFFFull;
    u64 rq = r << 30;                                   // Q62
    u64 r2 = (u64)(((u128)rq * rq) >> 62);
    u64 r3 = (u64)(((u128)r2 * rq) >> 62);
    u64 er = ((u64)1 << 62) - rq + (r2 >> 1) - r3 / 6;  // exp(-r) Q62
    u64 a = (u64)(((u128)T.e_int[n] * T.e_hi[hi]) >> 62);
    return (u64)(((u128)a * er) >> 62);
}
// exp(-D) in Q30; 0 when D >= 22 (value below one Q30 ulp).
inline u32 exp_neg_q30(u64 D) {
    if ((D >> 32) >= 22) return 0;
    return (u32)(exp_neg_q62(D) >> 32);
}

// exp2(x/256)-style growth table for the depth scale: scale_q32(s) = round(2^((s-s0)/R)) in Q32,
// built by repeated integer multiplication with a root-of-two constant derived by integer Newton.
// (Only needs to be deterministic, accuracy is irrelevant: the encoder uses the same table.)
inline u64 isqrt_u128(u128 v) {
    u64 x = 0;
    for (int b = 63; b >= 0; --b) {
        u64 c = x | ((u64)1 << b);
        if ((u128)c * c <= v) x = c;
    }
    return x;
}
// 2^(1/R) in Q62 for R a power of two (repeated square roots of 2).
inline u64 root2_q62(int R) {
    u64 x = (u64)2 << 62 >> 0;           // 2.0 in Q62 would overflow (2^63 fits u64): ok
    x = (u64)1 << 63;                    // 2.0 * 2^62
    for (int r = R; r > 1; r >>= 1) {
        // sqrt in Q62: sqrt(x * 2^62)
        x = isqrt_u128((u128)x << 62);
    }
    return x;
}

// ---------------------------------------------------------------------------------------------
// rANS64 (Fabian Giesen style). State in [2^31, 2^63). Output words are 32 bit.
// ---------------------------------------------------------------------------------------------
struct Sym { u64 packed; };   // start:28 | freq:28 | bits:8
inline Sym mk(u32 start, u32 freq, int bits) { return Sym{(u64)start | ((u64)freq << 28) | ((u64)bits << 56)}; }

struct RansEnc {
    std::vector<Sym> syms;
    void put(u32 start, u32 freq, int bits) {
#ifndef NDEBUG
        if (freq == 0 || (u64)start + freq > ((u64)1 << bits)) throw std::runtime_error("rans: bad symbol");
#endif
        syms.push_back(mk(start, freq, bits));
    }
    // raw bits (uniform), up to 16 per call
    void put_bits(u32 v, int nb) {
        while (nb > 0) { int c = nb > 16 ? 16 : nb; put(v & ((1u << c) - 1), 1, c); v >>= c; nb -= c; }
    }
    std::vector<uint8_t> finish() {
        std::vector<u32> w;
        const u64 L = (u64)1 << 31;
        u64 x = L;
        for (size_t i = syms.size(); i-- > 0;) {
            u64 p = syms[i].packed;
            u32 start = (u32)(p & 0xFFFFFFF), freq = (u32)((p >> 28) & 0xFFFFFFF); int bits = (int)(p >> 56);
            u64 x_max = ((L >> bits) << 32) * freq;
            if (x >= x_max) { w.push_back((u32)x); x >>= 32; }
            x = ((x / freq) << bits) + (x % freq) + start;
        }
        w.push_back((u32)(x >> 32)); w.push_back((u32)x);   // flush: decoder reads hi first
        std::vector<uint8_t> out(w.size() * 4);
        // words were produced last-first; reverse so decoder reads forward
        for (size_t i = 0; i < w.size(); ++i) { u32 v = w[w.size() - 1 - i]; std::memcpy(&out[i * 4], &v, 4); }
        return out;
    }
};

struct RansDec {
    const uint8_t* p = nullptr; const uint8_t* end = nullptr;
    u64 x = 0;
    void init(const uint8_t* b, size_t n) {
        p = b; end = b + n;
        u32 lo = rd(); u32 hi = rd();
        x = ((u64)hi << 32) | lo;
    }
    u32 rd() {
        if (p + 4 > end) throw std::runtime_error("rans: out of data");
        u32 v; std::memcpy(&v, p, 4); p += 4; return v;
    }
    inline u32 slot(int bits) const { return (u32)(x & (((u64)1 << bits) - 1)); }
    inline void advance(u32 start, u32 freq, int bits) {
        x = (u64)freq * (x >> bits) + (x & (((u64)1 << bits) - 1)) - start;
        if (x < ((u64)1 << 31)) x = (x << 32) | rd();
    }
    u32 get_bits(int nb) {
        u32 v = 0; int sh = 0;
        while (nb > 0) { int c = nb > 16 ? 16 : nb; u32 s = slot(c); advance(s, 1, c); v |= s << sh; sh += c; nb -= c; }
        return v;
    }
};

// ---------------------------------------------------------------------------------------------
// Adaptive frequency model over a small alphabet (<= 64 symbols), deterministic integer updates.
// Frequencies are renormalised to a power-of-two total (2^12) on demand so rANS can use them.
// ---------------------------------------------------------------------------------------------
struct AdaptModel {
    static const int SB = 12;                // scale bits of the coding table
    int n = 0;
    std::vector<u32> cnt;                    // raw counts
    std::vector<u32> cum;                    // cumulative table (n+1), total = 2^SB, rebuilt lazily
    u32 total = 0; int inc = 24, limit = 1 << 16; bool dirty = true;
    void init(int nsym, int init_cnt = 1, int increment = 24, int lim = 1 << 14) {
        n = nsym; cnt.assign(n, init_cnt); cum.assign(n + 1, 0); total = init_cnt * n; inc = increment; limit = lim; dirty = true;
    }
    void rebuild() {
        // freq_i = max(1, floor(cnt_i * (2^SB - n) / total)) + distribute remainder to largest
        u64 budget = ((u64)1 << SB) - n;
        u64 acc = 0; cum[0] = 0;
        std::vector<u32> f(n);
        for (int i = 0; i < n; ++i) { f[i] = 1 + (u32)((u64)cnt[i] * budget / total); acc += f[i]; }
        int big = (int)(std::max_element(cnt.begin(), cnt.end()) - cnt.begin());
        f[big] += (u32)(((u64)1 << SB) - acc);
        for (int i = 0; i < n; ++i) cum[i + 1] = cum[i] + f[i];
        dirty = false;
    }
    double bits(int s) { if (dirty) rebuild(); return (double)SB - std::log2((double)(cum[s + 1] - cum[s])); }
    int mode() const { return (int)(std::max_element(cnt.begin(), cnt.end()) - cnt.begin()); }
    void get(int s, u32& start, u32& freq) { if (dirty) rebuild(); start = cum[s]; freq = cum[s + 1] - cum[s]; }
    int find(u32 slot, u32& start, u32& freq) {
        if (dirty) rebuild();
        int lo = 0, hi = n - 1;
        while (lo < hi) { int mid = (lo + hi + 1) >> 1; if (cum[mid] <= slot) lo = mid; else hi = mid - 1; }
        start = cum[lo]; freq = cum[lo + 1] - cum[lo]; return lo;
    }
    void update(int s) {
        cnt[s] += inc; total += inc;
        if (total > (u32)limit) { total = 0; for (auto& c : cnt) { c = (c + 1) >> 1; total += c; } }
        dirty = true;
    }
};

}}  // namespace cs::ipp
