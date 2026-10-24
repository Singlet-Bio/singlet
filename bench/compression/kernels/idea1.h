// SPDX-License-Identifier: MIT
// Idea 1 + Idea 5 (brainstorm_fable.md): survival-function gap coding with a trained, out-of-sample
// per-feature x depth presence model and rANS.  Owned by the "Idea 1" agent; see notes/idea1.md.
//
// Representation of one column (cell), decoded left to right:
//   [depth bucket b]  then for every non-zero:  [next present feature j][count-1 | escape]  then [END]
// The presence model gives every feature t a probability p_t(b); the log-survival axis is
//   L_b[j] = sum_{t<j} -log2(1-p_t(b))      (fixed point, FR fractional bits, u64)
// so that P(next present = j | previous present = i) = prod_{i<t<j}(1-p_t) p_j
//   = 2^-(L[j]-L[i+1]) - 2^-(L[j+1]-L[i+1]).
// Coding j is therefore coding a draw of E ~ Exp(1) (in bits) on the L axis: the decoder turns the rANS
// slot into E with one integer log2, locates j by searching the (monotone) L table, and then evaluates
// the exact integer CDF  F(x) = M - G(x),  G(x) = 2^-x scaled to 2^31 (table + linear interpolation),
// at j and j+1 to obtain (cum, freq).  Encoder and decoder share G(), so the code is exact.
// Positions beyond XCUT (16 bits of surprise) or whose interval rounds to zero width use an escape
// symbol followed by the position coded uniformly.  END of column is position m.
//
// The feature order is the matrix's own row order (no frequency permutation is needed: the product
// model has the same probability in any order).  All decode-path arithmetic is integer.
#pragma once
#ifndef IDEA1_LIGHT
#include "common.h"
#include "ref_zstd_csc.h"
#else
// light mode (micro-benchmarks, no singlet/zstd headers): just what idea1.h itself needs
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
namespace cs {
inline uint64_t fnv64(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
inline std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::vector<uint8_t> buf(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return buf;
}
}  // namespace cs
#endif
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <memory>
#include <mutex>

namespace cs {
namespace i1 {

constexpr int FR = 24;                         // fractional bits of the log-survival axis
constexpr int MB = 31;                         // gap-symbol probability scale (bits)
constexpr uint64_t MSC = 1ull << MB;
constexpr int XB = 16;                         // escape beyond XB bits of surprise
constexpr uint64_t XCUT = (uint64_t)XB << FR;
constexpr uint64_t FCUT = MSC - (MSC >> XB);   // = M - G(XCUT)
constexpr uint64_t RL = 1ull << 31;            // rANS lower bound (64-bit state, 32-bit words)
constexpr int CB = 14;                         // count-symbol probability scale (bits)
constexpr int BB = 12;                         // bucket-symbol probability scale (bits)
constexpr uint64_t INF = 1ull << 60;
constexpr int NSYM = 16;                       // count alphabet: 1..15, 16 = escape (>= 16)

// ------------------------------------------------------------------------------------------------
// Integer tables (2^-x and log2).  Built once with long double; a checksum goes into every stream so a
// platform whose libm rounds differently is detected rather than silently mis-decoding.
struct Tabs {
    uint32_t ex[1025];   // round(2^31 * 2^(-k/1024))
    uint32_t lg[1025];   // round(2^FR * log2(1 + k/1024))
    uint64_t sum = 0;
    Tabs() {
        for (int k = 0; k <= 1024; ++k) {
            ex[k] = (uint32_t)llroundl(ldexpl(exp2l(-(long double)k / 1024.0L), 31));
            lg[k] = (uint32_t)llroundl(ldexpl(log2l(1.0L + (long double)k / 1024.0L), FR));
        }
        sum = fnv64(ex, sizeof ex);
        sum = fnv64(lg, sizeof lg, sum);
    }
};
inline const Tabs& tabs() { static const Tabs t; return t; }

// G(x) = 2^-(x / 2^FR) * 2^31, with x clamped to XCUT.
inline uint32_t G(const Tabs& T, uint64_t x) {
    if (x > XCUT) x = XCUT;
    unsigned xi = (unsigned)(x >> FR);
    uint32_t f = (uint32_t)(x & ((1u << FR) - 1));
    uint32_t k = f >> (FR - 10), r = f & ((1u << (FR - 10)) - 1);
    uint64_t a = T.ex[k], b = T.ex[k + 1];
    uint32_t v = (uint32_t)(a - (((a - b) * r) >> (FR - 10)));
    return v >> xi;
}
// log2(rem) in FR fixed point, rem in [1, 2^32).
inline uint64_t log2fix(const Tabs& T, uint32_t rem) {
    unsigned l = 31 - __builtin_clz(rem);
    uint64_t mant = (uint64_t)rem << (31 - l);       // [2^31, 2^32)
    uint32_t f = (uint32_t)(mant - (1ull << 31));    // 31 bits
    uint32_t k = f >> 21, r = f & ((1u << 21) - 1);
    uint64_t a = T.lg[k], b = T.lg[k + 1];
    return ((uint64_t)l << FR) + a + (((b - a) * r) >> 21);
}

// ------------------------------------------------------------------------------------------------
// Model
struct Model {
    std::string tag;                   // identifies the file the stream was coded against
    uint32_t m = 0, nb = 0, ncls = 0, cnb = 0, type = 0;
    uint32_t edges[18] = {0};          // bucket b holds edges[b] <= D < edges[b+1]
    uint64_t rnhash = 0;
    uint32_t step_lut[256];
    std::vector<uint8_t> codes;        // type 0: nb * m (bucket-major) step codes; type 1: m 'a' codes
    uint32_t A_lut[256];               // type 1: per-feature rate LUT (unit 2^-28 bits at the reference bucket)
    std::vector<uint32_t> Cb;          // type 1: per-bucket multiplier (unit 2^-31)
    std::vector<uint64_t> S;           // type 1: prefix sums of A, m+1 entries
    std::vector<uint8_t> cls;          // m        feature class for the count model
    std::vector<uint16_t> cfreq;       // ncls*cnb*16, each row sums to 2^CB
    size_t file_bytes = 0;
    // derived
    std::vector<uint64_t> Ld;          // nb*(m+2)
    std::vector<uint16_t> ccum;        // ncls*cnb*17
    int pos_bits = 0;

    uint32_t bucket_of(uint64_t D) const {
        if (nb <= 1) return 0;
        uint32_t b = 0;
        while (b + 1 < nb && D >= edges[b + 1]) ++b;
        return b;
    }
    const uint64_t* L(uint32_t b) const { return Ld.data() + (size_t)b * (m + 2); }
    uint32_t crow(uint32_t j, uint32_t b) const { return ((uint32_t)cls[j] * cnb + (cnb > 1 ? b : 0)) * 17u; }
};

inline uint64_t names_hash(const std::vector<std::string>& v) {
    uint64_t h = 1469598103934665603ull;
    for (auto& s : v) { h = fnv64(s.data(), s.size(), h); h ^= 0; h *= 1099511628211ull; }
    return h;
}

inline std::shared_ptr<Model> load_model(const std::string& path) {
    static std::mutex mu;
    static std::map<std::string, std::shared_ptr<Model>> cache;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(path);
    if (it != cache.end()) return it->second;
    auto buf = read_file(path);
    auto M = std::make_shared<Model>();
    M->file_bytes = buf.size();
    M->tag = path;
    const uint8_t* p = buf.data();
    if (buf.size() < 8 || std::memcmp(p, "IDEA1M02", 8) != 0) throw std::runtime_error("bad idea1 model " + path);
    p += 8;
    auto u32 = [&]() { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; };
    M->m = u32(); M->nb = u32(); M->ncls = u32(); M->cnb = u32(); M->type = u32();
    for (int i = 0; i < 18; ++i) M->edges[i] = u32();
    { uint64_t v; std::memcpy(&v, p, 8); p += 8; M->rnhash = v; }
    if (M->type == 0) {
        for (int i = 0; i < 256; ++i) M->step_lut[i] = u32();
        M->codes.assign(p, p + (size_t)M->nb * M->m); p += (size_t)M->nb * M->m;
    } else {
        for (int i = 0; i < 256; ++i) M->A_lut[i] = u32();
        M->Cb.resize(M->nb);
        for (uint32_t b = 0; b < M->nb; ++b) M->Cb[b] = u32();
        M->codes.assign(p, p + M->m); p += M->m;
    }
    M->cls.assign(p, p + M->m); p += M->m;
    size_t nc = (size_t)M->ncls * M->cnb * NSYM;
    M->cfreq.resize(nc);
    std::memcpy(M->cfreq.data(), p, nc * 2); p += nc * 2;
    if ((size_t)(p - buf.data()) != buf.size()) throw std::runtime_error("idea1 model size mismatch " + path);
    // derived tables (integer only)
    const uint32_t m = M->m;
    M->Ld.assign((size_t)M->nb * (m + 2), 0);
    if (M->type == 0) {
        for (uint32_t b = 0; b < M->nb; ++b) {
            uint64_t* L = M->Ld.data() + (size_t)b * (m + 2);
            const uint8_t* c = M->codes.data() + (size_t)b * m;
            uint64_t acc = 0;
            for (uint32_t t = 0; t < m; ++t) { L[t] = acc; acc += M->step_lut[c[t]]; }
            L[m] = acc; L[m + 1] = INF;
        }
    } else {
        M->S.assign(m + 1, 0);
        uint64_t acc = 0;
        for (uint32_t t = 0; t < m; ++t) { M->S[t] = acc; acc += M->A_lut[M->codes[t]]; }
        M->S[m] = acc;
        for (uint32_t b = 0; b < M->nb; ++b) {
            uint64_t* L = M->Ld.data() + (size_t)b * (m + 2);
            for (uint32_t t = 0; t <= m; ++t) L[t] = (uint64_t)(((unsigned __int128)M->S[t] * M->Cb[b]) >> 35);
            L[m + 1] = INF;
        }
    }
    M->ccum.assign((size_t)M->ncls * M->cnb * 17, 0);
    for (size_t r = 0; r < (size_t)M->ncls * M->cnb; ++r) {
        uint32_t acc = 0;
        for (int s = 0; s < NSYM; ++s) { M->ccum[r * 17 + s] = (uint16_t)acc; acc += M->cfreq[r * NSYM + s]; }
        if (acc != (1u << CB)) throw std::runtime_error("count row does not sum to 2^CB");
        M->ccum[r * 17 + 16] = 0;   // unused: cum_at() supplies 2^CB for index 16
    }
    int pb = 1;
    while ((1u << pb) < m + 1) ++pb;
    M->pos_bits = pb;
    cache[path] = M;
    return M;
}

// 16384 does not fit u16: store cum[0..15] in u16 and treat cum[16] as 2^CB.
inline uint32_t cum_at(const uint16_t* row, int s) { return s >= NSYM ? (1u << CB) : row[s]; }

// ------------------------------------------------------------------------------------------------
// Context set by the benchmark driver (which study is held out, where models live).
struct Ctx {
    std::string hold = "ins";                       // held-out GSE id, or "ins" for the in-sample model
    std::string dir = "results/idea1_models";
    std::string sel_hold = "ins";                   // model chosen for this file by the encoder-side selection (codec idea1_s5r)
    uint64_t names_bytes = 0;                       // last encode(): bytes of the name/kv streams
    uint64_t model_bytes = 0;                       // last encode(): size of the model file used
    uint64_t side_bytes = 0;                        // last encode(): bytes spent on bucket symbols (estimate)
    double dec_payload_ms_min = 1e300;              // decode(): min payload-only time since reset
    void reset_timing() { dec_payload_ms_min = 1e300; }
};
inline Ctx& ctx() { static Ctx c; return c; }

inline std::string org_member(uint32_t m, bool& ok) {
    ok = true;
    switch (m) {
        case 310797: return "human_exon";
        case 272191: return "human_intron";
        case 386028: return "mouse_exon";
        case 307730: return "mouse_intron";
    }
    ok = false;
    return "";
}

// ------------------------------------------------------------------------------------------------
// rANS, 64-bit state, 32-bit renormalisation.
struct Sym { uint32_t cum, freq; uint8_t bits; };

struct Enc {
    uint64_t x = RL;
    std::vector<uint32_t> out;
    inline void put(const Sym& s) {
        uint64_t xmax = ((RL >> s.bits) << 32) * s.freq;
        while (x >= xmax) { out.push_back((uint32_t)x); x >>= 32; }
        x = ((x / s.freq) << s.bits) + (x % s.freq) + s.cum;
    }
    void flush() { out.push_back((uint32_t)x); out.push_back((uint32_t)(x >> 32)); }
};

struct Dec {
    uint64_t x;
    const uint32_t* p;
    inline void adv(uint32_t cum, uint32_t freq, int bits) {
        x = (uint64_t)freq * (x >> bits) + (x & ((1ull << bits) - 1)) - cum;
        if (x < RL) x = (x << 32) | *p++;
    }
};

// ------------------------------------------------------------------------------------------------
// Symbol construction (encoder side).
inline void put_gap(std::vector<Sym>& v, const Tabs& T, const Model& md, const uint64_t* L, uint32_t i1,
                    uint32_t j) {
    uint64_t x0 = L[i1];
    uint64_t c0 = MSC - G(T, L[j] - x0);
    uint64_t c1 = MSC - G(T, L[j + 1] - x0);
    if (c1 == c0) {
        v.push_back({(uint32_t)FCUT, (uint32_t)(MSC - FCUT), (uint8_t)MB});
        v.push_back({j, 1u, (uint8_t)md.pos_bits});
    } else {
        v.push_back({(uint32_t)c0, (uint32_t)(c1 - c0), (uint8_t)MB});
    }
}
inline void put_count(std::vector<Sym>& v, const Model& md, uint32_t j, uint32_t b, uint32_t c) {
    const uint16_t* row = md.ccum.data() + md.crow(j, b);
    uint32_t s = c - 1 < 15 ? c - 1 : 15;
    v.push_back({cum_at(row, s), cum_at(row, s + 1) - cum_at(row, s), (uint8_t)CB});
    if (s == 15) {
        uint32_t u = c - 15;                        // >= 1
        int nbv = 31 - __builtin_clz(u);
        v.push_back({(uint32_t)nbv, 1u, 5});
        uint32_t low = u - (1u << nbv);
        if (nbv > 16) {
            v.push_back({low >> 16, 1u, (uint8_t)(nbv - 16)});
            v.push_back({low & 0xFFFF, 1u, 16});
        } else if (nbv > 0) {
            v.push_back({low, 1u, (uint8_t)nbv});
        }
    }
}

}  // namespace i1
}  // namespace cs
