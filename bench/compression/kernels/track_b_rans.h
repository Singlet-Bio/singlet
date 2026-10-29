// Track B: static-table rANS primitives (32-bit state, 16-bit renormalisation words) + table (de)serialisation.
//
// Encoder side is scalar/simple. Decoder side is table driven: one 32-bit entry per slot,
//   bits 0..7 symbol, bits 8..19 freq, bits 20..31 (slot - cum)
// so that   x' = freq * (x >> PB) + bias.   Probability bits PB <= 12.
#pragma once
#include "track_b_util.h"
#include <cmath>

namespace cs {
namespace tb {

constexpr uint32_t kRansL = 1u << 16;

// ---- frequency normalisation -----------------------------------------------------------------
// counts -> freqs summing to 1<<pb, every present symbol >= 1. A lone symbol gets 2^pb - 1 and a dummy
// partner symbol gets 1 so that every table has at least two symbols (freq field stays < 4096).
inline void normalize_freqs(const std::vector<uint64_t>& c, int pb, std::vector<uint16_t>& f) {
    const uint32_t M = 1u << pb;
    const size_t A = c.size();
    f.assign(A, 0);
    uint64_t N = 0;
    size_t npres = 0, last = 0;
    for (size_t i = 0; i < A; ++i) if (c[i]) { N += c[i]; ++npres; last = i; }
    if (!N) return;
    if (npres == 1) {
        f[last] = static_cast<uint16_t>(M - 1);
        f[last == 0 ? 1 : 0] = 1;
        return;
    }
    int64_t sum = 0;
    for (size_t i = 0; i < A; ++i)
        if (c[i]) {
            uint64_t v = (c[i] * M) / N;
            f[i] = static_cast<uint16_t>(std::max<uint64_t>(1, v));
            sum += f[i];
        }
    while (sum < static_cast<int64_t>(M)) {  // give the most valuable increments to the largest counts
        size_t best = 0;
        double bg = -1;
        for (size_t i = 0; i < A; ++i)
            if (c[i]) {
                double gain = static_cast<double>(c[i]) * std::log2((f[i] + 1.0) / f[i]);
                if (gain > bg) { bg = gain; best = i; }
            }
        f[best]++;
        ++sum;
    }
    while (sum > static_cast<int64_t>(M)) {
        size_t best = A;
        double bc = 1e300;
        for (size_t i = 0; i < A; ++i)
            if (c[i] && f[i] > 1) {
                double cost = static_cast<double>(c[i]) * std::log2(static_cast<double>(f[i]) / (f[i] - 1.0));
                if (cost < bc) { bc = cost; best = i; }
            }
        f[best]--;
        --sum;
    }
}

// ---- encoder tables ------------------------------------------------------------------------------
struct EncTables {
    int pb = 12;
    size_t A = 0, nctx = 0;
    std::vector<uint16_t> freq, cum;  // [ctx * A + sym]
    std::vector<uint8_t> used;        // ctx used
};
inline EncTables build_enc_tables(const std::vector<uint64_t>& hist, size_t nctx, size_t A, int pb) {
    EncTables T;
    T.pb = pb; T.A = A; T.nctx = nctx;
    T.freq.assign(nctx * A, 0);
    T.cum.assign(nctx * A, 0);
    T.used.assign(nctx, 0);
    std::vector<uint64_t> c(A);
    std::vector<uint16_t> f;
    for (size_t x = 0; x < nctx; ++x) {
        uint64_t tot = 0;
        for (size_t s = 0; s < A; ++s) { c[s] = hist[x * A + s]; tot += c[s]; }
        if (!tot) continue;
        T.used[x] = 1;
        normalize_freqs(c, pb, f);
        uint32_t cu = 0;
        for (size_t s = 0; s < A; ++s) { T.freq[x * A + s] = f[s]; T.cum[x * A + s] = static_cast<uint16_t>(cu); cu += f[s]; }
    }
    return T;
}
// serialise: [LEB nused] then per used ctx: LEB(ctx - prev), LEB(nsym), per sym: LEB(sym - prev), LEB(freq)
inline void serialize_tables(std::vector<uint8_t>& o, const EncTables& T) {
    std::vector<uint8_t> raw;
    size_t nused = 0;
    for (auto u : T.used) nused += u;
    leb_put(raw, nused);
    size_t prevc = 0;
    for (size_t x = 0; x < T.nctx; ++x) {
        if (!T.used[x]) continue;
        leb_put(raw, x - prevc);
        prevc = x;
        size_t ns = 0;
        for (size_t s = 0; s < T.A; ++s) ns += T.freq[x * T.A + s] != 0;
        leb_put(raw, ns);
        size_t ps = 0;
        for (size_t s = 0; s < T.A; ++s)
            if (T.freq[x * T.A + s]) {
                leb_put(raw, s - ps);
                ps = s;
                leb_put(raw, T.freq[x * T.A + s]);
            }
    }
    detail::put_stream(o, raw.data(), raw.size(), 19);
}

// ---- decoder tables --------------------------------------------------------------------------------
struct DecTables {
    int pb = 12;
    size_t nctx_alloc = 0;
    std::unique_ptr<uint32_t[]> tab;  // [ctx << pb | slot], uninitialised for unused ctx
};
inline DecTables parse_tables(const uint8_t*& p, int pb, size_t nctx) {
    DecTables D;
    D.pb = pb;
    D.nctx_alloc = nctx;
    D.tab.reset(new uint32_t[(nctx << pb) + 16]);
    auto raw = detail::get_stream(p);
    const uint8_t* q = raw.data();
    size_t nused = leb_get(q), ctx = 0;
    for (size_t u = 0; u < nused; ++u) {
        ctx += leb_get(q);
        size_t ns = leb_get(q), sym = 0;
        uint32_t cu = 0;
        uint32_t* t = D.tab.get() + (ctx << pb);
        for (size_t i = 0; i < ns; ++i) {
            sym += leb_get(q);
            uint32_t f = static_cast<uint32_t>(leb_get(q));
            for (uint32_t s = 0; s < f; ++s) t[cu + s] = static_cast<uint32_t>(sym) | (f << 8) | (s << 20);
            cu += f;
        }
    }
    return D;
}

// ---- rANS stream encoder (reverse), single lane -------------------------------------------------------
struct RansEnc {
    uint32_t x = kRansL;
    std::vector<uint16_t> w;  // emitted in generation order (= reverse of decode order)
    inline void put(uint32_t f, uint32_t c, int pb) {
        uint32_t xmax = f << (32 - pb);
        if (x >= xmax) { w.push_back(static_cast<uint16_t>(x)); x >>= 16; }
        x = ((x / f) << pb) + (x % f) + c;
    }
    // finish: words in decode order: state hi, state lo, then renorm words reversed
    void finish(std::vector<uint16_t>& out) {
        w.push_back(static_cast<uint16_t>(x));
        w.push_back(static_cast<uint16_t>(x >> 16));
        out.assign(w.rbegin(), w.rend());
    }
};

}  // namespace tb
}  // namespace cs
