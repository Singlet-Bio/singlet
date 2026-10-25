// SPDX-License-Identifier: MIT
// pz2 codec 1 core: class-order, lock-step multi-lane static rANS.  Format: docs/format-store.md section 5.1.
// Ported from bench/compression/kernels/w3_fast_*.h with: m up to 2^21 - 1, any u32 value (uniform fields wider than 16 bits are
// coded as a low and a high part), integer-only frequency normalisation (bit-identical encoder output on every platform),
// validated tables, and an end-of-stream integrity check.
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {
namespace r1 {

inline constexpr uint32_t kL = 1u << 16;     // rANS lower bound; states live in [2^16, 2^32), renormalisation moves 16 bits
inline constexpr int kNumCls = 124;          // classes 0..123 cover every u32 (g < 8 direct, then 4 per octave)
inline constexpr int kClsPad = 128;          // class tables are padded so a corrupt stream cannot index outside them
inline constexpr int kAlpha = 2 * kNumCls;   // gap alphabet: class * 2 + (value > 1)
inline constexpr int kResPos = 21;           // residual position buckets (posb(rank)/2 <= 20 for rank < 2^21)
inline constexpr int kResCtx = kResPos * 4;  // x depth class dk (0..3)
inline constexpr int kMaxLanes = 64;
inline constexpr uint32_t kMaxChunkN = 1u << 28;   // hard limit on non-zeros per chunk (REM packs dk into bits 28..31)

// ---- log-linear classes ------------------------------------------------------------------------------------------
// Class c = (e << 2) | mm.  c < 8: the value is c itself (no extra bits).  c >= 8 (e >= 2): value = ((4 + mm) << (e - 1)) + extra,
// where `extra` has e - 1 bits.  Classes 0..123 cover all u32 values.
inline uint32_t g_cls(uint32_t g, uint32_t& ex, uint32_t& nb) {
    if (g < 8) { ex = 0; nb = 0; return g; }
    int b = ilog2f(g);
    nb = uint32_t(b - 2);
    ex = g & ((1u << nb) - 1);
    return 8 + uint32_t(b - 3) * 4 + ((g >> nb) & 3);
}
struct ClsTab {
    uint32_t base[kClsPad];
    uint8_t nb[kClsPad];
    ClsTab() {
        for (int c = 0; c < kClsPad; ++c) {
            uint32_t e = uint32_t(c) >> 2, mm = uint32_t(c) & 3;
            uint32_t n = e > 1 ? e - 1 : 0;
            base[c] = ((mm | (e ? 4u : 0u)) << n);
            nb[c] = uint8_t(n);
        }
    }
};
inline const ClsTab& cls_tab() { static const ClsTab t; return t; }

// position bucket: two per octave of (p + 2); 0 for p = 0.  Vector form: (bits(float(p + 2)) >> 22) - 256.
inline uint32_t posb_raw(uint32_t p) {
    uint32_t q = p + 2;
    uint32_t lg = uint32_t(ilog2f(q));
    return 2 * lg + ((q >> (lg - 1)) & 1) - 2;
}
inline uint32_t dk_of(uint32_t kraw) { uint32_t d = (kraw * 11) >> 5; return d < 3 ? d : 3; }

struct Cfg {   // context configuration; `kind` is stored in the header (ctx_kind)
    uint8_t pb = 10, pbv = 10, sh = 0, cap = 39, ks = 0, kcap = 11, nk = 12;
    static Cfg make(int kind, int pb, int pbv) {
        Cfg c;
        c.pb = uint8_t(pb); c.pbv = uint8_t(pbv);
        switch (kind) {
            case 0: c.sh = 0; c.cap = 39; c.ks = 0; c.kcap = 11; break;
            case 1: c.sh = 0; c.cap = 39; c.ks = 1; c.kcap = 5; break;
            case 2: c.sh = 1; c.cap = 19; c.ks = 1; c.kcap = 5; break;
            case 3: c.sh = 0; c.cap = 39; c.ks = 0; c.kcap = 0; break;
            default: c.sh = 1; c.cap = 19; c.ks = 0; c.kcap = 0; break;
        }
        c.nk = uint8_t(c.kcap + 1);
        return c;
    }
    size_t npc() const { return size_t(cap) + 1; }
    size_t nctx() const { return npc() * nk; }
    uint32_t pcls(uint32_t b) const { b >>= sh; return b < cap ? b : cap; }
    uint32_t kcls(uint32_t kraw) const { kraw >>= ks; return kraw < kcap ? kraw : kcap; }
    uint32_t kbase(uint32_t k) const { return (kcls(uint32_t(ilog2f(k + 1))) * uint32_t(npc())) << pb; }
};

// ---- integer log2 (Q32) so that frequency normalisation and size estimates are identical on every platform ---------------
inline const std::vector<uint64_t>& lg_table() {
    static const std::vector<uint64_t> t = [] {
        std::vector<uint64_t> v(4098, 0);
        for (uint32_t f = 1; f < v.size(); ++f) {
            int ip = ilog2f(f);
            uint64_t y = (uint64_t(f) << 31) >> ip;   // mantissa in [2^31, 2^32), i.e. [1,2) as Q31
            uint64_t frac = 0;
            for (int i = 0; i < 32; ++i) {
                y = (y * y) >> 31;
                frac <<= 1;
                if (y >= (uint64_t(1) << 32)) { frac |= 1; y >>= 1; }
            }
            v[f] = (uint64_t(ip) << 32) | frac;
        }
        return v;
    }();
    return t;
}

// ---- frequency tables ---------------------------------------------------------------------------------------------
inline void normalize_freqs(const uint64_t* c, size_t A, int pb, std::vector<uint16_t>& f) {
    const uint32_t M = 1u << pb;
    const auto& lg = lg_table();
    f.assign(A, 0);
    uint64_t N = 0;
    size_t npres = 0, last = 0;
    for (size_t i = 0; i < A; ++i) if (c[i]) { N += c[i]; ++npres; last = i; }
    if (!N) return;
    if (npres == 1) { f[last] = uint16_t(M - 1); f[last == 0 ? 1 : 0] = 1; return; }
    int64_t sum = 0;
    for (size_t i = 0; i < A; ++i)
        if (c[i]) {
            uint64_t v = (c[i] * M) / N;
            f[i] = uint16_t(std::max<uint64_t>(1, v));
            sum += f[i];
        }
    while (sum < int64_t(M)) {   // give one unit to the symbol with the largest coding gain
        size_t best = 0;
        double bg = -1;
        for (size_t i = 0; i < A; ++i)
            if (c[i]) {
                double gain = double(c[i]) * double(lg[f[i] + 1] - lg[f[i]]);
                if (gain > bg) { bg = gain; best = i; }
            }
        f[best]++;
        ++sum;
    }
    while (sum > int64_t(M)) {   // take one unit from the symbol that loses the least
        size_t best = A;
        double bc = 1e300;
        for (size_t i = 0; i < A; ++i)
            if (c[i] && f[i] > 1) {
                double cost = double(c[i]) * double(lg[f[i]] - lg[f[i] - 1]);
                if (cost < bc) { bc = cost; best = i; }
            }
        f[best]--;
        --sum;
    }
}

struct EncTables {
    int pb = 10;
    size_t A = 0, nctx = 0;
    std::vector<uint16_t> freq, cum;   // [ctx * A + sym]
    std::vector<uint8_t> used;
};
inline EncTables build_enc_tables(const std::vector<uint64_t>& hist, size_t nctx, size_t A, int pb) {
    EncTables T;
    T.pb = pb; T.A = A; T.nctx = nctx;
    T.freq.assign(nctx * A, 0);
    T.cum.assign(nctx * A, 0);
    T.used.assign(nctx, 0);
    std::vector<uint16_t> f;
    for (size_t x = 0; x < nctx; ++x) {
        const uint64_t* c = &hist[x * A];
        uint64_t tot = 0;
        for (size_t s = 0; s < A; ++s) tot += c[s];
        if (!tot) continue;
        T.used[x] = 1;
        normalize_freqs(c, A, pb, f);
        uint32_t cu = 0;
        for (size_t s = 0; s < A; ++s) { T.freq[x * A + s] = f[s]; T.cum[x * A + s] = uint16_t(cu); cu += f[s]; }
    }
    return T;
}
// raw table bytes: LEB nused, per used context: LEB(ctx - prev) LEB(nsym), per symbol: LEB(sym - prev) LEB(freq); then framed
inline void serialize_tables(std::vector<uint8_t>& o, const EncTables& T, int zlevel) {
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
            if (T.freq[x * T.A + s]) { leb_put(raw, s - ps); ps = s; leb_put(raw, T.freq[x * T.A + s]); }
    }
    put_stream(o, raw.data(), raw.size(), zlevel);
}

struct DecTables {
    int pb = 10;
    size_t nctx = 0;
    std::vector<uint32_t> tab;   // [(ctx << pb) | slot], entry = sym << 24 | freq << 12 | bias; unused contexts are zero
    const uint32_t* data() const { return tab.data(); }
};
/// Parse and validate one table stream.  `A` = alphabet size (symbols must be < A).
inline DecTables parse_tables(const uint8_t*& p, const uint8_t* end, int pb, size_t nctx, size_t A) {
    DecTables D;
    D.pb = pb; D.nctx = nctx;
    D.tab.assign((nctx << pb) + 64, 0);
    auto raw = get_stream(p, end, uint64_t(1) << 28);
    const uint8_t* q = raw.data();
    const uint8_t* qe = q + raw.size();
    const uint64_t nused = leb_get(q, qe);
    if (nused > nctx) throw Error(Errc::Corrupt, "table: too many contexts");
    uint64_t ctx = 0;
    for (uint64_t u = 0; u < nused; ++u) {
        uint64_t dc = leb_get(q, qe);
        if (u > 0 && dc == 0) throw Error(Errc::Corrupt, "table: context order");
        ctx += dc;
        if (ctx >= nctx) throw Error(Errc::Corrupt, "table: context out of range");
        uint64_t ns = leb_get(q, qe);
        if (ns == 0 || ns > A) throw Error(Errc::Corrupt, "table: symbol count");
        uint64_t sym = 0;
        uint32_t cu = 0;
        uint32_t* t = D.tab.data() + (ctx << pb);
        for (uint64_t i = 0; i < ns; ++i) {
            uint64_t ds = leb_get(q, qe);
            if (i > 0 && ds == 0) throw Error(Errc::Corrupt, "table: symbol order");
            sym += ds;
            uint64_t f = leb_get(q, qe);
            if (sym >= A || f == 0 || f > (1u << pb) - 1 || cu + f > (1u << pb)) throw Error(Errc::Corrupt, "table: bad symbol/frequency");
            for (uint32_t s = 0; s < f; ++s) t[cu + s] = (uint32_t(sym) << 24) | (uint32_t(f) << 12) | s;
            cu += uint32_t(f);
        }
        if (cu != (1u << pb)) throw Error(Errc::Corrupt, "table: frequencies do not sum to 2^pb");
    }
    return D;
}

// ---- lock-step encoder (L lanes, one shared reversed 16-bit word stream) ---------------------------------------------
struct LockEnc {
    int L;
    uint32_t X[kMaxLanes];
    std::vector<uint16_t> w;
    explicit LockEnc(int L_) : L(L_) { for (auto& x : X) x = kL; }
    inline void put(int l, uint32_t f, uint32_t c, int pb) {
        uint32_t x = X[l];
        uint32_t xmax = f << (32 - pb);
        if (x >= xmax) { w.push_back(uint16_t(x)); x >>= 16; }
        X[l] = ((x / f) << pb) + (x % f) + c;
    }
    inline void put_uniform(int l, uint32_t v, int nb) {   // nb <= 16
        if (!nb) return;
        uint32_t x = X[l];
        uint32_t xmax = 1u << (32 - nb);
        if (x >= xmax) { w.push_back(uint16_t(x)); x >>= 16; }
        X[l] = (x << nb) | v;
    }
    /// A field of nb <= 32 bits.  Decode order is low 16 bits first, then the high nb - 16 bits, so the encoder (which runs
    /// backwards) writes the high part first.
    inline void put_bits(int l, uint32_t v, uint32_t nb) {
        if (nb <= 16) { put_uniform(l, v, int(nb)); return; }
        put_uniform(l, v >> 16, int(nb - 16));
        put_uniform(l, v & 0xFFFF, 16);
    }
    /// Appends L u32 states and then the words in decode order.
    size_t finish(std::vector<uint8_t>& o) {
        for (int l = 0; l < L; ++l) put_u32(o, X[l]);
        for (size_t i = w.size(); i-- > 0;) { o.push_back(uint8_t(w[i])); o.push_back(uint8_t(w[i] >> 8)); }
        return w.size();
    }
};

// ---- row order ("class order") ---------------------------------------------------------------------------------------
// key(count): 0 for empty rows (placed last); else 1 + lg * cpo + floor((count / 2^lg - 1) * cpo) with lg = floor(log2 count);
// cpo == 0 means exact rank (key = count).  Stored order = key descending, canonical id ascending inside a key.
inline uint32_t row_key(uint32_t cnt, int cpo) {
    if (!cnt) return 0;
    if (cpo == 0) return cnt;
    uint32_t lg = uint32_t(ilog2f(cnt));
    uint32_t frac = uint32_t((uint64_t(cnt - (1u << lg)) * uint64_t(cpo)) >> lg);
    return 1 + lg * uint32_t(cpo) + frac;
}
/// perm[stored id] = canonical id, from per-row keys.
inline std::vector<uint32_t> perm_from_keys(const std::vector<uint32_t>& key) {
    const uint32_t m = uint32_t(key.size());
    std::vector<uint32_t> perm(m);
    uint32_t mx = 0;
    for (uint32_t k : key) mx = std::max(mx, k);
    if (uint64_t(mx) <= 4ull * m + 4096) {   // counting sort
        std::vector<uint32_t> start(size_t(mx) + 2, 0);
        for (uint32_t i = 0; i < m; ++i) start[key[i]]++;
        uint32_t pos = 0;
        for (int64_t b = mx; b >= 0; --b) { uint32_t c = start[size_t(b)]; start[size_t(b)] = pos; pos += c; }
        for (uint32_t i = 0; i < m; ++i) perm[start[key[i]]++] = i;
    } else {
        for (uint32_t i = 0; i < m; ++i) perm[i] = i;
        std::stable_sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) { return key[a] > key[b]; });
    }
    return perm;
}

}  // namespace r1
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
