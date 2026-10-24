// SPDX-License-Identifier: MIT
// w3_fast core: format "W3F1" - lock-step multi-lane static rANS over class-order row ids.
//
// Differences from Track B's b_rans16 (kernels/track_b_lane.h), all aimed at decode speed with no sort:
//   * rows are renumbered in "class order" (frequency class desc, original id asc inside a class; cpo = classes per
//     octave, 0 = exact frequency rank = Track B's order). The decoder emits ascending class-order ids; the consumer
//     gets a CSC in that permuted row space (+ the permutation); canonical (original) order is an optional slower step.
//   * lanes split the chunk's element sequence into L equal slices (not on column boundaries); a lane that starts
//     inside a column gets its start position (pp0) in the chunk header. Every lane is busy for every step.
//   * the residual (value-2) is coded inline in the same lock-step loop (phases C1/C2), not in a second stream/pass.
//   * column counts are coded per chunk (self-contained chunks: range decode and parallel decode need no global state).
//   * table index layout [depth class][position class][slot], entry = sym<<24 | freq<<12 | bias, so the decoder needs no
//     multiply for the context address.
// Everything here is integer arithmetic; the SIMD decoders (w3_fast_avx512.h, w3_fast_avx2.h) decode the same bytes.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace cs {
namespace wf {

constexpr uint32_t kRansL = 1u << 16;
constexpr int kNumCls = 124;       // gap / value classes (g < 8 direct, then 4 per octave); only <= 72 used (g < 2^19)
constexpr int kMaxLanes = 64;
constexpr uint32_t kMaxRows = 1u << 19;

typedef uint16_t __attribute__((may_alias, aligned(1))) u16u;
typedef uint32_t __attribute__((may_alias, aligned(1))) u32u;

inline int ilog2u(uint32_t x) { return 31 - __builtin_clz(x | 1); }

inline void leb_put(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 128) { o.push_back(static_cast<uint8_t>(v | 128)); v >>= 7; }
    o.push_back(static_cast<uint8_t>(v));
}
inline uint64_t leb_get(const uint8_t*& p) {
    uint64_t v = 0;
    int s = 0;
    while (true) {
        uint8_t b = *p++;
        v |= static_cast<uint64_t>(b & 127) << s;
        if (!(b & 128)) break;
        s += 7;
    }
    return v;
}

// ---- log-linear classes -------------------------------------------------------------------------------------
// class c = (e << 2) | mm : e = 0 -> value mm (< 4); e >= 1 -> value = (4 + mm) << (e - 1) + extra, extra has e-1 bits.
inline uint32_t g_cls(uint32_t g, uint32_t& ex, uint32_t& nb) {
    if (g < 8) { ex = 0; nb = 0; return g; }
    int b = ilog2u(g);
    nb = static_cast<uint32_t>(b - 2);
    ex = g & ((1u << nb) - 1);
    return 8 + static_cast<uint32_t>(b - 3) * 4 + ((g >> nb) & 3);
}
struct ClsTab {
    uint32_t base[kNumCls];
    uint8_t nb[kNumCls];
    ClsTab() {
        for (int c = 0; c < kNumCls; ++c) {
            uint32_t e = static_cast<uint32_t>(c) >> 2, mm = static_cast<uint32_t>(c) & 3;
            uint32_t n = e > 1 ? e - 1 : 0;
            base[c] = ((mm | (e ? 4u : 0u)) << n);
            nb[c] = static_cast<uint8_t>(n);
        }
    }
};
inline const ClsTab& cls_tab() {
    static ClsTab t;
    return t;
}

// position bucket: 2 per octave of (p + 2), 0 for p = 0.   (vector form: (bits(float(p+2)) >> 22) - 256)
inline uint32_t posb_raw(uint32_t p) {
    uint32_t q = p + 2;
    uint32_t lg = 31u - static_cast<uint32_t>(__builtin_clz(q));
    return 2 * lg + ((q >> (lg - 1)) & 1) - 2;
}
inline uint32_t dk_of(uint32_t kraw) { uint32_t d = (kraw * 11) >> 5; return d < 3 ? d : 3; }

struct Cfg {  // context configuration (Track B kinds 0..4)
    uint8_t pb = 10, pbv = 10, sh = 0, cap = 39, ks = 0, kcap = 11, nk = 12;
    static Cfg make(int kind, int pb = 10, int pbv = 10) {
        Cfg c;
        c.pb = static_cast<uint8_t>(pb);
        c.pbv = static_cast<uint8_t>(pbv);
        switch (kind) {
            case 0: c.sh = 0; c.cap = 39; c.ks = 0; c.kcap = 11; break;
            case 1: c.sh = 0; c.cap = 39; c.ks = 1; c.kcap = 5; break;
            case 2: c.sh = 1; c.cap = 19; c.ks = 1; c.kcap = 5; break;
            case 3: c.sh = 0; c.cap = 39; c.ks = 0; c.kcap = 0; break;
            default: c.sh = 1; c.cap = 19; c.ks = 0; c.kcap = 0; break;
        }
        c.nk = static_cast<uint8_t>(c.kcap + 1);
        return c;
    }
    size_t npc() const { return static_cast<size_t>(cap) + 1; }
    size_t nctx() const { return npc() * nk; }
    uint32_t pcls(uint32_t b) const { b >>= sh; return b < cap ? b : cap; }
    uint32_t kcls(uint32_t kraw) const { kraw >>= ks; return kraw < kcap ? kraw : kcap; }
    // table offset (in entries) of the depth class of a column with k elements
    uint32_t kbase(uint32_t k) const { return (kcls(static_cast<uint32_t>(ilog2u(k + 1))) * static_cast<uint32_t>(npc())) << pb; }
};
constexpr int kResPos = 20;                 // own-rank buckets (posb/2)
constexpr int kResCtx = kResPos * 4;        // x depth class dk (0..3); index = (posb/2) * 4 + dk

// ---- frequency tables ----------------------------------------------------------------------------------------
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
    while (sum < static_cast<int64_t>(M)) {
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

struct EncTables {
    int pb = 10;
    size_t A = 0, nctx = 0;
    std::vector<uint16_t> freq, cum;  // [ctx * A + sym]
    std::vector<uint8_t> used;
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
// serialised (then zstd-19 framed): [LEB nused] per used ctx: LEB(ctx - prev) LEB(nsym) per sym: LEB(sym - prev) LEB(freq)
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

#ifdef _WIN32
extern "C" void* _aligned_malloc(size_t, size_t);
extern "C" void _aligned_free(void*);
struct AlignedFree { void operator()(void* p) const { if (p) _aligned_free(p); } };
inline void* aligned_alloc64(size_t bytes) { return _aligned_malloc(bytes, 64); }
#else
struct AlignedFree { void operator()(void* p) const { if (p) std::free(p); } };
inline void* aligned_alloc64(size_t bytes) { return std::aligned_alloc(64, bytes); }
#endif
template <class T> using AlignedPtr = std::unique_ptr<T[], AlignedFree>;
template <class T> inline AlignedPtr<T> aligned_array(size_t n) {
    size_t bytes = (n * sizeof(T) + 63) & ~size_t(63);
    void* p = aligned_alloc64(bytes ? bytes : 64);
    if (!p) throw std::bad_alloc();
    return AlignedPtr<T>(static_cast<T*>(p));
}

struct DecTables {
    int pb = 10;
    size_t nctx = 0;
    AlignedPtr<uint32_t> tab;  // [ctx << pb | slot] ; entry = sym << 24 | freq << 12 | bias ; unused ctx uninitialised
    size_t bytes() const { return (nctx << pb) * 4; }
};
inline DecTables parse_tables(const uint8_t*& p, int pb, size_t nctx) {
    DecTables D;
    D.pb = pb;
    D.nctx = nctx;
    D.tab = aligned_array<uint32_t>((nctx << pb) + 32);
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
            for (uint32_t s = 0; s < f; ++s) t[cu + s] = (static_cast<uint32_t>(sym) << 24) | (f << 12) | s;
            cu += f;
        }
    }
    return D;
}

// ---- lock-step encoder (L lanes, one shared reversed word stream) -------------------------------------------
struct LockEnc {
    int L;
    uint32_t X[kMaxLanes];
    std::vector<uint16_t> w;
    explicit LockEnc(int L_) : L(L_) { for (auto& x : X) x = kRansL; }
    inline void put(int l, uint32_t f, uint32_t c, int pb) {
        uint32_t x = X[l];
        uint32_t xmax = f << (32 - pb);
        if (x >= xmax) { w.push_back(static_cast<uint16_t>(x)); x >>= 16; }
        X[l] = ((x / f) << pb) + (x % f) + c;
    }
    inline void put_uniform(int l, uint32_t v, int nb) {
        if (!nb) return;
        uint32_t x = X[l];
        uint32_t xmax = 1u << (32 - nb);
        if (x >= xmax) { w.push_back(static_cast<uint16_t>(x)); x >>= 16; }
        X[l] = (x << nb) | v;
    }
    size_t finish(std::vector<uint8_t>& o) {  // appends L u32 states then the words in decode order
        for (int l = 0; l < L; ++l) detail::put_u32(o, X[l]);
        for (size_t i = w.size(); i-- > 0;) { o.push_back(static_cast<uint8_t>(w[i])); o.push_back(static_cast<uint8_t>(w[i] >> 8)); }
        return w.size();
    }
};

// ---- row order ("class order") -----------------------------------------------------------------------------------
// key(count): 0 for empty rows (placed last), else 1 + lg*cpo + floor((count / 2^lg - 1) * cpo); cpo = 0 -> key = count.
inline uint32_t row_key(uint32_t cnt, int cpo) {
    if (!cnt) return 0;
    if (cpo == 0) return cnt;
    uint32_t lg = static_cast<uint32_t>(ilog2u(cnt));
    uint32_t frac = static_cast<uint32_t>((static_cast<uint64_t>(cnt - (1u << lg)) * static_cast<uint64_t>(cpo)) >> lg);
    return 1 + lg * static_cast<uint32_t>(cpo) + frac;
}
struct RowOrder {
    std::vector<uint32_t> perm;  // class-order id -> original row id
    std::vector<uint32_t> inv;   // original row id -> class-order id
};
inline RowOrder order_from_counts(const std::vector<uint32_t>& rnnz, int cpo) {
    const uint32_t m = static_cast<uint32_t>(rnnz.size());
    std::vector<uint32_t> key(m);
    uint32_t mx = 0;
    for (uint32_t i = 0; i < m; ++i) { key[i] = row_key(rnnz[i], cpo); mx = std::max(mx, key[i]); }
    std::vector<uint32_t> start(static_cast<size_t>(mx) + 2, 0);
    for (uint32_t i = 0; i < m; ++i) start[key[i]]++;
    uint32_t pos = 0;
    for (int64_t b = mx; b >= 0; --b) { uint32_t c = start[b]; start[b] = pos; pos += c; }
    RowOrder R;
    R.perm.assign(m, 0);
    R.inv.assign(m, 0);
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t r = start[key[i]]++;
        R.perm[r] = i;
        R.inv[i] = r;
    }
    return R;
}

// Matrix in class-order row space: ids ascending within each column.
struct ClsMat {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    std::vector<uint32_t> ptr, rk, val;
};
inline ClsMat to_class_space(const CanonicalMatrix& M, const RowOrder& O) {
    ClsMat R;
    R.m = M.m; R.n = M.n; R.nnz = M.nnz; R.ptr = M.indptr;
    R.rk.resize(M.nnz);
    R.val.resize(M.nnz);
    std::vector<uint64_t> tmp;
    for (uint32_t j = 0; j < M.n; ++j) {
        tmp.clear();
        for (uint32_t k = M.indptr[j]; k < M.indptr[j + 1]; ++k)
            tmp.push_back((static_cast<uint64_t>(O.inv[M.indices[k]]) << 32) | M.data[k]);
        std::sort(tmp.begin(), tmp.end());
        for (size_t t = 0; t < tmp.size(); ++t) {
            R.rk[M.indptr[j] + t] = static_cast<uint32_t>(tmp[t] >> 32);
            R.val[M.indptr[j] + t] = static_cast<uint32_t>(tmp[t]);
        }
    }
    return R;
}

// ---- parameters / sizes -----------------------------------------------------------------------------------------
struct Params {
    int lanes = 32;                 // 16, 32 or 64
    int pb = 10, pbv = 10;          // probability bits of the gap/flag tables and of the residual tables (<= 12)
    int ctx = -1;                   // Track B context kind 0..4, -1 = pick the smallest by estimate
    int chunk_cols = 0;             // 0 = auto (about target_nnz per chunk, power of two in [128, 4096])
    uint32_t target_nnz = 96 * 1024;
    int cpo = 4;                    // classes per octave of row frequency; 0 = exact rank
};
struct Sizes {
    size_t total = 0, payload = 0, side = 0, names = 0, tables = 0, chunks = 0, header = 0;
    int ctx = 0, chunk_cols = 0;
};
inline Sizes& last_sizes() {
    static thread_local Sizes s;
    return s;
}

inline int auto_chunk_cols(uint32_t n, uint64_t nnz, uint32_t target) {
    if (n == 0 || nnz == 0) return 1024;
    double mean_k = static_cast<double>(nnz) / n;
    double want = target / std::max(mean_k, 1.0);
    int c = 128;
    while (c < 4096 && c * 1.414 < want) c *= 2;
    return c;
}

}  // namespace wf
}  // namespace cs
