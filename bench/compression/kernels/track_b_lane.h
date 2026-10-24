// Track B: 16-lane lock-step static rANS for rank-space gap coding with (position bucket x column depth) contexts.
//
// Payload layout:  cfg | counts stream | gap tables | residual tables | chunk sizes | chunk blobs
// Chunk blob:      LEB(main words) LEB(res words) | 32 x u32 main states | main u16 words | 32 x u32 res states | res u16 words
// Chunk columns are split into 16 lanes of contiguous columns with ~equal nnz (derived from the column counts).
// Decode order: for step s, for each active lane (ascending): phase A decode gap-class symbol (+ renorm word),
// then phase B take the extra gap bits as a uniform rANS symbol (+ renorm word). All words come from ONE shared
// stream, consumed in that order, so a vector decoder can use expand-loads.  m <= 2^19, gaps < 2^19.
// Values: the symbol also carries the flag "value>1"; flagged elements get (value-2) from a second flat 16-lane stream.
#pragma once
#include "track_b_rans.h"
#ifdef __AVX512F__
#include <immintrin.h>
#if defined(__AVX512BW__) && defined(__AVX512VBMI2__) && defined(__AVX512CD__) && defined(__AVX512DQ__) && defined(__BMI2__)
#define TB_HAVE_AVX512 1
#endif
#endif

namespace cs {
namespace tb {

constexpr int kLanes = 32;
constexpr int kPosB = 40;
constexpr uint32_t kMaxM = 1u << 19;

typedef uint16_t __attribute__((may_alias, aligned(1))) u16u;
typedef uint32_t __attribute__((may_alias, aligned(1))) u32u;

inline uint32_t posb_raw(uint32_t p) {  // 2 buckets per octave of (p+2); p = previous rank + 1
    uint32_t q = p + 2;
    uint32_t lg = 31u - static_cast<uint32_t>(__builtin_clz(q));
    return 2 * lg + ((q >> (lg - 1)) & 1) - 2;
}
inline uint32_t dk_of(uint32_t kraw) { uint32_t d = (kraw * 11) >> 5; return d < 3 ? d : 3; }

struct Cfg {
    uint8_t pb = 12, pbv = 10, sh = 0, cap = 39, ks = 0, kcap = 11, nk = 12;
    static Cfg make(int kind, int pb = 12) {
        Cfg c;
        c.pb = static_cast<uint8_t>(pb);
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
    size_t nctx() const { return static_cast<size_t>(cap + 1) * nk; }
    uint32_t pcls(uint32_t b) const { b >>= sh; return b < cap ? b : cap; }
    uint32_t kcls(uint32_t kraw) const { kraw >>= ks; return kraw < kcap ? kraw : kcap; }
    void put(std::vector<uint8_t>& o) const {
        const uint8_t v[7] = {pb, pbv, sh, cap, ks, kcap, nk};
        o.insert(o.end(), v, v + 7);
    }
    static Cfg get(const uint8_t*& p) {
        Cfg c;
        c.pb = p[0]; c.pbv = p[1]; c.sh = p[2]; c.cap = p[3]; c.ks = p[4]; c.kcap = p[5]; c.nk = p[6];
        p += 7;
        return c;
    }
};

constexpr int kResCtx = 20 * 4;  // (own rank bucket / 2) x depth class

// Lane partition of columns [0,nc) with counts kk into L lanes of ~equal nnz. bnd has L+1 entries.
inline void lane_partition(const uint32_t* kk, uint32_t nc, int L, uint32_t* bnd) {
    uint64_t total = 0;
    for (uint32_t j = 0; j < nc; ++j) total += kk[j];
    bnd[0] = 0;
    uint64_t pre = 0;
    uint32_t j = 0;
    for (int i = 1; i < L; ++i) {
        uint64_t target = (total * static_cast<uint64_t>(i) + static_cast<uint64_t>(L) - 1) / static_cast<uint64_t>(L);
        while (j < nc && pre < target) { pre += kk[j]; ++j; }
        bnd[i] = j;
    }
    bnd[L] = nc;
}

// ------------------------------------------------------------------------------------------------
// Lock-step encoder: shared word stream, 16 lane states.
// ------------------------------------------------------------------------------------------------
struct LockEnc {
    uint32_t X[kLanes];
    std::vector<uint16_t> w;
    LockEnc() { for (auto& x : X) x = kRansL; }
    inline void put(int l, uint32_t f, uint32_t c, int pb) {
        uint32_t x = X[l];
        uint32_t xmax = f << (32 - pb);
        if (x >= xmax) { w.push_back(static_cast<uint16_t>(x)); x >>= 16; }
        X[l] = ((x / f) << pb) + (x % f) + c;
    }
    inline void put_uniform(int l, uint32_t v, int nb) {  // f=1, M=2^nb
        if (!nb) return;
        uint32_t x = X[l];
        uint32_t xmax = 1u << (32 - nb);
        if (x >= xmax) { w.push_back(static_cast<uint16_t>(x)); x >>= 16; }
        X[l] = (x << nb) | v;
    }
    // append [16 x u32 states][reversed words] ; returns word count
    size_t finish(std::vector<uint8_t>& o) {
        for (int l = 0; l < kLanes; ++l) detail::put_u32(o, X[l]);
        for (size_t i = w.size(); i-- > 0;) { o.push_back(static_cast<uint8_t>(w[i])); o.push_back(static_cast<uint8_t>(w[i] >> 8)); }
        return w.size();
    }
};

struct ElemSyms {  // per element symbol description
    std::vector<uint8_t> sym;  // symbol (alphabet index)
    std::vector<uint16_t> ctx;
    std::vector<uint32_t> ex;
    std::vector<uint8_t> nb;
};

// Encode a set of lane element ranges in lock-step order. lane l covers elements idx(l, 0..left[l]).
template <class IdxFn>
inline size_t lockstep_encode(std::vector<uint8_t>& o, const size_t* left, IdxFn idx, const ElemSyms& S, const EncTables& T, int pb) {
    LockEnc E;
    size_t steps = 0;
    for (int l = 0; l < kLanes; ++l) steps = std::max(steps, left[l]);
    for (size_t s = steps; s-- > 0;) {
        for (int l = kLanes - 1; l >= 0; --l)
            if (s < left[l]) { size_t i = idx(l, s); E.put_uniform(l, S.ex[i], S.nb[i]); }
        for (int l = kLanes - 1; l >= 0; --l)
            if (s < left[l]) {
                size_t i = idx(l, s);
                size_t k = static_cast<size_t>(S.ctx[i]) * T.A + S.sym[i];
                E.put(l, T.freq[k], T.cum[k], pb);
            }
    }
    return E.finish(o);
}

// flat stream: element j in lane j&31
inline void encode_flat(std::vector<uint8_t>& o, const std::vector<uint32_t>& cls, const std::vector<uint32_t>& ctx,
                        const std::vector<uint32_t>& ex, const std::vector<uint32_t>& nb, size_t nctx, int pb) {
    const size_t N = cls.size();
    std::vector<uint64_t> hist(nctx * kNumCls, 0);
    for (size_t j = 0; j < N; ++j) hist[ctx[j] * kNumCls + cls[j]]++;
    EncTables T = build_enc_tables(hist, nctx, kNumCls, pb);
    serialize_tables(o, T);
    ElemSyms S;
    S.sym.resize(N); S.ctx.resize(N); S.ex.resize(N); S.nb.resize(N);
    for (size_t j = 0; j < N; ++j) {
        S.sym[j] = static_cast<uint8_t>(cls[j]); S.ctx[j] = static_cast<uint16_t>(ctx[j]);
        S.ex[j] = ex[j]; S.nb[j] = static_cast<uint8_t>(nb[j]);
    }
    size_t left[kLanes];
    for (int l = 0; l < kLanes; ++l) left[l] = N > static_cast<size_t>(l) ? (N - l + kLanes - 1) / kLanes : 0;
    std::vector<uint8_t> body;
    size_t nw = lockstep_encode(body, left, [](int l, size_t s) { return s * kLanes + l; }, S, T, pb);
    leb_put(o, nw);
    o.insert(o.end(), body.begin(), body.end());
}

// Scalar flat decode. p -> 16 states then nw words. element j in lane j&31; out[j] = base[cls] + extra.
// ctxb[j] is the table offset (ctx << pb), null = 0.
inline void decode_flat_scalar(const uint8_t* p, size_t nw, const uint32_t* tab, int pb, const uint32_t* ctxb, size_t N, uint32_t* out) {
    (void)nw;
    uint32_t X[kLanes];
    for (int l = 0; l < kLanes; ++l) X[l] = *reinterpret_cast<const u32u*>(p + 4 * l);
    const uint8_t* wp = p + 4 * kLanes;
    const auto& ct = cls_tab();
    const uint32_t mask = (1u << pb) - 1;
    uint32_t cls[kLanes];
    for (size_t j0 = 0; j0 < N; j0 += kLanes) {
        int na = static_cast<int>(std::min<size_t>(kLanes, N - j0));
        for (int l = 0; l < na; ++l) {
            uint32_t x = X[l];
            uint32_t e = tab[(ctxb ? ctxb[j0 + l] : 0) + (x & mask)];
            x = ((e >> 8) & 0xFFF) * (x >> pb) + (e >> 20);
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            X[l] = x;
            cls[l] = e & 0xFF;
        }
        for (int l = 0; l < na; ++l) {
            uint32_t x = X[l], nb = ct.nb[cls[l]];
            uint32_t ex = x & ((1u << nb) - 1);
            x >>= nb;
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            X[l] = x;
            out[j0 + l] = ct.base[cls[l]] + ex;
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Encoder
// ------------------------------------------------------------------------------------------------
struct EncodedPayload {
    std::vector<uint8_t> bytes;
    size_t tables = 0, counts = 0, chunks = 0;
};

// Cheap size estimate (bytes) of the gap stream + tables for a cfg: lets the encoder pick a cfg without encoding.
inline size_t estimate_payload(const RankMat& R, const Cfg& cfg) {
    const size_t A = 2 * kNumCls, nctx = cfg.nctx();
    std::vector<uint64_t> hist(nctx * A, 0);
    double extra = 0;
    for (uint32_t j = 0; j < R.n; ++j) {
        uint32_t k = R.ptr[j + 1] - R.ptr[j];
        if (!k) continue;
        uint32_t kc = cfg.kcls(static_cast<uint32_t>(ilog2u(k + 1)));
        uint32_t pp = 0;
        for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            uint32_t ex, nb;
            uint32_t cls = g_cls(R.rk[t] - pp, ex, nb);
            hist[(cfg.pcls(posb_raw(pp)) * cfg.nk + kc) * A + cls * 2 + (R.val[t] > 1)]++;
            extra += nb;
            pp = R.rk[t] + 1;
        }
    }
    auto T = build_enc_tables(hist, nctx, A, cfg.pb);
    double bits = extra;
    for (size_t x = 0; x < nctx; ++x)
        for (size_t s = 0; s < A; ++s) {
            uint64_t c = hist[x * A + s];
            if (c) bits += static_cast<double>(c) * (cfg.pb - std::log2(static_cast<double>(T.freq[x * A + s])));
        }
    std::vector<uint8_t> tb;
    serialize_tables(tb, T);
    return static_cast<size_t>(bits / 8.0) + tb.size();
}

inline EncodedPayload encode_payload(const RankMat& R, const Cfg& cfg, int chunk_cols) {
    if (R.m > kMaxM) throw std::runtime_error("track_b lane codec: m > 2^19 unsupported");
    EncodedPayload out;
    const uint32_t n = R.n;
    const size_t nnz = R.nnz;
    const size_t A = 2 * kNumCls;
    const size_t nctx = cfg.nctx();
    ElemSyms S;
    S.sym.resize(nnz); S.ctx.resize(nnz); S.ex.resize(nnz); S.nb.resize(nnz);
    std::vector<uint64_t> hist(nctx * A, 0);
    std::vector<uint64_t> hv(static_cast<size_t>(kResCtx) * kNumCls, 0);
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t k = R.ptr[j + 1] - R.ptr[j];
        if (!k) continue;
        uint32_t kraw = static_cast<uint32_t>(ilog2u(k + 1));
        uint32_t kc = cfg.kcls(kraw);
        uint32_t dk = dk_of(kraw);
        uint32_t pp = 0;
        for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            uint32_t r = R.rk[t];
            uint32_t g = r - pp;
            uint32_t ex, nb;
            uint32_t cls = g_cls(g, ex, nb);
            uint32_t flag = R.val[t] > 1;
            uint32_t ctx = cfg.pcls(posb_raw(pp)) * cfg.nk + kc;
            S.sym[t] = static_cast<uint8_t>(cls * 2 + flag);
            S.ctx[t] = static_cast<uint16_t>(ctx);
            S.ex[t] = ex;
            S.nb[t] = static_cast<uint8_t>(nb);
            hist[ctx * A + cls * 2 + flag]++;
            if (flag) {
                uint32_t vex, vnb;
                uint32_t vc = g_cls(R.val[t] - 2, vex, vnb);
                if (vnb > 16) throw std::runtime_error("track_b lane codec: value too large");
                hv[(posb_raw(r) / 2 * 4 + dk) * kNumCls + vc]++;
            }
            pp = r + 1;
        }
    }
    auto T = build_enc_tables(hist, nctx, A, cfg.pb);
    auto TV = build_enc_tables(hv, kResCtx, kNumCls, cfg.pbv);
    std::vector<uint8_t>& o = out.bytes;
    cfg.put(o);
    size_t t0 = o.size();
    {  // counts
        std::vector<uint32_t> cls(n), ctx(n, 0), ex(n), nb(n);
        for (uint32_t j = 0; j < n; ++j) cls[j] = g_cls(R.ptr[j + 1] - R.ptr[j], ex[j], nb[j]);
        encode_flat(o, cls, ctx, ex, nb, 1, 12);
    }
    out.counts = o.size() - t0;
    t0 = o.size();
    serialize_tables(o, T);
    serialize_tables(o, TV);
    out.tables = o.size() - t0;

    const uint32_t nchunks = (n + chunk_cols - 1) / chunk_cols;
    std::vector<std::vector<uint8_t>> blobs(nchunks);
    std::vector<uint32_t> kk;
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * chunk_cols, c1 = std::min(n, c0 + chunk_cols);
        uint32_t nc = c1 - c0;
        if (R.ptr[c1] - R.ptr[c0] >= (1u << 28)) throw std::runtime_error("track_b lane codec: chunk too large");
        kk.resize(nc);
        for (uint32_t j = 0; j < nc; ++j) kk[j] = R.ptr[c0 + j + 1] - R.ptr[c0 + j];
        uint32_t bnd[kLanes + 1];
        lane_partition(kk.data(), nc, kLanes, bnd);
        size_t left[kLanes], e0[kLanes];
        for (int l = 0; l < kLanes; ++l) {
            e0[l] = R.ptr[c0 + bnd[l]];
            left[l] = R.ptr[c0 + bnd[l + 1]] - e0[l];
        }
        std::vector<uint8_t> main_body;
        size_t nwm = lockstep_encode(main_body, left, [&](int l, size_t s) { return e0[l] + s; }, S, T, cfg.pb);

        // residual: flagged elements in lock-step (step, lane) order, exactly the order the decoder discovers them
        ElemSyms RS;
        {
            struct Fe { size_t step; uint32_t cl, ex, nb, ctx; };
            std::vector<Fe> lf[kLanes];
            size_t steps = 0;
            for (int l = 0; l < kLanes; ++l) {
                steps = std::max(steps, left[l]);
                size_t pos = 0;
                for (uint32_t j = c0 + bnd[l]; j < c0 + bnd[l + 1]; ++j) {
                    uint32_t k = R.ptr[j + 1] - R.ptr[j];
                    uint32_t dk = dk_of(static_cast<uint32_t>(ilog2u(k + 1)));
                    for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t, ++pos)
                        if (R.val[t] > 1) {
                            Fe f;
                            f.step = pos;
                            f.cl = g_cls(R.val[t] - 2, f.ex, f.nb);
                            f.ctx = posb_raw(R.rk[t]) / 2 * 4 + dk;
                            lf[l].push_back(f);
                        }
                }
            }
            (void)steps;
            std::vector<std::pair<uint64_t, const Fe*>> ord;  // key = step << 6 | lane
            for (int l = 0; l < kLanes; ++l)
                for (const Fe& f : lf[l]) ord.push_back({(static_cast<uint64_t>(f.step) << 6) | static_cast<uint64_t>(l), &f});
            std::sort(ord.begin(), ord.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (auto& [key, fp] : ord) {
                RS.sym.push_back(static_cast<uint8_t>(fp->cl));
                RS.ctx.push_back(static_cast<uint16_t>(fp->ctx));
                RS.ex.push_back(fp->ex);
                RS.nb.push_back(static_cast<uint8_t>(fp->nb));
            }
        }
        const size_t NR = RS.sym.size();
        size_t rleft[kLanes];
        for (int l = 0; l < kLanes; ++l) rleft[l] = NR > static_cast<size_t>(l) ? (NR - l + kLanes - 1) / kLanes : 0;
        std::vector<uint8_t> res_body;
        size_t nwr = lockstep_encode(res_body, rleft, [](int l, size_t s) { return s * kLanes + l; }, RS, TV, cfg.pbv);

        auto& b = blobs[c];
        leb_put(b, nwm);
        leb_put(b, nwr);
        b.insert(b.end(), main_body.begin(), main_body.end());
        b.insert(b.end(), res_body.begin(), res_body.end());
    }
    t0 = o.size();
    for (auto& b : blobs) leb_put(o, b.size());
    for (auto& b : blobs) o.insert(o.end(), b.begin(), b.end());
    out.chunks = o.size() - t0;
    for (int i = 0; i < 64; ++i) o.push_back(0);  // read padding
    return out;
}

// ------------------------------------------------------------------------------------------------
// Decoder state
// ------------------------------------------------------------------------------------------------
struct DecState {
    Cfg cfg;
    DecTables gap, res;
    std::vector<uint32_t> ptr;          // n+1
    std::vector<const uint8_t*> blob;   // per chunk
    uint32_t chunk_cols = 1024;
    uint32_t n = 0;
};

struct Scratch {
    std::vector<uint32_t> fl, flr, ctxb, vals, kk, kl;
    uint64_t prof[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};

inline bool& use_simd() { static bool v = true; return v; }

struct MainIn {
    const uint32_t* kl;          // concatenated per-lane lists of non-empty column counts (+ sentinel)
    uint32_t kl_start[kLanes];   // index of each lane's first list entry
    uint32_t left[kLanes];       // elements per lane
    uint32_t pos0[kLanes];       // output position of each lane's first element
};

// Flagged (value>1) elements found by the main pass, in lock-step (step, lane) order: item = chunk position | dk << 28.
struct FlagList {
    uint32_t* item;
    uint32_t* rank;
    uint32_t nf = 0;
};

// Scalar main pass (reference for the vector kernel). out_rk gets the rank; flagged elements are appended to fl.
inline void main_pass_scalar(const DecState& D, const uint8_t* wp, const MainIn& in, const uint32_t* st, uint32_t* out_rk, FlagList& fl) {
    const Cfg& cfg = D.cfg;
    const auto& ct = cls_tab();
    const uint32_t pb = cfg.pb, mask = (1u << pb) - 1;
    const uint32_t* tab = D.gap.tab.get();
    uint32_t X[kLanes], PP[kLanes], REM[kLanes], KC[kLanes], DK[kLanes], OUT[kLanes];
    size_t LEFT[kLanes];
    const uint32_t* KP[kLanes];
    size_t steps = 0;
    for (int l = 0; l < kLanes; ++l) {
        X[l] = st[l];
        LEFT[l] = in.left[l];
        steps = std::max(steps, LEFT[l]);
        OUT[l] = in.pos0[l];
        PP[l] = 0;
        KP[l] = in.kl + in.kl_start[l];
        if (LEFT[l]) {
            uint32_t k = *KP[l];
            uint32_t kraw = static_cast<uint32_t>(ilog2u(k + 1));
            REM[l] = k; KC[l] = cfg.kcls(kraw); DK[l] = dk_of(kraw);
        }
    }
    uint32_t E[kLanes];
    for (size_t s = 0; s < steps; ++s) {
        for (int l = 0; l < kLanes; ++l) {
            if (!LEFT[l]) continue;
            uint32_t ctx = cfg.pcls(posb_raw(PP[l])) * cfg.nk + KC[l];
            uint32_t x = X[l];
            uint32_t e = tab[(ctx << pb) + (x & mask)];
            x = ((e >> 8) & 0xFFF) * (x >> pb) + (e >> 20);
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            X[l] = x;
            E[l] = e;
        }
        for (int l = 0; l < kLanes; ++l) {
            if (!LEFT[l]) continue;
            uint32_t sym = E[l] & 0xFF, cls = sym >> 1;
            uint32_t nb = ct.nb[cls];
            uint32_t x = X[l];
            uint32_t ex = x & ((1u << nb) - 1);
            x >>= nb;
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            X[l] = x;
            uint32_t rank = PP[l] + ct.base[cls] + ex;
            if (sym & 1) { fl.item[fl.nf] = OUT[l] | (DK[l] << 28); fl.rank[fl.nf] = rank; ++fl.nf; }
            out_rk[OUT[l]++] = rank;
            PP[l] = rank + 1;
            --LEFT[l];
            if (--REM[l] == 0 && LEFT[l]) {
                uint32_t k = *++KP[l];
                uint32_t kraw = static_cast<uint32_t>(ilog2u(k + 1));
                REM[l] = k; KC[l] = cfg.kcls(kraw); DK[l] = dk_of(kraw); PP[l] = 0;
            }
        }
    }
}

}  // namespace tb
}  // namespace cs
