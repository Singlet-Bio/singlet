// SPDX-License-Identifier: MIT
// w3_fast encoder + container (header, names, row-count side stream, payload). See w3_fast_core.h for the design and
// notes/w3_fast.md for the byte-exact format.
#pragma once
#include "w3_fast_core.h"

namespace cs {
namespace wf {

// ---- container pieces (copied from Track B's side-info coding so the sizes are comparable) --------------------------
inline void put_names(std::vector<uint8_t>& o, const CanonicalMatrix& M) {
    using namespace detail;
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 3);
    put_stream(o, cn.data(), cn.size(), 3);
    std::vector<uint8_t> kv;
    for (auto& [k, v] : M.user_kv) {
        kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
        kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
    }
    put_u32(o, static_cast<uint32_t>(M.user_kv.size()));
    put_stream(o, kv.data(), kv.size(), 3);
}
inline void get_names(const uint8_t*& p, CanonicalMatrix& M) {
    using namespace detail;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    uint32_t nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * static_cast<size_t>(nkv));
    for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
}
inline void skip_stream(const uint8_t*& p) {
    uint64_t n = detail::get_u64(p), zn = detail::get_u64(p);
    (void)n;
    p += zn;
}
inline void put_rowcounts(std::vector<uint8_t>& o, const std::vector<uint32_t>& rnnz) {
    std::vector<uint8_t> b;
    for (auto c : rnnz) leb_put(b, c);
    detail::put_stream(o, b.data(), b.size(), 3);
}
inline std::vector<uint32_t> get_rowcounts(const uint8_t*& p, uint32_t m) {
    auto b = detail::get_stream(p);
    const uint8_t* q = b.data();
    std::vector<uint32_t> r(m);
    for (uint32_t i = 0; i < m; ++i) r[i] = static_cast<uint32_t>(leb_get(q));
    return r;
}

// ---- per-element coding info (whole file; independent of chunking and lane assignment) ----------------------------
struct ElemSyms {
    std::vector<uint8_t> sym;    // cls * 2 + flag(value > 1)
    std::vector<uint16_t> ctx;   // gap context = kc * npc + pc
    std::vector<uint32_t> ex;    // extra gap bits
    std::vector<uint8_t> nb;
    std::vector<uint8_t> rsym;   // residual (value - 2) class  (valid where flag)
    std::vector<uint8_t> rctx;   // (posb(rank) / 2) * 4 + dk
    std::vector<uint32_t> rex;
    std::vector<uint8_t> rnb;
};

inline void compute_syms(const ClsMat& R, const Cfg& cfg, ElemSyms& S, std::vector<uint64_t>& hg, std::vector<uint64_t>& hr) {
    const size_t nnz = R.nnz, A = 2 * kNumCls;
    S.sym.resize(nnz); S.ctx.resize(nnz); S.ex.resize(nnz); S.nb.resize(nnz);
    S.rsym.resize(nnz); S.rctx.resize(nnz); S.rex.resize(nnz); S.rnb.resize(nnz);
    hg.assign(cfg.nctx() * A, 0);
    hr.assign(static_cast<size_t>(kResCtx) * kNumCls, 0);
    const uint32_t npc = static_cast<uint32_t>(cfg.npc());
    for (uint32_t j = 0; j < R.n; ++j) {
        uint32_t k = R.ptr[j + 1] - R.ptr[j];
        if (!k) continue;
        uint32_t kraw = static_cast<uint32_t>(ilog2u(k + 1));
        uint32_t kc = cfg.kcls(kraw), dk = dk_of(kraw);
        uint32_t pp = 0;
        for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            uint32_t r = R.rk[t];
            uint32_t g = r - pp;
            uint32_t ex, nb;
            uint32_t cls = g_cls(g, ex, nb);
            uint32_t flag = R.val[t] > 1;
            uint32_t ctx = kc * npc + cfg.pcls(posb_raw(pp));
            S.sym[t] = static_cast<uint8_t>(cls * 2 + flag);
            S.ctx[t] = static_cast<uint16_t>(ctx);
            S.ex[t] = ex;
            S.nb[t] = static_cast<uint8_t>(nb);
            hg[ctx * A + cls * 2 + flag]++;
            if (flag) {
                uint32_t vex, vnb;
                uint32_t vc = g_cls(R.val[t] - 2, vex, vnb);
                if (vnb > 16) throw std::runtime_error("w3_fast: value too large");
                uint32_t rc = (posb_raw(r) >> 1) * 4 + dk;
                S.rsym[t] = static_cast<uint8_t>(vc);
                S.rctx[t] = static_cast<uint8_t>(rc);
                S.rex[t] = vex;
                S.rnb[t] = static_cast<uint8_t>(vnb);
                hr[rc * kNumCls + vc]++;
            }
            pp = r + 1;
        }
    }
}

// Estimated payload bytes (ideal static coding + table bytes) of the gap/flag stream for a context configuration.
inline size_t estimate_bytes(const ClsMat& R, const Cfg& cfg) {
    const size_t A = 2 * kNumCls;
    std::vector<uint64_t> hist(cfg.nctx() * A, 0);
    double extra = 0;
    const uint32_t npc = static_cast<uint32_t>(cfg.npc());
    for (uint32_t j = 0; j < R.n; ++j) {
        uint32_t k = R.ptr[j + 1] - R.ptr[j];
        if (!k) continue;
        uint32_t kc = cfg.kcls(static_cast<uint32_t>(ilog2u(k + 1)));
        uint32_t pp = 0;
        for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            uint32_t ex, nb;
            uint32_t cls = g_cls(R.rk[t] - pp, ex, nb);
            hist[(kc * npc + cfg.pcls(posb_raw(pp))) * A + cls * 2 + (R.val[t] > 1)]++;
            extra += nb;
            pp = R.rk[t] + 1;
        }
    }
    auto T = build_enc_tables(hist, cfg.nctx(), A, cfg.pb);
    double bits = extra;
    for (size_t x = 0; x < cfg.nctx(); ++x)
        for (size_t s = 0; s < A; ++s) {
            uint64_t c = hist[x * A + s];
            if (c) bits += static_cast<double>(c) * (cfg.pb - std::log2(static_cast<double>(T.freq[x * A + s])));
        }
    std::vector<uint8_t> tb;
    serialize_tables(tb, T);
    return static_cast<size_t>(bits / 8.0) + tb.size();
}

// ---- lane geometry shared by encoder and decoders ---------------------------------------------------------------------
struct Lanes {
    uint32_t N, S, r, steps;
    uint32_t e0[kMaxLanes], cnt[kMaxLanes];
    void init(uint32_t N_, int L) {
        N = N_; S = N / L; r = N % L; steps = S + (r ? 1 : 0);
        for (int l = 0; l < L; ++l) {
            e0[l] = static_cast<uint32_t>(l) * S + std::min<uint32_t>(static_cast<uint32_t>(l), r);
            cnt[l] = S + (static_cast<uint32_t>(l) < r ? 1 : 0);
        }
    }
};

inline void encode_chunk(const ClsMat& R, const ElemSyms& S, const Cfg& cfg, const EncTables& TG, const EncTables& TR,
                         const EncTables& TC, int L, uint32_t c0, uint32_t c1, std::vector<uint8_t>& blob) {
    const uint32_t nc = c1 - c0;
    const uint32_t base = R.ptr[c0];
    const uint32_t N = R.ptr[c1] - base;
    if (N >= (1u << 28)) throw std::runtime_error("w3_fast: chunk too large");
    std::vector<uint32_t> k(nc), colptr(nc + 1, 0);
    for (uint32_t j = 0; j < nc; ++j) { k[j] = R.ptr[c0 + j + 1] - R.ptr[c0 + j]; colptr[j + 1] = colptr[j] + k[j]; }

    // counts stream (single lane)
    std::vector<uint8_t> cb;
    size_t nwc;
    {
        LockEnc E(1);
        for (uint32_t j = nc; j-- > 0;) {
            uint32_t ex, nb;
            uint32_t cls = g_cls(k[j], ex, nb);
            E.put_uniform(0, ex, static_cast<int>(nb));
            E.put(0, TC.freq[cls], TC.cum[cls], 12);
        }
        nwc = E.finish(cb);
    }
    Lanes ln;
    ln.init(N, L);
    // lane start offsets inside columns -> pp0
    std::vector<uint8_t> ppb;
    {
        uint32_t j = 0;
        for (int l = 0; l < L; ++l) {
            if (!ln.cnt[l]) continue;
            while (colptr[j + 1] <= ln.e0[l]) ++j;
            uint32_t o = ln.e0[l] - colptr[j];
            if (o) leb_put(ppb, R.rk[base + ln.e0[l] - 1] + 1);
        }
    }
    LockEnc E(L);
    const size_t A = TG.A, AR = TR.A;
    for (uint32_t s = ln.steps; s-- > 0;) {
        for (int l = L - 1; l >= 0; --l)  // C2: residual extras
            if (s < ln.cnt[l]) { size_t t = base + ln.e0[l] + s; if (S.sym[t] & 1) E.put_uniform(l, S.rex[t], S.rnb[t]); }
        for (int l = L - 1; l >= 0; --l)  // C1: residual symbol
            if (s < ln.cnt[l]) {
                size_t t = base + ln.e0[l] + s;
                if (S.sym[t] & 1) { size_t q = static_cast<size_t>(S.rctx[t]) * AR + S.rsym[t]; E.put(l, TR.freq[q], TR.cum[q], cfg.pbv); }
            }
        for (int l = L - 1; l >= 0; --l)  // B: gap extras
            if (s < ln.cnt[l]) { size_t t = base + ln.e0[l] + s; E.put_uniform(l, S.ex[t], S.nb[t]); }
        for (int l = L - 1; l >= 0; --l)  // A: gap class + flag
            if (s < ln.cnt[l]) {
                size_t t = base + ln.e0[l] + s;
                size_t q = static_cast<size_t>(S.ctx[t]) * A + S.sym[t];
                E.put(l, TG.freq[q], TG.cum[q], cfg.pb);
            }
    }
    std::vector<uint8_t> mb;
    size_t nw = E.finish(mb);
    blob.clear();
    leb_put(blob, nwc);
    leb_put(blob, nw);
    blob.insert(blob.end(), cb.begin(), cb.end());
    blob.insert(blob.end(), ppb.begin(), ppb.end());
    blob.insert(blob.end(), mb.begin(), mb.end());
}

// File header fields needed by the decoder.
struct Hdr {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    uint8_t vt = 1, L = 32, cpo = 4;
    Cfg cfg;
    uint32_t chunk_cols = 1024;
};
constexpr uint8_t kMagic = 'W', kVersion = 1;

inline void put_hdr(std::vector<uint8_t>& o, const Hdr& h) {
    o.push_back(kMagic); o.push_back(kVersion);
    detail::put_u32(o, h.m); detail::put_u32(o, h.n); detail::put_u64(o, h.nnz);
    o.push_back(h.vt); o.push_back(h.L); o.push_back(h.cpo);
    const uint8_t v[7] = {h.cfg.pb, h.cfg.pbv, h.cfg.sh, h.cfg.cap, h.cfg.ks, h.cfg.kcap, h.cfg.nk};
    o.insert(o.end(), v, v + 7);
    detail::put_u32(o, h.chunk_cols);
}
inline Hdr get_hdr(const uint8_t*& p) {
    Hdr h;
    if (p[0] != kMagic || p[1] != kVersion) throw std::runtime_error("w3_fast: bad magic/version");
    p += 2;
    h.m = detail::get_u32(p); h.n = detail::get_u32(p); h.nnz = detail::get_u64(p);
    h.vt = *p++; h.L = *p++; h.cpo = *p++;
    h.cfg.pb = p[0]; h.cfg.pbv = p[1]; h.cfg.sh = p[2]; h.cfg.cap = p[3]; h.cfg.ks = p[4]; h.cfg.kcap = p[5]; h.cfg.nk = p[6];
    p += 7;
    h.chunk_cols = detail::get_u32(p);
    return h;
}

inline std::vector<uint8_t> encode_matrix(const CanonicalMatrix& M, const Params& prm) {
    if (M.m > kMaxRows) throw std::runtime_error("w3_fast: m > 2^19 unsupported");
    if (prm.lanes != 16 && prm.lanes != 32 && prm.lanes != 64) throw std::runtime_error("w3_fast: lanes must be 16/32/64");
    Sizes sz;
    Hdr h;
    h.m = M.m; h.n = M.n; h.nnz = M.nnz; h.L = static_cast<uint8_t>(prm.lanes); h.cpo = static_cast<uint8_t>(prm.cpo);
    {
        uint32_t mx = 0;
        for (auto v : M.data) mx = std::max(mx, v);
        h.vt = mx <= 255 ? 1 : mx <= 65535 ? 2 : 3;
    }
    h.chunk_cols = static_cast<uint32_t>(prm.chunk_cols > 0 ? prm.chunk_cols : auto_chunk_cols(M.n, M.nnz, prm.target_nnz));
    std::vector<uint32_t> rnnz(M.m, 0);
    for (auto r : M.indices) rnnz[r]++;
    RowOrder O = order_from_counts(rnnz, prm.cpo);
    ClsMat R = to_class_space(M, O);

    int kind = prm.ctx;
    if (kind < 0) {
        size_t bs = ~size_t(0);
        for (int kd = 0; kd < 5; ++kd) {
            size_t e = estimate_bytes(R, Cfg::make(kd, prm.pb, prm.pbv));
            if (e < bs) { bs = e; kind = kd; }
        }
    }
    h.cfg = Cfg::make(kind, prm.pb, prm.pbv);
    sz.ctx = kind;
    sz.chunk_cols = static_cast<int>(h.chunk_cols);

    std::vector<uint8_t> o;
    put_hdr(o, h);
    sz.header = o.size();
    size_t t0 = o.size();
    put_names(o, M);
    sz.names = o.size() - t0;
    t0 = o.size();
    put_rowcounts(o, rnnz);
    sz.side = o.size() - t0;
    t0 = o.size();

    ElemSyms S;
    std::vector<uint64_t> hg, hr;
    compute_syms(R, h.cfg, S, hg, hr);
    EncTables TG = build_enc_tables(hg, h.cfg.nctx(), 2 * kNumCls, h.cfg.pb);
    EncTables TR = build_enc_tables(hr, kResCtx, kNumCls, h.cfg.pbv);
    std::vector<uint64_t> hc(kNumCls, 0);
    for (uint32_t j = 0; j < M.n; ++j) { uint32_t ex, nb; hc[g_cls(R.ptr[j + 1] - R.ptr[j], ex, nb)]++; }
    if (M.n == 0) hc[0] = 1;
    EncTables TC = build_enc_tables(hc, 1, kNumCls, 12);
    serialize_tables(o, TC);
    serialize_tables(o, TG);
    serialize_tables(o, TR);
    sz.tables = o.size() - t0;

    const uint32_t nchunks = (M.n + h.chunk_cols - 1) / h.chunk_cols;
    std::vector<std::vector<uint8_t>> blobs(nchunks);
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * h.chunk_cols, c1 = std::min(M.n, c0 + h.chunk_cols);
        encode_chunk(R, S, h.cfg, TG, TR, TC, prm.lanes, c0, c1, blobs[c]);
    }
    t0 = o.size();
    detail::put_u32(o, nchunks);
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * h.chunk_cols, c1 = std::min(M.n, c0 + h.chunk_cols);
        leb_put(o, blobs[c].size());
        leb_put(o, R.ptr[c1] - R.ptr[c0]);
    }
    for (auto& b : blobs) o.insert(o.end(), b.begin(), b.end());
    for (int i = 0; i < 64; ++i) o.push_back(0);  // read padding
    sz.chunks = o.size() - t0;
    sz.payload = sz.tables + sz.chunks + 7 + 4;   // tables + chunk table + blobs + padding + cfg(7) + chunk_cols(4), like Track B
    sz.total = o.size();
    last_sizes() = sz;
    return o;
}

}  // namespace wf
}  // namespace cs
