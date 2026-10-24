// SPDX-License-Identifier: MIT
// pz2 codec 1 encoder.  Produces the `tables`, `rows` and `chunks` sections of a member (see api.h for the container).
#pragma once
#include <atomic>
#include <thread>

namespace singlet {
namespace pz2 {
namespace detail {

/// Run fn(i, thread_index) for i in [0, n) on up to `threads` threads with dynamic scheduling; rethrows the first exception.
template <class Fn>
inline void parallel_for(int threads, size_t n, Fn fn) {
    if (threads <= 1 || n <= 1) { for (size_t i = 0; i < n; ++i) fn(i, 0); return; }
    if (size_t(threads) > n) threads = int(n);
    std::atomic<size_t> next{0};
    std::exception_ptr err;
    std::mutex mu;
    auto work = [&](int t) {
        try {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= n) break;
                fn(i, t);
            }
        } catch (...) {
            std::lock_guard<std::mutex> g(mu);
            if (!err) err = std::current_exception();
            next.store(n);
        }
    };
    std::vector<std::thread> th;
    for (int t = 1; t < threads; ++t) th.emplace_back(work, t);
    work(0);
    for (auto& t : th) t.join();
    if (err) std::rethrow_exception(err);
}
inline int resolve_threads(int t, int cap) {
    if (t > 0) return t;
    unsigned h = std::thread::hardware_concurrency();
    return int(std::max(1u, std::min<unsigned>(h ? h : 1, unsigned(cap))));
}

namespace r1 {

/// A matrix in stored (class-order) row space: ids ascending in every column.
struct ClsMat {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    std::vector<uint64_t> ptr;       // n + 1
    std::vector<uint32_t> rk, val;   // nnz
};

/// Sorts every column into stored row space.  Parallel over column blocks.
inline void to_class_space(const CSC& in, const std::vector<uint64_t>& ptr, const uint32_t* val32,
                           const std::vector<uint32_t>& inv, ClsMat& R, int threads) {
    R.m = in.m; R.n = in.n; R.nnz = in.nnz; R.ptr = ptr;
    R.rk.resize(size_t(in.nnz));
    R.val.resize(size_t(in.nnz));
    const uint32_t blk = 256;
    const size_t nb = (size_t(in.n) + blk - 1) / blk;
    parallel_for(threads, nb, [&](size_t b, int) {
        std::vector<uint64_t> tmp;
        uint32_t j0 = uint32_t(b * blk), j1 = uint32_t(std::min<size_t>(in.n, size_t(j0) + blk));
        for (uint32_t j = j0; j < j1; ++j) {
            uint64_t a = ptr[j], e = ptr[j + 1];
            tmp.clear();
            for (uint64_t k = a; k < e; ++k) tmp.push_back((uint64_t(inv[in.indices[k]]) << 32) | val32[k]);
            std::sort(tmp.begin(), tmp.end());
            for (size_t t = 0; t < tmp.size(); ++t) { R.rk[a + t] = uint32_t(tmp[t] >> 32); R.val[a + t] = uint32_t(tmp[t]); }
        }
    });
}

// ---- per-element coding info of one chunk -----------------------------------------------------------------------------
struct ChunkSyms {
    std::vector<uint8_t> sym;    // cls * 2 + flag
    std::vector<uint16_t> ctx;   // gap context = kc * npc + pc
    std::vector<uint32_t> ex;    // gap extra bits
    std::vector<uint8_t> nb;
    std::vector<uint8_t> rsym;   // residual (value - 2) class, valid where flag
    std::vector<uint8_t> rctx;   // (posb(rank) / 2) * 4 + dk
    std::vector<uint32_t> rex;
    std::vector<uint8_t> rnb;
};

/// Walk columns [c0, c1): accumulate gap/residual histograms (hg, hr may be null) and fill `S` (may be null).
inline void scan_columns(const ClsMat& R, const Cfg& cfg, uint32_t c0, uint32_t c1, ChunkSyms* S, uint64_t* hg, uint64_t* hr) {
    const size_t A = kAlpha;
    const uint32_t npc = uint32_t(cfg.npc());
    const uint64_t base = R.ptr[c0];
    if (S) {
        size_t N = size_t(R.ptr[c1] - base);
        S->sym.resize(N); S->ctx.resize(N); S->ex.resize(N); S->nb.resize(N);
        S->rsym.resize(N); S->rctx.resize(N); S->rex.resize(N); S->rnb.resize(N);
    }
    for (uint32_t j = c0; j < c1; ++j) {
        uint64_t a = R.ptr[j], e = R.ptr[j + 1];
        uint32_t k = uint32_t(e - a);
        if (!k) continue;
        uint32_t kraw = uint32_t(ilog2f(k + 1));
        uint32_t kc = cfg.kcls(kraw), dk = dk_of(kraw);
        uint32_t pp = 0;
        for (uint64_t t = a; t < e; ++t) {
            uint32_t r = R.rk[t];
            uint32_t g = r - pp, ex, nb;
            uint32_t cls = g_cls(g, ex, nb);
            uint32_t flag = R.val[t] != 1;
            uint32_t ctx = kc * npc + cfg.pcls(posb_raw(pp));
            if (hg) hg[size_t(ctx) * A + cls * 2 + flag]++;
            size_t o = size_t(t - base);
            if (S) { S->sym[o] = uint8_t(cls * 2 + flag); S->ctx[o] = uint16_t(ctx); S->ex[o] = ex; S->nb[o] = uint8_t(nb); }
            if (flag) {
                uint32_t vex, vnb;
                uint32_t vc = g_cls(R.val[t] ? R.val[t] - 1 : 0, vex, vnb);   // residual: 0 for value 0, value - 1 otherwise
                uint32_t rc = (posb_raw(r) >> 1) * 4 + dk;
                if (hr) hr[size_t(rc) * kNumCls + vc]++;
                if (S) { S->rsym[o] = uint8_t(vc); S->rctx[o] = uint8_t(rc); S->rex[o] = vex; S->rnb[o] = uint8_t(vnb); }
            }
            pp = r + 1;
        }
    }
}

/// Estimated payload bytes (ideal static coding plus serialised tables) of the gap/flag stream for one context configuration.
inline size_t estimate_bytes(const ClsMat& R, const Cfg& cfg, int zlevel) {
    const size_t A = kAlpha;
    std::vector<uint64_t> hist(cfg.nctx() * A, 0);
    scan_columns(R, cfg, 0, R.n, nullptr, hist.data(), nullptr);
    double extra = 0;
    const uint32_t npc = uint32_t(cfg.npc());
    for (uint32_t j = 0; j < R.n; ++j) {
        uint32_t pp = 0;
        for (uint64_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) {
            uint32_t ex, nb;
            g_cls(R.rk[t] - pp, ex, nb);
            extra += nb;
            pp = R.rk[t] + 1;
        }
    }
    (void)npc;
    EncTables T = build_enc_tables(hist, cfg.nctx(), A, cfg.pb);
    const auto& lg = lg_table();
    double bits = extra;
    for (size_t x = 0; x < cfg.nctx(); ++x)
        for (size_t s = 0; s < A; ++s) {
            uint64_t c = hist[x * A + s];
            if (c) bits += double(c) * (double(uint64_t(cfg.pb) << 32) - double(lg[T.freq[x * A + s]])) / 4294967296.0;
        }
    std::vector<uint8_t> tb;
    serialize_tables(tb, T, zlevel);
    return size_t(bits / 8.0) + tb.size();
}

// ---- lane geometry shared by encoder and decoders ---------------------------------------------------------------------
struct Lanes {
    uint32_t N, S, r, steps;
    uint32_t e0[kMaxLanes], cnt[kMaxLanes];
    void init(uint32_t N_, int L) {
        N = N_; S = N / uint32_t(L); r = N % uint32_t(L); steps = S + (r ? 1 : 0);
        for (int l = 0; l < L; ++l) {
            e0[l] = uint32_t(l) * S + std::min<uint32_t>(uint32_t(l), r);
            cnt[l] = S + (uint32_t(l) < r ? 1 : 0);
        }
    }
};

/// Encode columns [c0, c1) into one chunk blob:
///   LEB nwc | LEB nw | u32 counts_state | u16 counts_words[nwc] | LEB pp0 (lanes ascending that start inside a column) |
///   u32 state[L] | u16 words[nw]
inline void encode_chunk(const ClsMat& R, const ChunkSyms& S, const Cfg& cfg, const EncTables& TG, const EncTables& TR,
                         const EncTables& TC, int L, uint32_t c0, uint32_t c1, std::vector<uint8_t>& blob) {
    const uint32_t nc = c1 - c0;
    const uint64_t base = R.ptr[c0];
    const uint64_t N64 = R.ptr[c1] - base;
    if (N64 >= kMaxChunkN) throw Error(Errc::Unsupported, "chunk has too many non-zeros");
    const uint32_t N = uint32_t(N64);
    std::vector<uint32_t> k(nc), colptr(nc + 1, 0);
    for (uint32_t j = 0; j < nc; ++j) { k[j] = uint32_t(R.ptr[c0 + j + 1] - R.ptr[c0 + j]); colptr[j + 1] = colptr[j] + k[j]; }

    // column counts: class symbol (12-bit table, one state) + extra bits
    std::vector<uint8_t> cb;
    size_t nwc;
    {
        LockEnc E(1);
        for (uint32_t j = nc; j-- > 0;) {
            uint32_t ex, nb;
            uint32_t cls = g_cls(k[j], ex, nb);
            E.put_bits(0, ex, nb);
            E.put(0, TC.freq[cls], TC.cum[cls], 12);
        }
        E.finish(cb);       // 4 state bytes + words
        nwc = E.w.size();
    }
    Lanes ln;
    ln.init(N, L);
    std::vector<uint8_t> ppb;   // start position (previous rank + 1) of every lane that starts inside a column
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
        // reverse of the decode order A, B, B2, C1, C2, C3
        for (int l = L - 1; l >= 0; --l)   // C3: residual extra, high part
            if (s < ln.cnt[l]) { size_t t = size_t(ln.e0[l]) + s; if ((S.sym[t] & 1) && S.rnb[t] > 16) E.put_uniform(l, S.rex[t] >> 16, int(S.rnb[t] - 16)); }
        for (int l = L - 1; l >= 0; --l)   // C2: residual extra, low part
            if (s < ln.cnt[l]) { size_t t = size_t(ln.e0[l]) + s; if (S.sym[t] & 1) E.put_uniform(l, S.rnb[t] > 16 ? (S.rex[t] & 0xFFFF) : S.rex[t], int(std::min<uint32_t>(S.rnb[t], 16))); }
        for (int l = L - 1; l >= 0; --l)   // C1: residual symbol
            if (s < ln.cnt[l]) {
                size_t t = size_t(ln.e0[l]) + s;
                if (S.sym[t] & 1) { size_t q = size_t(S.rctx[t]) * AR + S.rsym[t]; E.put(l, TR.freq[q], TR.cum[q], cfg.pbv); }
            }
        for (int l = L - 1; l >= 0; --l)   // B2: gap extra, high part
            if (s < ln.cnt[l]) { size_t t = size_t(ln.e0[l]) + s; if (S.nb[t] > 16) E.put_uniform(l, S.ex[t] >> 16, int(S.nb[t] - 16)); }
        for (int l = L - 1; l >= 0; --l)   // B: gap extra, low part
            if (s < ln.cnt[l]) { size_t t = size_t(ln.e0[l]) + s; E.put_uniform(l, S.nb[t] > 16 ? (S.ex[t] & 0xFFFF) : S.ex[t], int(std::min<uint32_t>(S.nb[t], 16))); }
        for (int l = L - 1; l >= 0; --l)   // A: gap class + flag
            if (s < ln.cnt[l]) {
                size_t t = size_t(ln.e0[l]) + s;
                size_t q = size_t(S.ctx[t]) * A + S.sym[t];
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

struct Codec1Params {
    int lanes = 32, pb = 10, pbv = 10, kind = -1, cpo = 4, zlevel = 19, threads = 1;
};

/// Encoded codec-1 payload pieces.
struct Codec1Out {
    std::vector<uint8_t> tables, rows;
    std::vector<std::vector<uint8_t>> blobs;   // one per chunk
    std::vector<uint32_t> first_col;           // n_chunks + 1 (sentinel = n)
    std::vector<uint64_t> chunk_nnz;           // per chunk
    uint8_t kind = 0;
};

/// Cut columns into chunks of at most `chunk_cols` columns and kMaxChunkN/2 non-zeros.
/// Chunk starts (n_chunks + 1, sentinel n): at most `chunk_cols` columns per chunk, and a chunk ends before the column that would
/// take it past `nnz_budget` non-zeros (at least one column per chunk); `max_nnz` is the hard limit of the coder.
inline std::vector<uint32_t> plan_chunks(const std::vector<uint64_t>& ptr, uint32_t n, uint32_t chunk_cols, uint64_t max_nnz,
                                         uint64_t nnz_budget = UINT64_MAX) {
    std::vector<uint32_t> starts;
    uint32_t j = 0;
    while (j < n) {
        starts.push_back(j);
        uint32_t e = uint32_t(std::min<uint64_t>(n, uint64_t(j) + chunk_cols));
        if (nnz_budget != UINT64_MAX && ptr[e] - ptr[j] > nnz_budget) {
            // the last column end k in (j, e] with ptr[k] - ptr[j] <= budget, but at least j + 1
            const uint64_t lim = ptr[j] + nnz_budget;
            auto it = std::upper_bound(ptr.begin() + j + 1, ptr.begin() + e + 1, lim);
            e = std::max<uint32_t>(j + 1, uint32_t((it - ptr.begin()) - 1));
        }
        // shrink until the chunk fits (a single column always fits: it has < 2^21 elements)
        while (e - j > 1 && ptr[e] - ptr[j] > max_nnz) e = j + std::max<uint32_t>(1, (e - j) / 2);
        j = e;
    }
    starts.push_back(n);
    return starts;
}

inline Codec1Out encode_codec1(const CSC& in, const std::vector<uint64_t>& ptr, const uint32_t* val32, uint32_t chunk_cols,
                               const Codec1Params& prm, uint64_t nnz_budget = UINT64_MAX) {
    const uint32_t m = in.m, n = in.n;
    Codec1Out out;
    // row classes
    std::vector<uint32_t> rnnz(m, 0);
    for (uint64_t t = 0; t < in.nnz; ++t) rnnz[in.indices[t]]++;
    std::vector<uint32_t> key(m);
    for (uint32_t i = 0; i < m; ++i) key[i] = row_key(rnnz[i], prm.cpo);
    {
        std::vector<uint8_t> raw;
        for (uint32_t k : key) leb_put(raw, k);
        put_stream(out.rows, raw.data(), raw.size(), prm.zlevel);
    }
    std::vector<uint32_t> perm = perm_from_keys(key), inv(m);
    for (uint32_t s = 0; s < m; ++s) inv[perm[s]] = s;
    ClsMat R;
    to_class_space(in, ptr, val32, inv, R, prm.threads);

    int kind = prm.kind;
    if (kind < 0) {
        if (in.nnz >= 2000000) kind = 0;
        else {
            size_t bs = ~size_t(0);
            for (int kd = 0; kd < 5; ++kd) {
                size_t e = estimate_bytes(R, Cfg::make(kd, prm.pb, prm.pbv), prm.zlevel);
                if (e < bs) { bs = e; kind = kd; }
            }
        }
    }
    out.kind = uint8_t(kind);
    const Cfg cfg = Cfg::make(kind, prm.pb, prm.pbv);

    // chunk plan
    out.first_col = plan_chunks(R.ptr, n, chunk_cols, kMaxChunkN / 2, nnz_budget);
    const size_t nch = out.first_col.size() - 1;

    // pass 1: histograms (per thread, merged)
    const size_t A = kAlpha;
    const int nt = std::max(1, std::min<int>(prm.threads, int(std::max<size_t>(1, nch))));
    std::vector<std::vector<uint64_t>> hgs(nt), hrs(nt);
    for (int t = 0; t < nt; ++t) { hgs[t].assign(cfg.nctx() * A, 0); hrs[t].assign(size_t(kResCtx) * kNumCls, 0); }
    parallel_for(nt, nch, [&](size_t c, int t) { scan_columns(R, cfg, out.first_col[c], out.first_col[c + 1], nullptr, hgs[t].data(), hrs[t].data()); });
    for (int t = 1; t < nt; ++t) {
        for (size_t i = 0; i < hgs[0].size(); ++i) hgs[0][i] += hgs[t][i];
        for (size_t i = 0; i < hrs[0].size(); ++i) hrs[0][i] += hrs[t][i];
    }
    std::vector<uint64_t> hc(kNumCls, 0);
    for (uint32_t j = 0; j < n; ++j) { uint32_t ex, nb; hc[g_cls(uint32_t(R.ptr[j + 1] - R.ptr[j]), ex, nb)]++; }
    if (n == 0) hc[0] = 1;
    EncTables TG = build_enc_tables(hgs[0], cfg.nctx(), A, cfg.pb);
    EncTables TR = build_enc_tables(hrs[0], kResCtx, kNumCls, cfg.pbv);
    EncTables TC = build_enc_tables(hc, 1, kNumCls, 12);
    serialize_tables(out.tables, TC, prm.zlevel);
    serialize_tables(out.tables, TG, prm.zlevel);
    serialize_tables(out.tables, TR, prm.zlevel);

    // pass 2: encode chunks
    out.blobs.resize(nch);
    out.chunk_nnz.resize(nch);
    parallel_for(nt, nch, [&](size_t c, int) {
        ChunkSyms S;
        uint32_t c0 = out.first_col[c], c1 = out.first_col[c + 1];
        scan_columns(R, cfg, c0, c1, &S, nullptr, nullptr);
        encode_chunk(R, S, cfg, TG, TR, TC, prm.lanes, c0, c1, out.blobs[c]);
        out.chunk_nnz[c] = R.ptr[c1] - R.ptr[c0];
    });
    return out;
}

}  // namespace r1
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
