// SPDX-License-Identifier: MIT
// w3_fast decoder side: file parse (Prep), per-chunk lane setup (shared by every ISA), scalar reference decoder.
//
// Decode semantics (the format definition; SIMD decoders must match it bit for bit). For chunk step s = 0..steps-1,
// every lane l with s < cnt[l] performs, in this global order (lanes ascending inside each phase):
//   A  : e = gap[KB + (pcls(posb(PP)) << pb) + (X & (2^pb - 1))] ; X = f * (X >> pb) + bias ; refill if X < 2^16
//   B  : cls = sym >> 1 ; flag = sym & 1 ; nb = nbits[cls] ; ex = X & (2^nb - 1) ; X >>= nb ; refill if X < 2^16
//        rank = PP + base[cls] + ex
//   C1 : (flagged lanes) e2 = res[(((posb(rank) >> 1) * 4 + dk) << pbv) + (X & (2^pbv - 1))] ; X update ; refill
//   C2 : (flagged lanes) cls2 = e2 >> 24 ; nb2 ; ex2 = X & mask ; X >>= nb2 ; refill ; value = 2 + base2 + ex2
// A refill reads the next little-endian u16 of the chunk's shared word stream: X = (X << 16) | word.
#pragma once
#include "w3_fast_enc.h"

namespace cs {
namespace wf {

struct ChunkInfo {
    const uint8_t* blob = nullptr;
    size_t bytes = 0;
    uint32_t c0 = 0, nc = 0;
    uint64_t nnz0 = 0;   // elements before this chunk
    uint32_t nnz = 0;
};

struct Prep {
    Hdr h;
    DecTables cnt, gap, res;
    std::vector<ChunkInfo> chunks;
    const uint8_t* names_p = nullptr;   // start of the names section
    const uint8_t* rowc_p = nullptr;    // start of the row-count stream
    const uint8_t* payload_p = nullptr;
    int L() const { return h.L; }
};

// Parse everything up to the chunk blobs. `b` must stay alive (and have >= 64 readable bytes past the last blob, which the
// encoder appends) while chunks are decoded.
inline void parse_file(const uint8_t* b, Prep& P) {
    const uint8_t* p = b;
    P.h = get_hdr(p);
    P.names_p = p;
    skip_stream(p); skip_stream(p);
    detail::get_u32(p);
    skip_stream(p);
    P.rowc_p = p;
    skip_stream(p);
    P.payload_p = p;
    P.cnt = parse_tables(p, 12, 1);
    P.gap = parse_tables(p, P.h.cfg.pb, P.h.cfg.nctx());
    P.res = parse_tables(p, P.h.cfg.pbv, kResCtx);
    uint32_t nchunks = detail::get_u32(p);
    std::vector<size_t> sz(nchunks);
    P.chunks.resize(nchunks);
    uint64_t acc = 0;
    for (uint32_t c = 0; c < nchunks; ++c) {
        sz[c] = leb_get(p);
        P.chunks[c].nnz = static_cast<uint32_t>(leb_get(p));
        P.chunks[c].nnz0 = acc;
        acc += P.chunks[c].nnz;
        P.chunks[c].c0 = c * P.h.chunk_cols;
        P.chunks[c].nc = std::min(P.h.n, P.chunks[c].c0 + P.h.chunk_cols) - P.chunks[c].c0;
    }
    for (uint32_t c = 0; c < nchunks; ++c) { P.chunks[c].blob = p; P.chunks[c].bytes = sz[c]; p += sz[c]; }
}

// Per-chunk lane setup. Everything the main loops need that is derivable from the column counts.
struct Setup {
    int L = 32;
    uint32_t nc = 0, N = 0, S = 0, r = 0, steps = 0;
    std::vector<uint32_t> k, colptr, kl;
    alignas(64) uint32_t e0[kMaxLanes], cnt[kMaxLanes], rem0[kMaxLanes], pp0[kMaxLanes], kb0[kMaxLanes], dk0[kMaxLanes], kp0[kMaxLanes];
    const uint8_t* states = nullptr;   // L x u32
    const uint8_t* words = nullptr;    // shared u16 word stream
};

inline void setup_chunk(const Prep& P, const ChunkInfo& ci, Setup& su) {
    const int L = P.h.L;
    const Cfg& cfg = P.h.cfg;
    su.L = L;
    su.nc = ci.nc;
    const uint8_t* q = ci.blob;
    size_t nwc = leb_get(q);
    size_t nw = leb_get(q);
    (void)nw;
    // counts
    su.k.resize(su.nc + 1);
    su.colptr.resize(su.nc + 1);
    {
        const uint32_t* ct = P.cnt.tab.get();
        const auto& cl = cls_tab();
        uint32_t x = *reinterpret_cast<const u32u*>(q);
        const uint8_t* wp = q + 4;
        uint32_t acc = 0;
        su.colptr[0] = 0;
        for (uint32_t j = 0; j < su.nc; ++j) {
            uint32_t e = ct[x & 4095];
            x = ((e >> 12) & 0xFFF) * (x >> 12) + (e & 0xFFF);
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            uint32_t c = e >> 24, nb = cl.nb[c];
            uint32_t ex = x & ((1u << nb) - 1);
            x >>= nb;
            if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; }
            uint32_t kk = cl.base[c] + ex;
            su.k[j] = kk;
            acc += kk;
            su.colptr[j + 1] = acc;
        }
        q += 4 + nwc * 2;
        su.N = acc;
    }
    // lanes
    const uint32_t N = su.N;
    su.S = N / L; su.r = N % L; su.steps = su.S + (su.r ? 1 : 0);
    su.kl.clear();
    uint32_t j = 0;
    for (int l = 0; l < L; ++l) {
        su.e0[l] = static_cast<uint32_t>(l) * su.S + std::min<uint32_t>(static_cast<uint32_t>(l), su.r);
        su.cnt[l] = su.S + (static_cast<uint32_t>(l) < su.r ? 1 : 0);
        su.kp0[l] = static_cast<uint32_t>(su.kl.size());
        if (!su.cnt[l]) { su.rem0[l] = 1u << 27; su.pp0[l] = 0; su.kb0[l] = 0; su.dk0[l] = 0; continue; }
        while (su.colptr[j + 1] <= su.e0[l]) ++j;
        uint32_t o = su.e0[l] - su.colptr[j];
        uint32_t kf = su.k[j];
        su.rem0[l] = kf - o;
        uint32_t kraw = static_cast<uint32_t>(ilog2u(kf + 1));
        su.kb0[l] = cfg.kbase(kf);
        su.dk0[l] = dk_of(kraw);
        su.pp0[l] = o ? static_cast<uint32_t>(leb_get(q)) : 0;   // read below in lane order
        int64_t remaining = static_cast<int64_t>(su.cnt[l]) - su.rem0[l];
        uint32_t jj = j + 1;
        while (remaining > 0) {
            uint32_t kk = su.k[jj++];
            if (kk) { su.kl.push_back(kk); remaining -= kk; }
        }
    }
    for (int i = 0; i < 32; ++i) su.kl.push_back(0);   // sentinels / gather padding
    su.states = q;
    su.words = q + 4 * L;
}

// ---- scalar reference decoder ---------------------------------------------------------------------------------------------
template <class OT> inline void store_val(OT* p, uint32_t v) { *p = static_cast<OT>(v); }

template <class OT>
inline void decode_chunk_scalar(const Prep& P, const ChunkInfo& ci, int32_t* idx_out, OT* val_out, Setup& su) {
    setup_chunk(P, ci, su);
    const int L = su.L;
    const Cfg& cfg = P.h.cfg;
    const int pb = cfg.pb, pbv = cfg.pbv;
    const uint32_t mask = (1u << pb) - 1, maskv = (1u << pbv) - 1;
    const uint32_t* gt = P.gap.tab.get();
    const uint32_t* rt = P.res.tab.get();
    const auto& cl = cls_tab();
    uint32_t X[kMaxLanes], PP[kMaxLanes], REM[kMaxLanes], KB[kMaxLanes], DK[kMaxLanes], E[kMaxLanes], RK[kMaxLanes], E2[kMaxLanes];
    const uint32_t* KP[kMaxLanes];
    uint32_t OUT[kMaxLanes];
    for (int l = 0; l < L; ++l) {
        X[l] = *reinterpret_cast<const u32u*>(su.states + 4 * l);
        PP[l] = su.pp0[l]; REM[l] = su.rem0[l]; KB[l] = su.kb0[l]; DK[l] = su.dk0[l];
        KP[l] = su.kl.data() + su.kp0[l];
        OUT[l] = su.e0[l];
    }
    const uint8_t* wp = su.words;
    auto refill = [&](uint32_t x) { if (x < kRansL) { x = (x << 16) | *reinterpret_cast<const u16u*>(wp); wp += 2; } return x; };
    for (uint32_t s = 0; s < su.steps; ++s) {
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l]) continue;
            uint32_t pc = cfg.pcls(posb_raw(PP[l]));
            uint32_t e = gt[KB[l] + (pc << pb) + (X[l] & mask)];
            uint32_t x = ((e >> 12) & 0xFFF) * (X[l] >> pb) + (e & 0xFFF);
            X[l] = refill(x);
            E[l] = e;
        }
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l]) continue;
            uint32_t sym = E[l] >> 24, c = sym >> 1, nb = cl.nb[c];
            uint32_t x = X[l];
            uint32_t ex = x & ((1u << nb) - 1);
            X[l] = refill(x >> nb);
            RK[l] = PP[l] + cl.base[c] + ex;
        }
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l] || !((E[l] >> 24) & 1)) continue;
            uint32_t cx = ((posb_raw(RK[l]) >> 1) * 4 + DK[l]) << pbv;
            uint32_t e = rt[cx + (X[l] & maskv)];
            uint32_t x = ((e >> 12) & 0xFFF) * (X[l] >> pbv) + (e & 0xFFF);
            X[l] = refill(x);
            E2[l] = e;
        }
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l]) continue;
            uint32_t v = 1;
            if ((E[l] >> 24) & 1) {
                uint32_t c = E2[l] >> 24, nb = cl.nb[c];
                uint32_t x = X[l];
                uint32_t ex = x & ((1u << nb) - 1);
                X[l] = refill(x >> nb);
                v = 2 + cl.base[c] + ex;
            }
            idx_out[OUT[l]] = static_cast<int32_t>(RK[l]);
            store_val<OT>(val_out + OUT[l], v);
            ++OUT[l];
            PP[l] = RK[l] + 1;
            if (--REM[l] == 0 && s + 1 < su.cnt[l]) {
                uint32_t kk = *KP[l]++;
                REM[l] = kk;
                PP[l] = 0;
                KB[l] = cfg.kbase(kk);
                DK[l] = dk_of(static_cast<uint32_t>(ilog2u(kk + 1)));
            }
        }
    }
}

}  // namespace wf
}  // namespace cs
