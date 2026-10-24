// SPDX-License-Identifier: MIT
// pz2 codec 1 decoder: tables, per-chunk setup (shared by every ISA) and the scalar reference decoder.
//
// Decode semantics (the format definition; the SIMD decoders must match it bit for bit).  For chunk step s = 0 .. steps-1 every lane
// l with s < cnt[l] performs, in this global order, lanes ascending inside each phase:
//   (flag = value != 1; residual r = 0 for value 0, value - 1 otherwise; decoded value = r ? r + 1 : 0)
//   A  : e = gap[KB + (pcls(posb(PP)) << pb) + (X & (2^pb - 1))] ; X = f * (X >> pb) + bias ; refill
//   B  : cls = (e >> 24) >> 1 ; nb = nbits[cls] ; lo = min(nb, 16) ; ex = X & (2^lo - 1) ; X >>= lo ; refill
//   B2 : lanes with nb > 16: hi = nb - 16 ; ex |= (X & (2^hi - 1)) << 16 ; X >>= hi ; refill
//        rank = PP + base[cls] + ex
//   C1 : lanes with flag ((e >> 24) & 1): e2 = res[((min(posb(rank) >> 1, 20) * 4 + dk) << pbv) + (X & (2^pbv - 1))] ; X update ; refill
//   C2 : flagged: nb2 = nbits[e2 >> 24] ; lo = min(nb2, 16) ; ex2 = X & (2^lo - 1) ; X >>= lo ; refill
//   C3 : flagged lanes with nb2 > 16: hi = nb2 - 16 ; ex2 |= (X & (2^hi - 1)) << 16 ; X >>= hi ; refill
//        r = base[e2 >> 24] + ex2 ; value = r ? r + 1 : 0   (unflagged lanes: value = 1)
// A refill reads the next little-endian u16 of the chunk's shared word stream: X = (X << 16) | word, when X < 2^16.
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {
namespace r1 {

inline constexpr size_t kPad = 1024;   // readable bytes the decoder may touch beyond the end of a chunk blob

struct Prep {              // everything decode needs, shared by all chunks
    uint32_t m = 0, n = 0;
    int L = 32;
    Cfg cfg;
    DecTables cnt, gap, res;
};

inline std::shared_ptr<Prep> make_prep(const Header& h, const uint8_t* tab, size_t tab_len) {
    auto P = std::make_shared<Prep>();
    P->m = h.m; P->n = h.n; P->L = h.lanes;
    if (h.lanes != 16 && h.lanes != 32 && h.lanes != 64) throw Error(Errc::Corrupt, "bad lane count");
    if (h.prob_bits < 8 || h.prob_bits > 12 || h.value_prob_bits < 8 || h.value_prob_bits > 12) throw Error(Errc::Corrupt, "bad probability bits");
    if (h.ctx_kind > 4) throw Error(Errc::Corrupt, "bad context kind");
    P->cfg = Cfg::make(h.ctx_kind, h.prob_bits, h.value_prob_bits);
    const uint8_t* p = tab;
    const uint8_t* end = tab + tab_len;
    P->cnt = parse_tables(p, end, 12, 1, kNumCls);
    P->gap = parse_tables(p, end, P->cfg.pb, P->cfg.nctx(), kAlpha);
    P->res = parse_tables(p, end, P->cfg.pbv, kResCtx, kNumCls);
    return P;
}

// Per-chunk lane setup: everything the main loops need that is derivable from the column counts.
struct Setup {
    int L = 32;
    uint32_t nc = 0, N = 0, S = 0, r = 0, steps = 0;
    std::vector<uint32_t> k, colptr, kl;
    alignas(64) uint32_t e0[kMaxLanes], cnt[kMaxLanes], rem0[kMaxLanes], pp0[kMaxLanes], kb0[kMaxLanes], dk0[kMaxLanes], kp0[kMaxLanes];
    const uint8_t* states = nullptr;      // L x u32
    const uint8_t* words = nullptr;       // shared u16 word stream
    const uint8_t* words_end = nullptr;
};

/// Decode only the column counts of a chunk.  `blob` / `bytes` is the chunk stream; throws Corrupt on any inconsistency.
inline void chunk_counts(const Prep& P, const uint8_t* blob, size_t bytes, uint32_t nc, std::vector<uint32_t>& k, size_t* hdr_end = nullptr,
                         size_t* nw_out = nullptr) {
    const uint8_t* q = blob;
    const uint8_t* end = blob + bytes;
    size_t nwc = size_t(leb_get(q, end));
    size_t nw = size_t(leb_get(q, end));
    if (nwc > bytes || nw > bytes) throw Error(Errc::Corrupt, "chunk: word counts exceed the chunk");
    if (size_t(end - q) < 4 + 2 * nwc) throw Error(Errc::Corrupt, "chunk: truncated count stream");
    k.resize(nc);
    const uint32_t* ct = P.cnt.data();
    const auto& cl = cls_tab();
    uint32_t x = ld_u32(q);
    const uint8_t* wp = q + 4;
    const uint8_t* wend = q + 4 + 2 * nwc;
    for (uint32_t j = 0; j < nc; ++j) {
        uint32_t e = ct[x & 4095];
        x = ((e >> 12) & 0xFFF) * (x >> 12) + (e & 0xFFF);
        if (x < kL) { if (wp >= wend) throw Error(Errc::Corrupt, "chunk: count stream underrun"); x = (x << 16) | ld_u16(wp); wp += 2; }
        uint32_t c = e >> 24, nb = cl.nb[c];
        uint32_t lo = nb < 16 ? nb : 16;
        uint32_t ex = x & ((1u << lo) - 1);
        x >>= lo;
        if (x < kL) { if (wp >= wend) throw Error(Errc::Corrupt, "chunk: count stream underrun"); x = (x << 16) | ld_u16(wp); wp += 2; }
        if (nb > 16) {
            uint32_t hi = nb - 16;
            ex |= (x & ((1u << hi) - 1)) << 16;
            x >>= hi;
            if (x < kL) { if (wp >= wend) throw Error(Errc::Corrupt, "chunk: count stream underrun"); x = (x << 16) | ld_u16(wp); wp += 2; }
        }
        k[j] = cl.base[c] + ex;
    }
    if (x != kL || wp != wend) throw Error(Errc::Corrupt, "chunk: count stream integrity check failed");
    if (hdr_end) *hdr_end = size_t(wend - blob);
    if (nw_out) *nw_out = nw;
}

inline void setup_chunk(const Prep& P, const uint8_t* blob, size_t bytes, uint32_t nc, uint64_t expect_nnz, Setup& su) {
    const int L = P.L;
    const Cfg& cfg = P.cfg;
    su.L = L;
    su.nc = nc;
    size_t after = 0, nw = 0;
    chunk_counts(P, blob, bytes, nc, su.k, &after, &nw);
    su.colptr.resize(size_t(nc) + 1);
    uint64_t acc = 0;
    su.colptr[0] = 0;
    for (uint32_t j = 0; j < nc; ++j) { acc += su.k[j]; if (acc >= kMaxChunkN) throw Error(Errc::Corrupt, "chunk: too many non-zeros"); su.colptr[j + 1] = uint32_t(acc); }
    if (acc != expect_nnz) throw Error(Errc::Corrupt, "chunk: column counts disagree with the chunk index");
    su.N = uint32_t(acc);
    const uint8_t* q = blob + after;
    const uint8_t* end = blob + bytes;
    const uint32_t N = su.N;
    su.S = N / uint32_t(L); su.r = N % uint32_t(L); su.steps = su.S + (su.r ? 1 : 0);
    su.kl.clear();
    uint32_t j = 0;
    for (int l = 0; l < L; ++l) {
        su.e0[l] = uint32_t(l) * su.S + std::min<uint32_t>(uint32_t(l), su.r);
        su.cnt[l] = su.S + (uint32_t(l) < su.r ? 1 : 0);
        su.kp0[l] = uint32_t(su.kl.size());
        if (!su.cnt[l]) { su.rem0[l] = 1u << 27; su.pp0[l] = 0; su.kb0[l] = 0; su.dk0[l] = 0; continue; }
        while (su.colptr[j + 1] <= su.e0[l]) ++j;
        uint32_t o = su.e0[l] - su.colptr[j];
        uint32_t kf = su.k[j];
        su.rem0[l] = kf - o;
        uint32_t kraw = uint32_t(ilog2f(kf + 1));
        su.kb0[l] = cfg.kbase(kf);
        su.dk0[l] = dk_of(kraw);
        su.pp0[l] = o ? uint32_t(leb_get(q, end)) : 0;
        int64_t remaining = int64_t(su.cnt[l]) - int64_t(su.rem0[l]);
        uint32_t jj = j + 1;
        while (remaining > 0) {
            uint32_t kk = su.k[jj++];
            if (kk) { su.kl.push_back(kk); remaining -= kk; }
        }
    }
    for (int i = 0; i < 32; ++i) su.kl.push_back(0);   // sentinels / gather padding
    if (size_t(end - q) < 4 * size_t(L) + 2 * nw) throw Error(Errc::Corrupt, "chunk: truncated lane stream");
    su.states = q;
    su.words = q + 4 * L;
    su.words_end = su.words + 2 * nw;
}

// ---- scalar reference decoder -----------------------------------------------------------------------------------------------
/// Writes stored-space ids and u32 values of the chunk's N elements.  Returns false if the end-of-stream integrity check fails.
inline bool decode_chunk_scalar(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    const int L = su.L;
    const Cfg& cfg = P.cfg;
    const int pb = cfg.pb, pbv = cfg.pbv;
    const uint32_t mask = (1u << pb) - 1, maskv = (1u << pbv) - 1;
    const uint32_t* gt = P.gap.data();
    const uint32_t* rt = P.res.data();
    const auto& cl = cls_tab();
    uint32_t X[kMaxLanes], PP[kMaxLanes], REM[kMaxLanes], KB[kMaxLanes], DK[kMaxLanes], E[kMaxLanes], RK[kMaxLanes], E2[kMaxLanes];
    uint32_t NB[kMaxLanes], EX[kMaxLanes], NB2[kMaxLanes], EX2[kMaxLanes], OUT[kMaxLanes];
    const uint32_t* KP[kMaxLanes];
    for (int l = 0; l < L; ++l) {
        X[l] = ld_u32(su.states + 4 * l);
        PP[l] = su.pp0[l]; REM[l] = su.rem0[l]; KB[l] = su.kb0[l]; DK[l] = su.dk0[l];
        KP[l] = su.kl.data() + su.kp0[l];
        OUT[l] = su.e0[l];
    }
    const uint8_t* wp = su.words;
    auto refill = [&](uint32_t x) -> uint32_t { if (x < kL) { x = (x << 16) | ld_u16(wp); wp += 2; } return x; };
    for (uint32_t s = 0; s < su.steps; ++s) {
        for (int l = 0; l < L; ++l) {   // A
            if (s >= su.cnt[l]) continue;
            uint32_t pc = cfg.pcls(posb_raw(PP[l]));
            uint32_t e = gt[KB[l] + (pc << pb) + (X[l] & mask)];
            uint32_t x = ((e >> 12) & 0xFFF) * (X[l] >> pb) + (e & 0xFFF);
            X[l] = refill(x);
            E[l] = e;
        }
        for (int l = 0; l < L; ++l) {   // B
            if (s >= su.cnt[l]) continue;
            uint32_t nb = cl.nb[(E[l] >> 24) >> 1];
            uint32_t lo = nb < 16 ? nb : 16;
            NB[l] = nb;
            EX[l] = X[l] & ((1u << lo) - 1);
            X[l] = refill(X[l] >> lo);
        }
        for (int l = 0; l < L; ++l) {   // B2
            if (s >= su.cnt[l] || NB[l] <= 16) continue;
            uint32_t hi = NB[l] - 16;
            EX[l] |= (X[l] & ((1u << hi) - 1)) << 16;
            X[l] = refill(X[l] >> hi);
        }
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l]) continue;
            RK[l] = PP[l] + cl.base[(E[l] >> 24) >> 1] + EX[l];
        }
        for (int l = 0; l < L; ++l) {   // C1
            if (s >= su.cnt[l] || !((E[l] >> 24) & 1)) continue;
            uint32_t pcr = posb_raw(RK[l]) >> 1;
            if (pcr > uint32_t(kResPos - 1)) pcr = kResPos - 1;
            uint32_t cx = (pcr * 4 + DK[l]) << pbv;
            uint32_t e = rt[cx + (X[l] & maskv)];
            uint32_t x = ((e >> 12) & 0xFFF) * (X[l] >> pbv) + (e & 0xFFF);
            X[l] = refill(x);
            E2[l] = e;
        }
        for (int l = 0; l < L; ++l) {   // C2
            if (s >= su.cnt[l] || !((E[l] >> 24) & 1)) continue;
            uint32_t nb = cl.nb[E2[l] >> 24];
            uint32_t lo = nb < 16 ? nb : 16;
            NB2[l] = nb;
            EX2[l] = X[l] & ((1u << lo) - 1);
            X[l] = refill(X[l] >> lo);
        }
        for (int l = 0; l < L; ++l) {   // C3
            if (s >= su.cnt[l] || !((E[l] >> 24) & 1) || NB2[l] <= 16) continue;
            uint32_t hi = NB2[l] - 16;
            EX2[l] |= (X[l] & ((1u << hi) - 1)) << 16;
            X[l] = refill(X[l] >> hi);
        }
        for (int l = 0; l < L; ++l) {
            if (s >= su.cnt[l]) continue;
            uint32_t v = 1;
            if ((E[l] >> 24) & 1) { uint32_t rr = cl.base[E2[l] >> 24] + EX2[l]; v = rr ? rr + 1 : 0; }
            idx_out[OUT[l]] = int32_t(RK[l]);
            val_out[OUT[l]] = v;
            ++OUT[l];
            PP[l] = RK[l] + 1;
            if (--REM[l] == 0 && s + 1 < su.cnt[l]) {
                uint32_t kk = *KP[l]++;
                REM[l] = kk;
                PP[l] = 0;
                KB[l] = cfg.kbase(kk);
                DK[l] = dk_of(uint32_t(ilog2f(kk + 1)));
            }
        }
        if (wp > su.words_end) return false;   // ran off the stream: stop before reading further
    }
    if (wp != su.words_end) return false;
    for (int l = 0; l < L; ++l) if (X[l] != kL) return false;
    return true;
}

}  // namespace r1
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
