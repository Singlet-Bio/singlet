// SPDX-License-Identifier: MIT
// w3_total: ONE codec that combines the separately measured pieces of the compression study.
//
//   w3_total_v1   = Track A container removal   (row names -> vocabulary id, barcodes -> whitelist index, no permutation block,
//                                                no ptr / colsums block; kernels/track_a.h enc_names_model / dec_names_model)
//                 + b_rans16 matrix payload     (kernels/track_b_*.h: 32-lane static rANS over rank-space gaps; its per-row
//                                                count side stream replaces the permutation)
//   w3_total_v2   = v1, but the empty droplets are a separate member coded with Idea 2's ambient multiset coder
//                   (kernels/idea2.h Engine::code_empties) conditioned on the decoded called cells; the called / empty split is a
//                   merge-context coded bitmap.  w3_total_v2r codes the empties member with b_rans16 instead.
//
// Everything is composed from the existing, read-only pieces; this header owns only the glue and the container.
// Row order: ORD_EXACT = exact descending row-nnz rank (b_rans16 native order, ties by row id);
//            ORD_CLASS = frequency class (tb::g_cls: counts < 8 exact, then 4 classes per octave), genomic order inside a class.
// The decoder can return the matrix in canonical (ascending row id, bit-identical to the input) or in native
// rank order (rows relabelled; rownames permuted to match), see dec_*(..., canonical).
//
// Stream byte accounting: the 64 zero read-padding bytes that tb::encode_payload appends are NOT part of the stream
// (the decoder re-adds them in its working copy).  Model files (vocabularies, whitelists) are counted separately.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include "track_a.h"
#include "track_b.h"
#include "idea2.h"
#include <algorithm>
#include <numeric>
#include <unordered_set>

namespace cs {
namespace w3 {

enum Order { ORD_EXACT = 0, ORD_CLASS = 1 };

inline std::string g_calls_path;   // cell_calls.tsv of the sample being encoded (v2 only; set by the driver)

// ---------------------------------------------------------------------------------------------------------
// Size / time breakdown of the last encode / decode on this thread
// ---------------------------------------------------------------------------------------------------------
struct Bd {
    size_t total = 0, hdr = 0, names = 0;
    size_t names_row = 0, names_col = 0, names_kv = 0;                           // split of `names`
    size_t cells_counts = 0, cells_payload = 0, cells_nnz = 0, cells_n = 0;     // v1: the single member is "cells"
    size_t cells_colcnt = 0, cells_tables = 0, cells_chunks = 0;                // split of cells_payload (b_rans16 payload layout)
    size_t empt_counts = 0, empt_payload = 0, empt_ambient = 0, empt_k = 0, empt_nnz = 0, empt_n = 0;
    size_t empt_colcnt = 0, empt_tables = 0, empt_chunks = 0;
    size_t bitmap = 0;
    int counts_tag = 0, empt_counts_tag = 0;
    double enc_ms_names = 0, enc_ms_cells = 0, enc_ms_empt = 0;
    void clear() { *this = Bd(); }
};
inline Bd& last_bd() { static thread_local Bd b; return b; }
struct Dt {   // decode phase times (ms), last decode on this thread
    double copy = 0, names = 0, counts = 0, payload = 0, empt = 0, split = 0, merge = 0, perm_names = 0;
    void clear() { *this = Dt(); }
};
inline Dt& last_dt() { static thread_local Dt d; return d; }

// ---------------------------------------------------------------------------------------------------------
// Count / class side stream: per-row values (nnz or class id) in row order, cheapest of
//   tags 1..6  : LEB128 (plain | delta-zigzag) + {zstd-19 | xz-9e | brotli-11}     (Track A's recipe for the permutation replacement)
//   tag  8     : adaptive Elias-gamma range code, context = log of the previous value (Idea 2's z_f coder)
// layout: u8 tag, u32 raw_len (tags 1-6; 0 for tag 8), u32 len, bytes
// ---------------------------------------------------------------------------------------------------------
inline int vctx(uint32_t prev) { return prev ? std::min(1 + i2::ilog2u(prev), 17) : 0; }

inline std::vector<uint8_t> enc_vals(const std::vector<uint32_t>& v, int* tag_out = nullptr) {
    std::vector<uint8_t> best;
    int best_tag = -1;
    auto consider = [&](int tag, uint32_t raw_len, const std::vector<uint8_t>& z) {
        if (best_tag < 0 || z.size() + 9 < best.size()) {
            best.clear();
            ta::ByteW w(best);
            w.u8(uint8_t(tag)); w.u32(raw_len); w.u32(uint32_t(z.size())); w.bytes(z.data(), z.size());
            best_tag = tag;
        }
    };
    for (int delta = 0; delta < 2; ++delta) {
        std::vector<uint8_t> raw;
        int64_t prev = 0;
        for (uint32_t x : v) {
            if (!delta) tb::leb_put(raw, x);
            else { int64_t d = int64_t(x) - prev; prev = x; tb::leb_put(raw, uint64_t((d << 1) ^ (d >> 63))); }
        }
        for (int t = 0; t < 3; ++t) {
            std::vector<uint8_t> z;
            ta::ec_compress(ta::coder_for_tag(uint8_t(t)), raw.data(), raw.size(), z);
            consider(1 + delta * 3 + t, uint32_t(raw.size()), z);
        }
    }
    {
        std::vector<uint8_t> o;
        i2::RcEnc rc(o);
        i2::NumCoder nc(18);
        uint32_t prev = 0;
        for (uint32_t x : v) { nc.code(rc, vctx(prev), x); prev = x; }
        rc.finish();
        consider(8, 0, o);
    }
    if (tag_out) *tag_out = best_tag;
    return best;
}
inline std::vector<uint32_t> dec_vals(ta::ByteR& r, uint32_t count, int* tag_out = nullptr) {
    int tag = r.u8();
    uint32_t raw_len = r.u32(), len = r.u32();
    const uint8_t* p = r.take(len);
    if (tag_out) *tag_out = tag;
    std::vector<uint32_t> v(count);
    if (tag == 8) {
        i2::RcDec rc(p, p + len);
        i2::NumCoder nc(18);
        uint32_t prev = 0;
        for (uint32_t i = 0; i < count; ++i) { v[i] = uint32_t(nc.code(rc, vctx(prev), 0)); prev = v[i]; }
        return v;
    }
    if (tag < 1 || tag > 6) throw std::runtime_error("w3: bad count tag");
    int delta = (tag - 1) / 3, t = (tag - 1) % 3;
    std::vector<uint8_t> raw(raw_len);
    ta::ec_decompress(ta::coder_for_tag(uint8_t(t)), p, len, raw.data(), raw_len);
    const uint8_t* q = raw.data();
    int64_t prev = 0;
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t x = tb::leb_get(q);
        if (delta) { int64_t d = int64_t(x >> 1) ^ -int64_t(x & 1); prev += d; v[i] = uint32_t(prev); }
        else v[i] = uint32_t(x);
    }
    return v;
}

// ---------------------------------------------------------------------------------------------------------
// b_rans16 member: a CSC matrix over the shared m rows.   [u32 n][u64 nnz] ([u32 chunk_cols][u8 order][counts section]
// [u64 payload_len][payload]) -- the bracketed part is absent when n == 0 or nnz == 0.
// ---------------------------------------------------------------------------------------------------------
inline std::vector<uint32_t> row_keys(const CanonicalMatrix& S, int order) {
    std::vector<uint32_t> c(S.m, 0);
    for (auto r : S.indices) c[r]++;
    if (order == ORD_CLASS)
        for (auto& x : c) { uint32_t ex, nb; x = tb::g_cls(x, ex, nb); }
    return c;
}

struct PaySplit { size_t colcnt = 0, tables = 0, chunks = 0; };   // b_rans16 payload: cfg+column-count stream | tables | chunk table + blobs
inline std::vector<uint8_t> enc_member(const CanonicalMatrix& S, int order, size_t* counts_bytes = nullptr,
                                       size_t* payload_bytes = nullptr, int* tag = nullptr, PaySplit* ps = nullptr) {
    std::vector<uint8_t> o;
    ta::ByteW w(o);
    w.u32(S.n); w.u64(S.nnz);
    if (counts_bytes) *counts_bytes = 0;
    if (payload_bytes) *payload_bytes = 0;
    if (S.n == 0 || S.nnz == 0) return o;
    const uint32_t cc = static_cast<uint32_t>(tb::auto_chunk_cols(S, 96 * 1024));
    w.u32(cc);
    w.u8(uint8_t(order));
    tb::RowPerm P;
    P.rnnz = row_keys(S, order);
    size_t t0 = o.size();
    {
        auto cs = enc_vals(P.rnnz, tag);
        w.bytes(cs.data(), cs.size());
    }
    if (counts_bytes) *counts_bytes = o.size() - t0;
    tb::perm_from_counts(P);                       // P.perm: rank -> row, P.inv: row -> rank (descending key, ties by row id)
    tb::RankMat R = tb::to_rank_space(S, P);
    int bestk = 0;
    size_t bs = ~size_t(0);
    for (int kind = 0; kind < 5; ++kind) {
        size_t e = tb::estimate_payload(R, tb::Cfg::make(kind, 10));
        if (e < bs) { bs = e; bestk = kind; }
    }
    tb::EncodedPayload ep = tb::encode_payload(R, tb::Cfg::make(bestk, 10), static_cast<int>(cc));
    ep.bytes.resize(ep.bytes.size() - 64);        // drop the read padding, the decoder re-adds it
    w.u64(ep.bytes.size());
    w.bytes(ep.bytes.data(), ep.bytes.size());
    if (payload_bytes) *payload_bytes = ep.bytes.size();
    if (ps) { ps->colcnt = ep.counts + 7; ps->tables = ep.tables; ps->chunks = ep.chunks; if (ps->colcnt + ps->tables + ps->chunks != ep.bytes.size()) throw std::runtime_error("w3: payload split mismatch"); }   // +7 = Cfg header
    return o;
}

struct MemberView {
    uint32_t n = 0, m = 0, chunk_cols = 0;
    uint64_t nnz = 0;
    uint8_t vt = 0, order = 0;
    tb::RowPerm P;                   // rank <-> row
    const uint8_t* payload = nullptr;
    size_t payload_len = 0;
};
inline MemberView parse_member(ta::ByteR& r, uint32_t m, uint8_t vt, int* tag = nullptr) {
    MemberView V;
    V.m = m; V.vt = vt;
    V.n = r.u32(); V.nnz = r.u64();
    if (V.n == 0 || V.nnz == 0) return V;
    V.chunk_cols = r.u32();
    V.order = r.u8();
    V.P.rnnz = dec_vals(r, m, tag);
    tb::perm_from_counts(V.P);
    V.payload_len = size_t(r.u64());
    V.payload = r.take(V.payload_len);
    return V;
}
inline void member_payload_decode(const MemberView& V, bool canonical, CanonicalMatrix& S, tb::DecState& D) {
    S.m = V.m; S.n = V.n; S.nnz = V.nnz; S.vt_code = V.vt;
    if (V.n == 0 || V.nnz == 0) {
        S.indptr.assign(size_t(V.n) + 1, 0);
        S.indices.clear(); S.data.clear();
        return;
    }
    tb::RansFile F;
    F.m = V.m; F.n = V.n; F.nnz = V.nnz; F.vt = V.vt; F.chunk_cols = V.chunk_cols;
    F.names_p = nullptr; F.rowc_p = nullptr; F.payload_p = V.payload;
    tb::rans_payload_decode(F, V.P, canonical, S, D);
}

// ---------------------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------------------
inline void put_container_hdr(std::vector<uint8_t>& o, const char* magic, uint8_t flags, const CanonicalMatrix& M) {
    ta::ByteW w(o);
    w.bytes(magic, 4); w.u8(flags); w.u32(M.m); w.u8(M.vt_code);
}

inline uint64_t colsum(const CanonicalMatrix& M, uint32_t j) {
    uint64_t s = 0;
    for (uint32_t q = M.indptr[j]; q < M.indptr[j + 1]; ++q) s += M.data[q];
    return s;
}

// split columns by flag (1 = first matrix)
inline void split_cols(const CanonicalMatrix& M, const std::vector<uint8_t>& flag, CanonicalMatrix& A, CanonicalMatrix& B) {
    CanonicalMatrix* T[2] = {&B, &A};
    for (auto* t : T) { *t = CanonicalMatrix(); t->m = M.m; t->vt_code = M.vt_code; t->indptr.assign(1, 0); }
    for (uint32_t j = 0; j < M.n; ++j) {
        CanonicalMatrix& t = *T[flag[j] ? 1 : 0];
        for (uint32_t q = M.indptr[j]; q < M.indptr[j + 1]; ++q) { t.indices.push_back(M.indices[q]); t.data.push_back(M.data[q]); }
        t.indptr.push_back(uint32_t(t.indices.size()));
        t.n++;
    }
    A.nnz = A.indices.size(); B.nnz = B.indices.size();
}

// called set from cell_calls.tsv (first column = barcode), mapped to column flags
inline std::vector<uint8_t> load_calls(const CanonicalMatrix& M, const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("w3: cannot open cell_calls.tsv at '" + path + "'");
    std::unordered_set<std::string> s;
    std::string line;
    std::getline(f, line);
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto t = line.find('\t');
        s.insert(line.substr(0, t));
    }
    std::vector<uint8_t> flag(M.n, 0);
    for (uint32_t j = 0; j < M.n; ++j) flag[j] = s.count(M.colnames[j]) ? 1 : 0;
    return flag;
}

// ---------------------------------------------------------------------------------------------------------
// called / empty split bitmap, merge-context coded.  The decoder already holds both columns lists (cells and empties,
// each in column order); at every position it knows the depth of the next unplaced cell and of the next unplaced empty and
// codes "which comes first" with the depth difference (quarter octaves) and the previous flag as context.
// ---------------------------------------------------------------------------------------------------------
inline int qbk(uint32_t k) {
    if (k < 4) return int(k);
    int l = 31 - __builtin_clz(k);
    return 4 * l + int((k >> (l - 2)) & 3) - 4;
}
template <class RC>
inline void code_split(RC& rc, std::vector<uint8_t>& flag, uint32_t n, const std::vector<uint32_t>& kC, const std::vector<uint32_t>& kE) {
    const uint32_t nc = uint32_t(kC.size()), ne = uint32_t(kE.size());
    std::vector<i2::BitM> bm(33 * 2);
    uint32_t ci = 0, ei = 0;
    int prev = 1;
    if (!RC::ENC) flag.assign(n, 0);
    for (uint32_t j = 0; j < n; ++j) {
        int f;
        if (ci == nc) f = 0;
        else if (ei == ne) f = 1;
        else {
            int d = qbk(kC[ci]) - qbk(kE[ei]);
            d = std::max(-16, std::min(16, d)) + 16;
            f = i2::code_bitm(rc, bm[size_t(d) * 2 + size_t(prev)], RC::ENC ? flag[j] : 0, 5);
        }
        flag[j] = uint8_t(f);
        if (f) ++ci; else ++ei;
        prev = f;
    }
}
inline std::vector<uint8_t> enc_split(const std::vector<uint8_t>& flag, const CanonicalMatrix& C, const CanonicalMatrix& E) {
    std::vector<uint32_t> kC(C.n), kE(E.n);
    for (uint32_t j = 0; j < C.n; ++j) kC[j] = uint32_t(colsum(C, j));
    for (uint32_t j = 0; j < E.n; ++j) kE[j] = uint32_t(colsum(E, j));
    std::vector<uint8_t> o;
    i2::RcEnc rc(o);
    std::vector<uint8_t> fl = flag;
    code_split(rc, fl, uint32_t(flag.size()), kC, kE);
    rc.finish();
    return o;
}

// ---------------------------------------------------------------------------------------------------------
// called-set stream for a single matrix (v1 bundle accounting): flag per column, context = quarter-octave depth bucket of the
// column's UMI total, exactly Idea 2's flag coder.  Decoder needs the matrix columns first.
// ---------------------------------------------------------------------------------------------------------
template <class RC>
inline void code_calls(RC& rc, std::vector<uint8_t>& flag, const std::vector<uint32_t>& k) {
    std::vector<i2::BitM> bm(256);
    if (!RC::ENC) flag.assign(k.size(), 0);
    for (size_t j = 0; j < k.size(); ++j) {
        int qb = std::min(qbk(k[j]), 255);
        flag[j] = uint8_t(i2::code_bitm(rc, bm[size_t(qb)], RC::ENC ? flag[j] : 0, 5));
    }
}
inline std::vector<uint8_t> enc_calls(const CanonicalMatrix& M, const std::vector<uint8_t>& flag) {
    std::vector<uint32_t> k(M.n);
    for (uint32_t j = 0; j < M.n; ++j) k[j] = uint32_t(colsum(M, j));
    std::vector<uint8_t> o;
    i2::RcEnc rc(o);
    std::vector<uint8_t> fl = flag;
    code_calls(rc, fl, k);
    rc.finish();
    return o;
}
inline std::vector<uint8_t> dec_calls(const CanonicalMatrix& M, const std::vector<uint8_t>& b) {
    std::vector<uint32_t> k(M.n);
    for (uint32_t j = 0; j < M.n; ++j) k[j] = uint32_t(colsum(M, j));
    i2::RcDec rc(b.data(), b.data() + b.size());
    std::vector<uint8_t> fl;
    code_calls(rc, fl, k);
    return fl;
}

// ---------------------------------------------------------------------------------------------------------
// ambient multiset coder for the empty droplets (Idea 2, Engine::code_empties), conditioned on the decoded cells.
// Virtual column ids: cells 0..nc-1, empties nc..nc+ne-1.  Stream = range code of [k per empty droplet (adaptive
// Elias-gamma, one context)] followed by the droplets as exact multinomial multisets under the ambient profile
// (sum of the cells' counts + running sum of decoded empties).  No model file.
// ---------------------------------------------------------------------------------------------------------
inline std::vector<uint8_t> enc_ambient(const CanonicalMatrix& C, const CanonicalMatrix& E, size_t* k_bytes = nullptr) {
    const uint32_t nc = C.n, ne = E.n;
    CanonicalMatrix V;
    V.m = C.m; V.n = nc + ne; V.nnz = C.nnz + E.nnz; V.vt_code = C.vt_code;
    V.indptr = C.indptr;
    for (uint32_t j = 1; j <= ne; ++j) V.indptr.push_back(uint32_t(C.nnz) + E.indptr[j]);
    V.indices = C.indices; V.indices.insert(V.indices.end(), E.indices.begin(), E.indices.end());
    V.data = C.data; V.data.insert(V.data.end(), E.data.begin(), E.data.end());
    i2::Engine<i2::RcEnc> eng;
    eng.P = i2::Params();                     // idea2_static: mask 0, K 1, pooled ambient, adaptive to decoded empties
    eng.m = V.m; eng.n = V.n; eng.M = &V;
    eng.kcol.assign(V.n, 0); eng.labc.assign(V.n, 0);
    for (uint32_t e = 0; e < ne; ++e) eng.kcol[nc + e] = uint32_t(colsum(V, nc + e));
    std::vector<uint32_t> cells(nc), empt(ne);
    std::iota(cells.begin(), cells.end(), 0u);
    std::iota(empt.begin(), empt.end(), nc);
    std::vector<uint8_t> o;
    i2::RcEnc rc(o);
    i2::NumCoder nk(1);
    rc.cat = i2::C_HK;
    for (uint32_t e = 0; e < ne; ++e) nk.code(rc, 0, eng.kcol[nc + e]);
    if (k_bytes) *k_bytes = size_t(rc.cost[i2::C_HK] / 8.0);
    std::vector<uint32_t> si, sv, sl;
    std::vector<uint64_t> so;
    eng.code_empties(rc, cells, empt, 1, si, sv, so, sl);
    rc.finish();
    return o;
}
inline void dec_ambient(const uint8_t* p, size_t len, const CanonicalMatrix& C, uint32_t ne, uint64_t nnz_e, CanonicalMatrix& E) {
    const uint32_t nc = C.n, nv = nc + ne;
    i2::Engine<i2::RcDec> eng;
    eng.P = i2::Params();
    eng.m = C.m; eng.n = nv; eng.M = nullptr;
    eng.kcol.assign(nv, 0); eng.labc.assign(nv, 0);
    i2::RcDec rc(p, p + len);
    i2::NumCoder nk(1);
    for (uint32_t e = 0; e < ne; ++e) eng.kcol[nc + e] = uint32_t(nk.code(rc, 0, 0));
    std::vector<uint32_t> si, sv, sl(nv, 0);
    std::vector<uint64_t> so(nv, 0);
    si.reserve(size_t(C.nnz + nnz_e + 16)); sv.reserve(size_t(C.nnz + nnz_e + 16));
    si.assign(C.indices.begin(), C.indices.end());
    sv.assign(C.data.begin(), C.data.end());
    for (uint32_t j = 0; j < nc; ++j) { so[j] = C.indptr[j]; sl[j] = C.indptr[j + 1] - C.indptr[j]; }
    std::vector<uint32_t> cells(nc), empt(ne);
    std::iota(cells.begin(), cells.end(), 0u);
    std::iota(empt.begin(), empt.end(), nc);
    eng.code_empties(rc, cells, empt, 1, si, sv, so, sl);
    E = CanonicalMatrix();
    E.m = C.m; E.n = ne; E.vt_code = C.vt_code;
    E.indptr.assign(size_t(ne) + 1, 0);
    for (uint32_t e = 0; e < ne; ++e) E.indptr[e + 1] = E.indptr[e] + sl[nc + e];
    E.nnz = E.indptr[ne];
    E.indices.resize(E.nnz); E.data.resize(E.nnz);
    for (uint32_t e = 0; e < ne; ++e) {
        std::memcpy(&E.indices[E.indptr[e]], &si[so[nc + e]], size_t(sl[nc + e]) * 4);
        std::memcpy(&E.data[E.indptr[e]], &sv[so[nc + e]], size_t(sl[nc + e]) * 4);
    }
}

// ---------------------------------------------------------------------------------------------------------
// containers
// ---------------------------------------------------------------------------------------------------------
inline std::vector<uint8_t> enc_v1(const CanonicalMatrix& M, int order) {
    Timer tt;
    auto& bd = last_bd();
    bd.clear();
    std::vector<uint8_t> o;
    put_container_hdr(o, "W3T1", uint8_t(order), M);
    bd.hdr = o.size();
    size_t t0 = o.size();
    {
        auto ns = ta::enc_names_model(M, o);
        bd.names_row = ns.row_bytes; bd.names_col = ns.col_bytes; bd.names_kv = ns.kv_bytes;
    }
    bd.names = o.size() - t0;
    t0 = o.size();
    PaySplit ps;
    auto mem = enc_member(M, order, &bd.cells_counts, &bd.cells_payload, &bd.counts_tag, &ps);
    bd.cells_colcnt = ps.colcnt; bd.cells_tables = ps.tables; bd.cells_chunks = ps.chunks;
    o.insert(o.end(), mem.begin(), mem.end());
    bd.hdr += 12;                                    // member n + nnz
    bd.cells_nnz = M.nnz; bd.cells_n = M.n;
    bd.total = o.size();
    return o;
}

// canonical = true: ascending row ids, bit-identical to the input.  false: native rank order (indices are ranks, rownames permuted).
inline CanonicalMatrix dec_v1(const std::vector<uint8_t>& b, bool canonical = true) {
    Timer t;
    auto& dt = last_dt();
    dt.clear();
    std::vector<uint8_t> buf(b.size() + 64, 0);
    std::memcpy(buf.data(), b.data(), b.size());
    dt.copy = t.ms();
    ta::ByteR r(buf.data(), b.size());
    if (std::memcmp(r.take(4), "W3T1", 4) != 0) throw std::runtime_error("w3: bad magic v1");
    r.u8();
    CanonicalMatrix M;
    M.m = r.u32(); M.vt_code = r.u8();
    t.t0 = std::chrono::steady_clock::now();
    ta::dec_names_model(r, M);
    dt.names = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    MemberView V = parse_member(r, M.m, M.vt_code);
    dt.counts = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    tb::DecState D;
    member_payload_decode(V, canonical, M, D);
    dt.payload = t.ms();
    if (!canonical && V.nnz && V.n) {
        t.t0 = std::chrono::steady_clock::now();
        std::vector<std::string> nr(M.m);
        for (uint32_t i = 0; i < M.m; ++i) nr[i] = std::move(M.rownames[V.P.perm[i]]);
        M.rownames = std::move(nr);
        dt.perm_names = t.ms();
    }
    return M;
}

inline std::vector<uint8_t> enc_v2(const CanonicalMatrix& M, int order, int empt_coder /*0 ambient, 1 rans16*/) {
    auto& bd = last_bd();
    bd.clear();
    std::vector<uint8_t> flag = load_calls(M, g_calls_path);
    CanonicalMatrix C, E;
    split_cols(M, flag, C, E);
    std::vector<uint8_t> o;
    put_container_hdr(o, "W3T2", uint8_t(order | (empt_coder << 1)), M);
    bd.hdr = o.size();
    size_t t0 = o.size();
    {
        auto ns = ta::enc_names_model(M, o);
        bd.names_row = ns.row_bytes; bd.names_col = ns.col_bytes; bd.names_kv = ns.kv_bytes;
    }
    bd.names = o.size() - t0;
    {   // cells member
        PaySplit ps;
        auto mem = enc_member(C, order, &bd.cells_counts, &bd.cells_payload, &bd.counts_tag, &ps);
        bd.cells_colcnt = ps.colcnt; bd.cells_tables = ps.tables; bd.cells_chunks = ps.chunks;
        o.insert(o.end(), mem.begin(), mem.end());
        bd.hdr += 12;
        bd.cells_nnz = C.nnz; bd.cells_n = C.n;
    }
    bd.empt_nnz = E.nnz; bd.empt_n = E.n;
    if (empt_coder == 0) {
        ta::ByteW w(o);
        w.u32(E.n); w.u64(E.nnz);
        auto amb = enc_ambient(C, E, &bd.empt_k);
        w.u64(amb.size());
        w.bytes(amb.data(), amb.size());
        bd.hdr += 12 + 8;
        bd.empt_ambient = amb.size();
    } else {
        PaySplit ps;
        auto mem = enc_member(E, order, &bd.empt_counts, &bd.empt_payload, &bd.empt_counts_tag, &ps);
        bd.empt_colcnt = ps.colcnt; bd.empt_tables = ps.tables; bd.empt_chunks = ps.chunks;
        o.insert(o.end(), mem.begin(), mem.end());
        bd.hdr += 12;
    }
    {
        auto bm = enc_split(flag, C, E);
        ta::ByteW w(o);
        w.u32(uint32_t(bm.size()));
        w.bytes(bm.data(), bm.size());
        bd.bitmap = bm.size();
        bd.hdr += 4;
    }
    bd.total = o.size();
    return o;
}

inline CanonicalMatrix dec_v2(const std::vector<uint8_t>& b, bool canonical = true) {
    Timer t;
    auto& dt = last_dt();
    dt.clear();
    std::vector<uint8_t> buf(b.size() + 64, 0);
    std::memcpy(buf.data(), b.data(), b.size());
    dt.copy = t.ms();
    ta::ByteR r(buf.data(), b.size());
    if (std::memcmp(r.take(4), "W3T2", 4) != 0) throw std::runtime_error("w3: bad magic v2");
    uint8_t flags = r.u8();
    int order = flags & 1, empt_coder = (flags >> 1) & 1;
    (void)order;
    CanonicalMatrix M;
    M.m = r.u32(); M.vt_code = r.u8();
    t.t0 = std::chrono::steady_clock::now();
    ta::dec_names_model(r, M);
    dt.names = t.ms();
    // cells (always canonical: the empties and the merge need original row ids)
    t.t0 = std::chrono::steady_clock::now();
    MemberView VC = parse_member(r, M.m, M.vt_code);
    dt.counts = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    CanonicalMatrix C, E;
    tb::DecState D;
    member_payload_decode(VC, true, C, D);
    dt.payload = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    if (empt_coder == 0) {
        uint32_t ne = r.u32();
        uint64_t nnz_e = r.u64();
        uint64_t len = r.u64();
        const uint8_t* p = r.take(size_t(len));
        dec_ambient(p, size_t(len), C, ne, nnz_e, E);
    } else {
        MemberView VE = parse_member(r, M.m, M.vt_code);
        tb::DecState D2;
        member_payload_decode(VE, true, E, D2);
    }
    dt.empt = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    std::vector<uint32_t> kC(C.n), kE(E.n);
    for (uint32_t j = 0; j < C.n; ++j) kC[j] = uint32_t(colsum(C, j));
    for (uint32_t j = 0; j < E.n; ++j) kE[j] = uint32_t(colsum(E, j));
    uint32_t bml = r.u32();
    const uint8_t* bp = r.take(bml);
    i2::RcDec rc(bp, bp + bml);
    std::vector<uint8_t> flag;
    M.n = C.n + E.n;
    code_split(rc, flag, M.n, kC, kE);
    dt.split = t.ms();
    t.t0 = std::chrono::steady_clock::now();
    M.nnz = C.nnz + E.nnz;
    M.indptr.assign(size_t(M.n) + 1, 0);
    M.indices.resize(M.nnz); M.data.resize(M.nnz);
    uint32_t ci = 0, ei = 0;
    size_t pos = 0;
    for (uint32_t j = 0; j < M.n; ++j) {
        const CanonicalMatrix& S = flag[j] ? C : E;
        uint32_t& idx = flag[j] ? ci : ei;
        uint32_t a = S.indptr[idx], len = S.indptr[idx + 1] - a;
        if (len) {
            std::memcpy(&M.indices[pos], &S.indices[a], size_t(len) * 4);
            std::memcpy(&M.data[pos], &S.data[a], size_t(len) * 4);
        }
        pos += len;
        M.indptr[j + 1] = uint32_t(pos);
        ++idx;
    }
    dt.merge = t.ms();
    (void)canonical;   // v2 always returns canonical row ids (the native layout is a v1 / per-member property)
    return M;
}

// ---------------------------------------------------------------------------------------------------------
// native-order check: map a native-decoded matrix back to canonical and compare with the original
// ---------------------------------------------------------------------------------------------------------
inline bool native_matches(const CanonicalMatrix& N, const CanonicalMatrix& M, std::string* why = nullptr) {
    auto fail = [&](const char* s) { if (why) *why = s; return false; };
    if (N.m != M.m || N.n != M.n || N.nnz != M.nnz || N.indptr != M.indptr || N.colnames != M.colnames || N.user_kv != M.user_kv)
        return fail("header/indptr/colnames");
    // N.rownames[r] must be the name of the row that rank r stands for; recover perm from names (unique within a build)
    std::unordered_map<std::string, uint32_t> row_of;
    row_of.reserve(M.m * 2);
    for (uint32_t i = 0; i < M.m; ++i) row_of.emplace(M.rownames[i], i);
    std::vector<uint32_t> perm(M.m);
    for (uint32_t r = 0; r < M.m; ++r) {
        auto it = row_of.find(N.rownames[r]);
        if (it == row_of.end()) return fail("rowname not in vocabulary");
        perm[r] = it->second;
    }
    std::vector<std::pair<uint32_t, uint32_t>> col;
    for (uint32_t j = 0; j < M.n; ++j) {
        col.clear();
        for (uint32_t q = N.indptr[j]; q < N.indptr[j + 1]; ++q) col.emplace_back(perm[N.indices[q]], N.data[q]);
        std::sort(col.begin(), col.end());
        for (uint32_t t = 0; t < col.size(); ++t) {
            uint32_t q = M.indptr[j] + t;
            if (col[t].first != M.indices[q] || col[t].second != M.data[q]) return fail("entries");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------------------------
inline Codec make_codec(const std::string& name, int version, int order, int empt_coder, size_t model_bytes) {
    Codec c;
    c.name = name;
    c.model_bytes = model_bytes;
    if (version == 1) {
        c.encode = [order](const CanonicalMatrix& M) { return enc_v1(M, order); };
        c.decode = [](const std::vector<uint8_t>& b) { return dec_v1(b, true); };
    } else {
        c.encode = [order, empt_coder](const CanonicalMatrix& M) { return enc_v2(M, order, empt_coder); };
        c.decode = [](const std::vector<uint8_t>& b) { return dec_v2(b, true); };
    }
    return c;
}

inline void register_w3(std::vector<Codec>& v) {
    size_t mb = ta::Models::get().model_bytes_total();
    v.push_back(make_codec("w3_total_v1", 1, ORD_EXACT, 0, mb));
    v.push_back(make_codec("w3_total_v1c", 1, ORD_CLASS, 0, mb));
    v.push_back(make_codec("w3_total_v2", 2, ORD_EXACT, 0, mb));
    v.push_back(make_codec("w3_total_v2r", 2, ORD_EXACT, 1, mb));
    v.push_back(make_codec("w3_total_v2c", 2, ORD_CLASS, 0, mb));
}

}  // namespace w3
}  // namespace cs
