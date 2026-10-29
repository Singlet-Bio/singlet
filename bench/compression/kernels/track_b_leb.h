// Track B: "plain" codecs without entropy coding of the symbols: rank-space gap + LEB128 varint streams per chunk,
// optionally followed by zstd (any level) or nothing.  These are the yardsticks for the rANS lane codec:
//   b_leb_raw   gap LEB128 + value LEB128, stored as is
//   b_leb_z3    same bytes, one zstd-3 frame per 1024-column chunk
//   b_leb_z19   same bytes, zstd-19
// Chunk bytes = [LEB k per column][LEB gap per nnz][LEB (v-1) per nnz]; row order = frequency rank (side info as in the
// lane codec), canonicalised on decode with the same sorter.
#pragma once
#include "track_b_codecs.h"

namespace cs {
namespace tb {

struct LebOpts {
    int level = 3;  // 0 = raw
    int chunk_cols = 1024;
    bool simd = true;
};

inline std::vector<uint8_t> leb_encode(const CanonicalMatrix& M, const LebOpts& op) {
    auto& bd = last_breakdown();
    bd.clear();
    std::vector<uint8_t> o;
    put_header(o, M);
    detail::put_u32(o, static_cast<uint32_t>(op.chunk_cols));
    o.push_back(static_cast<uint8_t>(op.level));
    size_t t0 = o.size();
    put_names(o, M);
    bd.names = o.size() - t0;
    RowPerm P = make_perm(M);
    t0 = o.size();
    put_rowcounts(o, P);
    bd.side = o.size() - t0;
    RankMat R = to_rank_space(M, P);
    t0 = o.size();
    const uint32_t nchunks = (M.n + op.chunk_cols - 1) / op.chunk_cols;
    std::vector<std::vector<uint8_t>> blobs(nchunks);
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * op.chunk_cols, c1 = std::min(M.n, c0 + op.chunk_cols);
        std::vector<uint8_t> raw;
        for (uint32_t j = c0; j < c1; ++j) leb_put(raw, R.ptr[j + 1] - R.ptr[j]);
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t pp = 0;
            for (uint32_t t = R.ptr[j]; t < R.ptr[j + 1]; ++t) { leb_put(raw, R.rk[t] - pp); pp = R.rk[t] + 1; }
        }
        for (uint32_t t = R.ptr[c0]; t < R.ptr[c1]; ++t) leb_put(raw, R.val[t] - 1);
        auto& b = blobs[c];
        detail::put_u32(b, static_cast<uint32_t>(raw.size()));
        if (op.level == 0) b.insert(b.end(), raw.begin(), raw.end());
        else {
            auto z = detail::zstd_c(raw.data(), raw.size(), op.level);
            b.insert(b.end(), z.begin(), z.end());
        }
    }
    for (auto& b : blobs) leb_put(o, b.size());
    for (auto& b : blobs) o.insert(o.end(), b.begin(), b.end());
    bd.payload = o.size() - t0;
    bd.chunks = bd.payload;
    return o;
}

struct LebFile {
    uint32_t m, n;
    uint64_t nnz;
    uint8_t vt;
    uint32_t chunk_cols;
    int level;
    const uint8_t* names_p;
    const uint8_t* rowc_p;
    const uint8_t* payload_p;
};
inline LebFile leb_parse(const std::vector<uint8_t>& b) {
    const uint8_t* p = b.data();
    LebFile F;
    F.m = detail::get_u32(p); F.n = detail::get_u32(p); F.nnz = detail::get_u64(p); F.vt = *p++;
    F.chunk_cols = detail::get_u32(p);
    F.level = *p++;
    F.names_p = p;
    skip_stream(p); skip_stream(p);
    detail::get_u32(p);
    skip_stream(p);
    F.rowc_p = p;
    skip_stream(p);
    F.payload_p = p;
    return F;
}

inline void leb_payload_decode(const LebFile& F, const RowPerm& P, bool canonical, CanonicalMatrix& M, size_t) {
    const uint32_t nchunks = (F.n + F.chunk_cols - 1) / F.chunk_cols;
    const uint8_t* p = F.payload_p;
    std::vector<size_t> sz(nchunks);
    for (auto& s : sz) s = leb_get(p);
    M.indptr.assign(static_cast<size_t>(F.n) + 1, 0);
    M.indices.resize(F.nnz);
    M.data.resize(F.nnz);
    std::vector<uint8_t> raw;
    std::unique_ptr<Canon> cn;
#ifdef TB_HAVE_AVX512
    std::unique_ptr<CanonVec> cv;
    const bool vec = canonical && use_simd();
    if (vec) cv.reset(new CanonVec(P.perm.data(), F.m));
    else
#endif
    if (canonical) cn.reset(new Canon(P.perm.data(), F.m));
    uint32_t pos = 0;
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * F.chunk_cols, c1 = std::min(F.n, c0 + F.chunk_cols);
        const uint8_t* q = p;
        uint32_t rawn = detail::get_u32(q);
        const uint8_t* src = q;
        if (F.level != 0) {
            raw.resize(rawn + 16);
            size_t z = ZSTD_decompress(raw.data(), rawn, q, sz[c] - 4);
            if (ZSTD_isError(z)) throw std::runtime_error("leb: zstd error");
            src = raw.data();
        }
        const uint8_t* kp = src;
        const uint32_t base = pos;
        for (uint32_t j = c0; j < c1; ++j) { uint32_t k = static_cast<uint32_t>(leb_get(kp)); M.indptr[j + 1] = M.indptr[j] + k; }
        const uint8_t* gp = kp;
        uint32_t* idx = M.indices.data();
        for (uint32_t j = c0; j < c1; ++j) {
            uint32_t pp = 0;
            for (uint32_t t = M.indptr[j]; t < M.indptr[j + 1]; ++t) {
                uint32_t b = *gp++, v = b & 127;
                if (b & 128) { b = *gp++; v |= (b & 127) << 7; if (b & 128) { b = *gp++; v |= (b & 127) << 14; if (b & 128) { b = *gp++; v |= (b & 127) << 21; } } }
                pp += v;
                idx[t] = pp;
                pp += 1;
            }
        }
        const uint8_t* vp = gp;
        uint32_t* dat = M.data.data();
        for (uint32_t t = M.indptr[c0]; t < M.indptr[c1]; ++t) {
            uint32_t b = *vp++, v = b & 127;
            if (b & 128) { b = *vp++; v |= (b & 127) << 7; if (b & 128) { b = *vp++; v |= (b & 127) << 14; if (b & 128) { b = *vp++; v |= (b & 127) << 21; } } }
            dat[t] = v + 1;
        }
        (void)base;
        pos = M.indptr[c1];
#ifdef TB_HAVE_AVX512
        if (vec) {
            for (uint32_t j = c0; j < c1; ++j) cv->column(idx + M.indptr[j], dat + M.indptr[j], M.indptr[j + 1] - M.indptr[j]);
        } else
#endif
        if (canonical)
            for (uint32_t j = c0; j < c1; ++j) cn->column(idx + M.indptr[j], dat + M.indptr[j], M.indptr[j + 1] - M.indptr[j]);
        p += sz[c];
    }
}

inline CanonicalMatrix leb_decode(const std::vector<uint8_t>& b, bool simd) {
    SimdScope sg(simd);
    LebFile F = leb_parse(b);
    CanonicalMatrix M;
    M.m = F.m; M.n = F.n; M.nnz = F.nnz; M.vt_code = F.vt;
    const uint8_t* p = F.names_p;
    get_names(p, M);
    p = F.rowc_p;
    RowPerm P;
    get_rowcounts(p, P, F.m);
    leb_payload_decode(F, P, true, M, 0);
    return M;
}

inline Codec make_b_leb(const std::string& name, LebOpts op) {
    Codec c;
    c.name = name;
    c.encode = [op](const CanonicalMatrix& M) { return leb_encode(M, op); };
    c.decode = [op](const std::vector<uint8_t>& b) { return leb_decode(b, op.simd); };
    payload_hooks()[name] = [op](const std::vector<uint8_t>& b, bool canonical) {
        struct St { LebFile F; RowPerm P; CanonicalMatrix M; };
        auto st = std::make_shared<St>();
        st->F = leb_parse(b);
        const uint8_t* p = st->F.rowc_p;
        get_rowcounts(p, st->P, st->F.m);
        return std::function<void()>([st, canonical, op]() {
            SimdScope sg(op.simd);
            leb_payload_decode(st->F, st->P, canonical, st->M, 0);
        });
    };
    return c;
}

inline void register_leb(std::vector<Codec>& v) {
    LebOpts o;
    o.level = 0;
    v.push_back(make_b_leb("b_leb_raw", o));
    o.level = 3;
    v.push_back(make_b_leb("b_leb_z3", o));
    o.level = 19;
    v.push_back(make_b_leb("b_leb_z19", o));
}

}  // namespace tb
}  // namespace cs
