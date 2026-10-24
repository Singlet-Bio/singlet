// Track B codec factories and registration.
#pragma once
#include "track_b_decode.h"
#include "track_b_sort.h"
#include <functional>
#include <map>

namespace cs {
namespace tb {

inline void skip_stream(const uint8_t*& p) {
    uint64_t n = detail::get_u64(p), zn = detail::get_u64(p);
    (void)n;
    p += zn;
}

// payload-only decode hooks for harness/track_b_bench.cpp: name -> factory(bytes, canonical) -> closure doing the timed work
using PayloadFn = std::function<std::function<void()>(const std::vector<uint8_t>&, bool)>;
inline std::map<std::string, PayloadFn>& payload_hooks() {
    static std::map<std::string, PayloadFn> h;
    return h;
}

// ---- lane-rANS codec -----------------------------------------------------------------------------------
struct RansOpts {
    int chunk_cols = 0;   // 0 = auto: ~target_nnz_per_chunk nnz per chunk (power-of-two columns in [128, 4096])
    int cfg_kind = -1, pb = 10;
    uint32_t target_nnz_per_chunk = 96 * 1024;   // keeps the chunk's output arrays (8 B/nnz) inside the 1.25 MB L2
    bool simd = true;
};

inline int auto_chunk_cols(const CanonicalMatrix& M, uint32_t target) {
    if (M.n == 0 || M.nnz == 0) return 1024;
    double mean_k = static_cast<double>(M.nnz) / M.n;
    double want = target / std::max(mean_k, 1.0);
    int c = 128;
    while (c < 4096 && c * 1.414 < want) c *= 2;
    return c;
}

inline std::vector<uint8_t> rans_encode(const CanonicalMatrix& M, const RansOpts& op_in) {
    RansOpts op = op_in;
    if (op.chunk_cols <= 0) op.chunk_cols = auto_chunk_cols(M, op.target_nnz_per_chunk);
    auto& bd = last_breakdown();
    bd.clear();
    std::vector<uint8_t> o;
    put_header(o, M);
    detail::put_u32(o, static_cast<uint32_t>(op.chunk_cols));
    size_t t0 = o.size();
    put_names(o, M);
    bd.names = o.size() - t0;
    RowPerm P = make_perm(M);
    t0 = o.size();
    put_rowcounts(o, P);
    bd.side = o.size() - t0;
    RankMat R = to_rank_space(M, P);
    int bestk = op.cfg_kind;
    if (bestk < 0) {
        size_t bs = ~size_t(0);
        for (int kind = 0; kind < 5; ++kind) {
            size_t e = estimate_payload(R, Cfg::make(kind, op.pb));
            if (e < bs) { bs = e; bestk = kind; }
        }
    }
    EncodedPayload best = encode_payload(R, Cfg::make(bestk, op.pb), op.chunk_cols);
    o.insert(o.end(), best.bytes.begin(), best.bytes.end());
    bd.payload = best.bytes.size();
    bd.tables = best.tables;
    bd.counts = best.counts;
    bd.chunks = best.chunks;
    return o;
}

struct RansFile {
    uint32_t m, n;
    uint64_t nnz;
    uint8_t vt;
    uint32_t chunk_cols;
    const uint8_t* names_p;    // start of names section
    const uint8_t* rowc_p;     // start of rowcounts stream
    const uint8_t* payload_p;
};
inline RansFile rans_parse(const std::vector<uint8_t>& b) {
    const uint8_t* p = b.data();
    RansFile F;
    F.m = detail::get_u32(p); F.n = detail::get_u32(p); F.nnz = detail::get_u64(p); F.vt = *p++;
    F.chunk_cols = detail::get_u32(p);
    F.names_p = p;
    skip_stream(p); skip_stream(p);
    detail::get_u32(p);
    skip_stream(p);
    F.rowc_p = p;
    skip_stream(p);
    F.payload_p = p;
    return F;
}

// payload decode: fills indptr/indices/data. canonical=false leaves rank-space ids (native order).
inline void rans_payload_decode(const RansFile& F, const RowPerm& P, bool canonical, CanonicalMatrix& M, DecState& D) {
    const uint8_t* p = F.payload_p;
    decode_open(p, D, F.n, F.chunk_cols);
    M.indptr = D.ptr;
    M.indices.resize(F.nnz);
    M.data.resize(F.nnz);
    Scratch sc;
    std::unique_ptr<Canon> cn;
#ifdef TB_HAVE_AVX512
    std::unique_ptr<CanonVec> cv;
    const bool vec = canonical && use_simd();
    if (vec) cv.reset(new CanonVec(P.perm.data(), F.m));
    else
#endif
    if (canonical) cn.reset(new Canon(P.perm.data(), F.m));
    const uint32_t nchunks = (F.n + F.chunk_cols - 1) / F.chunk_cols;
    for (uint32_t c = 0; c < nchunks; ++c) {
        uint32_t c0 = c * F.chunk_cols, c1 = std::min(F.n, c0 + F.chunk_cols);
        uint32_t base = D.ptr[c0];
        decode_chunk_native(D, c, M.indices.data() + base, M.data.data() + base, sc);
#ifdef TB_HAVE_AVX512
        if (vec) {
            for (uint32_t j = c0; j < c1; ++j)
                cv->column(M.indices.data() + D.ptr[j], M.data.data() + D.ptr[j], D.ptr[j + 1] - D.ptr[j]);
        } else
#endif
        if (canonical)
            for (uint32_t j = c0; j < c1; ++j)
                cn->column(M.indices.data() + D.ptr[j], M.data.data() + D.ptr[j], D.ptr[j + 1] - D.ptr[j]);
    }
}

struct SimdScope {
    bool old;
    explicit SimdScope(bool v) : old(use_simd()) { use_simd() = v; }
    ~SimdScope() { use_simd() = old; }
};

inline CanonicalMatrix rans_decode(const std::vector<uint8_t>& b, bool simd) {
    SimdScope sg(simd);
    RansFile F = rans_parse(b);
    CanonicalMatrix M;
    M.m = F.m; M.n = F.n; M.nnz = F.nnz; M.vt_code = F.vt;
    const uint8_t* p = F.names_p;
    get_names(p, M);
    p = F.rowc_p;
    RowPerm P;
    get_rowcounts(p, P, F.m);
    DecState D;
    rans_payload_decode(F, P, true, M, D);
    return M;
}

inline Codec make_b_rans(const std::string& name, RansOpts op) {
    Codec c;
    c.name = name;
    c.encode = [op](const CanonicalMatrix& M) { return rans_encode(M, op); };
    c.decode = [op](const std::vector<uint8_t>& b) { return rans_decode(b, op.simd); };
    payload_hooks()[name] = [op](const std::vector<uint8_t>& b, bool canonical) {
        struct St { RansFile F; RowPerm P; CanonicalMatrix M; DecState D; };
        auto st = std::make_shared<St>();
        st->F = rans_parse(b);
        const uint8_t* p = st->F.rowc_p;
        get_rowcounts(p, st->P, st->F.m);
        return std::function<void()>([st, canonical, op]() {
            SimdScope sg(op.simd);
            st->D = DecState();
            rans_payload_decode(st->F, st->P, canonical, st->M, st->D);
        });
    };
    return c;
}

inline void register_all(std::vector<Codec>& v) {
    RansOpts o;
    v.push_back(make_b_rans("b_rans16", o));
    RansOpts s = o;
    s.simd = false;
    v.push_back(make_b_rans("b_rans16_scalar", s));
    // variants for the size/speed knob study (notes/track_b.md section 8)
    RansOpts c = o;
    c.chunk_cols = 256;
    v.push_back(make_b_rans("b_rans16_c256", c));
    c.chunk_cols = 4096;
    v.push_back(make_b_rans("b_rans16_c4096", c));
    c = o;
    c.pb = 12;
    v.push_back(make_b_rans("b_rans16_pb12", c));
    c = o;
    c.pb = 11;
    v.push_back(make_b_rans("b_rans16_pb11", c));
    // context-set x probability-precision grid (cfg kinds: 0 pos40 x depth12, 1 pos40 x depth6, 2 pos20 x depth6,
    // 3 pos40 only, 4 pos20 only); table footprint = contexts x 2^pb x 4 bytes
    const int grid[][2] = {{0, 10}, {1, 10}, {2, 10}, {3, 10}, {4, 10}, {4, 9}, {2, 9}, {0, 12}};
    for (auto& g : grid) {
        c = o;
        c.cfg_kind = g[0];
        c.pb = g[1];
        v.push_back(make_b_rans("b_rans16_k" + std::to_string(g[0]) + "_pb" + std::to_string(g[1]), c));
    }
}

}  // namespace tb
}  // namespace cs
