// Track A codecs and measurement helpers. Owned by the Track A agent; see PLAN.md section 4 and
// notes/track_a.md.
//
// Everything lives in cs::ta. The pieces other tracks may want to reuse:
//   ta::prepare()            the production VOCSC front half (gap/value groups, byte split, bit planes,
//                            bitmap zero filter) producing the exact bytes the writer hands to zstd, per chunk
//   ta::EC + ec_compress()   zstd (any level, window, long mode, dictionary) / xz / brotli / bzip2 / store
//   ta::encode_plan()        full container with pluggable names / permutation / entropy coder / chunk size
//   ta::Models               reference-build vocabularies and barcode whitelists loaded from results/*.bin
//
// Every codec registered here satisfies decode(encode(M)) == M for all CanonicalMatrix fields.
#pragma once
#include "common.h"
#include <algorithm>
#include <bzlib.h>
#include <brotli/decode.h>
#include <brotli/encode.h>
#include <cstdlib>
#include <filesystem>
#include <lzma.h>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <zdict.h>
#include <zstd.h>

namespace cs {
namespace ta {

namespace pz = singlet::pz;

// ============================================================================
// Little-endian byte IO
// ============================================================================
struct ByteW {
    std::vector<uint8_t>& o;
    explicit ByteW(std::vector<uint8_t>& v) : o(v) {}
    void u8(uint8_t v) { o.push_back(v); }
    void u16(uint16_t v) { for (int i = 0; i < 2; ++i) o.push_back(uint8_t(v >> (8 * i))); }
    void u32(uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i))); }
    void u64(uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back(uint8_t(v >> (8 * i))); }
    void bytes(const void* p, size_t n) { auto b = static_cast<const uint8_t*>(p); o.insert(o.end(), b, b + n); }
    void blob(const std::vector<uint8_t>& v) { u32(uint32_t(v.size())); bytes(v.data(), v.size()); }
};
struct ByteR {
    const uint8_t* p; const uint8_t* e;
    ByteR(const uint8_t* b, size_t n) : p(b), e(b + n) {}
    void need(size_t n) const { if (size_t(e - p) < n) throw std::runtime_error("track_a: truncated stream"); }
    uint8_t u8() { need(1); return *p++; }
    uint16_t u16() { need(2); uint16_t v = uint16_t(p[0] | (p[1] << 8)); p += 2; return v; }
    uint32_t u32() { need(4); uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
    uint64_t u64() { need(8); uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }
    const uint8_t* take(size_t n) { need(n); const uint8_t* r = p; p += n; return r; }
    // length-prefixed (u32) blob; returns pointer+length
    std::pair<const uint8_t*, size_t> blob() { uint32_t n = u32(); return {take(n), n}; }
};

inline size_t varint_get(const uint8_t* b, size_t len, size_t& off) {
    size_t v = 0; int sh = 0;
    for (;;) {
        if (off >= len) throw std::runtime_error("track_a: varint past end");
        uint8_t c = b[off++];
        v |= size_t(c & 0x7F) << sh;
        if (!(c & 0x80)) return v;
        sh += 7;
        if (sh > 56) throw std::runtime_error("track_a: varint overflow");
    }
}

// ============================================================================
// Entropy coders
// ============================================================================
struct DictPair {
    ZSTD_CDict* c = nullptr;
    ZSTD_DDict* d = nullptr;
    std::vector<uint8_t> raw;
    uint64_t id = 0;
    DictPair(const std::vector<uint8_t>& dict, int level) : raw(dict) {
        c = ZSTD_createCDict(raw.data(), raw.size(), level);
        d = ZSTD_createDDict(raw.data(), raw.size());
        id = fnv64(raw.data(), raw.size());
    }
    ~DictPair() { if (c) ZSTD_freeCDict(c); if (d) ZSTD_freeDDict(d); }
    DictPair(const DictPair&) = delete;
    DictPair& operator=(const DictPair&) = delete;
};
inline std::map<uint64_t, std::shared_ptr<DictPair>>& dict_registry() {
    static std::map<uint64_t, std::shared_ptr<DictPair>> r;
    return r;
}

enum EcKind : uint8_t { EC_STORE = 0, EC_ZSTD = 1, EC_XZ = 2, EC_BROTLI = 3, EC_BZIP2 = 4 };

struct EC {
    uint8_t kind = EC_ZSTD;
    int level = 3;            // zstd level / xz preset (9 => 9e) / brotli quality / bzip2 blockSize100k
    int window_log = 0;       // zstd: 0 = library default; brotli: lgwin (0 => 24)
    bool ldm = false;         // zstd long distance matching
    std::shared_ptr<DictPair> dict;
    std::string label;
};

inline EC ec_zstd(int level, int window_log = 0, bool ldm = false) {
    EC e; e.kind = EC_ZSTD; e.level = level; e.window_log = window_log; e.ldm = ldm;
    e.label = "zstd" + std::to_string(level) + (window_log ? "w" + std::to_string(window_log) : "") + (ldm ? "ldm" : "");
    return e;
}
inline EC ec_xz() { EC e; e.kind = EC_XZ; e.level = 9; e.label = "xz9e"; return e; }
inline EC ec_brotli(int q = 11, int lgwin = 24) {
    EC e; e.kind = EC_BROTLI; e.level = q; e.window_log = lgwin; e.label = "brotli" + std::to_string(q) + "w" + std::to_string(lgwin); return e;
}
inline EC ec_bzip2() { EC e; e.kind = EC_BZIP2; e.level = 9; e.label = "bzip2"; return e; }
inline EC ec_store() { EC e; e.kind = EC_STORE; e.label = "store"; return e; }

inline uint32_t lzma_dict_for(size_t n) {
    uint32_t d = 4096;
    while (d < n && d < (64u << 20)) d <<= 1;
    return d;
}

inline void ec_compress(const EC& ec, const uint8_t* src, size_t n, std::vector<uint8_t>& out) {
    switch (ec.kind) {
    case EC_STORE:
        out.assign(src, src + n);
        return;
    case EC_ZSTD: {
        thread_local ZSTD_CCtx* ctx = ZSTD_createCCtx();
        ZSTD_CCtx_reset(ctx, ZSTD_reset_session_and_parameters);
        if (ec.dict) ZSTD_CCtx_refCDict(ctx, ec.dict->c);
        else ZSTD_CCtx_setParameter(ctx, ZSTD_c_compressionLevel, ec.level);
        ZSTD_CCtx_setParameter(ctx, ZSTD_c_checksumFlag, 1);
        if (ec.window_log) ZSTD_CCtx_setParameter(ctx, ZSTD_c_windowLog, ec.window_log);
        if (ec.ldm) ZSTD_CCtx_setParameter(ctx, ZSTD_c_enableLongDistanceMatching, 1);
        out.resize(ZSTD_compressBound(n));
        size_t z = ZSTD_compress2(ctx, out.data(), out.size(), src, n);
        if (ZSTD_isError(z)) throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(z));
        out.resize(z);
        return;
    }
    case EC_XZ: {
        lzma_options_lzma opt;
        lzma_lzma_preset(&opt, uint32_t(ec.level) | LZMA_PRESET_EXTREME);
        opt.dict_size = lzma_dict_for(n);
        lzma_filter f[2] = {{LZMA_FILTER_LZMA2, &opt}, {LZMA_VLI_UNKNOWN, nullptr}};
        out.resize(n + n / 8 + 1024);
        size_t pos = 0;
        lzma_ret r = lzma_raw_buffer_encode(f, nullptr, src, n, out.data(), &pos, out.size());
        if (r != LZMA_OK) throw std::runtime_error("lzma encode failed");
        out.resize(pos);
        return;
    }
    case EC_BROTLI: {
        size_t cap = BrotliEncoderMaxCompressedSize(n);
        out.resize(cap ? cap : 64);
        size_t sz = out.size();
        if (!BrotliEncoderCompress(ec.level, ec.window_log ? ec.window_log : 24, BROTLI_MODE_GENERIC, n, src, &sz, out.data()))
            throw std::runtime_error("brotli encode failed");
        out.resize(sz);
        return;
    }
    case EC_BZIP2: {
        unsigned int cap = unsigned(n + n / 100 + 1024);
        out.resize(cap);
        int r = BZ2_bzBuffToBuffCompress(reinterpret_cast<char*>(out.data()), &cap,
                                         reinterpret_cast<char*>(const_cast<uint8_t*>(src)), unsigned(n), ec.level, 0, 0);
        if (r != BZ_OK) throw std::runtime_error("bzip2 encode failed");
        out.resize(cap);
        return;
    }
    }
    throw std::runtime_error("unknown entropy coder");
}

// Decompress to exactly out_n bytes (the container always records the decoded size).
inline void ec_decompress(const EC& ec, const uint8_t* src, size_t n, uint8_t* dst, size_t out_n) {
    switch (ec.kind) {
    case EC_STORE:
        if (n != out_n) throw std::runtime_error("store: size mismatch");
        std::memcpy(dst, src, n);
        return;
    case EC_ZSTD: {
        thread_local ZSTD_DCtx* ctx = ZSTD_createDCtx();
        size_t z = ec.dict ? ZSTD_decompress_usingDDict(ctx, dst, out_n, src, n, ec.dict->d)
                           : ZSTD_decompressDCtx(ctx, dst, out_n, src, n);
        if (ZSTD_isError(z) || z != out_n) throw std::runtime_error("zstd decode failed");
        return;
    }
    case EC_XZ: {
        lzma_options_lzma opt;
        lzma_lzma_preset(&opt, 0);
        opt.dict_size = lzma_dict_for(out_n);
        lzma_filter f[2] = {{LZMA_FILTER_LZMA2, &opt}, {LZMA_VLI_UNKNOWN, nullptr}};
        size_t ip = 0, op = 0;
        lzma_ret r = lzma_raw_buffer_decode(f, nullptr, src, &ip, n, dst, &op, out_n);
        if (r != LZMA_OK && r != LZMA_STREAM_END) throw std::runtime_error("lzma decode failed");
        if (op != out_n) throw std::runtime_error("lzma size mismatch");
        return;
    }
    case EC_BROTLI: {
        size_t sz = out_n;
        if (BrotliDecoderDecompress(n, src, &sz, dst) != BROTLI_DECODER_RESULT_SUCCESS || sz != out_n)
            throw std::runtime_error("brotli decode failed");
        return;
    }
    case EC_BZIP2: {
        unsigned int sz = unsigned(out_n);
        int r = BZ2_bzBuffToBuffDecompress(reinterpret_cast<char*>(dst), &sz,
                                           reinterpret_cast<char*>(const_cast<uint8_t*>(src)), unsigned(n), 0, 0);
        if (r != BZ_OK || sz != out_n) throw std::runtime_error("bzip2 decode failed");
        return;
    }
    }
    throw std::runtime_error("unknown entropy coder");
}

inline std::vector<uint8_t> z_compress(const EC& ec, const std::vector<uint8_t>& v) {
    std::vector<uint8_t> o; ec_compress(ec, v.data(), v.size(), o); return o;
}

// ============================================================================
// VOCSC front half: exactly the bytes pz_writer.h hands to zstd, per chunk
// ============================================================================
inline std::vector<uint32_t> perm_from_rnnz(const std::vector<uint32_t>& rnnz) {
    // Counting sort, descending by row nnz, ties by ascending row id (identical to
    // pz_writer.h encode_vocsc_section).
    const uint32_t m = uint32_t(rnnz.size());
    uint32_t mx = 0;
    for (uint32_t v : rnnz) mx = std::max(mx, v);
    std::vector<uint32_t> bucket(size_t(mx) + 1, 0), pos(size_t(mx) + 1);
    for (uint32_t i = 0; i < m; ++i) bucket[rnnz[i]]++;
    uint32_t p = 0;
    for (int64_t b = mx; b >= 0; --b) { pos[b] = p; p += bucket[b]; }
    std::vector<uint32_t> perm(m);
    for (uint32_t i = 0; i < m; ++i) perm[pos[rnnz[i]]++] = i;
    return perm;
}

struct Chunk {
    uint32_t ng = 0, msz = 0;
    std::vector<uint8_t> packed;   // bitmap_pack(prefilter): what zstd sees in production
};

struct Prepared {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    uint8_t vt = 0;
    int chunk_cols = 1024, nc = 0;
    bool gap16 = true;
    int gw = 2;
    uint8_t ptr_width = 2;
    std::vector<uint32_t> rnnz, perm, cc;
    std::vector<Chunk> chunks;
    size_t packed_total() const { size_t s = 0; for (auto& c : chunks) s += c.packed.size(); return s; }
    size_t pf_size(const Chunk& c) const {
        size_t bp = c.ng > 0 ? 8 * ((size_t(c.ng) + 7) / 8) : 0;
        return size_t(c.msz) + bp + (c.ng > 0 ? size_t(gw - 1) * c.ng : 0);
    }
};

inline Prepared prepare(const CanonicalMatrix& M, int chunk_cols, const std::vector<uint32_t>* perm_override = nullptr,
                        int nthreads = 1) {
    Prepared P;
    P.m = M.m; P.n = M.n; P.nnz = M.nnz; P.vt = M.vt_code; P.chunk_cols = chunk_cols;
    P.rnnz.assign(M.m, 0);
    for (uint64_t k = 0; k < M.nnz; ++k) P.rnnz[M.indices[k]]++;
    P.perm = perm_override ? *perm_override : perm_from_rnnz(P.rnnz);
    std::vector<uint32_t> inv(M.m);
    for (uint32_t i = 0; i < M.m; ++i) inv[P.perm[i]] = i;
    P.cc.assign(M.n, 0);
    for (uint32_t j = 0; j < M.n; ++j) {
        int64_t d = int64_t(M.indptr[j + 1]) - int64_t(M.indptr[j]);
        P.cc[j] = d > 0 ? uint32_t(d) : 0;
    }
    bool use16p = true;
    for (uint32_t j = 0; j < M.n; ++j) if (P.cc[j] > 65535) { use16p = false; break; }
    P.ptr_width = use16p ? 2 : 4;
    uint64_t tnnz = 0;
    for (uint32_t j = 0; j < M.n; ++j) tnnz += P.cc[j];
    pz::CSCView mat{M.m, M.n, tnnz, M.indptr.data(), M.indices.data(), M.data.data()};
    const int nc = M.n ? (int(M.n) + chunk_cols - 1) / chunk_cols : 0;
    P.nc = nc;
    std::vector<pz::VocscResult> res(nc);
    auto run = [&](auto&& fn) {
        if (nthreads <= 1 || nc <= 1) { fn(0, nc); return; }
        int nw = std::min(nthreads, nc), per = nc / nw, rem = nc % nw, start = 0;
        std::vector<std::thread> th;
        for (int w = 0; w < nw; ++w) { int cnt = per + (w < rem ? 1 : 0); th.emplace_back(fn, start, start + cnt); start += cnt; }
        for (auto& t : th) t.join();
    };
    run([&](int a, int b) {
        for (int c = a; c < b; ++c) {
            int64_t cs = int64_t(c) * chunk_cols, ce = std::min(cs + int64_t(chunk_cols), int64_t(M.n));
            res[c] = pz::vocsc_encode_chunk(mat, inv.data(), P.cc.data(), cs, ce);
        }
    });
    bool any32 = false;
    for (auto& r : res) if (r.max_gap >= 65536) any32 = true;
    P.gap16 = !any32;
    P.gw = P.gap16 ? 2 : 4;
    const int gw = P.gw;
    P.chunks.resize(nc);
    run([&](int a, int b) {
        std::vector<uint8_t> raw, pf;
        for (int c = a; c < b; ++c) {
            auto& v = res[c];
            size_t ng = v.n_gaps, msz = v.metadata.size();
            raw.resize(msz + ng * gw);
            std::memcpy(raw.data(), v.metadata.data(), msz);
            if (ng > 0) {
                if (P.gap16) pz::byte_split_16(v.gaps.data(), ng, raw.data() + msz);
                else pz::byte_split_32(v.gaps.data(), ng, raw.data() + msz);
            }
            size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
            size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
            pf.resize(pf_sz);
            std::memcpy(pf.data(), raw.data(), msz);
            if (ng > 0) {
                pz::bit_planes_encode(raw.data() + msz, ng, pf.data() + msz);
                if (gw > 1) std::memcpy(pf.data() + msz + bp_sz, raw.data() + msz + ng, ng * (gw - 1));
            }
            auto& ch = P.chunks[c];
            ch.ng = uint32_t(ng); ch.msz = uint32_t(msz);
            ch.packed.resize(pf_sz + (pf_sz + 7) / 8);
            size_t psz = pz::bitmap_pack(pf.data(), pf_sz, ch.packed.data());
            ch.packed.resize(psz);
            v.gaps.clear(); v.gaps.shrink_to_fit();
            v.metadata.clear(); v.metadata.shrink_to_fit();
        }
    });
    return P;
}

// Split one chunk's prefilter (before the bitmap zero filter) into its three logical streams and return
// their sizes: varint metadata, bit-planes of the low gap byte, upper gap bytes. Used for accounting only.
struct ChunkParts { std::vector<uint8_t> meta, planes, upper; };
inline ChunkParts split_chunk(const Prepared& P, const Chunk& c) {
    size_t pf_sz = P.pf_size(c);
    std::vector<uint8_t> pf(pf_sz, 0);
    size_t bm = (pf_sz + 7) / 8, wp = bm;
    for (size_t i = 0; i < pf_sz; ++i)
        if (c.packed[i / 8] & (1u << (i & 7))) pf[i] = c.packed[wp++];
    size_t bp_sz = c.ng > 0 ? 8 * ((size_t(c.ng) + 7) / 8) : 0;
    ChunkParts r;
    r.meta.assign(pf.begin(), pf.begin() + c.msz);
    r.planes.assign(pf.begin() + c.msz, pf.begin() + c.msz + bp_sz);
    r.upper.assign(pf.begin() + c.msz + bp_sz, pf.end());
    return r;
}

// ============================================================================
// Entropy stage: blobs in the production layout [ng][msz][zsz][crc][psz][payload]
// ============================================================================
inline std::vector<std::vector<uint8_t>> entropy_chunks(const Prepared& P, const EC& ec) {
    std::vector<std::vector<uint8_t>> blobs(P.nc);
    std::vector<uint8_t> z;
    for (int c = 0; c < P.nc; ++c) {
        const Chunk& ch = P.chunks[c];
        uint32_t crc = pz::CRC32::compute(ch.packed.data(), ch.packed.size());
        ec_compress(ec, ch.packed.data(), ch.packed.size(), z);
        auto& b = blobs[c];
        ByteW w(b);
        w.u32(ch.ng); w.u32(ch.msz); w.u32(uint32_t(z.size())); w.u32(crc); w.u32(uint32_t(ch.packed.size()));
        w.bytes(z.data(), z.size());
    }
    return blobs;
}

// Mirror of pz_reader.h decode_vocsc_section, with a pluggable entropy decoder. Same checks, same
// per-column re-sort, so decode timings are comparable with the production reader.
inline void decode_chunks(const uint8_t* base, const std::vector<uint32_t>& ctable, const EC& ec,
                          uint32_t m, uint32_t n, uint64_t nnz, int chunk_cols, bool gap16, uint8_t vt,
                          const std::vector<uint32_t>& perm, const std::vector<uint32_t>& cc, CanonicalMatrix& M) {
    const size_t nc = ctable.size();
    const int gw = gap16 ? 2 : 4;
    const uint64_t vmax = vt == 1 ? 0xFFull : vt == 2 ? 0xFFFFull : 0xFFFFFFFFull;
    M.indices.assign(size_t(nnz), 0);
    M.data.assign(size_t(nnz), 0);
    const bool derive = cc.empty();   // no ptr block: column lengths come from the value-group counts
    if (derive) { M.indptr.assign(size_t(n) + 1, 0); }
    std::vector<uint8_t> raw, packed, prefilter;
    std::vector<uint32_t> gaps;
    std::vector<std::pair<uint32_t, uint32_t>> rv;
    const uint8_t* blob = base;
    uint64_t total_ng = 0;
    for (size_t c = 0; c < nc; ++c) {
        uint32_t ng, msz, zsz, crc, psz;
        std::memcpy(&ng, blob, 4); std::memcpy(&msz, blob + 4, 4); std::memcpy(&zsz, blob + 8, 4);
        std::memcpy(&crc, blob + 12, 4); std::memcpy(&psz, blob + 16, 4);
        if (zsz != ctable[c] - 20) throw std::runtime_error("track_a: chunk size mismatch");
        total_ng += ng;
        const size_t col_start = c * size_t(chunk_cols);
        const size_t col_end = std::min(col_start + size_t(chunk_cols), size_t(n));
        const size_t bp_sz = ng > 0 ? 8 * ((size_t(ng) + 7) / 8) : 0;
        const size_t pf_sz = size_t(msz) + bp_sz + (ng > 0 ? size_t(gw - 1) * ng : 0);
        if (psz > (pf_sz + 7) / 8 + pf_sz) throw std::runtime_error("track_a: packed size out of range");
        packed.resize(psz);
        ec_decompress(ec, blob + 20, zsz, packed.data(), psz);
        if (pz::CRC32::compute(packed.data(), packed.size()) != crc) throw std::runtime_error("track_a: chunk CRC mismatch");
        prefilter.assign(pf_sz, 0);
        pz::bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);
        raw.resize(size_t(msz) + size_t(ng) * gw);
        std::memcpy(raw.data(), prefilter.data(), msz);
        if (ng > 0) {
            pz::bit_planes_decode(prefilter.data() + msz, ng, raw.data() + msz);
            if (gw > 1) std::memcpy(raw.data() + msz + ng, prefilter.data() + msz + bp_sz, size_t(gw - 1) * ng);
        }
        gaps.assign(ng, 0);
        if (ng > 0) {
            if (gw == 2) pz::byte_unsplit_16(raw.data() + msz, ng, gaps.data());
            else pz::byte_unsplit_32(raw.data() + msz, ng, gaps.data());
        }
        size_t mp = 0, gp = 0;
        const uint8_t* meta = raw.data();
        for (size_t j = col_start; j < col_end; ++j) {
            uint32_t ng_col = pz::varint_read(meta, msz, mp);
            if (ng_col == 0) {
                if (derive) M.indptr[j + 1] = M.indptr[j];
                else if (cc[j] != 0) throw std::runtime_error("track_a: column has entries but empty value list");
                continue;
            }
            rv.clear();
            const uint64_t room = derive ? nnz - M.indptr[j] : cc[j];
            if (!derive) rv.reserve(cc[j]);
            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv = pz::varint_read(meta, msz, mp);
                uint32_t cnt = pz::varint_read(meta, msz, mp);
                if (cv > vmax) throw std::runtime_error("track_a: value exceeds type");
                if (cnt > room - rv.size()) throw std::runtime_error("track_a: value group longer than column");
                uint64_t prev = 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    if (gp >= ng) throw std::runtime_error("track_a: gap stream exhausted");
                    uint64_t prow = prev + gaps[gp++];
                    prev = prow + 1;
                    if (prow >= m) throw std::runtime_error("track_a: row out of bounds");
                    rv.emplace_back(perm[prow], cv);
                }
            }
            std::sort(rv.begin(), rv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            if (derive) M.indptr[j + 1] = M.indptr[j] + uint32_t(rv.size());
            else if (rv.size() != cc[j]) throw std::runtime_error("track_a: column length mismatch");
            const uint32_t b0 = M.indptr[j];
            for (size_t k = 0; k < rv.size(); ++k) { M.indices[b0 + k] = rv[k].first; M.data[b0 + k] = rv[k].second; }
        }
        if (gp != ng || mp != msz) throw std::runtime_error("track_a: leftover stream bytes");
        blob += ctable[c];
    }
    if (total_ng != nnz) throw std::runtime_error("track_a: gap counts do not add up to nnz");
    if (derive && M.indptr[n] != nnz) throw std::runtime_error("track_a: derived column lengths do not add up to nnz");
}

// ============================================================================
// Models: reference-build vocabularies, barcode whitelists
// ============================================================================
// Vocabulary file  results/track_a_vocab_<build>.bin :
//   "TAVOC001" u32 n_lists ; per list: u16 len + kind, u32 count, u64 hash, u8 coder, u64 raw_len, u64 z_len, u64 off
//   then the blobs. A list is the NUL-joined row names of one matrix kind (exon / intron / ...).
// Whitelist file   results/track_a_whitelist_<name>.bin :
//   "TAWL0001" u16 len + name, u32 count, u64 z_len, zstd-19 of delta-1 varints of the sorted 2-bit codes.
inline uint64_t names_hash(const std::vector<std::string>& v) {
    uint64_t h = 1469598103934665603ull;
    for (auto& s : v) { h = fnv64(s.data(), s.size(), h); uint8_t z = 0; h = fnv64(&z, 1, h); }
    return h;
}
inline std::vector<uint8_t> join_nul(const std::vector<std::string>& v) {
    std::vector<uint8_t> o;
    for (auto& s : v) { o.insert(o.end(), s.begin(), s.end()); o.push_back(0); }
    return o;
}
inline void split_nul(const uint8_t* b, size_t n, size_t count, std::vector<std::string>& out) {
    out.clear(); out.reserve(count);
    size_t i = 0;
    for (size_t k = 0; k < count; ++k) {
        size_t j = i;
        while (j < n && b[j] != 0) ++j;
        out.emplace_back(reinterpret_cast<const char*>(b + i), j - i);
        i = j + 1;
    }
}

inline int bc_code(const std::string& s, uint32_t& out) {   // 16-mer ACGT -> 32 bit, first base most significant
    if (s.size() != 16) return 0;
    uint32_t v = 0;
    for (char ch : s) {
        uint32_t x;
        switch (ch) { case 'A': x = 0; break; case 'C': x = 1; break; case 'G': x = 2; break; case 'T': x = 3; break; default: return 0; }
        v = (v << 2) | x;
    }
    out = v;
    return 1;
}
inline std::string bc_str(uint32_t v) {
    static const char L[4] = {'A', 'C', 'G', 'T'};
    std::string s(16, 'A');
    for (int i = 15; i >= 0; --i) { s[i] = L[v & 3]; v >>= 2; }
    return s;
}

struct VocabEntry {
    std::string path, build, kind;
    uint32_t count = 0;
    uint64_t hash = 0, raw_len = 0, z_len = 0, off = 0;
    uint8_t coder = 0;
    uint64_t file_bytes = 0;
    std::shared_ptr<std::vector<uint8_t>> joined;
};
struct WhitelistEntry {
    std::string path, name;
    uint32_t count = 0;
    uint64_t z_len = 0, off = 0, file_bytes = 0;
    std::shared_ptr<std::vector<uint32_t>> codes;   // sorted
};

inline EC coder_for_tag(uint8_t t) {
    switch (t) { case 0: return ec_zstd(19); case 1: return ec_xz(); case 2: return ec_brotli(); default: return ec_zstd(22); }
}

class Models {
public:
    std::map<uint64_t, VocabEntry> vocab;     // by names hash
    std::vector<WhitelistEntry> wls;
    static Models& get() { static Models m; return m; }
    static std::string dir() { const char* e = std::getenv("TRACK_A_MODELS"); return e ? e : "results"; }
    void rescan() {
        vocab.clear(); wls.clear();
        std::error_code ec;
        for (auto& de : std::filesystem::directory_iterator(dir(), ec)) {
            std::string fn = de.path().filename().string();
            if (fn.rfind("track_a_vocab_", 0) == 0 && fn.size() > 4 && fn.substr(fn.size() - 4) == ".bin") scan_vocab(de.path().string(), fn.substr(14, fn.size() - 18));
            else if (fn.rfind("track_a_whitelist_", 0) == 0 && fn.substr(fn.size() - 4) == ".bin") scan_wl(de.path().string());
        }
    }
    size_t model_bytes_total() const {
        size_t s = 0;
        for (auto& [h, v] : vocab) { (void)h; }
        std::map<std::string, uint64_t> files;
        for (auto& [h, v] : vocab) files[v.path] = v.file_bytes;
        for (auto& [p, b] : files) { (void)p; s += b; }
        for (auto& w : wls) s += w.file_bytes;
        return s;
    }
    std::shared_ptr<std::vector<uint8_t>> joined(uint64_t hash) {
        auto it = vocab.find(hash);
        if (it == vocab.end()) throw std::runtime_error("track_a: vocabulary not found (model file missing)");
        VocabEntry& v = it->second;
        if (!v.joined) {
            auto buf = read_file(v.path);
            std::vector<uint8_t> out(v.raw_len);
            ec_decompress(coder_for_tag(v.coder), buf.data() + v.off, v.z_len, out.data(), v.raw_len);
            v.joined = std::make_shared<std::vector<uint8_t>>(std::move(out));
        }
        return v.joined;
    }
    std::shared_ptr<std::vector<uint32_t>> codes(WhitelistEntry& w) {
        if (!w.codes) {
            auto buf = read_file(w.path);
            std::vector<uint8_t> raw;
            // raw length = sum of varint lengths; stored first
            ByteR r(buf.data() + w.off - 8, 8);
            uint64_t raw_len = r.u64();
            raw.resize(raw_len);
            ec_decompress(ec_zstd(19), buf.data() + w.off, w.z_len, raw.data(), raw_len);
            auto c = std::make_shared<std::vector<uint32_t>>();
            c->reserve(w.count);
            size_t off = 0; uint64_t prev = 0; bool first = true;
            while (off < raw.size()) {
                uint64_t d = varint_get(raw.data(), raw.size(), off);
                uint64_t v = first ? d : prev + d + 1;
                first = false; prev = v;
                c->push_back(uint32_t(v));
            }
            w.codes = c;
        }
        return w.codes;
    }
    WhitelistEntry* wl_by_name(const std::string& n) {
        for (auto& w : wls) if (w.name == n) return &w;
        return nullptr;
    }
private:
    Models() { rescan(); }
    void scan_vocab(const std::string& path, const std::string& build) {
        try {
            auto buf = read_file(path);
            ByteR r(buf.data(), buf.size());
            if (std::memcmp(r.take(8), "TAVOC001", 8) != 0) return;
            uint32_t nl = r.u32();
            for (uint32_t i = 0; i < nl; ++i) {
                VocabEntry v;
                v.path = path; v.build = build; v.file_bytes = buf.size();
                uint16_t kl = r.u16();
                v.kind.assign(reinterpret_cast<const char*>(r.take(kl)), kl);
                v.count = r.u32(); v.hash = r.u64(); v.coder = r.u8();
                v.raw_len = r.u64(); v.z_len = r.u64(); v.off = r.u64();
                vocab[v.hash] = v;
            }
        } catch (...) {}
    }
    void scan_wl(const std::string& path) {
        try {
            auto buf = read_file(path);
            ByteR r(buf.data(), buf.size());
            if (std::memcmp(r.take(8), "TAWL0001", 8) != 0) return;
            WhitelistEntry w;
            w.path = path; w.file_bytes = buf.size();
            uint16_t nl = r.u16();
            w.name.assign(reinterpret_cast<const char*>(r.take(nl)), nl);
            w.count = r.u32(); w.z_len = r.u64();
            w.off = size_t(r.p - buf.data()) + 8;   // 8 bytes of raw_len precede the blob
            wls.push_back(w);
        } catch (...) {}
    }
};

// ---- writers for the model files (used by harness/track_a_models.cpp) ----
inline std::vector<uint8_t> build_vocab_file(const std::vector<std::pair<std::string, std::vector<std::string>>>& lists) {
    struct L { std::string kind; uint32_t count; uint64_t hash, raw_len; uint8_t coder; std::vector<uint8_t> z; };
    std::vector<L> ls;
    for (auto& [kind, names] : lists) {
        L l; l.kind = kind; l.count = uint32_t(names.size()); l.hash = names_hash(names);
        auto raw = join_nul(names);
        l.raw_len = raw.size();
        // pick the smallest of zstd-19, xz-9e, brotli-11
        l.coder = 0; l.z = z_compress(coder_for_tag(0), raw);
        for (uint8_t t : {uint8_t(1), uint8_t(2)}) {
            auto z = z_compress(coder_for_tag(t), raw);
            if (z.size() < l.z.size()) { l.z = std::move(z); l.coder = t; }
        }
        ls.push_back(std::move(l));
    }
    size_t hdr = 8 + 4;
    for (auto& l : ls) hdr += 2 + l.kind.size() + 4 + 8 + 1 + 8 + 8 + 8;
    std::vector<uint8_t> o; ByteW w(o);
    w.bytes("TAVOC001", 8); w.u32(uint32_t(ls.size()));
    uint64_t off = hdr;
    for (auto& l : ls) {
        w.u16(uint16_t(l.kind.size())); w.bytes(l.kind.data(), l.kind.size());
        w.u32(l.count); w.u64(l.hash); w.u8(l.coder); w.u64(l.raw_len); w.u64(l.z.size()); w.u64(off);
        off += l.z.size();
    }
    for (auto& l : ls) w.bytes(l.z.data(), l.z.size());
    return o;
}
inline std::vector<uint8_t> build_whitelist_file(const std::string& name, std::vector<uint32_t> codes) {
    std::sort(codes.begin(), codes.end());
    std::vector<uint8_t> raw;
    uint32_t prev = 0;
    for (size_t i = 0; i < codes.size(); ++i) {
        pz::varint_push(raw, i == 0 ? codes[0] : codes[i] - prev - 1);
        prev = codes[i];
    }
    auto z = z_compress(ec_zstd(19), raw);
    std::vector<uint8_t> o; ByteW w(o);
    w.bytes("TAWL0001", 8); w.u16(uint16_t(name.size())); w.bytes(name.data(), name.size());
    w.u32(uint32_t(codes.size())); w.u64(z.size());
    w.u64(raw.size());
    w.bytes(z.data(), z.size());
    return o;
}

// ============================================================================
// Bit packing / Rice coding
// ============================================================================
struct BitW {
    std::vector<uint8_t>& o; uint64_t acc = 0; int nb = 0;
    explicit BitW(std::vector<uint8_t>& v) : o(v) {}
    void put(uint64_t v, int bits) {          // bits <= 32
        acc |= (v & ((1ull << bits) - 1)) << nb; nb += bits;
        while (nb >= 8) { o.push_back(uint8_t(acc)); acc >>= 8; nb -= 8; }
    }
    void flush() { if (nb) { o.push_back(uint8_t(acc)); acc = 0; nb = 0; } }
};
struct BitR {
    const uint8_t* p; size_t n, pos = 0; uint64_t acc = 0; int nb = 0;
    BitR(const uint8_t* b, size_t len) : p(b), n(len) {}
    uint64_t get(int bits) {
        while (nb < bits) { if (pos >= n) throw std::runtime_error("track_a: bit stream truncated"); acc |= uint64_t(p[pos++]) << nb; nb += 8; }
        uint64_t v = acc & ((1ull << bits) - 1); acc >>= bits; nb -= bits; return v;
    }
};
inline int ceil_log2(uint64_t x) { int b = 0; while ((1ull << b) < x) ++b; return b; }

// ============================================================================
// Names section
// ============================================================================
// names_mode 0: production metadata block (TLV rownames/colnames/user_kv, zstd-3).
// names_mode 1: model-side names:
//     u8 row_mode (0 inline NUL-joined + z19, 1 vocabulary id) [+ payload]
//     u8 bc_mode  (0 inline, 1 2-bit packed in file order, 2 whitelist index, 3 sorted deltas + order permutation)
//     user_kv TLV (zstd-3, usually empty)
struct NamesStats { size_t row_bytes = 0, col_bytes = 0, kv_bytes = 0; int row_mode = 0, bc_mode = 0; };

inline std::vector<uint8_t> tlv_kv(const std::map<std::string, std::string>& kv) {
    std::vector<uint8_t> raw;
    pz::push_kv_tlv(raw, kv);
    return raw;
}

inline std::vector<uint8_t> enc_names_inline_tlv(const CanonicalMatrix& M) {
    std::vector<uint8_t> raw;
    bool has_meta = !M.rownames.empty() || !M.colnames.empty() || !M.user_kv.empty();
    if (!has_meta) return {};
    pz::push_strings_tlv(raw, pz::META_TAG_ROWNAMES, M.rownames);
    pz::push_strings_tlv(raw, pz::META_TAG_COLNAMES, M.colnames);
    pz::push_kv_tlv(raw, M.user_kv);
    raw.push_back(pz::META_TAG_END);
    return z_compress(ec_zstd(3), raw);
}

// ---- barcode representations: each returns payload bytes or empty+false if not applicable ----
inline bool bc_pack2(const std::vector<std::string>& cn, std::vector<uint8_t>& out) {
    out.clear();
    for (auto& s : cn) {
        uint32_t v;
        if (!bc_code(s, v)) return false;
        for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i)));
    }
    return true;
}
inline bool bc_whitelist(const std::vector<std::string>& cn, std::vector<uint8_t>& out, std::string& wl_name) {
    auto& md = Models::get();
    std::vector<uint32_t> codes(cn.size());
    for (size_t i = 0; i < cn.size(); ++i) if (!bc_code(cn[i], codes[i])) return false;
    bool found = false;
    size_t best = SIZE_MAX;
    for (auto& w : md.wls) {
        auto wc = md.codes(w);
        std::vector<uint8_t> tmp;
        BitW bw(tmp);
        int bits = ceil_log2(wc->size());
        bool ok = true;
        for (uint32_t c : codes) {
            auto it = std::lower_bound(wc->begin(), wc->end(), c);
            if (it == wc->end() || *it != c) { ok = false; break; }
            bw.put(uint64_t(it - wc->begin()), bits);
        }
        if (!ok) continue;
        bw.flush();
        if (tmp.size() < best) { best = tmp.size(); out = std::move(tmp); wl_name = w.name; found = true; }
    }
    return found;
}
inline bool bc_sorted_delta(const std::vector<std::string>& cn, std::vector<uint8_t>& out) {
    const size_t n = cn.size();
    std::vector<uint32_t> v(n);
    for (size_t i = 0; i < n; ++i) if (!bc_code(cn[i], v[i])) return false;
    std::vector<uint32_t> order(n);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return v[a] < v[b]; });
    for (size_t i = 1; i < n; ++i) if (v[order[i]] == v[order[i - 1]]) return false;   // needs distinct barcodes
    // Rice-coded gaps (delta-1) of the sorted values
    std::vector<uint32_t> rank(n);
    for (size_t i = 0; i < n; ++i) rank[order[i]] = uint32_t(i);
    uint64_t span = n ? uint64_t(v[order[n - 1]]) - v[order[0]] + 1 : 1;
    int k = 0;
    if (n) { uint64_t mean = span / n; while ((2ull << k) <= mean) ++k; }
    std::vector<uint8_t> rice;
    {
        BitW bw(rice);
        for (size_t i = 1; i < n; ++i) {
            uint64_t d = uint64_t(v[order[i]]) - v[order[i - 1]] - 1;
            uint64_t q = d >> k;
            while (q >= 32) { bw.put(0, 32); q -= 32; }          // unary: q zeros ...
            bw.put(0, int(q)); bw.put(1, 1);                       // ... then a one
            if (k) bw.put(d & ((1ull << k) - 1), k);
        }
        bw.flush();
    }
    // permutation: column j holds sorted rank rank[j]; ceil(log2 n) bits each
    std::vector<uint8_t> pm;
    {
        BitW pw(pm);
        int pb = ceil_log2(n);
        for (size_t j = 0; j < n; ++j) pw.put(rank[j], pb);
        pw.flush();
    }
    out.clear();
    ByteW w(out);
    w.u32(uint32_t(n)); w.u8(uint8_t(k)); w.u32(n ? v[order[0]] : 0); w.u32(uint32_t(rice.size()));
    w.bytes(rice.data(), rice.size());
    w.bytes(pm.data(), pm.size());
    return true;
}

inline NamesStats enc_names_model(const CanonicalMatrix& M, std::vector<uint8_t>& out, int force_bc = -1) {
    NamesStats st;
    ByteW w(out);
    auto& md = Models::get();
    // rows
    uint64_t h = names_hash(M.rownames);
    auto it = md.vocab.find(h);
    bool use_vocab = false;
    if (it != md.vocab.end() && it->second.count == M.rownames.size()) {
        auto j = md.joined(h);
        use_vocab = (join_nul(M.rownames) == *j);   // exact verification, hash alone is not trusted
    }
    size_t s0 = out.size();
    if (use_vocab) { w.u8(1); w.u64(h); st.row_mode = 1; }
    else {
        w.u8(0);
        auto z = z_compress(ec_zstd(19), join_nul(M.rownames));
        w.u64(join_nul(M.rownames).size()); w.blob(z);
    }
    st.row_bytes = out.size() - s0;
    // columns
    s0 = out.size();
    std::vector<uint8_t> best; int best_mode = 0; std::string wl_name;
    {
        auto inl = z_compress(ec_zstd(19), join_nul(M.colnames));
        best = inl; best_mode = 0;
        std::vector<uint8_t> t; std::string wn;
        if (force_bc < 0 || force_bc == 1) if (bc_pack2(M.colnames, t) && (t.size() < best.size() || force_bc == 1)) { best = t; best_mode = 1; }
        if (force_bc < 0 || force_bc == 2) if (bc_whitelist(M.colnames, t, wn) && (t.size() < best.size() || force_bc == 2)) { best = t; best_mode = 2; wl_name = wn; }
        if (force_bc < 0 || force_bc == 3) if (bc_sorted_delta(M.colnames, t) && (t.size() < best.size() || force_bc == 3)) { best = t; best_mode = 3; }
        if (force_bc == 0) { best = inl; best_mode = 0; }
    }
    w.u8(uint8_t(best_mode)); st.bc_mode = best_mode;
    w.u32(uint32_t(M.colnames.size()));
    if (best_mode == 0) w.u64(join_nul(M.colnames).size());
    if (best_mode == 2) { w.u8(uint8_t(wl_name.size())); w.bytes(wl_name.data(), wl_name.size()); }
    w.blob(best);
    st.col_bytes = out.size() - s0;
    s0 = out.size();
    auto kv = tlv_kv(M.user_kv);
    w.blob(kv.empty() ? kv : z_compress(ec_zstd(3), kv));
    st.kv_bytes = out.size() - s0;
    return st;
}

inline void dec_names_model(ByteR& r, CanonicalMatrix& M) {
    auto& md = Models::get();
    uint8_t row_mode = r.u8();
    if (row_mode == 1) {
        uint64_t h = r.u64();
        auto j = md.joined(h);
        split_nul(j->data(), j->size(), M.m, M.rownames);
    } else {
        uint64_t raw_len = r.u64();
        auto [p, n] = r.blob();
        std::vector<uint8_t> raw(raw_len);
        ec_decompress(ec_zstd(19), p, n, raw.data(), raw_len);
        split_nul(raw.data(), raw.size(), M.m, M.rownames);
    }
    uint8_t bm = r.u8();
    uint32_t cnt = r.u32();
    M.colnames.clear(); M.colnames.reserve(cnt);
    if (bm == 0) {
        uint64_t raw_len = r.u64();
        auto [p, n] = r.blob();
        std::vector<uint8_t> raw(raw_len);
        ec_decompress(ec_zstd(19), p, n, raw.data(), raw_len);
        split_nul(raw.data(), raw.size(), cnt, M.colnames);
    } else if (bm == 1) {
        auto [p, n] = r.blob();
        if (n != size_t(cnt) * 4) throw std::runtime_error("track_a: packed barcode size");
        for (uint32_t i = 0; i < cnt; ++i) { uint32_t v; std::memcpy(&v, p + 4 * size_t(i), 4); M.colnames.push_back(bc_str(v)); }
    } else if (bm == 2) {
        uint8_t nl = r.u8();
        std::string wn(reinterpret_cast<const char*>(r.take(nl)), nl);
        auto* w = md.wl_by_name(wn);
        if (!w) throw std::runtime_error("track_a: whitelist model missing: " + wn);
        auto wc = md.codes(*w);
        auto [p, n] = r.blob();
        int bits = ceil_log2(wc->size());
        BitR br(p, n);
        for (uint32_t i = 0; i < cnt; ++i) M.colnames.push_back(bc_str((*wc)[br.get(bits)]));
    } else if (bm == 3) {
        auto [p, n] = r.blob();
        ByteR q(p, n);
        uint32_t nn = q.u32(); int k = q.u8(); uint32_t first = q.u32(); uint32_t rice_len = q.u32();
        const uint8_t* rice = q.take(rice_len);
        const uint8_t* pm = q.p; size_t pm_len = size_t(q.e - q.p);
        BitR br(rice, rice_len);
        std::vector<uint32_t> sorted(nn);
        if (nn) sorted[0] = first;
        for (uint32_t i = 1; i < nn; ++i) {
            uint64_t qv = 0;
            while (br.get(1) == 0) ++qv;
            uint64_t d = (qv << k) | (k ? br.get(k) : 0);
            sorted[i] = uint32_t(uint64_t(sorted[i - 1]) + d + 1);
        }
        int pb = ceil_log2(nn);
        BitR pr(pm, pm_len);
        for (uint32_t j = 0; j < cnt; ++j) M.colnames.push_back(bc_str(sorted[pr.get(pb)]));
    } else throw std::runtime_error("track_a: unknown barcode mode");
    auto [kp, kn] = r.blob();
    if (kn) {
        ZSTD_DCtx* d = ZSTD_createDCtx();
        unsigned long long csz = ZSTD_getFrameContentSize(kp, kn);
        std::vector<uint8_t> raw(static_cast<size_t>(csz));
        size_t z = ZSTD_decompressDCtx(d, raw.data(), raw.size(), kp, kn);
        ZSTD_freeDCtx(d);
        if (ZSTD_isError(z)) throw std::runtime_error("track_a: kv decode");
        std::vector<std::string> dummy_r, dummy_c;
        pz::parse_metadata_tlv(raw.data(), raw.size(), dummy_r, dummy_c, M.user_kv, 0, 0, 1u << 20);
    }
}

// ============================================================================
// Permutation section
// ============================================================================
// perm_mode 0: production (m u32, zstd-3).   perm_mode 1: per-row nnz counts, decoder re-derives the sort.
struct PermStats { size_t bytes = 0; int coder = 0; bool delta = false; };

inline std::vector<uint8_t> enc_perm(const Prepared& P, int mode, PermStats* ps = nullptr) {
    std::vector<uint8_t> o;
    ByteW w(o);
    if (mode == 0) {
        std::vector<uint8_t> raw(size_t(P.m) * 4);
        if (P.m) std::memcpy(raw.data(), P.perm.data(), raw.size());
        w.blob(z_compress(ec_zstd(3), raw));
        if (ps) { ps->bytes = o.size(); }
        return o;
    }
    // rowcounts: choose among {plain, zigzag-delta} x {zstd19, xz, brotli}
    std::vector<uint8_t> best; int bt = 0; bool bd = false;
    for (int delta = 0; delta < 2; ++delta) {
        std::vector<uint8_t> raw;
        int64_t prev = 0;
        for (uint32_t i = 0; i < P.m; ++i) {
            if (!delta) pz::varint_push(raw, P.rnnz[i]);
            else {
                int64_t d = int64_t(P.rnnz[i]) - prev; prev = P.rnnz[i];
                pz::varint_push(raw, uint32_t((d << 1) ^ (d >> 63)));
            }
        }
        for (uint8_t t = 0; t < 3; ++t) {
            auto z = z_compress(coder_for_tag(t), raw);
            if (best.empty() || z.size() + 8 < best.size()) {
                best.clear(); ByteW bw(best); bw.u64(raw.size()); bw.bytes(z.data(), z.size()); bt = t; bd = delta;
            }
        }
    }
    w.u8(uint8_t(bt | (bd ? 0x80 : 0)));
    w.blob(best);
    if (ps) { ps->bytes = o.size(); ps->coder = bt; ps->delta = bd; }
    return o;
}

inline std::vector<uint32_t> dec_perm(ByteR& r, int mode, uint32_t m) {
    std::vector<uint32_t> perm;
    if (mode == 0) {
        auto [p, n] = r.blob();
        std::vector<uint8_t> raw(size_t(m) * 4);
        ec_decompress(ec_zstd(3), p, n, raw.data(), raw.size());
        perm.resize(m);
        if (m) std::memcpy(perm.data(), raw.data(), raw.size());
        std::vector<uint8_t> seen(m, 0);
        for (uint32_t i = 0; i < m; ++i) {
            if (perm[i] >= m || seen[perm[i]]) throw std::runtime_error("track_a: perm invalid");
            seen[perm[i]] = 1;
        }
        return perm;
    }
    uint8_t tag = r.u8();
    auto [p, n] = r.blob();
    ByteR q(p, n);
    uint64_t raw_len = q.u64();
    std::vector<uint8_t> raw(raw_len);
    ec_decompress(coder_for_tag(tag & 0x7F), q.p, size_t(q.e - q.p), raw.data(), raw_len);
    std::vector<uint32_t> rnnz(m);
    size_t off = 0; int64_t prev = 0;
    for (uint32_t i = 0; i < m; ++i) {
        uint64_t v = varint_get(raw.data(), raw.size(), off);
        if (tag & 0x80) { int64_t d = int64_t(v >> 1) ^ -int64_t(v & 1); prev += d; rnnz[i] = uint32_t(prev); }
        else rnnz[i] = uint32_t(v);
    }
    return perm_from_rnnz(rnnz);
}

// ============================================================================
// Container
// ============================================================================
struct Plan {
    std::string name;
    EC ec = ec_zstd(3);
    int chunk_cols = 1024;
    int names_mode = 0;      // 0 production TLV, 1 model, 2 none (matrix-only, for measuring the payload)
    int perm_mode = 0;       // 0 production, 1 row counts
    int force_bc = -1;       // names_mode 1: force a barcode representation (-1 = smallest)
    bool derive_ptr = false; // omit the column-count block: it is the sum of the value-group counts in the chunk metadata
    int nthreads = 1;
};
struct Stats {
    size_t total = 0, names = 0, perm = 0, ptr = 0, ctable = 0, chunk_hdr = 0, chunk_payload = 0, fixed = 0;
    NamesStats ns; PermStats ps;
};

inline std::vector<uint8_t> encode_plan(const CanonicalMatrix& M, const Plan& pl, Stats* st = nullptr,
                                        const Prepared* pre = nullptr) {
    Prepared local;
    if (!pre) { local = prepare(M, pl.chunk_cols, nullptr, pl.nthreads); pre = &local; }
    const Prepared& P = *pre;
    std::vector<uint8_t> o;
    ByteW w(o);
    w.bytes("TAv1", 4);
    w.u8(pl.ec.kind); w.u8(P.vt); w.u8(P.gap16 ? 1 : 0); w.u8(pl.derive_ptr ? 0 : P.ptr_width);
    w.u32(M.m); w.u32(M.n); w.u64(M.nnz); w.u32(uint32_t(pl.chunk_cols)); w.u32(uint32_t(P.nc));
    w.u64(pl.ec.dict ? pl.ec.dict->id : 0);
    w.u8(uint8_t(pl.names_mode)); w.u8(uint8_t(pl.perm_mode));
    size_t fixed = o.size();
    // names
    size_t s0 = o.size();
    if (pl.names_mode == 2) { /* matrix-only stream: no names (measurement mode) */ }
    else if (pl.names_mode == 0) w.blob(enc_names_inline_tlv(M));
    else { auto ns = enc_names_model(M, o, pl.force_bc); if (st) st->ns = ns; }
    size_t names_bytes = o.size() - s0;
    // permutation
    s0 = o.size();
    auto pm = enc_perm(P, pl.perm_mode, st ? &st->ps : nullptr);
    w.bytes(pm.data(), pm.size());
    size_t perm_bytes = o.size() - s0;
    // column counts (ptr block), as production: ptr_width little-endian bytes per column, zstd-3
    s0 = o.size();
    if (!pl.derive_ptr) {
        std::vector<uint8_t> ptr_raw(size_t(M.n) * P.ptr_width);
        for (uint32_t j = 0; j < M.n; ++j)
            for (int b = 0; b < P.ptr_width; ++b) ptr_raw[size_t(j) * P.ptr_width + b] = uint8_t((P.cc[j] >> (8 * b)) & 0xFF);
        w.blob(z_compress(ec_zstd(3), ptr_raw));
    }
    size_t ptr_bytes = o.size() - s0;
    // chunks
    auto blobs = entropy_chunks(P, pl.ec);
    s0 = o.size();
    for (auto& b : blobs) w.u32(uint32_t(b.size()));
    size_t ctable_bytes = o.size() - s0;
    size_t payload = 0;
    for (auto& b : blobs) { w.bytes(b.data(), b.size()); payload += b.size() - 20; }
    if (st) {
        st->total = o.size(); st->fixed = fixed; st->names = names_bytes; st->perm = perm_bytes; st->ptr = ptr_bytes;
        st->ctable = ctable_bytes; st->chunk_hdr = 20 * blobs.size(); st->chunk_payload = payload;
    }
    return o;
}

inline CanonicalMatrix decode_plan(const std::vector<uint8_t>& b) {
    ByteR r(b.data(), b.size());
    if (std::memcmp(r.take(4), "TAv1", 4) != 0) throw std::runtime_error("track_a: bad magic");
    EC ec; ec.kind = r.u8();
    CanonicalMatrix M;
    M.vt_code = r.u8();
    bool gap16 = r.u8() != 0;
    uint8_t ptr_width = r.u8();
    M.m = r.u32(); M.n = r.u32(); M.nnz = r.u64();
    uint32_t chunk_cols = r.u32(), nc = r.u32();
    uint64_t dict_id = r.u64();
    if (dict_id) {
        auto it = dict_registry().find(dict_id);
        if (it == dict_registry().end()) throw std::runtime_error("track_a: dictionary model missing");
        ec.dict = it->second;
    }
    uint8_t names_mode = r.u8(), perm_mode = r.u8();
    // names (decoded last into M but parsed here)
    ByteR names_r = r;
    if (names_mode == 2) { /* no names stored */ }
    else if (names_mode == 0) { r.blob(); }
    else {
        // advance r past the names section by decoding it into a scratch matrix (also validates)
        CanonicalMatrix tmp; tmp.m = M.m;
        dec_names_model(r, tmp);
        M.rownames = std::move(tmp.rownames); M.colnames = std::move(tmp.colnames); M.user_kv = std::move(tmp.user_kv);
    }
    if (names_mode == 0) {
        auto [p, n] = names_r.blob();
        if (n) {
            unsigned long long csz = ZSTD_getFrameContentSize(p, n);
            std::vector<uint8_t> raw(static_cast<size_t>(csz));
            thread_local ZSTD_DCtx* dctx = ZSTD_createDCtx();
            size_t z = ZSTD_decompressDCtx(dctx, raw.data(), raw.size(), p, n);
            if (ZSTD_isError(z)) throw std::runtime_error("track_a: metadata decode");
            pz::parse_metadata_tlv(raw.data(), raw.size(), M.rownames, M.colnames, M.user_kv, M.m, M.n, 1u << 20);
        }
    }
    std::vector<uint32_t> perm = dec_perm(r, perm_mode, M.m);
    std::vector<uint32_t> cc;
    if (ptr_width != 0) {   // ptr_width 0 = no ptr block, column lengths are derived while decoding the chunks
        auto [pp, pn] = r.blob();
        std::vector<uint8_t> ptr_raw(size_t(M.n) * ptr_width);
        ec_decompress(ec_zstd(3), pp, pn, ptr_raw.data(), ptr_raw.size());
        cc.resize(M.n);
        for (uint32_t j = 0; j < M.n; ++j) {
            uint32_t v = 0;
            for (int k = 0; k < ptr_width; ++k) v |= uint32_t(ptr_raw[size_t(j) * ptr_width + k]) << (8 * k);
            cc[j] = v;
        }
        M.indptr.assign(size_t(M.n) + 1, 0);
        uint64_t run = 0;
        for (uint32_t j = 0; j < M.n; ++j) { M.indptr[j] = uint32_t(run); run += cc[j]; }
        M.indptr[M.n] = uint32_t(run);
        if (run != M.nnz) throw std::runtime_error("track_a: nnz mismatch");
    }
    std::vector<uint32_t> ctable(nc);
    for (uint32_t c = 0; c < nc; ++c) ctable[c] = r.u32();
    const uint8_t* base = r.p;
    decode_chunks(base, ctable, ec, M.m, M.n, M.nnz, int(chunk_cols), gap16, M.vt_code, perm, cc, M);
    return M;
}

inline Codec make_plan_codec(const Plan& pl, size_t model_bytes = 0) {
    Codec c;
    c.name = pl.name;
    c.model_bytes = model_bytes;
    c.encode = [pl](const CanonicalMatrix& M) { return encode_plan(M, pl); };
    c.decode = [](const std::vector<uint8_t>& b) { return decode_plan(b); };
    return c;
}

// ---- dictionary support (trained by harness/track_a_dict.cpp, looked up by matrix fingerprint) ----
inline std::string fingerprint(const CanonicalMatrix& M) {
    return std::to_string(M.m) + "x" + std::to_string(M.n) + "x" + std::to_string(M.nnz);
}
inline std::shared_ptr<DictPair> load_dict_for(const CanonicalMatrix& M) {
    // results/track_a_dict_manifest.tsv : fingerprint \t dict file
    const char* mf = std::getenv("TRACK_A_DICT_MANIFEST");
    std::ifstream f(Models::dir() + "/" + (mf ? mf : "track_a_dict_manifest.tsv"));
    std::string line, fp = fingerprint(M);
    while (std::getline(f, line)) {
        auto t = line.find('\t');
        if (t == std::string::npos || line.substr(0, t) != fp) continue;
        auto bytes = read_file(Models::dir() + "/" + line.substr(t + 1));
        auto dp = std::make_shared<DictPair>(bytes, 19);
        dict_registry()[dp->id] = dp;
        return dp;
    }
    return nullptr;
}

inline void register_track_a(std::vector<Codec>& v) {
    size_t mb = Models::get().model_bytes_total();
    {   // names to the model side, barcodes in their cheapest representation, matrix payload = production bytes
        Plan p; p.name = "a_names_as_model"; p.names_mode = 1; p.perm_mode = 0;
        v.push_back(make_plan_codec(p, mb));
    }
    {   // permutation re-derived from per-row nnz counts; names/barcodes as production
        Plan p; p.name = "a_perm_from_rowcounts"; p.names_mode = 0; p.perm_mode = 1;
        v.push_back(make_plan_codec(p));
    }
    {   // both, i.e. everything Track A can take out without touching the transform
        Plan p; p.name = "a_names_perm_model"; p.names_mode = 1; p.perm_mode = 1; p.derive_ptr = true;
        v.push_back(make_plan_codec(p, mb));
    }
    // entropy-coder swaps on the unchanged prefilter bytes; names/perm exactly as production
    {
        Plan p; p.name = "a_vocsc_zstd19"; p.ec = ec_zstd(19);
        v.push_back(make_plan_codec(p));
    }
    {
        Plan p; p.name = "a_vocsc_zstd22long"; p.ec = ec_zstd(22, 27, true);
        v.push_back(make_plan_codec(p));
    }
    {
        Plan p; p.name = "a_vocsc_xz"; p.ec = ec_xz();
        v.push_back(make_plan_codec(p));
    }
    {
        Plan p; p.name = "a_vocsc_brotli"; p.ec = ec_brotli(11, 24);
        v.push_back(make_plan_codec(p));
    }
    {   // dictionary trained on other studies (see track_a_dict); falls back to level 19 without one
        Codec c;
        c.name = "a_vocsc_zstd_dict";
        c.model_bytes = 1u << 20;
        c.encode = [](const CanonicalMatrix& M) {
            Plan p; p.name = "a_vocsc_zstd_dict";
            p.ec = ec_zstd(19);
            p.ec.dict = load_dict_for(M);
            return encode_plan(M, p);
        };
        c.decode = [](const std::vector<uint8_t>& b) { return decode_plan(b); };
        v.push_back(c);
    }
}

}  // namespace ta

inline void register_track_a(std::vector<Codec>& v) { ta::register_track_a(v); }

}  // namespace cs
