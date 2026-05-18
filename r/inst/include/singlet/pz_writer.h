// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: pz_writer.h
// Native .1pz (VOCSC) writer — format-compatible with singlepress pz_codec.cpp
//
// Single function: write_1pz<ValT>(path, csc, rownames, colnames, opts)
// Produces bit-exact files readable by singlepress.read().

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <zstd.h>

namespace singlet {
namespace pz {

// ============================================================================
// Constants (must match pz_codec.cpp exactly)
// ============================================================================
static constexpr uint32_t TP1_MAGIC   = 0x5A315054;  // "TP1Z"
static constexpr uint16_t TP1_VERSION = 1;

// PZHeader.flags
static constexpr uint8_t FLAG_HAS_PERM      = 0x01;
static constexpr uint8_t FLAG_GAP16         = 0x02;
static constexpr uint8_t FLAG_HAS_METADATA  = 0x04;
static constexpr uint8_t FLAG_HAS_COLSUMS   = 0x10;

// PZHeader.feature_flags
static constexpr uint32_t FEAT_ZSTD_CHECKSUMS  = 0x01;
static constexpr uint32_t FEAT_BITPLANE_BITMAP = 0x02;

// Metadata tags
static constexpr uint8_t META_TAG_END      = 0;
static constexpr uint8_t META_TAG_ROWNAMES = 1;
static constexpr uint8_t META_TAG_COLNAMES = 2;
static constexpr uint8_t META_TAG_USER_KV  = 3;  ///< Null-delimited key\0value\0 pairs

// ============================================================================
// Structs (must match pz_codec.cpp layout exactly)
// ============================================================================
#pragma pack(push, 1)
struct PZHeader {
    uint32_t magic;            // 0
    uint16_t version;          // 4
    uint8_t  vt_code;          // 6
    uint8_t  flags;            // 7
    uint32_t m;                // 8
    uint32_t n;                // 12
    uint64_t nnz;              // 16
    uint8_t  ptr_width;        // 24
    uint8_t  codec_level;      // 25
    uint16_t _pad0;            // 26
    uint32_t num_chunks;       // 28
    uint32_t perm_z_sz;        // 32
    uint32_t ptr_z_sz;         // 36
    uint32_t chunk_cols;       // 40
    uint32_t feature_flags;    // 44
    uint64_t metadata_offset;  // 48
    uint32_t metadata_z_sz;    // 56
    uint32_t colsums_z_sz;     // 60
    uint64_t transpose_offset; // 64
    uint32_t transpose_z_sz;   // 72
    uint32_t transpose_chunks; // 76
    uint8_t  reserved[16];     // 80-95
};
static_assert(sizeof(PZHeader) == 96, "PZHeader must be 96 bytes");

struct PZFooter {
    uint32_t file_crc32;
    uint32_t _reserved;
    uint32_t num_chunks;
    uint32_t magic;
};
static_assert(sizeof(PZFooter) == 16, "PZFooter must be 16 bytes");
#pragma pack(pop)

// ============================================================================
// CRC32 (ISO 3309, slicing-by-8 — matches pz_codec.cpp)
// ============================================================================
struct CRC32 {
    uint32_t state_;
    CRC32() : state_(0xFFFFFFFF) {}
    void update(const uint8_t* data, size_t len) {
        static const auto& T = get_table8();
        uint32_t crc = state_;
        while (len >= 8) {
            uint32_t a, b;
            std::memcpy(&a, data, 4);
            std::memcpy(&b, data + 4, 4);
            a ^= crc;
            crc = T[7][a & 0xFF] ^ T[6][(a >> 8) & 0xFF] ^
                  T[5][(a >> 16) & 0xFF] ^ T[4][(a >> 24) & 0xFF] ^
                  T[3][b & 0xFF] ^ T[2][(b >> 8) & 0xFF] ^
                  T[1][(b >> 16) & 0xFF] ^ T[0][(b >> 24) & 0xFF];
            data += 8; len -= 8;
        }
        while (len--) crc = (crc >> 8) ^ T[0][(crc ^ *data++) & 0xFF];
        state_ = crc;
    }
    void update(const void* data, size_t len) {
        update(reinterpret_cast<const uint8_t*>(data), len);
    }
    uint32_t finalize() const { return state_ ^ 0xFFFFFFFF; }
    static uint32_t compute(const void* data, size_t len) {
        CRC32 c; c.update(data, len); return c.finalize();
    }
private:
    static const uint32_t (&get_table8())[8][256] {
        static uint32_t T[8][256]{};
        static bool init = false;
        if (!init) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j)
                    c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                T[0][i] = c;
            }
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = T[0][i];
                for (int k = 1; k < 8; ++k) {
                    c = (c >> 8) ^ T[0][c & 0xFF];
                    T[k][i] = c;
                }
            }
            init = true;
        }
        return T;
    }
};

// ============================================================================
// ZSTD RAII + helpers
// ============================================================================
struct ZstdCCtx {
    ZSTD_CCtx* ctx;
    ZstdCCtx() : ctx(ZSTD_createCCtx()) {}
    ~ZstdCCtx() { if (ctx) ZSTD_freeCCtx(ctx); }
    ZstdCCtx(const ZstdCCtx&) = delete;
    ZstdCCtx& operator=(const ZstdCCtx&) = delete;
};

static inline std::vector<uint8_t> zstd_compress(ZSTD_CCtx* ctx,
                                                  const uint8_t* data, size_t sz,
                                                  int level) {
    ZSTD_CCtx_reset(ctx, ZSTD_reset_session_only);
    ZSTD_CCtx_setParameter(ctx, ZSTD_c_compressionLevel, level);
    ZSTD_CCtx_setParameter(ctx, ZSTD_c_checksumFlag, 1);
    size_t bound = ZSTD_compressBound(sz);
    std::vector<uint8_t> out(bound);
    size_t csz = ZSTD_compress2(ctx, out.data(), bound, data, sz);
    if (ZSTD_isError(csz))
        throw std::runtime_error(std::string("ZSTD compress: ") + ZSTD_getErrorName(csz));
    out.resize(csz);
    return out;
}

// ============================================================================
// Varint (LEB128) — matches pz_codec.cpp
// ============================================================================
static inline void varint_push(std::vector<uint8_t>& out, uint32_t v) {
    while (v >= 0x80) { out.push_back((v & 0x7F) | 0x80); v >>= 7; }
    out.push_back(v);
}

// ============================================================================
// Byte-split (scalar, 4x unrolled — matches pz_codec.cpp)
// ============================================================================
static void byte_split_16(const uint32_t* __restrict__ src, size_t n,
                           uint8_t* __restrict__ dst) {
    uint8_t* p0 = dst;
    uint8_t* p1 = dst + n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t a = src[i], b = src[i+1], c = src[i+2], d = src[i+3];
        p0[i]   = (uint8_t)(a);       p0[i+1] = (uint8_t)(b);
        p0[i+2] = (uint8_t)(c);       p0[i+3] = (uint8_t)(d);
        p1[i]   = (uint8_t)(a >> 8);  p1[i+1] = (uint8_t)(b >> 8);
        p1[i+2] = (uint8_t)(c >> 8);  p1[i+3] = (uint8_t)(d >> 8);
    }
    for (; i < n; ++i) {
        p0[i] = (uint8_t)(src[i]);
        p1[i] = (uint8_t)(src[i] >> 8);
    }
}

static void byte_split_32(const uint32_t* __restrict__ src, size_t n,
                           uint8_t* __restrict__ dst) {
    uint8_t *p0 = dst, *p1 = dst+n, *p2 = dst+2*n, *p3 = dst+3*n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        uint32_t a = src[i], b = src[i+1], c = src[i+2], d = src[i+3];
        p0[i]=(uint8_t)a;        p0[i+1]=(uint8_t)b;        p0[i+2]=(uint8_t)c;        p0[i+3]=(uint8_t)d;
        p1[i]=(uint8_t)(a>>8);   p1[i+1]=(uint8_t)(b>>8);   p1[i+2]=(uint8_t)(c>>8);   p1[i+3]=(uint8_t)(d>>8);
        p2[i]=(uint8_t)(a>>16);  p2[i+1]=(uint8_t)(b>>16);  p2[i+2]=(uint8_t)(c>>16);  p2[i+3]=(uint8_t)(d>>16);
        p3[i]=(uint8_t)(a>>24);  p3[i+1]=(uint8_t)(b>>24);  p3[i+2]=(uint8_t)(c>>24);  p3[i+3]=(uint8_t)(d>>24);
    }
    for (; i < n; ++i) {
        uint32_t v = src[i];
        p0[i]=(uint8_t)v; p1[i]=(uint8_t)(v>>8); p2[i]=(uint8_t)(v>>16); p3[i]=(uint8_t)(v>>24);
    }
}

// ============================================================================
// Bit-plane decomposition (plane 0 transform — matches pz_codec.cpp)
// ============================================================================
static void bit_planes_encode(const uint8_t* __restrict__ src, size_t n,
                               uint8_t* __restrict__ dst) {
    size_t pb = (n + 7) / 8;
    std::memset(dst, 0, 8 * pb);
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = src[i];
        if (v == 0) continue;
        size_t byte_idx = i / 8;
        uint8_t bit_mask = 1u << (i & 7);
        for (int b = 0; b < 8; ++b) {
            if (v & (1 << b))
                dst[b * pb + byte_idx] |= bit_mask;
        }
    }
}

// ============================================================================
// Bitmap zero-filter — matches pz_codec.cpp
// ============================================================================
static size_t bitmap_pack(const uint8_t* __restrict__ src, size_t n,
                           uint8_t* __restrict__ dst) {
    size_t bm_bytes = (n + 7) / 8;
    std::memset(dst, 0, bm_bytes);
    size_t wp = bm_bytes;
    for (size_t i = 0; i < n; ++i) {
        if (src[i] != 0) {
            dst[i / 8] |= (1u << (i & 7));
            dst[wp++] = src[i];
        }
    }
    return wp;
}

// ============================================================================
// Non-owning CSC view
// ============================================================================
struct CSCView {
    uint32_t m, n;
    uint64_t nnz;
    const uint32_t* p;
    const uint32_t* i;
    const uint32_t* x;  // values cast to uint32
};

// ============================================================================
// VOCSC encode one chunk — matches pz_codec.cpp exactly
// ============================================================================
struct VocscResult {
    std::vector<uint32_t> gaps;
    std::vector<uint8_t> metadata;
    size_t n_gaps;
    uint32_t max_gap;
};

static VocscResult vocsc_encode_chunk(
    const CSCView& mat, const uint32_t* inv_perm,
    const uint32_t* col_counts,
    int64_t col_start, int64_t col_end)
{
    VocscResult r;
    auto& gaps = r.gaps;
    auto& meta = r.metadata;
    r.max_gap = 0;

    uint64_t cnnz = 0;
    for (int64_t j = col_start; j < col_end; ++j) cnnz += col_counts[j];
    gaps.reserve(cnnz);
    meta.reserve((col_end - col_start) * 8);

    std::vector<std::vector<uint32_t>> val_rows;
    std::vector<uint32_t> active_vals;

    for (int64_t j = col_start; j < col_end; ++j) {
        uint32_t cnt = col_counts[j];
        if (cnt == 0) { varint_push(meta, 0); continue; }
        uint64_t s = mat.p[j];
        if (s + cnt > mat.nnz) {
            cnt = (s < mat.nnz) ? (uint32_t)(mat.nnz - s) : 0;
            if (cnt == 0) { varint_push(meta, 0); continue; }
        }

        active_vals.clear();
        uint32_t max_val = 0;
        for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t v = mat.x[s+k];
            if (v > max_val) max_val = v;
        }
        if (val_rows.size() <= max_val) val_rows.resize(max_val + 1);

        for (uint32_t k = 0; k < cnt; ++k) {
            uint32_t v = mat.x[s+k];
            uint32_t prow = inv_perm[mat.i[s+k]];
            if (val_rows[v].empty()) active_vals.push_back(v);
            val_rows[v].push_back(prow);
        }

        std::sort(active_vals.begin(), active_vals.end());

        uint32_t ng = (uint32_t)active_vals.size();
        varint_push(meta, ng);

        for (uint32_t vi = 0; vi < ng; ++vi) {
            uint32_t cv = active_vals[vi];
            auto& rows = val_rows[cv];
            std::sort(rows.begin(), rows.end());

            varint_push(meta, cv);
            varint_push(meta, (uint32_t)rows.size());
            uint32_t prev = 0;
            for (uint32_t row : rows) {
                uint32_t gap = row - prev;
                if (gap > r.max_gap) r.max_gap = gap;
                gaps.push_back(gap);
                prev = row + 1;
            }
            rows.clear();
        }
    }
    r.n_gaps = gaps.size();
    return r;
}

// ============================================================================
// Encoded section (output of full VOCSC encoding)
// ============================================================================
struct EncodedSection {
    std::vector<uint8_t> perm_z;
    std::vector<uint8_t> ptr_z;
    std::vector<uint32_t> ctable;
    std::vector<std::vector<uint8_t>> blobs;
    bool gap16;
    uint8_t ptr_width;
    int num_chunks;
};

// ============================================================================
// encode_vocsc_section — matches pz_codec.cpp
// ============================================================================
static EncodedSection encode_vocsc_section(
    const uint32_t* p_ptr, const uint32_t* i_ptr, const uint32_t* x_ptr,
    uint32_t m, uint32_t n, uint64_t nnz,
    int chunk_cols, int level, int nt)
{
    EncodedSection sec;

    // Row frequency-sort permutation (counting sort, descending)
    std::vector<int32_t> rnnz(m, 0);
    for (uint64_t k = 0; k < nnz; ++k) rnnz[i_ptr[k]]++;

    int32_t max_rnnz = 0;
    for (uint32_t i = 0; i < m; ++i)
        if (rnnz[i] > max_rnnz) max_rnnz = rnnz[i];

    std::vector<uint32_t> bucket_count(max_rnnz + 1, 0);
    for (uint32_t i = 0; i < m; ++i) bucket_count[rnnz[i]]++;

    std::vector<uint32_t> bucket_pos(max_rnnz + 1);
    {
        uint32_t pos = 0;
        for (int32_t b = max_rnnz; b >= 0; --b) {
            bucket_pos[b] = pos;
            pos += bucket_count[b];
        }
    }

    std::vector<uint32_t> perm(m);
    for (uint32_t i = 0; i < m; ++i)
        perm[bucket_pos[rnnz[i]]++] = i;

    std::vector<uint32_t> inv_perm(m);
    for (uint32_t i = 0; i < m; ++i) inv_perm[perm[i]] = i;

    // Column counts + ptr_width
    std::vector<uint32_t> cc(n, 0);
    for (uint32_t j = 0; j < n; ++j) {
        int64_t d = (int64_t)p_ptr[j+1] - (int64_t)p_ptr[j];
        cc[j] = d > 0 ? (uint32_t)d : 0;
    }
    bool use16p = true;
    for (uint32_t j = 0; j < n; ++j)
        if (cc[j] > 65535) { use16p = false; break; }
    sec.ptr_width = use16p ? 2 : 4;

    uint64_t tnnz = 0;
    for (uint32_t j = 0; j < n; ++j) tnnz += cc[j];

    CSCView mat;
    mat.m = m; mat.n = n; mat.nnz = tnnz;
    mat.p = p_ptr; mat.i = i_ptr; mat.x = x_ptr;

    // Compress perm and ptrs
    std::vector<uint8_t> perm_raw(m * 4);
    std::memcpy(perm_raw.data(), perm.data(), m * 4);
    std::vector<uint8_t> ptr_raw(n * sec.ptr_width);
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t d = cc[j];
        for (int b = 0; b < sec.ptr_width; ++b)
            ptr_raw[j * sec.ptr_width + b] = (d >> (8*b)) & 0xFF;
    }
    ZstdCCtx cctx;
    sec.perm_z = zstd_compress(cctx.ctx, perm_raw.data(), perm_raw.size(), 3);
    sec.ptr_z  = zstd_compress(cctx.ctx, ptr_raw.data(), ptr_raw.size(), 3);

    int nc = ((int)n + chunk_cols - 1) / chunk_cols;
    sec.num_chunks = nc;
    sec.blobs.resize(nc);
    bool any_gap32 = false;

    // Pass 1: VOCSC encode (threaded)
    struct ChunkData { VocscResult vocsc; };
    std::vector<ChunkData> cdata(nc);

    // Use std::thread pool for chunk encoding
    auto encode_chunks = [&](int start, int end) {
        for (int c = start; c < end; ++c) {
            int64_t cs = (int64_t)c * chunk_cols;
            int64_t ce = std::min(cs + (int64_t)chunk_cols, (int64_t)n);
            cdata[c].vocsc = vocsc_encode_chunk(mat, inv_perm.data(), cc.data(), cs, ce);
        }
    };

    if (nt <= 1 || nc <= 1) {
        encode_chunks(0, nc);
    } else {
        int nw = std::min(nt, nc);
        std::vector<std::thread> threads;
        int per = nc / nw;
        int rem = nc % nw;
        int start = 0;
        for (int w = 0; w < nw; ++w) {
            int cnt = per + (w < rem ? 1 : 0);
            threads.emplace_back(encode_chunks, start, start + cnt);
            start += cnt;
        }
        for (auto& t : threads) t.join();
    }

    for (int c = 0; c < nc; ++c)
        if (cdata[c].vocsc.max_gap >= 65536) any_gap32 = true;

    sec.gap16 = !any_gap32;
    int gw = sec.gap16 ? 2 : 4;

    // Pass 2: byte-split + bit-plane(p0) + bitmap + compress + per-chunk CRC (threaded)
    auto compress_chunks = [&](int start, int end) {
        ZstdCCtx tcctx;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> prefilter;
        std::vector<uint8_t> packed;

        for (int c = start; c < end; ++c) {
            auto& v = cdata[c].vocsc;
            size_t ng = v.n_gaps;
            size_t msz = v.metadata.size();
            size_t plane_sz = ng * gw;
            size_t raw_sz = msz + plane_sz;

            raw.resize(raw_sz);
            std::memcpy(raw.data(), v.metadata.data(), msz);
            if (ng > 0) {
                if (sec.gap16)
                    byte_split_16(v.gaps.data(), ng, raw.data() + msz);
                else
                    byte_split_32(v.gaps.data(), ng, raw.data() + msz);
            }

            // Bit-plane decompose plane 0, then bitmap zero-filter
            size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
            size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
            prefilter.resize(pf_sz);

            std::memcpy(prefilter.data(), raw.data(), msz);
            if (ng > 0) {
                bit_planes_encode(raw.data() + msz, ng, prefilter.data() + msz);
                if (gw > 1)
                    std::memcpy(prefilter.data() + msz + bp_sz,
                                raw.data() + msz + ng, ng * (gw - 1));
            }

            packed.resize(pf_sz + (pf_sz + 7) / 8);
            size_t packed_sz = bitmap_pack(prefilter.data(), pf_sz, packed.data());

            uint32_t chunk_crc = CRC32::compute(packed.data(), packed_sz);
            auto z = zstd_compress(tcctx.ctx, packed.data(), packed_sz, level);

            uint32_t ng32  = (uint32_t)ng;
            uint32_t msz32 = (uint32_t)msz;
            uint32_t zsz   = (uint32_t)z.size();
            uint32_t psz32 = (uint32_t)packed_sz;

            auto& blob = sec.blobs[c];
            blob.resize(20 + z.size());
            uint8_t* wp = blob.data();
            std::memcpy(wp, &ng32, 4);       wp += 4;
            std::memcpy(wp, &msz32, 4);      wp += 4;
            std::memcpy(wp, &zsz, 4);        wp += 4;
            std::memcpy(wp, &chunk_crc, 4);  wp += 4;
            std::memcpy(wp, &psz32, 4);      wp += 4;
            std::memcpy(wp, z.data(), z.size());

            v.gaps.clear(); v.gaps.shrink_to_fit();
            v.metadata.clear(); v.metadata.shrink_to_fit();
        }
    };

    if (nt <= 1 || nc <= 1) {
        compress_chunks(0, nc);
    } else {
        int nw = std::min(nt, nc);
        std::vector<std::thread> threads;
        int per = nc / nw;
        int rem = nc % nw;
        int start = 0;
        for (int w = 0; w < nw; ++w) {
            int cnt = per + (w < rem ? 1 : 0);
            threads.emplace_back(compress_chunks, start, start + cnt);
            start += cnt;
        }
        for (auto& t : threads) t.join();
    }

    sec.ctable.resize(nc);
    for (int c = 0; c < nc; ++c) sec.ctable[c] = (uint32_t)sec.blobs[c].size();
    return sec;
}

// ============================================================================
// Metadata serialization
// ============================================================================
static void push_strings_tlv(std::vector<uint8_t>& out, uint8_t tag,
                             const std::vector<std::string>& names) {
    if (names.empty()) return;
    size_t data_sz = 0;
    for (const auto& s : names) data_sz += s.size() + 1;
    out.push_back(tag);
    uint32_t sz = (uint32_t)data_sz;
    out.insert(out.end(), (uint8_t*)&sz, (uint8_t*)&sz + 4);
    for (const auto& s : names) {
        out.insert(out.end(), s.begin(), s.end());
        out.push_back(0);
    }
}

/// Serialize a string→string map as META_TAG_USER_KV TLV entry.
/// Layout: tag(1) + length(4) + [key\0value\0]...
static void push_kv_tlv(std::vector<uint8_t>& out,
                        const std::map<std::string, std::string>& kv) {
    if (kv.empty()) return;
    size_t data_sz = 0;
    for (const auto& p : kv) data_sz += p.first.size() + 1 + p.second.size() + 1;
    out.push_back(META_TAG_USER_KV);
    uint32_t sz = (uint32_t)data_sz;
    out.insert(out.end(), (uint8_t*)&sz, (uint8_t*)&sz + 4);
    for (const auto& p : kv) {
        out.insert(out.end(), p.first.begin(), p.first.end());
        out.push_back(0);
        out.insert(out.end(), p.second.begin(), p.second.end());
        out.push_back(0);
    }
}

// ============================================================================
// write_1pz — main entry point
// ============================================================================
// Writes a CSC sparse matrix as a .1pz file compatible with singlepress.
// ValT should be uint8_t, uint16_t, or uint32_t.
//
// indptr/indices use int32_t (from SparseAccumulator); cast internally to uint32.
template<typename ValT>
bool write_1pz(
    const std::string& path,
    uint32_t n_features,
    uint32_t n_barcodes,
    const std::vector<int32_t>& indptr,
    const std::vector<int32_t>& indices,
    const std::vector<ValT>& data,
    const std::vector<std::string>& rownames,
    const std::vector<std::string>& colnames,
    int zstd_level = 3,
    int chunk_cols = 1024,
    int threads = 4,
    const std::map<std::string, std::string>& user_meta = {})
{
    uint64_t nnz = data.size();
    uint32_t m = n_features;
    uint32_t n = n_barcodes;

    // Cast int32 → uint32 for indptr/indices/data
    std::vector<uint32_t> u_indptr(indptr.begin(), indptr.end());
    std::vector<uint32_t> u_indices(indices.begin(), indices.end());
    std::vector<uint32_t> u_data(nnz);
    uint32_t mx = 0;
    for (uint64_t k = 0; k < nnz; ++k) {
        u_data[k] = static_cast<uint32_t>(data[k]);
        if (u_data[k] > mx) mx = u_data[k];
    }
    int vt_code = (mx <= 255) ? 1 : (mx <= 65535) ? 2 : 3;

    // 1. Encode VOCSC section
    auto fwd = encode_vocsc_section(
        u_indptr.data(), u_indices.data(), u_data.data(),
        m, n, nnz, chunk_cols, zstd_level, threads);

    // 2. Compute column sums
    std::vector<uint64_t> colsums(n, 0);
    for (uint32_t j = 0; j < n; ++j) {
        uint64_t s = 0;
        for (int64_t k = indptr[j]; k < indptr[j+1]; ++k)
            s += static_cast<uint64_t>(data[k]);
        colsums[j] = s;
    }

    ZstdCCtx cctx;
    auto colsums_z = zstd_compress(cctx.ctx,
        (const uint8_t*)colsums.data(), n * 8, 3);

    // 3. Serialize metadata (TLV)
    std::vector<uint8_t> meta_raw;
    bool has_meta = !rownames.empty() || !colnames.empty() || !user_meta.empty();
    if (has_meta) {
        push_strings_tlv(meta_raw, META_TAG_ROWNAMES, rownames);
        push_strings_tlv(meta_raw, META_TAG_COLNAMES, colnames);
        push_kv_tlv(meta_raw, user_meta);
        meta_raw.push_back(META_TAG_END);
    }
    auto meta_z = has_meta ?
        zstd_compress(cctx.ctx, meta_raw.data(), meta_raw.size(), 3) :
        std::vector<uint8_t>{};

    // 4. Compute offsets (sequential layout matching pz_codec.cpp)
    size_t off = 96;  // header
    off += fwd.perm_z.size() + fwd.ptr_z.size();
    off += fwd.num_chunks * 4;  // chunk table
    for (int c = 0; c < fwd.num_chunks; ++c) off += fwd.blobs[c].size();
    size_t colsums_offset = off; off += colsums_z.size();
    size_t metadata_offset = off; off += meta_z.size();
    size_t footer_offset = off;

    // 5. Build header
    PZHeader hdr{};
    hdr.magic = TP1_MAGIC;
    hdr.version = TP1_VERSION;
    hdr.vt_code = (uint8_t)vt_code;
    hdr.flags = FLAG_HAS_PERM | FLAG_HAS_COLSUMS;
    if (fwd.gap16) hdr.flags |= FLAG_GAP16;
    if (has_meta) hdr.flags |= FLAG_HAS_METADATA;
    hdr.m = m;
    hdr.n = n;
    hdr.nnz = nnz;
    hdr.ptr_width = fwd.ptr_width;
    hdr.codec_level = (uint8_t)zstd_level;
    hdr.num_chunks = (uint32_t)fwd.num_chunks;
    hdr.perm_z_sz = (uint32_t)fwd.perm_z.size();
    hdr.ptr_z_sz = (uint32_t)fwd.ptr_z.size();
    hdr.chunk_cols = (uint32_t)chunk_cols;
    hdr.feature_flags = FEAT_ZSTD_CHECKSUMS | FEAT_BITPLANE_BITMAP;
    hdr.metadata_offset = has_meta ? metadata_offset : 0;
    hdr.metadata_z_sz = (uint32_t)meta_z.size();
    hdr.colsums_z_sz = (uint32_t)colsums_z.size();
    hdr.transpose_offset = 0;
    hdr.transpose_z_sz = 0;
    hdr.transpose_chunks = 0;
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));

    // 6. Assemble file buffer
    std::vector<uint8_t> buf(footer_offset);
    size_t wp = 0;
    auto append = [&](const void* data_ptr, size_t sz) {
        std::memcpy(buf.data() + wp, data_ptr, sz);
        wp += sz;
    };
    append(&hdr, 96);
    append(fwd.perm_z.data(), fwd.perm_z.size());
    append(fwd.ptr_z.data(), fwd.ptr_z.size());
    append(fwd.ctable.data(), fwd.num_chunks * 4);
    for (int c = 0; c < fwd.num_chunks; ++c)
        append(fwd.blobs[c].data(), fwd.blobs[c].size());
    append(colsums_z.data(), colsums_z.size());
    if (has_meta) append(meta_z.data(), meta_z.size());

    // 7. Compute file CRC and write
    uint32_t file_crc = CRC32::compute(buf.data(), buf.size());
    PZFooter ftr{};
    ftr.file_crc32 = file_crc;
    ftr._reserved = 0;
    ftr.num_chunks = (uint32_t)fwd.num_chunks;
    ftr.magic = TP1_MAGIC;

    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        std::cerr << "[pz_writer] Cannot open: " << path << "\n";
        return false;
    }
    fout.write(reinterpret_cast<const char*>(buf.data()), buf.size());
    fout.write(reinterpret_cast<const char*>(&ftr), 16);
    fout.close();

    size_t total_bytes = footer_offset + 16;
    std::cerr << "[pz_writer] Wrote " << path << ": "
              << m << "×" << n << ", " << nnz << " nnz, "
              << total_bytes << " bytes (vt=" << vt_code << ")\n";
    return true;
}

// ============================================================================
// Convenience: write CSCMatrix directly
// ============================================================================
template<typename ValT>
bool write_1pz_csc(
    const std::string& path,
    const typename singlet::SparseAccumulator<ValT>::CSCMatrix& csc,
    const std::vector<std::string>& rownames,
    const std::vector<std::string>& colnames,
    int zstd_level = 3,
    int chunk_cols = 1024,
    int threads = 4,
    const std::map<std::string, std::string>& user_meta = {})
{
    return write_1pz<ValT>(path, csc.nrows, csc.ncols,
                           csc.indptr, csc.indices, csc.data,
                           rownames, colnames,
                           zstd_level, chunk_cols, threads, user_meta);
}

}  // namespace pz
}  // namespace singlet
