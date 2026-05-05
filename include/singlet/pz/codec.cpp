/**
 * pz_codec.cpp — .1pz codec for singlepress package
 *
 * Format: VOCSC + adaptive byte-split + zstd-3
 *   - 13x compression vs raw CSC at 200+ MB/s encode, 4000+ MB/s decode
 *   - CRC32 integrity (per-chunk + whole-file)
 *   - Metadata: gene names, cell barcodes, key-value pairs
 *   - Column sums for on-demand log-normalization
 *   - Column-range reads for streaming/partial access
 *   - Optional transpose storage for row-range reads
 *   - Native int32 return path
 *   - SSE2-accelerated byte-unsplit
 *
 * Python API:
 *   pz_write(indptr, indices, data, nrows, path, ...) -> dict
 *   pz_write_int(indptr, indices, data_int, nrows, path, ...) -> dict
 *   pz_read(path) -> dict {m, n, nnz, indptr, indices, values, rownames, colnames, colsums}
 *   pz_read_int(path) -> dict (values as int32)
 *   pz_read_columns(path, col_start, col_end) -> dict (partial read)
 *   pz_info(path) -> dict
 *   pz_validate(path) -> dict
 *   pz_colsums(path) -> numpy array
 */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <zstd.h>
#include <lz4.h>
#include <lz4hc.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

namespace py = pybind11;

// ============================================================================
// Constants
// ============================================================================
static constexpr uint32_t TP1_MAGIC   = 0x5A315054;  // "TP1Z"
static constexpr uint16_t TP1_VERSION = 1;

// PZHeader.flags
static constexpr uint8_t FLAG_HAS_PERM      = 0x01;
static constexpr uint8_t FLAG_GAP16         = 0x02;
static constexpr uint8_t FLAG_HAS_METADATA  = 0x04;
static constexpr uint8_t FLAG_HAS_TRANSPOSE = 0x08;
static constexpr uint8_t FLAG_HAS_COLSUMS   = 0x10;
static constexpr uint8_t FLAG_HAS_OBS_VAR   = 0x20;  // Has obs/var DataFrames

// PZHeader.feature_flags
static constexpr uint32_t FEAT_ZSTD_CHECKSUMS   = 0x01;
static constexpr uint32_t FEAT_BITPLANE_BITMAP   = 0x02;  // bit-plane(p0) + bitmap pre-filter

// Codec IDs (stored in PZHeader.reserved[0])
static constexpr uint8_t CODEC_ZSTD   = 0;  // zstd (default, backward compatible)
static constexpr uint8_t CODEC_LZ4    = 1;  // LZ4 (fast read/write)
static constexpr uint8_t CODEC_LZ4HC  = 2;  // LZ4-HC (fast read, better ratio)

// Metadata tags
static constexpr uint8_t META_TAG_END      = 0;
static constexpr uint8_t META_TAG_ROWNAMES = 1;
static constexpr uint8_t META_TAG_COLNAMES = 2;
static constexpr uint8_t META_TAG_KV       = 3;  // Key-value string pairs (uns)
static constexpr uint8_t META_TAG_OBS      = 4;  // Cell metadata (native DataFrame)
static constexpr uint8_t META_TAG_VAR      = 5;  // Feature metadata (native DataFrame)

// DataFrame column types for native obs/var serialization
// Wire format per column: [u16 name_len][name][u8 dtype][u32 data_bytes][data]
enum DFDtype : uint8_t {
    DF_STRING      = 0,  // null-separated UTF-8 strings
    DF_INT32       = 1,  // nrows * 4 bytes LE
    DF_INT64       = 2,  // nrows * 8 bytes LE
    DF_FLOAT32     = 3,  // nrows * 4 bytes LE
    DF_FLOAT64     = 4,  // nrows * 8 bytes LE
    DF_UINT8       = 5,  // nrows bytes (booleans: 0/1)
    DF_CATEGORICAL = 6,  // [u32 nlev][null-sep levels][nrows*4 int32 codes]
};

// ============================================================================
// CRC32 (ISO 3309, polynomial 0xEDB88320)
// ============================================================================
struct CRC32 {
    uint32_t state_;
    CRC32() : state_(0xFFFFFFFF) {}
    void update(const uint8_t* data, size_t len) {
        static const auto& table = get_table();
        for (size_t i = 0; i < len; ++i)
            state_ = (state_ >> 8) ^ table[(state_ ^ data[i]) & 0xFF];
    }
    void update(const void* data, size_t len) {
        update(reinterpret_cast<const uint8_t*>(data), len);
    }
    uint32_t finalize() const { return state_ ^ 0xFFFFFFFF; }
    static uint32_t compute(const uint8_t* data, size_t len) {
        CRC32 c; c.update(data, len); return c.finalize();
    }
    static uint32_t compute(const void* data, size_t len) {
        return compute(reinterpret_cast<const uint8_t*>(data), len);
    }
private:
    struct Table {
        uint32_t tab[256];
        constexpr Table() : tab{} {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j)
                    c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                tab[i] = c;
            }
        }
        const uint32_t& operator[](size_t idx) const { return tab[idx]; }
    };
    static const Table& get_table() {
        static constexpr Table t{};
        return t;
    }
};

// ============================================================================
// .1pz header (96 bytes)
// ============================================================================
struct PZHeader {
    uint32_t magic;            // 0:  TP1_MAGIC
    uint16_t version;          // 4:  1
    uint8_t  vt_code;          // 6:  1=u8, 2=u16, 3=u32
    uint8_t  flags;            // 7:  FLAG_*
    uint32_t m;                // 8:  rows
    uint32_t n;                // 12: columns
    uint64_t nnz;              // 16: non-zeros
    uint8_t  ptr_width;        // 24: 2 or 4
    uint8_t  codec_level;      // 25: zstd level
    uint16_t _pad0;            // 26: reserved
    uint32_t num_chunks;       // 28:
    uint32_t perm_z_sz;        // 32:
    uint32_t ptr_z_sz;         // 36:
    uint32_t chunk_cols;       // 40: columns per chunk
    uint32_t feature_flags;    // 44:
    uint64_t metadata_offset;  // 48: byte offset to metadata section
    uint32_t metadata_z_sz;    // 56: compressed metadata size
    uint32_t colsums_z_sz;     // 60: compressed colsums size
    uint64_t transpose_offset; // 64: byte offset to transpose section
    uint32_t transpose_z_sz;   // 72: total transpose section size
    uint32_t transpose_chunks; // 76: number of transpose chunks
    uint8_t  reserved[16];     // 80-95
};
static_assert(sizeof(PZHeader) == 96, "PZHeader must be 96 bytes");

// .1pz footer (16 bytes)
struct PZFooter {
    uint32_t file_crc32;   // CRC32 of all bytes before footer
    uint32_t _reserved;    // 0
    uint32_t num_chunks;   // redundant check against header
    uint32_t magic;        // TP1_MAGIC
};
static_assert(sizeof(PZFooter) == 16, "PZFooter must be 16 bytes");

// Transpose section header (16 bytes)
struct TransposeHeader {
    uint32_t perm_z_sz;
    uint32_t ptr_z_sz;
    uint32_t chunk_cols;
    uint8_t  ptr_width;
    uint8_t  flags;        // bit0 = gap16
    uint16_t _pad;
};
static_assert(sizeof(TransposeHeader) == 16, "TransposeHeader must be 16 bytes");

// ============================================================================
// Non-owning CSC view
// ============================================================================
struct CSCView {
    uint32_t m, n;
    uint64_t nnz;
    const uint32_t* p;
    const uint32_t* i;
    const double* x;
};

// ============================================================================
// ZSTD RAII
// ============================================================================
struct ZstdCCtx {
    ZSTD_CCtx* ctx;
    ZstdCCtx() : ctx(ZSTD_createCCtx()) {}
    ~ZstdCCtx() { if (ctx) ZSTD_freeCCtx(ctx); }
    ZstdCCtx(const ZstdCCtx&) = delete;
    ZstdCCtx& operator=(const ZstdCCtx&) = delete;
};
struct ZstdDCtx {
    ZSTD_DCtx* ctx;
    ZstdDCtx() : ctx(ZSTD_createDCtx()) {}
    ~ZstdDCtx() { if (ctx) ZSTD_freeDCtx(ctx); }
    ZstdDCtx(const ZstdDCtx&) = delete;
    ZstdDCtx& operator=(const ZstdDCtx&) = delete;
};

// ============================================================================
// ZSTD helpers (with content checksums)
// ============================================================================
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
static inline size_t zstd_decompress(ZSTD_DCtx* ctx, void* dst, size_t cap,
                                     const void* src, size_t src_sz) {
    size_t ret = ZSTD_decompressDCtx(ctx, dst, cap, src, src_sz);
    if (ZSTD_isError(ret))
        throw std::runtime_error(std::string("ZSTD decompress: ") + ZSTD_getErrorName(ret));
    return ret;
}

// ============================================================================
// Generic chunk compress / decompress (dispatch on codec_id)
// ============================================================================
static inline std::vector<uint8_t> chunk_compress(ZSTD_CCtx* ctx,
                                                   const uint8_t* data, size_t sz,
                                                   int level, uint8_t codec_id) {
    switch (codec_id) {
    case CODEC_LZ4: {
        int bound = LZ4_compressBound((int)sz);
        std::vector<uint8_t> out(bound);
        int csz = LZ4_compress_default((const char*)data, (char*)out.data(),
                                        (int)sz, bound);
        if (csz <= 0) throw std::runtime_error("LZ4 compress failed");
        out.resize(csz);
        return out;
    }
    case CODEC_LZ4HC: {
        int bound = LZ4_compressBound((int)sz);
        std::vector<uint8_t> out(bound);
        int csz = LZ4_compress_HC((const char*)data, (char*)out.data(),
                                   (int)sz, bound, level);
        if (csz <= 0) throw std::runtime_error("LZ4-HC compress failed");
        out.resize(csz);
        return out;
    }
    default:  // CODEC_ZSTD
        return zstd_compress(ctx, data, sz, level);
    }
}

static inline size_t chunk_decompress(ZSTD_DCtx* ctx, void* dst, size_t cap,
                                       const void* src, size_t src_sz,
                                       uint8_t codec_id) {
    switch (codec_id) {
    case CODEC_LZ4:
    case CODEC_LZ4HC: {
        int ret = LZ4_decompress_safe((const char*)src, (char*)dst,
                                       (int)src_sz, (int)cap);
        if (ret < 0) throw std::runtime_error("LZ4 decompress failed");
        return (size_t)ret;
    }
    default:  // CODEC_ZSTD
        return zstd_decompress(ctx, dst, cap, src, src_sz);
    }
}

// ============================================================================
// Varint (LEB128)
// ============================================================================
static inline void varint_push(std::vector<uint8_t>& out, uint32_t v) {
    while (v >= 0x80) { out.push_back((v & 0x7F) | 0x80); v >>= 7; }
    out.push_back(v);
}
static inline uint32_t varint_read(const uint8_t*& p) {
    uint32_t v = 0; int s = 0;
    while (*p & 0x80) { v |= (uint32_t)(*p & 0x7F) << s; s += 7; ++p; }
    v |= (uint32_t)*p << s; ++p;
    return v;
}

// ============================================================================
// Byte-split encode (scalar, 4x unrolled)
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
// Byte-unsplit decode (SSE2-accelerated with scalar fallback)
// ============================================================================
#ifdef __SSE2__
static void byte_unsplit_16(const uint8_t* __restrict__ src, size_t n,
                             uint32_t* __restrict__ dst) {
    const uint8_t* p0 = src;
    const uint8_t* p1 = src + n;
    __m128i zero = _mm_setzero_si128();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i lo = _mm_loadu_si128((const __m128i*)(p0 + i));
        __m128i hi = _mm_loadu_si128((const __m128i*)(p1 + i));
        __m128i mix_lo = _mm_unpacklo_epi8(lo, hi);
        __m128i mix_hi = _mm_unpackhi_epi8(lo, hi);
        _mm_storeu_si128((__m128i*)(dst + i),      _mm_unpacklo_epi16(mix_lo, zero));
        _mm_storeu_si128((__m128i*)(dst + i + 4),  _mm_unpackhi_epi16(mix_lo, zero));
        _mm_storeu_si128((__m128i*)(dst + i + 8),  _mm_unpacklo_epi16(mix_hi, zero));
        _mm_storeu_si128((__m128i*)(dst + i + 12), _mm_unpackhi_epi16(mix_hi, zero));
    }
    for (; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i] << 8);
}

static void byte_unsplit_32(const uint8_t* __restrict__ src, size_t n,
                             uint32_t* __restrict__ dst) {
    const uint8_t *p0=src, *p1=src+n, *p2=src+2*n, *p3=src+3*n;
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m128i b0 = _mm_loadu_si128((const __m128i*)(p0 + i));
        __m128i b1 = _mm_loadu_si128((const __m128i*)(p1 + i));
        __m128i b2 = _mm_loadu_si128((const __m128i*)(p2 + i));
        __m128i b3 = _mm_loadu_si128((const __m128i*)(p3 + i));
        __m128i lo01_l = _mm_unpacklo_epi8(b0, b1);
        __m128i lo01_h = _mm_unpackhi_epi8(b0, b1);
        __m128i hi23_l = _mm_unpacklo_epi8(b2, b3);
        __m128i hi23_h = _mm_unpackhi_epi8(b2, b3);
        _mm_storeu_si128((__m128i*)(dst + i),      _mm_unpacklo_epi16(lo01_l, hi23_l));
        _mm_storeu_si128((__m128i*)(dst + i + 4),  _mm_unpackhi_epi16(lo01_l, hi23_l));
        _mm_storeu_si128((__m128i*)(dst + i + 8),  _mm_unpacklo_epi16(lo01_h, hi23_h));
        _mm_storeu_si128((__m128i*)(dst + i + 12), _mm_unpackhi_epi16(lo01_h, hi23_h));
    }
    for (; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i]<<8) | ((uint32_t)p2[i]<<16) | ((uint32_t)p3[i]<<24);
}
#else
static void byte_unsplit_16(const uint8_t* __restrict__ src, size_t n,
                             uint32_t* __restrict__ dst) {
    const uint8_t* p0 = src;
    const uint8_t* p1 = src + n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        dst[i]   = p0[i]   | ((uint32_t)p1[i]   << 8);
        dst[i+1] = p0[i+1] | ((uint32_t)p1[i+1] << 8);
        dst[i+2] = p0[i+2] | ((uint32_t)p1[i+2] << 8);
        dst[i+3] = p0[i+3] | ((uint32_t)p1[i+3] << 8);
    }
    for (; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i] << 8);
}

static void byte_unsplit_32(const uint8_t* __restrict__ src, size_t n,
                             uint32_t* __restrict__ dst) {
    const uint8_t *p0=src, *p1=src+n, *p2=src+2*n, *p3=src+3*n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        dst[i]   = p0[i]   | ((uint32_t)p1[i]<<8)   | ((uint32_t)p2[i]<<16)   | ((uint32_t)p3[i]<<24);
        dst[i+1] = p0[i+1] | ((uint32_t)p1[i+1]<<8) | ((uint32_t)p2[i+1]<<16) | ((uint32_t)p3[i+1]<<24);
        dst[i+2] = p0[i+2] | ((uint32_t)p1[i+2]<<8) | ((uint32_t)p2[i+2]<<16) | ((uint32_t)p3[i+2]<<24);
        dst[i+3] = p0[i+3] | ((uint32_t)p1[i+3]<<8) | ((uint32_t)p2[i+3]<<16) | ((uint32_t)p3[i+3]<<24);
    }
    for (; i < n; ++i)
        dst[i] = p0[i] | ((uint32_t)p1[i]<<8) | ((uint32_t)p2[i]<<16) | ((uint32_t)p3[i]<<24);
}
#endif

// ============================================================================
// Bit-plane decomposition (plane 0 transform)
// ============================================================================
// Decomposes n bytes into 8 bit-planes, each ceil(n/8) bytes.
// Output size: 8 * ceil(n/8) bytes. dst must be pre-zeroed or will be memset.
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

// Reconstitutes n original bytes from 8 bit-planes (each ceil(n/8) bytes).
static void bit_planes_decode(const uint8_t* __restrict__ src, size_t n,
                               uint8_t* __restrict__ dst) {
    size_t pb = (n + 7) / 8;
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = 0;
        size_t byte_idx = i / 8;
        uint8_t bit_mask = 1u << (i & 7);
        for (int b = 0; b < 8; ++b) {
            if (src[b * pb + byte_idx] & bit_mask)
                v |= (1 << b);
        }
        dst[i] = v;
    }
}

// ============================================================================
// Bitmap zero-filter: encodes only non-zero bytes with a bitmap header
// ============================================================================
// Output layout: [bitmap(ceil(n/8) bytes) | non-zero values]
// Returns actual packed size written to dst.
// dst must have capacity >= n + ceil(n/8).
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

// Unpacks bitmap-encoded data back to original n bytes.
static void bitmap_unpack(const uint8_t* __restrict__ src, size_t /*packed_sz*/,
                           uint8_t* __restrict__ dst, size_t n) {
    size_t bm_bytes = (n + 7) / 8;
    std::memset(dst, 0, n);
    size_t rp = bm_bytes;
    for (size_t i = 0; i < n; ++i) {
        if (src[i / 8] & (1u << (i & 7))) {
            dst[i] = src[rp++];
        }
    }
}

// ============================================================================
// VOCSC encode one chunk
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
    int64_t col_start, int64_t col_end,
    std::vector<std::pair<uint32_t, uint32_t>>& vr)
{
    VocscResult r;
    auto& gaps = r.gaps;
    auto& meta = r.metadata;
    r.max_gap = 0;

    uint64_t cnnz = 0;
    for (int64_t j = col_start; j < col_end; ++j) cnnz += col_counts[j];
    gaps.reserve(cnnz);
    meta.reserve((col_end - col_start) * 8);

    for (int64_t j = col_start; j < col_end; ++j) {
        uint32_t cnt = col_counts[j];
        if (cnt == 0) { varint_push(meta, 0); continue; }
        uint64_t s = mat.p[j];
        if (s + cnt > mat.nnz) {
            cnt = (s < mat.nnz) ? (uint32_t)(mat.nnz - s) : 0;
            if (cnt == 0) { varint_push(meta, 0); continue; }
        }
        vr.resize(cnt);
        for (uint32_t k = 0; k < cnt; ++k)
            vr[k] = { (uint32_t)mat.x[s+k], inv_perm[mat.i[s+k]] };
        std::sort(vr.begin(), vr.end());

        uint32_t ng = 1;
        for (size_t p = 1; p < vr.size(); ++p)
            if (vr[p].first != vr[p-1].first) ng++;
        varint_push(meta, ng);

        size_t pos = 0;
        while (pos < vr.size()) {
            uint32_t cv = vr[pos].first;
            size_t gs = pos;
            while (pos < vr.size() && vr[pos].first == cv) pos++;
            varint_push(meta, cv);
            varint_push(meta, (uint32_t)(pos - gs));
            uint32_t prev = 0;
            for (size_t i = gs; i < pos; ++i) {
                uint32_t row = vr[i].second;
                uint32_t gap = row - prev;
                if (gap > r.max_gap) r.max_gap = gap;
                gaps.push_back(gap);
                prev = row + 1;
            }
        }
    }
    r.n_gaps = gaps.size();
    return r;
}

// ============================================================================
// VOCSC decode one chunk (templated for double/int32 values)
// ============================================================================
template<typename ValT, typename IdxT = int32_t>
static void vocsc_decode_chunk(
    const uint8_t* meta, const uint32_t* gaps, const uint32_t* perm,
    int64_t col_start, int64_t col_end,
    const IdxT* indptr, IdxT* out_ix, ValT* out_vx)
{
    const uint8_t* mp = meta;
    size_t gp = 0;
    int64_t wp = indptr[col_start];

    for (int64_t j = col_start; j < col_end; ++j) {
        uint32_t ngrp = varint_read(mp);
        for (uint32_t g = 0; g < ngrp; ++g) {
            uint32_t val = varint_read(mp);
            uint32_t gc  = varint_read(mp);
            ValT dv = (ValT)val;
            uint32_t prev = 0;
            for (uint32_t k = 0; k < gc; ++k) {
                uint32_t mr = prev + gaps[gp++];
                prev = mr + 1;
                if (k + 1 < gc)
                    __builtin_prefetch(&perm[prev + gaps[gp]], 0, 1);
                out_ix[wp] = (IdxT)perm[mr];
                out_vx[wp] = dv;
                wp++;
            }
        }
    }
}

// ============================================================================
// Metadata serialization
// ============================================================================

// Push a null-delimited string list as a TLV entry
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

// Push a raw byte vector as a TLV entry
static void push_raw_tlv(std::vector<uint8_t>& out, uint8_t tag,
                         const std::vector<uint8_t>& data) {
    if (data.empty()) return;
    out.push_back(tag);
    uint32_t sz = (uint32_t)data.size();
    out.insert(out.end(), (uint8_t*)&sz, (uint8_t*)&sz + 4);
    out.insert(out.end(), data.begin(), data.end());
}

// ============================================================================
// Native DataFrame serialization (C++ native — no Parquet dependency)
//
// Wire format (inside META_TAG_OBS / META_TAG_VAR TLV blob):
//   [u32 nrows] [u32 ncols] [u32 index_bytes]
//   [index: null-terminated strings concatenated]
//   Per column (repeated ncols times):
//     [u16 name_len] [name bytes] [u8 DFDtype] [u32 data_bytes] [data]
// ============================================================================

// Serialize DataFrame from Python objects → native binary blob
static std::vector<uint8_t> serialize_dataframe_py(
    uint32_t nrows,
    const py::list& index,       // string index values
    const py::list& col_specs)   // list of (name:str, data:object) tuples
{
    std::vector<uint8_t> buf;
    auto push_u32 = [&](uint32_t v) {
        buf.insert(buf.end(), (uint8_t*)&v, (uint8_t*)&v + 4);
    };
    auto push_u16 = [&](uint16_t v) {
        buf.insert(buf.end(), (uint8_t*)&v, (uint8_t*)&v + 2);
    };

    uint32_t ncols = (uint32_t)py::len(col_specs);
    push_u32(nrows);
    push_u32(ncols);

    // Index: null-terminated strings
    std::vector<uint8_t> idx_buf;
    for (auto item : index) {
        std::string s = py::cast<std::string>(item);
        idx_buf.insert(idx_buf.end(), s.begin(), s.end());
        idx_buf.push_back(0);
    }
    push_u32((uint32_t)idx_buf.size());
    buf.insert(buf.end(), idx_buf.begin(), idx_buf.end());

    // Columns
    for (auto item : col_specs) {
        auto t = py::cast<py::tuple>(item);
        std::string name = py::cast<std::string>(t[0]);
        auto data = t[1];

        // Column name
        push_u16((uint16_t)name.size());
        buf.insert(buf.end(), name.begin(), name.end());

        if (py::isinstance<py::list>(data)) {
            // String column
            buf.push_back(DF_STRING);
            std::vector<uint8_t> str_buf;
            auto lst = py::cast<py::list>(data);
            for (auto s : lst) {
                auto sv = py::cast<std::string>(s);
                str_buf.insert(str_buf.end(), sv.begin(), sv.end());
                str_buf.push_back(0);
            }
            push_u32((uint32_t)str_buf.size());
            buf.insert(buf.end(), str_buf.begin(), str_buf.end());
        } else if (py::isinstance<py::dict>(data)) {
            // Categorical: {"levels": list[str], "codes": np.array[int32]}
            buf.push_back(DF_CATEGORICAL);
            auto d = py::cast<py::dict>(data);
            auto levels = py::cast<py::list>(d["levels"]);
            auto codes = py::cast<py::array_t<int32_t>>(d["codes"]);

            std::vector<uint8_t> cat_buf;
            uint32_t nlev = (uint32_t)py::len(levels);
            cat_buf.insert(cat_buf.end(), (uint8_t*)&nlev, (uint8_t*)&nlev + 4);
            for (auto lv : levels) {
                auto s = py::cast<std::string>(lv);
                cat_buf.insert(cat_buf.end(), s.begin(), s.end());
                cat_buf.push_back(0);
            }
            auto cbuf = codes.request();
            cat_buf.insert(cat_buf.end(),
                (uint8_t*)cbuf.ptr, (uint8_t*)cbuf.ptr + nrows * 4);

            push_u32((uint32_t)cat_buf.size());
            buf.insert(buf.end(), cat_buf.begin(), cat_buf.end());
        } else {
            // Numpy array — inspect dtype
            auto arr = py::cast<py::array>(data);
            auto abuf = arr.request();

            if (arr.dtype().is(py::dtype::of<int32_t>())) {
                buf.push_back(DF_INT32);
                push_u32(nrows * 4);
                buf.insert(buf.end(), (uint8_t*)abuf.ptr, (uint8_t*)abuf.ptr + nrows * 4);
            } else if (arr.dtype().is(py::dtype::of<int64_t>())) {
                buf.push_back(DF_INT64);
                push_u32(nrows * 8);
                buf.insert(buf.end(), (uint8_t*)abuf.ptr, (uint8_t*)abuf.ptr + nrows * 8);
            } else if (arr.dtype().is(py::dtype::of<float>())) {
                buf.push_back(DF_FLOAT32);
                push_u32(nrows * 4);
                buf.insert(buf.end(), (uint8_t*)abuf.ptr, (uint8_t*)abuf.ptr + nrows * 4);
            } else if (arr.dtype().is(py::dtype::of<double>())) {
                buf.push_back(DF_FLOAT64);
                push_u32(nrows * 8);
                buf.insert(buf.end(), (uint8_t*)abuf.ptr, (uint8_t*)abuf.ptr + nrows * 8);
            } else if (arr.dtype().is(py::dtype::of<uint8_t>())) {
                buf.push_back(DF_UINT8);
                push_u32(nrows);
                buf.insert(buf.end(), (uint8_t*)abuf.ptr, (uint8_t*)abuf.ptr + nrows);
            } else {
                throw std::runtime_error("Unsupported dtype for obs/var column '" + name + "'");
            }
        }
    }
    return buf;
}

// Deserialize native DataFrame binary → Python dict
// Returns {"__index__": list[str], "col_name": array_or_list, ...}
// Categorical columns return {"levels": list, "codes": array} dicts.
static py::dict deserialize_dataframe_py(const uint8_t* data, size_t len) {
    const uint8_t* p = data;
    const uint8_t* end = data + len;

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v; std::memcpy(&v, p, 4); p += 4; return v;
    };
    auto read_u16 = [&]() -> uint16_t {
        uint16_t v; std::memcpy(&v, p, 2); p += 2; return v;
    };

    if (p + 12 > end) throw std::runtime_error("DataFrame blob too short");
    uint32_t nrows = read_u32();
    uint32_t ncols = read_u32();
    uint32_t index_bytes = read_u32();

    // Parse index
    py::list index;
    if (index_bytes > 0 && p + index_bytes <= end) {
        const uint8_t* idx_end = p + index_bytes;
        const uint8_t* start = p;
        while (p < idx_end) {
            if (*p == 0) {
                index.append(py::str(std::string((const char*)start, p - start)));
                start = p + 1;
            }
            p++;
        }
    }

    py::dict result;
    result["__index__"] = index;

    for (uint32_t c = 0; c < ncols && p + 3 <= end; ++c) {
        uint16_t name_len = read_u16();
        if (p + name_len + 5 > end) break;
        std::string name((const char*)p, name_len);
        p += name_len;
        uint8_t dtype = *p++;
        uint32_t data_bytes = read_u32();
        if (p + data_bytes > end) break;

        switch (dtype) {
            case DF_STRING: {
                py::list vals;
                const uint8_t* col_end = p + data_bytes;
                const uint8_t* start = p;
                while (p < col_end) {
                    if (*p == 0) {
                        vals.append(py::str(std::string((const char*)start, p - start)));
                        start = p + 1;
                    }
                    p++;
                }
                result[py::str(name)] = vals;
                break;
            }
            case DF_INT32: {
                py::array_t<int32_t> arr(nrows);
                std::memcpy(arr.mutable_data(), p, nrows * 4);
                p += data_bytes;
                result[py::str(name)] = arr;
                break;
            }
            case DF_INT64: {
                py::array_t<int64_t> arr(nrows);
                std::memcpy(arr.mutable_data(), p, nrows * 8);
                p += data_bytes;
                result[py::str(name)] = arr;
                break;
            }
            case DF_FLOAT32: {
                py::array_t<float> arr(nrows);
                std::memcpy(arr.mutable_data(), p, nrows * 4);
                p += data_bytes;
                result[py::str(name)] = arr;
                break;
            }
            case DF_FLOAT64: {
                py::array_t<double> arr(nrows);
                std::memcpy(arr.mutable_data(), p, nrows * 8);
                p += data_bytes;
                result[py::str(name)] = arr;
                break;
            }
            case DF_UINT8: {
                // Return as numpy bool array
                py::array_t<bool> arr(nrows);
                auto ptr = arr.mutable_data();
                for (uint32_t i = 0; i < nrows; ++i) ptr[i] = p[i] != 0;
                p += data_bytes;
                result[py::str(name)] = arr;
                break;
            }
            case DF_CATEGORICAL: {
                const uint8_t* cat_start = p;
                uint32_t nlev;
                std::memcpy(&nlev, p, 4); p += 4;
                py::list levels;
                for (uint32_t i = 0; i < nlev; ++i) {
                    const char* start = (const char*)p;
                    while (p < end && *p != 0) p++;
                    levels.append(py::str(std::string(start, (const char*)p)));
                    if (p < end) p++;
                }
                py::array_t<int32_t> codes(nrows);
                std::memcpy(codes.mutable_data(), p, nrows * 4);
                p = cat_start + data_bytes;

                py::dict cat;
                cat["levels"] = levels;
                cat["codes"] = codes;
                result[py::str(name)] = cat;
                break;
            }
            default:
                p += data_bytes;  // Skip unknown dtypes (forward compat)
                break;
        }
    }
    return result;
}

// ============================================================================
// TLV metadata serialization (combines rownames, colnames, kv, obs, var)
// ============================================================================
static void serialize_metadata(std::vector<uint8_t>& out,
                               const std::vector<std::string>& rownames,
                               const std::vector<std::string>& colnames,
                               const std::vector<std::pair<std::string,std::string>>& kv_pairs = {},
                               const std::vector<uint8_t>& obs_blob = {},
                               const std::vector<uint8_t>& var_blob = {}) {
    push_strings_tlv(out, META_TAG_ROWNAMES, rownames);
    push_strings_tlv(out, META_TAG_COLNAMES, colnames);

    // Key-value pairs: key\0value\0key\0value\0...
    if (!kv_pairs.empty()) {
        size_t data_sz = 0;
        for (const auto& [k, v] : kv_pairs) data_sz += k.size() + 1 + v.size() + 1;
        out.push_back(META_TAG_KV);
        uint32_t sz = (uint32_t)data_sz;
        out.insert(out.end(), (uint8_t*)&sz, (uint8_t*)&sz + 4);
        for (const auto& [k, v] : kv_pairs) {
            out.insert(out.end(), k.begin(), k.end());
            out.push_back(0);
            out.insert(out.end(), v.begin(), v.end());
            out.push_back(0);
        }
    }

    // Obs/Var as native DataFrame blobs
    push_raw_tlv(out, META_TAG_OBS, obs_blob);
    push_raw_tlv(out, META_TAG_VAR, var_blob);

    out.push_back(META_TAG_END);
}

struct MetadataResult {
    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
    std::vector<std::pair<std::string,std::string>> kv_pairs;
    std::vector<uint8_t> obs_blob;
    std::vector<uint8_t> var_blob;
};

static MetadataResult deserialize_metadata(const uint8_t* data, size_t len) {
    MetadataResult m;
    const uint8_t* p = data;
    const uint8_t* end = data + len;
    auto read_strings = [](const uint8_t* p, size_t sz) {
        std::vector<std::string> result;
        const uint8_t* end = p + sz;
        const uint8_t* start = p;
        while (p < end) {
            if (*p == 0) {
                result.emplace_back((const char*)start, p - start);
                start = p + 1;
            }
            p++;
        }
        return result;
    };
    while (p < end) {
        uint8_t tag = *p++;
        if (tag == META_TAG_END) break;
        if (p + 4 > end) break;
        uint32_t sz;
        std::memcpy(&sz, p, 4);
        p += 4;
        if (p + sz > end) break;
        if (tag == META_TAG_ROWNAMES) {
            m.rownames = read_strings(p, sz);
        } else if (tag == META_TAG_COLNAMES) {
            m.colnames = read_strings(p, sz);
        } else if (tag == META_TAG_KV) {
            auto strings = read_strings(p, sz);
            for (size_t i = 0; i + 1 < strings.size(); i += 2) {
                m.kv_pairs.emplace_back(strings[i], strings[i + 1]);
            }
        } else if (tag == META_TAG_OBS) {
            m.obs_blob.assign(p, p + sz);
        } else if (tag == META_TAG_VAR) {
            m.var_blob.assign(p, p + sz);
        }
        // Unknown tags are silently skipped (forward compatibility)
        p += sz;
    }
    return m;
}

// ============================================================================
// CSC -> CSR conversion (for transpose storage)
// ============================================================================
static void csc_to_csr(
    const uint32_t* csc_p, const uint32_t* csc_i, const double* csc_x,
    uint32_t m, uint32_t n, uint64_t nnz,
    std::vector<uint32_t>& csr_p,
    std::vector<uint32_t>& csr_i,
    std::vector<double>& csr_x)
{
    std::vector<uint32_t> row_count(m, 0);
    for (uint64_t k = 0; k < nnz; ++k) row_count[csc_i[k]]++;
    csr_p.resize(m + 1, 0);
    for (uint32_t i = 0; i < m; ++i) csr_p[i+1] = csr_p[i] + row_count[i];
    csr_i.resize(nnz);
    csr_x.resize(nnz);
    std::vector<uint32_t> pos(m, 0);
    for (uint32_t j = 0; j < n; ++j) {
        for (uint64_t k = csc_p[j]; k < csc_p[j+1]; ++k) {
            uint32_t row = csc_i[k];
            uint64_t dest = csr_p[row] + pos[row]++;
            csr_i[dest] = j;
            csr_x[dest] = csc_x[k];
        }
    }
}

// ============================================================================
// Encode a CSC matrix into VOCSC chunks (reusable for forward + transpose)
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

static EncodedSection encode_vocsc_section(
    const uint32_t* p_ptr, const uint32_t* i_ptr, const double* x_ptr,
    uint32_t m, uint32_t n, uint64_t nnz,
    int chunk_cols, int level, int nt, uint8_t codec_id = CODEC_ZSTD)
{
    EncodedSection sec;

    // Row frequency-sort permutation
    std::vector<int32_t> rnnz(m, 0);
    for (uint64_t k = 0; k < nnz; ++k) rnnz[i_ptr[k]]++;
    std::vector<uint32_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](uint32_t a, uint32_t b) {
        return rnnz[a] > rnnz[b];
    });
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

    // Pass 1: VOCSC encode
    struct ChunkData { VocscResult vocsc; };
    std::vector<ChunkData> cdata(nc);

    #pragma omp parallel num_threads(nt)
    {
        std::vector<std::pair<uint32_t, uint32_t>> tvr;
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < nc; ++c) {
            int64_t cs = (int64_t)c * chunk_cols;
            int64_t ce = std::min(cs + (int64_t)chunk_cols, (int64_t)n);
            cdata[c].vocsc = vocsc_encode_chunk(mat, inv_perm.data(), cc.data(), cs, ce, tvr);
            if (cdata[c].vocsc.max_gap >= 65536) {
                #pragma omp atomic write
                any_gap32 = true;
            }
        }
    }

    sec.gap16 = !any_gap32;
    int gw = sec.gap16 ? 2 : 4;

    // Pass 2: byte-split + bit-plane(p0) + bitmap + compress + per-chunk CRC
    #pragma omp parallel num_threads(nt)
    {
        ZstdCCtx tcctx;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> prefilter;
        std::vector<uint8_t> packed;
        #pragma omp for schedule(dynamic)
        for (int c = 0; c < nc; ++c) {
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

            // Metadata stays at front
            std::memcpy(prefilter.data(), raw.data(), msz);
            if (ng > 0) {
                // Bit-plane decompose p0
                bit_planes_encode(raw.data() + msz, ng, prefilter.data() + msz);
                // Copy remaining byte-planes (p1, p2, p3) unchanged
                if (gw > 1)
                    std::memcpy(prefilter.data() + msz + bp_sz,
                                raw.data() + msz + ng, ng * (gw - 1));
            }

            // Bitmap pack (max output = pf_sz + ceil(pf_sz/8))
            packed.resize(pf_sz + (pf_sz + 7) / 8);
            size_t packed_sz = bitmap_pack(prefilter.data(), pf_sz, packed.data());

            // Per-chunk CRC32 over packed (pre-compression) data
            uint32_t chunk_crc = CRC32::compute(packed.data(), packed_sz);

            auto z = chunk_compress(tcctx.ctx, packed.data(), packed_sz, level, codec_id);

            uint32_t ng32  = (uint32_t)ng;
            uint32_t msz32 = (uint32_t)msz;
            uint32_t zsz   = (uint32_t)z.size();
            uint32_t psz32 = (uint32_t)packed_sz;

            // chunk blob: ng(4) + msz(4) + zsz(4) + crc(4) + packed_sz(4) + compressed
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
    }

    sec.ctable.resize(nc);
    for (int c = 0; c < nc; ++c) sec.ctable[c] = (uint32_t)sec.blobs[c].size();
    return sec;
}

// ============================================================================
// Core .1pz write
// ============================================================================
static py::dict pz_write_core(
    const uint32_t* p_ptr, const uint32_t* i_ptr, const double* x_ptr,
    uint32_t m, uint32_t n, uint64_t nnz,
    const std::string& out_path, int nt, int level, int chunk_cols,
    const std::vector<std::string>& rownames,
    const std::vector<std::string>& colnames,
    bool store_transpose,
    const std::vector<std::pair<std::string,std::string>>& kv_pairs = {},
    const std::vector<uint8_t>& obs_blob = {},
    const std::vector<uint8_t>& var_blob = {},
    uint8_t codec_id = CODEC_ZSTD)
{
    // --- Encode forward section ---
    auto fwd = encode_vocsc_section(p_ptr, i_ptr, x_ptr, m, n, nnz,
                                    chunk_cols, level, nt, codec_id);

    // Value type code
    uint32_t mx = 0;
    for (uint64_t k = 0; k < nnz; ++k) {
        uint32_t v = (uint32_t)x_ptr[k];
        if (v > mx) mx = v;
    }
    int vt_code = (mx <= 255) ? 1 : (mx <= 65535) ? 2 : 3;

    // --- Compute column sums ---
    std::vector<uint64_t> colsums(n, 0);
    for (uint32_t j = 0; j < n; ++j)
        for (uint64_t k = p_ptr[j]; k < p_ptr[j+1]; ++k)
            colsums[j] += (uint64_t)x_ptr[k];

    ZstdCCtx cctx;
    auto colsums_z = zstd_compress(cctx.ctx,
        (const uint8_t*)colsums.data(), n * 8, 3);

    // --- Serialize metadata ---
    std::vector<uint8_t> meta_raw;
    bool has_obs_var = !obs_blob.empty() || !var_blob.empty();
    bool has_meta = !rownames.empty() || !colnames.empty()
                    || !kv_pairs.empty() || has_obs_var;
    if (has_meta) serialize_metadata(meta_raw, rownames, colnames,
                                     kv_pairs, obs_blob, var_blob);
    auto meta_z = has_meta ?
        zstd_compress(cctx.ctx, meta_raw.data(), meta_raw.size(), 3) :
        std::vector<uint8_t>{};

    // --- Encode transpose (optional) ---
    EncodedSection trans;
    TransposeHeader thdr{};
    if (store_transpose) {
        std::vector<uint32_t> csr_p, csr_i;
        std::vector<double> csr_x;
        csc_to_csr(p_ptr, i_ptr, x_ptr, m, n, nnz, csr_p, csr_i, csr_x);
        // CSR of X = CSC of X^T: m_t=n rows, n_t=m cols
        trans = encode_vocsc_section(
            csr_p.data(), csr_i.data(), csr_x.data(),
            n, m, nnz, chunk_cols, level, nt, codec_id);
        thdr.perm_z_sz = (uint32_t)trans.perm_z.size();
        thdr.ptr_z_sz = (uint32_t)trans.ptr_z.size();
        thdr.chunk_cols = chunk_cols;
        thdr.ptr_width = trans.ptr_width;
        thdr.flags = trans.gap16 ? 0x01 : 0x00;
    }

    // --- Compute offsets ---
    size_t off = 96;  // header
    off += fwd.perm_z.size() + fwd.ptr_z.size();
    off += fwd.num_chunks * 4;  // chunk table
    for (int c = 0; c < fwd.num_chunks; ++c) off += fwd.blobs[c].size();

    size_t colsums_offset = off;
    off += colsums_z.size();

    size_t metadata_offset = off;
    off += meta_z.size();

    size_t transpose_offset = off;
    size_t transpose_total_sz = 0;
    if (store_transpose) {
        transpose_total_sz = 16;  // TransposeHeader
        transpose_total_sz += trans.perm_z.size() + trans.ptr_z.size();
        transpose_total_sz += trans.num_chunks * 4;
        for (int c = 0; c < trans.num_chunks; ++c)
            transpose_total_sz += trans.blobs[c].size();
        off += transpose_total_sz;
    }

    size_t footer_offset = off;

    // --- Build header ---
    PZHeader hdr{};
    hdr.magic = TP1_MAGIC;
    hdr.version = TP1_VERSION;
    hdr.vt_code = vt_code;
    hdr.flags = FLAG_HAS_PERM | FLAG_HAS_COLSUMS;
    if (fwd.gap16) hdr.flags |= FLAG_GAP16;
    if (has_meta) hdr.flags |= FLAG_HAS_METADATA;
    if (has_obs_var) hdr.flags |= FLAG_HAS_OBS_VAR;
    if (store_transpose) hdr.flags |= FLAG_HAS_TRANSPOSE;
    hdr.m = m; hdr.n = n; hdr.nnz = nnz;
    hdr.ptr_width = fwd.ptr_width;
    hdr.codec_level = level;
    hdr.reserved[0] = codec_id;  // Codec ID: 0=zstd, 1=lz4, 2=lz4hc
    hdr.num_chunks = fwd.num_chunks;
    hdr.perm_z_sz = (uint32_t)fwd.perm_z.size();
    hdr.ptr_z_sz = (uint32_t)fwd.ptr_z.size();
    hdr.chunk_cols = chunk_cols;
    hdr.feature_flags = FEAT_ZSTD_CHECKSUMS | FEAT_BITPLANE_BITMAP;
    hdr.metadata_offset = has_meta ? metadata_offset : 0;
    hdr.metadata_z_sz = (uint32_t)meta_z.size();
    hdr.colsums_z_sz = (uint32_t)colsums_z.size();
    hdr.transpose_offset = store_transpose ? transpose_offset : 0;
    hdr.transpose_z_sz = (uint32_t)transpose_total_sz;
    hdr.transpose_chunks = store_transpose ? trans.num_chunks : 0;

    // --- Build file buffer ---
    std::vector<uint8_t> buf(footer_offset);
    size_t wp = 0;
    auto append = [&](const void* data, size_t sz) {
        std::memcpy(buf.data() + wp, data, sz); wp += sz;
    };
    append(&hdr, 96);
    append(fwd.perm_z.data(), fwd.perm_z.size());
    append(fwd.ptr_z.data(), fwd.ptr_z.size());
    append(fwd.ctable.data(), fwd.num_chunks * 4);
    for (int c = 0; c < fwd.num_chunks; ++c)
        append(fwd.blobs[c].data(), fwd.blobs[c].size());
    append(colsums_z.data(), colsums_z.size());
    if (has_meta)
        append(meta_z.data(), meta_z.size());
    if (store_transpose) {
        append(&thdr, 16);
        append(trans.perm_z.data(), trans.perm_z.size());
        append(trans.ptr_z.data(), trans.ptr_z.size());
        append(trans.ctable.data(), trans.num_chunks * 4);
        for (int c = 0; c < trans.num_chunks; ++c)
            append(trans.blobs[c].data(), trans.blobs[c].size());
    }

    // --- Compute file CRC and footer ---
    uint32_t file_crc = CRC32::compute(buf.data(), buf.size());
    PZFooter ftr{};
    ftr.file_crc32 = file_crc;
    ftr.num_chunks = fwd.num_chunks;
    ftr.magic = TP1_MAGIC;

    // --- Write ---
    std::ofstream fout(out_path, std::ios::binary);
    if (!fout) throw std::runtime_error("Cannot write: " + out_path);
    fout.write((const char*)buf.data(), buf.size());
    fout.write((const char*)&ftr, 16);
    fout.close();

    size_t osz = footer_offset + 16;
    uint64_t tnnz = nnz;
    size_t raw_csc = ((uint64_t)n+1)*4 + tnnz*4 + tnnz*8 + 16;

    py::dict r;
    r["compressed_bytes"] = osz;
    r["raw_bytes"] = raw_csc;
    r["ratio"] = (double)raw_csc / osz;
    r["m"] = m; r["n"] = n; r["nnz"] = tnnz;
    r["num_chunks"] = fwd.num_chunks;
    r["gap16"] = fwd.gap16;
    r["has_metadata"] = has_meta;
    r["has_colsums"] = true;
    r["has_transpose"] = store_transpose;
    r["has_obs"] = !obs_blob.empty();
    r["has_var"] = !var_blob.empty();
    r["has_uns"] = !kv_pairs.empty();
    return r;
}

// ============================================================================
// Helper: decode forward chunks from file blob
// ============================================================================
template<typename ValT>
static void decode_forward_chunks(
    const uint8_t* blob, size_t blob_sz,
    const PZHeader& hdr, int nt,
    int32_t* out_ip, int32_t* out_ix, ValT* out_vx,
    int c_start, int c_end)
{
    int64_t n = hdr.n;
    int cc = hdr.chunk_cols;
    bool gap16 = (hdr.flags & FLAG_GAP16) != 0;
    int gw = gap16 ? 2 : 4;
    uint8_t codec_id = hdr.reserved[0];
    bool has_bp = (hdr.feature_flags & FEAT_BITPLANE_BITMAP) != 0;

    // Locate chunk table and chunk offsets
    size_t off = 96 + hdr.perm_z_sz + hdr.ptr_z_sz;
    const uint32_t* all_ct_base = nullptr;
    std::vector<uint32_t> ct_buf;
    // Read chunk table
    ct_buf.resize(hdr.num_chunks);
    std::memcpy(ct_buf.data(), blob + off, hdr.num_chunks * 4);
    all_ct_base = ct_buf.data();
    off += hdr.num_chunks * 4;

    std::vector<size_t> coff(hdr.num_chunks);
    for (uint32_t c = 0; c < hdr.num_chunks; ++c) {
        coff[c] = off;
        off += all_ct_base[c];
    }

    // Decode chunks [c_start, c_end)
    #pragma omp parallel num_threads(nt)
    {
        ZstdDCtx td;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> packed_buf, pf_buf;
        std::vector<uint32_t> gaps;

        #pragma omp for schedule(dynamic)
        for (int c = c_start; c < c_end; ++c) {
            int64_t cs = (int64_t)c * cc;
            int64_t ce = std::min(cs + (int64_t)cc, n);
            const uint8_t* cp = blob + coff[c];

            uint32_t ng, msz, zsz;
            std::memcpy(&ng, cp, 4);  cp += 4;
            std::memcpy(&msz, cp, 4); cp += 4;
            std::memcpy(&zsz, cp, 4); cp += 4;

            uint32_t stored_crc = 0;
            std::memcpy(&stored_crc, cp, 4); cp += 4;

            uint32_t packed_sz = 0;
            if (has_bp) { std::memcpy(&packed_sz, cp, 4); cp += 4; }

            size_t raw_sz = msz + (size_t)ng * gw;

            if (has_bp) {
                packed_buf.resize(packed_sz);
                chunk_decompress(td.ctx, packed_buf.data(), packed_sz, cp, zsz, codec_id);

                {
                    uint32_t actual_crc = CRC32::compute(packed_buf.data(), packed_sz);
                    if (actual_crc != stored_crc)
                        throw std::runtime_error("Chunk CRC32 mismatch at chunk " + std::to_string(c));
                }

                size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
                size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
                pf_buf.resize(pf_sz);
                bitmap_unpack(packed_buf.data(), packed_sz, pf_buf.data(), pf_sz);

                raw.resize(raw_sz);
                std::memcpy(raw.data(), pf_buf.data(), msz);
                if (ng > 0) {
                    bit_planes_decode(pf_buf.data() + msz, ng, raw.data() + msz);
                    if (gw > 1)
                        std::memcpy(raw.data() + msz + ng,
                                    pf_buf.data() + msz + bp_sz, ng * (gw - 1));
                }
            } else {
                raw.resize(raw_sz);
                chunk_decompress(td.ctx, raw.data(), raw_sz, cp, zsz, codec_id);

                {
                    uint32_t actual_crc = CRC32::compute(raw.data(), raw_sz);
                    if (actual_crc != stored_crc)
                        throw std::runtime_error("Chunk CRC32 mismatch at chunk " + std::to_string(c));
                }
            }

            gaps.resize(ng);
            if (ng > 0) {
                if (gap16)
                    byte_unsplit_16(raw.data() + msz, ng, gaps.data());
                else
                    byte_unsplit_32(raw.data() + msz, ng, gaps.data());
            }

            // Load perm for decode
            // (perm is shared, already decompressed by caller and passed via out_ix pattern)
        }
    }
    // NOTE: actual chunk decode with perm is done by the caller to avoid
    // threading the perm array through this helper. See pz_read_impl.
}

// ============================================================================
// Decompress .1pz (full read, returns dict)
// ============================================================================
template<typename ValT, typename IdxT = int32_t>
static py::dict pz_read_typed(const std::string& path, int nt) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read((char*)blob.data(), sz); f.close();
    if (sz < 112) throw std::runtime_error("File too small for .1pz");

    PZHeader hdr{};
    std::memcpy(&hdr, blob.data(), 96);
    if (hdr.magic != TP1_MAGIC)
        throw std::runtime_error("Not a .1pz file (wrong magic)");

    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("Unsupported .1pz version " + std::to_string(hdr.version));

    // Verify footer CRC
    if (sz >= 96 + 16) {
        PZFooter ftr;
        std::memcpy(&ftr, blob.data() + sz - 16, 16);
        if (ftr.magic != TP1_MAGIC)
            throw std::runtime_error(".1pz footer magic mismatch");
        uint32_t actual_crc = CRC32::compute(blob.data(), sz - 16);
        if (actual_crc != ftr.file_crc32)
            throw std::runtime_error(".1pz file CRC32 mismatch (corrupt file)");
    }

    int64_t m = hdr.m, n = hdr.n, nnz = (int64_t)hdr.nnz;
    int pw = hdr.ptr_width, nc = hdr.num_chunks, cc = (int)hdr.chunk_cols;
    bool gap16 = (hdr.flags & FLAG_GAP16) != 0;
    int gw = gap16 ? 2 : 4;
    uint8_t codec_id = hdr.reserved[0];
    bool has_bp = (hdr.feature_flags & FEAT_BITPLANE_BITMAP) != 0;

    size_t off = 96;
    const uint8_t* pz_data = blob.data() + off; off += hdr.perm_z_sz;
    const uint8_t* pt_data = blob.data() + off; off += hdr.ptr_z_sz;

    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), blob.data() + off, nc * 4);
    off += nc * 4;

    std::vector<size_t> coff(nc);
    for (int c = 0; c < nc; ++c) { coff[c] = off; off += ctable[c]; }

    // Decompress perm and ptrs
    ZstdDCtx dctx;
    std::vector<uint32_t> perm(m);
    zstd_decompress(dctx.ctx, perm.data(), m*4, pz_data, hdr.perm_z_sz);

    std::vector<uint8_t> pr(n * pw);
    zstd_decompress(dctx.ctx, pr.data(), pr.size(), pt_data, hdr.ptr_z_sz);

    py::array_t<IdxT> indptr(n + 1);
    auto ip = indptr.mutable_data();
    ip[0] = 0;
    if (pw == 2) {
        const uint16_t* c16 = (const uint16_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) ip[j+1] = ip[j] + (IdxT)c16[j];
    } else {
        const uint32_t* c32 = (const uint32_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) ip[j+1] = ip[j] + (IdxT)c32[j];
    }

    py::array_t<IdxT> indices(nnz);
    py::array_t<ValT> values(nnz);
    auto oix = indices.mutable_data();
    auto ovx = values.mutable_data();

    // Decode all chunks
    #pragma omp parallel num_threads(nt)
    {
        ZstdDCtx td;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> packed_buf, pf_buf;
        std::vector<uint32_t> gaps;

        #pragma omp for schedule(dynamic)
        for (int c = 0; c < nc; ++c) {
            int64_t cs = (int64_t)c * cc;
            int64_t ce = std::min(cs + (int64_t)cc, n);
            const uint8_t* cp = blob.data() + coff[c];

            uint32_t ng, msz, zsz;
            std::memcpy(&ng, cp, 4);  cp += 4;
            std::memcpy(&msz, cp, 4); cp += 4;
            std::memcpy(&zsz, cp, 4); cp += 4;

            uint32_t stored_crc = 0;
            std::memcpy(&stored_crc, cp, 4); cp += 4;

            uint32_t packed_sz = 0;
            if (has_bp) { std::memcpy(&packed_sz, cp, 4); cp += 4; }

            size_t raw_sz = msz + (size_t)ng * gw;

            if (has_bp) {
                packed_buf.resize(packed_sz);
                chunk_decompress(td.ctx, packed_buf.data(), packed_sz, cp, zsz, codec_id);

                {
                    uint32_t actual_crc = CRC32::compute(packed_buf.data(), packed_sz);
                    if (actual_crc != stored_crc)
                        throw std::runtime_error("Chunk CRC32 mismatch at chunk " + std::to_string(c));
                }

                size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
                size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
                pf_buf.resize(pf_sz);
                bitmap_unpack(packed_buf.data(), packed_sz, pf_buf.data(), pf_sz);

                raw.resize(raw_sz);
                std::memcpy(raw.data(), pf_buf.data(), msz);
                if (ng > 0) {
                    bit_planes_decode(pf_buf.data() + msz, ng, raw.data() + msz);
                    if (gw > 1)
                        std::memcpy(raw.data() + msz + ng,
                                    pf_buf.data() + msz + bp_sz, ng * (gw - 1));
                }
            } else {
                raw.resize(raw_sz);
                chunk_decompress(td.ctx, raw.data(), raw_sz, cp, zsz, codec_id);

                {
                    uint32_t actual_crc = CRC32::compute(raw.data(), raw_sz);
                    if (actual_crc != stored_crc)
                        throw std::runtime_error("Chunk CRC32 mismatch at chunk " + std::to_string(c));
                }
            }

            gaps.resize(ng);
            if (ng > 0) {
                if (gap16)
                    byte_unsplit_16(raw.data() + msz, ng, gaps.data());
                else
                    byte_unsplit_32(raw.data() + msz, ng, gaps.data());
            }

            vocsc_decode_chunk<ValT, IdxT>(raw.data(), gaps.data(), perm.data(),
                                     cs, ce, ip, oix, ovx);
        }
    }

    // Build result dict
    py::dict result;
    result["m"] = (int64_t)m;
    result["n"] = (int64_t)n;
    result["nnz"] = (int64_t)nnz;
    result["indptr"] = indptr;
    result["indices"] = indices;
    result["values"] = values;
    result["vt_code"] = (int)hdr.vt_code;

    // Read colsums if present
    if ((hdr.flags & FLAG_HAS_COLSUMS) && hdr.colsums_z_sz > 0) {
        // Colsums are right after the forward chunks
        size_t cs_off = 96 + hdr.perm_z_sz + hdr.ptr_z_sz + nc * 4;
        for (int c = 0; c < nc; ++c) cs_off += ctable[c];

        py::array_t<uint64_t> colsums_arr(n);
        zstd_decompress(dctx.ctx, colsums_arr.mutable_data(), n * 8,
                        blob.data() + cs_off, hdr.colsums_z_sz);
        result["colsums"] = colsums_arr;
    }

    // Read metadata if present
    if ((hdr.flags & FLAG_HAS_METADATA) && hdr.metadata_z_sz > 0
        && hdr.metadata_offset < sz) {
        const uint8_t* meta_src = blob.data() + hdr.metadata_offset;
        unsigned long long meta_decom_sz = ZSTD_getFrameContentSize(
            meta_src, hdr.metadata_z_sz);
        if (meta_decom_sz == ZSTD_CONTENTSIZE_UNKNOWN ||
            meta_decom_sz == ZSTD_CONTENTSIZE_ERROR)
            meta_decom_sz = (unsigned long long)hdr.metadata_z_sz * 100 + 65536;
        std::vector<uint8_t> meta_raw((size_t)meta_decom_sz);
        size_t msz = zstd_decompress(dctx.ctx, meta_raw.data(), meta_raw.size(),
                                     meta_src, hdr.metadata_z_sz);
        auto md = deserialize_metadata(meta_raw.data(), msz);
        if (!md.rownames.empty()) {
            py::list rn;
            for (const auto& s : md.rownames) rn.append(py::str(s));
            result["rownames"] = rn;
        }
        if (!md.colnames.empty()) {
            py::list cn;
            for (const auto& s : md.colnames) cn.append(py::str(s));
            result["colnames"] = cn;
        }
        if (!md.kv_pairs.empty()) {
            py::dict kv;
            for (const auto& [k, v] : md.kv_pairs)
                kv[py::str(k)] = py::str(v);
            result["uns"] = kv;
        }
        if (!md.obs_blob.empty())
            result["obs"] = deserialize_dataframe_py(md.obs_blob.data(), md.obs_blob.size());
        if (!md.var_blob.empty())
            result["var"] = deserialize_dataframe_py(md.var_blob.data(), md.var_blob.size());
    }

    return result;
}

// ============================================================================
// Column-range read
// ============================================================================
template<typename ValT, typename IdxT = int32_t>
static py::dict pz_read_columns_typed(const std::string& path,
                                       int64_t col_start, int64_t col_end,
                                       int nt) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read((char*)blob.data(), sz); f.close();

    PZHeader hdr{};
    std::memcpy(&hdr, blob.data(), 96);
    if (hdr.magic != TP1_MAGIC)
        throw std::runtime_error("Not a .1pz file");
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("Unsupported .1pz version");

    int64_t m = hdr.m, n = hdr.n;
    int pw = hdr.ptr_width, nc = hdr.num_chunks, cc = (int)hdr.chunk_cols;
    bool gap16 = (hdr.flags & FLAG_GAP16) != 0;
    int gw = gap16 ? 2 : 4;
    uint8_t codec_id = hdr.reserved[0];
    bool has_bp = (hdr.feature_flags & FEAT_BITPLANE_BITMAP) != 0;

    if (col_start < 0) col_start = 0;
    if (col_end > n) col_end = n;
    if (col_start >= col_end) {
        py::dict r;
        r["m"] = m; r["n"] = (int64_t)0; r["nnz"] = (int64_t)0;
        r["indptr"] = py::array_t<IdxT>(1);
        r["indices"] = py::array_t<IdxT>(0);
        r["values"] = py::array_t<ValT>(0);
        return r;
    }

    // Read perm and ptrs
    size_t off = 96;
    ZstdDCtx dctx;
    std::vector<uint32_t> perm(m);
    zstd_decompress(dctx.ctx, perm.data(), m*4, blob.data() + off, hdr.perm_z_sz);
    off += hdr.perm_z_sz;

    std::vector<uint8_t> pr(n * pw);
    zstd_decompress(dctx.ctx, pr.data(), pr.size(), blob.data() + off, hdr.ptr_z_sz);
    off += hdr.ptr_z_sz;

    // Build full indptr
    std::vector<IdxT> full_ip(n + 1);
    full_ip[0] = 0;
    if (pw == 2) {
        const uint16_t* c16 = (const uint16_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) full_ip[j+1] = full_ip[j] + (IdxT)c16[j];
    } else {
        const uint32_t* c32 = (const uint32_t*)pr.data();
        for (int64_t j = 0; j < n; ++j) full_ip[j+1] = full_ip[j] + (IdxT)c32[j];
    }

    // Chunk table
    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), blob.data() + off, nc * 4);
    off += nc * 4;

    std::vector<size_t> coff(nc);
    for (int c = 0; c < nc; ++c) { coff[c] = off; off += ctable[c]; }

    // Determine chunk range
    int c_start = (int)(col_start / cc);
    int c_end = (int)((col_end - 1) / cc) + 1;
    if (c_end > nc) c_end = nc;

    // Full chunk range cols
    int64_t full_cs = (int64_t)c_start * cc;
    int64_t full_ce = std::min((int64_t)c_end * cc, n);

    // Temp arrays for full chunk range
    int64_t base_nnz = full_ip[full_cs];
    int64_t range_nnz = full_ip[full_ce] - base_nnz;

    std::vector<IdxT> temp_ix(range_nnz);
    std::vector<ValT> temp_vx(range_nnz);

    // Decode relevant chunks
    #pragma omp parallel num_threads(nt)
    {
        ZstdDCtx td;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> packed_buf, pf_buf;
        std::vector<uint32_t> gaps;

        #pragma omp for schedule(dynamic)
        for (int c = c_start; c < c_end; ++c) {
            int64_t cs = (int64_t)c * cc;
            int64_t ce = std::min(cs + (int64_t)cc, n);
            const uint8_t* cp = blob.data() + coff[c];

            uint32_t ng, msz, zsz;
            std::memcpy(&ng, cp, 4);  cp += 4;
            std::memcpy(&msz, cp, 4); cp += 4;
            std::memcpy(&zsz, cp, 4); cp += 4;
            cp += 4;  // skip CRC

            uint32_t packed_sz = 0;
            if (has_bp) { std::memcpy(&packed_sz, cp, 4); cp += 4; }

            size_t raw_sz = msz + (size_t)ng * gw;

            if (has_bp) {
                packed_buf.resize(packed_sz);
                chunk_decompress(td.ctx, packed_buf.data(), packed_sz, cp, zsz, codec_id);

                size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
                size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (gw - 1) : 0);
                pf_buf.resize(pf_sz);
                bitmap_unpack(packed_buf.data(), packed_sz, pf_buf.data(), pf_sz);

                raw.resize(raw_sz);
                std::memcpy(raw.data(), pf_buf.data(), msz);
                if (ng > 0) {
                    bit_planes_decode(pf_buf.data() + msz, ng, raw.data() + msz);
                    if (gw > 1)
                        std::memcpy(raw.data() + msz + ng,
                                    pf_buf.data() + msz + bp_sz, ng * (gw - 1));
                }
            } else {
                raw.resize(raw_sz);
                chunk_decompress(td.ctx, raw.data(), raw_sz, cp, zsz, codec_id);
            }

            gaps.resize(ng);
            if (ng > 0) {
                if (gap16)
                    byte_unsplit_16(raw.data() + msz, ng, gaps.data());
                else
                    byte_unsplit_32(raw.data() + msz, ng, gaps.data());
            }

            // Decode with offset pointers
            vocsc_decode_chunk<ValT, IdxT>(
                raw.data(), gaps.data(), perm.data(),
                cs, ce, full_ip.data(),
                temp_ix.data() - base_nnz,
                temp_vx.data() - base_nnz);
        }
    }

    // Extract [col_start, col_end) from temp arrays
    int64_t n_out = col_end - col_start;
    int64_t local_nnz = full_ip[col_end] - full_ip[col_start];
    int64_t local_base = full_ip[col_start] - base_nnz;

    py::array_t<IdxT> out_indptr(n_out + 1);
    py::array_t<IdxT> out_indices(local_nnz);
    py::array_t<ValT> out_values(local_nnz);

    auto oip = out_indptr.mutable_data();
    oip[0] = 0;
    for (int64_t j = 0; j < n_out; ++j)
        oip[j+1] = (IdxT)(full_ip[col_start + j + 1] - full_ip[col_start]);

    if (local_nnz > 0) {
        std::memcpy(out_indices.mutable_data(),
                    temp_ix.data() + local_base, local_nnz * sizeof(IdxT));
        std::memcpy(out_values.mutable_data(),
                    temp_vx.data() + local_base, local_nnz * sizeof(ValT));
    }

    py::dict result;
    result["m"] = m;
    result["n"] = n_out;
    result["nnz"] = local_nnz;
    result["indptr"] = out_indptr;
    result["indices"] = out_indices;
    result["values"] = out_values;
    result["col_offset"] = col_start;
    return result;
}

// ============================================================================
// Row-range read (via transpose)
// ============================================================================
template<typename ValT, typename IdxT = int32_t>
static py::dict pz_read_rows_typed(const std::string& path,
                                    int64_t row_start, int64_t row_end,
                                    int nt) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t sz = f.tellg(); f.seekg(0);
    std::vector<uint8_t> blob(sz);
    f.read((char*)blob.data(), sz); f.close();

    PZHeader hdr{};
    std::memcpy(&hdr, blob.data(), 96);
    if (hdr.magic != TP1_MAGIC)
        throw std::runtime_error("Not a .1pz file");
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("Row-range reads require current .1pz format");
    if (!(hdr.flags & FLAG_HAS_TRANSPOSE) || hdr.transpose_offset == 0)
        throw std::runtime_error("File has no stored transpose; row-range reads unavailable");

    int64_t m = hdr.m, n = hdr.n;
    uint8_t codec_id = hdr.reserved[0];
    bool has_bp = (hdr.feature_flags & FEAT_BITPLANE_BITMAP) != 0;
    if (row_start < 0) row_start = 0;
    if (row_end > m) row_end = m;

    // Read transpose section
    size_t t_off = hdr.transpose_offset;
    TransposeHeader thdr;
    std::memcpy(&thdr, blob.data() + t_off, 16);
    t_off += 16;

    int t_pw = thdr.ptr_width;
    int t_nc = hdr.transpose_chunks;
    int t_cc = (int)thdr.chunk_cols;
    bool t_gap16 = (thdr.flags & 0x01) != 0;
    int t_gw = t_gap16 ? 2 : 4;

    // Transpose has n rows (original cols) and m columns (original rows)
    // Reading "columns" [row_start, row_end) of transpose = rows of original

    ZstdDCtx dctx;
    std::vector<uint32_t> t_perm(n);
    zstd_decompress(dctx.ctx, t_perm.data(), n*4,
                    blob.data() + t_off, thdr.perm_z_sz);
    t_off += thdr.perm_z_sz;

    std::vector<uint8_t> t_pr(m * t_pw);
    zstd_decompress(dctx.ctx, t_pr.data(), t_pr.size(),
                    blob.data() + t_off, thdr.ptr_z_sz);
    t_off += thdr.ptr_z_sz;

    // Build full transpose indptr (m+1 entries, columns of transpose = rows of original)
    std::vector<IdxT> t_ip(m + 1);
    t_ip[0] = 0;
    if (t_pw == 2) {
        const uint16_t* c16 = (const uint16_t*)t_pr.data();
        for (int64_t j = 0; j < m; ++j) t_ip[j+1] = t_ip[j] + (IdxT)c16[j];
    } else {
        const uint32_t* c32 = (const uint32_t*)t_pr.data();
        for (int64_t j = 0; j < m; ++j) t_ip[j+1] = t_ip[j] + (IdxT)c32[j];
    }

    // Chunk table
    std::vector<uint32_t> t_ctable(t_nc);
    std::memcpy(t_ctable.data(), blob.data() + t_off, t_nc * 4);
    t_off += t_nc * 4;

    std::vector<size_t> t_coff(t_nc);
    for (int c = 0; c < t_nc; ++c) { t_coff[c] = t_off; t_off += t_ctable[c]; }

    // Determine chunks
    int c_start = (int)(row_start / t_cc);
    int c_end = (int)((row_end - 1) / t_cc) + 1;
    if (c_end > t_nc) c_end = t_nc;

    int64_t full_cs = (int64_t)c_start * t_cc;
    int64_t full_ce = std::min((int64_t)c_end * t_cc, m);
    int64_t base_nnz = t_ip[full_cs];
    int64_t range_nnz = t_ip[full_ce] - base_nnz;

    std::vector<IdxT> temp_ix(range_nnz);
    std::vector<ValT> temp_vx(range_nnz);

    #pragma omp parallel num_threads(nt)
    {
        ZstdDCtx td;
        std::vector<uint8_t> raw;
        std::vector<uint8_t> packed_buf, pf_buf;
        std::vector<uint32_t> gaps;

        #pragma omp for schedule(dynamic)
        for (int c = c_start; c < c_end; ++c) {
            int64_t cs = (int64_t)c * t_cc;
            int64_t ce = std::min(cs + (int64_t)t_cc, m);
            const uint8_t* cp = blob.data() + t_coff[c];

            uint32_t ng, msz, zsz;
            std::memcpy(&ng, cp, 4); cp += 4;
            std::memcpy(&msz, cp, 4); cp += 4;
            std::memcpy(&zsz, cp, 4); cp += 4;
            cp += 4; // skip CRC

            uint32_t packed_sz = 0;
            if (has_bp) { std::memcpy(&packed_sz, cp, 4); cp += 4; }

            size_t raw_sz = msz + (size_t)ng * t_gw;

            if (has_bp) {
                packed_buf.resize(packed_sz);
                chunk_decompress(td.ctx, packed_buf.data(), packed_sz, cp, zsz, codec_id);

                size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
                size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * (t_gw - 1) : 0);
                pf_buf.resize(pf_sz);
                bitmap_unpack(packed_buf.data(), packed_sz, pf_buf.data(), pf_sz);

                raw.resize(raw_sz);
                std::memcpy(raw.data(), pf_buf.data(), msz);
                if (ng > 0) {
                    bit_planes_decode(pf_buf.data() + msz, ng, raw.data() + msz);
                    if (t_gw > 1)
                        std::memcpy(raw.data() + msz + ng,
                                    pf_buf.data() + msz + bp_sz, ng * (t_gw - 1));
                }
            } else {
                raw.resize(raw_sz);
                chunk_decompress(td.ctx, raw.data(), raw_sz, cp, zsz, codec_id);
            }

            gaps.resize(ng);
            if (ng > 0) {
                if (t_gap16)
                    byte_unsplit_16(raw.data() + msz, ng, gaps.data());
                else
                    byte_unsplit_32(raw.data() + msz, ng, gaps.data());
            }

            vocsc_decode_chunk<ValT, IdxT>(
                raw.data(), gaps.data(), t_perm.data(),
                cs, ce, t_ip.data(),
                temp_ix.data() - base_nnz,
                temp_vx.data() - base_nnz);
        }
    }

    // Extract [row_start, row_end)
    int64_t n_out = row_end - row_start;
    int64_t local_nnz = t_ip[row_end] - t_ip[row_start];
    int64_t local_base = t_ip[row_start] - base_nnz;

    py::array_t<IdxT> out_indptr(n_out + 1);
    py::array_t<IdxT> out_indices(local_nnz);
    py::array_t<ValT> out_values(local_nnz);

    auto oip = out_indptr.mutable_data();
    oip[0] = 0;
    for (int64_t j = 0; j < n_out; ++j)
        oip[j+1] = (IdxT)(t_ip[row_start + j + 1] - t_ip[row_start]);

    if (local_nnz > 0) {
        std::memcpy(out_indices.mutable_data(),
                    temp_ix.data() + local_base, local_nnz * sizeof(IdxT));
        std::memcpy(out_values.mutable_data(),
                    temp_vx.data() + local_base, local_nnz * sizeof(ValT));
    }

    // Result is a CSC submatrix of X^T[:, row_start:row_end]
    // = transpose of X[row_start:row_end, :]
    py::dict result;
    result["m"] = n;  // rows of transpose = original cols
    result["n"] = n_out;  // cols of transpose = selected original rows
    result["nnz"] = local_nnz;
    result["indptr"] = out_indptr;
    result["indices"] = out_indices;
    result["values"] = out_values;
    result["row_offset"] = row_start;
    result["is_transposed"] = true;  // caller must transpose CSC -> CSR
    return result;
}

// ============================================================================
// Read just colsums (fast, header + colsums section only)
// ============================================================================
static py::array_t<uint64_t> pz_colsums_impl(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);

    PZHeader hdr;
    f.read((char*)&hdr, 96);
    if (hdr.magic != TP1_MAGIC) throw std::runtime_error("Not a .1pz file");
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("Colsums require current .1pz format");
    if (!(hdr.flags & FLAG_HAS_COLSUMS) || hdr.colsums_z_sz == 0)
        throw std::runtime_error("File has no stored colsums");

    // Compute colsums offset: right after forward chunks
    size_t cs_off = 96 + hdr.perm_z_sz + hdr.ptr_z_sz + hdr.num_chunks * 4;
    // Need chunk sizes to compute total chunk data size
    f.seekg(96 + hdr.perm_z_sz + hdr.ptr_z_sz);
    std::vector<uint32_t> ctable(hdr.num_chunks);
    f.read((char*)ctable.data(), hdr.num_chunks * 4);
    for (uint32_t c = 0; c < hdr.num_chunks; ++c) cs_off += ctable[c];

    // Read colsums compressed blob
    std::vector<uint8_t> cs_z(hdr.colsums_z_sz);
    f.seekg(cs_off);
    f.read((char*)cs_z.data(), hdr.colsums_z_sz);
    f.close();

    py::array_t<uint64_t> colsums(hdr.n);
    ZstdDCtx dctx;
    zstd_decompress(dctx.ctx, colsums.mutable_data(), hdr.n * 8,
                    cs_z.data(), hdr.colsums_z_sz);
    return colsums;
}

// ============================================================================
// Read .1pz header + optional metadata
// ============================================================================
static py::dict pz_info_impl(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    size_t file_sz = f.tellg();
    f.seekg(0);

    PZHeader hdr;
    f.read((char*)&hdr, 96);
    if (hdr.magic != TP1_MAGIC)
        throw std::runtime_error("Not a .1pz file (wrong magic)");
    if (hdr.version != 1 && hdr.version != 3 && hdr.version != 4)
        throw std::runtime_error("Unsupported .1pz version");

    py::dict d;
    d["version"] = (int)hdr.version;
    d["m"] = hdr.m;
    d["n"] = hdr.n;
    d["nnz"] = hdr.nnz;
    d["vt_code"] = (int)hdr.vt_code;
    d["codec_level"] = (int)hdr.codec_level;
    d["num_chunks"] = hdr.num_chunks;
    d["gap16"] = (bool)(hdr.flags & FLAG_GAP16);
    d["ptr_width"] = (int)hdr.ptr_width;
    d["codec"] = "vocsc+zstd";
    d["chunk_cols"] = hdr.chunk_cols;
    d["has_metadata"] = (bool)(hdr.flags & FLAG_HAS_METADATA);
    d["has_colsums"] = (bool)(hdr.flags & FLAG_HAS_COLSUMS);
    d["has_transpose"] = (bool)(hdr.flags & FLAG_HAS_TRANSPOSE);
    d["has_obs_var"] = (bool)(hdr.flags & FLAG_HAS_OBS_VAR);
    d["zstd_checksums"] = (bool)(hdr.feature_flags & FEAT_ZSTD_CHECKSUMS);
    d["bitplane_bitmap"] = (bool)(hdr.feature_flags & FEAT_BITPLANE_BITMAP);
    d["file_size"] = file_sz;

    // Verify footer CRC
    if (file_sz >= 112) {
        f.seekg(file_sz - 16);
        PZFooter ftr;
        f.read((char*)&ftr, 16);
        d["footer_valid"] = (ftr.magic == TP1_MAGIC);
    }
    f.close();
    return d;
}

// ============================================================================
// Validate .1pz file (CRC check without full decompression)
// ============================================================================
static py::dict pz_validate_impl(const std::string& path) {
    py::dict r;
    r["valid"] = false;
    r["file_crc_ok"] = false;
    r["footer_ok"] = false;
    r["error"] = py::none();

    try {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) { r["error"] = "Cannot open file"; return r; }
        size_t sz = f.tellg(); f.seekg(0);
        if (sz < 112) { r["error"] = "File too small"; return r; }

        std::vector<uint8_t> blob(sz);
        f.read((char*)blob.data(), sz); f.close();

        PZHeader hdr;
        std::memcpy(&hdr, blob.data(), 96);
        if (hdr.magic != TP1_MAGIC) { r["error"] = "Wrong magic"; return r; }
        r["version"] = (int)hdr.version;

        PZFooter ftr;
        std::memcpy(&ftr, blob.data() + sz - 16, 16);
        r["footer_ok"] = (ftr.magic == TP1_MAGIC);

        uint32_t actual_crc = CRC32::compute(blob.data(), sz - 16);
        r["file_crc_ok"] = (actual_crc == ftr.file_crc32);
        r["valid"] = (actual_crc == ftr.file_crc32) && (ftr.magic == TP1_MAGIC);
    } catch (const std::exception& e) {
        r["error"] = std::string(e.what());
    }
    return r;
}

// ============================================================================
// Quick header nnz read (for int32/int64 dispatch)
// ============================================================================
static uint64_t read_pz_nnz(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    PZHeader hdr{};
    f.read((char*)&hdr, 96);
    if (hdr.magic != TP1_MAGIC)
        throw std::runtime_error("Not a .1pz file (wrong magic)");
    return hdr.nnz;
}

// ============================================================================
// Python bindings
// ============================================================================
PYBIND11_MODULE(_pz_codec, m) {
    m.doc() = ".1pz codec: VOCSC + byte-split + zstd-3 with CRC, metadata, colsums, partial reads";

    // --- Write (float64 values) ---
    m.def("pz_write", [](
        py::array_t<int32_t> indptr,
        py::array_t<int32_t> indices,
        py::array_t<double>  data_arr,
        int32_t nrows,
        const std::string& path,
        int num_threads, int level, int chunk_cols,
        std::vector<std::string> rownames,
        std::vector<std::string> colnames,
        bool store_transpose,
        std::vector<std::pair<std::string,std::string>> kv_pairs,
        py::list obs_index,
        py::list obs_columns,
        py::list var_index,
        py::list var_columns,
        int codec_id)
    {
        auto ip = indptr.request();
        auto ix = indices.request();
        auto dx = data_arr.request();
        if (ip.ndim != 1 || ix.ndim != 1 || dx.ndim != 1)
            throw std::runtime_error("All inputs must be 1-D arrays");

        uint32_t n_val = (uint32_t)(ip.shape[0] - 1);
        uint32_t m_val = (uint32_t)nrows;
        uint64_t nnz = (uint64_t)ix.shape[0];

        std::vector<uint32_t> p_buf(n_val + 1), i_buf(nnz);
        auto* pp = static_cast<int32_t*>(ip.ptr);
        auto* pi = static_cast<int32_t*>(ix.ptr);
        for (uint32_t j = 0; j <= n_val; ++j) p_buf[j] = (uint32_t)pp[j];
        for (uint64_t k = 0; k < nnz; ++k) i_buf[k] = (uint32_t)pi[k];

        // Serialize DataFrames to native binary blobs
        std::vector<uint8_t> obs_blob, var_blob;
        if (py::len(obs_columns) > 0)
            obs_blob = serialize_dataframe_py(n_val, obs_index, obs_columns);
        if (py::len(var_columns) > 0)
            var_blob = serialize_dataframe_py(m_val, var_index, var_columns);

        return pz_write_core(
            p_buf.data(), i_buf.data(), static_cast<double*>(dx.ptr),
            m_val, n_val, nnz, path, num_threads, level, chunk_cols,
            rownames, colnames, store_transpose, kv_pairs, obs_blob, var_blob,
            (uint8_t)codec_id);
    }, py::arg("indptr"), py::arg("indices"), py::arg("data"),
       py::arg("nrows"), py::arg("path"),
       py::arg("num_threads")=8, py::arg("level")=3, py::arg("chunk_cols")=1024,
       py::arg("rownames")=std::vector<std::string>(),
       py::arg("colnames")=std::vector<std::string>(),
       py::arg("store_transpose")=false,
       py::arg("kv_pairs")=std::vector<std::pair<std::string,std::string>>(),
       py::arg("obs_index")=py::list(),
       py::arg("obs_columns")=py::list(),
       py::arg("var_index")=py::list(),
       py::arg("var_columns")=py::list(),
       py::arg("codec_id")=0,
       "Write CSC arrays to a .1pz file with optional obs/var/uns metadata.");

    // --- Write (int32 values) ---
    m.def("pz_write_int", [](
        py::array_t<int32_t> indptr,
        py::array_t<int32_t> indices,
        py::array_t<int32_t> data_arr,
        int32_t nrows,
        const std::string& path,
        int num_threads, int level, int chunk_cols,
        std::vector<std::string> rownames,
        std::vector<std::string> colnames,
        bool store_transpose,
        std::vector<std::pair<std::string,std::string>> kv_pairs,
        py::list obs_index,
        py::list obs_columns,
        py::list var_index,
        py::list var_columns,
        int codec_id)
    {
        auto ip = indptr.request();
        auto ix = indices.request();
        auto dx = data_arr.request();
        if (ip.ndim != 1 || ix.ndim != 1 || dx.ndim != 1)
            throw std::runtime_error("All inputs must be 1-D arrays");

        uint32_t n_val = (uint32_t)(ip.shape[0] - 1);
        uint32_t m_val = (uint32_t)nrows;
        uint64_t nnz = (uint64_t)ix.shape[0];

        std::vector<uint32_t> p_buf(n_val + 1), i_buf(nnz);
        std::vector<double> x_buf(nnz);
        auto* pp = static_cast<int32_t*>(ip.ptr);
        auto* pi = static_cast<int32_t*>(ix.ptr);
        auto* pd = static_cast<int32_t*>(dx.ptr);
        for (uint32_t j = 0; j <= n_val; ++j) p_buf[j] = (uint32_t)pp[j];
        for (uint64_t k = 0; k < nnz; ++k) {
            i_buf[k] = (uint32_t)pi[k];
            x_buf[k] = (double)pd[k];
        }

        // Serialize DataFrames to native binary blobs
        std::vector<uint8_t> obs_blob, var_blob;
        if (py::len(obs_columns) > 0)
            obs_blob = serialize_dataframe_py(n_val, obs_index, obs_columns);
        if (py::len(var_columns) > 0)
            var_blob = serialize_dataframe_py(m_val, var_index, var_columns);

        return pz_write_core(
            p_buf.data(), i_buf.data(), x_buf.data(),
            m_val, n_val, nnz, path, num_threads, level, chunk_cols,
            rownames, colnames, store_transpose, kv_pairs, obs_blob, var_blob,
            (uint8_t)codec_id);
    }, py::arg("indptr"), py::arg("indices"), py::arg("data"),
       py::arg("nrows"), py::arg("path"),
       py::arg("num_threads")=8, py::arg("level")=3, py::arg("chunk_cols")=1024,
       py::arg("rownames")=std::vector<std::string>(),
       py::arg("colnames")=std::vector<std::string>(),
       py::arg("store_transpose")=false,
       py::arg("kv_pairs")=std::vector<std::pair<std::string,std::string>>(),
       py::arg("obs_index")=py::list(),
       py::arg("obs_columns")=py::list(),
       py::arg("var_index")=py::list(),
       py::arg("var_columns")=py::list(),
       py::arg("codec_id")=0,
       "Write CSC int32 arrays to a .1pz file with optional obs/var/uns metadata.");

    // --- Write (float64 values, int64 indices) ---
    m.def("pz_write_i64", [](
        py::array_t<int64_t> indptr,
        py::array_t<int64_t> indices,
        py::array_t<double>  data_arr,
        int32_t nrows,
        const std::string& path,
        int num_threads, int level, int chunk_cols,
        std::vector<std::string> rownames,
        std::vector<std::string> colnames,
        bool store_transpose,
        std::vector<std::pair<std::string,std::string>> kv_pairs,
        py::list obs_index,
        py::list obs_columns,
        py::list var_index,
        py::list var_columns,
        int codec_id)
    {
        auto ip = indptr.request();
        auto ix = indices.request();
        auto dx = data_arr.request();
        if (ip.ndim != 1 || ix.ndim != 1 || dx.ndim != 1)
            throw std::runtime_error("All inputs must be 1-D arrays");

        uint32_t n_val = (uint32_t)(ip.shape[0] - 1);
        uint32_t m_val = (uint32_t)nrows;
        uint64_t nnz = (uint64_t)ix.shape[0];

        std::vector<uint32_t> p_buf(n_val + 1), i_buf(nnz);
        auto* pp = static_cast<int64_t*>(ip.ptr);
        auto* pi = static_cast<int64_t*>(ix.ptr);
        for (uint32_t j = 0; j <= n_val; ++j) p_buf[j] = (uint32_t)pp[j];
        for (uint64_t k = 0; k < nnz; ++k) i_buf[k] = (uint32_t)pi[k];

        std::vector<uint8_t> obs_blob, var_blob;
        if (py::len(obs_columns) > 0)
            obs_blob = serialize_dataframe_py(n_val, obs_index, obs_columns);
        if (py::len(var_columns) > 0)
            var_blob = serialize_dataframe_py(m_val, var_index, var_columns);

        return pz_write_core(
            p_buf.data(), i_buf.data(), static_cast<double*>(dx.ptr),
            m_val, n_val, nnz, path, num_threads, level, chunk_cols,
            rownames, colnames, store_transpose, kv_pairs, obs_blob, var_blob,
            (uint8_t)codec_id);
    }, py::arg("indptr"), py::arg("indices"), py::arg("data"),
       py::arg("nrows"), py::arg("path"),
       py::arg("num_threads")=8, py::arg("level")=3, py::arg("chunk_cols")=1024,
       py::arg("rownames")=std::vector<std::string>(),
       py::arg("colnames")=std::vector<std::string>(),
       py::arg("store_transpose")=false,
       py::arg("kv_pairs")=std::vector<std::pair<std::string,std::string>>(),
       py::arg("obs_index")=py::list(),
       py::arg("obs_columns")=py::list(),
       py::arg("var_index")=py::list(),
       py::arg("var_columns")=py::list(),
       py::arg("codec_id")=0,
       "Write CSC arrays with int64 indices to a .1pz file.");

    // --- Write (int32 values, int64 indices) ---
    m.def("pz_write_int_i64", [](
        py::array_t<int64_t> indptr,
        py::array_t<int64_t> indices,
        py::array_t<int32_t> data_arr,
        int32_t nrows,
        const std::string& path,
        int num_threads, int level, int chunk_cols,
        std::vector<std::string> rownames,
        std::vector<std::string> colnames,
        bool store_transpose,
        std::vector<std::pair<std::string,std::string>> kv_pairs,
        py::list obs_index,
        py::list obs_columns,
        py::list var_index,
        py::list var_columns,
        int codec_id)
    {
        auto ip = indptr.request();
        auto ix = indices.request();
        auto dx = data_arr.request();
        if (ip.ndim != 1 || ix.ndim != 1 || dx.ndim != 1)
            throw std::runtime_error("All inputs must be 1-D arrays");

        uint32_t n_val = (uint32_t)(ip.shape[0] - 1);
        uint32_t m_val = (uint32_t)nrows;
        uint64_t nnz = (uint64_t)ix.shape[0];

        std::vector<uint32_t> p_buf(n_val + 1), i_buf(nnz);
        std::vector<double> x_buf(nnz);
        auto* pp = static_cast<int64_t*>(ip.ptr);
        auto* pi = static_cast<int64_t*>(ix.ptr);
        auto* pd = static_cast<int32_t*>(dx.ptr);
        for (uint32_t j = 0; j <= n_val; ++j) p_buf[j] = (uint32_t)pp[j];
        for (uint64_t k = 0; k < nnz; ++k) {
            i_buf[k] = (uint32_t)pi[k];
            x_buf[k] = (double)pd[k];
        }

        std::vector<uint8_t> obs_blob, var_blob;
        if (py::len(obs_columns) > 0)
            obs_blob = serialize_dataframe_py(n_val, obs_index, obs_columns);
        if (py::len(var_columns) > 0)
            var_blob = serialize_dataframe_py(m_val, var_index, var_columns);

        return pz_write_core(
            p_buf.data(), i_buf.data(), x_buf.data(),
            m_val, n_val, nnz, path, num_threads, level, chunk_cols,
            rownames, colnames, store_transpose, kv_pairs, obs_blob, var_blob,
            (uint8_t)codec_id);
    }, py::arg("indptr"), py::arg("indices"), py::arg("data"),
       py::arg("nrows"), py::arg("path"),
       py::arg("num_threads")=8, py::arg("level")=3, py::arg("chunk_cols")=1024,
       py::arg("rownames")=std::vector<std::string>(),
       py::arg("colnames")=std::vector<std::string>(),
       py::arg("store_transpose")=false,
       py::arg("kv_pairs")=std::vector<std::pair<std::string,std::string>>(),
       py::arg("obs_index")=py::list(),
       py::arg("obs_columns")=py::list(),
       py::arg("var_index")=py::list(),
       py::arg("var_columns")=py::list(),
       py::arg("codec_id")=0,
       "Write CSC int32 arrays with int64 indices to a .1pz file.");

    // --- Read (float64 values) ---
    m.def("pz_read", [](const std::string& path, int nt) {
        if (read_pz_nnz(path) > (uint64_t)INT32_MAX)
            return pz_read_typed<double, int64_t>(path, nt);
        return pz_read_typed<double, int32_t>(path, nt);
    }, py::arg("path"), py::arg("num_threads")=8,
       "Read a .1pz file -> dict {m, n, nnz, indptr, indices, values, rownames, colnames, colsums}");

    // --- Read (int32 values) ---
    m.def("pz_read_int", [](const std::string& path, int nt) {
        if (read_pz_nnz(path) > (uint64_t)INT32_MAX)
            return pz_read_typed<int32_t, int64_t>(path, nt);
        return pz_read_typed<int32_t, int32_t>(path, nt);
    }, py::arg("path"), py::arg("num_threads")=8,
       "Read a .1pz file with int32 values.");

    // --- Column-range read ---
    m.def("pz_read_columns", [](const std::string& path,
                                 int64_t col_start, int64_t col_end, int nt) {
        if (read_pz_nnz(path) > (uint64_t)INT32_MAX)
            return pz_read_columns_typed<double, int64_t>(path, col_start, col_end, nt);
        return pz_read_columns_typed<double, int32_t>(path, col_start, col_end, nt);
    }, py::arg("path"), py::arg("col_start"), py::arg("col_end"),
       py::arg("num_threads")=8,
       "Read columns [col_start, col_end) from a .1pz file.");

    // --- Column-range read (int32) ---
    m.def("pz_read_columns_int", [](const std::string& path,
                                     int64_t col_start, int64_t col_end, int nt) {
        if (read_pz_nnz(path) > (uint64_t)INT32_MAX)
            return pz_read_columns_typed<int32_t, int64_t>(path, col_start, col_end, nt);
        return pz_read_columns_typed<int32_t, int32_t>(path, col_start, col_end, nt);
    }, py::arg("path"), py::arg("col_start"), py::arg("col_end"),
       py::arg("num_threads")=8,
       "Read columns [col_start, col_end) from a .1pz file with int32 values.");

    // --- Row-range read (via transpose) ---
    m.def("pz_read_rows", [](const std::string& path,
                               int64_t row_start, int64_t row_end, int nt) {
        if (read_pz_nnz(path) > (uint64_t)INT32_MAX)
            return pz_read_rows_typed<double, int64_t>(path, row_start, row_end, nt);
        return pz_read_rows_typed<double, int32_t>(path, row_start, row_end, nt);
    }, py::arg("path"), py::arg("row_start"), py::arg("row_end"),
       py::arg("num_threads")=8,
       "Read rows [row_start, row_end) via stored transpose.");

    // --- Read colsums only ---
    m.def("pz_colsums", &pz_colsums_impl,
          py::arg("path"),
          "Read column sums without full decompression.");

    // --- Info ---
    m.def("pz_info", &pz_info_impl,
          py::arg("path"),
          "Read .1pz file header.");

    // --- Validate ---
    m.def("pz_validate", &pz_validate_impl,
          py::arg("path"),
          "Validate .1pz file CRC integrity.");
}
