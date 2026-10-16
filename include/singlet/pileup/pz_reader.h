// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: pz_reader.h
// Native .1pz (VOCSC) reader — bit-exact inverse of pz_writer.h.
//
// This header re-uses the structs and helpers from pz_writer.h
// (PZHeader, PZFooter, CRC32, TP1_MAGIC, TP1_VERSION, FLAG_* constants),
// so it must be compiled with pz_writer.h also in scope.
//
// Entry points:
//   read_1pz(path)                -> ReadResult
//   read_1pz_buffer(data, size)   -> ReadResult   (no file I/O; used by tests and fuzzers)
//
// The returned result contains a CSC matrix, rownames/colnames, and an
// optional user key-value map — everything that write_1pz() embedded.
//
// Only the one layout that write_1pz() emits is accepted: version 1, feature
// flags ZSTD_CHECKSUMS | BITPLANE_BITMAP, 16-bit or 32-bit column counts.
// Anything else is rejected with PZReadError. The reader treats every byte of
// the input as untrusted: the CRC32 only detects corruption, it does not
// authenticate, so every size, offset and index is bounds-checked and every
// allocation is capped by ReadLimits.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <zstd.h>

#include "pz_writer.h"  // for PZHeader, PZFooter, CRC32, constants, and helpers

namespace singlet {
namespace pz {

// ============================================================================
// ReadResult — everything read_1pz() returns
// ============================================================================
struct ReadResult {
    uint32_t m = 0;                     // rows (features)
    uint32_t n = 0;                     // cols (cells)
    uint64_t nnz = 0;                   // total non-zeros
    uint8_t  vt_code = 0;               // value type code (1=uint8, 2=uint16, 3=uint32)

    // CSC layout: indptr has length n+1, indices/data have length nnz.
    // Values are widened to uint32_t regardless of vt_code to keep the API
    // uniform; callers that want the native narrow width can downcast.
    std::vector<uint32_t> indptr;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> data;

    std::vector<std::string> rownames;
    std::vector<std::string> colnames;
    std::map<std::string, std::string> user_kv;
};

// ============================================================================
// ReadLimits — hard caps applied before any header-driven allocation
// ============================================================================
struct ReadLimits {
    uint64_t max_file_bytes = 1ull << 33;   // 8 GiB
    uint64_t max_rows       = 1ull << 22;   // 4,194,304 features
    uint64_t max_cols       = 1ull << 24;   // 16,777,216 cells
    uint64_t max_nnz        = (1ull << 31) - 1;
    uint32_t max_chunk_cols = 1u << 20;
};

// ============================================================================
// PZReadError — thrown on any structural / CRC / decode failure
// ============================================================================
class PZReadError : public std::runtime_error {
public:
    explicit PZReadError(const std::string& msg) : std::runtime_error(msg) {}
};

// ============================================================================
// zstd helpers (decompress)
// ============================================================================
struct ZstdDCtx {
    ZSTD_DCtx* ctx;
    ZstdDCtx() : ctx(ZSTD_createDCtx()) {
        if (!ctx) throw PZReadError("zstd: cannot allocate decompression context");
    }
    ~ZstdDCtx() { if (ctx) ZSTD_freeDCtx(ctx); }
    ZstdDCtx(const ZstdDCtx&) = delete;
    ZstdDCtx& operator=(const ZstdDCtx&) = delete;
};

// Decompress one zstd frame whose declared content size must not exceed
// ``max_out``. The declared size is checked before anything is allocated, so
// a crafted frame cannot request an arbitrary buffer.
static inline std::vector<uint8_t>
zstd_decompress(ZSTD_DCtx* ctx, const uint8_t* data, size_t sz, size_t max_out) {
    unsigned long long dsz = ZSTD_getFrameContentSize(data, sz);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw PZReadError("zstd: frame content size unknown or error");
    }
    if (dsz > max_out) {
        throw PZReadError("zstd: frame content size exceeds the bound for this block");
    }
    std::vector<uint8_t> out(static_cast<size_t>(dsz));
    size_t got = ZSTD_decompressDCtx(ctx, out.data(), out.size(), data, sz);
    if (ZSTD_isError(got)) {
        throw PZReadError(std::string("zstd decompress: ") + ZSTD_getErrorName(got));
    }
    out.resize(got);
    return out;
}

// ============================================================================
// Inverse helpers (must bit-exactly reverse the transforms in pz_writer.h)
// ============================================================================

// varint_read: inverse of varint_push. Reads a LEB128-encoded uint32 from
// ``buf[offset..len)`` and advances ``offset`` past the terminator.
static inline uint32_t varint_read(const uint8_t* buf, size_t len, size_t& offset) {
    uint32_t v = 0;
    int shift = 0;
    while (true) {
        if (offset >= len) throw PZReadError("varint: read past end of block");
        uint8_t b = buf[offset++];
        if (shift == 28 && (b & 0xF0) != 0) throw PZReadError("varint overflow");
        v |= static_cast<uint32_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return v;
        shift += 7;
        if (shift > 28) throw PZReadError("varint overflow");
    }
}

// bitmap_unpack: inverse of pz_writer's bitmap_pack.
// Input:
//   ``src`` — packed buffer of size ``packed_sz``. First (pf_sz+7)/8 bytes
//             are the bitmap, then one byte per set bit in the bitmap.
//   ``pf_sz`` — the expected size of the output buffer (i.e. the original
//             pre-pack prefilter size).
// Output:
//   ``dst`` — pf_sz bytes with original non-zero bytes restored at their
//             original positions and zero-bytes elsewhere.
static inline void bitmap_unpack(
    const uint8_t* src, size_t packed_sz,
    uint8_t* dst, size_t pf_sz)
{
    size_t bm_bytes = (pf_sz + 7) / 8;
    if (packed_sz < bm_bytes) {
        throw PZReadError("bitmap_unpack: packed size smaller than bitmap");
    }
    std::memset(dst, 0, pf_sz);
    size_t wp = bm_bytes;
    for (size_t i = 0; i < pf_sz; ++i) {
        if (src[i / 8] & (1u << (i & 7))) {
            if (wp >= packed_sz) {
                throw PZReadError("bitmap_unpack: ran off end of packed buffer");
            }
            dst[i] = src[wp++];
        }
    }
    if (wp != packed_sz) {
        throw PZReadError("bitmap_unpack: trailing bytes in packed buffer");
    }
}

// bit_planes_decode: inverse of pz_writer's bit_planes_encode.
// Reads 8*pb bytes from ``src`` where ``pb = (n+7)/8`` and reconstructs
// the ``n`` original byte values into ``dst``.
static inline void bit_planes_decode(
    const uint8_t* src, size_t n, uint8_t* dst)
{
    if (n == 0) return;
    size_t pb = (n + 7) / 8;
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = 0;
        size_t byte_idx = i / 8;
        uint8_t bit_mask = 1u << (i & 7);
        for (int b = 0; b < 8; ++b) {
            if (src[b * pb + byte_idx] & bit_mask) {
                v |= static_cast<uint8_t>(1 << b);
            }
        }
        dst[i] = v;
    }
}

// ============================================================================
// Metadata TLV tags (must match pz_writer.h)
// ============================================================================
// META_TAG_END / ROWNAMES / COLNAMES / USER_KV are defined in pz_writer.h.

// ============================================================================
// Parse a single TLV stream of (tag:1, length:4, payload:length) records
// terminated by META_TAG_END. Populates rownames, colnames, user_kv.
// ``max_names`` caps the number of row/column names a block may hold.
// ============================================================================
static inline void parse_metadata_tlv(
    const uint8_t* buf, size_t len,
    std::vector<std::string>& rownames,
    std::vector<std::string>& colnames,
    std::map<std::string, std::string>& user_kv,
    uint64_t max_rownames, uint64_t max_colnames)
{
    size_t offset = 0;
    while (offset < len) {
        uint8_t tag = buf[offset++];
        if (tag == META_TAG_END) break;
        if (len - offset < 4) {
            throw PZReadError("metadata TLV: truncated length field");
        }
        uint32_t length;
        std::memcpy(&length, buf + offset, 4);
        offset += 4;
        if (static_cast<uint64_t>(length) > len - offset) {
            throw PZReadError("metadata TLV: payload extends past end of block");
        }

        if (tag == META_TAG_ROWNAMES || tag == META_TAG_COLNAMES) {
            const bool is_row = (tag == META_TAG_ROWNAMES);
            auto& dst = is_row ? rownames : colnames;
            const uint64_t cap = is_row ? max_rownames : max_colnames;
            size_t p = offset;
            size_t end = offset + length;
            while (p < end) {
                size_t start = p;
                while (p < end && buf[p] != 0) ++p;
                if (dst.size() >= cap) {
                    throw PZReadError("metadata TLV: more names than rows/columns");
                }
                dst.emplace_back(reinterpret_cast<const char*>(buf + start), p - start);
                if (p < end && buf[p] == 0) ++p;  // skip NUL
            }
        } else if (tag == META_TAG_USER_KV) {
            size_t p = offset;
            size_t end = offset + length;
            while (p < end) {
                size_t ks = p;
                while (p < end && buf[p] != 0) ++p;
                if (p >= end) throw PZReadError("user_kv: unterminated key");
                std::string key(reinterpret_cast<const char*>(buf + ks), p - ks);
                ++p;  // skip key NUL
                size_t vs = p;
                while (p < end && buf[p] != 0) ++p;
                if (p >= end) throw PZReadError("user_kv: unterminated value");
                std::string value(reinterpret_cast<const char*>(buf + vs), p - vs);
                ++p;  // skip value NUL
                user_kv[std::move(key)] = std::move(value);
            }
        }
        // Unknown tags: skip their payload for forward compatibility.
        offset += length;
    }
}

// ============================================================================
// Validate header + footer + whole-file CRC32 of an in-memory file image and
// return the parsed header. Every structural field the later stages rely on
// is range-checked here, against ``limits`` and the actual buffer size.
// ============================================================================
static inline PZHeader validate_buffer(
    const uint8_t* buf, size_t size, const ReadLimits& limits)
{
    if (size < sizeof(PZHeader) + sizeof(PZFooter)) {
        throw PZReadError("file too short (" + std::to_string(size) + " bytes)");
    }
    if (static_cast<uint64_t>(size) > limits.max_file_bytes) {
        throw PZReadError("file larger than the configured limit");
    }

    PZHeader hdr;
    std::memcpy(&hdr, buf, sizeof(PZHeader));
    if (hdr.magic != TP1_MAGIC) throw PZReadError("bad magic");
    if (hdr.version != TP1_VERSION) {
        throw PZReadError("unsupported version " + std::to_string(hdr.version) +
                          " (only version " + std::to_string(TP1_VERSION) + " is supported)");
    }

    PZFooter ftr;
    std::memcpy(&ftr, buf + size - sizeof(PZFooter), sizeof(PZFooter));
    if (ftr.magic != TP1_MAGIC) throw PZReadError("bad footer magic");
    if (ftr.num_chunks != hdr.num_chunks) {
        throw PZReadError("chunk count mismatch header/footer");
    }

    // The one layout write_1pz() emits.
    constexpr uint8_t  kKnownFlags = FLAG_HAS_PERM | FLAG_GAP16 | FLAG_HAS_METADATA | FLAG_HAS_COLSUMS;
    constexpr uint32_t kFeatures   = FEAT_ZSTD_CHECKSUMS | FEAT_BITPLANE_BITMAP;
    if (hdr.vt_code < 1 || hdr.vt_code > 3) throw PZReadError("unsupported value type code");
    if ((hdr.flags & ~kKnownFlags) != 0)    throw PZReadError("unknown header flags");
    if (!(hdr.flags & FLAG_HAS_PERM))       throw PZReadError("header lacks the permutation block");
    if (hdr.feature_flags != kFeatures)     throw PZReadError("unsupported feature flags");
    if (hdr.ptr_width != 2 && hdr.ptr_width != 4) throw PZReadError("unsupported ptr_width");

    // Dimensions and allocation caps (checked before anything is sized from them).
    if (hdr.m > limits.max_rows) throw PZReadError("row count exceeds the configured limit");
    if (hdr.n > limits.max_cols) throw PZReadError("column count exceeds the configured limit");
    if (hdr.nnz > limits.max_nnz || hdr.nnz > std::numeric_limits<uint32_t>::max()) {
        throw PZReadError("nnz exceeds the configured limit");
    }
    if (hdr.nnz > static_cast<uint64_t>(hdr.m) * hdr.n) {
        throw PZReadError("nnz exceeds m*n");
    }

    // Chunk geometry: the chunks must tile the columns exactly.
    if (hdr.n == 0) {
        if (hdr.num_chunks != 0) throw PZReadError("chunks present for an empty matrix");
    } else {
        if (hdr.chunk_cols == 0 || hdr.chunk_cols > limits.max_chunk_cols) {
            throw PZReadError("chunk_cols out of range");
        }
        const uint64_t expect = (static_cast<uint64_t>(hdr.n) + hdr.chunk_cols - 1) / hdr.chunk_cols;
        if (hdr.num_chunks != expect) throw PZReadError("num_chunks does not cover the columns");
    }

    // Block sizes must fit inside the body (everything between header and footer).
    const uint64_t body_end = size - sizeof(PZFooter);
    if (hdr.perm_z_sz == 0 || hdr.ptr_z_sz == 0) throw PZReadError("empty perm/ptr block");
    const uint64_t table_off = sizeof(PZHeader) + static_cast<uint64_t>(hdr.perm_z_sz) + hdr.ptr_z_sz;
    if (table_off + 4ull * hdr.num_chunks > body_end) {
        throw PZReadError("perm/ptr/chunk table extend past the end of the file");
    }
    if (hdr.flags & FLAG_HAS_METADATA) {
        if (hdr.metadata_z_sz == 0) throw PZReadError("FLAG_HAS_METADATA set but metadata is empty");
        if (hdr.metadata_offset < sizeof(PZHeader) ||
            hdr.metadata_offset > body_end ||
            hdr.metadata_z_sz > body_end - hdr.metadata_offset) {
            throw PZReadError("metadata block offset/size out of range");
        }
    }

    // CRC32 over file body (everything before the footer).
    if (CRC32::compute(buf, size - sizeof(PZFooter)) != ftr.file_crc32) {
        throw PZReadError("file CRC32 mismatch");
    }
    return hdr;
}

// ============================================================================
// Decode the permutation block and the column-count ptr block into two
// vectors. Populates ``perm`` (length m, a verified permutation of [0, m)) and
// ``cc`` (length n). Advances ``body_offset`` past the end of both blocks.
// ============================================================================
static inline void decode_perm_and_ptr(
    const uint8_t* body, size_t& body_offset,
    const PZHeader& hdr,
    std::vector<uint32_t>& perm,
    std::vector<uint32_t>& cc)
{
    ZstdDCtx dctx;

    // Permutation block
    const size_t perm_bytes = static_cast<size_t>(hdr.m) * 4;
    auto raw = zstd_decompress(dctx.ctx, body + body_offset, hdr.perm_z_sz, perm_bytes);
    if (raw.size() != perm_bytes) {
        throw PZReadError("perm block wrong size after decompression");
    }
    perm.resize(hdr.m);
    if (perm_bytes) std::memcpy(perm.data(), raw.data(), perm_bytes);
    {
        std::vector<uint8_t> seen(hdr.m, 0);
        for (uint32_t i = 0; i < hdr.m; ++i) {
            if (perm[i] >= hdr.m || seen[perm[i]]) {
                throw PZReadError("perm block is not a permutation of [0, m)");
            }
            seen[perm[i]] = 1;
        }
    }
    body_offset += hdr.perm_z_sz;

    // Ptr block — compressed column-count array (ptr_width bytes per column)
    const size_t ptr_bytes = static_cast<size_t>(hdr.n) * hdr.ptr_width;
    auto raw_ptr = zstd_decompress(dctx.ctx, body + body_offset, hdr.ptr_z_sz, ptr_bytes);
    if (raw_ptr.size() != ptr_bytes) {
        throw PZReadError("ptr block wrong size after decompression");
    }
    cc.resize(hdr.n);
    for (uint32_t j = 0; j < hdr.n; ++j) {
        uint32_t v = 0;
        for (int b = 0; b < hdr.ptr_width; ++b) {
            v |= static_cast<uint32_t>(raw_ptr[static_cast<size_t>(j) * hdr.ptr_width + b]) << (8 * b);
        }
        if (v > hdr.m) throw PZReadError("column count exceeds the number of rows");
        cc[j] = v;
    }
    body_offset += hdr.ptr_z_sz;
}

// ============================================================================
// Locate and decode the metadata block (if present). Appends results to
// the supplied vectors / map. Offsets were range-checked in validate_buffer.
// ============================================================================
static inline void decode_metadata_block(
    const uint8_t* file,
    const PZHeader& hdr,
    std::vector<std::string>& rownames,
    std::vector<std::string>& colnames,
    std::map<std::string, std::string>& user_kv)
{
    if (!(hdr.flags & FLAG_HAS_METADATA)) return;
    ZstdDCtx dctx;
    // The TLV expands names; allow generous but bounded growth.
    const size_t max_out = std::min<size_t>(
        static_cast<size_t>(hdr.metadata_z_sz) * 4096 + (1u << 20), size_t(1) << 31);
    auto raw = zstd_decompress(dctx.ctx, file + hdr.metadata_offset, hdr.metadata_z_sz, max_out);
    parse_metadata_tlv(raw.data(), raw.size(), rownames, colnames, user_kv, hdr.m, hdr.n);
    if (!rownames.empty() && rownames.size() != hdr.m) {
        throw PZReadError("rowname count does not match the number of rows");
    }
    if (!colnames.empty() && colnames.size() != hdr.n) {
        throw PZReadError("colname count does not match the number of columns");
    }
}

// ============================================================================
// Byte-unsplit decode (scalar fallback; performance can be tuned later).
//
// byte_unsplit_16: reconstruct n uint32 values from their low bytes (plane 0)
//   and high bytes (plane 1) stored contiguously:
//   src[0..n)  = low bytes (bits 0-7),  src[n..2n) = high bytes (bits 8-15)
//
// byte_unsplit_32: reconstruct n uint32 values from four planes:
//   src[0..n)=p0  src[n..2n)=p1  src[2n..3n)=p2  src[3n..4n)=p3
// ============================================================================
static inline void byte_unsplit_16(const uint8_t* src, size_t n, uint32_t* dst) {
    const uint8_t* p0 = src;
    const uint8_t* p1 = src + n;
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<uint32_t>(p0[i]) | (static_cast<uint32_t>(p1[i]) << 8);
}

static inline void byte_unsplit_32(const uint8_t* src, size_t n, uint32_t* dst) {
    const uint8_t* p0 = src;
    const uint8_t* p1 = src + n;
    const uint8_t* p2 = src + 2 * n;
    const uint8_t* p3 = src + 3 * n;
    for (size_t i = 0; i < n; ++i)
        dst[i] = static_cast<uint32_t>(p0[i])
               | (static_cast<uint32_t>(p1[i]) <<  8)
               | (static_cast<uint32_t>(p2[i]) << 16)
               | (static_cast<uint32_t>(p3[i]) << 24);
}

// ============================================================================
// Decode all VOCSC chunks, filling r.indices / r.data.
// ``body`` is the whole file image, ``body_end`` the offset of the footer,
// and ``chunk_table_offset`` the absolute offset of the chunk table.
// ``perm`` and ``cc`` are the already-decoded permutation and column-count
// arrays; ``r.indptr`` is already filled from ``cc``.
//
// Chunk blob layout (the only one write_1pz emits, FEAT_BITPLANE_BITMAP):
//   [ng:4][msz:4][zsz:4][crc:4][packed_sz:4][compressed:zsz]  — 20-byte header
//   The zstd payload decompresses to packed_sz bytes (bitmap-packed
//   pre-filter), which are then unpacked and un-bit-planed to produce
//   raw = meta(msz) + gaps(ng*gw). CRC32 is over the packed bytes.
// ============================================================================
static inline void decode_vocsc_section(
    const uint8_t* body, size_t body_end, size_t chunk_table_offset,
    const PZHeader& hdr,
    const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& cc,
    ReadResult& r)
{
    const size_t nc = hdr.num_chunks;
    constexpr uint32_t kBlobHdr = 20;

    // Chunk table: nc uint32 blob sizes, little-endian, unaligned-safe.
    // Its extent was checked against body_end in validate_buffer.
    std::vector<uint32_t> ctable(nc);
    if (nc) std::memcpy(ctable.data(), body + chunk_table_offset, nc * 4);

    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;
    const uint32_t chunk_cols = hdr.chunk_cols;
    const uint32_t n = hdr.n;
    const uint64_t vmax = (hdr.vt_code == 1) ? 0xFFull : (hdr.vt_code == 2) ? 0xFFFFull : 0xFFFFFFFFull;

    r.indices.assign(static_cast<size_t>(hdr.nnz), 0);
    r.data.assign(static_cast<size_t>(hdr.nnz), 0);

    // blob_cursor = absolute offset (within ``body``) of the next chunk blob
    uint64_t blob_cursor = static_cast<uint64_t>(chunk_table_offset) + nc * 4;

    // Scratch buffers reused across chunks
    ZstdDCtx dctx;
    std::vector<uint8_t> raw;          // final decoded: meta + byte-split gaps
    std::vector<uint8_t> packed;       // bitmap-packed chunk body
    std::vector<uint8_t> prefilter;    // bitmap-unpacked body
    std::vector<uint32_t> gaps;
    uint64_t total_ng = 0;

    for (size_t c = 0; c < nc; ++c) {
        const uint32_t blob_sz = ctable[c];
        if (blob_sz < kBlobHdr) {
            throw PZReadError("chunk blob smaller than expected header (20 bytes)");
        }
        if (blob_cursor > body_end || blob_sz > body_end - blob_cursor) {
            throw PZReadError("chunk blob extends past the end of the file");
        }
        const uint8_t* blob = body + blob_cursor;

        uint32_t ng32, msz32, zsz, chunk_crc, psz32;
        std::memcpy(&ng32,      blob +  0, 4);
        std::memcpy(&msz32,     blob +  4, 4);
        std::memcpy(&zsz,       blob +  8, 4);
        std::memcpy(&chunk_crc, blob + 12, 4);
        std::memcpy(&psz32,     blob + 16, 4);

        if (zsz != blob_sz - kBlobHdr) {
            throw PZReadError("chunk blob header size mismatch");
        }

        // The gap counts must add up to exactly nnz across all chunks.
        total_ng += ng32;
        if (total_ng > hdr.nnz) throw PZReadError("chunk gap counts exceed nnz");

        // Columns covered by this chunk, and the largest metadata stream they
        // could legitimately need (varint group count per column, plus a
        // value varint and a count varint per group, at most one group per gap).
        const size_t col_start = c * static_cast<size_t>(chunk_cols);
        const size_t col_end_raw = col_start + static_cast<size_t>(chunk_cols);
        const size_t col_end = col_end_raw < n ? col_end_raw : static_cast<size_t>(n);
        const uint64_t msz_max = 5ull * (col_end - col_start) + 10ull * ng32;
        if (msz32 > msz_max) throw PZReadError("chunk metadata size out of range");

        const size_t raw_sz = static_cast<size_t>(msz32) + static_cast<size_t>(ng32) * gw;

        // pf_sz = msz + bp_sz + (gw-1)*ng where bp_sz = 8 * ceil(ng/8) (8 bit-planes).
        const size_t bp_sz = ng32 > 0 ? 8 * ((static_cast<size_t>(ng32) + 7) / 8) : 0;
        const size_t pf_sz = static_cast<size_t>(msz32) + bp_sz
                             + (ng32 > 0 ? static_cast<size_t>(gw - 1) * ng32 : 0);
        const size_t packed_max = (pf_sz + 7) / 8 + pf_sz;
        if (psz32 > packed_max) throw PZReadError("chunk packed size out of range");

        // Decompress → packed buffer (bitmap-filtered pre-filter bytes).
        packed = zstd_decompress(dctx.ctx, blob + kBlobHdr, zsz, psz32);
        if (packed.size() != psz32) {
            throw PZReadError("chunk decompressed size mismatch");
        }

        // CRC32 is over the packed (post-bitmap-pack, pre-compress) bytes.
        if (CRC32::compute(packed.data(), packed.size()) != chunk_crc) {
            throw PZReadError("chunk CRC32 mismatch");
        }

        // Bitmap-unpack packed → prefilter of size pf_sz.
        prefilter.assign(pf_sz, 0);
        bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);

        // Reconstruct raw from prefilter:
        //   raw[0..msz)          = prefilter[0..msz)      (metadata, unchanged)
        //   raw[msz..msz+ng)     = bit_planes_decode(prefilter[msz..msz+bp_sz))
        //   raw[msz+ng..raw_sz)  = prefilter[msz+bp_sz..) (upper byte planes)
        raw.resize(raw_sz);
        std::memcpy(raw.data(), prefilter.data(), msz32);
        if (ng32 > 0) {
            bit_planes_decode(prefilter.data() + msz32, ng32, raw.data() + msz32);
            if (gw > 1) {
                std::memcpy(raw.data() + msz32 + ng32,
                            prefilter.data() + msz32 + bp_sz,
                            static_cast<size_t>(gw - 1) * ng32);
            }
        }

        // Reconstruct gap stream from byte-split planes in raw[msz32..].
        gaps.assign(ng32, 0);
        if (ng32 > 0) {
            if (gw == 2)
                byte_unsplit_16(raw.data() + msz32, ng32, gaps.data());
            else
                byte_unsplit_32(raw.data() + msz32, ng32, gaps.data());
        }

        // Walk the column TLV stream and consume gaps in order
        size_t mp = 0;  // position within raw[0..msz32] (varint stream)
        size_t gp = 0;  // position within gaps[]
        const uint8_t* meta = raw.data();  // raw always holds: meta(msz) + gaps(ng*gw)

        // Per-column buffer of (row, value) pairs, re-sorted to original row order
        std::vector<std::pair<uint32_t, uint32_t>> rv;
        for (size_t j = col_start; j < col_end; ++j) {
            uint32_t ng_col = varint_read(meta, msz32, mp);
            if (ng_col == 0) {
                if (cc[j] != 0) throw PZReadError("column has entries but an empty value list");
                continue;
            }

            rv.clear();
            rv.reserve(cc[j]);

            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv  = varint_read(meta, msz32, mp);
                uint32_t cnt = varint_read(meta, msz32, mp);
                if (cv > vmax) throw PZReadError("value exceeds the declared value type");
                if (cnt > cc[j] - rv.size()) {
                    throw PZReadError("value group longer than the column");
                }

                uint64_t prev_row = 0;  // gap stream resets to 0 for each value group
                for (uint32_t k = 0; k < cnt; ++k) {
                    if (gp >= ng32) {
                        throw PZReadError("gap stream exhausted mid-column");
                    }
                    uint64_t prow = prev_row + gaps[gp++];
                    prev_row = prow + 1;
                    if (prow >= hdr.m) {
                        throw PZReadError("decoded row index out of bounds");
                    }
                    rv.emplace_back(perm[prow], cv);
                }
            }

            // The encoder wrote rows in value-group order (sorted within a
            // group, but not across groups or after un-permutation). Sort by
            // original row so the output is canonical CSC.
            std::sort(rv.begin(), rv.end(),
                      [](const auto& a, const auto& b){ return a.first < b.first; });

            if (rv.size() != cc[j]) {
                throw PZReadError("column length mismatch vs cc[]");
            }
            for (size_t k = 1; k < rv.size(); ++k) {
                if (rv[k].first == rv[k - 1].first) {
                    throw PZReadError("duplicate row index within a column");
                }
            }

            // Emit to the pre-sized indices/data arrays using indptr[j] as start.
            // indptr was built from cc[] with running <= nnz, so this stays in range.
            const uint32_t base = r.indptr[j];
            for (size_t k = 0; k < rv.size(); ++k) {
                r.indices[base + k] = rv[k].first;
                r.data   [base + k] = rv[k].second;
            }
        }

        if (gp != ng32) {
            throw PZReadError("gap stream had leftover values after column walk");
        }
        if (mp != msz32) {
            throw PZReadError("chunk metadata stream had leftover bytes");
        }

        blob_cursor += blob_sz;
    }

    if (total_ng != hdr.nnz) {
        throw PZReadError("chunk gap counts do not add up to nnz");
    }
}

// ============================================================================
// read_1pz_buffer — decode a complete .1pz image held in memory.
// ============================================================================
inline ReadResult read_1pz_buffer(
    const uint8_t* file, size_t file_size,
    const ReadLimits& limits = ReadLimits())
{
    try {
        PZHeader hdr = validate_buffer(file, file_size, limits);

        ReadResult r;
        r.m = hdr.m;
        r.n = hdr.n;
        r.nnz = hdr.nnz;
        r.vt_code = hdr.vt_code;

        size_t body_offset = sizeof(PZHeader);
        std::vector<uint32_t> perm, cc;
        decode_perm_and_ptr(file, body_offset, hdr, perm, cc);

        decode_metadata_block(file, hdr, r.rownames, r.colnames, r.user_kv);

        // Populate indptr from the decoded column counts.
        r.indptr.assign(static_cast<size_t>(hdr.n) + 1, 0);
        uint64_t running = 0;
        for (uint32_t j = 0; j < hdr.n; ++j) {
            r.indptr[j] = static_cast<uint32_t>(running);
            running += cc[j];
            if (running > hdr.nnz) throw PZReadError("column counts exceed nnz");
        }
        r.indptr[hdr.n] = static_cast<uint32_t>(running);
        if (running != hdr.nnz) {
            throw PZReadError(
                "nnz mismatch: indptr says " + std::to_string(running) +
                ", header says " + std::to_string(hdr.nnz));
        }

        decode_vocsc_section(file, file_size - sizeof(PZFooter), body_offset, hdr, perm, cc, r);
        return r;
    } catch (const std::bad_alloc&) {
        throw PZReadError("out of memory while decoding (file declares sizes beyond available memory)");
    } catch (const std::length_error&) {
        throw PZReadError("declared sizes exceed implementation limits");
    }
}

// ============================================================================
// read_1pz — read a file from disk and decode it.
// ============================================================================
inline ReadResult read_1pz(const std::string& path, const ReadLimits& limits = ReadLimits()) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw PZReadError("cannot open " + path);
    std::streamoff sz = fin.tellg();
    if (sz < 0) throw PZReadError(path + ": cannot determine file size");
    if (static_cast<uint64_t>(sz) > limits.max_file_bytes) {
        throw PZReadError(path + ": file larger than the configured limit");
    }
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    if (sz > 0) fin.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!fin) throw PZReadError(path + ": read failed");
    fin.close();

    try {
        return read_1pz_buffer(buf.data(), buf.size(), limits);
    } catch (const PZReadError& e) {
        throw PZReadError(path + ": " + e.what());
    }
}

}  // namespace pz
}  // namespace singlet
