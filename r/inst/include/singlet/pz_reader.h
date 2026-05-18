// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: pz_reader.h
// Native .1pz (VOCSC) reader — bit-exact inverse of pz_writer.h.
//
// This header re-uses the structs and helpers from pz_writer.h
// (PZHeader, PZFooter, CRC32, TP1_MAGIC, TP1_VERSION, FLAG_* constants),
// so it must be compiled with pz_writer.h also in scope.
//
// Single function: read_1pz(path) -> ReadResult
// The returned result contains a CSC matrix, rownames/colnames, and an
// optional user key-value map — everything that write_1pz() embedded.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
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
    uint8_t  vt_code = 0;               // value type code (1=uint8, 2=uint16, 3=uint32, etc.)

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
    ZstdDCtx() : ctx(ZSTD_createDCtx()) {}
    ~ZstdDCtx() { if (ctx) ZSTD_freeDCtx(ctx); }
    ZstdDCtx(const ZstdDCtx&) = delete;
    ZstdDCtx& operator=(const ZstdDCtx&) = delete;
};

static inline std::vector<uint8_t>
zstd_decompress(ZSTD_DCtx* ctx, const uint8_t* data, size_t sz) {
    unsigned long long dsz = ZSTD_getFrameContentSize(data, sz);
    if (dsz == ZSTD_CONTENTSIZE_ERROR || dsz == ZSTD_CONTENTSIZE_UNKNOWN) {
        throw PZReadError("zstd: frame content size unknown or error");
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
// ``buf`` starting at ``offset`` and advances ``offset`` past the terminator.
static inline uint32_t varint_read(const uint8_t* buf, size_t& offset) {
    uint32_t v = 0;
    int shift = 0;
    while (true) {
        uint8_t b = buf[offset++];
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
// ============================================================================
static inline void parse_metadata_tlv(
    const uint8_t* buf, size_t len,
    std::vector<std::string>& rownames,
    std::vector<std::string>& colnames,
    std::map<std::string, std::string>& user_kv)
{
    size_t offset = 0;
    while (offset < len) {
        uint8_t tag = buf[offset++];
        if (tag == META_TAG_END) break;
        if (offset + 4 > len) {
            throw PZReadError("metadata TLV: truncated length field");
        }
        uint32_t length;
        std::memcpy(&length, buf + offset, 4);
        offset += 4;
        if (offset + length > len) {
            throw PZReadError("metadata TLV: payload extends past end of block");
        }

        if (tag == META_TAG_ROWNAMES || tag == META_TAG_COLNAMES) {
            auto& dst = (tag == META_TAG_ROWNAMES) ? rownames : colnames;
            size_t p = offset;
            size_t end = offset + length;
            while (p < end) {
                size_t start = p;
                while (p < end && buf[p] != 0) ++p;
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
        // Unknown tags: silently skip their payload for forward compatibility.
        offset += length;
    }
}

// ============================================================================
// Read entire file into memory, validate header + footer + CRC.
// Returns the in-memory buffer; the caller owns it.
// ============================================================================
static inline std::vector<uint8_t> slurp_and_validate(const std::string& path) {
    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) throw PZReadError("cannot open " + path);
    std::streamsize sz = fin.tellg();
    if (sz < static_cast<std::streamsize>(sizeof(PZHeader) + sizeof(PZFooter))) {
        throw PZReadError(path + ": file too short (" + std::to_string(sz) + " bytes)");
    }
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    fin.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!fin) throw PZReadError(path + ": read failed");
    fin.close();

    // Header
    PZHeader hdr;
    std::memcpy(&hdr, buf.data(), sizeof(PZHeader));
    if (hdr.magic != TP1_MAGIC) {
        throw PZReadError(path + ": bad magic");
    }
    if (hdr.version != TP1_VERSION) {
        throw PZReadError(path + ": unsupported version " + std::to_string(hdr.version));
    }

    // Footer
    PZFooter ftr;
    std::memcpy(&ftr, buf.data() + buf.size() - sizeof(PZFooter), sizeof(PZFooter));
    if (ftr.magic != TP1_MAGIC) {
        throw PZReadError(path + ": bad footer magic");
    }
    if (ftr.num_chunks != hdr.num_chunks) {
        throw PZReadError(path + ": chunk count mismatch header/footer");
    }

    // CRC32 over file body (everything before the footer)
    uint32_t actual = CRC32::compute(buf.data(), buf.size() - sizeof(PZFooter));
    if (actual != ftr.file_crc32) {
        throw PZReadError(path + ": file CRC32 mismatch");
    }

    return buf;
}

// ============================================================================
// Decode the permutation block (if FLAG_HAS_PERM set) and the column-count
// ptr block into two vectors. Populates ``perm`` (length m) and ``cc``
// (length n). Advances ``body_offset`` past the end of both blocks.
// ============================================================================
static inline void decode_perm_and_ptr(
    const uint8_t* body, size_t& body_offset,
    const PZHeader& hdr,
    std::vector<uint32_t>& perm,
    std::vector<uint32_t>& cc)
{
    ZstdDCtx dctx;

    // Permutation block
    perm.resize(hdr.m);
    if (hdr.flags & FLAG_HAS_PERM) {
        if (hdr.perm_z_sz == 0) {
            throw PZReadError("FLAG_HAS_PERM set but perm_z_sz == 0");
        }
        auto raw = zstd_decompress(dctx.ctx, body + body_offset, hdr.perm_z_sz);
        if (raw.size() != static_cast<size_t>(hdr.m) * 4) {
            throw PZReadError("perm block wrong size after decompression");
        }
        std::memcpy(perm.data(), raw.data(), raw.size());
        body_offset += hdr.perm_z_sz;
    } else {
        // Identity permutation
        for (uint32_t i = 0; i < hdr.m; ++i) perm[i] = i;
    }

    // Ptr block — compressed column-count array (ptr_width bytes per column)
    cc.resize(hdr.n);
    if (hdr.ptr_z_sz == 0) {
        throw PZReadError("ptr_z_sz == 0 (all-empty matrix not yet supported)");
    }
    auto raw_ptr = zstd_decompress(dctx.ctx, body + body_offset, hdr.ptr_z_sz);
    if (raw_ptr.size() != static_cast<size_t>(hdr.n) * hdr.ptr_width) {
        throw PZReadError("ptr block wrong size after decompression");
    }
    for (uint32_t j = 0; j < hdr.n; ++j) {
        uint32_t v = 0;
        for (int b = 0; b < hdr.ptr_width; ++b) {
            v |= static_cast<uint32_t>(raw_ptr[j * hdr.ptr_width + b]) << (8 * b);
        }
        cc[j] = v;
    }
    body_offset += hdr.ptr_z_sz;
}

// ============================================================================
// Locate and decode the metadata block (if present). Appends results to
// the supplied vectors / map.
// ============================================================================
static inline void decode_metadata_block(
    const uint8_t* file, size_t file_size,
    const PZHeader& hdr,
    std::vector<std::string>& rownames,
    std::vector<std::string>& colnames,
    std::map<std::string, std::string>& user_kv)
{
    if (!(hdr.flags & FLAG_HAS_METADATA) || hdr.metadata_z_sz == 0) return;
    if (hdr.metadata_offset == 0 ||
        hdr.metadata_offset + hdr.metadata_z_sz > file_size) {
        throw PZReadError("metadata block offset/size out of range");
    }
    ZstdDCtx dctx;
    auto raw = zstd_decompress(
        dctx.ctx, file + hdr.metadata_offset, hdr.metadata_z_sz);
    parse_metadata_tlv(raw.data(), raw.size(), rownames, colnames, user_kv);
}

// ============================================================================
// Decode all VOCSC chunks, filling r.indices / r.data.
// ``body`` points at the start of the chunk table. ``perm`` and ``cc`` are
// the already-decoded permutation and column-count arrays.
// ============================================================================
static inline void decode_vocsc_section(
    const uint8_t* body, size_t chunk_table_offset,
    const PZHeader& hdr,
    const std::vector<uint32_t>& perm,
    const std::vector<uint32_t>& cc,
    ReadResult& r)
{
    const size_t nc = hdr.num_chunks;

    // Chunk table: nc uint32 blob sizes, little-endian, unaligned-safe
    std::vector<uint32_t> ctable(nc);
    std::memcpy(ctable.data(), body + chunk_table_offset, nc * 4);

    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;
    const uint32_t chunk_cols = hdr.chunk_cols;
    const uint32_t n = hdr.n;

    r.indices.assign(hdr.nnz, 0);
    r.data.assign(hdr.nnz, 0);

    // blob_cursor = absolute offset (within ``body``) of the next chunk blob
    size_t blob_cursor = chunk_table_offset + nc * 4;

    // Scratch buffers reused across chunks
    ZstdDCtx dctx;
    std::vector<uint8_t> packed;       // bitmap-packed chunk body
    std::vector<uint8_t> prefilter;    // unpacked chunk body
    std::vector<uint8_t> low_bytes;    // decoded bit-plane 0
    std::vector<uint32_t> gaps;

    for (size_t c = 0; c < nc; ++c) {
        const uint32_t blob_sz = ctable[c];
        if (blob_sz < 20) {
            throw PZReadError("chunk blob smaller than 20-byte header");
        }
        const uint8_t* blob = body + blob_cursor;

        uint32_t ng32, msz32, zsz, chunk_crc, psz32;
        std::memcpy(&ng32,      blob +  0, 4);
        std::memcpy(&msz32,     blob +  4, 4);
        std::memcpy(&zsz,       blob +  8, 4);
        std::memcpy(&chunk_crc, blob + 12, 4);
        std::memcpy(&psz32,     blob + 16, 4);

        if (static_cast<size_t>(20) + zsz > blob_sz) {
            throw PZReadError("chunk blob header size mismatch");
        }

        // zstd-decompress blob body → packed
        packed = zstd_decompress(dctx.ctx, blob + 20, zsz);
        if (packed.size() != psz32) {
            throw PZReadError("chunk decompressed size mismatch");
        }

        // CRC32 over the packed (post-bitmap-pack) bytes
        if (CRC32::compute(packed.data(), packed.size()) != chunk_crc) {
            throw PZReadError("chunk CRC32 mismatch");
        }

        // Prefilter size: msz + bp_sz + (gw-1)*ng  (matches encoder)
        const size_t bp_sz = ng32 > 0 ? 8 * ((ng32 + 7) / 8) : 0;
        const size_t extra_planes_sz = ng32 > 0 ? static_cast<size_t>(gw - 1) * ng32 : 0;
        const size_t pf_sz = static_cast<size_t>(msz32) + bp_sz + extra_planes_sz;
        prefilter.assign(pf_sz, 0);
        bitmap_unpack(packed.data(), packed.size(), prefilter.data(), pf_sz);

        // Rebuild the gap stream
        gaps.assign(ng32, 0);
        if (ng32 > 0) {
            low_bytes.assign(ng32, 0);
            bit_planes_decode(prefilter.data() + msz32, ng32, low_bytes.data());
            for (uint32_t i = 0; i < ng32; ++i) gaps[i] = low_bytes[i];
            if (gw > 1) {
                const uint8_t* upper = prefilter.data() + msz32 + bp_sz;
                for (int b = 1; b < gw; ++b) {
                    const uint8_t* plane = upper + static_cast<size_t>(b - 1) * ng32;
                    for (uint32_t i = 0; i < ng32; ++i) {
                        gaps[i] |= static_cast<uint32_t>(plane[i]) << (8 * b);
                    }
                }
            }
        }

        // Walk the column TLV stream and consume gaps in order
        const size_t col_start = c * static_cast<size_t>(chunk_cols);
        const size_t col_end_raw = col_start + static_cast<size_t>(chunk_cols);
        const size_t col_end = col_end_raw < n ? col_end_raw : static_cast<size_t>(n);

        size_t mp = 0;  // position within prefilter[0..msz32] (varint stream)
        size_t gp = 0;  // position within gaps[]
        const uint8_t* meta = prefilter.data();

        // Per-column buffer of (row, value) pairs, re-sorted to original row order
        std::vector<std::pair<uint32_t, uint32_t>> rv;
        for (size_t j = col_start; j < col_end; ++j) {
            uint32_t ng_col = varint_read(meta, mp);
            if (mp > msz32) throw PZReadError("column TLV overrun");
            if (ng_col == 0) continue;

            rv.clear();
            rv.reserve(cc[j]);

            for (uint32_t vi = 0; vi < ng_col; ++vi) {
                uint32_t cv  = varint_read(meta, mp);
                uint32_t cnt = varint_read(meta, mp);
                if (mp > msz32) throw PZReadError("column TLV overrun");

                uint32_t prev_row = 0;
                for (uint32_t k = 0; k < cnt; ++k) {
                    if (gp >= ng32) {
                        throw PZReadError("gap stream exhausted mid-column");
                    }
                    uint32_t gap = gaps[gp++];
                    uint32_t prow = prev_row + gap;
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

            // Emit to the pre-sized indices/data arrays using indptr[j] as start
            const uint32_t base = r.indptr[j];
            if (rv.size() != cc[j]) {
                throw PZReadError("column length mismatch vs cc[]");
            }
            for (size_t k = 0; k < rv.size(); ++k) {
                r.indices[base + k] = rv[k].first;
                r.data   [base + k] = rv[k].second;
            }
        }

        if (gp != ng32) {
            throw PZReadError("gap stream had leftover values after column walk");
        }

        blob_cursor += blob_sz;
    }
}

// ============================================================================
// read_1pz — CPP-1 partial implementation
// ----------------------------------------------------------------------------
// At CPP-1, we parse the header+footer, validate CRC, decode the permutation
// and ptr blocks, and decode the metadata TLV. The VOCSC matrix decode itself
// is implemented in CPP-2. Calling read_1pz() in this file right now therefore
// leaves ``result.indptr`` / ``result.indices`` / ``result.data`` EMPTY (with
// only the ``indptr`` size set to ``n+1``). CPP-2 will fill those arrays.
// ============================================================================
inline ReadResult read_1pz(const std::string& path) {
    auto buf = slurp_and_validate(path);
    const uint8_t* file = buf.data();
    const size_t file_size = buf.size();

    PZHeader hdr;
    std::memcpy(&hdr, file, sizeof(PZHeader));

    ReadResult r;
    r.m = hdr.m;
    r.n = hdr.n;
    r.nnz = hdr.nnz;
    r.vt_code = hdr.vt_code;
    r.indptr.assign(hdr.n + 1, 0);

    // Decode the simple blocks (perm, cc, metadata) in-place.
    size_t body_offset = sizeof(PZHeader);
    std::vector<uint32_t> perm, cc;
    decode_perm_and_ptr(file, body_offset, hdr, perm, cc);

    decode_metadata_block(
        file, file_size, hdr, r.rownames, r.colnames, r.user_kv);

    // Populate indptr from the (already decoded) column counts so that
    // even the CPP-1 stub exposes usable column boundaries.
    uint32_t running = 0;
    for (uint32_t j = 0; j < hdr.n; ++j) {
        r.indptr[j] = running;
        running += cc[j];
    }
    r.indptr[hdr.n] = running;
    if (static_cast<uint64_t>(running) != hdr.nnz) {
        throw PZReadError(
            "nnz mismatch: indptr says " + std::to_string(running) +
            ", header says " + std::to_string(hdr.nnz));
    }

    // CPP-2: walk the chunk table and decode VOCSC chunks into r.indices/data.
    decode_vocsc_section(file, body_offset, hdr, perm, cc, r);

    return r;
}

}  // namespace pz
}  // namespace singlet
