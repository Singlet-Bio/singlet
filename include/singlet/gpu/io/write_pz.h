// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/io/write_pz.h
//
// Minimal host-only .1pz writer for test utilities.
//
// Produces TP1Z v1 files bit-exact with singlify/include/singlet-pileup/pz_writer.h.
// This is NOT a production kernel — it is a test helper used only in
// streaming_pipeline_correctness.cpp and other test fixtures that need to
// create synthetic .1pz files on disk for round-trip verification.
//
// API (within singlet_gpu::io):
//
//   write_pz(path, n_rows, n_cols, nnz,
//            values, indptr, indices,
//            rownames, colnames)
//
// Parameters:
//   path      — output file path
//   n_rows    — number of features (rows / genes)
//   n_cols    — number of cells (columns)
//   nnz       — number of non-zero entries
//   values    — uint16_t data array [nnz]; caller ensures values fit in uint16
//   indptr    — int32_t column pointer array [n_cols + 1]
//   indices   — int32_t row index array [nnz]
//   rownames  — feature names (may be empty)
//   colnames  — cell / barcode names (may be empty)
//
// Dependencies: zstd (already required by pz_device_loader.h).
// No CUDA headers included — purely CPU.

#pragma once

#include <singlet-gpu/io/pz_device_loader.h>  // pz_fmt constants + structs

#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace singlet_gpu {
namespace io {

// ---------------------------------------------------------------------------
// Internal — all helpers in an anonymous namespace to avoid ODR conflicts.
// ---------------------------------------------------------------------------
namespace {  // write_pz internals

// CRC32 writer-side (ISO 3309, slicing-by-8).
// Identical to Crc32Loader in pz_device_loader.h — just renamed.
struct Crc32Writer {
    uint32_t state_;
    Crc32Writer() : state_(0xFFFFFFFFu) {}
    void update(const uint8_t* data, size_t len) {
        static const auto& T = table8();
        uint32_t crc = state_;
        while (len >= 8) {
            uint32_t a, b;
            std::memcpy(&a, data, 4); std::memcpy(&b, data + 4, 4);
            a ^= crc;
            crc = T[7][a&0xFF]^T[6][(a>>8)&0xFF]^T[5][(a>>16)&0xFF]^T[4][(a>>24)&0xFF]^
                  T[3][b&0xFF]^T[2][(b>>8)&0xFF]^T[1][(b>>16)&0xFF]^T[0][(b>>24)&0xFF];
            data += 8; len -= 8;
        }
        while (len--) crc = (crc >> 8) ^ T[0][(crc ^ *data++) & 0xFF];
        state_ = crc;
    }
    uint32_t finalize() const { return state_ ^ 0xFFFFFFFFu; }
    static uint32_t compute(const void* d, size_t n) {
        Crc32Writer c; c.update(static_cast<const uint8_t*>(d), n); return c.finalize();
    }
    static const uint32_t (&table8())[8][256] {
        static uint32_t T[8][256]{};
        static bool init = false;
        if (!init) {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                T[0][i] = c;
            }
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = T[0][i];
                for (int k = 1; k < 8; ++k) { c = (c >> 8) ^ T[0][c & 0xFF]; T[k][i] = c; }
            }
            init = true;
        }
        return T;
    }
};

struct ZstdCCtxW {
    ZSTD_CCtx* ctx;
    ZstdCCtxW() : ctx(ZSTD_createCCtx()) { if (!ctx) throw std::runtime_error("ZSTD_createCCtx"); }
    ~ZstdCCtxW() { if (ctx) ZSTD_freeCCtx(ctx); }
    ZstdCCtxW(const ZstdCCtxW&) = delete;
};

inline std::vector<uint8_t> zstd_comp_w(ZSTD_CCtx* ctx, const uint8_t* data, size_t sz, int lvl=3) {
    ZSTD_CCtx_reset(ctx, ZSTD_reset_session_only);
    ZSTD_CCtx_setParameter(ctx, ZSTD_c_compressionLevel, lvl);
    ZSTD_CCtx_setParameter(ctx, ZSTD_c_checksumFlag, 1);
    size_t bound = ZSTD_compressBound(sz);
    std::vector<uint8_t> out(bound);
    size_t csz = ZSTD_compress2(ctx, out.data(), bound, data, sz);
    if (ZSTD_isError(csz)) throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(csz));
    out.resize(csz); return out;
}

inline void varint_w(std::vector<uint8_t>& out, uint32_t v) {
    while (v >= 0x80) { out.push_back((v & 0x7F) | 0x80); v >>= 7; }
    out.push_back(v);
}

// bit-plane encode plane-0 transform (matches pz_writer.h bit_planes_encode)
inline void bp_encode_w(const uint8_t* src, size_t n, uint8_t* dst) {
    size_t pb = (n + 7) / 8; std::memset(dst, 0, 8 * pb);
    for (size_t i = 0; i < n; ++i) {
        uint8_t v = src[i]; if (!v) continue;
        size_t bi = i / 8; uint8_t bm = uint8_t(1u << (i & 7));
        for (int b = 0; b < 8; ++b) if (v & (1 << b)) dst[b * pb + bi] |= bm;
    }
}

// bitmap zero-filter pack (matches pz_writer.h bitmap_pack)
inline size_t bm_pack_w(const uint8_t* src, size_t n, uint8_t* dst) {
    size_t bm_bytes = (n + 7) / 8; std::memset(dst, 0, bm_bytes); size_t wp = bm_bytes;
    for (size_t i = 0; i < n; ++i) if (src[i]) { dst[i/8] |= (1u << (i & 7)); dst[wp++] = src[i]; }
    return wp;
}

// byte-split 2 planes (gap16, matches pz_writer.h byte_split_16)
inline void bsplit16_w(const uint32_t* src, size_t n, uint8_t* dst) {
    uint8_t *p0 = dst, *p1 = dst + n;
    for (size_t i = 0; i < n; ++i) { p0[i] = uint8_t(src[i]); p1[i] = uint8_t(src[i] >> 8); }
}

// TLV helpers (matches pz_writer.h push_strings_tlv / push_kv_tlv)
inline void push_strvec(std::vector<uint8_t>& out, uint8_t tag, const std::vector<std::string>& v) {
    if (v.empty()) return;
    size_t dsz = 0; for (auto& s : v) dsz += s.size() + 1;
    out.push_back(tag); uint32_t sz = uint32_t(dsz);
    out.insert(out.end(), (uint8_t*)&sz, (uint8_t*)&sz + 4);
    for (auto& s : v) { out.insert(out.end(), s.begin(), s.end()); out.push_back(0); }
}

}  // anonymous namespace (write_pz internals)

// ---------------------------------------------------------------------------
// write_pz — main entry point.
//
// Takes a CSC matrix (uint16 values, int32 indptr / indices) and writes a
// complete TP1Z v1 .1pz file that pz_device_loader.h can round-trip.
//
// Constraints / simplifications (test helper, not production):
//  - vt_code always 2 (uint16); caller must ensure values fit.
//  - chunk_cols fixed at 1024 (matches pz_writer.h default for small matrices).
//  - No colsums block written (colsums_z_sz=0 in header, FLAG_HAS_COLSUMS clear).
//  - Single-threaded (nt=1); fine for small test fixtures.
//  - No transpose block.
//  - Row frequency-sort permutation written (FLAG_HAS_PERM set).
//  - FLAG_GAP16 set when all gaps fit in 16 bits (common for small matrices).
// ---------------------------------------------------------------------------
inline void write_pz(
    const std::string& path,
    int n_rows,
    int n_cols,
    int64_t nnz_in,
    const uint16_t* values,    // [nnz]
    const int32_t*  indptr,    // [n_cols + 1]
    const int32_t*  indices,   // [nnz]
    const std::vector<std::string>& rownames = {},
    const std::vector<std::string>& colnames = {})
{
    using namespace pz_fmt;

    const uint32_t m = static_cast<uint32_t>(n_rows);
    const uint32_t n = static_cast<uint32_t>(n_cols);
    const uint64_t nnz = static_cast<uint64_t>(nnz_in);
    const int chunk_cols = 1024;

    // --- Row-frequency sort permutation (descending row nnz) ------------------
    std::vector<int32_t> rnnz(m, 0);
    for (uint64_t k = 0; k < nnz; ++k) rnnz[indices[k]]++;
    std::vector<uint32_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0u);
    std::stable_sort(perm.begin(), perm.end(),
        [&](uint32_t a, uint32_t b){ return rnnz[a] > rnnz[b]; });
    std::vector<uint32_t> inv_perm(m);
    for (uint32_t i = 0; i < m; ++i) inv_perm[perm[i]] = i;

    // --- Column counts + ptr_width --------------------------------------------
    std::vector<uint32_t> cc(n);
    for (uint32_t j = 0; j < n; ++j)
        cc[j] = static_cast<uint32_t>(indptr[j + 1] - indptr[j]);
    bool use16p = true;
    for (uint32_t j = 0; j < n; ++j) if (cc[j] > 65535) { use16p = false; break; }
    uint8_t ptr_width = use16p ? 2 : 4;

    ZstdCCtxW cctx;

    // Compress perm
    std::vector<uint8_t> perm_raw(m * 4);
    std::memcpy(perm_raw.data(), perm.data(), m * 4);
    auto perm_z = zstd_comp_w(cctx.ctx, perm_raw.data(), perm_raw.size());

    // Compress ptr
    std::vector<uint8_t> ptr_raw(n * ptr_width);
    for (uint32_t j = 0; j < n; ++j)
        for (int b = 0; b < ptr_width; ++b)
            ptr_raw[j * ptr_width + b] = uint8_t(cc[j] >> (8 * b));
    auto ptr_z = zstd_comp_w(cctx.ctx, ptr_raw.data(), ptr_raw.size());

    // --- VOCSC chunk encoding -------------------------------------------------
    int nc = (int(n) + chunk_cols - 1) / chunk_cols;
    std::vector<std::vector<uint8_t>> blobs(nc);
    bool any_gap32 = false;

    // First pass: compute gaps and varint metadata per chunk.
    struct ChunkEnc { std::vector<uint32_t> gaps; std::vector<uint8_t> meta; uint32_t max_gap; };
    std::vector<ChunkEnc> chunks(nc);

    for (int c = 0; c < nc; ++c) {
        uint32_t col_start = static_cast<uint32_t>(c) * chunk_cols;
        uint32_t col_end   = std::min(col_start + uint32_t(chunk_cols), n);
        auto& ch = chunks[c];
        ch.max_gap = 0;

        // Track val→rows mapping per column.
        // Values are uint16; max possible = 65535.
        std::vector<std::vector<uint32_t>> val_rows;
        std::vector<uint32_t> active_vals;

        for (uint32_t j = col_start; j < col_end; ++j) {
            uint32_t cnt = cc[j];
            if (cnt == 0) { varint_w(ch.meta, 0); continue; }
            uint32_t s = static_cast<uint32_t>(indptr[j]);

            // Bucket entries by value.
            active_vals.clear();
            uint32_t max_v = 0;
            for (uint32_t k = 0; k < cnt; ++k) {
                uint32_t v = values[s + k];
                if (v > max_v) max_v = v;
            }
            if (val_rows.size() <= max_v) val_rows.resize(max_v + 1);

            for (uint32_t k = 0; k < cnt; ++k) {
                uint32_t v = values[s + k];
                uint32_t prow = inv_perm[static_cast<uint32_t>(indices[s + k])];
                if (val_rows[v].empty()) active_vals.push_back(v);
                val_rows[v].push_back(prow);
            }
            std::sort(active_vals.begin(), active_vals.end());

            varint_w(ch.meta, static_cast<uint32_t>(active_vals.size()));
            for (uint32_t vi : active_vals) {
                auto& rows = val_rows[vi];
                std::sort(rows.begin(), rows.end());
                varint_w(ch.meta, vi);
                varint_w(ch.meta, static_cast<uint32_t>(rows.size()));
                uint32_t prev = 0;
                for (uint32_t row : rows) {
                    uint32_t gap = row - prev;
                    if (gap > ch.max_gap) ch.max_gap = gap;
                    ch.gaps.push_back(gap);
                    prev = row + 1;
                }
                rows.clear();
            }
        }
        if (ch.max_gap >= 65536) any_gap32 = true;
    }

    bool gap16 = !any_gap32;
    int gw = gap16 ? 2 : 4;

    // Second pass: byte-split + bit-plane + bitmap + zstd → blob per chunk.
    for (int c = 0; c < nc; ++c) {
        auto& ch = chunks[c];
        size_t ng  = ch.gaps.size();
        size_t msz = ch.meta.size();

        std::vector<uint8_t> raw(msz + ng * gw);
        std::memcpy(raw.data(), ch.meta.data(), msz);
        if (ng > 0) {
            if (gap16)
                bsplit16_w(ch.gaps.data(), ng, raw.data() + msz);
            else {
                // gap32: 4 planes
                uint8_t *p0=raw.data()+msz, *p1=p0+ng, *p2=p1+ng, *p3=p2+ng;
                for (size_t i = 0; i < ng; ++i) {
                    p0[i]=uint8_t(ch.gaps[i]);     p1[i]=uint8_t(ch.gaps[i]>>8);
                    p2[i]=uint8_t(ch.gaps[i]>>16); p3[i]=uint8_t(ch.gaps[i]>>24);
                }
            }
        }

        size_t bp_sz = ng > 0 ? 8 * ((ng + 7) / 8) : 0;
        size_t pf_sz = msz + bp_sz + (ng > 0 ? ng * size_t(gw - 1) : 0);
        std::vector<uint8_t> prefilter(pf_sz, 0);
        std::memcpy(prefilter.data(), raw.data(), msz);
        if (ng > 0) {
            bp_encode_w(raw.data() + msz, ng, prefilter.data() + msz);
            if (gw > 1)
                std::memcpy(prefilter.data() + msz + bp_sz,
                            raw.data() + msz + ng, ng * size_t(gw - 1));
        }

        std::vector<uint8_t> packed(pf_sz + (pf_sz + 7) / 8);
        size_t packed_sz = bm_pack_w(prefilter.data(), pf_sz, packed.data());

        uint32_t chunk_crc = Crc32Writer::compute(packed.data(), packed_sz);
        auto z = zstd_comp_w(cctx.ctx, packed.data(), packed_sz);

        uint32_t ng32  = uint32_t(ng);
        uint32_t msz32 = uint32_t(msz);
        uint32_t zsz   = uint32_t(z.size());
        uint32_t psz32 = uint32_t(packed_sz);

        auto& blob = blobs[c];
        blob.resize(20 + z.size());
        uint8_t* wp = blob.data();
        std::memcpy(wp, &ng32,      4); wp += 4;
        std::memcpy(wp, &msz32,     4); wp += 4;
        std::memcpy(wp, &zsz,       4); wp += 4;
        std::memcpy(wp, &chunk_crc, 4); wp += 4;
        std::memcpy(wp, &psz32,     4); wp += 4;
        std::memcpy(wp, z.data(), z.size());
    }

    // --- Metadata TLV ---------------------------------------------------------
    std::vector<uint8_t> meta_raw;
    bool has_meta = !rownames.empty() || !colnames.empty();
    if (has_meta) {
        push_strvec(meta_raw, META_TAG_ROWNAMES, rownames);
        push_strvec(meta_raw, META_TAG_COLNAMES, colnames);
        meta_raw.push_back(META_TAG_END);
    }
    auto meta_z = has_meta
        ? zstd_comp_w(cctx.ctx, meta_raw.data(), meta_raw.size())
        : std::vector<uint8_t>{};

    // --- Compute byte offsets -------------------------------------------------
    size_t off = 96;                                    // header
    off += perm_z.size() + ptr_z.size();               // perm + ptr blocks
    off += size_t(nc) * 4;                             // chunk table
    for (int c = 0; c < nc; ++c) off += blobs[c].size();
    // (no colsums block)
    size_t metadata_offset = off; off += meta_z.size();
    size_t footer_offset = off;

    // --- Build header ---------------------------------------------------------
    PZHeader hdr{};
    hdr.magic          = TP1_MAGIC;
    hdr.version        = TP1_VERSION;
    hdr.vt_code        = 2;                            // uint16
    hdr.flags          = FLAG_HAS_PERM;
    if (gap16)    hdr.flags |= FLAG_GAP16;
    if (has_meta) hdr.flags |= FLAG_HAS_METADATA;
    hdr.m              = m;
    hdr.n              = n;
    hdr.nnz            = nnz;
    hdr.ptr_width      = ptr_width;
    hdr.codec_level    = 3;
    hdr.num_chunks     = uint32_t(nc);
    hdr.perm_z_sz      = uint32_t(perm_z.size());
    hdr.ptr_z_sz       = uint32_t(ptr_z.size());
    hdr.chunk_cols     = uint32_t(chunk_cols);
    hdr.feature_flags  = FEAT_ZSTD_CHECKSUMS | FEAT_BITPLANE_BITMAP;
    hdr.metadata_offset = has_meta ? uint64_t(metadata_offset) : 0;
    hdr.metadata_z_sz  = uint32_t(meta_z.size());
    hdr.colsums_z_sz   = 0;
    hdr.transpose_offset = 0;
    hdr.transpose_z_sz   = 0;
    hdr.transpose_chunks = 0;
    std::memset(hdr.reserved, 0, sizeof(hdr.reserved));

    // --- Assemble body buffer + CRC -------------------------------------------
    std::vector<uint8_t> buf(footer_offset);
    size_t wp = 0;
    auto append = [&](const void* d, size_t s) { std::memcpy(buf.data() + wp, d, s); wp += s; };
    append(&hdr, 96);
    append(perm_z.data(), perm_z.size());
    append(ptr_z.data(), ptr_z.size());
    // Chunk table
    for (int c = 0; c < nc; ++c) {
        uint32_t bsz = uint32_t(blobs[c].size());
        append(&bsz, 4);
    }
    for (int c = 0; c < nc; ++c) append(blobs[c].data(), blobs[c].size());
    if (has_meta) append(meta_z.data(), meta_z.size());

    uint32_t file_crc = Crc32Writer::compute(buf.data(), buf.size());

    PZFooter ftr{};
    ftr.file_crc32  = file_crc;
    ftr._reserved   = 0;
    ftr.num_chunks  = uint32_t(nc);
    ftr.magic       = TP1_MAGIC;

    // --- Write to disk --------------------------------------------------------
    std::ofstream fout(path, std::ios::binary);
    if (!fout) throw std::runtime_error("write_pz: cannot open: " + path);
    fout.write(reinterpret_cast<const char*>(buf.data()), std::streamsize(buf.size()));
    fout.write(reinterpret_cast<const char*>(&ftr), 16);
    if (!fout) throw std::runtime_error("write_pz: write error: " + path);
}

}  // namespace io
}  // namespace singlet_gpu
