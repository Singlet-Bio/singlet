// w4_reader_dec.h: prototype multi-threaded .1pz (TP1Z v1) chunk decoder writing straight into caller-owned buffers.
// Base: singlet/pileup/pz_reader.h (read-only). Differences from the production reader:
//   * names are never touched (no metadata block parse)
//   * file-level CRC-32 is optional (per-chunk CRC-32 stays)
//   * chunks decode in parallel (std::thread), each into its slice of the caller's indices/values arrays
//   * output is int32 indices and a caller-chosen value width (u8/u16/i32/f32/f64): no uint32 -> int32 copy loops
//   * a chunk range [c0, c1) can be decoded alone (cell-subset reads via the chunk table)
// The per-chunk kernels (bitmap unpack, bit planes, byte unsplit) are the production ones, unchanged, so the speedups
// measured with this header are threading + zero-copy + no names (+ optional packed-key sort), not new kernels.
#pragma once
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "singlet/pileup/sparse_accumulator.h"
#include "singlet/pileup/pz_reader.h"

namespace w4r {
using namespace singlet::pz;

struct Handle {
    const uint8_t* buf = nullptr;
    size_t size = 0;
    PZHeader hdr{};
    std::vector<uint32_t> perm, cc;     // perm: stored row -> original row; cc: nnz per column
    std::vector<uint32_t> indptr;       // n+1, absolute
    std::vector<uint64_t> blob_off;     // nc+1 absolute member offsets of the chunk blobs
};

struct Scratch {
    ZstdDCtx dctx;
    std::vector<uint8_t> raw, packed, prefilter;
    std::vector<uint32_t> gaps;
    std::vector<std::pair<uint32_t, uint32_t>> rv;
    std::vector<uint64_t> keys;
};

inline void open(Handle& H, const uint8_t* buf, size_t size, bool verify_file) {
    ReadLimits lim;
    H.buf = buf; H.size = size;
    if (verify_file) {
        H.hdr = validate_buffer(buf, size, lim);
    } else {
        if (size < sizeof(PZHeader) + sizeof(PZFooter)) throw PZReadError("file too short");
        std::memcpy(&H.hdr, buf, sizeof(PZHeader));
        if (H.hdr.magic != TP1_MAGIC || H.hdr.version != TP1_VERSION) throw PZReadError("bad magic/version");
        if (!(H.hdr.flags & FLAG_HAS_PERM) || H.hdr.feature_flags != (FEAT_ZSTD_CHECKSUMS | FEAT_BITPLANE_BITMAP))
            throw PZReadError("unsupported layout");
        if (H.hdr.n && (H.hdr.chunk_cols == 0 || H.hdr.num_chunks != (uint64_t(H.hdr.n) + H.hdr.chunk_cols - 1) / H.hdr.chunk_cols))
            throw PZReadError("chunk geometry");
        uint64_t table_off = sizeof(PZHeader) + uint64_t(H.hdr.perm_z_sz) + H.hdr.ptr_z_sz;
        if (table_off + 4ull * H.hdr.num_chunks > size - sizeof(PZFooter)) throw PZReadError("tables past end");
    }
    CRC32::compute("", 0);  // build the CRC table before any thread uses it (production table init is not thread safe)
    size_t off = sizeof(PZHeader);
    decode_perm_and_ptr(buf, off, H.hdr, H.perm, H.cc);
    H.indptr.assign(size_t(H.hdr.n) + 1, 0);
    uint64_t run = 0;
    for (uint32_t j = 0; j < H.hdr.n; ++j) { H.indptr[j] = (uint32_t)run; run += H.cc[j]; if (run > H.hdr.nnz) throw PZReadError("cc exceeds nnz"); }
    H.indptr[H.hdr.n] = (uint32_t)run;
    if (run != H.hdr.nnz) throw PZReadError("nnz mismatch");
    const size_t nc = H.hdr.num_chunks;
    H.blob_off.assign(nc + 1, 0);
    uint64_t cur = off + 4ull * nc, body_end = size - sizeof(PZFooter);
    for (size_t c = 0; c < nc; ++c) {
        uint32_t sz; std::memcpy(&sz, buf + off + 4 * c, 4);
        if (sz < 20 || cur > body_end || sz > body_end - cur) throw PZReadError("chunk blob out of range");
        H.blob_off[c] = cur; cur += sz;
    }
    H.blob_off[nc] = cur;
}

template <class V>
inline void decode_chunk(const Handle& H, size_t c, Scratch& S, int32_t* idx_out, V* val_out, uint64_t base, bool packed_sort) {
    const PZHeader& hdr = H.hdr;
    const uint8_t* blob = H.buf + H.blob_off[c];
    const uint32_t blob_sz = (uint32_t)(H.blob_off[c + 1] - H.blob_off[c]);
    uint32_t ng32, msz32, zsz, chunk_crc, psz32;
    std::memcpy(&ng32, blob, 4); std::memcpy(&msz32, blob + 4, 4); std::memcpy(&zsz, blob + 8, 4);
    std::memcpy(&chunk_crc, blob + 12, 4); std::memcpy(&psz32, blob + 16, 4);
    if (zsz != blob_sz - 20) throw PZReadError("chunk header size mismatch");
    const int gw = (hdr.flags & FLAG_GAP16) ? 2 : 4;
    const size_t col_start = c * size_t(hdr.chunk_cols);
    const size_t col_end = std::min<size_t>(col_start + hdr.chunk_cols, hdr.n);
    const uint64_t vmax = (hdr.vt_code == 1) ? 0xFFull : (hdr.vt_code == 2) ? 0xFFFFull : 0xFFFFFFFFull;
    if (msz32 > 5ull * (col_end - col_start) + 10ull * ng32) throw PZReadError("chunk metadata size");
    const size_t raw_sz = size_t(msz32) + size_t(ng32) * gw;
    const size_t bp_sz = ng32 ? 8 * ((size_t(ng32) + 7) / 8) : 0;
    const size_t pf_sz = size_t(msz32) + bp_sz + (ng32 ? size_t(gw - 1) * ng32 : 0);
    if (psz32 > (pf_sz + 7) / 8 + pf_sz) throw PZReadError("chunk packed size");

    S.packed = zstd_decompress(S.dctx.ctx, blob + 20, zsz, psz32);
    if (S.packed.size() != psz32) throw PZReadError("chunk decompressed size");
    if (CRC32::compute(S.packed.data(), S.packed.size()) != chunk_crc) throw PZReadError("chunk CRC32 mismatch");
    S.prefilter.assign(pf_sz, 0);
    bitmap_unpack(S.packed.data(), S.packed.size(), S.prefilter.data(), pf_sz);
    S.raw.resize(raw_sz);
    std::memcpy(S.raw.data(), S.prefilter.data(), msz32);
    if (ng32) {
        bit_planes_decode(S.prefilter.data() + msz32, ng32, S.raw.data() + msz32);
        if (gw > 1) std::memcpy(S.raw.data() + msz32 + ng32, S.prefilter.data() + msz32 + bp_sz, size_t(gw - 1) * ng32);
    }
    S.gaps.assign(ng32, 0);
    if (ng32) { if (gw == 2) byte_unsplit_16(S.raw.data() + msz32, ng32, S.gaps.data()); else byte_unsplit_32(S.raw.data() + msz32, ng32, S.gaps.data()); }

    size_t mp = 0, gp = 0;
    const uint8_t* meta = S.raw.data();
    for (size_t j = col_start; j < col_end; ++j) {
        uint32_t ng_col = varint_read(meta, msz32, mp);
        if (ng_col == 0) { if (H.cc[j] != 0) throw PZReadError("column has entries but empty value list"); continue; }
        const uint32_t ccj = H.cc[j];
        uint32_t filled = 0;
        if (packed_sort) { S.keys.clear(); S.keys.reserve(ccj); } else { S.rv.clear(); S.rv.reserve(ccj); }
        for (uint32_t vi = 0; vi < ng_col; ++vi) {
            uint32_t cv = varint_read(meta, msz32, mp);
            uint32_t cnt = varint_read(meta, msz32, mp);
            if (cv > vmax) throw PZReadError("value exceeds type");
            if (cnt > ccj - filled) throw PZReadError("group longer than column");
            uint64_t prev = 0;
            for (uint32_t k = 0; k < cnt; ++k) {
                if (gp >= ng32) throw PZReadError("gap stream exhausted");
                uint64_t prow = prev + S.gaps[gp++];
                prev = prow + 1;
                if (prow >= hdr.m) throw PZReadError("row out of bounds");
                if (packed_sort) S.keys.push_back((uint64_t(H.perm[prow]) << 32) | cv); else S.rv.emplace_back(H.perm[prow], cv);
            }
            filled += cnt;
        }
        const uint64_t o = uint64_t(H.indptr[j]) - base;
        if (packed_sort) {
            std::sort(S.keys.begin(), S.keys.end());
            if (S.keys.size() != ccj) throw PZReadError("column length mismatch");
            for (size_t k = 0; k < S.keys.size(); ++k) {
                if (k && (S.keys[k] >> 32) == (S.keys[k - 1] >> 32)) throw PZReadError("duplicate row");
                idx_out[o + k] = (int32_t)(S.keys[k] >> 32);
                val_out[o + k] = (V)(uint32_t)S.keys[k];
            }
        } else {
            std::sort(S.rv.begin(), S.rv.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            if (S.rv.size() != ccj) throw PZReadError("column length mismatch");
            for (size_t k = 0; k < S.rv.size(); ++k) {
                if (k && S.rv[k].first == S.rv[k - 1].first) throw PZReadError("duplicate row");
                idx_out[o + k] = (int32_t)S.rv[k].first;
                val_out[o + k] = (V)S.rv[k].second;
            }
        }
    }
    if (gp != ng32 || mp != msz32) throw PZReadError("chunk leftover");
}

// Decode chunks [c0, c1) into idx_out/val_out, which hold the entries of columns [c0*chunk_cols, min(c1*chunk_cols, n)).
template <class V>
inline void decode_range(const Handle& H, size_t c0, size_t c1, int32_t* idx_out, V* val_out, int threads, bool packed_sort) {
    const size_t col0 = c0 * size_t(H.hdr.chunk_cols);
    const uint64_t base = H.indptr[std::min<size_t>(col0, H.hdr.n)];
    std::atomic<size_t> next(c0);
    std::string err; std::mutex mu; std::atomic<bool> failed(false);
    auto work = [&]() {
        Scratch S;
        try {
            for (;;) {
                size_t c = next.fetch_add(1);
                if (c >= c1 || failed.load()) break;
                decode_chunk<V>(H, c, S, idx_out, val_out, base, packed_sort);
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> g(mu);
            if (!failed.exchange(true)) err = e.what();
        }
    };
    if (threads <= 1) work();
    else {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back(work);
        for (auto& t : th) t.join();
    }
    if (failed.load()) throw PZReadError(err);
}
}  // namespace w4r
