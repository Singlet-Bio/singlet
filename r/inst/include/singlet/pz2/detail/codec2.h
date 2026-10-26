// SPDX-License-Identifier: MIT
// pz2 codec 2 ("generic"): per chunk three zstd streams.  Any m, any values, any chunk shape.  Format: docs/format-store.md 5.2.
//   chunk := STREAM(counts) STREAM(gaps) STREAM(values)      STREAM := u32 raw_len, u32 zstd_len, zstd frame (raw_len 0: no frame)
//   counts : LEB128 of the number of non-zeros of each column of the chunk
//   gaps   : for each column, for each element: LEB128(row - previous_row - 1), previous_row = -1 at the column start
//   values : Int / F32Int: LEB128(value) per element;  F32Raw: 4 bytes little-endian bit pattern per element
// Row ids are canonical (row_order 0), so native and canonical decode are identical.
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {
namespace c2 {

inline void encode_chunk(const CSC& in, const std::vector<uint64_t>& ptr, const uint32_t* val, VMode vm, uint32_t c0, uint32_t c1, int zlevel,
                         std::vector<uint8_t>& blob) {
    std::vector<uint8_t> cnt, gap, vals;
    for (uint32_t j = c0; j < c1; ++j) {
        leb_put(cnt, ptr[j + 1] - ptr[j]);
        int64_t prev = -1;
        for (uint64_t t = ptr[j]; t < ptr[j + 1]; ++t) {
            leb_put(gap, uint64_t(int64_t(in.indices[t]) - prev - 1));
            prev = in.indices[t];
            if (vm == VMode::F32Raw) put_u32(vals, val[t]);
            else leb_put(vals, val[t]);
        }
    }
    blob.clear();
    put_stream(blob, cnt.data(), cnt.size(), zlevel);
    put_stream(blob, gap.data(), gap.size(), zlevel);
    put_stream(blob, vals.data(), vals.size(), zlevel);
}

inline void chunk_counts(const uint8_t* blob, size_t bytes, uint32_t nc, std::vector<uint32_t>& k, const uint8_t** after = nullptr) {
    const uint8_t* p = blob;
    const uint8_t* end = blob + bytes;
    auto cnt = get_stream(p, end, uint64_t(nc) * 5 + 8);
    const uint8_t* q = cnt.data();
    const uint8_t* qe = q + cnt.size();
    k.resize(nc);
    for (uint32_t j = 0; j < nc; ++j) {
        uint64_t v = leb_get(q, qe);
        if (v > 0xFFFFFFFFull) throw Error(Errc::Corrupt, "codec2: count overflow");
        k[j] = uint32_t(v);
    }
    if (q != qe) throw Error(Errc::Corrupt, "codec2: trailing bytes in counts");
    if (after) *after = p;
}

/// Decode a chunk into counts, row ids (canonical) and coded values.  `m` bounds row ids.
inline void decode_chunk(const uint8_t* blob, size_t bytes, uint32_t nc, uint64_t expect_nnz, uint32_t m, VMode vm, std::vector<uint32_t>& k, int32_t* idx_out,
                         uint32_t* val_out) {
    const uint8_t* p = nullptr;
    chunk_counts(blob, bytes, nc, k, &p);
    const uint8_t* end = blob + bytes;
    uint64_t N = 0;
    for (uint32_t x : k) N += x;
    if (N != expect_nnz) throw Error(Errc::Corrupt, "codec2: counts disagree with the chunk index");
    auto gap = get_stream(p, end, N * 5 + 8);
    auto vals = get_stream(p, end, N * 5 + 8);
    const uint8_t* g = gap.data();
    const uint8_t* ge = g + gap.size();
    const uint8_t* v = vals.data();
    const uint8_t* ve = v + vals.size();
    if (vm == VMode::F32Raw && vals.size() != N * 4) throw Error(Errc::Corrupt, "codec2: value stream length");
    size_t o = 0;
    for (uint32_t j = 0; j < nc; ++j) {
        int64_t prev = -1;
        for (uint32_t t = 0; t < k[j]; ++t, ++o) {
            uint64_t d = leb_get(g, ge);
            int64_t id = prev + 1 + int64_t(d);
            if (d > 0xFFFFFFFFull || id >= int64_t(m)) throw Error(Errc::Corrupt, "codec2: row id out of range");
            idx_out[o] = int32_t(id);
            prev = id;
            if (vm == VMode::F32Raw) { val_out[o] = ld_u32(v); v += 4; }
            else {
                uint64_t x = leb_get(v, ve);
                if (x > 0xFFFFFFFFull) throw Error(Errc::Corrupt, "codec2: value overflow");
                val_out[o] = uint32_t(x);
            }
        }
    }
    if (g != ge || v != ve) throw Error(Errc::Corrupt, "codec2: trailing bytes");
}

}  // namespace c2
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
