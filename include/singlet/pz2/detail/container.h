// SPDX-License-Identifier: MIT
// pz2 internals: member header, chunk index, names parts.  Byte layouts are specified in docs/format-store.md section 4.
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {

inline constexpr char kMagic[4] = {'P', 'Z', '2', 'M'};
inline PackId cells_pack_id() { return PackId{{'C', 'E', 'L', 'L', 'S', 0, 0, 0}}; }

// Header byte offsets (all little-endian).  Total 256 bytes; bytes [252,256) are the CRC-32 of bytes [0,252).
//   0 magic[4]  4 version u16  6 codec u16  8 m u32  12 n u32  16 nnz u64
//   24 value_type u8  25 row_order u8  26 classes_per_octave u8  27 flags u8  28 chunk_cols u32  32 n_chunks u32
//   36 lanes u8  37 prob_bits u8  38 value_prob_bits u8  39 ctx_kind u8
//   40 row_names_id[8]  48 col_names_id[8]  56 digest[16]  72 section[7] x {off u64, len u64}  184 total_len u64
//   192 rows_kind u8  193 cols_kind u8  194..251 zero  252 crc32
inline std::array<uint8_t, kHeaderBytes> write_header(const Header& h) {
    std::array<uint8_t, kHeaderBytes> b{};
    uint8_t* p = b.data();
    std::memcpy(p, kMagic, 4);
    p[4] = uint8_t(h.version); p[5] = uint8_t(h.version >> 8);
    p[6] = uint8_t(uint16_t(h.codec)); p[7] = uint8_t(uint16_t(h.codec) >> 8);
    st_u32(p + 8, h.m); st_u32(p + 12, h.n); st_u64(p + 16, h.nnz);
    p[24] = uint8_t(h.value_type); p[25] = h.row_order; p[26] = h.classes_per_octave; p[27] = h.flags;
    st_u32(p + 28, h.chunk_cols); st_u32(p + 32, h.n_chunks);
    p[36] = h.lanes; p[37] = h.prob_bits; p[38] = h.value_prob_bits; p[39] = h.ctx_kind;
    std::memcpy(p + 40, h.row_names_id.data(), 8);
    std::memcpy(p + 48, h.col_names_id.data(), 8);
    std::memcpy(p + 56, h.digest.b.data(), 16);
    for (int s = 0; s < kNumSections; ++s) { st_u64(p + 72 + 16 * s, h.section[s].off); st_u64(p + 80 + 16 * s, h.section[s].len); }
    st_u64(p + 184, h.total_len);
    p[192] = uint8_t(h.rows_kind); p[193] = uint8_t(h.cols_kind);
    st_u32(p + 252, crc32(p, 252));
    return b;
}

}  // namespace detail

inline Header read_header(const uint8_t* p, size_t n) {
    using namespace detail;
    if (n < kHeaderBytes) throw Error(Errc::Truncated, "header needs 256 bytes", 0, kHeaderBytes);
    if (std::memcmp(p, kMagic, 4) != 0) throw Error(Errc::BadMagic, "bad magic (not a pz2 member)");
    Header h;
    h.version = ld_u16(p + 4);
    if (h.version != kFormatVersion) throw Error(Errc::Unsupported, "unsupported pz2 version " + std::to_string(h.version));
    if (ld_u32(p + 252) != crc32(p, 252)) throw Error(Errc::Corrupt, "header CRC mismatch");
    uint16_t codec = ld_u16(p + 6);
    if (codec != 1 && codec != 2) throw Error(Errc::Unsupported, "unknown codec " + std::to_string(codec));
    h.codec = CodecId(codec);
    h.m = ld_u32(p + 8); h.n = ld_u32(p + 12); h.nnz = ld_u64(p + 16);
    h.value_type = ValueType(p[24]);
    if (p[24] < 1 || p[24] > 4) throw Error(Errc::Corrupt, "bad value_type");
    h.row_order = p[25]; h.classes_per_octave = p[26]; h.flags = p[27];
    h.chunk_cols = ld_u32(p + 28); h.n_chunks = ld_u32(p + 32);
    h.lanes = p[36]; h.prob_bits = p[37]; h.value_prob_bits = p[38]; h.ctx_kind = p[39];
    std::memcpy(h.row_names_id.data(), p + 40, 8);
    std::memcpy(h.col_names_id.data(), p + 48, 8);
    std::memcpy(h.digest.b.data(), p + 56, 16);
    for (int s = 0; s < kNumSections; ++s) { h.section[s].off = ld_u64(p + 72 + 16 * s); h.section[s].len = ld_u64(p + 80 + 16 * s); }
    h.total_len = ld_u64(p + 184);
    if (p[192] > 4 || p[193] > 4) throw Error(Errc::Corrupt, "bad names kind");
    h.rows_kind = NamesKind(p[192]); h.cols_kind = NamesKind(p[193]);
    if (h.rows_kind == NamesKind::Cells) throw Error(Errc::Corrupt, "rows cannot reference the cells table");
    // structural checks: every section lies inside the member and after the header
    for (int s = 0; s < kNumSections; ++s) {
        const SectionRef& r = h.section[s];
        if (r.len == 0) continue;
        if (r.off < kHeaderBytes || r.off > h.total_len || r.len > h.total_len - r.off) throw Error(Errc::Corrupt, "section outside the member");
    }
    if (h.section[kChunkIndex].len != (uint64_t(h.n_chunks) + 1) * kChunkRecBytes) throw Error(Errc::Corrupt, "chunk index length does not match n_chunks");
    if (h.codec == CodecId::Rans && h.m > kRansMaxRows) throw Error(Errc::Corrupt, "codec 1 member with m >= 2^21");
    return h;
}

inline ChunkIndex read_chunk_index(const Header& h, const uint8_t* p, size_t n) {
    using namespace detail;
    const uint64_t need = (uint64_t(h.n_chunks) + 1) * kChunkRecBytes;
    if (n < need) throw Error(Errc::Truncated, "chunk index truncated", h.section[kChunkIndex].off, need);
    ChunkIndex ix;
    ix.rec.resize(size_t(h.n_chunks) + 1);
    for (size_t c = 0; c < ix.rec.size(); ++c) {
        const uint8_t* q = p + c * kChunkRecBytes;
        ix.rec[c].off = ld_u64(q); ix.rec[c].len = ld_u32(q + 8); ix.rec[c].first_col = ld_u32(q + 12); ix.rec[c].nnz_before = ld_u64(q + 16);
    }
    // validate monotonicity and the sentinel
    const uint64_t clen = h.section[kChunks].len;
    for (size_t c = 0; c < h.n_chunks; ++c) {
        const ChunkRec &a = ix.rec[c], &b = ix.rec[c + 1];
        if (b.first_col <= a.first_col || b.nnz_before < a.nnz_before || b.off != a.off + a.len) throw Error(Errc::Corrupt, "chunk index not monotone");
    }
    const ChunkRec& s = ix.rec.back();
    if (ix.rec[0].first_col != 0 || ix.rec[0].off != 0 || ix.rec[0].nnz_before != 0) throw Error(Errc::Corrupt, "chunk index must start at 0");
    if (s.first_col != h.n || s.nnz_before != h.nnz || s.off != clen || s.len != 0) throw Error(Errc::Corrupt, "chunk index sentinel mismatch");
    return ix;
}

namespace detail {

inline void write_chunk_index(std::vector<uint8_t>& o, const std::vector<ChunkRec>& rec) {
    for (const ChunkRec& r : rec) { put_u64(o, r.off); put_u32(o, r.len); put_u32(o, r.first_col); put_u64(o, r.nnz_before); }
}

// ---- names parts ---------------------------------------------------------------------------------------------
// Inline    : STREAM(zstd of the NUL-terminated names)             -- u32 raw_len, u32 zstd_len, frame
// PackSubset: STREAM(zstd of LEB128(zigzag(index[i] - index[i-1])), index[-1] = 0)
// Pack/Cells/None: empty.
inline uint64_t zigzag(int64_t v) { return (uint64_t(v) << 1) ^ uint64_t(v >> 63); }
inline int64_t unzigzag(uint64_t u) { return int64_t(u >> 1) ^ -int64_t(u & 1); }

inline std::vector<uint8_t> encode_names_part(const NamesSpec& s, uint64_t count, int level, bool allow_cells) {
    std::vector<uint8_t> o;
    switch (s.kind) {
        case NamesKind::Inline: {
            if (blob_count(s.blob) != count) throw Error(Errc::BadArgument, "inline names: blob holds " + std::to_string(blob_count(s.blob)) + " names, expected " + std::to_string(count));
            if (count && s.blob.back() != 0) throw Error(Errc::BadArgument, "inline names: blob must end with NUL");
            put_stream(o, s.blob.data(), s.blob.size(), level);
            break;
        }
        case NamesKind::PackSubset: {
            if (s.pack_index.size() != count) throw Error(Errc::BadArgument, "pack subset: index list length != count");
            std::vector<uint8_t> raw;
            int64_t prev = 0;
            for (uint32_t v : s.pack_index) { leb_put(raw, zigzag(int64_t(v) - prev)); prev = v; }
            put_stream(o, raw.data(), raw.size(), level);
            break;
        }
        case NamesKind::Pack: case NamesKind::None: break;
        case NamesKind::Cells:
            if (!allow_cells) throw Error(Errc::BadArgument, "rows cannot reference the cells table");
            break;
    }
    return o;
}

inline NamesSpec decode_names_part(NamesKind kind, const PackId& id, const uint8_t* p, size_t n, uint64_t count) {
    NamesSpec s;
    s.kind = kind; s.pack_id = id; s.count = count;
    const uint8_t* end = p + n;
    if (kind == NamesKind::Inline) {
        s.blob = get_stream(p, end, uint64_t(1) << 34);
        if (blob_count(s.blob) != count) throw Error(Errc::Corrupt, "inline names: wrong count");
    } else if (kind == NamesKind::PackSubset) {
        auto raw = get_stream(p, end, count * 5 + 16);
        const uint8_t* q = raw.data();
        const uint8_t* qe = q + raw.size();
        s.pack_index.resize(size_t(count));
        int64_t prev = 0;
        for (uint64_t i = 0; i < count; ++i) { prev += unzigzag(leb_get(q, qe)); if (prev < 0 || prev > int64_t(0xFFFFFFFFll)) throw Error(Errc::Corrupt, "pack index out of range"); s.pack_index[size_t(i)] = uint32_t(prev); }
    }
    return s;
}

}  // namespace detail
}  // namespace pz2
}  // namespace singlet
