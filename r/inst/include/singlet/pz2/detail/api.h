// SPDX-License-Identifier: MIT
// pz2 internals: public entry points (encode_matrix, Matrix, decode, canonical_digest).  Container layout: docs/format-v2.md 4.
#pragma once

namespace singlet {
namespace pz2 {

// ===========================================================================================================
// Matrix::Impl
// ===========================================================================================================
struct Matrix::Impl {
    Header h;
    ChunkIndex ix;
    std::vector<uint32_t> perm;                        // stored -> canonical (row_order 1)
    mutable std::vector<uint32_t> inv;                 // canonical -> stored, lazy
    mutable std::once_flag inv_once;
    std::shared_ptr<const detail::r1::Prep> prep;      // codec 1 decode tables
};

inline const Header& Matrix::header() const { return impl_->h; }
inline const ChunkIndex& Matrix::chunk_index() const { return impl_->ix; }
inline const std::vector<uint32_t>& Matrix::row_perm() const { return impl_->perm; }
inline const std::vector<uint32_t>& Matrix::row_inv() const {
    const Impl& I = *impl_;
    std::call_once(I.inv_once, [&I] {
        if (I.perm.empty()) return;
        I.inv.assign(I.perm.size(), 0);
        for (size_t s = 0; s < I.perm.size(); ++s) I.inv[I.perm[s]] = uint32_t(s);
    });
    return I.inv;
}

inline ColumnPlan Matrix::plan_columns(uint32_t col0, uint32_t col1) const {
    const Impl& I = *impl_;
    if (col0 > col1 || col1 > I.h.n) throw Error(Errc::BadArgument, "column range out of bounds");
    ColumnPlan p;
    p.col0 = col0; p.col1 = col1;
    if (col0 == col1) return p;
    p.chunk0 = I.ix.chunk_of_column(col0);
    p.chunk1 = I.ix.chunk_of_column(col1 - 1) + 1;
    const ChunkRec &a = I.ix.rec[p.chunk0], &b = I.ix.rec[p.chunk1];
    p.byte_off = I.h.section[kChunks].off + a.off;
    p.byte_len = b.off - a.off;
    p.max_nnz = b.nnz_before - a.nnz_before;
    p.first_col = a.first_col;
    return p;
}

inline Matrix Matrix::open(const ByteSource& src) {
    using namespace detail;
    auto impl = std::make_shared<Impl>();
    impl->h = read_header(src.get(0, kHeaderBytes), kHeaderBytes);
    const Header& h = impl->h;
    const SectionRef& ci = h.section[kChunkIndex];
    impl->ix = read_chunk_index(h, src.get(ci.off, ci.len), size_t(ci.len));
    if (h.codec == CodecId::Rans) {
        const SectionRef& tb = h.section[kTables];
        impl->prep = r1::make_prep(h, src.get(tb.off, tb.len), size_t(tb.len));
        if (h.row_order != 1) throw Error(Errc::Corrupt, "codec 1 requires row_order 1");
        const SectionRef& rw = h.section[kRows];
        const uint8_t* p = src.get(rw.off, rw.len);
        auto raw = get_stream(p, p + rw.len, uint64_t(h.m) * 5 + 16);
        std::vector<uint32_t> key(h.m);
        const uint8_t* q = raw.data();
        const uint8_t* qe = q + raw.size();
        for (uint32_t i = 0; i < h.m; ++i) { uint64_t v = leb_get(q, qe); if (v > 0xFFFFFFFFull) throw Error(Errc::Corrupt, "row key overflow"); key[i] = uint32_t(v); }
        if (q != qe) throw Error(Errc::Corrupt, "rows: trailing bytes");
        impl->perm = r1::perm_from_keys(key);
    } else if (h.row_order != 0) {
        throw Error(Errc::Corrupt, "codec 2 requires row_order 0");
    }
    Matrix mx;
    mx.impl_ = std::move(impl);
    return mx;
}

// ===========================================================================================================
// names / kv
// ===========================================================================================================
inline NamesSpec Matrix::row_names(const ByteSource& src) const {
    const Header& h = impl_->h;
    const SectionRef& s = h.section[kExtra];
    const uint8_t* p = (h.rows_kind == NamesKind::Inline || h.rows_kind == NamesKind::PackSubset) ? src.get(s.off, s.len) : nullptr;
    return detail::decode_names_part(h.rows_kind, h.row_names_id, p, size_t(s.len), h.rows_kind == NamesKind::None ? 0 : h.m);
}
inline NamesSpec Matrix::col_names(const ByteSource& src) const {
    const Header& h = impl_->h;
    const SectionRef& s = h.section[kCols];
    const uint8_t* p = (h.cols_kind == NamesKind::Inline || h.cols_kind == NamesKind::PackSubset) ? src.get(s.off, s.len) : nullptr;
    return detail::decode_names_part(h.cols_kind, h.col_names_id, p, size_t(s.len), h.cols_kind == NamesKind::None ? 0 : h.n);
}
inline KvMap Matrix::kv(const ByteSource& src) const {
    using namespace detail;
    KvMap out;
    const SectionRef& s = impl_->h.section[kKv];
    if (!s.len) return out;
    const uint8_t* p = src.get(s.off, s.len);
    const uint8_t* end = p + s.len;
    if (end - p < 4) throw Error(Errc::Corrupt, "kv truncated");
    uint32_t npairs = ld_u32(p);
    p += 4;
    auto blob = get_stream(p, end, uint64_t(1) << 30);
    auto parts = split_names(blob, 2 * size_t(npairs));
    if (blob_count(blob) != 2 * uint64_t(npairs)) throw Error(Errc::Corrupt, "kv: wrong number of strings");
    for (uint32_t i = 0; i < npairs; ++i) out[parts[2 * i]] = parts[2 * i + 1];
    return out;
}

// ===========================================================================================================
// decoding
// ===========================================================================================================
namespace detail {

/// Per-thread workspace of the canonical sort: a two-level occupancy bitmap over the canonical row space plus a value table.
struct CanonWork {
    std::vector<uint64_t> l0, l1;   // bit x of l0 = canonical row x present in the current column; bit w of l1 = l0 word w non-empty
    std::vector<uint32_t> vt;       // value of canonical row x
    std::vector<uint64_t> tmp;
    void prepare(uint32_t m) {
        const size_t w0 = (size_t(m) >> 6) + 2, w1 = (w0 >> 6) + 2;
        if (l0.size() < w0) l0.assign(w0, 0);
        if (l1.size() < w1) l1.assign(w1, 0);
        if (vt.size() < size_t(m) + 1) vt.resize(size_t(m) + 1);
    }
};

struct Scratch {
    r1::Setup su;
    std::vector<uint32_t> k;
    std::vector<int32_t> idx;
    std::vector<uint32_t> val;
    std::vector<uint8_t> pad;
    CanonWork canon;
};

inline bool run_kernel(Isa isa, const r1::Prep& P, const r1::Setup& su, int32_t* idx, uint32_t* val) {
#ifdef PZ2_COMPILED_AVX512
    if (isa == Isa::Avx512) return r1::decode_chunk_avx512(P, su, idx, val);
#endif
#ifdef PZ2_COMPILED_AVX2
    if (isa == Isa::Avx2) return r1::decode_chunk_avx2(P, su, idx, val);
#endif
    (void)isa;
    return r1::decode_chunk_scalar(P, su, idx, val);
}

/// Map stored ids to canonical ids through perm and sort every column ascending (ids are unique).  Columns of up to 24 entries use
/// an insertion sort, larger ones the bitmap scatter/scan (O(k + m/4096)); the bitmap is left clean after every column.
inline void canonicalize(const uint32_t* perm, uint32_t m, const std::vector<uint32_t>& k, int32_t* idx32, uint32_t* val, CanonWork& w) {
    uint32_t* idx = reinterpret_cast<uint32_t*>(idx32);
    size_t o = 0;
    for (uint32_t kk : k) {
        if (!kk) continue;
        bool sorted = true;
        uint32_t prev = 0;
        for (uint32_t i = 0; i < kk; ++i) {
            uint32_t s = idx[o + i];
            if (s >= m) throw Error(Errc::Corrupt, "decoded row id out of range");
            uint32_t c = perm[s];
            idx[o + i] = c;
            if (i && c < prev) sorted = false;
            prev = c;
        }
        if (!sorted) {
            if (kk <= 24) {   // insertion sort of (id, value) pairs
                for (uint32_t i = 1; i < kk; ++i) {
                    uint32_t ci = idx[o + i], vi = val[o + i];
                    uint32_t j = i;
                    while (j > 0 && idx[o + j - 1] > ci) { idx[o + j] = idx[o + j - 1]; val[o + j] = val[o + j - 1]; --j; }
                    idx[o + j] = ci; val[o + j] = vi;
                }
            } else {
                w.prepare(m);
                uint64_t* l0 = w.l0.data();
                uint64_t* l1 = w.l1.data();
                uint32_t* vt = w.vt.data();
                for (uint32_t i = 0; i < kk; ++i) {
                    const uint32_t x = idx[o + i];
                    l0[x >> 6] |= uint64_t(1) << (x & 63);
                    l1[x >> 12] |= uint64_t(1) << ((x >> 6) & 63);
                    vt[x] = val[o + i];
                }
                size_t p = o;
                const size_t n1 = (size_t(m) >> 12) + 1;
                for (size_t i1 = 0; i1 < n1; ++i1) {
                    uint64_t b1 = l1[i1];
                    if (!b1) continue;
                    l1[i1] = 0;
                    do {
                        const uint32_t i0 = uint32_t(i1 << 6) | uint32_t(ctz64(b1));
                        b1 &= b1 - 1;
                        uint64_t b0 = l0[i0];
                        l0[i0] = 0;
                        do {
                            const uint32_t x = (i0 << 6) | uint32_t(ctz64(b0));
                            b0 &= b0 - 1;
                            idx[p] = x;
                            val[p] = vt[x];
                            ++p;
                        } while (b0);
                    } while (b1);
                }
            }
        }
        o += kk;
    }
}

inline void write_int(IntType t, void* dst, size_t i, uint64_t v) {
    switch (t) {
        case IntType::U32: static_cast<uint32_t*>(dst)[i] = uint32_t(v); break;
        case IntType::I32: static_cast<int32_t*>(dst)[i] = int32_t(v); break;
        case IntType::U64: static_cast<uint64_t*>(dst)[i] = v; break;
        case IntType::I64: static_cast<int64_t*>(dst)[i] = int64_t(v); break;
    }
}

}  // namespace detail

namespace {
inline void check_out(const Header& h, const Output& out, uint64_t n_ptr, uint64_t nnz) {
    if (out.indices && out.indices_type != IntType::U32 && out.indices_type != IntType::I32) throw Error(Errc::BadArgument, "indices must be U32 or I32");
    if (out.indptr) {
        if (out.indptr_capacity < n_ptr) throw Error(Errc::OutputTooSmall, "indptr buffer too small", n_ptr, 0);
        if (out.indptr_type == IntType::I32 && nnz > 0x7FFFFFFFull) throw Error(Errc::TypeMismatch, "nnz does not fit an int32 indptr");
        if (out.indptr_type == IntType::U32 && nnz > 0xFFFFFFFFull) throw Error(Errc::TypeMismatch, "nnz does not fit a uint32 indptr");
    }
    if (out.indices) {
        if (out.indices_capacity < nnz) throw Error(Errc::OutputTooSmall, "indices buffer too small", nnz, 0);
        if (out.indices_type == IntType::I32 && h.m > 0x7FFFFFFFu) throw Error(Errc::TypeMismatch, "row ids do not fit int32");
    }
    if (out.data && out.data_capacity < nnz) throw Error(Errc::OutputTooSmall, "data buffer too small", nnz, 0);
}
}  // namespace

namespace detail {

/// Decode one chunk into idx_dst / val_dst (N elements each; coded values) and sc.k (counts).  Handles tail padding, ISA, canonical.
inline void decode_one_chunk(const Matrix::Impl& I, const ByteSource& src, size_t c, Isa isa, bool canonical, Scratch& sc, int32_t* idx_dst, uint32_t* val_dst) {
    const Header& h = I.h;
    const ChunkRec& rc = I.ix.rec[c];
    const uint64_t off = h.section[kChunks].off + rc.off;
    const uint32_t nc = I.ix.cols_in_chunk(c);
    const uint64_t N = I.ix.nnz_in_chunk(c);
    if (h.codec == CodecId::Generic) {
        const uint8_t* blob = src.get(off, rc.len);
        c2::decode_chunk(blob, rc.len, nc, N, h.m, vmode_of(h), sc.k, idx_dst, val_dst);
        return;
    }
    const uint8_t* blob;
    if (src.has(off, uint64_t(rc.len) + r1::kPad)) blob = src.get(off, rc.len);
    else {   // the chunk ends at the end of the available bytes: decode from a padded copy so the SIMD refills cannot read past it
        sc.pad.assign(size_t(rc.len) + r1::kPad, 0);
        std::memcpy(sc.pad.data(), src.get(off, rc.len), rc.len);
        blob = sc.pad.data();
    }
    r1::setup_chunk(*I.prep, blob, rc.len, nc, N, sc.su);
    if (N && !run_kernel(isa, *I.prep, sc.su, idx_dst, val_dst)) throw Error(Errc::Corrupt, "chunk " + std::to_string(c) + ": entropy stream integrity check failed");
    sc.k.swap(sc.su.k);   // the kernels do not read su.k
    if (N && canonical) canonicalize(I.perm.data(), h.m, sc.k, idx_dst, val_dst, sc.canon);
}

}  // namespace detail

inline void Matrix::decode(const ByteSource& src, Output& out, const DecodeOptions& opt) const {
    using namespace detail;
    const Impl& I = *impl_;
    const Header& h = I.h;
    check_out(h, out, uint64_t(h.n) + 1, h.nnz);
    const Isa isa = resolve_isa(opt.isa);
    const bool canonical = opt.mode == Mode::Canonical;
    const VMode vm = vmode_of(h);
    const bool direct_val = out.data && out.data_type == DType::U32 && vm != VMode::F32Int;
    const size_t vsz = out.data ? dtype_size(out.data_type) : 0;
    if (out.indptr) write_int(out.indptr_type, out.indptr, 0, 0);
    const size_t nch = I.ix.n_chunks();
    if (nch) src.get(h.section[kChunks].off, h.section[kChunks].len);   // whole chunks section must be available
    const int nthr = resolve_threads(opt.threads, 4);
    std::vector<Scratch> scr(size_t(std::max(1, std::min<int>(nthr, int(std::max<size_t>(1, nch))))));
    parallel_for(nthr, nch, [&](size_t c, int ti) {
        Scratch& sc = scr[size_t(ti)];
        const ChunkRec& rc = I.ix.rec[c];
        const uint64_t N = I.ix.nnz_in_chunk(c), base = rc.nnz_before;
        int32_t* idx_dst;
        if (out.indices) idx_dst = static_cast<int32_t*>(out.indices) + base;
        else { sc.idx.resize(size_t(N) + 64); idx_dst = sc.idx.data(); }
        uint32_t* val_dst;
        if (direct_val) val_dst = static_cast<uint32_t*>(out.data) + base;
        else { sc.val.resize(size_t(N) + 64); val_dst = sc.val.data(); }
        decode_one_chunk(I, src, c, isa, canonical, sc, idx_dst, val_dst);
        if (out.data && !direct_val) write_values(out.data_type, vm, val_dst, size_t(N), static_cast<uint8_t*>(out.data) + base * vsz);
        if (out.indptr) {
            uint64_t acc = base;
            const uint32_t nc = I.ix.cols_in_chunk(c), first = rc.first_col;
            for (uint32_t j = 0; j < nc; ++j) { acc += sc.k[j]; write_int(out.indptr_type, out.indptr, size_t(first) + j + 1, acc); }
        }
    });
}

inline std::vector<uint32_t> Matrix::column_counts(const ByteSource& src, uint32_t col0, uint32_t col1) const {
    using namespace detail;
    const Impl& I = *impl_;
    ColumnPlan pl = plan_columns(col0, col1);
    std::vector<uint32_t> out;
    out.reserve(col1 - col0);
    std::vector<uint32_t> k;
    for (size_t c = pl.chunk0; c < pl.chunk1; ++c) {
        const ChunkRec& rc = I.ix.rec[c];
        const uint8_t* blob = src.get(I.h.section[kChunks].off + rc.off, rc.len);
        const uint32_t nc = I.ix.cols_in_chunk(c);
        if (I.h.codec == CodecId::Generic) c2::chunk_counts(blob, rc.len, nc, k);
        else r1::chunk_counts(*I.prep, blob, rc.len, nc, k);
        uint32_t a = std::max(col0, rc.first_col), b = std::min<uint32_t>(col1, rc.first_col + nc);
        for (uint32_t j = a; j < b; ++j) out.push_back(k[j - rc.first_col]);
    }
    return out;
}

inline uint64_t Matrix::decode_columns(const ByteSource& src, uint32_t col0, uint32_t col1, Output& out, const DecodeOptions& opt) const {
    using namespace detail;
    const Impl& I = *impl_;
    const Header& h = I.h;
    ColumnPlan pl = plan_columns(col0, col1);
    if (col0 == col1) { if (out.indptr) { if (out.indptr_capacity < 1) throw Error(Errc::OutputTooSmall, "indptr buffer too small", 1, 0); write_int(out.indptr_type, out.indptr, 0, 0); } return 0; }
    src.get(pl.byte_off, pl.byte_len);   // NeedRange early
    // phase 1: exact counts of the covered columns -> element ranges and output offsets
    const size_t nch = pl.chunk1 - pl.chunk0;
    std::vector<uint32_t> lo_el(nch), hi_el(nch);
    std::vector<uint64_t> out_off(nch + 1, 0);
    std::vector<uint32_t> k;
    for (size_t t = 0; t < nch; ++t) {
        const size_t c = pl.chunk0 + t;
        const ChunkRec& rc = I.ix.rec[c];
        const uint8_t* blob = src.get(h.section[kChunks].off + rc.off, rc.len);
        const uint32_t nc = I.ix.cols_in_chunk(c);
        if (h.codec == CodecId::Generic) c2::chunk_counts(blob, rc.len, nc, k); else r1::chunk_counts(*I.prep, blob, rc.len, nc, k);
        uint32_t a = std::max(col0, rc.first_col) - rc.first_col, b = std::min<uint32_t>(col1, rc.first_col + nc) - rc.first_col;
        uint64_t before = 0, within = 0;
        for (uint32_t j = 0; j < a; ++j) before += k[j];
        for (uint32_t j = a; j < b; ++j) within += k[j];
        lo_el[t] = uint32_t(before); hi_el[t] = uint32_t(before + within);
        out_off[t + 1] = out_off[t] + within;
    }
    const uint64_t total = out_off[nch];
    check_out(h, out, uint64_t(col1 - col0) + 1, total);
    const Isa isa = resolve_isa(opt.isa);
    const bool canonical = opt.mode == Mode::Canonical;
    const VMode vm = vmode_of(h);
    const size_t vsz = out.data ? dtype_size(out.data_type) : 0;
    if (out.indptr) write_int(out.indptr_type, out.indptr, 0, 0);
    const int nthr = resolve_threads(opt.threads, 4);
    std::vector<Scratch> scr(size_t(std::max(1, std::min<int>(nthr, int(nch)))));
    parallel_for(nthr, nch, [&](size_t t, int ti) {
        Scratch& sc = scr[size_t(ti)];
        const size_t c = pl.chunk0 + t;
        const ChunkRec& rc = I.ix.rec[c];
        const uint64_t N = I.ix.nnz_in_chunk(c);
        sc.idx.resize(size_t(N) + 64);
        sc.val.resize(size_t(N) + 64);
        decode_one_chunk(I, src, c, isa, canonical, sc, sc.idx.data(), sc.val.data());
        const uint32_t nc = I.ix.cols_in_chunk(c);
        const uint32_t a = std::max(col0, rc.first_col) - rc.first_col, b = std::min<uint32_t>(col1, rc.first_col + nc) - rc.first_col;
        const size_t cnt = hi_el[t] - lo_el[t];
        if (out.indices && cnt) std::memcpy(static_cast<int32_t*>(out.indices) + out_off[t], sc.idx.data() + lo_el[t], cnt * 4);
        if (out.data && cnt) write_values(out.data_type, vm, sc.val.data() + lo_el[t], cnt, static_cast<uint8_t*>(out.data) + out_off[t] * vsz);
        if (out.indptr) {
            uint64_t acc = out_off[t];
            for (uint32_t j = a; j < b; ++j) { acc += sc.k[j]; write_int(out.indptr_type, out.indptr, size_t(rc.first_col) + j + 1 - col0, acc); }
        }
    });
    return total;
}

inline void decode(const uint8_t* member, size_t n, Output& out, const DecodeOptions& opt) {
    ByteSource src(member, n);
    Matrix::open(src).decode(src, out, opt);
}

// ===========================================================================================================
// canonical digest
// ===========================================================================================================
namespace detail {

inline uint32_t load_value(const CSC& c, uint64_t i) {
    switch (c.data_type) {
        case DType::U8: return static_cast<const uint8_t*>(c.data)[i];
        case DType::U16: return static_cast<const uint16_t*>(c.data)[i];
        case DType::U32: return static_cast<const uint32_t*>(c.data)[i];
        case DType::F32: return f32_bits(static_cast<const float*>(c.data)[i]);
        default: throw Error(Errc::BadArgument, "unsupported input data type");
    }
}

}  // namespace detail

inline Digest16 canonical_digest(const CSC& c, ValueType vt, const std::vector<uint8_t>& row_names_blob, const std::vector<uint8_t>& col_names_blob, const KvMap& kv) {
    using namespace detail;
    if (vt == ValueType::Auto) throw Error(Errc::BadArgument, "canonical_digest needs a concrete value type");
    if (!c.indptr && !c.counts && c.n) throw Error(Errc::BadArgument, "CSC needs indptr or counts");
    Sha256 h;
    h.update("PZ2CANON1", 9);
    h.update_u32(c.m); h.update_u32(c.n); h.update_u64(c.nnz);
    const uint8_t vtb = uint8_t(vt);
    h.update(&vtb, 1);
    // little-endian hosts hash the index and u32 value arrays in place; other cases go through a small staging buffer
    static const uint32_t one = 1;
    const bool little = *reinterpret_cast<const uint8_t*>(&one) == 1;
    std::vector<uint8_t> buf;
    uint64_t pos = 0;
    for (uint32_t j = 0; j < c.n; ++j) {
        const uint32_t k = c.indptr ? c.indptr[j + 1] - c.indptr[j] : c.counts[j];
        h.update_u32(k);
        if (little) h.update(c.indices + pos, 4 * size_t(k));
        else { buf.resize(4 * size_t(k)); for (uint32_t t = 0; t < k; ++t) st_u32(buf.data() + 4 * size_t(t), c.indices[pos + t]); h.update(buf.data(), buf.size()); }
        if (little && c.data_type == DType::U32) h.update(static_cast<const uint32_t*>(c.data) + pos, 4 * size_t(k));
        else {
            buf.resize(4 * size_t(k));
            for (uint32_t t = 0; t < k; ++t) st_u32(buf.data() + 4 * size_t(t), load_value(c, pos + t));
            h.update(buf.data(), buf.size());
        }
        pos += k;
    }
    h.update_u32(uint32_t(blob_count(row_names_blob)));
    h.update(row_names_blob.data(), row_names_blob.size());
    h.update_u32(uint32_t(blob_count(col_names_blob)));
    h.update(col_names_blob.data(), col_names_blob.size());
    for (const auto& kvp : kv) {
        h.update(kvp.first.data(), kvp.first.size()); const uint8_t z = 0; h.update(&z, 1);
        h.update(kvp.second.data(), kvp.second.size()); h.update(&z, 1);
    }
    uint8_t full[32];
    h.final(full);
    Digest16 d;
    std::memcpy(d.b.data(), full, 16);
    return d;
}

// ===========================================================================================================
// encoding
// ===========================================================================================================
namespace detail {

struct Analysis {
    std::vector<uint64_t> ptr;       // n + 1
    std::vector<uint32_t> own;       // coded values when data is not already a u32 Int array
    const uint32_t* val = nullptr;   // coded values (nnz)
    VMode vm = VMode::Int;
    ValueType vt = ValueType::U32;
    uint32_t maxv = 0;
    bool has_zero = false;
};

inline Analysis analyse(const CSC& in, const EncodeOptions& opt) {
    Analysis a;
    if (in.n && !in.indptr && !in.counts) throw Error(Errc::BadArgument, "CSC needs indptr or counts");
    if (in.nnz && (!in.indices || !in.data)) throw Error(Errc::BadArgument, "CSC indices/data missing");
    if (in.indptr && in.nnz > 0xFFFFFFFFull) throw Error(Errc::BadArgument, "indptr form needs nnz < 2^32; pass counts");
    a.ptr.assign(size_t(in.n) + 1, 0);
    for (uint32_t j = 0; j < in.n; ++j) {
        uint64_t k;
        if (in.indptr) { if (in.indptr[j + 1] < in.indptr[j]) throw Error(Errc::BadArgument, "indptr not monotone"); k = in.indptr[j + 1] - in.indptr[j]; }
        else k = in.counts[j];
        a.ptr[j + 1] = a.ptr[j] + k;
    }
    if (in.indptr && (in.indptr[0] != 0)) throw Error(Errc::BadArgument, "indptr[0] must be 0");
    if (a.ptr[in.n] != in.nnz) throw Error(Errc::BadArgument, "indptr/counts do not sum to nnz");
    // row ids strictly ascending and < m
    for (uint32_t j = 0; j < in.n; ++j) {
        int64_t prev = -1;
        for (uint64_t t = a.ptr[j]; t < a.ptr[j + 1]; ++t) {
            int64_t id = in.indices[t];
            if (id <= prev || id >= int64_t(in.m)) throw Error(Errc::BadArgument, "row ids must be strictly ascending and < m (column " + std::to_string(j) + ")");
            prev = id;
        }
    }
    // values
    const bool f32_in = in.data_type == DType::F32;
    if (opt.value_type != ValueType::Auto && (opt.value_type == ValueType::F32) != f32_in) throw Error(Errc::BadArgument, "value_type F32 requires F32 data and vice versa");
    if (in.data_type != DType::U8 && in.data_type != DType::U16 && in.data_type != DType::U32 && in.data_type != DType::F32) throw Error(Errc::BadArgument, "data_type must be U8/U16/U32/F32");
    if (!f32_in) {
        a.vm = VMode::Int;
        if (in.data_type == DType::U32) a.val = static_cast<const uint32_t*>(in.data);
        else {
            a.own.resize(size_t(in.nnz));
            for (uint64_t i = 0; i < in.nnz; ++i) a.own[size_t(i)] = load_value(in, i);
            a.val = a.own.data();
        }
        for (uint64_t i = 0; i < in.nnz; ++i) { uint32_t v = a.val[i]; a.maxv = std::max(a.maxv, v); a.has_zero |= (v == 0); }
        ValueType need = a.maxv <= 255 ? ValueType::U8 : a.maxv <= 65535 ? ValueType::U16 : ValueType::U32;
        if (opt.value_type == ValueType::Auto) a.vt = need;
        else {
            if (uint8_t(opt.value_type) < uint8_t(need)) throw Error(Errc::BadArgument, "values do not fit the requested value_type");
            a.vt = opt.value_type;
        }
    } else {
        a.vt = ValueType::F32;
        a.own.resize(size_t(in.nnz));
        bool integral = true;
        for (uint64_t i = 0; i < in.nnz; ++i) {
            uint32_t bits = load_value(in, i), v;
            a.own[size_t(i)] = bits;
            if (integral && !f32_is_integral(bits, v)) integral = false;
        }
        if (integral) {
            for (uint64_t i = 0; i < in.nnz; ++i) { uint32_t v = 0; f32_is_integral(a.own[size_t(i)], v); a.own[size_t(i)] = v; a.maxv = std::max(a.maxv, v); a.has_zero |= (v == 0); }
            a.vm = VMode::F32Int;
        } else {
            a.vm = VMode::F32Raw;   // coded values are bit patterns; zeros/negative numbers are fine, codec 2 only
        }
        a.val = a.own.data();
    }
    return a;
}

struct Built {
    CodecId codec = CodecId::Rans;
    std::vector<uint8_t> tables, rows, extra_unused;
    std::vector<std::vector<uint8_t>> blobs;
    std::vector<uint32_t> first_col;
    std::vector<uint64_t> chunk_nnz;
    uint8_t kind = 0;
};

inline Built build_codec2(const CSC& in, const Analysis& a, uint32_t chunk_cols, const EncodeOptions& opt) {
    Built b;
    b.codec = CodecId::Generic;
    b.first_col = r1::plan_chunks(a.ptr, in.n, chunk_cols, r1::kMaxChunkN / 2);
    const size_t nch = b.first_col.size() - 1;
    b.blobs.resize(nch);
    b.chunk_nnz.resize(nch);
    parallel_for(resolve_threads(opt.threads, 8), nch, [&](size_t c, int) {
        c2::encode_chunk(in, a.ptr, a.val, a.vm, b.first_col[c], b.first_col[c + 1], opt.zstd_level, b.blobs[c]);
        b.chunk_nnz[c] = a.ptr[b.first_col[c + 1]] - a.ptr[b.first_col[c]];
    });
    return b;
}

inline Built build_codec1(const CSC& in, const Analysis& a, uint32_t chunk_cols, const EncodeOptions& opt) {
    r1::Codec1Params p;
    p.lanes = opt.lanes; p.pb = opt.prob_bits; p.pbv = opt.value_prob_bits; p.kind = opt.ctx_kind; p.cpo = opt.classes_per_octave;
    p.zlevel = opt.zstd_level; p.threads = resolve_threads(opt.threads, 8);
    r1::Codec1Out o = r1::encode_codec1(in, a.ptr, a.val, chunk_cols, p);
    Built b;
    b.codec = CodecId::Rans;
    b.tables = std::move(o.tables); b.rows = std::move(o.rows); b.blobs = std::move(o.blobs);
    b.first_col = std::move(o.first_col); b.chunk_nnz = std::move(o.chunk_nnz); b.kind = o.kind;
    return b;
}

inline std::vector<uint8_t> assemble(const CSC& in, const Analysis& a, const Built& b, const EncodeOptions& opt, uint32_t chunk_cols, const Digest16& digest,
                                     EncodeStats* st) {
    Header h;
    h.codec = b.codec;
    h.m = in.m; h.n = in.n; h.nnz = in.nnz;
    h.value_type = a.vt;
    h.flags = (a.vm == VMode::F32Int) ? 1 : 0;
    h.row_order = b.codec == CodecId::Rans ? 1 : 0;
    h.classes_per_octave = opt.classes_per_octave;
    h.chunk_cols = chunk_cols;
    h.n_chunks = uint32_t(b.blobs.size());
    h.lanes = uint8_t(opt.lanes); h.prob_bits = uint8_t(opt.prob_bits); h.value_prob_bits = uint8_t(opt.value_prob_bits); h.ctx_kind = b.kind;
    h.digest = digest;
    h.rows_kind = opt.row_names.kind; h.cols_kind = opt.col_names.kind;
    if (h.rows_kind == NamesKind::Pack || h.rows_kind == NamesKind::PackSubset) h.row_names_id = opt.row_names.pack_id;
    if (h.cols_kind == NamesKind::Pack || h.cols_kind == NamesKind::PackSubset) h.col_names_id = opt.col_names.pack_id;
    if (h.cols_kind == NamesKind::Cells) h.col_names_id = cells_pack_id();

    // side sections
    std::vector<uint8_t> cols = encode_names_part(opt.col_names, in.n, opt.zstd_level, true);
    std::vector<uint8_t> extra = encode_names_part(opt.row_names, in.m, opt.zstd_level, false);
    std::vector<uint8_t> kvs;
    if (!opt.kv.empty()) {
        std::vector<uint8_t> blob;
        for (const auto& kvp : opt.kv) {
            blob.insert(blob.end(), kvp.first.begin(), kvp.first.end()); blob.push_back(0);
            blob.insert(blob.end(), kvp.second.begin(), kvp.second.end()); blob.push_back(0);
        }
        put_u32(kvs, uint32_t(opt.kv.size()));
        put_stream(kvs, blob.data(), blob.size(), opt.zstd_level);
    }
    // chunk index
    std::vector<ChunkRec> rec(b.blobs.size() + 1);
    uint64_t off = 0, nz = 0;
    for (size_t c = 0; c < b.blobs.size(); ++c) {
        if (b.blobs[c].size() > 0xFFFFFFFFull) throw Error(Errc::Unsupported, "a chunk stream would exceed 4 GiB; use a smaller chunk_cols");
        rec[c].off = off; rec[c].len = uint32_t(b.blobs[c].size()); rec[c].first_col = b.first_col[c]; rec[c].nnz_before = nz;
        off += b.blobs[c].size(); nz += b.chunk_nnz[c];
    }
    rec.back().off = off; rec.back().len = 0; rec.back().first_col = in.n; rec.back().nnz_before = nz;
    std::vector<uint8_t> ci;
    write_chunk_index(ci, rec);

    uint64_t pos = kHeaderBytes;
    auto place = [&](int s, uint64_t len) { h.section[s].off = pos; h.section[s].len = len; pos += len; };
    place(kChunkIndex, ci.size());
    place(kTables, b.tables.size());
    place(kRows, b.rows.size());
    place(kChunks, off);
    place(kCols, cols.size());
    place(kKv, kvs.size());
    place(kExtra, extra.size());
    h.total_len = pos;

    std::vector<uint8_t> out;
    out.reserve(size_t(pos));
    auto hb = write_header(h);
    out.insert(out.end(), hb.begin(), hb.end());
    out.insert(out.end(), ci.begin(), ci.end());
    out.insert(out.end(), b.tables.begin(), b.tables.end());
    out.insert(out.end(), b.rows.begin(), b.rows.end());
    for (const auto& bl : b.blobs) out.insert(out.end(), bl.begin(), bl.end());
    out.insert(out.end(), cols.begin(), cols.end());
    out.insert(out.end(), kvs.begin(), kvs.end());
    out.insert(out.end(), extra.begin(), extra.end());
    if (st) {
        st->codec = b.codec; st->value_type = a.vt; st->chunk_cols = chunk_cols; st->n_chunks = h.n_chunks;
        st->total = out.size(); st->header = kHeaderBytes; st->chunk_index = ci.size(); st->tables = b.tables.size(); st->rows = b.rows.size();
        st->chunks = off; st->cols = cols.size(); st->kv = kvs.size(); st->extra = extra.size();
    }
    return out;
}

}  // namespace detail

inline std::vector<uint8_t> encode_matrix(const CSC& in, const EncodeOptions& opt0, EncodeStats* stats) {
    using namespace detail;
    EncodeOptions opt = opt0;
    if (opt.lanes != 16 && opt.lanes != 32 && opt.lanes != 64) throw Error(Errc::BadArgument, "lanes must be 16, 32 or 64");
    if (opt.prob_bits < 8 || opt.prob_bits > 12 || opt.value_prob_bits < 8 || opt.value_prob_bits > 12) throw Error(Errc::BadArgument, "prob bits must be 8..12");
    if (opt.classes_per_octave > 8) throw Error(Errc::BadArgument, "classes_per_octave must be 0..8");
    if (opt.ctx_kind > 4) throw Error(Errc::BadArgument, "ctx_kind must be -1..4");
    if (opt.col_names.kind == NamesKind::Inline && opt.col_names.blob.empty() && in.n) throw Error(Errc::BadArgument, "inline column names: blob required");
    if (opt.row_names.kind == NamesKind::Inline && opt.row_names.blob.empty() && in.m) throw Error(Errc::BadArgument, "inline row names: blob required");
    Analysis a = analyse(in, opt);
    const uint32_t chunk_cols = opt.chunk_cols ? opt.chunk_cols : default_chunk_cols(in.n, in.nnz, opt.chunk_nnz_target);

    // digest
    Digest16 dg;
    if (opt.has_digest) dg = opt.digest;
    else if (opt.compute_digest) {
        const bool rows_ok = !opt.row_names.blob.empty() || in.m == 0 || opt.row_names.kind == NamesKind::None;
        const bool cols_ok = !opt.col_names.blob.empty() || in.n == 0 || opt.col_names.kind == NamesKind::None;
        if (!rows_ok || !cols_ok) throw Error(Errc::BadArgument, "canonical digest needs the logical row and column names (NamesSpec::blob), or pass has_digest / compute_digest=false");
        dg = canonical_digest(in, a.vt, opt.row_names.blob, opt.col_names.blob, opt.kv);
    }

    const bool rans_ok = in.m <= kRansMaxRows && a.vm != VMode::F32Raw;
    if (opt.codec == CodecId::Rans && !rans_ok) throw Error(Errc::Unsupported, "codec 1 cannot represent this matrix (m >= 2^21 or non-integral f32 values)");
    std::vector<uint8_t> best;
    EncodeStats st1;
    uint64_t sz1 = 0, sz2 = 0;
    if (opt.codec != CodecId::Generic && rans_ok) {
        Built b1 = build_codec1(in, a, chunk_cols, opt);
        best = assemble(in, a, b1, opt, chunk_cols, dg, &st1);
        sz1 = best.size();
    }
    if (opt.codec == CodecId::Generic || !rans_ok || (opt.codec == CodecId::Auto && best.size() < 65536)) {
        Built b2 = build_codec2(in, a, chunk_cols, opt);
        EncodeStats st2;
        std::vector<uint8_t> o2 = assemble(in, a, b2, opt, chunk_cols, dg, &st2);
        sz2 = o2.size();
        if (best.empty() || o2.size() < best.size()) { best = std::move(o2); st1 = st2; }
    }
    if (stats) { *stats = st1; stats->codec1_bytes = sz1; stats->codec2_bytes = sz2; }
    return best;
}

}  // namespace pz2
}  // namespace singlet
