// SPDX-License-Identifier: MIT
// singlet R package -- Rcpp binding for the store matrix codec (inst/include/singlet/pz2/) and the small store side formats a client
// has to read: sample cells tables ("CEL2"), vocabulary packs ("SPK2"/"SVP0") and barcode whitelists.
//
// Design: the R code does the I/O (HTTP ranges, files); this file only turns bytes into R objects.
//   * pz2_header_ / pz2_plan_ / pz2_chunk_first_cols_ read the member prefix (header, chunk index, tables, row classes).
//   * pz2_decode_ decodes a column range straight into the slots of a dgCMatrix (0-based row ids `i`, column pointers `p`,
//     double values `x`): mode "gene" aggregates features to genes through a canonical-row -> gene map (the rows of a
//     stored-order member are mapped once, no string work), mode "canonical" returns the legacy feature matrix. A set of
//     columns can be kept (the called cells); only chunks that cover the range are decoded; threads decode chunks in parallel.
//   * pz2_cells_ decodes the columns of a cells table; pz2_pack_* decode vocabulary packs; pz2_sha256_ hashes a raw vector.
//
// Format: docs/format-store.md.  Codec: bench/compression/notes/m_w1.md.

#include <Rcpp.h>

#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "singlet/pz2/pz2.h"
#include "singlet/pz2_client.h"

namespace pz = singlet::pz2;
namespace pc = singlet::pz2c;

namespace {

std::string hex_bytes(const uint8_t* p, size_t n) {
    static const char* d = "0123456789abcdef";
    std::string s(2 * n, '0');
    for (size_t i = 0; i < n; ++i) {
        s[2 * i] = d[p[i] >> 4];
        s[2 * i + 1] = d[p[i] & 15];
    }
    return s;
}

const char* kind_name(pz::NamesKind k) {
    switch (k) {
        case pz::NamesKind::Inline: return "inline";
        case pz::NamesKind::Pack: return "pack";
        case pz::NamesKind::PackSubset: return "pack_subset";
        case pz::NamesKind::Cells: return "cells";
        case pz::NamesKind::None: return "none";
    }
    return "inline";
}

const char* vt_name(pz::ValueType v) {
    switch (v) {
        case pz::ValueType::U8: return "u8";
        case pz::ValueType::U16: return "u16";
        case pz::ValueType::U32: return "u32";
        case pz::ValueType::F32: return "f32";
        default: return "auto";
    }
}

pz::Isa parse_isa(const std::string& s) {
    if (s.empty() || s == "auto") return pz::Isa::Auto;
    if (s == "scalar") return pz::Isa::Scalar;
    if (s == "avx2") return pz::Isa::Avx2;
    if (s == "avx512") return pz::Isa::Avx512;
    Rcpp::stop("isa must be 'auto', 'scalar', 'avx2' or 'avx512'");
    return pz::Isa::Auto;
}

Rcpp::List header_list(const pz::Header& h) {
    return Rcpp::List::create(
        Rcpp::Named("m") = static_cast<double>(h.m), Rcpp::Named("n") = static_cast<double>(h.n),
        Rcpp::Named("nnz") = static_cast<double>(h.nnz), Rcpp::Named("codec") = static_cast<int>(h.codec),
        Rcpp::Named("value_type") = std::string(vt_name(h.value_type)), Rcpp::Named("row_order") = static_cast<int>(h.row_order),
        Rcpp::Named("chunk_cols") = static_cast<double>(h.chunk_cols), Rcpp::Named("n_chunks") = static_cast<double>(h.n_chunks),
        Rcpp::Named("row_names_kind") = std::string(kind_name(h.rows_kind)), Rcpp::Named("row_names_id") = hex_bytes(h.row_names_id.data(), 8),
        Rcpp::Named("col_names_kind") = std::string(kind_name(h.cols_kind)), Rcpp::Named("col_names_id") = hex_bytes(h.col_names_id.data(), 8),
        Rcpp::Named("digest") = h.digest.hex(), Rcpp::Named("total_len") = static_cast<double>(h.total_len),
        Rcpp::Named("prefix_len") = static_cast<double>(h.prefix_len()),
        Rcpp::Named("chunks_off") = static_cast<double>(h.section[pz::kChunks].off),
        Rcpp::Named("chunks_len") = static_cast<double>(h.section[pz::kChunks].len),
        Rcpp::Named("extra_off") = static_cast<double>(h.section[pz::kExtra].off),
        Rcpp::Named("extra_len") = static_cast<double>(h.section[pz::kExtra].len));
}

// bytes of an R raw vector, as a borrowed pointer
inline const uint8_t* rawp(const Rcpp::RawVector& v) { return reinterpret_cast<const uint8_t*>(RAW(v)); }

// A member assembled from the prefix plus any number of fetched ranges ((offset, raw) pairs).
struct Assembled {
    pz::ByteSource src;
    pz::Matrix mx;
};

Assembled assemble(const Rcpp::RawVector& prefix, const Rcpp::NumericVector& offs, const Rcpp::List& bufs) {
    Assembled a;
    a.src.add(0, rawp(prefix), static_cast<size_t>(prefix.size()));
    if (offs.size() != bufs.size()) Rcpp::stop("ranges: offsets and buffers differ in length");
    for (R_xlen_t i = 0; i < offs.size(); ++i) {
        Rcpp::RawVector b = Rcpp::as<Rcpp::RawVector>(bufs[i]);
        a.src.add(static_cast<uint64_t>(offs[i]), rawp(b), static_cast<size_t>(b.size()));
    }
    try {
        a.mx = pz::Matrix::open(a.src);
    } catch (const pz::Error& e) {
        if (e.code == pz::Errc::NeedRange) Rcpp::stop("pz2: the member prefix is incomplete: need %f bytes at offset %f", static_cast<double>(e.need_len), static_cast<double>(e.need_off));
        Rcpp::stop("%s", e.what());
    }
    return a;
}

// ---- LEB128 / streams / frames (pack and cells decoding) -------------------------------------------------------------------
struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    template <class T>
    T get() {
        if (end - p < static_cast<ptrdiff_t>(sizeof(T))) Rcpp::stop("store: truncated data");
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
};

std::vector<uint8_t> zstd_frame(const uint8_t* p, size_t n) {
    unsigned long long sz = ZSTD_getFrameContentSize(p, n);
    if (sz == ZSTD_CONTENTSIZE_ERROR) Rcpp::stop("store: not a zstd frame");
    if (sz == ZSTD_CONTENTSIZE_UNKNOWN) Rcpp::stop("store: zstd frame without content size");
    if (sz > (1ull << 32)) Rcpp::stop("store: zstd frame too large");
    std::vector<uint8_t> out(static_cast<size_t>(sz));
    size_t r = sz ? ZSTD_decompress(out.data(), out.size(), p, n) : 0;
    if (ZSTD_isError(r) || r != sz) Rcpp::stop("store: zstd decompression failed");
    return out;
}

std::vector<uint8_t> zstd_take(Cur& c, uint32_t clen) {
    if (static_cast<size_t>(c.end - c.p) < clen) Rcpp::stop("store: truncated frame");
    std::vector<uint8_t> out = zstd_frame(c.p, clen);
    c.p += clen;
    return out;
}

std::vector<uint64_t> leb_all(const std::vector<uint8_t>& raw, size_t n) {
    std::vector<uint64_t> v(n);
    size_t q = 0;
    for (size_t i = 0; i < n; ++i) {
        uint64_t x = 0;
        int s = 0;
        for (;;) {
            if (q >= raw.size() || s > 63) Rcpp::stop("store: bad LEB128 stream");
            uint8_t b = raw[q++];
            x |= static_cast<uint64_t>(b & 127) << s;
            if (b < 128) break;
            s += 7;
        }
        v[i] = x;
    }
    return v;
}

// u8 mode (0 LEB128, 1 four byte planes), u32 n, u32 clen, zstd frame
std::vector<uint64_t> pack_stream(Cur& c) {
    uint8_t mode = c.get<uint8_t>();
    uint32_t n = c.get<uint32_t>();
    uint32_t cl = c.get<uint32_t>();
    std::vector<uint8_t> raw = zstd_take(c, cl);
    if (mode == 0) return leb_all(raw, n);
    if (raw.size() != static_cast<size_t>(n) * 4) Rcpp::stop("store: bad plane stream");
    std::vector<uint64_t> v(n);
    for (uint32_t i = 0; i < n; ++i)
        v[i] = raw[i] | (static_cast<uint64_t>(raw[n + i]) << 8) | (static_cast<uint64_t>(raw[2 * static_cast<size_t>(n) + i]) << 16) |
               (static_cast<uint64_t>(raw[3 * static_cast<size_t>(n) + i]) << 24);
    return v;
}

std::vector<uint8_t> pack_blob(Cur& c) {  // u32 clen + zstd frame
    uint32_t cl = c.get<uint32_t>();
    return zstd_take(c, cl);
}

// terminated = true: every string is followed by a NUL ("a\0b\0"); false: strings are separated by NULs ("a\0b", empty blob = no strings)
std::vector<std::string> split_nul(const std::vector<uint8_t>& b, bool terminated) {
    std::vector<std::string> out;
    size_t s = 0;
    for (size_t i = 0; i < b.size(); ++i)
        if (b[i] == 0) {
            out.emplace_back(reinterpret_cast<const char*>(b.data()) + s, i - s);
            s = i + 1;
        }
    if (!terminated && !b.empty()) out.emplace_back(reinterpret_cast<const char*>(b.data()) + s, b.size() - s);
    return out;
}

// SPK2 wrapper: "SPK2" u16 version u16 kind u32 header_len header body
struct PackFile {
    uint16_t kind = 0;
    std::string header;
    const uint8_t* body = nullptr;
    size_t body_len = 0;
};

PackFile parse_pack(const Rcpp::RawVector& raw) {
    const uint8_t* p = rawp(raw);
    size_t n = static_cast<size_t>(raw.size());
    if (n < 12 || std::memcmp(p, "SPK2", 4) != 0) Rcpp::stop("not an SPK2 pack");
    uint16_t ver, kind;
    uint32_t hl;
    std::memcpy(&ver, p + 4, 2);
    std::memcpy(&kind, p + 6, 2);
    std::memcpy(&hl, p + 8, 4);
    if (ver != 1) Rcpp::stop("unsupported pack version %d", static_cast<int>(ver));
    if (12 + static_cast<size_t>(hl) > n) Rcpp::stop("truncated pack header");
    PackFile f;
    f.kind = kind;
    f.header.assign(reinterpret_cast<const char*>(p) + 12, hl);
    f.body = p + 12 + hl;
    f.body_len = n - 12 - hl;
    return f;
}

std::string codes_to_barcode(uint64_t code, int k) {
    static const char alpha[4] = {'A', 'C', 'G', 'T'};
    std::string s(static_cast<size_t>(k), 'A');
    for (int j = 0; j < k; ++j) s[static_cast<size_t>(k - 1 - j)] = alpha[(code >> (2 * j)) & 3];
    return s;
}

uint64_t read_bits(const uint8_t* p, size_t nbytes, uint64_t bit0, int bits) {  // LSB-first, value = sum bit_b << b
    uint64_t v = 0;
    for (int b = 0; b < bits; ++b) {
        uint64_t j = bit0 + static_cast<uint64_t>(b);
        if ((j >> 3) >= nbytes) Rcpp::stop("store: bit-packed column is truncated");
        v |= static_cast<uint64_t>((p[j >> 3] >> (j & 7)) & 1) << b;
    }
    return v;
}

struct Whitelist {
    int k = 0;
    std::vector<uint64_t> codes;
};

Whitelist parse_whitelist(const Rcpp::RawVector& raw) {
    PackFile f = parse_pack(raw);
    if (f.kind != 2) Rcpp::stop("not a barcode whitelist pack");
    Cur c{f.body, f.body + f.body_len};
    Whitelist w;
    w.k = c.get<uint8_t>();
    uint32_t n = c.get<uint32_t>();
    uint8_t mode = c.get<uint8_t>();
    if (mode != 0) Rcpp::stop("unknown whitelist mode");
    std::vector<uint8_t> raw_gaps = zstd_frame(c.p, static_cast<size_t>(c.end - c.p));
    std::vector<uint64_t> gaps = leb_all(raw_gaps, n);
    w.codes.resize(n);
    uint64_t acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
        acc += gaps[i];
        w.codes[i] = acc;
    }
    return w;
}

}  // namespace

// ================================================================================================================================
// matrix members
// ================================================================================================================================

//' Parse the 256-byte header of a pz2 member.
//' @param buf Raw vector with at least the first 256 bytes of the member.
//' @return A list: m, n, nnz, codec, value_type, names kinds and pack ids, digest, total_len, prefix_len, section offsets.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_header_(Rcpp::RawVector buf) {
    try {
        return header_list(pz::read_header(rawp(buf), static_cast<size_t>(buf.size())));
    } catch (const pz::Error& e) {
        Rcpp::stop("%s", e.what());
    }
    return Rcpp::List();
}

//' First column of every chunk (n_chunks + 1 entries, the last is n) of a member, from its prefix.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::NumericVector pz2_chunk_first_cols_(Rcpp::RawVector prefix) {
    Assembled a = assemble(prefix, Rcpp::NumericVector(), Rcpp::List());
    const auto& ix = a.mx.chunk_index();
    Rcpp::NumericVector out(static_cast<R_xlen_t>(ix.rec.size()));
    for (size_t i = 0; i < ix.rec.size(); ++i) out[static_cast<R_xlen_t>(i)] = ix.rec[i].first_col;
    return out;
}

//' Member byte range that holds the chunks covering columns [c0, c1) (0-based, c1 exclusive).
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_plan_(Rcpp::RawVector prefix, double c0, double c1) {
    Assembled a = assemble(prefix, Rcpp::NumericVector(), Rcpp::List());
    try {
        pz::ColumnPlan p = a.mx.plan_columns(static_cast<uint32_t>(c0), static_cast<uint32_t>(c1));
        return Rcpp::List::create(Rcpp::Named("byte_off") = static_cast<double>(p.byte_off), Rcpp::Named("byte_len") = static_cast<double>(p.byte_len),
                                  Rcpp::Named("max_nnz") = static_cast<double>(p.max_nnz), Rcpp::Named("chunk0") = static_cast<double>(p.chunk0),
                                  Rcpp::Named("chunk1") = static_cast<double>(p.chunk1), Rcpp::Named("first_col") = static_cast<double>(p.first_col));
    } catch (const pz::Error& e) {
        Rcpp::stop("%s", e.what());
    }
    return Rcpp::List();
}

//' Decode columns [c0, c1) of a member into the slots of a dgCMatrix.
//'
//' @param prefix Raw vector: the member prefix (header, chunk index, tables, rows) -- or the whole member.
//' @param offs,bufs Fetched member ranges: offsets (member-relative, numeric) and the matching raw vectors.
//' @param c0,c1 Column range, 0-based, c1 exclusive.
//' @param mode "gene" (rows summed into genes through `gene_map`) or "canonical" (legacy feature rows).
//' @param cols Optional integer vector of columns to keep, 0-based relative to `c0`, strictly ascending; NULL keeps all.
//' @param gene_map Integer vector over the canonical rows of the member: 0-based gene, or -1 to drop the row ("gene" mode).
//' @param n_genes Number of genes of the target axis.
//' @param threads Decoder threads (0 = up to 4).
//' @param isa "auto", "scalar", "avx2" or "avx512".
//' @return list(i, p, x, nrow, ncol): 0-based row ids, column pointers, double values.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_decode_(Rcpp::RawVector prefix, Rcpp::NumericVector offs, Rcpp::List bufs, double c0, double c1, std::string mode,
                       SEXP cols, SEXP gene_map, int n_genes, int threads, std::string isa) {
    Assembled a = assemble(prefix, offs, bufs);
    const pz::Header& h = a.mx.header();
    const uint32_t col0 = static_cast<uint32_t>(c0), col1 = static_cast<uint32_t>(c1);
    if (col0 > col1 || col1 > h.n) Rcpp::stop("column range out of bounds");
    const uint32_t ncol = col1 - col0;
    const bool gene = (mode == "gene");
    if (!gene && mode != "canonical") Rcpp::stop("mode must be 'gene' or 'canonical'");

    std::vector<uint32_t> keep;
    if (!Rf_isNull(cols)) {
        Rcpp::IntegerVector cv(cols);
        keep.resize(static_cast<size_t>(cv.size()));
        for (R_xlen_t i = 0; i < cv.size(); ++i) {
            if (cv[i] < 0) Rcpp::stop("cols must be non-negative");
            keep[static_cast<size_t>(i)] = static_cast<uint32_t>(cv[i]);
        }
    }
    std::vector<int32_t> smap;
    if (gene) {
        if (Rf_isNull(gene_map)) Rcpp::stop("mode 'gene' needs gene_map");
        if (n_genes <= 0) Rcpp::stop("mode 'gene' needs n_genes > 0");
        Rcpp::IntegerVector gm(gene_map);
        const auto& perm = a.mx.row_perm();
        try {
            smap = pc::compose_gene_map(INTEGER(gm), static_cast<size_t>(gm.size()), perm.empty() ? nullptr : perm.data(), h.m);
        } catch (const std::exception& e) {
            Rcpp::stop("%s", e.what());
        }
    }

    const bool whole = (col0 == 0 && col1 == h.n);
    pz::ColumnPlan plan;
    uint64_t bound = h.nnz;
    try {
        if (!whole) {
            plan = a.mx.plan_columns(col0, col1);
            bound = plan.max_nnz;
        }
    } catch (const pz::Error& e) {
        Rcpp::stop("%s", e.what());
    }
    if (bound >= (1ull << 31)) Rcpp::stop("too many non-zeros for one R sparse matrix (%f); read fewer columns", static_cast<double>(bound));

    std::vector<int32_t> ptr(static_cast<size_t>(ncol) + 1), idx(static_cast<size_t>(bound));
    std::vector<double> val(static_cast<size_t>(bound));
    pz::Output out;
    out.indptr = ptr.data();
    out.indptr_type = pz::IntType::I32;
    out.indptr_capacity = ptr.size();
    out.indices = idx.data();
    out.indices_type = pz::IntType::I32;
    out.indices_capacity = idx.size();
    out.data = val.data();
    out.data_type = pz::DType::F64;
    out.data_capacity = val.size();
    pz::DecodeOptions opt;
    opt.mode = gene ? pz::Mode::Native : pz::Mode::Canonical;
    opt.threads = threads;
    opt.isa = parse_isa(isa);

    uint64_t nnz = 0;
    try {
        if (whole) {
            a.mx.decode(a.src, out, opt);
            nnz = h.nnz;
        } else {
            nnz = a.mx.decode_columns(a.src, col0, col1, out, opt);
        }
    } catch (const pz::Error& e) {
        if (e.code == pz::Errc::NeedRange) Rcpp::stop("pz2: byte range not available: need %f bytes at member offset %f", static_cast<double>(e.need_len), static_cast<double>(e.need_off));
        Rcpp::stop("%s", e.what());
    }

    size_t ncol_out = ncol;
    std::vector<int32_t> sel_ptr;
    if (!keep.empty() || !Rf_isNull(cols)) {
        sel_ptr.resize(keep.size() + 1);
        try {
            nnz = static_cast<uint64_t>(pc::select_columns_inplace<int32_t>(ptr.data(), ncol, keep.data(), keep.size(), idx.data(), val.data(), sel_ptr.data()));
        } catch (const std::exception& e) {
            Rcpp::stop("%s", e.what());
        }
        ptr.swap(sel_ptr);
        ncol_out = keep.size();
    }
    if (gene) {
        try {
            nnz = static_cast<uint64_t>(pc::aggregate_genes<int32_t, double>(ptr.data(), ncol_out, idx.data(), val.data(), smap.data(), n_genes, threads));
        } catch (const std::exception& e) {
            Rcpp::stop("%s", e.what());
        }
    }
    Rcpp::IntegerVector p(static_cast<R_xlen_t>(ncol_out) + 1), i(static_cast<R_xlen_t>(nnz));
    Rcpp::NumericVector x(static_cast<R_xlen_t>(nnz));
    std::memcpy(INTEGER(p), ptr.data(), (ncol_out + 1) * sizeof(int32_t));
    if (nnz) {
        std::memcpy(INTEGER(i), idx.data(), nnz * sizeof(int32_t));
        std::memcpy(REAL(x), val.data(), nnz * sizeof(double));
    }
    return Rcpp::List::create(Rcpp::Named("i") = i, Rcpp::Named("p") = p, Rcpp::Named("x") = x,
                              Rcpp::Named("nrow") = gene ? n_genes : static_cast<int>(h.m), Rcpp::Named("ncol") = static_cast<int>(ncol_out));
}

//' Inline row or column names of a whole member (raw vector), or the pack subset index.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_names_(Rcpp::RawVector member, bool rows) {
    Assembled a = assemble(member, Rcpp::NumericVector(), Rcpp::List());
    try {
        pz::NamesSpec ns = rows ? a.mx.row_names(a.src) : a.mx.col_names(a.src);
        Rcpp::CharacterVector names;
        if (ns.kind == pz::NamesKind::Inline) {
            std::vector<std::string> v = pz::split_names(ns.blob, static_cast<size_t>(ns.count));
            names = Rcpp::CharacterVector(v.begin(), v.end());
        }
        Rcpp::IntegerVector idx;
        if (ns.kind == pz::NamesKind::PackSubset) {
            idx = Rcpp::IntegerVector(ns.pack_index.size());
            for (size_t k = 0; k < ns.pack_index.size(); ++k) idx[static_cast<R_xlen_t>(k)] = static_cast<int>(ns.pack_index[k]);
        }
        return Rcpp::List::create(Rcpp::Named("kind") = std::string(kind_name(ns.kind)), Rcpp::Named("pack_id") = hex_bytes(ns.pack_id.data(), 8),
                                  Rcpp::Named("names") = names, Rcpp::Named("pack_index") = idx);
    } catch (const pz::Error& e) {
        Rcpp::stop("%s", e.what());
    }
    return Rcpp::List();
}

//' Decompress one zstd frame that records its content size.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::RawVector pz2_zstd_(Rcpp::RawVector x) {
    std::vector<uint8_t> v = zstd_frame(rawp(x), static_cast<size_t>(x.size()));
    Rcpp::RawVector out(static_cast<R_xlen_t>(v.size()));
    if (!v.empty()) std::memcpy(RAW(out), v.data(), v.size());
    return out;
}

//' One line: the instruction-set paths of the codec.
//' @keywords internal
// [[Rcpp::export]]
std::string pz2_isa_report_() { return pz::isa_report(); }

//' SHA-256 of a raw vector, as 64 hex characters.
//' @keywords internal
// [[Rcpp::export]]
std::string pz2_sha256_(Rcpp::RawVector x) {
    pz::detail::Sha256 h;
    h.update(rawp(x), static_cast<size_t>(x.size()));
    uint8_t out[32];
    h.final(out);
    return hex_bytes(out, 32);
}

// ================================================================================================================================
// packs
// ================================================================================================================================

//' Header JSON, kind and (for composite / derived packs) the referenced pack digests of an SPK2 pack.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_pack_info_(Rcpp::RawVector pack) {
    PackFile f = parse_pack(pack);
    Rcpp::List parts;
    std::string derived;
    Cur c{f.body, f.body + f.body_len};
    if (f.kind == 3) {  // composite: u8 n_parts, per part 16 B digest, u8 tag_len, tag
        uint8_t np = c.get<uint8_t>();
        parts = Rcpp::List(np);
        for (int k = 0; k < np; ++k) {
            if (c.end - c.p < 17) Rcpp::stop("truncated composite pack");
            std::string id = hex_bytes(c.p, 16);
            uint8_t tl = c.p[16];
            if (c.end - c.p < 17 + tl) Rcpp::stop("truncated composite pack");
            std::string tag(reinterpret_cast<const char*>(c.p) + 17, tl);
            c.p += 17 + tl;
            parts[k] = Rcpp::List::create(Rcpp::Named("id") = id, Rcpp::Named("tag") = tag);
        }
    } else if (f.kind == 1 && f.body_len == 16) {  // SVPD: the body is the digest of the exon pack
        derived = hex_bytes(f.body, 16);
    }
    return Rcpp::List::create(Rcpp::Named("kind") = static_cast<int>(f.kind), Rcpp::Named("header") = f.header, Rcpp::Named("parts") = parts,
                              Rcpp::Named("derived_from") = derived);
}

//' Gene table of an SVP0 vocabulary pack: gene ids, gene names and the number of exons of every gene.
//'
//' Exon row `r` belongs to gene `rep(seq_along(nex), nex)[r]`, intron row `r` to `rep(seq_along(nex), nex - 1)[r]`: the
//' feature -> gene map of a matrix is this vector, no string work.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_pack_genes_(Rcpp::RawVector pack) {
    PackFile f = parse_pack(pack);
    if (f.kind != 1 || f.body_len < 4 || std::memcmp(f.body, "SVP0", 4) != 0) Rcpp::stop("not an SVP0 exon vocabulary pack");
    Cur c{f.body + 4, f.body + f.body_len};
    c.get<uint16_t>();  // version
    c.get<uint16_t>();  // flags
    uint8_t prefix_len = c.get<uint8_t>();
    uint8_t width = c.get<uint8_t>();
    c.get<uint16_t>();  // n chromosomes
    uint32_t n_genes = c.get<uint32_t>();
    c.get<uint32_t>();  // n exons
    uint8_t bl = c.get<uint8_t>();
    if (c.end - c.p < static_cast<ptrdiff_t>(bl) + prefix_len + 8) Rcpp::stop("truncated SVP0 header");
    std::string build(reinterpret_cast<const char*>(c.p), bl);
    c.p += bl;
    std::string prefix(reinterpret_cast<const char*>(c.p), prefix_len);
    c.p += prefix_len + 8;  // + schema
    std::vector<uint64_t> d = pack_stream(c);
    std::vector<uint8_t> kind = pack_blob(c);
    std::vector<uint8_t> symb = pack_blob(c);
    pack_blob(c);                   // chromosome names
    pack_stream(c);                 // chromosome per gene
    std::vector<uint64_t> nx = pack_stream(c);
    if (d.size() != n_genes || nx.size() != n_genes || kind.size() != n_genes) Rcpp::stop("SVP0 counts do not match the header");
    std::vector<std::string> syms;
    bool any_explicit = std::find(kind.begin(), kind.end(), static_cast<uint8_t>(2)) != kind.end();
    if (any_explicit) syms = split_nul(symb, false);
    Rcpp::CharacterVector ids(n_genes), names(n_genes);
    Rcpp::IntegerVector nex(n_genes);
    uint64_t num = 0;
    size_t sk = 0;
    for (uint32_t g = 0; g < n_genes; ++g) {
        num += (g == 0) ? d[0] : d[g] + 1;
        std::string digits = std::to_string(num);
        if (digits.size() < width) digits.insert(0, width - digits.size(), '0');
        std::string gid = prefix + digits;
        ids[g] = gid;
        if (kind[g] == 2) {
            if (sk >= syms.size()) Rcpp::stop("SVP0 symbol table is short");
            names[g] = syms[sk++];
        } else {
            names[g] = gid;
        }
        nex[g] = static_cast<int>(nx[g] + 1);
    }
    return Rcpp::List::create(Rcpp::Named("build") = build, Rcpp::Named("gene_id") = ids, Rcpp::Named("gene_name") = names, Rcpp::Named("nex") = nex);
}

//' Barcodes of a whitelist pack at the given 0-based indices (all barcodes when `idx` is NULL).
//' @keywords internal
// [[Rcpp::export]]
Rcpp::CharacterVector pz2_whitelist_barcodes_(Rcpp::RawVector pack, SEXP idx) {
    Whitelist w = parse_whitelist(pack);
    if (Rf_isNull(idx)) {
        Rcpp::CharacterVector out(static_cast<R_xlen_t>(w.codes.size()));
        for (size_t k = 0; k < w.codes.size(); ++k) out[static_cast<R_xlen_t>(k)] = codes_to_barcode(w.codes[k], w.k);
        return out;
    }
    Rcpp::IntegerVector iv(idx);
    Rcpp::CharacterVector out(iv.size());
    for (R_xlen_t k = 0; k < iv.size(); ++k) {
        if (iv[k] < 0 || static_cast<size_t>(iv[k]) >= w.codes.size()) Rcpp::stop("whitelist index out of range");
        out[k] = codes_to_barcode(w.codes[static_cast<size_t>(iv[k])], w.k);
    }
    return out;
}

// ================================================================================================================================
// cells tables
// ================================================================================================================================

//' Decode columns of a sample cells table ("CEL2").
//'
//' @param buf Raw vector with the whole cells member.
//' @param columns Names of the columns wanted (character); empty = all.
//' @param whitelist Optional raw vector: the whitelist pack barcodes may be coded against.
//' @return A list with n_rows, the column names present, and one element per wanted column: numeric for numbers, logical for
//'   bitmaps, character for strings and barcodes. A barcode column coded against a whitelist that was not supplied comes back as
//'   list(whitelist = <8-byte pack id, hex>, index = <0-based integer vector>).
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List pz2_cells_(Rcpp::RawVector buf, Rcpp::CharacterVector columns, SEXP whitelist) {
    const uint8_t* base = rawp(buf);
    size_t total = static_cast<size_t>(buf.size());
    if (total < 12 || std::memcmp(base, "CEL2", 4) != 0) Rcpp::stop("not a cells table");
    Cur c{base + 4, base + total};
    uint16_t ver = c.get<uint16_t>();
    uint16_t ncols = c.get<uint16_t>();
    uint32_t nrows = c.get<uint32_t>();
    if (ver != 1) Rcpp::stop("unsupported cells table version %d", static_cast<int>(ver));
    std::vector<std::string> want;
    for (R_xlen_t i = 0; i < columns.size(); ++i) want.push_back(Rcpp::as<std::string>(columns[i]));
    Rcpp::CharacterVector present(ncols);
    Rcpp::List out;
    std::vector<std::string> out_names;
    std::vector<Rcpp::RObject> out_vals;
    bool have_wl = !Rf_isNull(whitelist);
    Whitelist wl;
    if (have_wl) wl = parse_whitelist(Rcpp::RawVector(whitelist));
    for (uint16_t k = 0; k < ncols; ++k) {
        uint16_t nl = c.get<uint16_t>();
        if (c.end - c.p < nl) Rcpp::stop("truncated cells directory");
        std::string name(reinterpret_cast<const char*>(c.p), nl);
        c.p += nl;
        uint8_t dt = c.get<uint8_t>();
        uint8_t enc = c.get<uint8_t>();
        uint64_t off = c.get<uint64_t>();
        uint64_t len = c.get<uint64_t>();
        present[k] = name;
        if (!want.empty() && std::find(want.begin(), want.end(), name) == want.end()) continue;
        if (off + len > total) Rcpp::stop("cells column %s runs past the end of the table", name.c_str());
        const uint8_t* body = base + off;
        std::vector<uint8_t> tmp;
        const uint8_t* raw = body;
        size_t rlen = static_cast<size_t>(len);
        if (dt != 9 && enc != 0 && len) {
            tmp = zstd_frame(body, static_cast<size_t>(len));
            raw = tmp.data();
            rlen = tmp.size();
        }
        Rcpp::RObject val;
        if (dt >= 1 && dt <= 6) {
            const size_t w = (dt == 1) ? 1 : (dt == 2) ? 2 : (dt == 6) ? 8 : 4;
            if (rlen != static_cast<size_t>(nrows) * w) Rcpp::stop("cells column %s has the wrong length", name.c_str());
            Rcpp::NumericVector v(nrows);
            std::vector<double> tmpd;
            for (uint32_t r = 0; r < nrows; ++r) {
                const uint8_t* q = raw + static_cast<size_t>(r) * w;
                double x = 0;
                switch (dt) {
                    case 1: x = q[0]; break;
                    case 2: { uint16_t t; std::memcpy(&t, q, 2); x = t; break; }
                    case 3: { uint32_t t; std::memcpy(&t, q, 4); x = t; break; }
                    case 4: { int32_t t; std::memcpy(&t, q, 4); x = t; break; }
                    case 5: { float t; std::memcpy(&t, q, 4); x = t; break; }
                    case 6: { std::memcpy(&x, q, 8); break; }
                }
                v[r] = x;
            }
            if (enc == 2 && dt <= 4) {  // delta: wrapping first differences in the column's width
                uint64_t acc = 0;
                const uint64_t mask = (dt == 1) ? 0xFFull : (dt == 2) ? 0xFFFFull : 0xFFFFFFFFull;
                for (uint32_t r = 0; r < nrows; ++r) {
                    const uint8_t* q = raw + static_cast<size_t>(r) * w;
                    uint64_t t = 0;
                    std::memcpy(&t, q, w);
                    acc = (acc + t) & mask;
                    v[r] = (dt == 4) ? static_cast<double>(static_cast<int32_t>(static_cast<uint32_t>(acc))) : static_cast<double>(acc);
                }
            }
            val = v;
        } else if (dt == 7) {
            Rcpp::LogicalVector v(nrows);
            if (rlen < (static_cast<size_t>(nrows) + 7) / 8) Rcpp::stop("cells bitmap %s is short", name.c_str());
            for (uint32_t r = 0; r < nrows; ++r) v[r] = (raw[r >> 3] >> (r & 7)) & 1;
            val = v;
        } else if (dt == 8) {
            std::vector<uint8_t> b(raw, raw + rlen);
            std::vector<std::string> s = split_nul(b, true);
            if (s.size() != nrows) Rcpp::stop("cells string column %s has %d values, expected %d", name.c_str(), static_cast<int>(s.size()), static_cast<int>(nrows));
            val = Rcpp::CharacterVector(s.begin(), s.end());
        } else if (dt == 9) {
            if (len < 1) Rcpp::stop("empty barcode column");
            uint8_t mode = body[0];
            if (mode == 0) {
                std::vector<uint8_t> b = nrows ? zstd_frame(body + 1, static_cast<size_t>(len) - 1) : std::vector<uint8_t>();
                std::vector<std::string> s = split_nul(b, true);
                if (s.size() != nrows) Rcpp::stop("barcode column has the wrong count");
                val = Rcpp::CharacterVector(s.begin(), s.end());
            } else if (mode == 1) {
                if (len < 10) Rcpp::stop("truncated whitelist-coded barcode column");
                std::string pid = hex_bytes(body + 1, 8);
                int bits = body[9];
                Rcpp::IntegerVector ix(nrows);
                for (uint32_t r = 0; r < nrows; ++r) ix[r] = static_cast<int>(read_bits(body + 10, static_cast<size_t>(len) - 10, static_cast<uint64_t>(r) * bits, bits));
                if (have_wl) {
                    Rcpp::CharacterVector s(nrows);
                    for (uint32_t r = 0; r < nrows; ++r) {
                        if (static_cast<size_t>(ix[r]) >= wl.codes.size()) Rcpp::stop("whitelist index out of range");
                        s[r] = codes_to_barcode(wl.codes[static_cast<size_t>(ix[r])], wl.k);
                    }
                    val = s;
                } else {
                    val = Rcpp::List::create(Rcpp::Named("whitelist") = pid, Rcpp::Named("index") = ix);
                }
            } else if (mode == 2) {
                int kk = body[1];
                Rcpp::CharacterVector s(nrows);
                for (uint32_t r = 0; r < nrows; ++r)
                    s[r] = codes_to_barcode(read_bits(body + 2, static_cast<size_t>(len) - 2, static_cast<uint64_t>(r) * 2 * kk, 2 * kk), kk);
                val = s;
            } else {
                Rcpp::stop("unknown barcode mode %d", static_cast<int>(mode));
            }
        } else {
            Rcpp::stop("unknown cells dtype %d", static_cast<int>(dt));
        }
        out_names.push_back(name);
        out_vals.push_back(val);
    }
    Rcpp::List cols(out_vals.size());
    for (size_t k = 0; k < out_vals.size(); ++k) cols[k] = out_vals[k];
    cols.names() = Rcpp::wrap(out_names);
    return Rcpp::List::create(Rcpp::Named("n_rows") = static_cast<double>(nrows), Rcpp::Named("names") = present, Rcpp::Named("columns") = cols);
}
