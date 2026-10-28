// SPDX-License-Identifier: MIT
// singlet R package -- write a count matrix as a .s1 sample object, and list the members of one (docs/format-s1.md).
//
// s1_write_ encodes a canonical CSC with the header-only pz2 encoder (inst/include/singlet/pz2/) using exactly the options the
// Python binding uses for singlet.write_s1 (codec "auto", u32 values, default chunking, zstd level 19, the canonical digest
// computed from the matrix and its names), then lays out a sample object with one pz2 member as
// python/singlet/s1/container.py S2Writer.build does. The two packages therefore write the same bytes for the same matrix;
// tests/testthat/test-s1-write.R checks the layout and the Python suite checks byte identity on a shared fixture.
//
// s1_members_ parses the 64-byte header and the member table of an object (bytes of its first part) into an R list.

#include <Rcpp.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "singlet/pz2/pz2.h"

namespace pz = singlet::pz2;

namespace {

constexpr uint64_t kAlign = 64;
constexpr size_t kHeader = 64;
constexpr size_t kEntryFixed = 1 + 1 + 8 + 8 + 8 + 4 + 16 + 4;  // kind flags off len out_len crc32 digest meta_len

uint64_t align_up(uint64_t x) { return (x + kAlign - 1) / kAlign * kAlign; }

template <typename T>
void put_le(std::vector<uint8_t>& o, T v) {
    for (size_t k = 0; k < sizeof(T); ++k) o.push_back(static_cast<uint8_t>((static_cast<uint64_t>(v) >> (8 * k)) & 0xFF));
}

template <typename T>
T get_le(const uint8_t* p) {
    uint64_t v = 0;
    for (size_t k = 0; k < sizeof(T); ++k) v |= static_cast<uint64_t>(p[k]) << (8 * k);
    return static_cast<T>(v);
}

std::vector<std::string> as_strings(const Rcpp::CharacterVector& v) {
    std::vector<std::string> out(static_cast<size_t>(v.size()));
    for (R_xlen_t k = 0; k < v.size(); ++k) out[static_cast<size_t>(k)] = Rcpp::as<std::string>(v[k]);
    return out;
}

pz::PackId parse_pack(const std::string& hx) {
    if (hx.size() != 16) Rcpp::stop("gene_pack must be 16 hex characters");
    pz::PackId id{};
    for (int k = 0; k < 8; ++k) id[static_cast<size_t>(k)] = static_cast<uint8_t>(std::stoi(hx.substr(2 * k, 2), nullptr, 16));
    return id;
}

}  // namespace

//' Encode a canonical CSC (genes x cells) as a .s1 sample object with one pz2 member; returns the object bytes.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::RawVector s1_write_(Rcpp::IntegerVector p, Rcpp::IntegerVector i, Rcpp::NumericVector x, int m, int n, Rcpp::CharacterVector rownames,
                          Rcpp::CharacterVector colnames, std::string sample_id, std::string member, std::string gene_pack, int threads) {
    if (m < 0 || n < 0) Rcpp::stop("negative dimensions");
    if (p.size() != static_cast<R_xlen_t>(n) + 1) Rcpp::stop("p must have ncol + 1 entries");
    if (rownames.size() != m || colnames.size() != n) Rcpp::stop("the matrix needs a name for every row and column");
    if (sample_id.size() > 16) Rcpp::stop("sample_id longer than 16 bytes");
    for (unsigned char c : sample_id)
        if (c > 127) Rcpp::stop("sample_id must be ASCII");
    const uint64_t nnz = static_cast<uint64_t>(p[n]);
    if (static_cast<uint64_t>(i.size()) != nnz || static_cast<uint64_t>(x.size()) != nnz) Rcpp::stop("i and x must have p[ncol] entries");

    std::vector<uint32_t> counts(static_cast<size_t>(n)), idx(static_cast<size_t>(nnz)), val(static_cast<size_t>(nnz));
    if (n && p[0] != 0) Rcpp::stop("p[1] must be 0");
    for (int j = 0; j < n; ++j) {
        if (p[j + 1] < p[j]) Rcpp::stop("p must be non-decreasing");
        counts[static_cast<size_t>(j)] = static_cast<uint32_t>(p[j + 1] - p[j]);
        for (int k = p[j]; k < p[j + 1]; ++k) {
            if (i[k] < 0 || i[k] >= m) Rcpp::stop("row index out of range");
            if (k > p[j] && i[k] <= i[k - 1]) Rcpp::stop("row indices must be strictly increasing within each column");
            const double v = x[k];
            if (!(v >= 1) || v != static_cast<double>(static_cast<uint64_t>(v)) || v > 4294967295.0)
                Rcpp::stop("values must be integers in [1, 2^32 - 1] (drop explicit zeros first)");
            idx[static_cast<size_t>(k)] = static_cast<uint32_t>(i[k]);
            val[static_cast<size_t>(k)] = static_cast<uint32_t>(v);
        }
    }

    pz::CSC csc;
    csc.m = static_cast<uint32_t>(m);
    csc.n = static_cast<uint32_t>(n);
    csc.nnz = nnz;
    csc.indptr = nullptr;
    csc.counts = counts.data();
    csc.indices = idx.data();
    csc.data = val.data();
    csc.data_type = pz::DType::U32;

    const std::vector<uint8_t> row_blob = pz::join_names(as_strings(rownames)), col_blob = pz::join_names(as_strings(colnames));
    pz::EncodeOptions opt;  // the defaults of singlet._pz2.encode: codec auto, 32 lanes, 10-bit tables, zstd 19
    opt.codec = pz::CodecId::Auto;
    opt.value_type = pz::ValueType::U32;
    opt.row_names.count = static_cast<uint64_t>(m);
    opt.col_names.count = static_cast<uint64_t>(n);
    if (gene_pack.empty()) {
        opt.row_names.kind = pz::NamesKind::Inline;
        opt.row_names.blob = row_blob;
    } else {
        opt.row_names.kind = pz::NamesKind::Pack;
        opt.row_names.pack_id = parse_pack(gene_pack);
    }
    opt.col_names.kind = pz::NamesKind::Inline;
    opt.col_names.blob = col_blob;
    pz::Digest16 digest;
    std::vector<uint8_t> body;
    try {
        digest = pz::canonical_digest(csc, pz::ValueType::U32, row_blob, col_blob, pz::KvMap());
        opt.digest = digest;
        opt.has_digest = true;
        opt.compute_digest = false;
        opt.threads = threads;
        opt.zstd_level = 19;
        body = pz::encode_matrix(csc, opt, nullptr);
    } catch (const pz::Error& e) {
        Rcpp::stop("%s", e.what());
    }

    // one-member sample object, as container.py S2Writer.build lays it out
    const std::string meta = "{\"modality\":\"counts\"}";  // canonical JSON (sorted keys, compact)
    if (member.empty() || member.size() > 0xFFFF) Rcpp::stop("bad member name");
    const uint64_t table_len = 2 + member.size() + kEntryFixed + meta.size();
    const uint64_t body_off = body.empty() ? 0 : align_up(kHeader + table_len);
    const uint64_t file_len = body.empty() ? kHeader + table_len : body_off + body.size();
    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(file_len));
    const char magic[] = "SINGLET2";
    out.insert(out.end(), magic, magic + 8);
    put_le<uint16_t>(out, 1);  // version
    put_le<uint16_t>(out, 0);  // flags
    put_le<uint32_t>(out, 1);  // n_members
    put_le<uint64_t>(out, kHeader);
    put_le<uint64_t>(out, table_len);
    put_le<uint64_t>(out, file_len);
    for (size_t k = 0; k < 16; ++k) out.push_back(k < sample_id.size() ? static_cast<uint8_t>(sample_id[k]) : 0);
    for (int k = 0; k < 8; ++k) out.push_back(0);
    put_le<uint16_t>(out, static_cast<uint16_t>(member.size()));
    out.insert(out.end(), member.begin(), member.end());
    out.push_back(1);  // kind: pz2 matrix
    out.push_back(0);  // flags
    put_le<uint64_t>(out, body_off);
    put_le<uint64_t>(out, body.size());
    put_le<uint64_t>(out, 0);  // out_len
    put_le<uint32_t>(out, 0);  // crc32
    out.insert(out.end(), digest.b.begin(), digest.b.end());
    put_le<uint32_t>(out, static_cast<uint32_t>(meta.size()));
    out.insert(out.end(), meta.begin(), meta.end());
    out.resize(static_cast<size_t>(body_off ? body_off : file_len), 0);
    out.insert(out.end(), body.begin(), body.end());
    if (out.size() != file_len) Rcpp::stop("internal error: object length");

    Rcpp::RawVector r(static_cast<R_xlen_t>(out.size()));
    if (!out.empty()) std::memcpy(RAW(r), out.data(), out.size());
    return r;
}

//' Members of a .s1 sample object from its first bytes (header and member table): name, kind, offset, length.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List s1_members_(Rcpp::RawVector head) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(RAW(head));
    const size_t len = static_cast<size_t>(head.size());
    if (len < kHeader || std::memcmp(b, "SINGLET2", 8) != 0) Rcpp::stop("not a .s1 sample object (bad magic)");
    if (get_le<uint16_t>(b + 8) != 1) Rcpp::stop("unsupported .s1 version");
    const uint32_t n_members = get_le<uint32_t>(b + 12);
    const uint64_t table_off = get_le<uint64_t>(b + 16), table_len = get_le<uint64_t>(b + 24), file_len = get_le<uint64_t>(b + 32);
    if (table_off != kHeader) Rcpp::stop("unexpected member table offset");
    if (table_off + table_len > len) Rcpp::stop("need %f bytes for the member table", static_cast<double>(table_off + table_len));
    std::string sid(reinterpret_cast<const char*>(b + 40), 16);
    sid = sid.substr(0, sid.find('\0'));
    Rcpp::CharacterVector name(n_members);
    Rcpp::IntegerVector kind(n_members);
    Rcpp::NumericVector off(n_members), size(n_members);
    size_t q = static_cast<size_t>(table_off);
    const size_t end = static_cast<size_t>(table_off + table_len);
    for (uint32_t k = 0; k < n_members; ++k) {
        if (q + 2 > end) Rcpp::stop("member table truncated");
        const uint16_t nl = get_le<uint16_t>(b + q);
        q += 2;
        if (q + nl + kEntryFixed > end) Rcpp::stop("member table truncated");
        name[k] = std::string(reinterpret_cast<const char*>(b + q), nl);
        q += nl;
        kind[k] = b[q];
        off[k] = static_cast<double>(get_le<uint64_t>(b + q + 2));
        size[k] = static_cast<double>(get_le<uint64_t>(b + q + 10));
        const uint32_t ml = get_le<uint32_t>(b + q + kEntryFixed - 4);
        q += kEntryFixed + ml;
    }
    if (q != end) Rcpp::stop("trailing bytes in the member table");
    return Rcpp::List::create(Rcpp::Named("sample_id") = sid, Rcpp::Named("file_len") = static_cast<double>(file_len),
                              Rcpp::Named("name") = name, Rcpp::Named("kind") = kind, Rcpp::Named("off") = off, Rcpp::Named("len") = size);
}
