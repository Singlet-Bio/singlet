// w4_vocab: reader for "SVP0" feature-name packs (see tools/w4_vocab_build.py for the writer and notes/w4_vocab.md for the spec).
// Decodes the structured payload (gene table + exon lengths/gaps) and materialises exon / intron row names, eagerly (blob +
// offsets) or lazily (one name on demand from an index).  Needs zstd only.  Not covered: composite packs (barnyard), vocab JSON.
#pragma once
#include <zstd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace w4v {

struct Pack {
    std::string build, prefix, schema;
    int width = 0;
    std::vector<uint32_t> nums, nex, gstart, gch, elen, egap;
    std::vector<uint8_t> kind;
    std::vector<uint32_t> sym_off;       // explicit symbols: offsets into sym_blob (size n_explicit + 1)
    std::string sym_blob;
    std::vector<std::string> chroms;
    // index (build_index): exon_row0[g] = first exon row of gene g; intron_row0[g]; absolute exon start/end; explicit-symbol rank
    std::vector<uint32_t> exon_row0, intron_row0, estart, eend, symrank;
    size_t n_genes() const { return nums.size(); }
    size_t n_exons() const { return elen.size(); }
    size_t n_introns() const { return egap.size(); }
};

namespace detail {
struct Cur {
    const uint8_t* p;
    const uint8_t* end;
    template <class T> T get() { T v; if (end - p < (ptrdiff_t)sizeof(T)) throw std::runtime_error("svp: truncated"); std::memcpy(&v, p, sizeof(T)); p += sizeof(T); return v; }
};
inline std::vector<uint8_t> zdec(Cur& c, uint32_t clen) {
    if ((size_t)(c.end - c.p) < clen) throw std::runtime_error("svp: truncated frame");
    unsigned long long n = ZSTD_getFrameContentSize(c.p, clen);
    if (n == ZSTD_CONTENTSIZE_UNKNOWN || n == ZSTD_CONTENTSIZE_ERROR) throw std::runtime_error("svp: frame without content size");
    std::vector<uint8_t> out(n);
    size_t r = ZSTD_decompress(out.data(), n, c.p, clen);
    if (ZSTD_isError(r) || r != n) throw std::runtime_error("svp: zstd error");
    c.p += clen;
    return out;
}
inline std::vector<uint32_t> stream(Cur& c) {    // u8 mode, u32 n, u32 clen, zstd frame
    uint8_t mode = c.get<uint8_t>(); uint32_t n = c.get<uint32_t>(); uint32_t cl = c.get<uint32_t>();
    std::vector<uint8_t> raw = zdec(c, cl);
    std::vector<uint32_t> v(n);
    if (mode == 0) {
        size_t q = 0;
        for (uint32_t i = 0; i < n; i++) {
            uint64_t x = 0; int s = 0;
            for (;;) { uint8_t b = raw[q++]; x |= (uint64_t)(b & 127) << s; if (b < 128) break; s += 7; }
            v[i] = (uint32_t)x;
        }
    } else {
        for (uint32_t i = 0; i < n; i++)
            v[i] = raw[i] | (raw[n + i] << 8) | (raw[2 * (size_t)n + i] << 16) | ((uint32_t)raw[3 * (size_t)n + i] << 24);
    }
    return v;
}
inline std::vector<uint8_t> blob(Cur& c) { uint32_t cl = c.get<uint32_t>(); return zdec(c, cl); }
}  // namespace detail

inline Pack decode(const uint8_t* buf, size_t len) {
    detail::Cur c{buf, buf + len};
    if (len < 4 || std::memcmp(buf, "SVP0", 4)) throw std::runtime_error("svp: bad magic");
    c.p += 4;
    c.get<uint16_t>(); c.get<uint16_t>();                      // version, flags
    uint8_t pl = c.get<uint8_t>(); uint8_t width = c.get<uint8_t>(); c.get<uint16_t>();
    uint32_t ng = c.get<uint32_t>(); uint32_t ne = c.get<uint32_t>();
    Pack P; P.width = width;
    uint8_t bl = c.get<uint8_t>();
    P.build.assign((const char*)c.p, bl); c.p += bl;
    P.prefix.assign((const char*)c.p, pl); c.p += pl;
    P.schema.assign((const char*)c.p, 8); c.p += 8;
    std::vector<uint32_t> d = detail::stream(c);
    P.nums.resize(ng); uint64_t acc = 0;
    for (uint32_t g = 0; g < ng; g++) { acc += (g == 0 ? d[0] : (uint64_t)d[g] + 1); P.nums[g] = (uint32_t)acc; }
    auto kb = detail::blob(c); P.kind.assign(kb.begin(), kb.end());
    auto sb = detail::blob(c);
    P.sym_blob.assign((const char*)sb.data(), sb.size());
    P.sym_off.push_back(0);
    for (size_t i = 0; i < sb.size(); i++) if (sb[i] == 0) P.sym_off.push_back((uint32_t)i + 1);
    if (!sb.empty()) P.sym_off.push_back((uint32_t)sb.size() + 1);   // end sentinel (the last symbol has no terminator)
    auto cb = detail::blob(c);
    { size_t s = 0; for (size_t i = 0; i <= cb.size(); i++) if (i == cb.size() || cb[i] == 0) { P.chroms.emplace_back((const char*)cb.data() + s, i - s); s = i + 1; } }
    P.gch = detail::stream(c);
    P.nex = detail::stream(c); for (auto& x : P.nex) x += 1;
    P.gstart = detail::stream(c);
    P.elen = detail::stream(c);
    P.egap = detail::stream(c);
    if (c.p != c.end || P.elen.size() != ne) throw std::runtime_error("svp: size mismatch");
    return P;
}

inline void build_index(Pack& P) {
    size_t ng = P.n_genes();
    P.exon_row0.assign(ng + 1, 0); P.intron_row0.assign(ng + 1, 0); P.symrank.assign(ng + 1, 0);
    for (size_t g = 0; g < ng; g++) {
        P.exon_row0[g + 1] = P.exon_row0[g] + P.nex[g];
        P.intron_row0[g + 1] = P.intron_row0[g] + P.nex[g] - 1;
        P.symrank[g + 1] = P.symrank[g] + (P.kind[g] == 2);
    }
    P.estart.resize(P.elen.size()); P.eend.resize(P.elen.size());
    for (size_t g = 0; g < ng; g++) {
        uint32_t a = P.gstart[g];
        uint32_t e0 = P.exon_row0[g], i0 = P.intron_row0[g];
        for (uint32_t k = 0; k < P.nex[g]; k++) {
            P.estart[e0 + k] = a; a += P.elen[e0 + k]; P.eend[e0 + k] = a;
            if (k + 1 < P.nex[g]) a += P.egap[i0 + k];
        }
    }
}

// ---- formatting
inline char* put_u(char* p, uint32_t v) {
    static const char* LUT =
        "00010203040506070809101112131415161718192021222324252627282930313233343536373839"
        "40414243444546474849505152535455565758596061626364656667686970717273747576777879"
        "8081828384858687888990919293949596979899";
    char t[12]; char* e = t + 12; char* q = e;
    while (v >= 100) { uint32_t r = v % 100; v /= 100; q -= 2; std::memcpy(q, LUT + 2 * r, 2); }
    if (v >= 10) { q -= 2; std::memcpy(q, LUT + 2 * v, 2); } else { *--q = (char)('0' + v); }
    size_t n = (size_t)(e - q);
    std::memcpy(p, q, 12 > n ? n : 12);
    return p + n;
}
// "<gid>_<symbol>_[intron_]" into out; returns end
inline char* gene_head(const Pack& P, size_t g, bool intron, char* out) {
    char* p = out;
    std::memcpy(p, P.prefix.data(), P.prefix.size()); p += P.prefix.size();
    char d[12]; uint32_t v = P.nums[g];
    for (int i = P.width - 1; i >= 0; i--) { d[i] = (char)('0' + v % 10); v /= 10; }
    std::memcpy(p, d, P.width); p += P.width;
    *p++ = '_';
    if (P.kind[g] == 1) { std::memcpy(p, out, P.prefix.size() + P.width); p += P.prefix.size() + P.width; }
    else if (P.kind[g] == 2) { uint32_t r = P.symrank[g]; uint32_t a = P.sym_off[r], b = P.sym_off[r + 1] - 1; std::memcpy(p, P.sym_blob.data() + a, b - a); p += b - a; }
    *p++ = '_';
    if (intron) { std::memcpy(p, "intron_", 7); p += 7; }
    return p;
}
inline char* tail(const Pack& P, size_t g, uint32_t a, uint32_t b, char* p) {
    const std::string& c = P.chroms[P.gch[g]];
    std::memcpy(p, c.data(), c.size()); p += c.size();
    *p++ = ':'; p = put_u(p, a); *p++ = '-'; p = put_u(p, b); *p++ = 0;
    return p;
}

// Eager: all names NUL-terminated in one blob (same bytes as the NUL-joined row-name list), offsets[i] = start of name i.
inline void materialise(const Pack& P, bool intron, std::vector<char>& blob, std::vector<uint32_t>& offs) {
    size_t rows = intron ? P.n_introns() : P.n_exons();
    size_t max_sym = 0, max_chr = 0;
    for (size_t i = 0; i + 1 < P.sym_off.size(); i++) max_sym = std::max<size_t>(max_sym, P.sym_off[i + 1] - P.sym_off[i]);
    for (auto& c : P.chroms) max_chr = std::max(max_chr, c.size());
    size_t cap = rows * (P.prefix.size() + P.width + max_sym + max_chr + 48) + 64;     // upper bound per name
    blob.resize(cap); offs.resize(rows + 1);
    char* base = blob.data(); char* p = base; size_t r = 0;
    char head[256];
    for (size_t g = 0; g < P.n_genes(); g++) {
        size_t hl = gene_head(P, g, intron, head) - head;
        uint32_t e0 = P.exon_row0[g];
        uint32_t cnt = intron ? P.nex[g] - 1 : P.nex[g];
        for (uint32_t k = 0; k < cnt; k++) {
            offs[r++] = (uint32_t)(p - base);
            std::memcpy(p, head, hl); p += hl;
            uint32_t a = intron ? P.eend[e0 + k] : P.estart[e0 + k];
            uint32_t b = intron ? P.estart[e0 + k + 1] : P.eend[e0 + k];
            p = tail(P, g, a, b, p);
        }
    }
    offs[rows] = (uint32_t)(p - base);
    blob.resize(p - base);
}

// Lazy: one name on demand (needs build_index).  Returns length (without NUL); out must hold >= 512 bytes.
inline size_t name_at(const Pack& P, bool intron, size_t row, char* out) {
    const std::vector<uint32_t>& r0 = intron ? P.intron_row0 : P.exon_row0;
    size_t g = (size_t)(std::upper_bound(r0.begin(), r0.end(), (uint32_t)row) - r0.begin()) - 1;
    // genes with zero introns share the same r0 value: upper_bound lands past them, which is the owning gene
    uint32_t k = (uint32_t)row - r0[g];
    uint32_t e = P.exon_row0[g] + k;
    char* p = gene_head(P, g, intron, out);
    uint32_t a = intron ? P.eend[e] : P.estart[e], b = intron ? P.estart[e + 1] : P.eend[e];
    p = tail(P, g, a, b, p);
    return (size_t)(p - out) - 1;
}

}  // namespace w4v
