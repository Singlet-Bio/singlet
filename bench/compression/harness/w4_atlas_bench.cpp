// w4_atlas: recoding-potential driver for text members taken from public bundles.
//
//   bin/w4_atlas_bench.exe text <file> <label> <out.tsv> [cap_bytes [dzlxb]]   general-purpose codecs on the bytes of <file>
//                                         (cap: use only the first cap bytes, cut at a line; letters: d deflate, z zstd19, l zstd19+long, x xz, b brotli)
//   bin/w4_atlas_bench.exe mtx  <file> <label> <out.tsv>   parse a MatrixMarket coordinate text to CSC, then the study's codecs
//
// text: raw-deflate (zlib level 6 and 9; the bundle's own method), zstd-19, zstd-19 + long-distance window 2^27, xz -9e,
//       brotli-11 (lgwin 24). One TSV row: label, bytes, sizes, ms per codec.
// mtx : header "%%MatrixMarket matrix coordinate integer general", comment lines, "m n nnz", then "row col value" (1-based).
//       The file may be a PREFIX of the real matrix (cut at a line boundary): n is the last column seen.
//       Names are empty strings, so name blocks cost ~0 and the sizes compare the payload (indptr + indices + values).
//       Codecs: production .1pz writer (zstd-3 VOCSC, 1024 cols/chunk), naive zstd-3/zstd-19 over raw CSC, b_rans16 (Track B).
//       decode(encode(M)) == M is checked for the study codecs.
#include "common.h"
#include "ref_zstd_csc.h"
#include "track_b.h"
#include <algorithm>
#include <brotli/encode.h>
#include <iostream>
#include <lzma.h>
#include <zlib.h>
#include <zstd.h>

using namespace cs;

static size_t deflate_raw(const std::vector<uint8_t>& in, int level) {
    z_stream s{};
    deflateInit2(&s, level, Z_DEFLATED, -15, 9, Z_DEFAULT_STRATEGY);
    std::vector<uint8_t> out(deflateBound(&s, (uLong)in.size()) + 64);
    s.next_in = const_cast<Bytef*>(in.data());
    s.avail_in = (uInt)in.size();
    s.next_out = out.data();
    s.avail_out = (uInt)out.size();
    deflate(&s, Z_FINISH);
    size_t n = s.total_out;
    deflateEnd(&s);
    return n;
}
static size_t zstd_size(const std::vector<uint8_t>& in, int level, int window_log) {
    ZSTD_CCtx* c = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level);
    if (window_log > 0) {
        ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, window_log);
        ZSTD_CCtx_setParameter(c, ZSTD_c_enableLongDistanceMatching, 1);
    }
    std::vector<uint8_t> out(ZSTD_compressBound(in.size()));
    size_t z = ZSTD_compress2(c, out.data(), out.size(), in.data(), in.size());
    ZSTD_freeCCtx(c);
    if (ZSTD_isError(z)) throw std::runtime_error(ZSTD_getErrorName(z));
    return z;
}
static size_t xz_size(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> out(lzma_stream_buffer_bound(in.size()));
    size_t pos = 0;
    lzma_ret r = lzma_easy_buffer_encode(9 | LZMA_PRESET_EXTREME, LZMA_CHECK_NONE, nullptr, in.data(), in.size(), out.data(), &pos, out.size());
    if (r != LZMA_OK) throw std::runtime_error("lzma");
    return pos;
}
static size_t brotli_size(const std::vector<uint8_t>& in) {
    size_t n = BrotliEncoderMaxCompressedSize(in.size());
    std::vector<uint8_t> out(n);
    if (!BrotliEncoderCompress(11, 24, BROTLI_MODE_TEXT, in.size(), in.data(), &n, out.data())) throw std::runtime_error("brotli");
    return n;
}

template <class F> static double timed(F f, size_t& sz) {
    Timer t;
    sz = f();
    return t.ms();
}

static int run_text(const std::string& path, const std::string& label, const std::string& outp, size_t cap, const std::string& which) {
    auto buf = read_file(path);
    size_t full = buf.size();
    if (cap && buf.size() > cap) {
        size_t cut = cap;
        while (cut > 0 && buf[cut - 1] != '\n') --cut;
        if (cut) buf.resize(cut);
    }
    auto has = [&](char c) { return which.find(c) != std::string::npos; };
    size_t z6 = 0, z9 = 0, s19 = 0, s19l = 0, x = 0, b = 0;
    double t6 = 0, t9 = 0, ts = 0, tl = 0, tx = 0, tb = 0;
    if (has('d')) { t6 = timed([&] { return deflate_raw(buf, 6); }, z6); t9 = timed([&] { return deflate_raw(buf, 9); }, z9); }
    if (has('z')) ts = timed([&] { return zstd_size(buf, 19, 0); }, s19);
    if (has('l')) tl = timed([&] { return zstd_size(buf, 19, 27); }, s19l);
    if (has('x')) tx = timed([&] { return xz_size(buf); }, x);
    if (has('b')) tb = timed([&] { return brotli_size(buf); }, b);
    bool nf = !std::ifstream(outp).good();
    std::ofstream o(outp, std::ios::app);
    if (nf) o << "label\tbytes\tdeflate6\tdeflate9\tzstd19\tzstd19_long27\txz9e\tbrotli11\tms_d6\tms_d9\tms_zstd19\tms_zstd19long\tms_xz\tms_brotli\tfull_bytes\n";
    o << label << '\t' << buf.size() << '\t' << z6 << '\t' << z9 << '\t' << s19 << '\t' << s19l << '\t' << x << '\t' << b << '\t' << t6 << '\t' << t9
      << '\t' << ts << '\t' << tl << '\t' << tx << '\t' << tb << '\t' << full << '\n';
    std::printf("%s: %zu of %zu B -> deflate6 %zu deflate9 %zu zstd19 %zu zstd19L %zu xz9e %zu brotli11 %zu\n", label.c_str(), buf.size(), full, z6, z9, s19, s19l, x, b);
    return 0;
}

static bool parse_mtx(const std::string& path, CanonicalMatrix& M, uint64_t& declared_nnz, uint32_t& declared_n, bool& sorted_ok) {
    auto buf = read_file(path);
    const char* p = reinterpret_cast<const char*>(buf.data());
    const char* e = p + buf.size();
    auto skip_line = [&]() { while (p < e && *p != '\n') ++p; if (p < e) ++p; };
    while (p < e && *p == '%') skip_line();
    char* q;
    uint64_t m = strtoull(p, &q, 10); p = q;
    uint64_t n = strtoull(p, &q, 10); p = q;
    declared_nnz = strtoull(p, &q, 10); p = q;
    declared_n = (uint32_t)n;
    skip_line();
    std::vector<uint32_t> rows, cols, vals;
    rows.reserve(buf.size() / 8);
    cols.reserve(buf.size() / 8);
    vals.reserve(buf.size() / 8);
    while (p < e) {
        uint64_t r = strtoull(p, &q, 10);
        if (q == p) break;
        p = q;
        uint64_t c = strtoull(p, &q, 10); p = q;
        uint64_t v = strtoull(p, &q, 10); p = q;
        while (p < e && *p != '\n') ++p;
        if (p < e) ++p;
        rows.push_back((uint32_t)(r - 1));
        cols.push_back((uint32_t)(c - 1));
        vals.push_back((uint32_t)v);
    }
    size_t nnz = rows.size();
    sorted_ok = true;
    for (size_t i = 1; i < nnz; ++i)
        if (cols[i] < cols[i - 1] || (cols[i] == cols[i - 1] && rows[i] <= rows[i - 1])) { sorted_ok = false; break; }
    std::vector<uint32_t> ord;
    if (!sorted_ok) {
        ord.resize(nnz);
        for (size_t i = 0; i < nnz; ++i) ord[i] = (uint32_t)i;
        std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return cols[a] != cols[b] ? cols[a] < cols[b] : rows[a] < rows[b]; });
    }
    uint32_t nc = nnz ? cols[sorted_ok ? nnz - 1 : ord[nnz - 1]] + 1 : 0;
    M.m = (uint32_t)m; M.n = nc; M.nnz = nnz;
    M.indptr.assign(nc + 1, 0);
    M.indices.resize(nnz);
    M.data.resize(nnz);
    uint32_t mx = 0;
    for (size_t i = 0; i < nnz; ++i) {
        size_t k = sorted_ok ? i : ord[i];
        M.indices[i] = rows[k];
        M.data[i] = vals[k];
        mx = std::max(mx, vals[k]);
        M.indptr[cols[k] + 1]++;
    }
    for (uint32_t j = 0; j < nc; ++j) M.indptr[j + 1] += M.indptr[j];
    M.vt_code = (mx <= 255) ? 1 : (mx <= 65535) ? 2 : 3;
    M.rownames.assign(M.m, std::string());
    M.colnames.assign(M.n, std::string());
    return true;
}

static int run_mtx(const std::string& path, const std::string& label, const std::string& outp) {
    CanonicalMatrix M;
    uint64_t dnnz; uint32_t dn; bool sorted_ok;
    parse_mtx(path, M, dnnz, dn, sorted_ok);
    uint64_t nz_cols = 0;
    for (uint32_t j = 0; j < M.n; ++j) nz_cols += (M.indptr[j + 1] > M.indptr[j]);
    std::printf("%s: m=%u n=%u (declared n=%u) nnz=%llu (declared %llu) nonempty cols=%llu sorted=%d vt=%d\n", label.c_str(), M.m, M.n, dn,
                (unsigned long long)M.nnz, (unsigned long long)dnnz, (unsigned long long)nz_cols, (int)sorted_ok, M.vt_code);
    std::vector<Codec> codecs;
    codecs.push_back(make_ref_zstd_csc(3));
    codecs.push_back(make_ref_zstd_csc(19));
    std::vector<Codec> tb;
    register_track_b(tb);
    for (auto& c : tb) if (c.name == "b_rans16") codecs.push_back(c);
    bool nf = !std::ifstream(outp).good();
    std::ofstream o(outp, std::ios::app);
    if (nf) o << "label\tcodec\tenc_bytes\tm\tn\tnnz\tnonempty_cols\tbits_per_nnz\tenc_ms\tdec_ms\troundtrip_ok\n";
    // production writer
    {
        std::string tmp = "results/.w4_atlas_tmp.1pz";
        Timer t;
        auto b = rewrite_1pz(M, tmp);
        double ms = t.ms();
        std::remove(tmp.c_str());
        o << label << '\t' << "prod_1pz_v1_L3" << '\t' << b.size() << '\t' << M.m << '\t' << M.n << '\t' << M.nnz << '\t' << nz_cols << '\t' << b.size() * 8.0 / (double)M.nnz << '\t' << ms << "\t0\t1\n";
        std::printf("  %-18s %10zu B  %.3f bits/nnz\n", "prod_1pz_v1_L3", b.size(), b.size() * 8.0 / (double)M.nnz);
    }
    for (auto& c : codecs) {
        Timer te;
        auto enc = c.encode(M);
        double em = te.ms();
        Timer td;
        auto D = c.decode(enc);
        double dm = td.ms();
        bool ok = (D == M);
        o << label << '\t' << c.name << '\t' << enc.size() << '\t' << M.m << '\t' << M.n << '\t' << M.nnz << '\t' << nz_cols << '\t' << enc.size() * 8.0 / (double)M.nnz << '\t' << em << '\t' << dm << '\t' << (ok ? 1 : 0) << '\n';
        std::printf("  %-18s %10zu B  %.3f bits/nnz  enc %.0f ms dec %.0f ms  %s\n", c.name.c_str(), enc.size(), enc.size() * 8.0 / (double)M.nnz, em, dm, ok ? "roundtrip OK" : "ROUNDTRIP FAIL");
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) { std::cerr << "usage: w4_atlas_bench (text|mtx) file label out.tsv\n"; return 2; }
    std::string mode = argv[1];
    try {
        if (mode == "text") return run_text(argv[2], argv[3], argv[4], argc > 5 ? (size_t)atoll(argv[5]) : 0, argc > 6 ? argv[6] : "dzxb");
        if (mode == "mtx") return run_mtx(argv[2], argv[3], argv[4]);
    } catch (std::exception& e) { std::cerr << "error: " << e.what() << "\n"; return 1; }
    return 2;
}
