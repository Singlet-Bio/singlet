// SPDX-License-Identifier: MIT
// pz2 round trip on every legacy .1pz matrix of the compression-study corpora:
//   bench/compression/corpus/*/*.1pz and bench/compression/corpus_w4_corpus/*/*.1pz   (skips quietly when they are absent)
// For each matrix: encode (codec Auto, names inline, kv, digest), decode on every supported ISA in both modes, compare with the
// legacy arrays byte for byte, check names / kv / stored digest, ISA-identical native output, thread-independence, and random column
// ranges decoded from a *partial* byte source (prefix + one chunk run, as a remote client would).
//
// usage: test_pz2_roundtrip [files.1pz ...]        env: PZ2_TEST_MAX_MB (skip files larger than this), PZ2_CORPUS (extra root)
#define SINGLET_TEST_HARNESS_QUIET
#include "test_harness.h"
#include <singlet/pileup/pz_reader.h>
#include <singlet/pz2/pz2.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>

using namespace singlet::pz2;
namespace fs = std::filesystem;

#ifndef PZ2_SOURCE_DIR
#define PZ2_SOURCE_DIR "."
#endif

static std::vector<std::string> discover() {
    std::vector<std::string> out;
    std::vector<fs::path> roots = {fs::path(PZ2_SOURCE_DIR) / "bench/compression/corpus", fs::path(PZ2_SOURCE_DIR) / "bench/compression/corpus_w4_corpus"};
    if (const char* e = std::getenv("PZ2_CORPUS")) roots.push_back(e);
    for (auto& r : roots) {
        std::error_code ec;
        if (!fs::is_directory(r, ec)) continue;
        for (auto& d : fs::directory_iterator(r, ec)) {
            if (!d.is_directory()) continue;
            for (auto& f : fs::directory_iterator(d.path(), ec))
                if (f.path().extension() == ".1pz") out.push_back(f.path().string());
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Case {
    std::string path;
    singlet::pz::ReadResult r;
};

static void check_one(const std::string& path, size_t file_bytes) {
    singlet::pz::ReadResult r;
    try { r = singlet::pz::read_1pz(path); }
    catch (const std::exception& e) { CHECK(false, path + ": legacy reader failed: " + e.what()); return; }

    CSC c;
    c.m = r.m; c.n = r.n; c.nnz = r.nnz;
    c.indptr = r.indptr.data(); c.indices = r.indices.data(); c.data = r.data.data(); c.data_type = DType::U32;
    EncodeOptions opt;
    opt.row_names.blob = join_names(r.rownames);
    opt.col_names.blob = join_names(r.colnames);
    opt.kv = KvMap(r.user_kv.begin(), r.user_kv.end());
    if (r.vt_code >= 1 && r.vt_code <= 3) opt.value_type = ValueType(r.vt_code);   // keep the legacy storage type
    opt.threads = 4;
    opt.zstd_level = 3;   // side streams only; decoding does not depend on the level
    EncodeStats st;
    double t0 = now_s();
    std::vector<uint8_t> bytes;
    try { bytes = encode_matrix(c, opt, &st); }
    catch (const std::exception& e) { CHECK(false, path + ": encode failed: " + e.what()); return; }
    double t_enc = now_s() - t0;

    ByteSource src(bytes);
    Matrix mx = Matrix::open(src);
    const Header& h = mx.header();
    CHECK(h.m == r.m && h.n == r.n && h.nnz == r.nnz && uint8_t(h.value_type) == r.vt_code, path + ": header fields");
    CHECK(h.total_len == bytes.size(), path + ": total_len");

    // names, kv, digest
    NamesSpec rn = mx.row_names(src), cn = mx.col_names(src);
    CHECK(rn.blob == opt.row_names.blob && cn.blob == opt.col_names.blob, path + ": names round trip");
    CHECK(mx.kv(src) == opt.kv, path + ": kv round trip");

    std::vector<std::vector<uint32_t>> nat_idx_by_isa, nat_val_by_isa;
    double t_dec_best = 1e9;
    for (Isa isa : {Isa::Scalar, Isa::Avx2, Isa::Avx512}) {
        if (!isa_supported(isa)) continue;
        for (Mode mode : {Mode::Native, Mode::Canonical}) {
            for (int threads : {1, 3}) {
                if (threads == 3 && !(isa == best_isa())) continue;
                std::vector<uint32_t> ip(r.n + 1), ix(r.nnz + 1), dv(r.nnz + 1);
                Output o;
                o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
                DecodeOptions d;
                d.mode = mode; d.isa = isa; d.threads = threads;
                double t1 = now_s();
                try { mx.decode(src, o, d); }
                catch (const std::exception& e) { CHECK(false, path + ": decode failed " + isa_name(isa) + ": " + e.what()); continue; }
                if (threads == 1) t_dec_best = std::min(t_dec_best, now_s() - t1);
                const std::string tag = path + " [" + isa_name(isa) + (mode == Mode::Native ? " native" : " canonical") + " thr" + std::to_string(threads) + "]";
                const bool ptr_ok = std::equal(r.indptr.begin(), r.indptr.end(), ip.begin());
                CHECK(ptr_ok, tag + ": indptr");
                if (mode == Mode::Canonical) {
                    CHECK(std::equal(r.indices.begin(), r.indices.end(), ix.begin()), tag + ": indices");
                    CHECK(std::equal(r.data.begin(), r.data.end(), dv.begin()), tag + ": data");
                    if (isa == best_isa() && threads == 1) {   // digest of the decoded canonical matrix equals the stored one
                        CSC dc = c;
                        dc.indices = ix.data(); dc.data = dv.data(); dc.data_type = DType::U32;
                        CHECK(canonical_digest(dc, h.value_type, rn.blob, cn.blob, opt.kv) == h.digest, tag + ": stored digest");
                    }
                } else if (ptr_ok) {
                    const auto& perm = mx.row_perm();
                    bool ok = true;
                    std::vector<std::pair<uint32_t, uint32_t>> a, b;
                    for (uint32_t j = 0; j < r.n && ok; ++j) {
                        a.clear(); b.clear();
                        for (uint32_t t = r.indptr[j]; t < r.indptr[j + 1]; ++t) {
                            a.push_back({r.indices[t], r.data[t]});
                            b.push_back({perm.empty() ? ix[t] : perm[ix[t]], dv[t]});
                            if (t > r.indptr[j] && ix[t] <= ix[t - 1]) ok = false;   // stored ids ascend inside a column
                        }
                        std::sort(b.begin(), b.end());
                        ok = ok && a == b;
                    }
                    CHECK(ok, tag + ": native content (via row_perm)");
                    if (threads == 1) { nat_idx_by_isa.push_back(ix); nat_val_by_isa.push_back(dv); }
                }
            }
        }
    }
    for (size_t i = 1; i < nat_idx_by_isa.size(); ++i)
        CHECK(nat_idx_by_isa[i] == nat_idx_by_isa[0] && nat_val_by_isa[i] == nat_val_by_isa[0], path + ": native output identical across ISAs");

    // random column ranges from a partial source: prefix + the planned chunk run only
    std::mt19937 rng(12345);
    const uint64_t pref = h.prefix_len();
    for (int it = 0; it < 4 && r.n > 0; ++it) {
        uint32_t a = rng() % r.n, len = 1 + (it == 0 ? 0 : rng() % std::min<uint32_t>(r.n, it == 3 ? 2000 : 300));
        uint32_t b = std::min<uint32_t>(r.n, a + len);
        ByteSource part;
        part.add(0, bytes.data(), size_t(pref));
        Matrix pm = Matrix::open(part);
        ColumnPlan pl = pm.plan_columns(a, b);
        part.add(pl.byte_off, bytes.data() + pl.byte_off, size_t(pl.byte_len));
        for (Mode mode : {Mode::Canonical, Mode::Native}) {
            std::vector<uint32_t> ip(b - a + 1), ix(pl.max_nnz + 1), dv(pl.max_nnz + 1);
            Output o;
            o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
            DecodeOptions d;
            d.mode = mode; d.threads = it & 1 ? 2 : 1;
            uint64_t nz;
            try { nz = pm.decode_columns(part, a, b, o, d); }
            catch (const std::exception& e) { CHECK(false, path + ": decode_columns failed: " + e.what()); continue; }
            const uint32_t base = r.indptr[a];
            bool ok = nz == r.indptr[b] - base;
            for (uint32_t j = 0; j <= b - a && ok; ++j) ok = ip[j] == r.indptr[a + j] - base;
            if (mode == Mode::Canonical) for (uint64_t t = 0; t < nz && ok; ++t) ok = ix[t] == r.indices[base + t] && dv[t] == r.data[base + t];
            else {
                const auto& perm = pm.row_perm();
                for (uint32_t j = a; j < b && ok; ++j) {
                    std::vector<std::pair<uint32_t, uint32_t>> x, y;
                    for (uint32_t t = r.indptr[j]; t < r.indptr[j + 1]; ++t) { x.push_back({r.indices[t], r.data[t]}); y.push_back({perm.empty() ? ix[t - base] : perm[ix[t - base]], dv[t - base]}); }
                    std::sort(y.begin(), y.end());
                    ok = x == y;
                }
            }
            CHECK(ok, path + ": decode_columns [" + std::to_string(a) + "," + std::to_string(b) + ") " + (mode == Mode::Canonical ? "canonical" : "native"));
            if (it == 0 && mode == Mode::Canonical) {
                auto cc = pm.column_counts(part, a, b);
                CHECK(cc.size() == b - a && (cc.empty() || cc[0] == r.indptr[a + 1] - r.indptr[a]), path + ": column_counts");
            }
        }
    }
    std::printf("  %-52s m=%-7u n=%-7u nnz=%-9llu legacy %9zu -> pz2 %9zu (%.3f) codec %d cols/chunk %-4u enc %.2fs dec1 %.0f Mnnz/s\n", path.substr(path.size() > 52 ? path.size() - 52 : 0).c_str(),
                r.m, r.n, (unsigned long long)r.nnz, file_bytes, bytes.size(), file_bytes ? double(bytes.size()) / double(file_bytes) : 0.0, int(st.codec), st.chunk_cols, t_enc,
                t_dec_best > 0 ? double(r.nnz) / t_dec_best / 1e6 : 0.0);
}

int main(int argc, char** argv) {
    std::printf("%s\n", isa_report().c_str());
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) files.push_back(argv[i]);
    if (files.empty()) files = discover();
    if (files.empty()) { std::printf("test_pz2_roundtrip: no corpus found, skipped\n"); return 0; }
    double max_mb = 0;
    if (const char* e = std::getenv("PZ2_TEST_MAX_MB")) max_mb = std::atof(e);
    size_t done = 0, skipped = 0;
    for (auto& f : files) {
        std::error_code ec;
        size_t sz = size_t(fs::file_size(f, ec));
        if (max_mb > 0 && double(sz) > max_mb * 1048576.0) { ++skipped; continue; }
        check_one(f, sz);
        ++done;
    }
    std::printf("test_pz2_roundtrip: %zu matrices (%zu skipped by size), %d checks passed, %d failed\n", done, skipped, g_pass, g_fail);
    return g_fail ? 1 : 0;
}
