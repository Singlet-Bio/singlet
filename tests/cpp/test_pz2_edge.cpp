// SPDX-License-Identifier: MIT
// pz2 synthetic edge cases: empty / degenerate shapes, m = 2^21 - 1, values up to 2^32 - 1, escapes, f32, zero values, every option,
// names kinds, output types, partial sources, corruption robustness, thread-independent encoding, SHA-256 vectors.
#define SINGLET_TEST_HARNESS_QUIET
#include "test_harness.h"
#include <singlet/pz2/pz2.h>

#include <cstdio>
#include <random>

using namespace singlet::pz2;

struct Mat {
    uint32_t m = 0, n = 0;
    std::vector<uint32_t> ptr{0}, idx, val;
    uint64_t nnz() const { return idx.size(); }
    void add_col(const std::vector<std::pair<uint32_t, uint32_t>>& c) {   // (row, value), rows ascending
        for (auto& p : c) { idx.push_back(p.first); val.push_back(p.second); }
        ptr.push_back(uint32_t(idx.size()));
        ++n;
    }
    CSC csc() const {
        CSC c;
        c.m = m; c.n = n; c.nnz = nnz();
        c.indptr = ptr.data(); c.indices = idx.data(); c.data = val.data(); c.data_type = DType::U32;
        return c;
    }
};

static EncodeOptions NN() {   // options for a matrix without names
    EncodeOptions o;
    o.row_names.kind = NamesKind::None; o.col_names.kind = NamesKind::None;
    return o;
}

static std::vector<Isa> isas() {
    std::vector<Isa> v;
    for (Isa i : {Isa::Scalar, Isa::Avx2, Isa::Avx512}) if (isa_supported(i)) v.push_back(i);
    return v;
}

static std::vector<std::string> make_names(const char* pre, uint32_t n) {
    std::vector<std::string> v;
    for (uint32_t i = 0; i < n; ++i) v.push_back(std::string(pre) + std::to_string(i));
    return v;
}

// Encode, then verify everything on every ISA / mode.  Returns the member bytes (empty on failure).
static std::vector<uint8_t> roundtrip(const Mat& M, EncodeOptions opt, const std::string& name, EncodeStats* stp = nullptr, bool names = true) {
    CSC c = M.csc();
    if (names) {
        if (opt.row_names.kind == NamesKind::Inline && opt.row_names.blob.empty()) opt.row_names.blob = join_names(make_names("g", M.m));
        if (opt.col_names.kind == NamesKind::Inline && opt.col_names.blob.empty()) opt.col_names.blob = join_names(make_names("c", M.n));
    } else {   // no names at all
        opt.row_names.kind = NamesKind::None; opt.col_names.kind = NamesKind::None;
    }
    EncodeStats st;
    std::vector<uint8_t> bytes;
    try { bytes = encode_matrix(c, opt, &st); }
    catch (const std::exception& e) { CHECK(false, name + ": encode threw " + e.what()); return {}; }
    if (stp) *stp = st;
    ByteSource src(bytes);
    Matrix mx;
    try { mx = Matrix::open(src); }
    catch (const std::exception& e) { CHECK(false, name + ": open threw " + e.what()); return {}; }
    CHECK(mx.header().m == M.m && mx.header().n == M.n && mx.header().nnz == M.nnz(), name + ": header shape");
    for (Isa isa : isas()) {
        for (Mode mode : {Mode::Native, Mode::Canonical}) {
            std::vector<uint32_t> ip(M.n + 1, 77), ix(M.nnz() + 1, 77), dv(M.nnz() + 1, 77);
            Output o;
            o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
            DecodeOptions d;
            d.mode = mode; d.isa = isa;
            try { mx.decode(src, o, d); }
            catch (const std::exception& e) { CHECK(false, name + ": decode threw " + e.what()); continue; }
            const std::string tag = name + " [" + isa_name(isa) + (mode == Mode::Native ? " native" : " canonical") + "]";
            CHECK(ip == M.ptr, tag + ": indptr");
            const auto& perm = mx.row_perm();
            bool ok = true;
            for (uint32_t j = 0; j < M.n && ok; ++j) {
                std::vector<std::pair<uint32_t, uint32_t>> a, b;
                for (uint32_t t = M.ptr[j]; t < M.ptr[j + 1]; ++t) {
                    a.push_back({M.idx[t], M.val[t]});
                    uint32_t id = (mode == Mode::Canonical || perm.empty()) ? ix[t] : perm[ix[t]];
                    b.push_back({id, dv[t]});
                    if (t > M.ptr[j] && ix[t] <= ix[t - 1]) ok = false;
                }
                std::sort(b.begin(), b.end());
                ok = ok && a == b;
                if (mode == Mode::Canonical) ok = ok && std::equal(a.begin(), a.end(), b.begin());   // canonical = ascending ids as given
            }
            CHECK(ok, tag + ": content");
            CHECK(ix[M.nnz()] == 77 && dv[M.nnz()] == 77 && ip[M.n] == M.nnz(), tag + ": no write past the end");
        }
    }
    if (opt.row_names.kind == NamesKind::Inline && names) CHECK(mx.row_names(src).blob == opt.row_names.blob, name + ": row names");
    if (opt.col_names.kind == NamesKind::Inline && names) CHECK(mx.col_names(src).blob == opt.col_names.blob, name + ": col names");
    return bytes;
}

static Mat random_mat(uint32_t m, uint32_t n, double mean_k, uint32_t seed, uint32_t maxval, bool big_values = false, double empty_frac = 0.1) {
    std::mt19937_64 g(seed);
    Mat M;
    M.m = m;
    for (uint32_t j = 0; j < n; ++j) {
        std::vector<std::pair<uint32_t, uint32_t>> col;
        if ((g() % 1000) / 1000.0 >= empty_frac && m) {
            uint32_t k = std::min<uint64_t>(m, uint64_t(mean_k * (0.2 + (g() % 1000) / 400.0)) + 1);
            std::vector<uint32_t> rows;
            for (uint32_t i = 0; i < k; ++i) rows.push_back(uint32_t(g() % m));
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
            for (uint32_t r : rows) {
                uint32_t v = 1;
                if (g() % 3 == 0) v = 1 + uint32_t(g() % maxval);
                if (big_values && g() % 17 == 0) v = uint32_t(g());
                if (v == 0) v = 1;
                col.push_back({r, v});
            }
        }
        M.add_col(col);
    }
    return M;
}

int main() {
    std::printf("%s\n", isa_report().c_str());

    // ---- SHA-256 vectors ------------------------------------------------------------------------------------------------
    for (int use_ni = 0; use_ni <= 1; ++use_ni) {
        if (use_ni && !detail::Sha256::sha_ni_available()) { std::printf("  (SHA-NI not available on this CPU: portable path only)\n"); continue; }
        const std::string which = use_ni ? " [sha-ni]" : " [portable]";
        auto hex = [&](const std::string& s) {
            detail::Sha256 h(use_ni);
            h.update(s.data(), s.size());
            uint8_t o[32];
            h.final(o);
            Digest16 d;
            std::memcpy(d.b.data(), o, 16);
            return d.hex();
        };
        CHECK(hex("abc") == "ba7816bf8f01cfea414140de5dae2223", "sha256(abc)" + which);
        CHECK(hex("") == "e3b0c44298fc1c149afbf4c8996fb924", "sha256(empty)" + which);
        CHECK(hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") == "248d6a61d20638b8e5c026930c3e6039", "sha256(448-bit message)" + which);
        CHECK(hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67", "sha256(one million a)" + which);
        detail::Sha256 h(use_ni);   // streaming in odd pieces equals one shot
        std::string s(1000, 'x');
        for (size_t i = 0; i < s.size();) { size_t n = std::min<size_t>(1 + (i * 7) % 61, s.size() - i); h.update(s.data() + i, n); i += n; }
        uint8_t o1[32], o2[32];
        h.final(o1);
        detail::Sha256 h2(use_ni);
        h2.update(s.data(), s.size());
        h2.final(o2);
        CHECK(std::memcmp(o1, o2, 32) == 0, "sha256 streaming == one shot");
    }

    if (detail::Sha256::sha_ni_available()) {   // SHA-NI and the portable block function agree on every length
        std::mt19937_64 g(9);
        bool ok = true;
        for (int it = 0; it < 400 && ok; ++it) {
            size_t len = it < 300 ? size_t(it) : size_t(g() % 200000);
            std::vector<uint8_t> d(len);
            for (auto& x : d) x = uint8_t(g());
            detail::Sha256 a0(0), a1(1);
            size_t cut = len ? g() % len : 0;
            a0.update(d.data(), len);
            a1.update(d.data(), cut); a1.update(d.data() + cut, len - cut);
            uint8_t o0[32], o1[32];
            a0.final(o0); a1.final(o1);
            ok = std::memcmp(o0, o1, 32) == 0;
        }
        CHECK(ok, "sha-ni == portable on 400 random messages");
    }

    // ---- golden digests, computed by the independent Python implementation (bench/pz2/ref_decode.py digest(), spec 4.6) -----------------
    {
        Mat M;
        M.m = 3;
        M.add_col({{0, 5}, {2, 1}});
        M.add_col({{1, 70000}});
        CHECK(canonical_digest(M.csc(), ValueType::U32, join_names({"a", "b", "c"}), join_names({"x", "y"}), {{"k", "v"}, {"a", ""}}).hex() == "fffe6a424a4871ee09826cf236aee637", "golden digest: small u32 matrix with names and kv");
        Mat E;
        CHECK(canonical_digest(E.csc(), ValueType::U8, {}, {}, {}).hex() == "e33f41b7bd34678fef1ffe591f4f4a09", "golden digest: empty matrix");
        Mat F;
        F.m = 2;
        F.add_col({{0, 0}, {1, 0}});
        const float fv[2] = {0.5f, 3.0f};
        CSC c = F.csc();
        c.data = fv; c.data_type = DType::F32;
        CHECK(canonical_digest(c, ValueType::F32, join_names({"p", "q"}), join_names({"c"}), {}).hex() == "6283c8715084ba16a9066a699294865b", "golden digest: f32 matrix (bit patterns)");
    }

    // ---- empty and degenerate shapes ------------------------------------------------------------------------------------------
    for (CodecId codec : {CodecId::Auto, CodecId::Generic}) {
        EncodeOptions o;
        o.codec = codec;
        Mat a;                     roundtrip(a, o, "m0n0");
        Mat b; b.m = 5;            roundtrip(b, o, "n0");
        Mat c; c.m = 0; for (int i = 0; i < 3; ++i) c.add_col({}); roundtrip(c, o, "m0n3");
        Mat d; d.m = 100; for (int i = 0; i < 50; ++i) d.add_col({}); roundtrip(d, o, "nnz0");
        Mat e; e.m = 1; e.add_col({{0, 1}}); roundtrip(e, o, "1x1");
        Mat f; f.m = 3; f.add_col({{0, 2}, {2, 9}}); roundtrip(f, o, "tiny 3 nnz");
    }
    {
        EncodeOptions o;
        o.codec = CodecId::Rans;
        Mat a;                     roundtrip(a, o, "rans m0n0");
        Mat d; d.m = 100; for (int i = 0; i < 50; ++i) d.add_col({}); roundtrip(d, o, "rans nnz0");
        Mat f; f.m = 3; f.add_col({{0, 2}, {2, 9}}); roundtrip(f, o, "rans tiny 3 nnz");
    }

    // ---- sizes, escapes, huge values ------------------------------------------------------------------------------------------
    EncodeOptions r1 = NN();
    r1.codec = CodecId::Rans;
    {   // m = 2^21 - 1, sparse random columns: gaps up to 2^21 exercise the >16-bit gap escape (phase B2)
        Mat M = random_mat(kRansMaxRows, 40, 6, 11, 5);
        EncodeStats st;
        roundtrip(M, r1, "m=2^21-1 sparse", &st, false);
        CHECK(st.codec == CodecId::Rans, "m=2^21-1 uses codec 1");
    }
    {   // one dense column of 2^21 - 1 rows with large values (also exercises the 16+ bit residual escape, phase C3)
        Mat M;
        M.m = kRansMaxRows;
        std::vector<std::pair<uint32_t, uint32_t>> col;
        std::mt19937_64 g(5);
        for (uint32_t i = 0; i < M.m; ++i) col.push_back({i, g() % 5 == 0 ? uint32_t(g()) | 2u : 1u + uint32_t(g() % 3)});
        M.add_col(col);
        roundtrip(M, r1, "single dense column m=2^21-1", nullptr, false);
    }
    {   // values 2^32 - 1 and friends
        Mat M = random_mat(5000, 300, 40, 12, 9, true);
        M.val[0] = 0xFFFFFFFFu;
        M.val[M.val.size() / 2] = 0x80000000u;
        M.val[M.val.size() / 3] = 65536; M.val[M.val.size() / 4] = 65535; M.val[M.val.size() / 5] = 2; M.val[M.val.size() / 6] = 3;
        EncodeStats st;
        roundtrip(M, r1, "values up to 2^32-1", &st);
        CHECK(st.value_type == ValueType::U32, "value type u32 for big values");
    }
    {   // m = 2^21: codec 1 refuses, Auto falls back to codec 2
        Mat M = random_mat(1u << 21, 20, 5, 13, 5);
        bool threw = false;
        try { encode_matrix(M.csc(), r1); } catch (const Error& e) { threw = e.code == Errc::Unsupported; }
        CHECK(threw, "m = 2^21 rejected by codec 1");
        EncodeStats st;
        roundtrip(M, NN(), "m=2^21 auto", &st, false);
        CHECK(st.codec == CodecId::Generic, "m = 2^21 falls back to codec 2");
    }
    {   // N < lanes, N = lanes, N = lanes + 1, lots of empty columns
        for (uint32_t k : {1u, 2u, 31u, 32u, 33u, 63u, 64u, 65u, 1000u}) {
            Mat M;
            M.m = 2000;
            for (uint32_t j = 0; j < 3; ++j) M.add_col({});
            std::vector<std::pair<uint32_t, uint32_t>> col;
            for (uint32_t i = 0; i < k; ++i) col.push_back({i * 2, 1 + i % 3});
            M.add_col(col);
            M.add_col({});
            roundtrip(M, r1, "N=" + std::to_string(k), nullptr, false);
        }
    }

    // ---- option grid --------------------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(3000, 400, 25, 21, 12);
        for (int lanes : {16, 32, 64})
            for (int pb : {8, 11, 12})
                for (int kind : {-1, 0, 2, 4}) {
                    EncodeOptions o = r1;
                    o.lanes = uint8_t(lanes); o.prob_bits = uint8_t(pb); o.value_prob_bits = uint8_t(pb == 8 ? 9 : pb); o.ctx_kind = kind;
                    roundtrip(M, o, "lanes" + std::to_string(lanes) + " pb" + std::to_string(pb) + " kind" + std::to_string(kind), nullptr, false);
                }
        for (int cpo : {0, 1, 2, 8}) { EncodeOptions o = r1; o.classes_per_octave = uint8_t(cpo); roundtrip(M, o, "cpo" + std::to_string(cpo), nullptr, false); }
        for (uint32_t cc : {1u, 2u, 3u, 7u, 100u, 399u, 400u, 401u, 100000u}) { EncodeOptions o = r1; o.chunk_cols = cc; roundtrip(M, o, "chunk_cols" + std::to_string(cc), nullptr, false); }
        EncodeOptions o2; o2.codec = CodecId::Generic; o2.chunk_cols = 77; roundtrip(M, o2, "codec2 chunk77", nullptr, false);
        // per-chunk non-zero budget: round trip, and the geometry it promises (<= chunk_cols columns; <= budget non-zeros unless the
        // chunk is one column; a chunk ends only where the next column would not fit, or at chunk_cols, or at the end)
        for (CodecId codec : {CodecId::Rans, CodecId::Generic})
            for (uint32_t bud : {1u, 60u, 700u, 5000u, 0u, 0xFFFFFFFFu}) {
                EncodeOptions o = r1; o.codec = codec; o.chunk_cols = 64; o.chunk_nnz_max = bud;
                const std::string nm = std::string(codec == CodecId::Rans ? "rans" : "generic") + " chunk_nnz_max" + std::to_string(bud);
                roundtrip(M, o, nm, nullptr, false);
                o.row_names.kind = NamesKind::None; o.col_names.kind = NamesKind::None; o.compute_digest = false;
                auto b = encode_matrix(M.csc(), o);
                ByteSource src(b);
                auto mx = Matrix::open(src);
                const ChunkIndex& ci = mx.chunk_index();
                const uint64_t lim = bud == 0xFFFFFFFFu ? UINT64_MAX : (bud ? bud : kDefaultChunkNnzMax);
                bool ok = true;
                for (size_t k = 0; k < ci.n_chunks(); ++k) {
                    const uint32_t nc = ci.cols_in_chunk(k), j0 = ci.rec[k].first_col, j1 = j0 + nc;
                    const uint64_t z = ci.nnz_in_chunk(k);
                    if (nc > 64 || (nc > 1 && z > lim)) ok = false;
                    // greedy: the next column would not have fitted (unless the chunk is at its column cap or the end)
                    if (j1 < M.n && nc < 64 && lim != UINT64_MAX && z + (M.ptr[j1 + 1] - M.ptr[j1]) <= lim) ok = false;
                }
                CHECK(ok, nm + ": chunk geometry");
                if (bud == 0xFFFFFFFFu) {
                    bool fixed = true;
                    for (size_t k = 0; k + 1 < ci.n_chunks(); ++k) fixed = fixed && ci.cols_in_chunk(k) == 64;
                    CHECK(fixed, nm + ": no budget gives fixed-width chunks");
                }
            }
    }
    {   // encoder output does not depend on the thread count
        Mat M = random_mat(8000, 1500, 30, 22, 20);
        EncodeOptions o = r1;
        o.chunk_cols = 100; o.threads = 1;
        auto b1 = encode_matrix(M.csc(), o);
        o.threads = 4;
        auto b4 = encode_matrix(M.csc(), o);
        CHECK(b1 == b4, "encode is thread-count independent");
        o.threads = 1;
        CHECK(encode_matrix(M.csc(), o) == b1, "encode is deterministic");
    }

    // ---- zero values and f32 --------------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(500, 60, 10, 31, 5);
        M.val[3] = 0;
        EncodeStats st;
        roundtrip(M, r1, "explicit zero, codec 1", &st);   // value 0 is a first-class value in codec 1 (flag = value != 1)
        CHECK(st.codec == CodecId::Rans, "explicit zero stays in codec 1");
        // psi-like: half zeros, half ones, a few big values
        Mat P = random_mat(3000, 400, 30, 33, 3, true);
        std::mt19937 g(5);
        for (auto& v : P.val) { uint32_t x = g() % 8; if (x < 3) v = 0; else if (x < 6) v = 1; }
        roundtrip(P, r1, "zeros and ones with big values", nullptr, false);
        Mat Z = P;
        for (auto& v : Z.val) v = 0;
        roundtrip(Z, r1, "all zero values", nullptr, false);
    }
    {
        Mat M = random_mat(800, 90, 12, 32, 30);
        std::vector<float> f(M.val.begin(), M.val.end());   // integral floats
        CSC c = M.csc();
        c.data = f.data(); c.data_type = DType::F32;
        EncodeOptions o;
        o.row_names.blob = join_names(make_names("g", M.m)); o.col_names.blob = join_names(make_names("c", M.n));
        o.codec = CodecId::Rans;
        auto bytes = encode_matrix(c, o);
        ByteSource src(bytes);
        Matrix mx = Matrix::open(src);
        CHECK(mx.header().value_type == ValueType::F32 && (mx.header().flags & 1) && mx.header().codec == CodecId::Rans, "integral f32 -> codec 1 with the integral flag");
        std::vector<uint32_t> ip(M.n + 1), ix(M.nnz());
        std::vector<float> fv(M.nnz());
        std::vector<double> dv(M.nnz());
        std::vector<uint32_t> uv(M.nnz());
        Output o1;
        o1.indptr = ip.data(); o1.indptr_capacity = ip.size(); o1.indices = ix.data(); o1.indices_capacity = ix.size(); o1.data = fv.data(); o1.data_type = DType::F32; o1.data_capacity = fv.size();
        DecodeOptions d;
        d.mode = Mode::Canonical;
        mx.decode(src, o1, d);
        CHECK(fv == f && ix == M.idx, "integral f32 round trip");
        o1.data = dv.data(); o1.data_type = DType::F64; o1.data_capacity = dv.size();
        mx.decode(src, o1, d);
        bool ok = true;
        for (size_t i = 0; i < dv.size(); ++i) ok = ok && dv[i] == double(f[i]);
        CHECK(ok, "f32 -> f64");
        o1.data = uv.data(); o1.data_type = DType::U32; o1.data_capacity = uv.size();
        mx.decode(src, o1, d);
        ok = true;
        for (size_t i = 0; i < uv.size(); ++i) { uint32_t b; std::memcpy(&b, &f[i], 4); ok = ok && uv[i] == b; }
        CHECK(ok, "f32 -> u32 bit patterns");
        o1.data = uv.data(); o1.data_type = DType::U8;
        bool threw = false;
        try { mx.decode(src, o1, d); } catch (const Error& e) { threw = e.code == Errc::TypeMismatch; }
        CHECK(threw, "f32 matrix cannot decode to u8");
        CHECK(canonical_digest(c, ValueType::F32, o.row_names.blob, o.col_names.blob, {}) == mx.header().digest, "f32 digest");

        // non-integral floats, -0.0 and NaN need codec 2 and round trip bit for bit
        std::vector<float> g2(f);
        g2[0] = 0.5f; g2[1] = -0.0f; g2[2] = std::numeric_limits<float>::quiet_NaN(); g2[3] = 1e30f; g2[4] = -3.25f;
        c.data = g2.data();
        o.codec = CodecId::Auto;
        auto b2 = encode_matrix(c, o);
        ByteSource s2(b2);
        Matrix m2 = Matrix::open(s2);
        CHECK(m2.header().codec == CodecId::Generic && !(m2.header().flags & 1), "non-integral f32 -> codec 2 raw bits");
        o1.data = fv.data(); o1.data_type = DType::F32; o1.data_capacity = fv.size();
        m2.decode(s2, o1, d);
        CHECK(std::memcmp(fv.data(), g2.data(), g2.size() * 4) == 0, "non-integral f32 round trip (bit exact)");
        threw = false;
        try { EncodeOptions oo = o; oo.codec = CodecId::Rans; encode_matrix(c, oo); } catch (const Error& e) { threw = e.code == Errc::Unsupported; }
        CHECK(threw, "codec 1 refuses non-integral f32");
    }

    // ---- value types / output types ---------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(900, 120, 15, 41, 200);
        std::vector<uint8_t> v8(M.val.begin(), M.val.end());
        CSC c = M.csc();
        c.data = v8.data(); c.data_type = DType::U8;
        auto bytes = encode_matrix(c, NN());
        ByteSource src(bytes);
        Matrix mx = Matrix::open(src);
        CHECK(mx.header().value_type == ValueType::U8, "u8 input -> u8 value type");
        // all output integer widths
        std::vector<uint64_t> ip64(M.n + 1);
        std::vector<int32_t> i32(M.nnz());
        std::vector<uint8_t> d8(M.nnz());
        std::vector<uint16_t> d16(M.nnz());
        std::vector<float> df(M.nnz());
        std::vector<int32_t> di(M.nnz());
        DecodeOptions d;
        d.mode = Mode::Canonical;
        Output o;
        o.indptr = ip64.data(); o.indptr_type = IntType::U64; o.indptr_capacity = ip64.size();
        o.indices = i32.data(); o.indices_type = IntType::I32; o.indices_capacity = i32.size();
        o.data = d8.data(); o.data_type = DType::U8; o.data_capacity = d8.size();
        mx.decode(src, o, d);
        bool ok = true;
        for (size_t i = 0; i < M.ptr.size(); ++i) ok = ok && ip64[i] == M.ptr[i];
        for (size_t i = 0; i < M.nnz(); ++i) ok = ok && uint32_t(i32[i]) == M.idx[i] && d8[i] == M.val[i];
        CHECK(ok, "u64 indptr / i32 indices / u8 data");
        o.data = d16.data(); o.data_type = DType::U16; o.data_capacity = d16.size();
        mx.decode(src, o, d);
        ok = true; for (size_t i = 0; i < M.nnz(); ++i) ok = ok && d16[i] == M.val[i];
        CHECK(ok, "u16 data");
        o.data = df.data(); o.data_type = DType::F32; o.data_capacity = df.size();
        mx.decode(src, o, d);
        ok = true; for (size_t i = 0; i < M.nnz(); ++i) ok = ok && df[i] == float(M.val[i]);
        CHECK(ok, "f32 data from an integer matrix");
        o.data = di.data(); o.data_type = DType::I32; o.data_capacity = di.size();
        mx.decode(src, o, d);
        ok = true; for (size_t i = 0; i < M.nnz(); ++i) ok = ok && di[i] == int32_t(M.val[i]);
        CHECK(ok, "i32 data");
        // capacity and range errors
        bool threw = false;
        o.data = d8.data(); o.data_type = DType::U8; o.data_capacity = 3;
        try { mx.decode(src, o, d); } catch (const Error& e) { threw = e.code == Errc::OutputTooSmall; }
        CHECK(threw, "OutputTooSmall");
        // a matrix whose values need u16 cannot decode to u8
        Mat B = random_mat(900, 120, 15, 42, 200000, true);
        auto bb = encode_matrix(B.csc(), NN());
        ByteSource sb(bb);
        Matrix mb = Matrix::open(sb);
        std::vector<uint8_t> d8b(B.nnz());
        std::vector<uint32_t> ipb(B.n + 1), ixb(B.nnz());
        Output ob;
        ob.indptr = ipb.data(); ob.indptr_capacity = ipb.size(); ob.indices = ixb.data(); ob.indices_capacity = ixb.size(); ob.data = d8b.data(); ob.data_type = DType::U8; ob.data_capacity = d8b.size();
        threw = false;
        try { mb.decode(sb, ob, d); } catch (const Error& e) { threw = e.code == Errc::TypeMismatch; }
        CHECK(threw, "TypeMismatch when values exceed u8");
        // explicit value_type smaller than needed
        EncodeOptions bad = NN();
        bad.value_type = ValueType::U8;
        threw = false;
        try { encode_matrix(B.csc(), bad); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
        CHECK(threw, "value_type U8 rejected for big values");
        // legacy: u32 storage type kept although values are small
        EncodeOptions keep = NN();
        keep.value_type = ValueType::U32;
        auto kb = encode_matrix(M.csc(), keep);
        CHECK(read_header(kb.data(), kb.size()).value_type == ValueType::U32, "explicit u32 value type is kept");
    }

    // ---- names kinds, kv, digest options -------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(400, 70, 8, 51, 5);
        EncodeOptions o;
        o.row_names.kind = NamesKind::Pack;
        o.row_names.pack_id = PackId{{1, 2, 3, 4, 5, 6, 7, 8}};
        o.row_names.blob = join_names(make_names("g", M.m));   // logical names, used for the digest only
        o.col_names.kind = NamesKind::Cells;
        o.col_names.blob = join_names(make_names("c", M.n));
        o.kv = {{"alpha", "1"}, {"zeta", "value with spaces"}, {"b", ""}};
        auto bytes = encode_matrix(M.csc(), o);
        ByteSource src(bytes);
        Matrix mx = Matrix::open(src);
        NamesSpec rn = mx.row_names(src), cn = mx.col_names(src);
        CHECK(rn.kind == NamesKind::Pack && rn.pack_id == o.row_names.pack_id && rn.blob.empty() && rn.count == M.m, "Pack row names");
        CHECK(cn.kind == NamesKind::Cells && cn.count == M.n && cn.blob.empty(), "Cells column names");
        CHECK(mx.header().col_names_id == (PackId{{'C', 'E', 'L', 'L', 'S', 0, 0, 0}}), "CELLS marker in the header");
        CHECK(mx.kv(src) == o.kv, "kv round trip");
        CHECK(canonical_digest(M.csc(), mx.header().value_type, o.row_names.blob, o.col_names.blob, o.kv) == mx.header().digest, "digest from logical names");
        // the digest changes with a value, a name, a kv entry
        Mat M2 = M; M2.val[0] += 1;
        CHECK(canonical_digest(M2.csc(), mx.header().value_type, o.row_names.blob, o.col_names.blob, o.kv) != mx.header().digest, "digest sensitive to values");
        auto kv2 = o.kv; kv2["alpha"] = "2";
        CHECK(canonical_digest(M.csc(), mx.header().value_type, o.row_names.blob, o.col_names.blob, kv2) != mx.header().digest, "digest sensitive to kv");
        auto rn2 = o.row_names.blob; rn2[0] = 'h';
        CHECK(canonical_digest(M.csc(), mx.header().value_type, rn2, o.col_names.blob, o.kv) != mx.header().digest, "digest sensitive to names");

        EncodeOptions s;   // PackSubset columns, inline rows
        s.col_names.kind = NamesKind::PackSubset;
        s.col_names.pack_id = PackId{{9, 9, 9, 9, 9, 9, 9, 9}};
        std::mt19937 g(3);
        for (uint32_t j = 0; j < M.n; ++j) s.col_names.pack_index.push_back(uint32_t(g() % 3000000));
        s.row_names.blob = join_names(make_names("g", M.m));
        s.compute_digest = false;
        auto sb = encode_matrix(M.csc(), s);
        ByteSource ss(sb);
        Matrix ms = Matrix::open(ss);
        CHECK(ms.col_names(ss).pack_index == s.col_names.pack_index && ms.col_names(ss).kind == NamesKind::PackSubset, "PackSubset column names");
        CHECK(ms.header().digest.is_zero(), "compute_digest=false leaves a zero digest");
        s.has_digest = true; s.digest.b[0] = 0x5A;
        CHECK(read_header(encode_matrix(M.csc(), s).data(), 256).digest.b[0] == 0x5A, "caller-supplied digest is stored");
        EncodeOptions need;   // digest requested but logical names missing
        need.col_names.kind = NamesKind::Cells;
        bool threw = false;
        try { encode_matrix(M.csc(), need); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
        CHECK(threw, "digest without logical names is an error");
    }

    // ---- partial sources -------------------------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(2000, 900, 20, 61, 8);
        EncodeOptions o = r1;
        o.chunk_cols = 64;
        auto bytes = roundtrip(M, o, "partial base", nullptr, false);
        Header h = read_header(bytes.data(), bytes.size());
        ByteSource part;
        part.add(0, bytes.data(), size_t(h.prefix_len()));
        Matrix pm = Matrix::open(part);
        ColumnPlan pl = pm.plan_columns(130, 300);
        std::vector<uint32_t> ip(171), ix(pl.max_nnz), dv(pl.max_nnz);
        Output out;
        out.indptr = ip.data(); out.indptr_capacity = ip.size(); out.indices = ix.data(); out.indices_capacity = ix.size(); out.data = dv.data(); out.data_capacity = dv.size();
        bool threw = false;
        try { pm.decode_columns(part, 130, 300, out); } catch (const Error& e) { threw = e.code == Errc::NeedRange && e.need_off == pl.byte_off && e.need_len == pl.byte_len; }
        CHECK(threw, "decode_columns reports the missing chunk run");
        part.add(pl.byte_off, bytes.data() + pl.byte_off, size_t(pl.byte_len));
        DecodeOptions d;
        d.mode = Mode::Canonical;
        uint64_t nz = pm.decode_columns(part, 130, 300, out, d);
        bool ok = nz == M.ptr[300] - M.ptr[130];
        for (uint64_t t = 0; t < nz && ok; ++t) ok = ix[t] == M.idx[M.ptr[130] + t] && dv[t] == M.val[M.ptr[130] + t];
        CHECK(ok, "decode_columns from the prefix and one chunk run");
        threw = false;
        std::vector<uint32_t> fip(M.n + 1), fix(M.nnz()), fdv(M.nnz());
        Output fo;
        fo.indptr = fip.data(); fo.indptr_capacity = fip.size(); fo.indices = fix.data(); fo.indices_capacity = fix.size(); fo.data = fdv.data(); fo.data_capacity = fdv.size();
        try { pm.decode(part, fo); } catch (const Error& e) { threw = e.code == Errc::NeedRange; }
        CHECK(threw, "whole-matrix decode from a partial source reports NeedRange");
        // the last chunk sits at the very end of a region: must decode without reading past it
        ColumnPlan last = pm.plan_columns(M.n - 3, M.n);
        ByteSource only;
        only.add(0, bytes.data(), size_t(h.prefix_len()));
        std::vector<uint8_t> copy(bytes.begin() + long(last.byte_off), bytes.begin() + long(last.byte_off + last.byte_len));   // exact-size heap copy
        only.add(last.byte_off, copy.data(), copy.size());
        std::vector<uint32_t> ip3(4), ix3(last.max_nnz), dv3(last.max_nnz);
        Output o3;
        o3.indptr = ip3.data(); o3.indptr_capacity = 4; o3.indices = ix3.data(); o3.indices_capacity = ix3.size(); o3.data = dv3.data(); o3.data_capacity = dv3.size();
        nz = pm.decode_columns(only, M.n - 3, M.n, o3, d);
        CHECK(nz == M.ptr[M.n] - M.ptr[M.n - 3], "last chunk at the end of a region");
    }

    // ---- corruption must never crash ---------------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(1500, 300, 20, 71, 9, true);
        EncodeOptions o = r1;
        o.chunk_cols = 50;
        auto good = encode_matrix(M.csc(), o);
        std::mt19937 g(99);
        int detected = 0, total = 0;
        for (int it = 0; it < 400; ++it) {
            auto bad = good;
            int flips = 1 + int(g() % 3);
            for (int f = 0; f < flips; ++f) bad[g() % bad.size()] ^= uint8_t(1u << (g() % 8));
            ++total;
            try {
                ByteSource src(bad);
                Matrix mx = Matrix::open(src);
                std::vector<uint32_t> ip(M.n + 1), ix(M.nnz()), dv(M.nnz());
                Output out;
                out.indptr = ip.data(); out.indptr_capacity = ip.size(); out.indices = ix.data(); out.indices_capacity = ix.size(); out.data = dv.data(); out.data_capacity = dv.size();
                DecodeOptions d;
                d.mode = (it & 1) ? Mode::Canonical : Mode::Native;
                d.isa = (it % 3 == 0) ? Isa::Scalar : Isa::Auto;
                mx.decode(src, out, d);
                if (ip != M.ptr || ix != M.idx || dv != M.val) ++detected;   // silently different output is the "undetected" outcome for the rANS body
                (void)detected;
            } catch (const Error&) { ++detected; }
        }
        std::printf("  corruption: %d / %d mutated members reported an error or decoded differently without crashing\n", detected, total);
        CHECK(true, "corruption survived");
        // truncation
        for (size_t cut : {size_t(0), size_t(10), size_t(255), size_t(256), good.size() / 2, good.size() - 1}) {
            bool threw = false;
            try { ByteSource src(good.data(), cut); Matrix mx = Matrix::open(src); std::vector<uint32_t> ip(M.n + 1), ix(M.nnz()), dv(M.nnz()); Output out; out.indptr = ip.data(); out.indptr_capacity = ip.size(); out.indices = ix.data(); out.indices_capacity = ix.size(); out.data = dv.data(); out.data_capacity = dv.size(); mx.decode(src, out); }
            catch (const Error&) { threw = true; }
            CHECK(threw, "truncated member at " + std::to_string(cut) + " bytes is an error");
        }
    }

    // ---- bad arguments -------------------------------------------------------------------------------------------------------------------------
    {
        Mat M = random_mat(100, 10, 5, 81, 3);
        Mat bad = M;
        bad.idx[1] = bad.idx[0];   // not strictly ascending
        bool threw = false;
        try { encode_matrix(bad.csc(), NN()); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
        CHECK(threw, "non-ascending row ids rejected");
        Mat oob = M;
        oob.idx[0] = 100;
        threw = false;
        try { encode_matrix(oob.csc(), NN()); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
        CHECK(threw, "row id >= m rejected");
        EncodeOptions o;
        o.lanes = 24;
        threw = false;
        try { encode_matrix(M.csc(), o); } catch (const Error& e) { threw = e.code == Errc::BadArgument; }
        CHECK(threw, "bad lane count rejected");
        std::vector<uint8_t> junk(300, 7);
        threw = false;
        try { read_header(junk.data(), junk.size()); } catch (const Error& e) { threw = e.code == Errc::BadMagic; }
        CHECK(threw, "bad magic");
    }

    std::printf("test_pz2_edge: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
