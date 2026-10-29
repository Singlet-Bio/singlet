// quick synthetic round-trip while developing (not a shipped test)
#include <singlet/pz2/pz2.h>
#include <cstdio>
#include <random>
using namespace singlet::pz2;
struct Mat { uint32_t m, n; std::vector<uint32_t> ptr, idx, val; };
static Mat make(uint32_t m, uint32_t n, double dens, uint32_t seed, uint32_t maxval) {
    std::mt19937_64 g(seed);
    Mat M; M.m = m; M.n = n; M.ptr.push_back(0);
    // gene popularity ~ zipf-ish
    std::vector<double> w(m); for (uint32_t i = 0; i < m; ++i) w[i] = 1.0 / (1.0 + (g() % 1000) * 0.01 * (i % 97 + 1));
    for (uint32_t j = 0; j < n; ++j) {
        double d = dens * (0.2 + (g() % 1000) / 500.0);
        for (uint32_t i = 0; i < m; ++i) {
            double p = d * w[i] * 5; if (p > 1) p = 1;
            if ((g() % 1000000) < p * 1e6) { M.idx.push_back(i); uint32_t v = 1 + (g() % 4 == 0 ? g() % maxval : 0); M.val.push_back(v); }
        }
        M.ptr.push_back((uint32_t)M.idx.size());
    }
    return M;
}
static int run(const Mat& M, const char* name, EncodeOptions opt = EncodeOptions()) {
    CSC c; c.m = M.m; c.n = M.n; c.nnz = M.idx.size(); c.indptr = M.ptr.data(); c.indices = M.idx.data(); c.data = M.val.data(); c.data_type = DType::U32;
    std::vector<std::string> rn, cn; for (uint32_t i = 0; i < M.m; ++i) rn.push_back("g" + std::to_string(i)); for (uint32_t j = 0; j < M.n; ++j) cn.push_back("c" + std::to_string(j));
    opt.row_names.blob = join_names(rn); opt.col_names.blob = join_names(cn);
    EncodeStats st;
    std::printf("[%s] encoding\n", name);
    auto bytes = encode_matrix(c, opt, &st);
    std::printf("[%s] encoded %zu bytes\n", name, bytes.size());
    int bad = 0;
    for (Isa isa : {Isa::Scalar, Isa::Avx2, Isa::Avx512}) {
        if (!isa_supported(isa)) continue;
        for (Mode mode : {Mode::Native, Mode::Canonical}) {
            ByteSource src(bytes); Matrix mx = Matrix::open(src);
            std::vector<uint32_t> ip(M.n + 1), ix(c.nnz + 1), dv(c.nnz + 1);
            Output o; o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
            DecodeOptions d; d.mode = mode; d.isa = isa;
            std::printf("[%s] decode isa=%s mode=%d\n", name, isa_name(isa), (int)mode);
            mx.decode(src, o, d);
            bool ok = ip == std::vector<uint32_t>(M.ptr.begin(), M.ptr.end());
            if (mode == Mode::Canonical) { ok = ok && std::equal(M.idx.begin(), M.idx.end(), ix.begin()) && std::equal(M.val.begin(), M.val.end(), dv.begin()); }
            else {
                // native: map through perm and compare as sets per column
                const auto& perm0 = mx.row_perm(); std::vector<uint32_t> idp; if (perm0.empty()) { idp.resize(M.m); for (uint32_t i = 0; i < M.m; ++i) idp[i] = i; } const auto& perm = perm0.empty() ? idp : perm0;
                for (uint32_t j = 0; j < M.n && ok; ++j) {
                    std::vector<std::pair<uint32_t,uint32_t>> a, b;
                    for (uint32_t t = M.ptr[j]; t < M.ptr[j+1]; ++t) { a.push_back({M.idx[t], M.val[t]}); b.push_back({perm[ix[t]], dv[t]}); }
                    // native ids ascending in stored space
                    for (uint32_t t = M.ptr[j] + 1; t < M.ptr[j+1]; ++t) if (ix[t] <= ix[t-1]) ok = false;
                    std::sort(b.begin(), b.end()); ok = ok && a == b;
                }
            }
            if (!ok) { std::printf("  FAIL %s isa=%s mode=%d\n", name, isa_name(isa), (int)mode); ++bad; }
        }
    }
    std::printf("%s: m=%u n=%u nnz=%zu bytes=%zu (%.3f bits/nnz) codec=%d chunk_cols=%u chunks=%u %s\n", name, M.m, M.n, M.idx.size(), bytes.size(),
                8.0 * bytes.size() / std::max<size_t>(1, M.idx.size()), (int)st.codec, st.chunk_cols, st.n_chunks, bad ? "FAILED" : "ok");
    return bad;
}
int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    int bad = 0;
    std::printf("%s\n", isa_report().c_str());
    bad += run(make(2000, 300, 0.02, 1, 5), "small");
    bad += run(make(30000, 600, 0.03, 2, 50), "mid");
    bad += run(make(30000, 2500, 0.004, 3, 300), "sparse");
    bad += run(make(500, 100, 0.2, 4, 1000000), "dense-bigvals");
    EncodeOptions o16; o16.lanes = 16; bad += run(make(5000, 700, 0.02, 5, 9), "lanes16", o16);
    EncodeOptions o64; o64.lanes = 64; bad += run(make(5000, 700, 0.02, 6, 9), "lanes64", o64);
    EncodeOptions oc; oc.codec = CodecId::Generic; bad += run(make(5000, 700, 0.02, 7, 9), "codec2", oc);
    std::printf("bad=%d\n", bad);
    return bad != 0;
}
