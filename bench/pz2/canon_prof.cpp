// Profile of the canonical mode: native decode vs the stored->canonical permutation and per-column sort, on a cached member.
//   canon_prof member.pz2
#include <singlet/pz2/pz2.h>

#include <chrono>
#include <cstdio>
#include <fstream>

using namespace singlet::pz2;
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    std::ifstream f(argv[1], std::ios::binary);
    std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    ByteSource src(bytes);
    Matrix mx = Matrix::open(src);
    const Header& h = mx.header();
    std::vector<uint32_t> ip(h.n + 1), ix(h.nnz + 64), dv(h.nnz + 64);
    Output o;
    o.indptr = ip.data(); o.indptr_capacity = ip.size(); o.indices = ix.data(); o.indices_capacity = ix.size(); o.data = dv.data(); o.data_capacity = dv.size();
    DecodeOptions d;
    d.mode = Mode::Native;
    mx.decode(src, o, d);
    std::vector<uint32_t> ix0 = ix, dv0 = dv;
    const uint32_t* perm = mx.row_perm().data();
    auto run = [&](const char* name, auto&& body) {
        double best = 1e9;
        for (int r = 0; r < 5; ++r) { ix = ix0; dv = dv0; double a = now_s(); body(); best = std::min(best, now_s() - a); }
        std::printf("%-34s %8.2f ms  %6.1f Mnnz/s\n", name, best * 1e3, double(h.nnz) / best / 1e6);
    };
    // 1: just the permutation (gather + bounds check), no sort
    run("perm gather only", [&] {
        uint32_t* idx = ix.data();
        for (uint64_t t = 0; t < h.nnz; ++t) { uint32_t s = idx[t]; if (s >= h.m) throw 1; idx[t] = perm[s]; }
    });
    // 2: full canonicalize per chunk of columns
    std::vector<uint32_t> k(h.n);
    for (uint32_t j = 0; j < h.n; ++j) k[j] = ip[j + 1] - ip[j];
    run("canonicalize (all columns)", [&] {
        detail::CanonWork w;
        std::vector<uint32_t> kk;
        for (uint32_t j0 = 0; j0 < h.n; j0 += 256) {
            uint32_t j1 = std::min<uint32_t>(h.n, j0 + 256);
            kk.assign(k.begin() + j0, k.begin() + j1);
            detail::canonicalize(perm, h.m, kk, reinterpret_cast<int32_t*>(ix.data()) + ip[j0], dv.data() + ip[j0], w);
        }
    });
    // 3: std::sort baseline on packed keys
    run("perm + std::sort packed u64", [&] {
        std::vector<uint64_t> tmp;
        for (uint32_t j = 0; j < h.n; ++j) {
            uint32_t a = ip[j], e = ip[j + 1];
            tmp.resize(e - a);
            for (uint32_t t = a; t < e; ++t) tmp[t - a] = (uint64_t(perm[ix[t]]) << 32) | dv[t];
            std::sort(tmp.begin(), tmp.end());
            for (uint32_t t = a; t < e; ++t) { ix[t] = uint32_t(tmp[t - a] >> 32); dv[t] = uint32_t(tmp[t - a]); }
        }
    });
    double mean_k = double(h.nnz) / std::max<uint32_t>(1, h.n);
    std::printf("m=%u n=%u nnz=%llu mean k=%.1f\n", h.m, h.n, (unsigned long long)h.nnz, mean_k);
    return 0;
}
