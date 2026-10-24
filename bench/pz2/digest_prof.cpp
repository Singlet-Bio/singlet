// digest_prof member.pz2 : time canonical_digest() on the decoded canonical matrix
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
    d.mode = Mode::Canonical;
    d.threads = 4;
    mx.decode(src, o, d);
    CSC c;
    c.m = h.m; c.n = h.n; c.nnz = h.nnz; c.indptr = ip.data(); c.indices = ix.data(); c.data = dv.data();
    double best = 1e9;
    Digest16 dg;
    for (int r = 0; r < 3; ++r) { double a = now_s(); dg = canonical_digest(c, h.value_type, {}, {}, {}); best = std::min(best, now_s() - a); }
    double mb = double(h.nnz) * 8 / 1e6;
    std::printf("digest %s over %.1f MB: %.1f ms = %.0f MB/s  (%.1f Mnnz/s)\n", dg.hex().c_str(), mb, best * 1e3, mb / best, double(h.nnz) / best / 1e6);
    return 0;
}
