// w4_vocab: roundtrip + materialisation timing of an SVP0 pack.
//   w4_vocab_mat <pack.svp> [ref_exon.nul ref_intron.nul]      (reference files = NUL-terminated row names, see tools/w4_vocab_dump.py)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <random>
#include "w4_vocab.h"

using Clk = std::chrono::steady_clock;
static std::vector<uint8_t> slurp(const char* f) { std::ifstream i(f, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(i)), {}); }
template <class F> static double best_ms(int reps, F f) {
    double b = 1e30;
    for (int r = 0; r < reps; r++) { auto t0 = Clk::now(); f(); double ms = std::chrono::duration<double, std::milli>(Clk::now() - t0).count(); if (ms < b) b = ms; }
    return b;
}
int main(int argc, char** argv) {
    if (argc < 2) return 2;
    auto pk = slurp(argv[1]);
    w4v::Pack P;
    double t_dec = best_ms(7, [&] { P = w4v::decode(pk.data(), pk.size()); });
    double t_idx = best_ms(7, [&] { w4v::build_index(P); });
    std::vector<char> be, bi; std::vector<uint32_t> oe, oi;
    double t_ex = best_ms(7, [&] { w4v::materialise(P, false, be, oe); });
    double t_in = best_ms(7, [&] { w4v::materialise(P, true, bi, oi); });
    // lazy lookups
    std::mt19937 rng(1);
    size_t NE = P.n_exons(), NI = P.n_introns();
    std::vector<uint32_t> q(1 << 20);
    for (auto& x : q) x = rng();
    char buf[1024]; size_t sink = 0;
    double t_lazy_e = best_ms(5, [&] { for (uint32_t x : q) sink += w4v::name_at(P, false, x % NE, buf); }) * 1e6 / q.size();
    double t_lazy_i = best_ms(5, [&] { for (uint32_t x : q) sink += w4v::name_at(P, true, x % NI, buf); }) * 1e6 / q.size();
    int ok_e = -1, ok_i = -1, ok_lazy = -1;
    if (argc >= 4) {
        auto re = slurp(argv[2]), ri = slurp(argv[3]);
        ok_e = re.size() == be.size() && !std::memcmp(re.data(), be.data(), be.size());
        ok_i = ri.size() == bi.size() && !std::memcmp(ri.data(), bi.data(), bi.size());
        ok_lazy = 1;
        for (size_t t = 0; t < 20000 && ok_lazy; t++) {   // lazy path vs reference blob
            size_t r = rng() % NE; size_t n = w4v::name_at(P, false, r, buf);
            if (n != oe[r + 1] - oe[r] - 1 || std::memcmp(buf, be.data() + oe[r], n)) ok_lazy = 0;
            r = rng() % NI; n = w4v::name_at(P, true, r, buf);
            if (n != oi[r + 1] - oi[r] - 1 || std::memcmp(buf, bi.data() + oi[r], n)) ok_lazy = 0;
        }
    }
    printf("pack=%zu B genes=%zu exons=%zu introns=%zu | decode(zstd+varint) %.2f ms  index %.2f ms  eager exon %.2f ms (%.0f ns/name, %zu B)  eager intron %.2f ms (%.0f ns/name, %zu B)"
           " | cold total exon+intron %.2f ms | lazy exon %.0f ns/name  lazy intron %.0f ns/name | roundtrip exon=%d intron=%d lazy=%d (sink %zu)\n",
           pk.size(), P.n_genes(), NE, NI, t_dec, t_idx, t_ex, t_ex * 1e6 / NE, be.size(), t_in, t_in * 1e6 / NI, bi.size(),
           t_dec + t_idx + t_ex + t_in, t_lazy_e, t_lazy_i, ok_e, ok_i, ok_lazy, sink);
    return 0;
}
