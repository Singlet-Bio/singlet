// w3_total second pass: re-compress already-coded w3_total streams with zstd-19 and xz -9e (and report the ratio).
//   bin/w3_total_second.exe [--out results/w3_total_second.tsv] stream.bin ...
#include "common.h"
#include <lzma.h>
#include <zstd.h>
#include <cstdio>
using namespace cs;
int main(int argc, char** argv) {
    std::string out = "results/w3_total_second.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i]; else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "file\tin_bytes\tzstd19\tratio_zstd19\txz9e\tratio_xz9e\n";
    for (auto& f : files) {
        auto in = read_file(f);
        std::vector<uint8_t> o(ZSTD_compressBound(in.size()));
        size_t z = ZSTD_compress(o.data(), o.size(), in.data(), in.size(), 19);
        std::vector<uint8_t> x(lzma_stream_buffer_bound(in.size()));
        size_t pos = 0;
        lzma_ret r = lzma_easy_buffer_encode(9 | LZMA_PRESET_EXTREME, LZMA_CHECK_NONE, nullptr, in.data(), in.size(), x.data(), &pos, x.size());
        if (ZSTD_isError(z) || r != LZMA_OK) { std::fprintf(stderr, "failed %s\n", f.c_str()); continue; }
        std::printf("%-90s %10zu  zstd19 %10zu (%.4f)  xz9e %10zu (%.4f)\n", f.c_str(), in.size(), z, (double)z / in.size(), pos, (double)pos / in.size());
        tsv << f << '\t' << in.size() << '\t' << z << '\t' << (double)z / in.size() << '\t' << pos << '\t' << (double)pos / in.size() << '\n';
    }
    return 0;
}
