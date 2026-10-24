// w4_reader_native: C++-side timing of the production .1pz reader stages on a file (read-only use of pz_reader.h).
// Usage: w4_reader_native.exe file.1pz [reps]
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>
#include "singlet/pileup/sparse_accumulator.h"
#include "singlet/pileup/pz_reader.h"
using namespace singlet::pz;
using Clk = std::chrono::steady_clock;
static double ms(Clk::time_point a, Clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    int reps = argc > 2 ? atoi(argv[2]) : 5;
    std::ifstream f(argv[1], std::ios::binary | std::ios::ate);
    size_t sz = (size_t)f.tellg(); f.seekg(0);
    std::vector<uint8_t> buf(sz); f.read((char*)buf.data(), sz);
    ReadLimits lim;
    double best_total = 1e18, best_crc = 1e18, best_pp = 1e18, best_meta = 1e18, best_read = 1e18;
    size_t nnz = 0, m = 0, n = 0;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clk::now();
        ReadResult res = read_1pz_buffer(buf.data(), buf.size(), lim);
        auto t1 = Clk::now();
        best_total = std::min(best_total, ms(t0, t1));
        nnz = res.nnz; m = res.m; n = res.n;
        // file-level CRC32 (validate_buffer does this)
        t0 = Clk::now();
        PZHeader hdr = validate_buffer(buf.data(), buf.size(), lim);
        t1 = Clk::now();
        best_crc = std::min(best_crc, ms(t0, t1));
        // perm + ptr
        size_t off = sizeof(PZHeader);
        std::vector<uint32_t> perm, cc;
        t0 = Clk::now();
        decode_perm_and_ptr(buf.data(), off, hdr, perm, cc);
        t1 = Clk::now();
        best_pp = std::min(best_pp, ms(t0, t1));
        // names (metadata block): zstd + TLV parse into vector<string>
        std::vector<std::string> rn, cn; std::map<std::string, std::string> kv;
        t0 = Clk::now();
        decode_metadata_block(buf.data(), hdr, rn, cn, kv, lim);
        t1 = Clk::now();
        best_meta = std::min(best_meta, ms(t0, t1));
    }
    double chunks = best_total - best_crc - best_pp - best_meta;
    printf("file=%s bytes=%zu m=%zu n=%zu nnz=%zu reps=%d\n", argv[1], sz, m, n, nnz, reps);
    printf("native_total_ms=%.2f  file_crc32_ms=%.2f  perm+ptr_ms=%.2f  names_zstd+parse_ms=%.2f  chunks(rest)_ms=%.2f  chunks_Mnnz_s=%.1f\n",
           best_total, best_crc, best_pp, best_meta, chunks, nnz / chunks / 1e3);
    return 0;
}
