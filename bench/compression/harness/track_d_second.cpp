// Track D: second-pass compression of arbitrary files (codec dumps) with zstd-22, xz -9e, brotli-11,
// bzip2 -9 and the Track D context-mixing coder. Appends rows to a TSV.
//   bin/track_d_second.exe [--methods zstd22,xz,brotli,bz2,cm] [--out results/track_d_second.tsv]
//                          [--threads N] [--cm-roundtrip] file...
// TSV columns: file  method  in_bytes  out_bytes  ratio  ms
#include "common.h"
#include "track_d.h"
#include <algorithm>
#include <atomic>
#include <brotli/encode.h>
#include <bzlib.h>
#include <iostream>
#include <lzma.h>
#include <mutex>
#include <sstream>
#include <thread>
#include <zstd.h>

using namespace cs;

static size_t m_zstd22(const std::vector<uint8_t>& in) {
    ZSTD_CCtx* c = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, 22);
    ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, 27);
    ZSTD_CCtx_setParameter(c, ZSTD_c_enableLongDistanceMatching, 1);
    std::vector<uint8_t> o(ZSTD_compressBound(in.size()));
    size_t z = ZSTD_compress2(c, o.data(), o.size(), in.data(), in.size());
    ZSTD_freeCCtx(c);
    if (ZSTD_isError(z)) throw std::runtime_error(ZSTD_getErrorName(z));
    return z;
}
static size_t m_xz(const std::vector<uint8_t>& in) {
    std::vector<uint8_t> o(lzma_stream_buffer_bound(in.size()));
    size_t pos = 0;
    lzma_ret r = lzma_easy_buffer_encode(9 | LZMA_PRESET_EXTREME, LZMA_CHECK_NONE, nullptr, in.data(), in.size(), o.data(), &pos, o.size());
    if (r != LZMA_OK) throw std::runtime_error("lzma failed");
    return pos;
}
static size_t m_brotli(const std::vector<uint8_t>& in) {
    size_t n = BrotliEncoderMaxCompressedSize(in.size());
    std::vector<uint8_t> o(n);
    if (!BrotliEncoderCompress(11, 24, BROTLI_MODE_GENERIC, in.size(), in.data(), &n, o.data())) throw std::runtime_error("brotli failed");
    return n;
}
static size_t m_bz2(const std::vector<uint8_t>& in) {
    unsigned int n = (unsigned int)(in.size() + in.size() / 100 + 1000);
    std::vector<char> o(n);
    int r = BZ2_bzBuffToBuffCompress(o.data(), &n, (char*)in.data(), (unsigned int)in.size(), 9, 0, 30);
    if (r != BZ_OK) throw std::runtime_error("bz2 failed");
    return n;
}

int main(int argc, char** argv) {
    std::vector<std::string> files, methods = {"zstd22", "xz", "brotli", "bz2", "cm"};
    std::string out = "results/track_d_second.tsv";
    int threads = 1;
    bool rt = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--methods" && i + 1 < argc) {
            methods.clear();
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) methods.push_back(t);
        } else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--cm-roundtrip") rt = true;
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "file\tmethod\tin_bytes\tout_bytes\tratio\tms\n";
    std::mutex mu;
    std::atomic<size_t> next{0};
    auto work = [&]() {
        for (;;) {
            size_t i = next++;
            if (i >= files.size()) return;
            auto in = read_file(files[i]);
            std::string base = files[i];
            auto s = base.find_last_of("/\\");
            if (s != std::string::npos) base = base.substr(s + 1);
            for (auto& m : methods) {
                Timer t;
                size_t o = 0;
                try {
                    if (m == "zstd22") o = m_zstd22(in);
                    else if (m == "xz") o = m_xz(in);
                    else if (m == "brotli") o = m_brotli(in);
                    else if (m == "bz2") o = m_bz2(in);
                    else if (m == "cm") {
                        auto z = dcm::compress(in.data(), in.size());
                        o = z.size();
                        if (rt && dcm::decompress(z.data(), z.size()) != in) std::fprintf(stderr, "CM MISMATCH %s\n", base.c_str());
                    } else continue;
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "%s %s: %s\n", base.c_str(), m.c_str(), e.what());
                    continue;
                }
                double ms = t.ms();
                std::lock_guard<std::mutex> g(mu);
                tsv << base << '\t' << m << '\t' << in.size() << '\t' << o << '\t' << (double)o / in.size() << '\t' << ms << '\n';
                tsv.flush();
                std::printf("%-60s %-7s %10zu -> %10zu  %.4f  %.0f ms\n", base.c_str(), m.c_str(), in.size(), o, (double)o / in.size(), ms);
                std::fflush(stdout);
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < threads; ++i) th.emplace_back(work);
    for (auto& t : th) t.join();
    return 0;
}
