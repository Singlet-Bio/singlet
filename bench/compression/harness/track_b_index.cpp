// Track B index-coding survey: bits/nnz (raw / +zstd3 / +zstd19) and decode Mnnz/s for each row-index coder, in rank space
// (frequency order, as the lane codec) and in original row order.
//   bin/track_b_index.exe [--out results/track_b_index.tsv] file.1pz ...
#include "common.h"
#include "track_b_util.h"
#include "track_b_index.h"
#include <algorithm>
#include <iostream>

using namespace cs;
using namespace cs::tb;

static double calib_ghz() {
    uint64_t s = 0;
    Timer t;
    for (uint64_t i = 0; i < 20000000ull; ++i) {
        asm volatile("add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0" : "+r"(s));
    }
    return 2e8 / t.ms() / 1e6;
}

int main(int argc, char** argv) {
    std::string out = "results/track_b_index.tsv";
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) out = argv[++i];
        else files.push_back(a);
    }
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tspace\tcoder\tstage\tnnz\tbytes\tidx_bits_nnz\tdec_ms\tdec_Mnnz_s\tcore_GHz\tcyc_per_nnz\tok\n";
    std::vector<IdxCoder> coders = {coder_leb(), coder_bytesplit(), coder_ef(), coder_bic(), coder_roaring(), coder_headbm(2048), coder_headbm(8192)};
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        std::string sample = f;
        {
            auto s = f.find_last_of("/\\");
            auto t = f.find_last_of("/\\", s - 1);
            sample = f.substr(t + 1, s - t - 1) + "/" + f.substr(s + 1);
        }
        RowPerm P = make_perm(M);
        RankMat R = to_rank_space(M, P);
        const double nnz = (double)M.nnz;
        std::printf("== %s m=%u n=%u nnz=%llu\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz);
        for (int space = 0; space < 2; ++space) {
            const uint32_t* idx = space == 0 ? R.rk.data() : M.indices.data();
            const char* sn = space == 0 ? "rank" : "orig";
            for (auto& cd : coders) {
                if (space == 1 && cd.name.rfind("headbm", 0) == 0) continue;
                const uint32_t nchunks = (M.n + 1023) / 1024;
                std::vector<std::vector<uint8_t>> raw(nchunks);
                for (uint32_t c = 0; c < nchunks; ++c) cd.enc(M.indptr.data(), idx, c * 1024, std::min(M.n, (c + 1) * 1024), M.m, raw[c]);
                for (int stage = 0; stage < 3; ++stage) {
                    const int level = stage == 0 ? 0 : (stage == 1 ? 3 : 19);
                    std::vector<std::vector<uint8_t>> st(nchunks);
                    std::vector<size_t> rawlen(nchunks);
                    size_t total = 0;
                    for (uint32_t c = 0; c < nchunks; ++c) {
                        rawlen[c] = raw[c].size();
                        if (level == 0) st[c] = raw[c];
                        else {
                            st[c].resize(ZSTD_compressBound(raw[c].size()));
                            st[c].resize(ZSTD_compress(st[c].data(), st[c].size(), raw[c].data(), raw[c].size(), level));
                        }
                        total += st[c].size();
                    }
                    double ghz0 = calib_ghz();
                    std::vector<uint32_t> o(M.nnz + 64);
                    std::vector<uint8_t> buf;
                    double best = 1e300;
                    bool ok = true;
                    for (int rep = 0; rep < 3; ++rep) {
                        Timer t;
                        for (uint32_t c = 0; c < nchunks; ++c) {
                            const uint8_t* src = st[c].data();
                            if (level != 0) {
                                buf.resize(rawlen[c] + 64);
                                ZSTD_decompress(buf.data(), rawlen[c], st[c].data(), st[c].size());
                                src = buf.data();
                            }
                            cd.dec(src, rawlen[c], M.indptr.data(), c * 1024, std::min(M.n, (c + 1) * 1024), M.m, o.data());
                        }
                        best = std::min(best, t.ms());
                    }
                    for (size_t i = 0; i < M.nnz; ++i) if (o[i] != idx[i]) { ok = false; break; }
                    double ghz = std::max(ghz0, calib_ghz());
                    const char* stn = stage == 0 ? "raw" : (stage == 1 ? "zstd3" : "zstd19");
                    std::printf("  %-4s %-18s %-6s %9zu B  %6.2f idx bits/nnz  dec %8.2f ms  %7.1f Mnnz/s (%6.1f cyc/nnz @%.2fGHz) %s\n", sn, cd.name.c_str(), stn,
                                total, total * 8.0 / nnz, best, nnz / best / 1e3, best * ghz * 1e6 / nnz, ghz, ok ? "OK" : "FAIL");
                    tsv << sample << '\t' << sn << '\t' << cd.name << '\t' << stn << '\t' << M.nnz << '\t' << total << '\t' << total * 8.0 / nnz << '\t' << best
                        << '\t' << nnz / best / 1e3 << '\t' << ghz << '\t' << best * ghz * 1e6 / nnz << '\t' << (ok ? 1 : 0) << '\n';
                    tsv.flush();
                }
            }
        }
    }
}
