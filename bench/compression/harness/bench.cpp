// SPDX-License-Identifier: MIT
// Compression-study driver.
//
//   bin/bench.exe [--codec NAME[,NAME...]] [--out results/run.tsv] [--reps 3] [--no-file-check] file.1pz [...]
//
// For every (file, codec): encode, decode, verify matrix roundtrip (and file roundtrip unless
// disabled), time both, append one TSV row. Codecs come from kernels/registry.h.
#include "common.h"
#include "registry.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

static std::string base_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? p : p.substr(s + 1);
}
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1);
    size_t start = (t == std::string::npos) ? 0 : t + 1;
    return p.substr(start, s - start);
}

int main(int argc, char** argv) {
    std::vector<std::string> files, want;
    std::string out = "results/run.tsv";
    int reps = 3;
    bool file_check = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) want.push_back(tok);
        } else if (a == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else if (a == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (a == "--no-file-check") {
            file_check = false;
        } else if (a == "--list") {
            for (auto& c : all_codecs()) std::cout << c.name << "\n";
            return 0;
        } else {
            files.push_back(a);
        }
    }
    if (files.empty()) {
        std::cerr << "usage: bench [--codec a,b] [--out tsv] [--reps n] [--no-file-check] file.1pz...\n";
        return 2;
    }

    auto codecs = all_codecs();
    if (!want.empty()) {
        std::vector<Codec> sel;
        for (auto& c : codecs)
            if (std::find(want.begin(), want.end(), c.name) != want.end()) sel.push_back(c);
        codecs = sel;
    }

    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tratio_vs_1pz\tbits_per_nnz\tnnz\tm\tn\t"
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\n";

    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)\n", f.c_str(), M.m, M.n,
                    (unsigned long long)M.nnz, orig.size(), orig.size() * 8.0 / (double)M.nnz);
        for (auto& c : codecs) {
            Timer te;
            auto enc = c.encode(M);
            double enc_ms = te.ms();
            double dec_min = 1e300;
            CanonicalMatrix D;
            for (int r = 0; r < reps; ++r) {
                Timer td;
                D = c.decode(enc);
                dec_min = std::min(dec_min, td.ms());
            }
            bool mok = (D == M);
            bool fok = false;
            if (mok && file_check) {
                std::string tmp = out + ".tmp.1pz";
                auto re = rewrite_1pz(D, tmp);
                fok = (re == orig);
                std::remove(tmp.c_str());
            }
            double mnnz_s = (double)M.nnz / dec_min / 1e3;
            std::printf("  %-28s %10zu B  ratio %.3f  %6.2f bits/nnz  enc %8.1f ms  dec %8.1f ms (%.1f Mnnz/s)  %s%s\n",
                        c.name.c_str(), enc.size(), (double)enc.size() / (double)orig.size(),
                        enc.size() * 8.0 / (double)M.nnz, enc_ms, dec_min, mnnz_s,
                        mok ? "matrix=OK" : "matrix=FAIL",
                        file_check ? (fok ? " file=OK" : " file=FAIL") : "");
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << orig.size() << '\t'
                << enc.size() << '\t' << (double)enc.size() / (double)orig.size() << '\t'
                << enc.size() * 8.0 / (double)M.nnz << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t'
                << (file_check ? (fok ? 1 : 0) : -1) << '\t' << c.model_bytes << '\n';
        }
    }
    return 0;
}
