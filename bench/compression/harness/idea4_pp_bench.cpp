// SPDX-License-Identifier: MIT
// Idea 4 (Poisson-process variant) driver: a copy of harness/bench.cpp with a local codec list and extra detail columns.
//   IDEA4_MODEL_STATIC=... IDEA4_MODEL_NMF=... IDEA4_TAG=x bin/idea4_pp_bench.exe --calls --codec idea4_nmf_x --out results/idea4_pp_run.tsv f.1pz
// --calls : take the called/empty split from <dir of f.1pz>/cell_calls.tsv (the producer knows it; flags cost ~0.03 bit/column)
#include "common.h"
#include "ref_zstd_csc.h"
#include "idea4_pp_codec.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

namespace cs {
inline std::vector<Codec> all_codecs() {
    std::vector<Codec> v;
    v.push_back(make_ref_identity_1pz());
    v.push_back(make_ref_zstd_csc(3));
    ipp::register_idea4(v);
    return v;
}
}

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
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<std::string> files, want;
    std::string out = "results/idea4_pp_run.tsv";
    int reps = 3;
    bool file_check = true, use_calls = false;
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
        } else if (a == "--calls") {
            use_calls = true;
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
        std::cerr << "usage: idea4_pp_bench [--codec a,b] [--out tsv] [--reps n] [--calls] [--no-file-check] file.1pz...\n";
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
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\t"
               "payload_bytes\tpayload_bits_per_nnz\tnames_bytes\tcells\tamb\tnnz_cells\tnnz_amb\tbits_nnz_cells\tbits_nnz_amb\tside_bytes_per_cell\tmatvec_ms_per_cell\tdetail\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)\n", f.c_str(), M.m, M.n,
                    (unsigned long long)M.nnz, orig.size(), orig.size() * 8.0 / (double)M.nnz);
        if (use_calls) {
            auto sl = f.find_last_of("/\\");
            ipp::calls_path() = (sl == std::string::npos ? std::string(".") : f.substr(0, sl)) + "/cell_calls.tsv";
        }
        for (auto& c : codecs) {
            Timer te;
            auto enc = c.encode(M);
            double enc_ms = te.ms();
            double dec_min = 1e300, mv = 1e300;
            CanonicalMatrix D;
            double dec_cpu = 1e300;
            for (int r = 0; r < reps; ++r) {
                Timer td;
                double c0 = ipp::now_ms();
                D = c.decode(enc);
                double t = td.ms();
                double tc = ipp::now_ms() - c0;
                if (t < dec_min) dec_min = t;
                if (tc < dec_cpu) { dec_cpu = tc; mv = ipp::stats().dec_matvec_ms; }
            }
            bool mok = (D == M);
            if (!mok) {
                std::printf("  MISMATCH: m %d n %d nnz %d indptr %d indices %d data %d rn %d cn %d kv %d\n", D.m == M.m, D.n == M.n, D.nnz == M.nnz,
                            D.indptr == M.indptr, D.indices == M.indices, D.data == M.data, D.rownames == M.rownames, D.colnames == M.colnames, D.user_kv == M.user_kv);
            }
            bool fok = false;
            if (mok && file_check) {
                std::string tmp = out + ".tmp.1pz";
                auto re = rewrite_1pz(D, tmp);
                fok = (re == orig);
                std::remove(tmp.c_str());
            }
            double mnnz_s = (double)M.nnz / dec_cpu / 1e3;   // CPU-time based (the machine is shared)
            const auto& S = ipp::stats();
            bool is_i4 = c.name.rfind("idea4", 0) == 0;
            size_t payload = is_i4 ? S.rans_bytes : 0;
            double side_per_cell = (is_i4 && S.n_cell) ? (S.bits_theta + S.sc_cell) / 8.0 / S.n_cell : 0;
            double mvc = (is_i4 && S.dec_cells) ? mv / S.dec_cells : 0;
            std::printf("  %-28s %10zu B  ratio %.3f  %6.2f bits/nnz  enc %8.1f ms  dec %8.1f ms (%.1f Mnnz/s)  %s%s\n",
                        c.name.c_str(), enc.size(), (double)enc.size() / (double)orig.size(),
                        enc.size() * 8.0 / (double)M.nnz, enc_ms, dec_min, mnnz_s,
                        mok ? "matrix=OK" : "matrix=FAIL", file_check ? (fok ? " file=OK" : " file=FAIL") : "");
            if (is_i4) {
                std::printf("      decode shares of the column loop: gap(called) %.0f%%  amb-profile rebuild %.0f%%  gap(amb) %.0f%%\n", 100 * S.dec_share_gapc, 100 * S.dec_share_ambr, 100 * S.dec_share_gapa);
                std::printf("      payload %zu B = %.3f bits/nnz | names %zu B | side/cell %.2f B | matvec %.3f ms/cell | %s\n", payload,
                            payload * 8.0 / (double)M.nnz, S.names_bytes, side_per_cell, mvc, S.str().c_str());
            }
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << orig.size() << '\t'
                << enc.size() << '\t' << (double)enc.size() / (double)orig.size() << '\t'
                << enc.size() * 8.0 / (double)M.nnz << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t'
                << (file_check ? (fok ? 1 : 0) : -1) << '\t' << c.model_bytes << '\t'
                << payload << '\t' << payload * 8.0 / (double)M.nnz << '\t' << (is_i4 ? S.names_bytes : 0) << '\t'
                << (is_i4 ? S.n_cell : 0) << '\t' << (is_i4 ? S.n_amb : 0) << '\t'
                << (is_i4 ? S.nnz_cell : 0) << '\t' << (is_i4 ? S.nnz_amb : 0) << '\t'
                << (is_i4 && S.nnz_cell ? (S.bits_gap_cell + S.bits_cnt_cell + S.bits_theta + S.sc_cell) / S.nnz_cell : 0) << '\t'
                << (is_i4 && S.nnz_amb ? (S.bits_gap_amb + S.bits_cnt_amb + S.sc_amb) / S.nnz_amb : 0) << '\t'
                << side_per_cell << '\t' << mvc << '\t' << (is_i4 ? S.str() : std::string()) << '\n';
            tsv.flush();
        }
    }
    return 0;
}
