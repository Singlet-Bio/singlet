// SPDX-License-Identifier: MIT
// E2 benchmark driver: copy of bench.cpp with a local codec registry (reference codecs + E2) and extra TSV columns
// (payload-only bits/nnz, class split, payload-only decode time, calibrated-clock throughput).
//   bin/e2_bench.exe [--codec a,b] [--out tsv] [--reps n] [--no-file-check] file.1pz ...
#include "common.h"
#include "ref_zstd_csc.h"
#include "e2_codec.h"
#ifdef E2_PROF_BUILD
#include "e2_prof.h"
#endif
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
    v.push_back(make_ref_zstd_csc(19));
    register_e2(v);
    return v;
}
}  // namespace cs

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
static std::string dir_of(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? "." : p.substr(0, s);
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<std::string> files, want;
    std::string out = "results/e2_run.tsv";
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
        std::cerr << "usage: e2_bench [--codec a,b] [--out tsv] [--reps n] [--no-file-check] file.1pz...\n";
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
               "names_bytes\tpayload_bits_per_nnz\thdr_bits_per_nnz\tcells_bits_per_nnz\tempt_bits_per_nnz\t"
               "blockhdr_bits_per_nnz\tpayload_dec_ms\tpayload_Mnnz_s\trefit_ms\tgather_ms\thdr_dec_ms\t"
               "pay_cyc_per_nnz\tcal_cyc_per_iter\tpay_Mnnz_s_norm\tesc_per_nnz\tcorr_per_nnz\ttag\n";
#ifdef E2_PROF_BUILD
    e2prof::start();
#endif
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        e2::g_calls_path = dir_of(f) + "/cell_calls.tsv";
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)\n", f.c_str(), M.m, M.n,
                    (unsigned long long)M.nnz, orig.size(), orig.size() * 8.0 / (double)M.nnz);
        for (auto& c : codecs) {
            bool is_e2 = c.name.rfind("e2", 0) == 0;
            Timer te;
            std::vector<uint8_t> enc;
            try { enc = c.encode(M); }
            catch (std::exception& e) { std::printf("  %-20s ENCODE ERROR: %s\n", c.name.c_str(), e.what()); continue; }
            double enc_ms = te.ms();
            e2::Report rep = e2::g_rep;
            if (is_e2) e2::g_rep.reset_dec();
            double dec_min = 1e300;
            CanonicalMatrix D;
            double cal = e2::calibrate_chain();
            for (int r = 0; r < reps; ++r) {
                e2::g_rep.nesc = 0; e2::g_rep.ncorr = 0;
                Timer td;
#ifdef E2_PROF_BUILD
                e2prof::active = true;
#endif
                D = c.decode(enc);
#ifdef E2_PROF_BUILD
                e2prof::active = false;
#endif
                dec_min = std::min(dec_min, td.ms());
            }
            cal = std::min(cal, e2::calibrate_chain());
            bool mok = (D == M);
            bool fok = false;
            if (mok && file_check) {
                std::string tmp = out + ".tmp.1pz";
                auto re = rewrite_1pz(D, tmp);
                fok = (re == orig);
                std::remove(tmp.c_str());
            }
            double nnz = (double)M.nnz;
            double mnnz_s = nnz / dec_min / 1e3;
            double names = is_e2 ? (double)rep.names_bytes : -1;
            double pay_bits = is_e2 ? (enc.size() - rep.names_bytes) * 8.0 / nnz : -1;
            double pay_ms = is_e2 ? e2::g_rep.dec_payload_ms : -1;
            double pay_mnnz = is_e2 ? nnz / pay_ms / 1e3 : -1;
            double pay_cyc = is_e2 ? e2::g_rep.dec_payload_cyc / nnz : -1;
            double pay_norm = is_e2 ? 3000.0 / (pay_cyc / (cal / 4.0)) : -1;
            double hdr_b = is_e2 ? rep.hdr_bytes * 8.0 / nnz : -1;
            double cells_b = is_e2 && rep.nnz_cells ? rep.cells_bytes * 8.0 / (double)rep.nnz_cells : -1;
            double empt_b = is_e2 && rep.nnz_empt ? rep.empt_bytes * 8.0 / (double)rep.nnz_empt : -1;
            std::printf("  %-14s %10zu B  %6.3f bits/nnz (payload %6.3f: hdr %5.3f cells %6.3f empt %6.3f)  enc %8.1f ms  dec %8.1f ms (%.1f Mnnz/s; payload-only %.1f ms = %.1f Mnnz/s, refit %.1f gather %.1f; %.0f cyc/nnz cal %.1f -> %.1f norm)  esc %.2e corr %.2e  %s%s\n",
                        c.name.c_str(), enc.size(), enc.size() * 8.0 / nnz, pay_bits, hdr_b, cells_b, empt_b, enc_ms, dec_min, mnnz_s,
                        pay_ms, pay_mnnz, is_e2 ? e2::g_rep.dec_refit_ms : 0.0, is_e2 ? e2::g_rep.dec_gather_ms : 0.0, pay_cyc, cal, pay_norm,
                        is_e2 ? (double)rep.nesc / nnz : 0.0, is_e2 ? (double)e2::g_rep.ncorr / nnz : 0.0,
                        mok ? "matrix=OK" : "matrix=FAIL", file_check ? (fok ? " file=OK" : " file=FAIL") : "");
            std::fflush(stdout);
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << orig.size() << '\t' << enc.size() << '\t'
                << (double)enc.size() / (double)orig.size() << '\t' << enc.size() * 8.0 / nnz << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t' << (file_check ? (fok ? 1 : 0) : -1) << '\t' << 0 << '\t'
                << names << '\t' << pay_bits << '\t' << hdr_b << '\t' << cells_b << '\t' << empt_b << '\t'
                << (is_e2 ? rep.blockhdr_bytes * 8.0 / nnz : -1) << '\t' << pay_ms << '\t' << pay_mnnz << '\t'
                << (is_e2 ? e2::g_rep.dec_refit_ms : -1) << '\t' << (is_e2 ? e2::g_rep.dec_gather_ms : -1) << '\t' << (is_e2 ? e2::g_rep.dec_hdr_ms : -1) << '\t'
                << pay_cyc << '\t' << cal << '\t' << pay_norm << '\t' << (is_e2 ? (double)rep.nesc / nnz : -1) << '\t'
                << (is_e2 ? (double)e2::g_rep.ncorr / nnz : -1) << '\t' << (std::getenv("E2_TAG") ? std::getenv("E2_TAG") : "") << '\n';
            tsv.flush();
        }
    }
#ifdef E2_PROF_BUILD
    e2prof::stop_and_print(std::getenv("E2_PROF") ? std::getenv("E2_PROF") : "results/e2/prof.txt", 0x140000000ull);
#endif
#ifdef E2_STATS
    { uint64_t tot = 0, c8 = 0, c16 = 0, c24 = 0, c64 = 0; for (int i = 0; i < 4096; ++i) { tot += e2::g_win[i]; if (i <= 8) c8 += e2::g_win[i]; if (i <= 16) c16 += e2::g_win[i]; if (i <= 24) c24 += e2::g_win[i]; if (i <= 64) c64 += e2::g_win[i]; }
      std::printf("window sizes (queries %llu): <=8 %.3f <=16 %.3f <=24 %.3f <=64 %.3f ; w0 %.3f w1 %.3f w2 %.3f\n", (unsigned long long)tot, (double)c8/tot, (double)c16/tot, (double)c24/tot, (double)c64/tot, (double)e2::g_win[0]/tot, (double)e2::g_win[1]/tot, (double)e2::g_win[2]/tot); }
#endif
    return 0;
}
