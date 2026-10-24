// SPDX-License-Identifier: MIT
// Idea 1 benchmark driver: copy of bench.cpp with a local codec registry and extra TSV columns
// (payload-only bits/nnz, name bytes, payload-only decode time).
//   bin/idea1_bench.exe [--codec a,b] [--out tsv] [--reps 3] [--no-file-check] file.1pz ...
#include "common.h"
#include "ref_zstd_csc.h"
#include "idea1_codec.h"
#include "idea1_fast.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <windows.h>

static LONG WINAPI crash_handler(EXCEPTION_POINTERS* ep) {
    std::fprintf(stderr, "[crash] code=0x%lx rip=0x%llx base=0x%llx rva=0x%llx",
                 (unsigned long)ep->ExceptionRecord->ExceptionCode, (unsigned long long)ep->ContextRecord->Rip,
                 (unsigned long long)GetModuleHandleA(nullptr),
                 (unsigned long long)(ep->ContextRecord->Rip - (ULONG_PTR)GetModuleHandleA(nullptr)));
    if (ep->ExceptionRecord->NumberParameters >= 2)
        std::fprintf(stderr, " access=%llu addr=0x%llx", (unsigned long long)ep->ExceptionRecord->ExceptionInformation[0],
                     (unsigned long long)ep->ExceptionRecord->ExceptionInformation[1]);
    std::fprintf(stderr, " rsp=0x%llx\n",(unsigned long long)ep->ContextRecord->Rsp);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}

using namespace cs;

namespace cs {
inline std::vector<Codec> all_codecs() {
    std::vector<Codec> v;
    v.push_back(make_ref_identity_1pz());
    v.push_back(make_ref_zstd_csc(3));
    v.push_back(make_ref_zstd_csc(19));
    register_idea1(v);
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
static std::string gse_of(const std::string& sample) {
    static const std::map<std::string, std::string> g = {
        {"pbmc_v2_typical", "GSE142444"}, {"pbmc_v2_shallow", "GSE142444"}, {"pbmc_v3_lowgenes", "GSE244263"},
        {"tcell_5p", "GSE152522"}, {"tumor_3p", "GSE145370"}, {"brain_sn_v3_deep", "GSE285936"},
        {"brain_sn_v3_intronic", "GSE285936"}, {"h_pbmc_a", "GSE165080"}, {"h_b", "GSE285335"},
        {"h_c", "GSE244515"}, {"m_brain_a", "GSE249382"}, {"m_brain_c", "GSE245201"}, {"m_brain_d", "GSE254360"}};
    auto it = g.find(sample);
    return it == g.end() ? "unknown" : it->second;
}

int main(int argc, char** argv) {
    SetUnhandledExceptionFilter(crash_handler);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    std::vector<std::string> files, want;
    std::string out = "results/idea1_run.tsv";
    int reps = 3;
    bool file_check = true;
    std::string dump_dir, rawdump_dir, choice_file = "results/idea1_choice.tsv";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--choice" && i + 1 < argc) {
            choice_file = argv[++i];
        } else if (a == "--rawdump" && i + 1 < argc) {
            rawdump_dir = argv[++i];
        } else if (a == "--dump" && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (a == "--codec" && i + 1 < argc) {
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
        std::cerr << "usage: idea1_bench [--codec a,b] [--out tsv] [--reps n] [--no-file-check] file.1pz...\n";
        return 2;
    }
    auto codecs = all_codecs();
    if (!want.empty()) {
        std::vector<Codec> sel;
        for (auto& c : codecs)
            if (std::find(want.begin(), want.end(), c.name) != want.end()) sel.push_back(c);
        codecs = sel;
    }
    // per-file model selection table (sample 	 member 	 hold-tag), written by tools/idea1_choose.py
    const char TAB = 9;
    std::map<std::string, std::string> choice;
    {
        std::ifstream cf(choice_file);
        std::string line;
        while (std::getline(cf, line)) {
            std::stringstream ls(line);
            std::string a, b, c;
            if (std::getline(ls, a, TAB) && std::getline(ls, b, TAB) && std::getline(ls, c, TAB) && a != "sample") choice[a + "/" + b] = c;
        }
    }
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tratio_vs_1pz\tbits_per_nnz\tnnz\tm\tn\t"
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\t"
               "names_bytes\tpayload_bits_per_nnz\tside_bits_per_nnz\tpayload_dec_ms\tpayload_Mnnz_s\thold\t"
               "dec_cyc_per_nnz\tpay_cyc_per_nnz\tcal_cyc_per_iter\tpay_Mnnz_s_norm\n";

    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        std::string sample = parent_name(f);
        i1::ctx().hold = gse_of(sample);
        {
            auto it = choice.find(sample + "/" + base_name(f).substr(0, base_name(f).find('_')));
            i1::ctx().sel_hold = it == choice.end() ? "ins" : it->second;
        }
        if (!rawdump_dir.empty()) {
            std::vector<uint8_t> r(24, 0);
            std::memcpy(r.data(), &M.m, 4); std::memcpy(r.data() + 4, &M.n, 4); std::memcpy(r.data() + 8, &M.nnz, 8); r[16] = M.vt_code;
            auto app = [&](const void* d, size_t n) { const uint8_t* b = (const uint8_t*)d; r.insert(r.end(), b, b + n); };
            app(M.indptr.data(), M.indptr.size() * 4); app(M.indices.data(), M.indices.size() * 4); app(M.data.data(), M.data.size() * 4);
            write_file(rawdump_dir + "/" + sample + "_" + base_name(f) + ".raw", r);
        }
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)  hold=%s\n", f.c_str(), M.m, M.n,
                    (unsigned long long)M.nnz, orig.size(), orig.size() * 8.0 / (double)M.nnz, i1::ctx().hold.c_str());
        for (auto& c : codecs) {
            i1::ctx().names_bytes = 0; i1::ctx().side_bytes = 0; i1::ctx().reset_timing();
            Timer te;
            std::vector<uint8_t> enc;
            try { enc = c.encode(M); }
            catch (std::exception& e) { std::printf("  %-28s ENCODE ERROR: %s\n", c.name.c_str(), e.what()); continue; }
            double enc_ms = te.ms();
            if (!dump_dir.empty()) write_file(dump_dir + "/" + sample + "_" + base_name(f) + "_" + c.name + ".bin", enc);
            double nb_names = (double)i1::ctx().names_bytes, side = (double)i1::ctx().side_bytes;
            double dec_min = 1e300, dec_cyc = 1e300;
            CanonicalMatrix D;
            i1::g_pay_cyc_min = 1e300;
            double cal = i1::calibrate_chain();
            for (int r = 0; r < reps; ++r) {
                Timer td;
                double c0 = i1::thread_cycles();
                D = c.decode(enc);
                dec_cyc = std::min(dec_cyc, i1::thread_cycles() - c0);
                dec_min = std::min(dec_min, td.ms());
            }
            cal = std::min(cal, i1::calibrate_chain());
            double pay_cyc = i1::g_pay_cyc_min < 1e299 ? i1::g_pay_cyc_min / (double)M.nnz : -1;
            // normalised payload throughput: cycles/nnz divided by the contention factor (cal/4), at 3.0 GHz
            double pay_norm = pay_cyc > 0 ? 3000.0 / (pay_cyc / (cal / 4.0)) : -1;
            double pay_ms = i1::ctx().dec_payload_ms_min;
            bool mok = (D == M);
            bool fok = false;
            if (mok && file_check) {
                std::string tmp = out + ".tmp.1pz";
                auto re = rewrite_1pz(D, tmp);
                fok = (re == orig);
                std::remove(tmp.c_str());
            }
            double mnnz_s = (double)M.nnz / dec_min / 1e3;
            double pay_bits = nb_names > 0 ? (enc.size() - nb_names) * 8.0 / (double)M.nnz : -1;
            double pay_mnnz = pay_ms < 1e299 ? (double)M.nnz / pay_ms / 1e3 : -1;
            std::printf("  %-20s %10zu B  %6.3f bits/nnz (payload %6.3f, side %5.3f)  enc %8.1f ms  dec %8.1f ms (%.1f Mnnz/s; payload-only %.1f; %.0f cyc/nnz, cal %.1f -> norm %.1f Mnnz/s)  %s%s\n",
                        c.name.c_str(), enc.size(), enc.size() * 8.0 / (double)M.nnz, pay_bits, side * 8.0 / (double)M.nnz,
                        enc_ms, dec_min, mnnz_s, pay_mnnz, dec_cyc / (double)M.nnz, cal, pay_norm, mok ? "matrix=OK" : "matrix=FAIL",
                        file_check ? (fok ? " file=OK" : " file=FAIL") : "");
            std::fflush(stdout);
            tsv << sample << '\t' << base_name(f) << '\t' << c.name << '\t' << orig.size() << '\t'
                << enc.size() << '\t' << (double)enc.size() / (double)orig.size() << '\t'
                << enc.size() * 8.0 / (double)M.nnz << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t'
                << (file_check ? (fok ? 1 : 0) : -1) << '\t' << i1::ctx().model_bytes << '\t'
                << nb_names << '\t' << pay_bits << '\t' << side * 8.0 / (double)M.nnz << '\t' << pay_ms << '\t'
                << pay_mnnz << '\t' << i1::ctx().hold << '\t' << dec_cyc / (double)M.nnz << '\t' << pay_cyc << '\t' << cal
                << '\t' << pay_norm << '\n';
            tsv.flush();
        }
    }
    return 0;
}
