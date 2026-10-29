// SPDX-License-Identifier: MIT
// w3_total driver: derived from harness/bench.cpp.  Runs ref_1pz_v1_L3, a_names_perm_model, b_rans16 (as the yardsticks)
// and the w3_total codecs on .1pz files; checks decode(encode(M)) == M on every field, times single-thread decode
// (min of N reps), and records a per-component size breakdown, native-order decode timing and the idle-machine clock.
//
//   bin/w3_total_bench.exe [--codec a,b] [--out results/w3_total_x.tsv] [--detail results/w3_total_x_detail.tsv]
//                          [--reps 5] [--dump DIR] [--calls FILE] [--no-native] [--list] file.1pz ...
// cell_calls.tsv is taken from the directory of each input file unless --calls is given.
#include "common.h"
#include "ref_zstd_csc.h"
#include "w3_total.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

namespace cs {
inline std::vector<Codec> all_codecs() {
    std::vector<Codec> v, t;
    v.push_back(make_ref_identity_1pz());
    register_track_a(t);
    register_track_b(t);
    for (auto& c : t)
        if (c.name == "a_names_perm_model" || c.name == "b_rans16") v.push_back(c);
    w3::register_w3(v);
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
static std::string dir_name(const std::string& p) {
    auto s = p.find_last_of("/\\");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}

// Effective core clock (GHz) from a 1-cycle dependency chain (as Track B's driver).
static double calib_ghz() {
    uint64_t s = 0;
    Timer t;
    for (uint64_t i = 0; i < 10000000ull; ++i) {
        asm volatile("add $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0\n\tadd $1,%0" : "+r"(s));
    }
    return 1e8 / t.ms() / 1e6;
}
struct Meas { double wall_ms = 1e300, cyc = 1e300, ghz = 0; };
template <class F> static Meas measure(int reps, F&& fn) {
    Meas m;
    m.ghz = calib_ghz();
    for (int r = 0; r < reps; ++r) {
        tb::CycTimer ct;
        Timer t;
        fn();
        m.wall_ms = std::min(m.wall_ms, t.ms());
        m.cyc = std::min(m.cyc, ct.cycles());
    }
    m.ghz = std::max(m.ghz, calib_ghz());
    return m;
}

#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentProcess(void);
extern "C" __declspec(dllimport) int __stdcall SetPriorityClass(void*, unsigned long);
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void*, int);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), 0x80 /*HIGH_PRIORITY_CLASS*/);
    SetThreadPriority(GetCurrentThread(), 2 /*THREAD_PRIORITY_HIGHEST*/);
#endif
    std::vector<std::string> files, want;
    std::string out = "results/w3_total_run.tsv", detail, dump, calls_fixed;
    int reps = 5;
    bool native = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string tok;
            while (std::getline(ss, tok, ',')) want.push_back(tok);
        } else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--detail" && i + 1 < argc) detail = argv[++i];
        else if (a == "--dump" && i + 1 < argc) dump = argv[++i];
        else if (a == "--calls" && i + 1 < argc) calls_fixed = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--no-native") native = false;
        else if (a == "--list") {
            for (auto& c : all_codecs()) std::cout << c.name << "\n";
            return 0;
        } else files.push_back(a);
    }
    if (files.empty()) {
        std::cerr << "usage: w3_total_bench [--codec a,b] [--out tsv] [--detail tsv] [--reps n] [--dump dir] file.1pz...\n";
        return 2;
    }
    if (detail.empty()) detail = out.substr(0, out.size() - (out.size() > 4 && out.substr(out.size() - 4) == ".tsv" ? 4 : 0)) + "_detail.tsv";
    auto codecs = all_codecs();
    if (!want.empty()) {
        std::vector<Codec> sel;
        for (auto& w : want)
            for (auto& c : codecs) if (c.name == w) sel.push_back(c);
        codecs = sel;
    }
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tratio_vs_1pz\tbits_per_nnz\tnnz\tm\tn\t"
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\n";
    bool new_detail = !std::ifstream(detail).good();
    std::ofstream dtsv(detail, std::ios::app);
    if (new_detail)
        dtsv << "sample\tmember\tcodec\tenc_bytes\tnnz\thdr\tnames\tnames_row\tnames_col\tnames_kv\tcells_colcnt\tcells_tables\tcells_chunks\tempt_colcnt\tempt_tables\tempt_chunks\tcells_n\tcells_nnz\tcells_counts\tcells_payload\tcells_counts_tag\t"
                "empt_n\tempt_nnz\tempt_counts\tempt_payload\tempt_ambient\tempt_k\tbitmap\t"
                "dec_ms_min\tph_copy\tph_names\tph_counts\tph_payload\tph_empt\tph_split\tph_merge\t"
                "payload_ms_canon\tpayload_ms_native\tfull_ms_native\tnative_ok\tcore_ghz\tmatrix_ok\n";

    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        w3::g_calls_path = calls_fixed.empty() ? dir_name(f) + "/cell_calls.tsv" : calls_fixed;
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz,
                    orig.size(), orig.size() * 8.0 / (double)M.nnz);
        for (auto& c : codecs) {
            const bool is_w3 = c.name.rfind("w3_total", 0) == 0;
            std::vector<uint8_t> enc;
            double enc_ms = 0;
            w3::Bd bd;
            try {
                Timer te;
                enc = c.encode(M);
                enc_ms = te.ms();
                if (is_w3) bd = w3::last_bd();
            } catch (const std::exception& e) {
                std::printf("  %-22s ENCODE FAILED: %s\n", c.name.c_str(), e.what());
                continue;
            }
            if (!dump.empty() && is_w3) write_file(dump + "/" + parent_name(f) + "__" + base_name(f) + "__" + c.name + ".bin", enc);
            double dec_min = 1e300;
            w3::Dt best_dt;
            CanonicalMatrix D;
            try {
                for (int r = 0; r < reps; ++r) {
                    Timer td;
                    D = c.decode(enc);
                    double ms = td.ms();
                    if (ms < dec_min) { dec_min = ms; if (is_w3) best_dt = w3::last_dt(); }
                }
            } catch (const std::exception& e) {
                std::printf("  %-22s DECODE FAILED: %s\n", c.name.c_str(), e.what());
                continue;
            }
            bool mok = (D == M);
            double mnnz_s = (double)M.nnz / dec_min / 1e3;
            std::printf("  %-22s %10zu B  ratio %.3f  %6.2f bits/nnz  enc %8.1f ms  dec %8.1f ms (%.1f Mnnz/s)  %s\n", c.name.c_str(),
                        enc.size(), (double)enc.size() / (double)orig.size(), enc.size() * 8.0 / (double)M.nnz, enc_ms, dec_min, mnnz_s,
                        mok ? "matrix=OK" : "matrix=FAIL");
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << orig.size() << '\t' << enc.size() << '\t'
                << (double)enc.size() / (double)orig.size() << '\t' << enc.size() * 8.0 / (double)M.nnz << '\t' << M.nnz << '\t' << M.m
                << '\t' << M.n << '\t' << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t' << -1 << '\t'
                << c.model_bytes << '\n';
            tsv.flush();
            if (!mok) std::printf("  !!! roundtrip FAILED for %s\n", c.name.c_str());

            // payload-only and native-order timings (v1 / b_rans16 only)
            double pc_ms = -1, pn_ms = -1, fn_ms = -1, ghz = 0;
            int nat_ok = -1;
            if (native && (c.name == "w3_total_v1" || c.name == "w3_total_v1c")) {
                std::vector<uint8_t> buf(enc.size() + 64, 0);
                std::memcpy(buf.data(), enc.data(), enc.size());
                ta::ByteR r(buf.data(), enc.size());
                r.take(4); r.u8();
                CanonicalMatrix T;
                T.m = r.u32(); T.vt_code = r.u8();
                ta::dec_names_model(r, T);
                auto V = std::make_shared<w3::MemberView>(w3::parse_member(r, T.m, T.vt_code));
                auto S = std::make_shared<CanonicalMatrix>();
                auto run = [&](bool canonical) {
                    return [V, S, canonical]() { tb::DecState Dq; w3::member_payload_decode(*V, canonical, *S, Dq); };
                };
                Meas mn = measure(reps, run(false));
                Meas mc = measure(reps, run(true));
                pn_ms = mn.wall_ms; pc_ms = mc.wall_ms; ghz = std::max(mn.ghz, mc.ghz);
                CanonicalMatrix N;
                Meas mf = measure(reps, [&] { N = w3::dec_v1(enc, false); });
                fn_ms = mf.wall_ms;
                std::string why;
                nat_ok = w3::native_matches(N, M, &why) ? 1 : 0;
                std::printf("  %-22s payload-only: canonical %.1f ms (%.1f Mnnz/s)  native %.1f ms (%.1f Mnnz/s)  | full native %.1f ms (%.1f Mnnz/s) native_ok=%d %s  [core %.2f GHz]\n",
                            c.name.c_str(), pc_ms, M.nnz / pc_ms / 1e3, pn_ms, M.nnz / pn_ms / 1e3, fn_ms, M.nnz / fn_ms / 1e3, nat_ok,
                            why.c_str(), ghz);
            } else if (native && c.name == "b_rans16") {
                auto it = tb::payload_hooks().find(c.name);
                if (it != tb::payload_hooks().end()) {
                    auto fn_n = it->second(enc, false);
                    Meas mn = measure(reps, fn_n);
                    auto fn_c = it->second(enc, true);
                    Meas mc = measure(reps, fn_c);
                    pn_ms = mn.wall_ms; pc_ms = mc.wall_ms; ghz = std::max(mn.ghz, mc.ghz);
                    std::printf("  %-22s payload-only: canonical %.1f ms (%.1f Mnnz/s)  native %.1f ms (%.1f Mnnz/s)  [core %.2f GHz]\n",
                                c.name.c_str(), pc_ms, M.nnz / pc_ms / 1e3, pn_ms, M.nnz / pn_ms / 1e3, ghz);
                }
            }
            if (is_w3) {
                std::printf("      bytes: hdr %zu names %zu (row %zu col %zu kv %zu) | cells colcnt %zu tables %zu chunks %zu | cells n=%zu nnz=%zu counts %zu payload %zu (tag %d) | empt n=%zu nnz=%zu counts %zu payload %zu ambient %zu (k %zu) | bitmap %zu\n",
                            bd.hdr, bd.names, bd.names_row, bd.names_col, bd.names_kv, bd.cells_colcnt, bd.cells_tables, bd.cells_chunks, bd.cells_n, bd.cells_nnz, bd.cells_counts, bd.cells_payload, bd.counts_tag, bd.empt_n,
                            bd.empt_nnz, bd.empt_counts, bd.empt_payload, bd.empt_ambient, bd.empt_k, bd.bitmap);
                std::printf("      dec phases (ms): copy %.1f names %.1f counts %.1f payload %.1f empt %.1f split %.1f merge %.1f\n", best_dt.copy,
                            best_dt.names, best_dt.counts, best_dt.payload, best_dt.empt, best_dt.split, best_dt.merge);
            }
            dtsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << enc.size() << '\t' << M.nnz << '\t' << bd.hdr << '\t'
                 << bd.names << '\t' << bd.names_row << '\t' << bd.names_col << '\t' << bd.names_kv << '\t' << bd.cells_colcnt << '\t'
                 << bd.cells_tables << '\t' << bd.cells_chunks << '\t' << bd.empt_colcnt << '\t' << bd.empt_tables << '\t'
                 << bd.empt_chunks << '\t' << bd.cells_n << '\t' << bd.cells_nnz << '\t' << bd.cells_counts << '\t' << bd.cells_payload << '\t'
                 << bd.counts_tag << '\t' << bd.empt_n << '\t' << bd.empt_nnz << '\t' << bd.empt_counts << '\t' << bd.empt_payload << '\t'
                 << bd.empt_ambient << '\t' << bd.empt_k << '\t' << bd.bitmap << '\t' << dec_min << '\t' << best_dt.copy << '\t'
                 << best_dt.names << '\t' << best_dt.counts << '\t' << best_dt.payload << '\t' << best_dt.empt << '\t' << best_dt.split
                 << '\t' << best_dt.merge << '\t' << pc_ms << '\t' << pn_ms << '\t' << fn_ms << '\t' << nat_ok << '\t' << ghz << '\t'
                 << (mok ? 1 : 0) << '\n';
            dtsv.flush();
        }
    }
    return 0;
}
