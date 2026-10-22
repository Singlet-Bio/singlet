// SPDX-License-Identifier: MIT
// Compression-study driver.
//
//   bin/bench.exe [--codec NAME[,NAME...]] [--out results/run.tsv] [--reps 3] [--no-file-check] file.1pz [...]
//
// For every (file, codec): encode, decode, verify matrix roundtrip (and file roundtrip unless
// disabled), time both, append one TSV row. Codecs come from kernels/registry.h.
#include "common.h"
#include "ref_zstd_csc.h"
#include "idea2m.h"
#include <algorithm>
#include <windows.h>
static double cpu_ms() { FILETIME c, e, k, u; GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u); return (((unsigned long long)u.dwHighDateTime << 32) | u.dwLowDateTime) / 1e4; }
#include <cstdlib>
#include <iostream>
#include <sstream>

using namespace cs;

namespace cs {
inline std::vector<Codec> all_codecs() {
    std::vector<Codec> v;
    v.push_back(make_ref_identity_1pz());
    v.push_back(make_ref_zstd_csc(3));
    i2::Params st; st.mask = 0; st.T = 1024; st.val_mode = 3; st.zq = 8;                 // static prior only (stage 1)
    v.push_back(i2::make_idea2("idea2_static", st));
    i2::Params k8 = st; k8.T = 4096; k8.mask = 15; k8.mixres = 1; k8.mixshift = 16; k8.K = 8;   // stage 2/3 adaptive, 8 clusters
    v.push_back(i2::make_idea2("idea2_k8", k8));
    v.push_back(i2::make_idea2("idea2_env", st, true));      // parameters from I2_* environment variables
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
            { auto sl = f.find_last_of("/\\"); i2::g_calls_path = (sl == std::string::npos ? std::string(".") : f.substr(0, sl)) + "/cell_calls.tsv"; }
            auto enc = c.encode(M);
            double enc_ms = te.ms();
            for (auto& x : i2::g_rep.cyc) x = 0;   // so the cycle counters printed below are decode-only (last rep)
            double dec_min = 1e300, dec_cpu = 1e300;
            CanonicalMatrix D;
            for (int r = 0; r < reps; ++r) {
                for (auto& x : i2::g_rep.cyc) x = 0;
                Timer td; double cpu0 = cpu_ms();
                D = c.decode(enc);
                dec_cpu = std::min(dec_cpu, cpu_ms() - cpu0);
                dec_min = std::min(dec_min, td.ms());
            }
            bool mok = (D == M);
            if (c.name.rfind("idea2", 0) == 0) {
                auto& R = i2::g_rep;
                std::printf("    [idea2] cell-loop Gcycles (last encode+decode): top %.2f tail %.2f val %.2f store %.2f" "%c", R.cyc[0]/1e9, R.cyc[1]/1e9, R.cyc[2]/1e9, R.cyc[3]/1e9, 10);
                if (reps > 0) std::printf("    [idea2] decode CPU %.0f ms = %.2f Mnnz/s (wall %.0f ms)" "%c", dec_cpu, (double)M.nnz / dec_cpu / 1e3, dec_min, 10);
                double pay_bits = 0; for (int q = 0; q < i2::C_NCAT; ++q) pay_bits += R.bits[q];
                std::printf("    [idea2] payload %.0f B (%.3f bits/nnz) | hdr %zu names %zu | cells %llu (nnz %llu) empties %llu (nnz %llu) S=%llu Tn=%llu fit %.0f ms hdr %.0f cells %.0f empt %.0f ms\n",
                            pay_bits / 8, pay_bits / (double)M.nnz, R.hdr_bytes, R.names_bytes, (unsigned long long)R.ncells,
                            (unsigned long long)R.nnz_cells, (unsigned long long)R.nempt, (unsigned long long)R.nnz_empt,
                            (unsigned long long)R.S, (unsigned long long)R.Tn, R.fit_ms, R.t_hdr, R.t_cells, R.t_empt);
                std::printf("    [idea2] cells: ");
                double cb = 0, eb = 0;
                const double hkc = R.hk_cells, hke = R.hk_empt;
                for (int q = 0; q < i2::C_NCAT; ++q) {
                    double v = R.bits[q];
                    if (q == i2::C_HK) v = hkc;
                    if (q == i2::C_EMPT) continue;
                    if (v > 0.5) std::printf("%s %.0f B (%.3f b/nnz) | ", i2::cat_name(q), v / 8, v / std::max<double>(1, (double)R.nnz_cells));
                    cb += v;
                }
                std::printf("TOTAL %.3f bits/nnz(cells)\n", cb / std::max<double>(1, (double)R.nnz_cells));
                if (std::getenv("I2_RANKPROF")) {
                    std::printf("    [idea2] rank profile (bucket ilog2(rank+1): nnz, bits/nnz-in-bucket):");
                    for (int q = 0; q < 16; ++q) if (R.rnnz[q]) std::printf(" %d:%llu,%.2f", q, (unsigned long long)R.rnnz[q], R.rbits[q] / (double)R.rnnz[q]);
                    std::printf("\n");
                }
                eb = hke + R.bits[i2::C_EMPT];
                {   // machine-readable breakdown
                    bool nf = !std::ifstream("results/idea2_m_breakdown.tsv").good();
                    std::ofstream bt("results/idea2_m_breakdown.tsv", std::ios::app);
                    if (nf) {
                        bt << "sample\tmember\tcodec\ttag\tnnz_cells\tnnz_empt\tncells\tnempt\tS\tTn";
                        for (int q = 0; q < i2::C_NCAT; ++q) bt << '\t' << i2::cat_name(q) << "_bits";
                        bt << "\thk_cells_bits\thk_empt_bits\tnames_bytes\tpayload_bytes\tdec_ms\tmatrix_ok\tt_cells_ms\tt_empt_ms\n";
                    }
                    const char* tg = std::getenv("I2_TAG");
                    std::string dir = parent_name(f);
                    bt << dir << '\t' << base_name(f) << '\t' << c.name << '\t' << (tg ? tg : "") << '\t' << R.nnz_cells << '\t' << R.nnz_empt << '\t'
                       << R.ncells << '\t' << R.nempt << '\t' << R.S << '\t' << R.Tn;
                    for (int q = 0; q < i2::C_NCAT; ++q) bt << '\t' << R.bits[q];
                    bt << '\t' << R.hk_cells << '\t' << R.hk_empt << '\t' << R.names_bytes << '\t' << R.payload_bytes << '\t' << (reps > 0 ? dec_cpu : -1.0) << '\t' << (mok ? 1 : 0)
                       << '\t' << R.t_cells << '\t' << R.t_empt << '\n';
                }
                std::printf("    [idea2] empties: hdr_k %.0f B (%.3f b/nnz) data %.0f B (%.3f b/nnz) TOTAL %.3f bits/nnz(empties)\n", hke / 8,
                            hke / std::max<double>(1, (double)R.nnz_empt), R.bits[i2::C_EMPT] / 8, R.bits[i2::C_EMPT] / std::max<double>(1, (double)R.nnz_empt),
                            eb / std::max<double>(1, (double)R.nnz_empt));
            }
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
