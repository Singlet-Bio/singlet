// Track C evaluation driver: encode/decode every file with the requested Track C codecs, verify, time, and print the
// payload-only accounting (names excluded) with the encoder's ideal cross-entropy breakdown.
//   bin/track_c_eval.exe --codec c_r2 --out results/track_c_cv.tsv [--reps 1] [--holdout STUDY] file.1pz ...
#include "common.h"
#include "track_c.h"
#include <iostream>
#include <sstream>
using namespace cs;

static std::string base_name(const std::string& p) { auto s = p.find_last_of("/\\"); return s == std::string::npos ? p : p.substr(s + 1); }
static std::string parent_name(const std::string& p) {
    auto s = p.find_last_of("/\\"); if (s == std::string::npos || s == 0) return "";
    auto t = p.find_last_of("/\\", s - 1); size_t st = (t == std::string::npos) ? 0 : t + 1; return p.substr(st, s - st);
}

int main(int argc, char** argv) {
    std::string colcost; bool show_b = false; std::vector<std::string> files, want; std::string out = "results/track_c_cv.tsv"; int reps = 1;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) { std::stringstream ss(argv[++i]); std::string t; while (std::getline(ss, t, ',')) want.push_back(t); }
        else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--refresh" && i + 1 < argc) tc::g_refresh = std::atoi(argv[++i]);
        else if (a == "--coarse" && i + 1 < argc) tc::g_coarse = std::atoi(argv[++i]);
        else if (a == "--colcost" && i + 1 < argc) colcost = argv[++i];
        else if (a == "--kappa" && i + 1 < argc) tc::g_kappa = std::atoi(argv[++i]);
        else if (a == "--cptn" && i + 1 < argc) tc::g_cptn = std::atoi(argv[++i]);
        else if (a == "--buckets") show_b = true;
        else if (a == "--names-level" && i + 1 < argc) tc::g_names_level = std::atoi(argv[++i]);
        else if (a == "--holdout" && i + 1 < argc) tc::g_holdout = argv[++i];
        else files.push_back(a);
    }
    std::vector<Codec> codecs; register_track_c(codecs);
    std::vector<Codec> sel;
    for (auto& c : codecs) if (std::find(want.begin(), want.end(), c.name) != want.end()) sel.push_back(c);
    bool nf = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (nf) tsv << "sample\tmember\tcodec\tmodel_tag\tnnz\tn_cols\tpayload_bytes\tpayload_bits_per_nnz\tside_bits_nnz\tbucket_bits_nnz\tpresence_bits_nnz\tvalue_bits_nnz\txent_bits_nnz\ttotal_bytes\ttotal_bits_per_nnz\tnames_bytes\tenc_ms\tdec_ms\tdec_Mnnz_s\tmodel_bytes\tmatrix_ok\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        std::printf("== %s m=%u n=%u nnz=%llu\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz); std::fflush(stdout);
        for (auto& c : sel) {
            Timer te; auto enc = c.encode(M); double enc_ms = te.ms();
            tc::Report r = tc::g_report;
            double dec_min = 1e300; CanonicalMatrix D;
            for (int k = 0; k < reps; ++k) { Timer td; D = c.decode(enc); dec_min = std::min(dec_min, td.ms()); }
            bool ok = (D == M);
            double nn = (double)M.nnz;
            double payload_bits = r.rc_bytes * 8.0;
            double xent = (r.side_bits + r.bucket_bits + r.pres_bits + r.val_bits) / nn;
            std::printf("  %-16s tag=%-26s payload %9zu B = %6.3f b/nnz (side %.3f bucket %.3f pres %.3f val %.3f; xent %.3f)  names %zu B  total %.3f b/nnz  enc %.0f ms dec %.0f ms (%.2f Mnnz/s) model %zu B  %s\n",
                        c.name.c_str(), r.tag.c_str(), r.rc_bytes, payload_bits / nn, r.side_bits / nn, r.bucket_bits / nn, r.pres_bits / nn, r.val_bits / nn, xent,
                        r.names_bytes, enc.size() * 8.0 / nn, enc_ms, dec_min, nn / dec_min / 1e3, r.model_bytes, ok ? "matrix=OK" : "matrix=FAIL");
            if (show_b) { for (int b = 0; b < 40; ++b) if (r.nb[b] > 0) std::printf("     bucket %2d: nnz share %.3f  pres %.2f val %.2f bits/nnz\n", b, r.nb[b] / nn, r.pb[b] / r.nb[b], r.vb[b] / r.nb[b]); }
            if (!colcost.empty()) { std::ofstream cf(colcost + "_" + c.name + "_" + parent_name(f) + "_" + base_name(f) + ".f32", std::ios::binary); cf.write((const char*)tc::g_col_bits.data(), tc::g_col_bits.size() * 4); }
            std::fflush(stdout);
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << c.name << '\t' << r.tag << '\t' << M.nnz << '\t' << M.n << '\t' << r.rc_bytes << '\t' << payload_bits / nn << '\t'
                << r.side_bits / nn << '\t' << r.bucket_bits / nn << '\t' << r.pres_bits / nn << '\t' << r.val_bits / nn << '\t' << xent << '\t' << enc.size() << '\t'
                << enc.size() * 8.0 / nn << '\t' << r.names_bytes << '\t' << enc_ms << '\t' << dec_min << '\t' << nn / dec_min / 1e3 << '\t' << r.model_bytes << '\t' << (ok ? 1 : 0) << '\n';
            tsv.flush();
        }
    }
    return 0;
}
