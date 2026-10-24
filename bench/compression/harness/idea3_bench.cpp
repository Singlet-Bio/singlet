// SPDX-License-Identifier: MIT
// Idea 3 benchmark driver (copy of harness/bench.cpp with a local codec list and out-of-sample pools).
//
//   bin/idea3_bench.exe [--codec NAME[,NAME...]] [--out results/idea3_run.tsv] [--reps 3] [--no-file-check]
//                       [--pool-spec results/idea3_pools.tsv] file.1pz [...]
// The pool spec maps a test file path to the comma-separated .1pz training files (leave-one-study-out) used to build
// the out-of-sample model for the *_oos codecs (built at first use, outside the timed encode/decode).
#include "common.h"
#include "ref_zstd_csc.h"
#include "idea3_codec.h"
#include "idea3_time.h"
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

static std::map<std::string, std::vector<std::string>> g_pools;
static void load_pools(const std::string& f) {
    std::ifstream in(f);
    std::string line;
    while (std::getline(in, line)) {
        auto t = line.find('\t');
        if (t == std::string::npos) continue;
        std::string key = line.substr(0, t), rest = line.substr(t + 1);
        std::stringstream ss(rest); std::string tok;
        while (std::getline(ss, tok, ',')) if (!tok.empty()) g_pools[key].push_back(tok);
    }
}

static std::map<std::string, std::shared_ptr<i3::OosModel>> g_acc;     // pooled statistics per test file (shared by all oos variants)
static std::shared_ptr<i3::OosModel> build_oos(const std::string& testfile, const CanonicalMatrix& Mtest, const i3::Params& P) {
    i3::GeneMap gm = i3::make_genemap(Mtest.rownames, P.flat);
    auto& acc = g_acc[testfile];
    if (!acc) {
        auto it = g_pools.find(testfile);
        if (it == g_pools.end()) throw std::runtime_error("no pool for " + testfile);
        acc = std::make_shared<i3::OosModel>();
        for (auto& tf : it->second) {
            CanonicalMatrix T = load_1pz(tf);
            if (T.rownames != Mtest.rownames) { std::fprintf(stderr, "  pool %s: rownames differ, skipped\n", tf.c_str()); continue; }
            acc->add(T, gm);
        }
    }
    auto model = std::make_shared<i3::OosModel>(*acc);
    model->finalize(gm, P.shrink, P.alpha, P.prior_mass);
    return model;
}

struct Variant { std::string name; i3::Params P; };
static std::vector<Variant> variants() {
    std::vector<Variant> v;
    { i3::Params P; v.push_back({"idea3_gene_then_interval", P}); }
    { i3::Params P; P.flat = true; v.push_back({"idea3_interval_direct", P}); }
    { i3::Params P; P.oos = true; P.split_adaptive = false; P.split_prior = true; P.prior_mass = 8192; v.push_back({"idea3_gene_then_interval_oos", P}); }
    { i3::Params P; P.oos = true; P.split_adaptive = true; P.split_prior = true; P.prior_mass = 64; v.push_back({"idea3_gti_oos_adapt", P}); }
    { i3::Params P; P.a0 = 1; v.push_back({"idea3_gti_self_a0_1", P}); }
    return v;
}

int main(int argc, char** argv) {
    std::vector<std::string> files, want;
    std::string out = "results/idea3_run.tsv";
    int reps = 3;
    bool file_check = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--codec" && i + 1 < argc) {
            std::stringstream ss(argv[++i]); std::string tok;
            while (std::getline(ss, tok, ',')) want.push_back(tok);
        } else if (a == "--out" && i + 1 < argc) out = argv[++i];
        else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (a == "--no-file-check") file_check = false;
        else if (a == "--pool-spec" && i + 1 < argc) load_pools(argv[++i]);
        else if (a == "--list") { for (auto& v : variants()) std::cout << v.name << "\n"; return 0; }
        else files.push_back(a);
    }
    if (files.empty()) { std::cerr << "usage: idea3_bench [--codec a,b] [--out tsv] [--reps n] [--pool-spec f] file.1pz...\n"; return 2; }
    auto vars = variants();
    if (!want.empty()) {
        std::vector<Variant> sel;
        for (auto& v : vars) if (std::find(want.begin(), want.end(), v.name) != want.end()) sel.push_back(v);
        vars = sel;
    }
    bool new_file = !std::ifstream(out).good();
    std::ofstream tsv(out, std::ios::app);
    if (new_file)
        tsv << "sample\tmember\tcodec\torig_bytes\tenc_bytes\tratio_vs_1pz\tbits_per_nnz\tnnz\tm\tn\t"
               "enc_ms\tdec_ms_min\tdec_Mnnz_s\tmatrix_ok\tfile_ok\tmodel_bytes\t"
               "payload_bytes\tpayload_bits_per_nnz\tnames_bytes\tside_bytes\tk_bits\tpres_bits\tcnt_bits\tsplit_bits\tnnz_gene\tumi\tenc_cpu_ms\tdec_cpu_ms_min\tdec_cpu_Mnnz_s\n";
    for (auto& f : files) {
        CanonicalMatrix M = load_1pz(f);
        auto orig = read_file(f);
        std::printf("== %s  m=%u n=%u nnz=%llu  %zu bytes (%.2f bits/nnz)\n", f.c_str(), M.m, M.n, (unsigned long long)M.nnz, orig.size(), orig.size() * 8.0 / (double)M.nnz);
        for (auto& v : vars) {
            std::shared_ptr<i3::OosModel> oos;
            if (v.P.oos || v.P.split_prior) oos = build_oos(f, M, v.P);
            Timer te; double cpu0 = i3::cpu_ms();
            auto enc = i3::encode_matrix(M, v.P, oos.get());
            double enc_ms = te.ms(); double enc_cpu = i3::cpu_ms() - cpu0;
            i3::LastStats ls = i3::last_stats();
            double dec_min = 1e300, dec_cpu_min = 1e300;
            CanonicalMatrix D;
            for (int r = 0; r < reps; ++r) {
                Timer td; double c0 = i3::cpu_ms();
                D = i3::decode_matrix(enc, v.P, oos.get());
                dec_min = std::min(dec_min, td.ms()); dec_cpu_min = std::min(dec_cpu_min, i3::cpu_ms() - c0);
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
            double nnzd = (double)M.nnz;
            std::printf("  %-30s %10zu B  payload %9zu B (%.3f b/nnz; K %.3f P %.3f C %.3f S %.3f side %.3f)  enc %7.0f ms (cpu %7.0f) dec %7.0f ms (cpu %7.0f, %.2f Mnnz/s cpu) model %.1f MB  %s\n",
                        v.name.c_str(), enc.size(), ls.payload_bytes, ls.payload_bytes * 8.0 / nnzd,
                        ls.bits[i3::C_K] / nnzd, ls.bits[i3::C_PRES] / nnzd, ls.bits[i3::C_CNT] / nnzd, ls.bits[i3::C_SPLIT] / nnzd, ls.bits[i3::C_SIDE] / nnzd,
                        enc_ms, enc_cpu, dec_min, dec_cpu_min, nnzd / dec_cpu_min / 1e3, (oos ? oos->model_bytes : 0) / 1e6, mok ? "matrix=OK" : "matrix=FAIL");
            std::fflush(stdout);
            tsv << parent_name(f) << '\t' << base_name(f) << '\t' << v.name << '\t' << orig.size() << '\t' << enc.size() << '\t'
                << (double)enc.size() / (double)orig.size() << '\t' << enc.size() * 8.0 / nnzd << '\t' << M.nnz << '\t' << M.m << '\t' << M.n << '\t'
                << enc_ms << '\t' << dec_min << '\t' << mnnz_s << '\t' << (mok ? 1 : 0) << '\t' << (file_check ? (fok ? 1 : 0) : -1) << '\t'
                << (oos ? oos->model_bytes : 0) << '\t' << ls.payload_bytes << '\t' << ls.payload_bytes * 8.0 / nnzd << '\t' << ls.names_bytes << '\t'
                << ls.side_bytes << '\t' << ls.bits[i3::C_K] << '\t' << ls.bits[i3::C_PRES] << '\t' << ls.bits[i3::C_CNT] << '\t' << ls.bits[i3::C_SPLIT] << '\t'
                << ls.nnz_gene << '\t' << ls.umi << '\t' << enc_cpu << '\t' << dec_cpu_min << '\t' << nnzd / dec_cpu_min / 1e3 << '\n';
            tsv.flush();
        }
    }
    return 0;
}
