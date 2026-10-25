// Run singlet::correct_ambient_gene_level_split (production, default AmbientDecontConfig) on one sample dumped by extract.py.
// usage: harness_split DUMP_DIR OUT_JSON [threads] [lower_umi]   (lower_umi 0 = production default of 100; else the pool rule UMI cap)
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include "singlet/pileup/ambient_correction.h"

template <class T> static std::vector<T> slurp(const std::string& p, size_t n) {
    std::vector<T> v(n);
    if (n == 0) return v;
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(2); }
    if (std::fread(v.data(), sizeof(T), n, f) != n) { std::cerr << "short read " << p << "\n"; std::exit(2); }
    std::fclose(f);
    return v;
}

static singlet::CSCu32 load(const std::string& d, const std::string& tag, size_t G, size_t B, size_t nnz) {
    auto ip = slurp<int64_t>(d + "/" + tag + ".indptr", B + 1);
    singlet::CSCu32 M;
    M.nrows = static_cast<uint32_t>(G); M.ncols = static_cast<uint32_t>(B);
    M.indptr.assign(ip.begin(), ip.end());
    M.indices = slurp<int32_t>(d + "/" + tag + ".indices", nnz);
    M.data = slurp<uint32_t>(d + "/" + tag + ".data", nnz);
    return M;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: harness_split DIR OUT_JSON [threads] [lower_umi]\n"; return 1; }
    const std::string d = argv[1];
    singlet::AmbientDecontConfig cfg;
    cfg.threads = argc > 3 ? std::stoul(argv[3]) : 4;
    if (argc > 4 && std::stoul(argv[4]) > 0) cfg.lower_umi = std::stoul(argv[4]);
    size_t G, B, nnzE, nnzI, nc, lower;
    { std::ifstream f(d + "/dims.txt"); f >> G >> B >> nnzE >> nnzI >> nc >> lower; }
    auto t0 = std::chrono::steady_clock::now();
    auto E = load(d, "exon", G, B, nnzE);
    auto I = load(d, "intron", G, B, nnzI);
    auto cells = slurp<uint32_t>(d + "/cells.idx", nc);
    auto R = singlet::correct_ambient_gene_level_split(E, I, cells, cfg);
    double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<double> rho;
    for (auto& c : R.contam) rho.push_back(c.rho);
    std::sort(rho.begin(), rho.end());
    auto q = [&](double p) { return rho.empty() ? 0.0 : rho[std::min(rho.size() - 1, static_cast<size_t>(p * rho.size()))]; };
    std::ofstream j(argv[2]);
    j << "{\"ok\": " << (R.ok ? 1 : 0) << ", \"lower_umi\": " << cfg.lower_umi << ", \"threads\": " << cfg.threads
      << ", \"n_cells\": " << nc << ", \"mean_rho\": " << R.mean_rho << ", \"median_rho\": " << R.median_rho
      << ", \"umi_weighted_rho\": " << R.umi_weighted_rho << ", \"rho_q10\": " << q(0.1) << ", \"rho_q90\": " << q(0.9)
      << ", \"n_clusters\": " << R.n_clusters << ", \"pool_barcodes\": " << R.profile.n_ambient_barcodes
      << ", \"pool_umi\": " << static_cast<uint64_t>(R.profile.total_ambient_umi) << ", \"exon_profile_share\": " << R.exon_profile_share
      << ", \"n_outlier_cells\": " << R.n_outlier_cells << ", \"n_low_depth_cells\": " << R.n_low_depth_cells
      << ", \"seconds\": " << R.seconds << ", \"wall_seconds\": " << wall << ", \"method\": \"" << R.method << "\"}\n";
    return 0;
}
