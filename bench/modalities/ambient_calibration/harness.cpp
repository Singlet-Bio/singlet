// Ambient calibration harness: runs correct_ambient_gene_level[_split]() (include/singlet/pileup/ambient_correction.h, compiled
// unchanged) on stored counts, with config overrides, and writes per-cell rho plus the estimator's diagnostics.
//
// usage: harness IN_DIR OUT_PREFIX [key=value ...]
//   IN_DIR with dims.txt of 6 fields "G B nnzE nnzI n_cells lower" -> exon.* / intron.* (split API, as in production)
//          or of 4 fields "G B nnz n_cells"                     -> raw.* (single gene matrix; barnyard inputs)
//   keys: cells=FILE.idx (uint32 cell columns instead of IN_DIR/cells.idx), threads, off_prior_reads, prior_strength,
//         cluster_prior_cells, min_cluster_cells, low_depth_frac, offgene_z, marker_fold, n_clusters, lower_umi, outlier_round
// writes OUT_PREFIX.rho.f64 .cluster.u32 .offx.f64 .offe.f64 .prior.f64 (cell order of the cells file) and OUT_PREFIX.json
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
template <class T> static void spit(const std::string& p, const std::vector<T>& v) {
    FILE* f = std::fopen(p.c_str(), "wb");
    if (!f) { std::cerr << "cannot write " << p << "\n"; std::exit(2); }
    std::fwrite(v.data(), sizeof(T), v.size(), f);
    std::fclose(f);
}
static size_t fsize(const std::string& p) {
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    return f ? static_cast<size_t>(f.tellg()) : 0;
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
    if (argc < 3) { std::cerr << "usage: harness IN_DIR OUT_PREFIX [key=value ...]\n"; return 1; }
    const std::string d = argv[1], out = argv[2];
    singlet::AmbientDecontConfig cfg;
    cfg.threads = 4;
    std::string cells_file = d + "/cells.idx";
    std::ostringstream params;
    for (int i = 3; i < argc; ++i) {
        const std::string kv = argv[i];
        const auto eq = kv.find('=');
        if (eq == std::string::npos) { std::cerr << "bad arg " << kv << "\n"; return 1; }
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if (k == "cells") cells_file = v;
        else if (k == "threads") cfg.threads = std::stoul(v);
        else if (k == "off_prior_reads") cfg.off_prior_reads = std::stod(v);
        else if (k == "prior_strength") cfg.prior_strength = std::stod(v);
        else if (k == "cluster_prior_cells") cfg.cluster_prior_cells = std::stod(v);
        else if (k == "min_cluster_cells") cfg.min_cluster_cells = std::stoul(v);
        else if (k == "cluster_prior_reads") cfg.cluster_prior_reads = std::stod(v);
        else if (k == "cluster_estimate_pooled") cfg.cluster_estimate_pooled = std::stoi(v) != 0;
        else if (k == "sample_target_pooled") cfg.sample_target_pooled = std::stoi(v) != 0;
        else if (k == "low_depth_frac") cfg.low_depth_frac = std::stod(v);
        else if (k == "offgene_z") cfg.offgene_z = std::stod(v);
        else if (k == "marker_fold") cfg.marker_fold = std::stod(v);
        else if (k == "n_clusters") cfg.n_clusters = std::stoul(v);
        else if (k == "lower_umi") cfg.lower_umi = std::stoul(v);
        else if (k == "outlier_round") cfg.outlier_round = std::stoi(v) != 0;
        else { std::cerr << "unknown key " << k << "\n"; return 1; }
        params << (params.tellp() > 0 ? ", " : "") << '"' << k << "\": \"" << v << '"';
    }
    std::vector<size_t> dims;
    { std::ifstream f(d + "/dims.txt"); size_t x; while (f >> x) dims.push_back(x); }
    const size_t nc = fsize(cells_file) / sizeof(uint32_t);
    auto cells = slurp<uint32_t>(cells_file, nc);
    singlet::AmbientCorrectionResult R;
    const auto t0 = std::chrono::steady_clock::now();
    if (dims.size() >= 6) {
        auto E = load(d, "exon", dims[0], dims[1], dims[2]);
        auto I = load(d, "intron", dims[0], dims[1], dims[3]);
        R = singlet::correct_ambient_gene_level_split(E, I, cells, cfg);
    } else {
        auto M = load(d, "raw", dims[0], dims[1], dims[2]);
        R = singlet::correct_ambient_gene_level(M, cells, cfg);
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<double> rho(nc), ox(nc), oe(nc), pr(nc);
    std::vector<uint32_t> cl(nc);
    for (size_t i = 0; i < nc; ++i) {
        const auto& c = R.contam[i];
        rho[i] = c.rho; cl[i] = c.cluster; ox[i] = c.off_reads; oe[i] = c.off_exposure; pr[i] = c.prior_rho;
    }
    spit(out + ".rho.f64", rho); spit(out + ".cluster.u32", cl); spit(out + ".offx.f64", ox); spit(out + ".offe.f64", oe);
    spit(out + ".prior.f64", pr);
    std::ofstream j(out + ".json");
    j << "{\"params\": {" << params.str() << "}, \"ok\": " << (R.ok ? 1 : 0) << ", \"n_cells\": " << nc
      << ", \"mean_rho\": " << R.mean_rho << ", \"median_rho\": " << R.median_rho << ", \"n_clusters\": " << R.n_clusters
      << ", \"sample_rho\": " << R.sample_rho << ", \"n_low_depth_cells\": " << R.n_low_depth_cells
      << ", \"n_outlier_cells\": " << R.n_outlier_cells << ", \"pool_barcodes\": " << R.profile.n_ambient_barcodes
      << ", \"seconds\": " << R.seconds << ", \"wall_seconds\": " << wall << ", \"clusters\": [";
    for (size_t k = 0; k < R.cluster_rho.size(); ++k)
        j << (k ? ", " : "") << "{\"c\": " << R.cluster_rho[k] << ", \"c_raw\": " << R.cluster_rho_raw[k]
          << ", \"cells\": " << R.cluster_cells[k] << ", \"off_mass\": " << R.cluster_off_mass[k] << "}";
    j << "]}\n";
    std::cerr << "[harness] " << out << " mean_rho=" << R.mean_rho << " K=" << R.n_clusters << " c_s=" << R.sample_rho
              << " t=" << R.seconds << "s\n";
    return 0;
}
