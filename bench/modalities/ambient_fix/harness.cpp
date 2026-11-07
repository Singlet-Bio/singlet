// Harness: run correct_ambient_gene_level() (include/singlet/pileup/ambient_correction.h, compiled unchanged) on the
// barnyard inputs written by dump.py. Usage: harness IN_DIR OUT_DIR [threads] [kappa] [offgene_z] [n_clusters]
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/ambient_correction.h"

template <class T> static std::vector<T> slurp(const std::string& p, size_t n) {
    std::vector<T> v(n);
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::cerr << "cannot open " << p << "\n"; std::exit(2); }
    if (std::fread(v.data(), sizeof(T), n, f) != n) { std::cerr << "short read " << p << "\n"; std::exit(2); }
    std::fclose(f);
    return v;
}
template <class T> static void spit(const std::string& p, const std::vector<T>& v) {
    FILE* f = std::fopen(p.c_str(), "wb");
    std::fwrite(v.data(), sizeof(T), v.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: harness IN OUT [threads] [kappa] [z] [K] [cell_iters] [phi_pseudocount]\n"; return 1; }
    const std::string in = argv[1], out = argv[2];
    singlet::AmbientDecontConfig cfg;
    cfg.threads = argc > 3 ? std::stoul(argv[3]) : 16;
    if (argc > 4) cfg.prior_strength = std::stod(argv[4]);
    if (argc > 5) cfg.offgene_z = std::stod(argv[5]);
    if (argc > 6) cfg.n_clusters = std::stoul(argv[6]);
    if (argc > 7) cfg.cell_iters = std::stoul(argv[7]);
    if (argc > 8) cfg.phi_pseudocount = std::stod(argv[8]);
    if (argc > 9) cfg.equal_cell_weight = std::stoi(argv[9]) != 0;
    if (argc > 10) cfg.outlier_round = std::stoi(argv[10]) != 0;
    if (argc > 11) cfg.use_off_estimator = std::stoi(argv[11]) != 0;
    if (argc > 12) cfg.off_prior_reads = std::stod(argv[12]);
    if (argc > 13) cfg.marker_fold = std::stod(argv[13]);
    if (argc > 14) cfg.low_depth_frac = std::stod(argv[14]);
    size_t G, B, nnz, nc;
    { std::ifstream d(in + "/dims.txt"); d >> G >> B >> nnz >> nc; }
    auto t_load0 = std::chrono::steady_clock::now();
    auto indptr = slurp<int64_t>(in + "/raw.indptr", B + 1);
    auto indices = slurp<int32_t>(in + "/raw.indices", nnz);
    auto data = slurp<uint32_t>(in + "/raw.data", nnz);
    auto cells = slurp<uint32_t>(in + "/cells.idx", nc);
    singlet::CSCu32 M;
    M.nrows = static_cast<uint32_t>(G); M.ncols = static_cast<uint32_t>(B);
    M.indptr.assign(indptr.begin(), indptr.end());
    M.indices = std::move(indices); M.data = std::move(data);
    double t_load = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load0).count();
    const std::string split_mode = argc > 15 ? argv[15] : "";   // "wrap": rows [0,G/2) exon, [G/2,G) intron -> split API
    singlet::AmbientCorrectionResult R;
    if (split_mode == "wrap") {
        const uint32_t h = static_cast<uint32_t>(G / 2);
        singlet::CSCu32 E, I;
        E.nrows = I.nrows = h; E.ncols = I.ncols = M.ncols;
        E.indptr.assign(B + 1, 0); I.indptr.assign(B + 1, 0);
        for (size_t b = 0; b < B; ++b) {
            for (int32_t k = M.indptr[b]; k < M.indptr[b + 1]; ++k) {
                if (static_cast<uint32_t>(M.indices[k]) < h) { E.indices.push_back(M.indices[k]); E.data.push_back(M.data[k]); }
                else { I.indices.push_back(M.indices[k] - static_cast<int32_t>(h)); I.data.push_back(M.data[k]); }
            }
            E.indptr[b + 1] = static_cast<int32_t>(E.indices.size()); I.indptr[b + 1] = static_cast<int32_t>(I.indices.size());
        }
        R = singlet::correct_ambient_gene_level_split(E, I, cells, cfg);
    } else if (split_mode == "est_exon_half") {
        cfg.estimate_features.assign(G, 0);
        std::fill(cfg.estimate_features.begin(), cfg.estimate_features.begin() + G / 2, 1);
        R = singlet::correct_ambient_gene_level(M, cells, cfg);
    } else {
        R = singlet::correct_ambient_gene_level(M, cells, cfg);
    }
    { std::ofstream nr(out + "/corr.nrows"); nr << R.corrected.nrows << '\n'; }
    std::cerr << "[harness] load=" << t_load << "s estimate+correct=" << R.seconds << "s clusters=" << R.n_clusters
              << " mean_rho=" << R.mean_rho << " median_rho=" << R.median_rho << " umi_weighted=" << R.umi_weighted_rho
              << " pool=" << R.profile.n_ambient_barcodes << "/" << static_cast<uint64_t>(R.profile.total_ambient_umi) << "\n";
    std::vector<double> rho(nc); std::vector<uint32_t> cl(nc), rem(nc);
    for (size_t i = 0; i < nc; ++i) { rho[i] = R.contam[i].rho; cl[i] = R.contam[i].cluster; rem[i] = R.contam[i].n_removed; }
    spit(out + "/rho.f64", rho); spit(out + "/cluster.u32", cl); spit(out + "/removed.u32", rem);
    spit(out + "/ambient.f64", R.profile.gene_fractions);
    std::vector<int64_t> ip(nc + 1, 0); std::vector<int32_t> ix; std::vector<uint32_t> dv;
    for (size_t i = 0; i < nc; ++i) {
        const uint32_t b = cells[i];
        for (int32_t k = R.corrected.indptr[b]; k < R.corrected.indptr[b + 1]; ++k) { ix.push_back(R.corrected.indices[k]); dv.push_back(R.corrected.data[k]); }
        ip[i + 1] = ix.size();
    }
    spit(out + "/corr.indptr", ip); spit(out + "/corr.indices", ix); spit(out + "/corr.data", dv);
    singlet::write_ambient_json(out + "/ambient_contamination.json", R, cfg);
    std::ofstream j(out + "/run.json");
    std::cerr << "[harness] n_outlier_cells=" << R.n_outlier_cells << "\n[harness] cluster_rho:";
    for (double c : R.cluster_rho) std::cerr << " " << c;
    std::cerr << '\n';
    j << "{\"seconds\": " << R.seconds << ", \"nnz_corrected\": " << ix.size() << ", \"threads\": " << cfg.threads
      << ", \"kappa\": " << cfg.prior_strength << ", \"z\": " << cfg.offgene_z << ", \"n_clusters\": " << R.n_clusters << "}\n";
    return 0;
}
