// SPDX-License-Identifier: MIT
// Ambient estimator on a pipeline run's counts, configured as export.h configures it (pool 5..100 UMIs, barcode-keyed k-means with 10 restarts,
// ensemble 32, cells in barcode-sequence order), through correct_ambient_gene_level_split().
//   harness IN_DIR OUT_DIR [key=value ...]     keys: see set_param() below
// IN_DIR (from dump_run.py): raw.indptr/indices/data (int64/int32/uint32; rows 0..G-1 exon genes, G..2G-1 intron genes, all barcodes),
// cells.idx (called cells, column indices, in barcode-sequence order), barcodes.txt (all barcodes in column order), dims.txt "2G B nnz nc".
// OUT_DIR: rho.f64, removed.u32 (one per called cell), run.json.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
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

static void set_param(singlet::AmbientDecontConfig& c, const std::string& k, double v) {
    if (k == "threads") c.threads = static_cast<uint32_t>(v);
    else if (k == "ensemble") c.ensemble = static_cast<uint32_t>(v);
    else if (k == "restarts") c.kmeans_restarts = static_cast<uint32_t>(v);
    else if (k == "cap") c.cluster_rho_cap = v;
    else if (k == "capcell") c.capped_cell_mult = v;
    else if (k == "trend") c.depth_trend = v != 0;
    else if (k == "prior_min") c.prior_reads_min = v;
    else if (k == "prior_max") c.prior_reads_max = v;
    else if (k == "low_frac") c.low_depth_frac = v;
    else if (k == "kappa") c.prior_strength = v;
    else if (k == "offprior") c.off_prior_reads = v;
    else if (k == "outlier") c.outlier_round = v != 0;
    else if (k == "cellcap") c.cell_rho_cap_mult = v;
    else if (k == "ashrink") c.alpha_shrink_cells = v;
    else { std::cerr << "unknown parameter " << k << "\n"; std::exit(2); }
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: harness IN OUT [key=value ...]\n"; return 1; }
    const std::string in = argv[1], out = argv[2];
    size_t G2, B, nnz, nc;
    { std::ifstream d(in + "/dims.txt"); d >> G2 >> B >> nnz >> nc; }
    auto indptr = slurp<int64_t>(in + "/raw.indptr", B + 1);
    auto indices = slurp<int32_t>(in + "/raw.indices", nnz);
    auto data = slurp<uint32_t>(in + "/raw.data", nnz);
    auto cells = slurp<uint32_t>(in + "/cells.idx", nc);
    std::vector<std::string> barcodes;
    barcodes.reserve(B);
    { std::ifstream f(in + "/barcodes.txt"); std::string l; while (std::getline(f, l)) barcodes.push_back(l); }
    if (barcodes.size() != B) { std::cerr << "barcodes.txt has " << barcodes.size() << " lines, expected " << B << "\n"; return 2; }
    const uint32_t h = static_cast<uint32_t>(G2 / 2);
    singlet::CSCu32 E, I;
    E.nrows = I.nrows = h;
    E.ncols = I.ncols = static_cast<uint32_t>(B);
    E.indptr.assign(B + 1, 0);
    I.indptr.assign(B + 1, 0);
    for (size_t b = 0; b < B; ++b) {
        for (int64_t k = indptr[b]; k < indptr[b + 1]; ++k) {
            if (static_cast<uint32_t>(indices[k]) < h) { E.indices.push_back(indices[k]); E.data.push_back(data[k]); }
            else { I.indices.push_back(indices[k] - static_cast<int32_t>(h)); I.data.push_back(data[k]); }
        }
        E.indptr[b + 1] = static_cast<int32_t>(E.indices.size());
        I.indptr[b + 1] = static_cast<int32_t>(I.indices.size());
    }
    // as export.h: sequence rank of every barcode, packed barcode keys
    std::vector<uint32_t> rank(B), by(B);
    std::iota(by.begin(), by.end(), 0u);
    std::sort(by.begin(), by.end(), [&](uint32_t a, uint32_t b) { return barcodes[a] != barcodes[b] ? barcodes[a] < barcodes[b] : a < b; });
    for (uint32_t r = 0; r < B; ++r) rank[by[r]] = r;
    std::vector<uint64_t> keys(B, 0);
    for (size_t b = 0; b < B; ++b) {
        uint64_t k = 0;
        const auto& s = barcodes[b];
        for (size_t i = 0; i < s.size() && i < 32; ++i) {
            const char c = static_cast<char>(s[i] | 32);
            k = (k << 2) | (c == 'a' ? 0u : c == 'c' ? 1u : c == 'g' ? 2u : 3u);
        }
        keys[b] = k;
    }
    singlet::AmbientDecontConfig cfg;
    cfg.lower_umi = 100;
    cfg.pool_min_umi = 5;
    cfg.pool_tie_rank = &rank;
    cfg.barcode_keys = &keys;
    cfg.kmeans_restarts = 10;
    cfg.ensemble = 32;
    cfg.threads = 16;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        const size_t e = a.find('=');
        if (e == std::string::npos) continue;
        set_param(cfg, a.substr(0, e), std::atof(a.c_str() + e + 1));
    }
    const auto t0 = std::chrono::steady_clock::now();
    auto R = singlet::correct_ambient_gene_level_split(E, I, cells, cfg);
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::vector<double> rho(nc);
    std::vector<uint32_t> rem(nc);
    for (size_t i = 0; i < nc; ++i) { rho[i] = R.contam[i].rho; rem[i] = R.contam[i].n_removed; }
    spit(out + "/rho.f64", rho);
    spit(out + "/removed.u32", rem);
    std::ofstream j(out + "/run.json");
    j << "{\"seconds\": " << sec << ", \"mean_rho\": " << R.mean_rho << ", \"median_rho\": " << R.median_rho << ", \"n_clusters\": " << R.n_clusters
      << ", \"depth_gamma\": " << R.depth_gamma << ", \"prior_reads\": " << R.prior_reads << ", \"n_clusters_capped\": " << R.n_clusters_capped
      << ", \"n_low_depth_cells\": " << R.n_low_depth_cells << ", \"skipped\": \"" << R.skipped_reason << "\"}\n";
    std::cerr << "[harness] " << sec << " s mean_rho=" << R.mean_rho << " median=" << R.median_rho << " clusters=" << R.n_clusters
              << " gamma=" << R.depth_gamma << " prior_reads=" << R.prior_reads << " capped=" << R.n_clusters_capped << "\n";
    return 0;
}
