// Timing of correct_ambient_gene_level() on a synthetic 15,000-cell x 38,000-gene library (+ 100,000 empty droplets).
// Usage: perf [threads] [n_cells] [n_genes]. Cell types with disjoint marker sets, heavy-tailed gene weights, ~3k UMIs per cell.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "singlet/pileup/ambient_correction.h"

using singlet::ambient_detail::SplitMix64;

int main(int argc, char** argv) {
    const unsigned threads = argc > 1 ? std::stoul(argv[1]) : 16;
    const uint32_t n_cells = argc > 2 ? std::stoul(argv[2]) : 15000;
    const uint32_t G = argc > 3 ? std::stoul(argv[3]) : 38000;
    const uint32_t n_empty = 100000, T = 12;
    SplitMix64 rng(1);
    std::vector<std::vector<double>> cdf(T, std::vector<double>(G));
    std::vector<double> amb(G, 0.0), acdf(G);
    for (uint32_t t = 0; t < T; ++t) {
        double s = 0.0;
        for (uint32_t g = 0; g < G; ++g) {
            const bool shared = g % 5 == 0, mine = (g % T) == t;
            const double w = (shared || mine) ? std::exp(-3.0 * std::log(1.0 + 50.0 * rng.uniform())) : 0.0;
            s += w; cdf[t][g] = s;
            amb[g] += w / 1.0;
        }
        for (uint32_t g = 0; g < G; ++g) cdf[t][g] /= s;
    }
    double s = 0.0;
    for (uint32_t g = 0; g < G; ++g) { s += amb[g]; acdf[g] = s; }
    for (auto& v : acdf) v /= s;
    auto draw = [&](const std::vector<double>& c) { return static_cast<uint32_t>(std::lower_bound(c.begin(), c.end(), rng.uniform()) - c.begin()); };

    const uint32_t B = n_cells + n_empty;
    singlet::CSCu32 M;
    M.nrows = G; M.ncols = B; M.indptr.assign(B + 1, 0);
    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0);
    std::vector<uint32_t> dense(G, 0), touched;
    for (uint32_t b = 0; b < B; ++b) {
        const bool cell = b < n_cells;
        const uint32_t N = cell ? 1500 + static_cast<uint32_t>(rng.uniform() * 3000) : 3 + static_cast<uint32_t>(rng.uniform() * 60);
        const double rho = 0.02 + 0.06 * rng.uniform();
        for (uint32_t m = 0; m < N; ++m) {
            const uint32_t g = (cell && rng.uniform() > rho) ? draw(cdf[b % T]) : draw(acdf);
            if (!dense[g]++) touched.push_back(g);
        }
        std::sort(touched.begin(), touched.end());
        for (uint32_t g : touched) { M.indices.push_back(static_cast<int32_t>(g)); M.data.push_back(dense[g]); dense[g] = 0; }
        touched.clear();
        M.indptr[b + 1] = static_cast<int32_t>(M.indices.size());
    }
    std::cerr << "[perf] matrix " << G << " x " << B << " nnz=" << M.data.size() << " cells=" << n_cells << "\n";
    singlet::AmbientDecontConfig cfg;
    cfg.threads = threads;
    auto R = singlet::correct_ambient_gene_level(M, cells, cfg);
    std::cerr << "[perf] threads=" << threads << " seconds=" << R.seconds << " clusters=" << R.n_clusters << " mean_rho=" << R.mean_rho
              << " removed=" << R.umis_removed << "/" << R.umis_observed << " outliers=" << R.n_outlier_cells << "\n";
    return 0;
}
