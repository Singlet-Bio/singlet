// Quick synthetic-data driver: planted heterotypic doublets, prints AUROC, calls, valley, runtime.
// Build: g++ -O2 -std=c++17 -I<repo>/include/singlet/pileup synth_check.cpp -o synth_check
// Usage: synth_check [n_cells=4000] [n_types=8] [dbl_frac=0.05] [threads=0]
#include "doublet_detect.h"
#include <cstdio>
#include <cstdlib>
struct E { int32_t r, c; uint16_t v; };
static double auroc(const std::vector<double>& s, const std::vector<char>& y) {
    std::vector<size_t> o(s.size()); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return s[a] < s[b]; });
    double rs = 0, np = 0; size_t i = 0;
    while (i < o.size()) { size_t j = i; while (j + 1 < o.size() && s[o[j + 1]] == s[o[i]]) ++j;
        double rk = 0.5 * (i + j) + 1; for (size_t t = i; t <= j; ++t) if (y[o[t]]) { rs += rk; ++np; } i = j + 1; }
    double nn = o.size() - np; return (rs - np * (np + 1) / 2) / (np * nn);
}
int main(int argc, char** argv) {
    uint32_t n_sing = argc > 1 ? atoi(argv[1]) : 4000; int types = argc > 2 ? atoi(argv[2]) : 8;
    double frac = argc > 3 ? atof(argv[3]) : 0.05; int thr = argc > 4 ? atoi(argv[4]) : 0;
    const uint32_t G = 3000; const uint32_t n_dbl = uint32_t(n_sing * frac); const uint32_t n = n_sing + n_dbl;
    std::mt19937 rng(1);
    std::vector<double> base(G); std::lognormal_distribution<double> ln(0, 1.2);
    for (auto& b : base) b = ln(rng);
    std::vector<std::vector<double>> prof(types, base);
    std::uniform_int_distribution<int> gd(0, G - 1);
    for (int t = 0; t < types; ++t) for (int k = 0; k < 120; ++k) prof[t][gd(rng)] *= 6.0;
    for (auto& p : prof) { double s = 0; for (double v : p) s += v; for (double& v : p) v /= s; }
    std::vector<E> ent; std::vector<char> y(n, 0);
    auto draw = [&](int t, double depth, uint32_t col) {
        for (uint32_t g = 0; g < G; ++g) { double mu = prof[t][g] * depth; if (mu < 1e-4) continue;
            std::poisson_distribution<int> pd(mu); int v = pd(rng); if (v) ent.push_back({(int32_t)g, (int32_t)col, (uint16_t)v}); } };
    std::lognormal_distribution<double> dep(std::log(3000.0), 0.35);
    for (uint32_t c = 0; c < n_sing; ++c) draw(c % types, dep(rng), c);
    for (uint32_t d = 0; d < n_dbl; ++d) { int a = rng() % types, b = (a + 1 + rng() % (types - 1)) % types; uint32_t col = n_sing + d;
        draw(a, dep(rng), col); draw(b, dep(rng), col); y[col] = 1; }
    std::stable_sort(ent.begin(), ent.end(), [](const E& a, const E& b) { return a.c < b.c; });
    singlet::SparseAccumulator<uint16_t>::CSCMatrix m; m.nrows = G; m.ncols = n; m.indptr.assign(n + 1, 0);
    for (auto& e : ent) m.indptr[e.c + 1]++;
    for (uint32_t c = 0; c < n; ++c) m.indptr[c + 1] += m.indptr[c];
    for (auto& e : ent) { m.indices.push_back(e.r); m.data.push_back(e.v); }
    std::vector<uint32_t> cells(n); std::iota(cells.begin(), cells.end(), 0u);
    singlet::DoubletParams p; p.threads = thr; singlet::DoubletSummary sm;
    if (getenv("CLF")) p.classifier = atoi(getenv("CLF")) != 0;
    if (getenv("MASK")) p.clf_mask = atoi(getenv("MASK"));
    if (getenv("RATE")) p.expected_rate = atof(getenv("RATE"));
    auto res = singlet::detect_doublets(m, cells, p, nullptr, 0, &sm);
    std::vector<double> s(n); int tp = 0, fp = 0; for (uint32_t i = 0; i < n; ++i) { s[i] = res[i].score; if (res[i].is_doublet) (y[i] ? tp : fp)++; }
    std::printf("n=%u dbl=%u AUROC=%.4f called=%u (rate %.2f%%) tp=%d fp=%d null=%.3f h=%.2f rho=%.4f hvg=%u pcs=%u k=%u sims=%u time=%.2fs\n",
        n, n_dbl, auroc(s, y), sm.n_called, 100.0 * sm.n_called / n, tp, fp, sm.null_score, sm.detectable_fraction, sm.expected_rate, sm.n_hvg, sm.n_pcs, sm.k, sm.n_sim, sm.seconds);
    std::printf("sim_hist:"); for (auto v : sm.sim_hist) std::printf(" %u", v);
    std::printf("\nobs_hist:"); for (auto v : sm.obs_hist) std::printf(" %u", v);
    std::printf("\n");
}
