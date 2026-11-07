// SPDX-License-Identifier: MIT
// test_doublet_calibration.cpp -- planted-doublet checks for the v7 Scrublet-style scorer and its rate-calibrated call
// (include/singlet/pileup/doublet_detect.h).
//
//  C1: planted heterotypic doublets: AUROC >= 0.95, precision of the calls >= 0.9, score is continuous (few exact ties at 0).
//  C2: the call rate follows the expected multiplet rate (never above rho = 0.8% per 1,000 cells), not the score distribution.
//  C3: an all-singlet library is not flagged beyond that rate.
//  C4: deterministic: identical scores and calls for 1 and 5 threads, and across repeated runs.
//  C5: exon-level input with a feature->gene map equals the gene-level matrix exactly.
//  C6: explicit expected_rate overrides the 10x rule; rate cap of 25%.
#include "singlet/pileup/doublet_detect.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

using CSC = singlet::SparseAccumulator<uint16_t>::CSCMatrix;
struct Ent { int32_t r, c; uint16_t v; };

static CSC to_csc(uint32_t nrows, uint32_t ncols, std::vector<Ent>& ent) {
    std::stable_sort(ent.begin(), ent.end(), [](const Ent& a, const Ent& b) { return a.c < b.c; });
    CSC m; m.nrows = nrows; m.ncols = ncols; m.indptr.assign(ncols + 1, 0);
    for (auto& e : ent) m.indptr[e.c + 1]++;
    for (uint32_t c = 0; c < ncols; ++c) m.indptr[c + 1] += m.indptr[c];
    for (auto& e : ent) { m.indices.push_back(e.r); m.data.push_back(e.v); }
    return m;
}

static double auroc(const std::vector<double>& s, const std::vector<char>& y) {
    std::vector<size_t> o(s.size());
    std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return s[a] < s[b]; });
    double rs = 0, np = 0;
    for (size_t i = 0; i < o.size();) {
        size_t j = i;
        while (j + 1 < o.size() && s[o[j + 1]] == s[o[i]]) ++j;
        const double rk = 0.5 * (i + j) + 1;
        for (size_t t = i; t <= j; ++t) if (y[o[t]]) { rs += rk; ++np; }
        i = j + 1;
    }
    const double nn = o.size() - np;
    return (rs - np * (np + 1) / 2) / (np * nn);
}

// n_sing singlets over `types` cell types (type-specific program on top of a shared log-normal profile) plus n_dbl
// heterotypic doublets (sum of two independently drawn cells of different types).  Gene-level.
struct Synth { CSC m; std::vector<char> is_dbl; uint32_t n; };
static Synth make_synth(uint32_t n_sing, uint32_t n_dbl, int types, uint32_t G, uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::vector<double> base(G);
    std::lognormal_distribution<double> ln(0, 1.2);
    for (auto& b : base) b = ln(rng);
    std::vector<std::vector<double>> prof(types, base);
    std::uniform_int_distribution<int> gd(0, G - 1);
    for (int t = 0; t < types; ++t) for (int k = 0; k < 80; ++k) prof[t][gd(rng)] *= 6.0;
    for (auto& p : prof) { double s = 0; for (double v : p) s += v; for (double& v : p) v /= s; }
    const uint32_t n = n_sing + n_dbl;
    std::vector<Ent> ent;
    Synth out; out.n = n; out.is_dbl.assign(n, 0);
    std::lognormal_distribution<double> dep(std::log(2500.0), 0.3);
    auto draw = [&](int t, double depth, uint32_t col) {
        for (uint32_t g = 0; g < G; ++g) {
            const double mu = prof[t][g] * depth;
            if (mu < 1e-4) continue;
            std::poisson_distribution<int> pd(mu);
            const int v = pd(rng);
            if (v) ent.push_back({(int32_t)g, (int32_t)col, (uint16_t)v});
        }
    };
    for (uint32_t c = 0; c < n_sing; ++c) draw(c % types, dep(rng), c);
    for (uint32_t d = 0; d < n_dbl; ++d) {
        const int a = rng() % types, b = (a + 1 + rng() % (types - 1)) % types;
        const uint32_t col = n_sing + d;
        draw(a, dep(rng), col); draw(b, dep(rng), col);
        out.is_dbl[col] = 1;
    }
    out.m = to_csc(G, n, ent);
    return out;
}

static std::vector<uint32_t> all_cells(uint32_t n) { std::vector<uint32_t> v(n); std::iota(v.begin(), v.end(), 0u); return v; }

int main() {
    // 2400 singlets + 120 doublets (4.8%); 10x rule gives rho = 0.8% * 2.52 = 2.0%
    auto S = make_synth(2400, 120, 6, 2000, 11);
    const auto cells = all_cells(S.n);

    std::cout << "\n=== C1/C2: planted doublets, auto rate ===\n";
    singlet::DoubletSummary sm;
    auto res = singlet::detect_doublets(S.m, cells, singlet::DoubletParams(), nullptr, 0, &sm);
    std::vector<double> sc(S.n);
    int tp = 0, fp = 0, zeros = 0;
    for (uint32_t i = 0; i < S.n; ++i) {
        sc[i] = res[i].score;
        if (res[i].is_doublet) (S.is_dbl[i] ? tp : fp)++;
        if (res[i].score == 0.0) ++zeros;
    }
    const double au = auroc(sc, S.is_dbl);
    const int called = tp + fp;
    const double rho = 0.008 * S.n / 1000.0;
    std::cout << "  AUROC=" << au << " called=" << called << " tp=" << tp << " fp=" << fp << " rho=" << sm.expected_rate
              << " null_score=" << sm.null_score << " h=" << sm.detectable_fraction << " zeros=" << zeros
              << " hvg=" << sm.n_hvg << " pcs=" << sm.n_pcs << " time=" << sm.seconds << "s\n";
    CHECK(res.size() == S.n, "C1: one result per cell");
    CHECK(au >= 0.95, "C1: AUROC >= 0.95 on planted heterotypic doublets");
    CHECK(called > 0 && tp >= 0.9 * called, "C1: precision of the calls >= 0.9");
    CHECK(zeros < 0.05 * S.n, "C1: score is not dominated by exact zeros");
    CHECK(std::fabs(sm.expected_rate - rho) < 1e-9, "C2: auto rate = 0.8% per 1,000 cells");
    CHECK(called <= std::llround(rho * S.n) + 1, "C2: called cells never exceed rho * n");
    CHECK(called >= 0.5 * rho * S.n * 0.5, "C2: calls reach a fair share of the expected multiplets");

    std::cout << "\n=== C3: all-singlet library ===\n";
    auto P = make_synth(2400, 0, 6, 2000, 12);
    auto rp = singlet::detect_doublets(P.m, all_cells(P.n));
    int calls_p = 0;
    for (const auto& r : rp) calls_p += r.is_doublet ? 1 : 0;
    std::cout << "  calls=" << calls_p << " of " << P.n << "\n";
    CHECK(calls_p <= std::llround(0.008 * P.n / 1000.0 * P.n) + 1, "C3: no more calls than the expected multiplet rate");

    std::cout << "\n=== C4: determinism ===\n";
    singlet::DoubletParams p1; p1.threads = 1;
    singlet::DoubletParams p5; p5.threads = 5;
    auto r1 = singlet::detect_doublets(S.m, cells, p1);
    auto r5 = singlet::detect_doublets(S.m, cells, p5);
    auto r1b = singlet::detect_doublets(S.m, cells, p1);
    bool same = true, same_rep = true;
    for (uint32_t i = 0; i < S.n; ++i) {
        if (r1[i].score != r5[i].score || r1[i].is_doublet != r5[i].is_doublet) same = false;
        if (r1[i].score != r1b[i].score || r1[i].is_doublet != r1b[i].is_doublet) same_rep = false;
    }
    CHECK(same, "C4: identical scores and calls for 1 and 5 threads");
    CHECK(same_rep, "C4: identical across repeated runs");

    std::cout << "\n=== C5: exon-level input with a feature->gene map ===\n";
    // split every gene into two exon features (counts split in two); the map folds them back
    const uint32_t G = S.m.nrows;
    std::vector<Ent> ex;
    for (uint32_t c = 0; c < S.m.ncols; ++c)
        for (int32_t k = S.m.indptr[c]; k < S.m.indptr[c + 1]; ++k) {
            const uint16_t v = S.m.data[k], a = v / 2, b = v - a;
            if (a) ex.push_back({2 * S.m.indices[k], (int32_t)c, a});
            if (b) ex.push_back({2 * S.m.indices[k] + 1, (int32_t)c, b});
        }
    CSC E = to_csc(2 * G, S.n, ex);
    std::vector<uint32_t> f2g(2 * G);
    for (uint32_t f = 0; f < 2 * G; ++f) f2g[f] = f / 2;
    auto rg = singlet::detect_doublets(S.m, cells);
    auto re = singlet::detect_doublets(E, cells, singlet::DoubletParams(), &f2g, G);
    bool eq = true;
    for (uint32_t i = 0; i < S.n; ++i)
        if (rg[i].score != re[i].score || rg[i].is_doublet != re[i].is_doublet) eq = false;
    CHECK(eq, "C5: mapped exon matrix gives the same scores and calls as the gene matrix");

    std::cout << "\n=== C6: explicit rate and the rate cap ===\n";
    singlet::DoubletParams pr; pr.expected_rate = 0.05;
    singlet::DoubletSummary s6;
    auto r6 = singlet::detect_doublets(S.m, cells, pr, nullptr, 0, &s6);
    int c6 = 0;
    for (const auto& r : r6) c6 += r.is_doublet ? 1 : 0;
    std::cout << "  rate 0.05: called=" << c6 << " h=" << s6.detectable_fraction << "\n";
    CHECK(c6 <= std::llround(0.05 * S.n) + 1 && c6 > called, "C6: explicit expected_rate controls the call count");
    singlet::DoubletParams pc; pc.expected_rate = 0.9;
    singlet::DoubletSummary s7;
    singlet::detect_doublets(S.m, cells, pc, nullptr, 0, &s7);
    CHECK(s7.expected_rate <= 0.25 + 1e-12, "C6: expected rate capped at 25%");

    std::cout << "\n===========================\nPASSED: " << n_pass << "  FAILED: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
