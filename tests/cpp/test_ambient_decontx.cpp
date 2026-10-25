// SPDX-License-Identifier: MIT
// test_ambient_decontx.cpp -- gene-level cluster-EM ambient estimator and integer corrected matrix
// (ambient_correction.h: correct_ambient_gene_level).
//
// Synthetic droplets with a known contamination fraction per cell:
//   3 cell types (300 marker genes each + 300 shared genes), ambient profile = the population mixture,
//   cell c = (1 - rho_c) native molecules + rho_c ambient molecules, rho_c ~ U(0.01, 0.15).
// Checks: the estimate tracks the truth (Pearson / Spearman, mean ratio), the correction removes about the
// ambient molecules and few native ones, corrected totals are integers that equal N - removed with no gene above its
// observed count, only called cells have a column, the result does not depend on the thread count, and a library with
// no empty droplets is returned unchanged.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

#include "singlet/pileup/ambient_correction.h"
#include "test_harness.h"

using namespace singlet;
using ambient_detail::SplitMix64;

namespace {

constexpr uint32_t G = 1200, TYPES = 3, MARK = 300, SHARED0 = 900;

struct Synth {
    CSCu32 counts;                       // genes x barcodes (cells + empties, shuffled)
    std::vector<uint32_t> cell_idx;      // barcode columns of the cells
    std::vector<double> rho_true;        // per cell (cell_idx order)
    std::vector<std::vector<uint32_t>> amb;  // per cell: ambient molecules per gene (dense G)
};

size_t draw(const std::vector<double>& cdf, SplitMix64& rng) {
    const double u = rng.uniform() * cdf.back();
    return static_cast<size_t>(std::lower_bound(cdf.begin(), cdf.end(), u) - cdf.begin());
}

Synth make_synth(bool with_empties, uint64_t seed) {
    SplitMix64 rng(seed);
    std::vector<std::vector<double>> prof(TYPES, std::vector<double>(G, 0.0));
    for (uint32_t t = 0; t < TYPES; ++t) {
        for (uint32_t g = t * MARK; g < (t + 1) * MARK; ++g) prof[t][g] = std::exp(2.0 * (rng.uniform() - 0.5) * 2.0);
        for (uint32_t g = SHARED0; g < G; ++g) prof[t][g] = std::exp(2.0 * (rng.uniform() - 0.5) * 2.0) * 0.8;
        double s = 0; for (double v : prof[t]) s += v;
        for (double& v : prof[t]) v /= s;
    }
    // shared genes get identical weights across types
    for (uint32_t g = SHARED0; g < G; ++g) { prof[1][g] = prof[0][g]; prof[2][g] = prof[0][g]; }
    std::vector<double> amb(G, 0.0);
    const double mix[TYPES] = {0.4, 0.4, 0.2};
    for (uint32_t t = 0; t < TYPES; ++t) for (uint32_t g = 0; g < G; ++g) amb[g] += mix[t] * prof[t][g];
    auto cdf_of = [](const std::vector<double>& p) {
        std::vector<double> c(p.size()); double s = 0; for (size_t i = 0; i < p.size(); ++i) { s += p[i]; c[i] = s; } return c; };
    std::vector<std::vector<double>> cdf(TYPES);
    for (uint32_t t = 0; t < TYPES; ++t) cdf[t] = cdf_of(prof[t]);
    const auto cdf_amb = cdf_of(amb);

    const uint32_t per_type = 150, n_cells = per_type * TYPES, n_empty = with_empties ? 3000 : 0, B = n_cells + n_empty;
    // barcode order: deterministic shuffle of cell / empty roles
    std::vector<uint32_t> perm(B);
    std::iota(perm.begin(), perm.end(), 0);
    for (uint32_t i = B; i > 1; --i) std::swap(perm[i - 1], perm[rng.next() % i]);

    Synth S;
    S.cell_idx.assign(perm.begin(), perm.begin() + n_cells);
    std::sort(S.cell_idx.begin(), S.cell_idx.end());
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> cols(B);   // (gene, count)
    auto flush = [&](uint32_t bc, std::vector<uint32_t>& dense) {
        for (uint32_t g = 0; g < G; ++g) if (dense[g]) { cols[bc].emplace_back(g, dense[g]); dense[g] = 0; }
    };
    std::vector<uint32_t> dense(G, 0);
    // cells in cell_idx order: type = i / per_type, interleaved by index permutation
    S.rho_true.resize(n_cells); S.amb.assign(n_cells, {});
    for (uint32_t i = 0; i < n_cells; ++i) {
        const uint32_t t = i % TYPES;
        const double rho = 0.01 + 0.14 * rng.uniform();
        const uint32_t N = 2500 + static_cast<uint32_t>(rng.uniform() * 3500);
        S.rho_true[i] = rho;
        S.amb[i].assign(G, 0);
        for (uint32_t m = 0; m < N; ++m) {
            if (rng.uniform() < rho) { size_t g = draw(cdf_amb, rng); ++dense[g]; ++S.amb[i][g]; }
            else                     { size_t g = draw(cdf[t], rng);  ++dense[g]; }
        }
        flush(S.cell_idx[i], dense);
    }
    // empties
    std::vector<char> is_cell(B, 0);
    for (uint32_t b : S.cell_idx) is_cell[b] = 1;
    for (uint32_t b = 0; b < B; ++b) {
        if (is_cell[b]) continue;
        const uint32_t N = 5 + static_cast<uint32_t>(rng.uniform() * 55);
        for (uint32_t m = 0; m < N; ++m) ++dense[draw(cdf_amb, rng)];
        flush(b, dense);
    }
    S.counts.nrows = G; S.counts.ncols = B; S.counts.indptr.assign(B + 1, 0);
    for (uint32_t b = 0; b < B; ++b) {
        for (auto& [g, v] : cols[b]) { S.counts.indices.push_back(static_cast<int32_t>(g)); S.counts.data.push_back(v); }
        S.counts.indptr[b + 1] = static_cast<int32_t>(S.counts.indices.size());
    }
    return S;
}

std::vector<double> ranks(const std::vector<double>& v) {
    std::vector<size_t> o(v.size()); std::iota(o.begin(), o.end(), 0);
    std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
    std::vector<double> r(v.size());
    for (size_t i = 0; i < o.size(); ++i) r[o[i]] = static_cast<double>(i);
    return r;
}
double pearson(const std::vector<double>& a, const std::vector<double>& b) {
    const double n = static_cast<double>(a.size());
    double ma = 0, mb = 0; for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); ++i) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb); }
    return sab / std::sqrt(saa * sbb);
}

}  // namespace

int main() {
    std::cout << "=== ambient decontx (gene-level cluster EM) ===\n";
    Synth S = make_synth(true, 7);
    AmbientDecontConfig cfg;
    cfg.lower_umi = 100;
    cfg.threads = 4;
    AmbientCorrectionResult R = correct_ambient_gene_level(S.counts, S.cell_idx, cfg);

    CHECK(R.ok, "estimator ran (ambient pool found)");
    CHECK(R.contam.size() == S.cell_idx.size(), "one contamination record per called cell");
    CHECK(R.n_clusters >= 2 && R.n_clusters <= 24, "auto clustering produced 2..24 clusters");
    CHECK(R.profile.n_ambient_barcodes == 3000, "ambient pool = the 3000 empty droplets");

    std::vector<double> est, tru;
    for (size_t i = 0; i < R.contam.size(); ++i) { est.push_back(R.contam[i].rho); tru.push_back(S.rho_true[i]); }
    const double pr = pearson(est, tru), sp = pearson(ranks(est), ranks(tru));
    double me = 0, mt = 0; for (size_t i = 0; i < est.size(); ++i) { me += est[i]; mt += tru[i]; }
    me /= est.size(); mt /= est.size();
    std::cout << "  pearson=" << pr << " spearman=" << sp << " mean_est=" << me << " mean_true=" << mt
              << " clusters=" << R.n_clusters << " removed=" << R.umis_removed << "/" << R.umis_observed << "\n";
    CHECK(pr > 0.85, "estimate correlates with the true rho (Pearson > 0.85)");
    CHECK(sp > 0.85, "estimate correlates with the true rho (Spearman > 0.85)");
    CHECK(me > 0.75 * mt && me < 1.25 * mt, "mean estimated rho within 25% of the truth");
    CHECK(std::fabs(R.mean_rho - me) < 1e-9, "result.mean_rho is the mean over called cells");
    CHECK(R.mean_rho > 0.0, "estimated_contamination_fraction is non-zero");

    // corrected matrix: integer totals, bounds, structure
    const CSCu32& C = R.corrected;
    CHECK(C.nrows == G && C.ncols == S.counts.ncols, "corrected matrix keeps genes x barcodes shape");
    bool structure_ok = true, totals_ok = true, bound_ok = true;
    std::vector<char> is_cell(S.counts.ncols, 0);
    for (uint32_t b : S.cell_idx) is_cell[b] = 1;
    double amb_total = 0, amb_removed = 0, nat_total = 0, nat_removed = 0;
    for (size_t i = 0; i < S.cell_idx.size(); ++i) {
        const uint32_t b = S.cell_idx[i];
        std::vector<uint32_t> obs(G, 0), cor(G, 0);
        for (int32_t k = S.counts.indptr[b]; k < S.counts.indptr[b + 1]; ++k) obs[S.counts.indices[k]] = S.counts.data[k];
        for (int32_t k = C.indptr[b]; k < C.indptr[b + 1]; ++k) cor[C.indices[k]] = C.data[k];
        uint64_t so = 0, sc = 0;
        for (uint32_t g = 0; g < G; ++g) {
            so += obs[g]; sc += cor[g];
            if (cor[g] > obs[g]) bound_ok = false;
            const double rem = obs[g] - cor[g], a = S.amb[i][g], nat = obs[g] - a;
            amb_total += a; nat_total += nat;
            amb_removed += std::min(rem, a);
            nat_removed += std::max(0.0, rem - a);
        }
        if (so - sc != R.contam[i].n_removed) totals_ok = false;
        if (R.contam[i].n_umi != so) totals_ok = false;
    }
    for (uint32_t b = 0; b < C.ncols; ++b)
        if (!is_cell[b] && C.indptr[b + 1] != C.indptr[b]) structure_ok = false;
    CHECK(structure_ok, "non-called barcodes have empty columns");
    CHECK(bound_ok, "no corrected count exceeds the observed count");
    CHECK(totals_ok, "per-cell integer totals equal N - n_removed");
    const double removed_frac = (amb_removed + nat_removed) / amb_total;
    std::cout << "  ambient molecules removed=" << amb_removed / amb_total << " native removed=" << nat_removed / nat_total
              << " total_removed/ambient=" << removed_frac << "\n";
    CHECK(amb_removed / amb_total > 0.45, "removes at least 45% of the true ambient molecules");
    CHECK(nat_removed / nat_total < 0.03, "removes < 3% of the native molecules");
    CHECK(removed_frac > 0.5 && removed_frac < 1.15, "removed UMIs are about the ambient UMIs (0.5..1.15)");
    CHECK(std::fabs(R.umi_weighted_rho - static_cast<double>(R.umis_removed) / R.umis_observed) < 1e-12, "umi_weighted_rho consistent");

    // determinism across thread counts
    cfg.threads = 1;
    AmbientCorrectionResult R1 = correct_ambient_gene_level(S.counts, S.cell_idx, cfg);
    bool same = R1.corrected.indices == R.corrected.indices && R1.corrected.data == R.corrected.data;
    for (size_t i = 0; i < R.contam.size() && same; ++i) same = R1.contam[i].rho == R.contam[i].rho;
    CHECK(same, "identical output with 1 and 4 threads");
    cfg.threads = 3;
    AmbientCorrectionResult R3 = correct_ambient_gene_level(S.counts, S.cell_idx, cfg);
    CHECK(R3.corrected.data == R.corrected.data, "identical output with 3 threads");

    // no empty droplets -> identity, rho 0
    {
        Synth N = make_synth(false, 11);
        AmbientCorrectionResult Rn = correct_ambient_gene_level(N.counts, N.cell_idx, cfg);
        CHECK(!Rn.ok, "no ambient pool -> ok == false");
        CHECK(Rn.mean_rho == 0.0, "no ambient pool -> rho 0");
        CHECK(Rn.corrected.data == N.counts.data && Rn.corrected.indices == N.counts.indices, "no ambient pool -> counts unchanged");
    }

    // tiny input (< 40 cells -> single cluster) does not crash
    {
        Synth T = make_synth(true, 5);
        std::vector<uint32_t> few(T.cell_idx.begin(), T.cell_idx.begin() + 10);
        AmbientCorrectionResult Rt = correct_ambient_gene_level(T.counts, few, cfg);
        CHECK(Rt.n_clusters == 1 && Rt.contam.size() == 10, "10 cells -> one cluster, 10 records");
    }

    // JSON / TSV writers run
    {
        write_ambient_json("ambient_decontx_test.json", R, cfg);
        std::vector<std::string> bcs(S.counts.ncols, "BC");
        write_contamination_tsv("ambient_decontx_test.tsv", R.contam, S.cell_idx, bcs);
        std::remove("ambient_decontx_test.json");
        std::remove("ambient_decontx_test.tsv");
        CHECK(true, "writers ran");
    }

    std::cout << "\nTotal: " << (n_pass + n_fail) << "  PASS: " << n_pass << "  FAIL: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
