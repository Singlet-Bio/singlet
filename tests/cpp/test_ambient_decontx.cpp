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

    // exon / intron split API: intron = a third of every exon count (same ambient shape); result folds back to genes
    {
        CSCu32 I = S.counts;
        for (auto& v : I.data) v = (v + 2) / 3;
        AmbientCorrectionResult Rs = correct_ambient_gene_level_split(S.counts, I, S.cell_idx, cfg);
        CHECK(Rs.ok && Rs.corrected.nrows == G && Rs.corrected.ncols == S.counts.ncols, "split API: genes x barcodes result");
        std::vector<double> es;
        for (const auto& c : Rs.contam) es.push_back(c.rho);
        CHECK(pearson(es, tru) > 0.85, "split API: rho tracks the truth");
        bool tot_ok = true, empty_ok = true;
        for (size_t i = 0; i < S.cell_idx.size(); ++i) {
            const uint32_t b = S.cell_idx[i];
            uint64_t so = 0, sc = 0;
            for (int32_t k = S.counts.indptr[b]; k < S.counts.indptr[b + 1]; ++k) so += S.counts.data[k];
            for (int32_t k = I.indptr[b]; k < I.indptr[b + 1]; ++k) so += I.data[k];
            for (int32_t k = Rs.corrected.indptr[b]; k < Rs.corrected.indptr[b + 1]; ++k) sc += Rs.corrected.data[k];
            if (so - sc != Rs.contam[i].n_removed) tot_ok = false;
        }
        for (uint32_t b = 0; b < Rs.corrected.ncols; ++b)
            if (!is_cell[b] && Rs.corrected.indptr[b + 1] != Rs.corrected.indptr[b]) empty_ok = false;
        CHECK(tot_ok, "split API: folded integer totals equal N - n_removed");
        CHECK(empty_ok, "split API: non-called barcodes empty");
        double fs = 0; for (double f : Rs.profile.gene_fractions) fs += f;
        CHECK(Rs.profile.gene_fractions.size() == G && std::fabs(fs - 1.0) < 1e-9, "split API: folded profile sums to 1");
    }

    // tiny input (< 40 cells -> single cluster) does not crash
    {
        Synth T = make_synth(true, 5);
        std::vector<uint32_t> few(T.cell_idx.begin(), T.cell_idx.begin() + 10);
        AmbientCorrectionResult Rt = correct_ambient_gene_level(T.counts, few, cfg);
        CHECK(Rt.n_clusters == 1 && Rt.contam.size() == 10, "10 cells -> one cluster, 10 records");
    }

    // Cell-set guard: duplicate and out-of-range indices are ignored by the fit; records stay aligned with the input
    {
        std::vector<uint32_t> cells = S.cell_idx;
        cells.push_back(S.cell_idx[0]);                 // duplicate
        cells.push_back(S.counts.ncols + 5);            // out of range
        AmbientCorrectionResult Rg = correct_ambient_gene_level(S.counts, cells, cfg);
        CHECK(Rg.contam.size() == cells.size(), "guard: one record per input index");
        CHECK(Rg.n_cells_dropped == 2, "guard: duplicate + out-of-range index counted");
        bool same = true;
        for (size_t i = 0; i < S.cell_idx.size(); ++i) same = same && Rg.contam[i].rho == R3.contam[i].rho;
        CHECK(same, "guard: the fit is unchanged by ignored indices");
        CHECK(Rg.contam[S.cell_idx.size()].rho == Rg.contam[0].rho, "guard: duplicate gets its first occurrence's record");
        CHECK(Rg.contam.back().rho == 0.0 && Rg.contam.back().n_umi == 0, "guard: out-of-range index gets rho 0");
        CHECK(R3.n_uncalled_above_pool == 0, "guard: full cell set leaves no uncalled barcode above the pool cap");
        // a knee-like subset (drop the 25% shallowest cells): they fall in neither the pool nor the fit, and are counted
        std::vector<std::pair<uint64_t, uint32_t>> depth;
        for (size_t i = 0; i < S.cell_idx.size(); ++i) {
            const uint32_t b = S.cell_idx[i];
            uint64_t t = 0;
            for (int32_t k = S.counts.indptr[b]; k < S.counts.indptr[b + 1]; ++k) t += S.counts.data[k];
            depth.emplace_back(t, b);
        }
        std::sort(depth.begin(), depth.end());
        std::vector<uint32_t> knee;
        for (size_t i = depth.size() / 4; i < depth.size(); ++i) knee.push_back(depth[i].second);
        std::sort(knee.begin(), knee.end());
        AmbientCorrectionResult Rk = correct_ambient_gene_level(S.counts, knee, cfg);
        CHECK(Rk.n_uncalled_above_pool == depth.size() / 4, "guard: a knee subset is flagged (uncalled barcodes above the pool cap)");
        CHECK(Rk.profile.n_ambient_barcodes == R3.profile.n_ambient_barcodes, "guard: dropped cells do not enter the ambient pool");
    }

    // Two-level shrinkage: per-cell prior toward the cluster c_k, cluster c_k toward the sample rho
    {
        AmbientDecontConfig c0 = cfg;
        c0.cluster_prior_reads = 0.0; c0.cluster_prior_cells = 0.0;
        AmbientCorrectionResult A0 = correct_ambient_gene_level(S.counts, S.cell_idx, c0);
        bool raw_ok = A0.cluster_rho.size() == A0.cluster_rho_raw.size();
        for (size_t k = 0; raw_ok && k < A0.cluster_rho.size(); ++k) raw_ok = std::fabs(A0.cluster_rho[k] - A0.cluster_rho_raw[k]) < 1e-12;
        CHECK(raw_ok, "shrinkage off: cluster c_k = raw c_k");
        AmbientDecontConfig c1 = cfg;
        c1.cluster_prior_reads = 1e12;
        AmbientCorrectionResult A1 = correct_ambient_gene_level(S.counts, S.cell_idx, c1);
        bool pooled_ok = A1.sample_rho > 0.0;
        for (double c : A1.cluster_rho) pooled_ok = pooled_ok && std::fabs(c / A1.sample_rho - 1.0) < 1e-3;
        CHECK(pooled_ok, "infinite cluster prior: every c_k equals the sample rho");
        bool prior_ok = true;
        for (const auto& c : A1.contam) prior_ok = prior_ok && std::fabs(c.prior_rho / A1.sample_rho - 1.0) < 1e-3;
        CHECK(prior_ok, "infinite cluster prior: every cell's prior mean is the sample rho");
        // per-cell prior strength: more pseudo-reads, less per-cell spread (information-weighted shrinkage)
        auto sd = [](const AmbientCorrectionResult& r) {
            double m = 0, v = 0;
            for (const auto& c : r.contam) m += c.rho_total > 0 ? c.rho_total : c.rho;
            m /= r.contam.size();
            for (const auto& c : r.contam) { const double x = c.rho_total > 0 ? c.rho_total : c.rho; v += (x - m) * (x - m); }
            return std::sqrt(v / r.contam.size());
        };
        AmbientDecontConfig cw = c1, cs = c1;
        cw.off_prior_reads = 1.0; cs.off_prior_reads = 1e9;
        const double sd_w = sd(correct_ambient_gene_level(S.counts, S.cell_idx, cw));
        const double sd_s = sd(correct_ambient_gene_level(S.counts, S.cell_idx, cs));
        std::cout << "  per-cell sd: weak prior " << sd_w << ", infinite prior " << sd_s << "\n";
        CHECK(sd_s < 0.05 * sd_w, "infinite per-cell prior collapses rho onto the sample rho");
        // minimum cluster size: small clusters dissolve, at least two remain
        AmbientDecontConfig cm = cfg;
        cm.min_cluster_cells = 1000;
        AmbientCorrectionResult Am = correct_ambient_gene_level(S.counts, S.cell_idx, cm);
        CHECK(Am.n_clusters == 2 && Am.ok, "min_cluster_cells dissolves small clusters but keeps two");
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
