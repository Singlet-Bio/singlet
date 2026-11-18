// test_demux_fixedpoint.cpp: the exact fixed-point exit of run_vb_inplace (SINGLET_DEMUX_FIXEDPOINT) leaves every output
// (q, beta, ELBO bits, n_iter) bitwise equal to the full loop, for K = 1..4 and for several min_iter / max_iter pairs.
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include "singlet/pileup/donor_demux.h"

using namespace singlet;

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) \
    do { if (cond) { ++n_pass; } else { ++n_fail; std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; } } while (0)

static CoveredSNPData make_data(int nc, int ns, int ndonor, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    std::vector<std::vector<float>> af(ndonor, std::vector<float>(ns));
    for (auto& d : af) for (auto& v : d) v = U(rng) < 0.5f ? 0.05f : 0.95f;
    CoveredSNPData cd;
    cd.n_cells = nc; cd.n_covered_snps = ns;
    cd.cells.resize(nc);
    cd.covered_to_original.resize(ns);
    for (int j = 0; j < ns; ++j) cd.covered_to_original[j] = j;
    for (int c = 0; c < nc; ++c) {
        const int dn = static_cast<int>(rng() % ndonor);
        if (c % 17 == 5) continue;                       // a cell without SNPs
        for (int j = 0; j < ns; ++j) {
            if (U(rng) > 0.15f) continue;
            const int dp = 1 + static_cast<int>(rng() % 4);
            int ad = 0;
            for (int t = 0; t < dp; ++t) if (U(rng) < af[dn][j]) ++ad;
            cd.cells[c].snp_idx.push_back(j);
            cd.cells[c].ad.push_back(static_cast<uint16_t>(ad));
            cd.cells[c].dp.push_back(static_cast<uint16_t>(dp));
        }
    }
    return cd;
}

static VBResult make_init(const CoveredSNPData& d, int K, uint64_t seed) {
    VBResult r;
    r.elbo = -std::numeric_limits<double>::infinity();
    r.n_iter = 0;
    r.beta_mu.assign(d.n_covered_snps, std::vector<float>(K, 0.5f));
    r.beta_sum.assign(d.n_covered_snps, std::vector<float>(K, 30.0f));
    r.id_prob.assign(d.n_cells, std::vector<float>(K));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    for (int c = 0; c < d.n_cells; ++c) {
        float s = 0.0f;
        for (int k = 0; k < K; ++k) { r.id_prob[c][k] = U(rng); s += r.id_prob[c][k]; }
        const float inv = 1.0f / s;
        for (int k = 0; k < K; ++k) r.id_prob[c][k] *= inv;
    }
    return r;
}

int main() {
    const int pairs[][2] = {{20, 300}, {20, 100}, {5, 50}, {20, 10}, {1, 30}, {20, 20}, {20, 21}, {20, 22}, {3, 8}};
    long long skipped_total = 0;
    for (int nd : {1, 3}) {
        CoveredSNPData data = make_data(400, 300, nd, 7 + nd);
        for (int K = 1; K <= 4; ++K) {
            for (const auto& p : pairs) {
                DonorDemuxConfig cfg;
                cfg.min_iter = p[0];
                const int mx = p[1];
                for (int nt : {1, 3}) {
                    VBResult a = make_init(data, K, 99 + K), b = a, c = a;
                    run_vb_inplace_mode(data, K, cfg, a, mx, nt, 0);
                    const long long s0 = demux_fp_stats().iters_skipped.load();
                    run_vb_inplace_mode(data, K, cfg, b, mx, nt, 1);
                    skipped_total += demux_fp_stats().iters_skipped.load() - s0;
                    const long long d0 = demux_fp_stats().check_differ.load();
                    run_vb_inplace_mode(data, K, cfg, c, mx, nt, 2);
                    CHECK(vb_result_bits_equal(a, b), "skip == full (bitwise)");
                    CHECK(demux_fp_stats().check_differ.load() == d0, "check mode: 0 differ");
                    CHECK(vb_result_bits_equal(a, c), "check-mode result == full");
                }
            }
        }
    }
    CHECK(skipped_total > 0, "at least one fit skipped iterations");
    std::cout << "iterations skipped over the test fits: " << skipped_total << "\n";
    std::cout << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
