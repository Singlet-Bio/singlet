// test_demux_autok_fast.cpp: the fast auto-K path (progressive PCA, folds prepared side by side, wider waves, worker pool;
// SINGLET_DEMUX_AUTOK_FAST) gives the chosen K and every number of the k_selection trace bit for bit as the earlier path, on
// single-donor and multi-donor synthetic data, for several thread counts; and a progressive PCA cache holds the scores a
// full computation gives.
#include <cstdio>
#include <cstring>
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
    for (auto& d : af) for (auto& v : d) v = U(rng) < 0.5f ? 0.03f : 0.97f;
    CoveredSNPData cd;
    cd.n_cells = nc; cd.n_covered_snps = ns;
    cd.cells.resize(nc);
    cd.covered_to_original.resize(ns);
    for (int j = 0; j < ns; ++j) cd.covered_to_original[j] = j;
    for (int c = 0; c < nc; ++c) {
        const int dn = static_cast<int>(rng() % ndonor);
        if (c % 23 == 5) continue;                       // a cell without SNPs
        for (int j = 0; j < ns; ++j) {
            if (U(rng) > 0.2f) continue;
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

static bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

int main() {
    // 1. progressive PCA == full PCA, column by column
    {
        CoveredSNPData d = make_data(500, 400, 3, 11);
        const int npc = 12;
        PcaCache full = compute_pca_cache(d, npc, 3, 123);
        PcaCache lazy = compute_pca_cache(d, npc, 1, 123, 4);
        CHECK(lazy.ready == 4, "progressive cache starts with 4 PCs");
        lazy.ensure(7, 2);
        lazy.ensure(npc, 5);
        CHECK(lazy.ready == npc, "progressive cache complete");
        CHECK(full.scores.size() == lazy.scores.size() &&
              std::memcmp(full.scores.data(), lazy.scores.data(), full.scores.size() * sizeof(float)) == 0,
              "progressive PCA scores bitwise equal to the full computation");
        PcaCache part = compute_pca_cache(d, npc, 2, 123, 6);
        bool eq = true;
        for (int c = 0; c < d.n_cells; ++c)
            for (int p = 0; p < 6; ++p)
                eq = eq && std::memcmp(&part.scores[static_cast<size_t>(c) * npc + p], &full.scores[static_cast<size_t>(c) * npc + p], 4) == 0;
        CHECK(eq, "first 6 PCs of a partial cache equal those of the full one");
    }
    // 2. auto-K both ways
    for (int nd : {1, 3, 5}) {
        CoveredSNPData data = make_data(900, 500, nd, 40 + nd);
        for (int threads : {1, 4, 16}) {
            DonorDemuxConfig cfg;
            cfg.threads = threads;
            cfg.max_donors = 8;
            std::vector<KSelectRow> tf, ts;
            demux_autok_fast_override() = 1;
            const int kf = select_k_heldout(data, cfg, &tf);
            demux_autok_fast_override() = 0;
            const int ks = select_k_heldout(data, cfg, &ts);
            demux_autok_fast_override() = -1;
            std::cout << "donors=" << nd << " threads=" << threads << ": chosen K fast=" << kf << " old=" << ks
                      << " trace rows " << tf.size() << " / " << ts.size() << "\n";
            CHECK(kf == ks, "chosen K equal");
            CHECK(tf.size() == ts.size(), "trace length equal");
            for (size_t i = 0; i < std::min(tf.size(), ts.size()); ++i) {
                CHECK(tf[i].K == ts[i].K && tf[i].valid == ts[i].valid && tf[i].accepted == ts[i].accepted, "trace row K/valid/accepted");
                CHECK(same_bits(tf[i].ll_per_cell, ts[i].ll_per_cell) && same_bits(tf[i].gain, ts[i].gain) && same_bits(tf[i].z, ts[i].z),
                      "trace row ll/gain/z bitwise equal");
            }
            if (nd == 1) CHECK(kf == 1, "single donor: K = 1");
        }
    }
    std::cout << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
