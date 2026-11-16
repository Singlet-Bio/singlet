// SPDX-License-Identifier: MIT
// genotype_same_pairs (auto-K genotype-distance check): components with identical genotypes are one donor, components
// of unrelated donors are not, and thinly typed pairs get no verdict.
#include <cstdio>
#include <random>
#include "singlet/pileup/donor_demux.h"

using namespace singlet;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL line %d: %s\n", __LINE__, #c); ++fails; } } while (0)

int main() {
    const int K = 3, ns = 600;
    std::mt19937 rng(7);
    AlleleTables T;
    T.K = K; T.ns = ns;
    T.F.assign(static_cast<size_t>(ns) * K, 0.5f);
    for (int j = 0; j < ns; ++j) {
        const int g = static_cast<int>(rng() % 3);                 // donor 0's genotype class
        const float f0 = g == 0 ? 0.02f : g == 1 ? 0.5f : 0.98f;
        // donor 1: the same genotypes with a little allelic skew (het sites move to 0.4, still het)
        const float f1 = g == 1 ? 0.42f : f0;
        // donor 2: an unrelated genotype at every SNP
        const int h = static_cast<int>(rng() % 3);
        const float f2 = h == 0 ? 0.02f : h == 1 ? 0.5f : 0.98f;
        T.F[static_cast<size_t>(j) * K + 0] = f0;
        T.F[static_cast<size_t>(j) * K + 1] = f1;
        T.F[static_cast<size_t>(j) * K + 2] = f2;
    }
    DonorDemuxConfig cfg;
    std::vector<char> same;
    std::vector<double> dist;
    genotype_same_pairs(T, cfg, same, &dist);
    CHECK(same[0 * K + 1] == 1);                 // 0 and 1: same donor
    CHECK(dist[0 * K + 1] < 0.01);
    CHECK(same[0 * K + 2] == 0 && same[1 * K + 2] == 0);   // unrelated donor stays
    CHECK(dist[0 * K + 2] > 0.4);

    cfg.k_select_geno_dist_min = 0.0;            // switched off: no pair is merged
    genotype_same_pairs(T, cfg, same, &dist);
    CHECK(same[0 * K + 1] == 0);

    cfg = DonorDemuxConfig();
    cfg.k_select_geno_dist_sites = 10000;        // too few typed SNPs for a verdict
    genotype_same_pairs(T, cfg, same, &dist);
    CHECK(same[0 * K + 1] == 0 && dist[0 * K + 1] < 0.0);

    std::printf(fails ? "test_demux_geno_same: FAILED\n" : "test_demux_geno_same: ok\n");
    return fails ? 1 : 0;
}
