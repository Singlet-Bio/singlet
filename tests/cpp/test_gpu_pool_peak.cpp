// test_gpu_pool_peak.cpp: the gpu-pool's peak-model figures (include/singlet/gpu_pool_peak.h) against the 18-class table of
// bench/gpu-runtime/t2-peakmodel-predictor.json and the phase table of docs/gpu-engine/runtime/T2-peakmodel.md section 2.
// Classes: n (M reads), bases (G, one decimal as in the doc), P_B from the json, then F0 / F1 / F2 as printed in section 2.
#include <cmath>
#include <cstdio>
#include <string>

#include "singlet/gpu_pool_peak.h"

using namespace singlet::poolpeak;

static int g_fail = 0, g_pass = 0;
#define CHECK(expr, name) do { if (expr) { ++g_pass; } else { ++g_fail; std::fprintf(stderr, "  FAIL: %s\n", name); } } while (0)
static bool near(double a, double b, double tol) { return std::fabs(a - b) <= tol; }

struct Row { const char* s; double n, bases, P, f0, f1, f2; };
static const Row kRows[] = {
    {"pbmc2m",           2.0,   0.2,   6.958775095452039,  19.0,  19.0,  18.1},
    {"pbmc10m",         10.0,   0.9,   8.376500195083558,  20.4,  20.4,  18.8},
    {"pbmc1k_cite_v3",  45.4,   4.1,  14.65939297901672,   26.7,  26.7,  22.0},
    {"pbmc5p_v1",       60.0,   5.5,  17.286596335211403,  29.3,  29.3,  23.3},
    {"hgmm1k_v3",       63.1,   5.7,  17.77504157983312,   29.8,  29.8,  23.5},
    {"pbmc1k_v3",       66.6,   6.0,  18.407442590743447,  30.4,  30.4,  23.8},
    {"hgmm1k_v2",       75.2,   6.8,  47.7579721184962,    59.8,  59.8,  38.5},
    {"neuron1k_v3",     92.9,   8.4,  57.552008931123254,  69.6,  69.6,  43.4},
    {"d100s_3p",       100.0,  14.9,  64.24640914747998,   76.2,  76.2,  46.8},
    {"nuclei2k_v2",    118.7,  11.6,  72.29019053610084,   84.3,  84.1,  50.8},
    {"d300_5p",        299.6,  27.3,  59.949189418226226,  71.9,  71.9,  44.6},
    {"d300_3p",        300.2,  27.2,  59.8901637967477,    71.9,  71.9,  44.6},
    {"hgmm5k_v3",      313.0,  28.3,  62.105304267819406,  74.1,  74.1,  45.7},
    {"d500s_3p",       500.5,  74.7, 126.54029981657817,  138.5, 133.0,  77.9},
    {"d700s_3p",       703.4,  62.8, 130.58763484178246,  142.6, 136.6,  79.9},
    {"d1000_3p",       999.8,  88.5, 181.97625993747488,  194.0, 182.9, 105.6},
    {"d1000s_3p",     1004.6,  91.2, 185.2389927371849,   197.2, 185.8, 107.3},
    {"d2600_3p",      1303.7, 115.7, 235.61537408870322,  247.6, 231.1, 132.5},
};

int main() {
    const Consts k;   // the defaults, not the environment
    CHECK(near(k.margin, 12.0, 1e-12), "default margin 12");
    for (const Row& r : kRows) {
        const double P = peakP(r.n, r.bases, k);
        // the doc's n and bases are rounded to one decimal: 0.05 G of bases is 0.053 GiB of P, 0.05 M reads 0.004
        const std::string nm = r.s;
        CHECK(near(P, r.P, 0.15), (nm + ": P_B within rounding of the json").c_str());
        // phase figures from the json's exact P: the doc prints them to 0.1
        CHECK(near(phaseFigure(r.P, 0, k), r.f0, 0.06), (nm + ": F0").c_str());
        CHECK(near(phaseFigure(r.P, 1, k), r.f1, 0.06), (nm + ": F1").c_str());
        CHECK(near(phaseFigure(r.P, 2, k), r.f2, 0.06), (nm + ": F2").c_str());
        CHECK(phaseFigure(r.P, 1, k) <= phaseFigure(r.P, 0, k) + 1e-12 && phaseFigure(r.P, 2, k) <= phaseFigure(r.P, 1, k) + 1e-12, (nm + ": figures never grow with the phase").c_str());
        CHECK(inZone(r.n, k) == (r.n >= 70.0 && r.n < 160.0), (nm + ": zone").c_str());
    }
    // --chimeric (T2-chimpass.md): every phase figure is at least 0.5 P + 2.6432 + 19.5 + 12; never lower than without it; chimGb 0 gives the floor 0.5 P + 2.6 + margin = F2
    for (const Row& r : kRows)
        for (int ph = 0; ph < 3; ++ph) CHECK(phaseFigure(r.P, ph, k, true) >= phaseFigure(r.P, ph, k) - 1e-12 && phaseFigure(r.P, ph, k, true) >= 0.5 * r.P + 22.1432 + 12.0 - 1e-6, (std::string(r.s) + ": chimeric figure is a floor").c_str());
    {
        // the three recordings of T2-chimpass.md with their real n and cDNA bases (the "[1fq-ingest] Scanned" lines): pbmc5p_v1 60.000 M / 5.460 G, GSM6250953 121.366 M / 18.205 G, d300_5p 299.628 M / 27.266 G
        const double P5 = peakP(60.0, 5.46, k), Pc = peakP(299.628383, 27.266182853, k), Pg = peakP(121.366072, 18.2049108, k), PgLin = pLinear(121.366072, 18.2049108, k);
        CHECK(near(phaseFigure(P5, 2, k, true), 0.5 * P5 + 22.1432 + 12.0, 1e-4) && near(phaseFigure(P5, 2, k, true), 42.8, 0.1), "chimeric pbmc5p_v1 F2");
        CHECK(near(phaseFigure(Pc, 2, k, true), 64.1, 0.1), "chimeric d300_5p F2");
        CHECK(near(phaseFigure(PgLin, 2, k, true), 52.0, 0.1) && near(phaseFigure(Pg, 2, k, true), 72.5, 0.1), "chimeric GSM6250953 F2 (after the first engine line, and at the start in the zone)");
        // T2 chimshare: the floor adds chimSharedGb when sharing is in effect (the default), chimGb when SINGLET_CHIM_SHARE_GENOME=0
        Consts sh; sh.chimSharedGb = 30.0;
        CHECK(near(phaseFigure(P5, 2, sh, true), 0.5 * P5 + 2.6432 + 30.0 + 12.0, 1e-4), "chimeric shared: floor uses chimSharedGb");
        sh.chimShare = false;
        CHECK(near(phaseFigure(P5, 2, sh, true), 0.5 * P5 + 22.1432 + 12.0, 1e-4), "chimeric not shared: floor uses chimGb");
        CHECK(Consts().chimShare && near(Consts().chimEff(), Consts().chimSharedGb, 1e-12), "sharing is the default");
        Consts z; z.chimGb = 0.0; z.chimSharedGb = 0.0;
        CHECK(near(phaseFigure(P5, 2, z, true), phaseFigure(P5, 2, z), 1e-9), "chimGb 0: the floor equals F2");
    }
    // zone: the four classes the doc over-predicts (fallback possible) are above the linear predictor
    CHECK(peakP(75.2, 6.8, k) > pLinear(75.2, 6.8, k) + 20.0, "zone raises hgmm1k_v2");
    CHECK(near(peakP(69.9, 6.3, k), pLinear(69.9, 6.3, k), 1e-12), "just below the zone: linear");
    CHECK(near(peakP(160.0, 14.4, k), pLinear(160.0, 14.4, k), 1e-12), "the zone is half open: 160 M is linear");
    // fallback formula with the real D: d100s_3p planes 9.2 + 6.06 + 0.70 x 61.83 = 58.5, nuclei2k_v2 70.0 (doc 1.3)
    CHECK(near(fallbackP(100.0, 14.9, 61.83, k), 58.5, 0.3), "fallback d100s_3p");
    CHECK(near(fallbackP(118.7, 11.6, 79.81, k), 70.0, 0.3), "fallback nuclei2k_v2");
    // the ledger on d500s_3p: planes 46.0 + region 41.7 + 0.9 x 11.35 + late 4.65 + 6.06 + 0.0238 x 500.5 = 120.5 (doc 1.3; region 42746 MiB, solo keep 11.35 GB, late 4.65 GiB)
    CHECK(near(ledgerL(500.5, 74.7, 42746.0 * 1048576.0, 11.35e9, 4.65 * kGiB, k), 120.5, 0.5), "ledger d500s_3p");
    // d2600_3p: 83.0 + 67.6 + 0.9 x 27.72 + 23.8 + 6.06 + 0.0238 x 1303.7 = 236.4
    CHECK(near(ledgerL(1303.7, 115.7, 69199.0 * 1048576.0, 27.72e9, 23.7573 * kGiB, k), 236.4, 0.6), "ledger d2600_3p");
    // the engine's fallback expectation: needs about 35.07 / 43.93 GiB against 31 for the two that fell back; neuron1k_v3 (D 34.83 M) and the 300 M classes did not
    CHECK(fallbackExpected(false, 61.83, 31.0, k), "fb expected d100s_3p");
    CHECK(fallbackExpected(false, 79.81, 31.0, k), "fb expected nuclei2k_v2");
    CHECK(!fallbackExpected(false, 34.83, 31.0, k), "no fb neuron1k_v3");
    CHECK(!fallbackExpected(false, 22.30, 31.0, k), "no fb hgmm1k_v2");
    CHECK(!fallbackExpected(true, 205.98, 31.0, k), "no fb in read chunks");
    CHECK(!fallbackExpected(false, 200.0, 0.0, k), "no budget, no fb");
    // fitValid: range and regime
    std::string why;
    CHECK(fitValid(300.0, 31.0, 2.0, false, k, why), "fit valid at 31 / 2");
    CHECK(!fitValid(1400.1, 31.0, 2.0, false, k, why) && why.find("range") != std::string::npos, "n above 1400 M");
    CHECK(fitValid(1400.0, 31.0, 2.0, false, k, why), "n = 1400 M is inside");
    CHECK(!fitValid(300.0, 31.0, 2.0, true, k, why) && why.find("hostcut") != std::string::npos, "a hostcut switch off");
    CHECK(!fitValid(300.0, 40.0, 2.0, false, k, why), "memory 40");
    CHECK(fitValid(300.0, 31.5, 2.0, false, k, why), "memory 31.5 is within the tolerance");
    CHECK(!fitValid(300.0, -1.0, 2.0, false, k, why), "memory unset");
    CHECK(!fitValid(300.0, 31.0, 1.0, false, k, why), "slots 1");
    CHECK(!fitValid(300.0, 31.0, 3.0, false, k, why), "slots 3");
    std::fprintf(stderr, "test_gpu_pool_peak: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
