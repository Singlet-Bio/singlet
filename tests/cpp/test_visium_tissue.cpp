// SPDX-License-Identifier: MIT
// test_visium_tissue.cpp — unit tests for visium_tissue.h (slide spot list, barcode matching, tissue detection).

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "singlet/pileup/visium_tissue.h"

using namespace singlet;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    double u() {  // uniform in (0, 1)
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        return ((s >> 11) + 0.5) / 9007199254740992.0;
    }
    double normal() {
        const double a = u(), b = u();
        return std::sqrt(-2.0 * std::log(a)) * std::cos(2.0 * 3.14159265358979323846 * b);
    }
    uint64_t lognormal10(double center, double sd) {
        return static_cast<uint64_t>(std::pow(10.0, std::log10(center) + sd * normal()));
    }
};

std::string temp_path(const char* name) {
    const char* d = std::getenv("TMPDIR");
    if (!d) d = std::getenv("TEMP");
    std::string p = d ? d : ".";
    return p + "/" + name;
}

struct Grid {
    VisiumSlide slide;
    std::vector<uint8_t> truth;
    std::vector<uint64_t> umis;
};

constexpr double kRowC = 38.5, kColC = 63.5, kRowA = 28.0, kColA = 45.0;

double ell(int r, int c) {  // < 1 inside the ellipse
    const double a = (r - kRowC) / kRowA, b = (c - kColC) / kColA;
    return a * a + b * b;
}

Grid make_grid(uint64_t seed) {
    Grid g;
    Rng rng(seed);
    for (int r = 0; r < 78; ++r)
        for (int c = 0; c < 128; ++c) {
            if ((r + c) % 2) continue;
            g.slide.barcode.push_back("S" + std::to_string(g.slide.barcode.size()));
            g.slide.row.push_back(r);
            g.slide.col.push_back(c);
            const bool t = ell(r, c) < 1.0;
            g.truth.push_back(t);
            g.umis.push_back(rng.lognormal10(t ? 5000 : 100, 0.35));
        }
    return g;
}

// Choose k mutually non-adjacent spots with lo <= ell < hi.
std::vector<size_t> pick_isolated(const Grid& g, double lo, double hi, int k, uint64_t seed) {
    Rng rng(seed);
    std::vector<size_t> out;
    int guard = 0;
    while (static_cast<int>(out.size()) < k && guard++ < 1000000) {
        const size_t i = static_cast<size_t>(rng.u() * g.slide.size());
        const double e = ell(g.slide.row[i], g.slide.col[i]);
        if (e < lo || e >= hi) continue;
        bool ok = true;
        for (size_t j : out) {
            if (std::abs(g.slide.row[i] - g.slide.row[j]) <= 2 && std::abs(g.slide.col[i] - g.slide.col[j]) <= 4) {
                ok = false;
                break;
            }
        }
        if (ok) out.push_back(i);
    }
    assert(static_cast<int>(out.size()) == k);
    return out;
}

void test_slide_load_match() {
    const std::string path = temp_path("test_visium_slide.tsv");
    {
        std::ofstream f(path);
        f << "barcode\tarray_row\tarray_col\n";
        f << "ACGTACGTACGTACGT-1\t3\t7\n";
        f << "TTTTAAAACCCCGGGA\t4\t8\n";  // no suffix
        f << "AAAAAAAAAAAAAAAC\t0\t0\n";   // Hamming-1 ambiguity pair
        f << "AAAAAAAAAAAAAAAG\t0\t2\n";
    }
    VisiumSlide s;
    assert(s.load(path));
    assert(s.size() == 4);
    assert(s.row[0] == 3 && s.col[0] == 7);
    assert(s.row[1] == 4 && s.col[1] == 8);
    assert(s.barcode[0] == "ACGTACGTACGTACGT");  // suffix stripped
    assert(s.index.at("ACGTACGTACGTACGT") == 0);
    assert(s.index.size() == 4);
    assert(s.match("ACGTACGTACGTACGT") == 0);
    assert(s.match("ACGTACGTACGTACGT-1") == 0);
    assert(s.match("TTTTAAAACCCCGGGA-1") == 1);
    assert(s.match("ACGTACGTACGTACGA") == 0);   // Hamming-1 unique (last base)
    assert(s.match("TCTTAAAACCCCGGGA") == 1);   // Hamming-1 unique (second base)
    assert(s.match("AAAAAAAAAAAAAAAT") == -1);  // one mismatch from both AAAA...C and AAAA...G
    assert(s.match("AAAAAAAAAAAAAAAC") == 2);   // exact still resolves
    assert(s.match("GGGGGGGGGGGGGGGG") == -1);  // no match
    assert(s.match("ACGTACGTACGTACAA") == -1);  // two mismatches
    VisiumSlide missing;
    assert(!missing.load(temp_path("test_visium_nonexistent.tsv")));
    std::remove(path.c_str());
    std::printf("  slide_load_match: PASS\n");
}

void test_detect_tissue() {
    Grid g = make_grid(12345);
    assert(g.slide.size() == 4992);
    size_t planted = 0;
    for (uint8_t t : g.truth) planted += t;
    const double frac = double(planted) / double(g.truth.size());
    TissueCall r = detect_tissue(g.slide, g.umis);
    assert(r.in_tissue.size() == g.slide.size());
    size_t agree = 0;
    for (size_t i = 0; i < g.truth.size(); ++i) agree += (r.in_tissue[i] != 0) == (g.truth[i] != 0);
    const double acc = double(agree) / double(g.truth.size());
    const double nrel = std::abs(double(r.n_tissue) - double(planted)) / double(planted);
    std::printf("    planted=%zu (%.1f%%) n_tissue=%u agree=%.4f nrel=%.4f mean_low=%.3f mean_high=%.3f sd=%.3f/%.3f sep=%d\n",
                planted, 100 * frac, r.n_tissue, acc, nrel, r.mean_low, r.mean_high, r.sd_low, r.sd_high,
                int(r.separated));
    assert(acc >= 0.97);
    assert(nrel <= 0.03);
    assert(r.separated);
    assert(r.mean_high > r.mean_low);
    std::printf("  detect_tissue: PASS\n");
}

void test_smoothing() {
    Grid g = make_grid(777);
    const auto holes = pick_isolated(g, 0.0, 0.5, 30, 1);   // well inside the ellipse
    const auto specks = pick_isolated(g, 1.5, 3.0, 30, 2);  // well outside
    Rng rng(99);
    for (size_t i : holes) g.umis[i] = rng.lognormal10(100, 0.35);
    for (size_t i : specks) g.umis[i] = rng.lognormal10(5000, 0.35);
    TissueCall r = detect_tissue(g.slide, g.umis);
    int holes_filled = 0, specks_removed = 0;
    for (size_t i : holes) holes_filled += r.in_tissue[i] != 0;
    for (size_t i : specks) specks_removed += r.in_tissue[i] == 0;
    std::printf("    holes called in_tissue: %d/30, specks called background: %d/30 (default beta)\n", holes_filled,
                specks_removed);
    assert(holes_filled >= 25);
    assert(specks_removed >= 25);
    std::printf("  smoothing: PASS\n");
}

void test_degenerate() {
    Grid g = make_grid(5);
    std::vector<uint64_t> flat(g.slide.size(), 1000);
    TissueCall r = detect_tissue(g.slide, flat);
    assert(r.in_tissue.size() == g.slide.size());
    assert(r.prob.size() == g.slide.size());
    std::printf("    flat UMIs: n_tissue=%u separated=%d mean_low=%.3f mean_high=%.3f weight_high=%.3f prob[0]=%.3f\n",
                r.n_tissue, int(r.separated), r.mean_low, r.mean_high, r.weight_high, double(r.prob[0]));
    std::vector<uint64_t> zeros(g.slide.size(), 0);
    TissueCall z = detect_tissue(g.slide, zeros);
    assert(z.in_tissue.size() == g.slide.size());
    std::printf("    zero UMIs: n_tissue=%u separated=%d\n", z.n_tissue, int(z.separated));
    VisiumSlide empty;
    TissueCall e = detect_tissue(empty, {});
    assert(e.in_tissue.empty());
    std::printf("  degenerate: PASS\n");
}

}  // namespace

int main() {
    std::printf("test_visium_tissue\n");
    test_slide_load_match();
    test_detect_tissue();
    test_smoothing();
    test_degenerate();
    std::printf("all passed\n");
    return 0;
}
