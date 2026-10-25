// SPDX-License-Identifier: MIT
// tests/cpp/test_called_v2.cpp
//
// The C++ port of called_v2 (include/singlet/pileup/called_v2.h) against the Python reference
// (python/singlet/v2/cellcall.py). called_v2_fixture.h holds cellcall.call's output on a synthetic droplet sample that
// synth() below regenerates bit for bit, plus reference values of the pieces (scipy gammaln, numba's RNG, scipy's
// UnivariateSpline knots, numpy's quantile, the Good-Turing proportions). Regenerate it with
// `python -P bench/called_v2/make_cpp_fixture.py` whenever cellcall.py changes.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "called_v2_fixture.h"
#include "singlet/pileup/called_v2.h"
#include "singlet/pileup/run_provenance.h"
#include "test_harness.h"

using namespace singlet::calledv2;

namespace {

struct SplitMix {
    uint64_t s;
    explicit SplitMix(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};

/// Same generator as bench/called_v2/make_cpp_fixture.py::synth (keep in step).
GeneCounts synth() {
    const int G = cv2fix::kG;
    SplitMix r(cv2fix::kSeed);
    std::vector<uint64_t> amb(G);
    for (int g = 0; g < G; ++g) {
        uint64_t a = r.next() % 50;
        uint64_t b = r.next() % 50;
        amb[g] = 1 + a * b;
    }
    std::vector<std::vector<uint64_t>> types(3, std::vector<uint64_t>(G));
    for (int t = 0; t < 3; ++t)
        for (int g = 0; g < G; ++g) {
            uint64_t a = r.next() % 40;
            uint64_t b = r.next() % 40;
            uint64_t c = r.next() % 4;
            types[t][g] = 1 + a * b * c;
        }
    auto cum = [](const std::vector<uint64_t>& w) {
        std::vector<uint64_t> o(w.size());
        uint64_t c = 0;
        for (size_t i = 0; i < w.size(); ++i) o[i] = (c += w[i]);
        return o;
    };
    auto camb = cum(amb);
    std::vector<std::vector<uint64_t>> ctypes;
    for (auto& w : types) ctypes.push_back(cum(w));
    auto draw = [&](const std::vector<uint64_t>& cw) {
        uint64_t u = r.next() % cw.back();
        size_t lo = 0, hi = cw.size() - 1;
        while (lo < hi) {
            size_t mid = (lo + hi) / 2;
            if (cw[mid] > u)
                hi = mid;
            else
                lo = mid + 1;
        }
        return lo;
    };
    const uint64_t n_cells = cv2fix::kCells, n_low = cv2fix::kLow, n_fat = 150, n_amb = cv2fix::kAmb;
    const uint64_t total = n_cells + n_low + n_fat + n_amb;
    GeneCounts X;
    X.nrows = G;
    X.ncols = static_cast<uint32_t>(total);
    X.indptr.push_back(0);
    for (uint64_t i = 0; i < total; ++i) {
        uint64_t kind = r.next() % total;
        uint64_t T, mix;
        if (kind < n_cells) {
            T = 300 + r.next() % 3000;
            mix = 85;
        } else if (kind < n_cells + n_low) {
            T = 60 + r.next() % 200;
            mix = 60;
        } else if (kind < n_cells + n_low + n_fat) {
            T = 50 + r.next() % 100;
            mix = r.next() % 8;
        } else {
            T = 1 + r.next() % 25;
            mix = r.next() % 20;
        }
        const auto& ty = ctypes[r.next() % 3];
        std::vector<int64_t> x(G, 0);
        for (uint64_t k = 0; k < T; ++k) {
            size_t g = (r.next() % 100) < mix ? draw(ty) : draw(camb);
            x[g] += 1;
        }
        for (int g = 0; g < G; ++g)
            if (x[g]) {
                X.indices.push_back(g);
                X.data.push_back(x[g]);
            }
        X.indptr.push_back(static_cast<int64_t>(X.indices.size()));
    }
    return X;
}

uint64_t bits(double v) {
    uint64_t b;
    std::memcpy(&b, &v, 8);
    return b;
}

}  // namespace

int main() {
    std::cout << "=== called_v2 vs cellcall.py ===\n";

    // ── pieces ──
    {
        bool ok = true;
        for (size_t i = 0; i < sizeof(cv2fix::kGlX) / sizeof(double); ++i)
            ok = ok && bits(detail::gammaln(cv2fix::kGlX[i])) == bits(cv2fix::kGlY[i]);
        CHECK(ok, "gammaln == scipy.special.gammaln bit for bit");
    }
    {
        detail::NumbaRng rng(20261023 + 5);
        double got[12];
        for (int i = 0; i < 4; ++i) got[i] = rng.random();
        const int64_t ns[6] = {2, 7, 1000, 123456789, 5000000000LL, 3};
        for (int i = 0; i < 6; ++i) got[4 + i] = static_cast<double>(rng.randint(ns[i]));
        got[10] = rng.random();
        got[11] = static_cast<double>(rng.randint(38606));
        bool ok = true;
        for (int i = 0; i < 12; ++i) ok = ok && got[i] == cv2fix::kRng[i];
        CHECK(ok, "MT19937 random()/randint() == numba np.random after np.random.seed");
    }
    {
        std::vector<double> xs, ys;
        for (int i = 1; i < 400; ++i) {
            double x = std::log10(i * 1.0 + 0.5);
            xs.push_back(x);
            ys.push_back(3.5 - 0.2 * x - 1.2 / (1 + std::exp(-(x - 1.7) * 6)) + 0.01 * std::sin(x * 37));
        }
        double mean = detail::np_sum(ys) / ys.size();
        std::vector<double> dev(ys.size());
        for (size_t i = 0; i < ys.size(); ++i) dev[i] = (ys[i] - mean) * (ys[i] - mean);
        double s = xs.size() * (detail::np_sum(dev) / ys.size()) * 0.002;
        auto spl = detail::curfit(xs, ys, s);
        bool ok = static_cast<int>(spl.t.size()) == cv2fix::kSplineN;
        // inputs come from libm exp/sin/log10 here and numpy there, so allow a last-bit difference in the knots
        for (int i = 0; ok && i < cv2fix::kSplineN; ++i) ok = std::fabs(spl.t[i] - cv2fix::kSplineT[i]) < 1e-12;
        CHECK(ok, "FITPACK curfit knots == scipy UnivariateSpline knots");
    }

    // ── the call ──
    GeneCounts X = synth();
    int64_t sum = 0;
    for (auto v : X.data) sum += v;
    CHECK(sum == cv2fix::kSum, "synthetic sample regenerated (same total count as the Python generator)");
    std::vector<int64_t> exon_tot(X.ncols, 0);
    for (uint32_t j = 0; j < X.ncols; ++j)
        for (int64_t p = X.indptr[j]; p < X.indptr[j + 1]; ++p) exon_tot[j] += X.data[p];
    auto called = cellranger22(exon_tot);
    int64_t n_called = 0;
    std::vector<int64_t> tc;
    for (uint32_t j = 0; j < X.ncols; ++j)
        if (called[j]) {
            ++n_called;
            tc.push_back(exon_tot[j]);
        }
    CHECK(n_called == cv2fix::kNCalled, "CellRanger2.2 knee set size");
    CHECK(detail::quantile_linear(tc, 0.05) == cv2fix::kQuantile, "np.quantile(linear) of called totals");

    for (int threads : {1, 4}) {
        Result R = call(X, called, "", Params(), threads);
        std::string t = " (threads=" + std::to_string(threads) + ")";
        CHECK(R.lower == cv2fix::kLower, "plateau lower" + t);
        // the port returns the knee total exactly; Python returns 10 ** log10(total), the same up to one ulp
        CHECK(R.retain_total == std::round(cv2fix::kRetain) && std::fabs(R.retain_total - cv2fix::kRetain) < 1e-9,
              "knee retain total" + t);
        // np.polyfit (LAPACK gelsd) in the Good-Turing step is not reproduced bit for bit, so the ambient profile can
        // differ in the last bit; on this nearly multinomial (flat-likelihood) ambient pool that moves alpha slightly
        // and can move a Monte Carlo count by one. The call itself must not change.
        std::printf("    alpha C++ %.17g, Python %.17g (rel %.2g)\n", R.alpha, cv2fix::kAlpha,
                    R.alpha / cv2fix::kAlpha - 1);
        CHECK(std::fabs(R.alpha / cv2fix::kAlpha - 1) < 1e-3, "Dirichlet-multinomial alpha" + t);
        CHECK(R.n_tested == cv2fix::kNTested, "tested barcodes" + t);
        CHECK(R.n_called_v2 == cv2fix::kNCalledV2, "called_v2 count" + t);
        int set_diff = 0, fdr_diff = 0;
        double fdr_max = 0;
        for (int64_t i = 0; i < cv2fix::kNTested; ++i) {
            int j = cv2fix::kTested[i];
            set_diff += (R.called_v2[j] != cv2fix::kCalledV2Tested[i]);
            fdr_diff += (R.fdr[j] != cv2fix::kFdr[i]);
            fdr_max = std::max(fdr_max, static_cast<double>(std::fabs(R.fdr[j] - cv2fix::kFdr[i])));
        }
        std::printf("    FDR: %d of %lld differ, max |diff| %.3g\n", fdr_diff, (long long)cv2fix::kNTested, fdr_max);
        CHECK(set_diff == 0, "called_v2 identical on every tested barcode" + t);
        CHECK(fdr_diff <= cv2fix::kNTested / 50 && fdr_max <= 2e-3, "FDR equal up to Monte Carlo ties" + t);
    }

    // combinatorial protocols keep the stored call
    {
        Result R = call(X, called, "SPLiT-seq", Params(), 1);
        CHECK(R.method == "stored-call" && R.n_called_v2 == n_called, "SPLiT-seq keeps the stored call");
    }
    // BH and gene keys
    {
        auto q = bh({0.01, 0.04, 0.03, 0.2});
        CHECK(std::fabs(q[0] - 0.04) < 1e-15 && std::fabs(q[1] - 0.16 / 3) < 1e-15 &&
                  std::fabs(q[2] - 0.16 / 3) < 1e-15 && std::fabs(q[3] - 0.2) < 1e-15,
              "Benjamini-Hochberg");
        CHECK(gene_key("ENSG01_TP53_intron_chr17:1-2") == "ENSG01", "gene key = name before the first '_'");
    }
    // STARsolo CellRanger2.2: robust max = 31st largest (round(3000 * 0.01) = 30, 0-based), threshold round(max / 10)
    {
        std::vector<int64_t> tot;
        for (int i = 0; i < 100; ++i) tot.push_back(1000 - i);  // 1000 .. 901; 31st largest = 970 -> min 97
        tot.push_back(97);
        tot.push_back(96);
        tot.push_back(0);
        int64_t umin = 0;
        auto c = cellranger22(tot, 3000, 0.99, 10, &umin);
        CHECK(umin == 97 && c[100] == 1 && c[101] == 0 && c[102] == 0, "CellRanger2.2 threshold as STARsolo");
    }
    // provenance helpers
    {
        singlet::Sha256Stream s;
        const char* abc = "abc";
        s.update(reinterpret_cast<const uint8_t*>(abc), 3);
        CHECK(s.hex() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256(\"abc\")");
        auto runs = singlet::split_list(" SRR1, SRR2 ,,SRR3");
        CHECK(runs.size() == 3 && runs[1] == "SRR2", "runs_used list parsing");
    }

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
