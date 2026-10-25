// SPDX-License-Identifier: MIT
// tests/cpp/test_called_v2.cpp
//
// The C++ port of called_v2 (include/singlet/pileup/called_v2.h) against the Python reference (cellcall.call, VERSION
// 2.1: bench/called_v2_truth/cellcall_ref.py until python/singlet/v2/cellcall.py carries the same change).
// called_v2_fixture.h holds the reference output on two synthetic droplet samples that synth() below regenerates bit for
// bit (the second one exercises the 2.1 tail p-values), a barcode-rank curve on which the unbounded knee falls in the
// ambient shoulder (the 2.1 curve bounds), plus reference values of the pieces (scipy gammaln, numba's RNG, scipy's
// UnivariateSpline knots, numpy's quantile, the Good-Turing proportions). Regenerate it with
// `python -P bench/called_v2/make_cpp_fixture.py` whenever the reference changes.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
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
GeneCounts synth(uint64_t seed, uint64_t n_cells, uint64_t n_low, uint64_t n_amb, uint64_t n_fat) {
    const int G = cv2fix::kG;
    SplitMix r(seed);
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

/// Call-level p-values against the reference. On these synthetic samples the ambient pool is nearly multinomial, so the
/// likelihood of alpha is flat and the last-bit Good-Turing difference (np.polyfit) moves alpha within fminbound's xatol
/// (~1e-4 relative, see the alpha note in main). That shifts the null: Monte Carlo counts can move by a count or two, and
/// the tail-extrapolated p (0 < p < 1 / (niters + 1)) by a few percent (z moves by ~1e-2). With alpha bit-equal the bounds
/// are one count on <= 2% of barcodes and 1e-9 relative, otherwise five counts on <= 20% and 15% relative. The tail
/// formula itself is checked exactly against numpy below (kernel check, with the reference prop / alpha).
bool pv_agree(const Result& R, const int* tested, const double* ref, int64_t n, bool alpha_equal) {
    const double floor_p = 1.0 / static_cast<double>(Params().niters + 1);
    const double tail_tol = alpha_equal ? 1e-9 : 0.15;
    const int64_t mc_tol = alpha_equal ? n / 50 : n / 5;
    const double mc_abs = (alpha_equal ? 1.5 : 5.5) * floor_p;
    int mc_diff = 0, bad = 0;
    double tail_rel = 0, mc_max = 0;
    for (int64_t i = 0; i < n; ++i) {
        double got = R.pvalue[tested[i]], want = ref[i];
        if (want > 0 && want < floor_p) {
            double rel = std::fabs(got / want - 1);
            tail_rel = std::max(tail_rel, rel);
            bad += !(rel < tail_tol);
        } else if (got != want) {
            ++mc_diff;
            mc_max = std::max(mc_max, std::fabs(got - want));
            if (!(std::fabs(got - want) < mc_abs)) ++bad;
        }
    }
    std::printf("    p: %d Monte Carlo p-values differ (max |diff| %.3g), tail max rel diff %.3g (alpha %s)\n", mc_diff,
                mc_max, tail_rel, alpha_equal ? "bit-equal" : "within xatol");
    return bad == 0 && mc_diff <= mc_tol;
}

// ── shared golden vectors (tests/data/called_v2_golden/, format singlet-called-v2-golden/1; README there) ──

std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// The number after "key": inside the JSON object text `obj` (NaN for null or absent). Enough for case.json.
double json_num(const std::string& obj, const std::string& key) {
    auto p = obj.find("\"" + key + "\"");
    if (p == std::string::npos) return std::nan("");
    p = obj.find(':', p);
    while (p + 1 < obj.size() && std::isspace(static_cast<unsigned char>(obj[p + 1]))) ++p;
    if (obj.compare(p + 1, 4, "null") == 0) return std::nan("");
    return std::strtod(obj.c_str() + p + 1, nullptr);
}

std::string json_str(const std::string& obj, const std::string& key) {
    auto p = obj.find("\"" + key + "\"");
    if (p == std::string::npos) return "";
    p = obj.find('"', obj.find(':', p));
    return obj.substr(p + 1, obj.find('"', p + 1) - p - 1);
}

bool rel_ok(double got, double want, double tol) {
    if (std::isnan(want)) return std::isnan(got);
    if (want == 0) return got == 0;
    return std::fabs(got / want - 1) <= tol;
}

void golden_case(const std::string& dir, const std::string& name) {
    const std::string cj = slurp(dir + "/case.json");
    const auto ep = cj.find("\"expected\"");
    const std::string ex = cj.substr(ep, cj.find('}', ep) - ep);
    const std::string opts = cj.substr(cj.find("\"options\""), cj.find('}', cj.find("\"options\"")) - cj.find("\"options\""));
    if (opts.find(':', 10) != std::string::npos && opts.find('"', 10) != std::string::npos) {
        CHECK(false, "golden " + name + ": parameter overrides are not supported by this reader");
        return;
    }
    // counts.mtx: genes x barcodes, 1-based, sorted by column then row
    GeneCounts X;
    {
        std::ifstream f(dir + "/counts.mtx");
        std::string line;
        while (std::getline(f, line) && !line.empty() && line[0] == '%') {}
        long long m, n, nnz;
        std::istringstream(line) >> m >> n >> nnz;
        X.nrows = static_cast<uint32_t>(m);
        X.ncols = static_cast<uint32_t>(n);
        X.indptr.assign(n + 1, 0);
        long long i, j, v;
        while (f >> i >> j >> v) {
            X.indices.push_back(static_cast<int32_t>(i - 1));
            X.data.push_back(v);
            ++X.indptr[j];
        }
        for (long long c = 0; c < n; ++c) X.indptr[c + 1] += X.indptr[c];
    }
    std::vector<uint8_t> called;
    {
        std::ifstream f(dir + "/called.txt");
        int v;
        while (f >> v) called.push_back(static_cast<uint8_t>(v));
    }
    struct Row {
        int col;
        double p, q;
        int cv2;
    };
    std::vector<Row> rows;
    {
        std::ifstream f(dir + "/expected.tsv");
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ss(line);
            Row r;
            long long tot;
            ss >> r.col >> tot >> r.p >> r.q >> r.cv2;
            rows.push_back(r);
        }
    }
    CHECK(called.size() == X.ncols && X.indices.size() == static_cast<size_t>(X.indptr.back()),
          "golden " + name + ": inputs read");
    const double floor_p = 1.0 / static_cast<double>(Params().niters + 1);
    const double alpha_ref = json_num(ex, "alpha");
    // Two passes. "free": the call as the pipeline runs it; everything discrete must be exact, alpha agrees to
    // fminbound's xatol (1e-4 in log alpha: the Good-Turing np.polyfit is not bit-exact, and on a flat likelihood a
    // last-bit change moves the minimiser within its tolerance, which then shifts Monte Carlo counts and tail p).
    // "pinned": alpha fixed to the reference value; then the README tolerances must hold for p-values and FDRs.
    for (int pass = 0; pass < 2; ++pass)
        for (int threads : {1, 4}) {
            Params P;
            if (pass == 1) P.alpha_override = alpha_ref;
            Result R = call(X, called, json_str(cj, "protocol"), P, threads);
            const std::string t =
                "golden " + name + (pass ? " pinned alpha" : " free") + " (threads=" + std::to_string(threads) + "): ";
            CHECK(R.lower == static_cast<int64_t>(json_num(ex, "lower")) && R.n_ambient == json_num(ex, "n_ambient") &&
                      R.ambient_umi == json_num(ex, "ambient_umi") && R.n_tested == json_num(ex, "n_tested") &&
                      rel_ok(R.retain_total, json_num(ex, "retain_total"), 1e-9),
                  t + "lower, ambient pool, tested (exact), retain_total (1e-9)");
            if (pass == 0) {
                std::printf("    %salpha C++ %.17g, reference %.17g (rel %.2g)\n", t.c_str(), R.alpha, alpha_ref,
                            R.alpha / alpha_ref - 1);
                CHECK(rel_ok(R.alpha, alpha_ref, 1e-4), t + "alpha (fminbound xatol)");
            }
            CHECK(R.n_tail == json_num(ex, "n_tail") && R.n_called_v2 == json_num(ex, "n_called_v2") &&
                      R.n_added == json_num(ex, "n_added"),
                  t + "n_tail, n_called_v2, n_added (exact)");
            int64_t ntested = 0;
            for (size_t j = 0; j < X.ncols; ++j) ntested += !std::isnan(R.pvalue[j]);
            int set_diff = 0, mc_diff = 0, tail_bad = 0, fdr_bad = 0;
            double tail_rel = 0;
            for (const Row& r : rows) {
                set_diff += (R.called_v2[r.col] != r.cv2);
                const double p = R.pvalue[r.col];
                if (r.p > 0 && r.p < floor_p) {
                    tail_bad += !rel_ok(p, r.p, 1e-6);
                    tail_rel = std::max(tail_rel, std::fabs(p / r.p - 1));
                } else if (!(p == r.p)) {
                    ++mc_diff;
                    if (pass == 1)
                        std::printf("      col %d: p C++ %.17g, reference %.17g (%+.3f counts)\n", r.col, p, r.p,
                                    (p - r.p) / floor_p);
                }
                fdr_bad += !rel_ok(R.fdr[r.col], r.q, 1e-6) && !(std::fabs(R.fdr[r.col] - r.q) < 1e-7);
            }
            std::printf("    %s%d called_v2, %d Monte Carlo p, %d tail p (max rel %.2g), %d FDR differ\n", t.c_str(),
                        set_diff, mc_diff, tail_bad, tail_rel, fdr_bad);
            CHECK(ntested == static_cast<int64_t>(rows.size()) && set_diff == 0, t + "tested set and called_v2 (exact)");
            if (pass == 1) {
                CHECK(mc_diff == 0 && tail_bad == 0, t + "p-values (Monte Carlo exact, tail 1e-6)");
                CHECK(fdr_bad == 0, t + "FDR (1e-6)");
            }
        }
}

/// Every case under tests/data/called_v2_golden/ (skipped, with a note, when the directory is absent).
void golden_vectors() {
#ifdef CV2_GOLDEN_DIR
    const std::string root = CV2_GOLDEN_DIR;
#else
    const std::string root = "tests/data/called_v2_golden";
#endif
    namespace fs = std::filesystem;
    if (!fs::is_directory(root)) {
        std::printf("    golden vectors: %s absent, skipped\n", root.c_str());
        return;
    }
    std::vector<std::string> cases;
    for (const auto& e : fs::directory_iterator(root))
        if (e.is_directory() && fs::exists(e.path() / "case.json")) cases.push_back(e.path().filename().string());
    std::sort(cases.begin(), cases.end());
    CHECK(!cases.empty(), "golden vectors present");
    for (const auto& c : cases) golden_case(root + "/" + c, c);
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
    GeneCounts X = synth(cv2fix::kSeed, cv2fix::kCells, cv2fix::kLow, cv2fix::kAmb, 150);
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
        CHECK(bits(R.retain_total) == bits(cv2fix::kRetain), "knee retain total" + t);
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
            // tail-extrapolated FDRs move by a few percent with alpha (see pv_agree); count larger differences
            double a = R.fdr[j], b = cv2fix::kFdr[i];
            fdr_diff += !(a == b || std::fabs(a - b) <= 0.15 * std::max(std::fabs(a), std::fabs(b)));
            fdr_max = std::max(fdr_max, std::fabs(a - b));
        }
        std::printf("    FDR: %d of %lld differ, max |diff| %.3g\n", fdr_diff, (long long)cv2fix::kNTested, fdr_max);
        CHECK(set_diff == 0, "called_v2 identical on every tested barcode" + t);
        CHECK(fdr_diff <= cv2fix::kNTested / 50 && fdr_max <= 2e-3, "FDR equal up to Monte Carlo ties" + t);
        CHECK(pv_agree(R, cv2fix::kTested, cv2fix::kPv, cv2fix::kNTested, bits(R.alpha) == bits(cv2fix::kAlpha)),
              "p-values equal up to Monte Carlo ties" + t);
    }

    // -- 2.1 curve bounds: the unbounded knee lands in the ambient shoulder (10-100 UMIs), the bounded one among cells --
    {
        std::vector<int64_t> tot;
        for (size_t i = 0; i < sizeof(cv2fix::kKneeVal) / sizeof(int64_t); ++i)
            tot.insert(tot.end(), static_cast<size_t>(cv2fix::kKneeCnt[i]), cv2fix::kKneeVal[i]);
        double ku = knee(tot, cv2fix::kKneeLower), kb = knee_bounded(tot, cv2fix::kKneeLower);
        std::printf("    knee unbounded %.17g (Python %.17g), bounded %.17g (Python %.17g)\n", ku, cv2fix::kKneeUnbounded,
                    kb, cv2fix::kKneeBounded);
        CHECK(ku == cv2fix::kKneeUnbounded && ku <= 100, "unbounded knee (2.0) == cellcall.knee, in the ambient shoulder");
        CHECK(kb == cv2fix::kKneeBounded && kb > 1000, "bounded knee (2.1) == knee_bounded, among the cells");
        CHECK(std::isnan(knee_bounded(std::vector<int64_t>(tot.begin(), tot.begin() + 5), 0)),
              "bounded knee NaN with < 10 totals");
    }

    // -- 2.1 tail p-values: low-RNA cells below every simulated path, fewer than n_tested / 10 of them --
    {
        GeneCounts Xt = synth(cv2fix::kTSeed, cv2fix::kTCells, cv2fix::kTLow, cv2fix::kTAmb, cv2fix::kTFat);
        int64_t tsum = 0;
        for (auto v : Xt.data) tsum += v;
        CHECK(tsum == cv2fix::kTSum, "tail sample regenerated");
        std::vector<int64_t> et(Xt.ncols, 0);
        for (uint32_t j = 0; j < Xt.ncols; ++j)
            for (int64_t p = Xt.indptr[j]; p < Xt.indptr[j + 1]; ++p) et[j] += Xt.data[p];
        auto ct = cellranger22(et);
        int64_t nct = 0;
        for (auto c : ct) nct += c;
        CHECK(nct == cv2fix::kTNCalled, "tail sample: CellRanger2.2 set size");
        for (int threads : {1, 4}) {
            Result R = call(Xt, ct, "", Params(), threads);
            std::string t = " (threads=" + std::to_string(threads) + ")";
            CHECK(R.lower == cv2fix::kTLower && bits(R.retain_total) == bits(cv2fix::kTRetain),
                  "tail sample: lower and bounded retain" + t);
            CHECK(R.n_tested == cv2fix::kTNTested && R.n_called_v2 == cv2fix::kTNCalledV2, "tail sample: counts" + t);
            int set_diff = 0, n_tail = 0;
            const double floor_p = 1.0 / static_cast<double>(Params().niters + 1);
            for (int64_t i = 0; i < cv2fix::kTNTested; ++i) {
                int j = cv2fix::kTTested[i];
                set_diff += (R.called_v2[j] != cv2fix::kTCalledV2Tested[i]);
                n_tail += (R.pvalue[j] > 0 && R.pvalue[j] < floor_p);
            }
            std::printf("    tail sample: %d tail-extrapolated p-values, called_v2 %lld\n", n_tail,
                        (long long)R.n_called_v2);
            CHECK(set_diff == 0, "tail sample: called_v2 identical on every tested barcode" + t);
            CHECK(n_tail > 50 && n_tail < cv2fix::kTNTested / 10, "tail sample: the floor binds and is extrapolated" + t);
            CHECK(pv_agree(R, cv2fix::kTTested, cv2fix::kTPv, cv2fix::kTNTested, bits(R.alpha) == bits(cv2fix::kTAlpha)),
                  "tail sample: p-values" + t);
        }

        // kernel check with the reference's prop / alpha: DM log-probabilities, Monte Carlo counts (RNG-exact) and the
        // null mean / sd numpy takes over the 10,000 sorted paths, then the tail p-value formula
        std::vector<double> prop(cv2fix::kTProp, cv2fix::kTProp + cv2fix::kG);
        std::vector<uint32_t> cols(cv2fix::kKCols, cv2fix::kKCols + cv2fix::kKN);
        ColumnView S = take_columns(Xt, cols);
        std::vector<double> lp = dm_logprob(S, prop, cv2fix::kTAlpha);
        double lp_rel = 0;
        for (int i = 0; i < cv2fix::kKN; ++i) lp_rel = std::max(lp_rel, std::fabs(lp[i] / cv2fix::kKLp[i] - 1));
        CHECK(lp_rel < 1e-13, "kernel: DM log-probability == cellcall.dm_logprob");
        std::vector<double> lpy(cv2fix::kKLp, cv2fix::kKLp + cv2fix::kKN);
        for (int threads : {1, 3}) {
            std::vector<double> mu, sd;
            Params P;
            auto nle = mc_counts(prop, cv2fix::kTAlpha, S.totals, lpy, P.niters, P.seed, threads, &mu, &sd);
            int nle_diff = 0, n_floor = 0;
            double mu_rel = 0, sd_rel = 0, p_rel = 0;
            const double floor_p = 1.0 / static_cast<double>(P.niters + 1);
            for (int i = 0; i < cv2fix::kKN; ++i) {
                nle_diff += nle[i] != cv2fix::kKNle[i];
                mu_rel = std::max(mu_rel, std::fabs(mu[i] / cv2fix::kKMu[i] - 1));
                sd_rel = std::max(sd_rel, std::fabs(sd[i] / cv2fix::kKSd[i] - 1));
                if (cv2fix::kKNle[i] == 0) {
                    ++n_floor;
                    double z = (lpy[i] - mu[i]) / std::max(sd[i], 1e-12);
                    double zp = (lpy[i] - cv2fix::kKMu[i]) / std::max(cv2fix::kKSd[i], 1e-12);
                    double pc = std::min(ndtr(z), floor_p), pp = std::min(ndtr(zp), floor_p);
                    p_rel = std::max(p_rel, std::fabs(pc / pp - 1));
                }
            }
            std::printf("    kernel (threads=%d): %d of %d counts differ, %d at the floor, null mean rel %.2g, sd rel %.2g, "
                        "tail p rel %.2g\n", threads, nle_diff, cv2fix::kKN, n_floor, mu_rel, sd_rel, p_rel);
            std::string t = " (threads=" + std::to_string(threads) + ")";
            CHECK(nle_diff == 0, "kernel: Monte Carlo counts == numba kernel" + t);
            CHECK(n_floor >= 5 && mu_rel < 1e-12 && sd_rel < 1e-9 && p_rel < 1e-8, "kernel: null mean / sd / tail p" + t);
        }
    }

    // ── the golden vectors shared with python/singlet/v2/cellcall.py ──
    golden_vectors();

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
