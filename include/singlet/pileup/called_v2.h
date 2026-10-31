// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: called_v2.h
// `called_v2`: the EmptyDrops cell call of docs/called-v2.md, ported from the reference implementation
// python/singlet/v2/cellcall.py (call(), METHOD "singlet-emptydrops") so that the pipeline writes the same call the v2
// migration computes from stored counts. VERSION "2.1" (docs/called-v2-truth.md) fits the retain knee between
// DropletUtils' curve bounds (knee_bounded = cellcall.knee) and extrapolates the null tail for barcodes below every
// simulated path. Shared golden vectors: tests/data/called_v2_golden/ (checked by tests/cpp/test_called_v2.cpp).
//
//   * counts: exon + intron per gene (gene = feature name before the first '_'), genes sorted by that key;
//   * lower: the "plateau" rule; ambient pool: uncalled barcodes with 0 < total <= lower;
//   * tested: uncalled barcodes with total > max(lower, 50); barcodes at or above the barcodeRanks knee (fitted between
//     DropletUtils' curve bounds) get p = 0;
//   * ambient profile: Simple Good-Turing proportions; Dirichlet-multinomial alpha by maximum likelihood (Brent's
//     bounded minimiser, as scipy's minimize_scalar(method="bounded"));
//   * Monte Carlo: 10,000 Polya-urn paths, path i seeded with seed + i on an MT19937 drawing exactly as numba's
//     np.random does (random(), randint()), so the simulated log-probabilities are the Python ones;
//   * below every simulated path: p = min(Phi((lp - mean) / sd), 1 / (niters + 1)) over the paths at that total;
//   * Benjamini-Hochberg over the tested barcodes; called_v2 = called OR FDR <= 0.001.
//
// Floating-point order follows numpy where it can matter (pairwise sums in 8192-element blocks, quantile lerp, the
// FITPACK smoothing spline of the knee, cephes lgamma). Contraction into FMA is disabled for this file, because the
// pipeline is built with -march=native and numpy/numba do not fuse.
//
// Also here: the STARsolo "CellRanger2.2 3000 0.99 10" knee filter (the starting `called` set), computed on the
// pileup's own gene counts after strand resolution.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

namespace singlet {
namespace calledv2 {

inline constexpr const char* kMethod = "singlet-emptydrops";
inline constexpr const char* kVersion = "2.1";
inline constexpr int64_t kMaxSimulatedTotal = 50000;

struct Params {
    std::string lower_rule = "plateau";  ///< "plateau" or an integer
    int64_t test_min = 50;
    double fdr = 0.001;
    std::string retain = "knee";         ///< "knee" or "none"
    int64_t niters = 10000;
    int64_t seed = 20261023;
    /// tests only: use this alpha instead of the MLE (NaN = fit it). The golden-vector test pins the reference alpha to check
    /// the rest of the call exactly, because alpha itself agrees only to fminbound's xatol (np.polyfit is not bit-exact).
    double alpha_override = std::numeric_limits<double>::quiet_NaN();
};

/// genes x barcodes counts, CSC, row indices ascending within each column.
struct GeneCounts {
    uint32_t nrows = 0, ncols = 0;
    std::vector<int64_t> indptr;     ///< ncols + 1
    std::vector<int32_t> indices;
    std::vector<int64_t> data;
    std::vector<std::string> genes;  ///< row keys (sorted)
};

struct Result {
    std::vector<uint8_t> called_v2;  ///< per barcode
    std::vector<double> fdr;         ///< BH FDR (NaN = not tested; double since 2.1: tail p reach 1e-150)
    std::vector<double> pvalue;      ///< Monte Carlo p (NaN = not tested)
    std::vector<double> log_prob;    ///< DM log-probability of simulated barcodes (NaN otherwise)
    std::vector<int64_t> totals;     ///< exon + intron total per barcode
    std::string method = kMethod;    ///< "singlet-emptydrops" or "stored-call"
    std::string protocol, protocol_family;
    bool validated = true;
    Params params;
    int64_t lower = -1;
    double retain_total = std::numeric_limits<double>::quiet_NaN();  ///< NaN = None
    int64_t n_ambient = 0, ambient_umi = 0, n_tested = 0;
    int64_t n_tail = 0;  ///< tested barcodes whose p came from the tail extrapolation (2.1)
    double alpha = std::numeric_limits<double>::quiet_NaN();
    int64_t n_called = 0, n_called_v2 = 0, n_added = 0;
    std::string error;
    double seconds = 0.0;
};

// ───────────────────────────────────────────────────────────────────────── numerics matching numpy / scipy
namespace detail {

/// numpy's pairwise summation of a contiguous float64 array (one inner-loop call).
inline double pairwise_block(const double* a, size_t n) {
    if (n < 8) {
        double r = 0.0;
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return pairwise_block(a, n2) + pairwise_block(a + n2, n - n2);
}

/// np.sum / np.add.reduce of a float64 vector: 0 + pairwise sums of 8192-element buffers.
inline double np_sum(const std::vector<double>& v) {
    double acc = 0.0;
    for (size_t i = 0; i < v.size(); i += 8192) acc += pairwise_block(v.data() + i, std::min<size_t>(8192, v.size() - i));
    return acc;
}

inline double polevl(double x, const double* c, int n) {
    double ans = c[0];
    for (int i = 1; i <= n; ++i) ans = ans * x + c[i];
    return ans;
}
inline double p1evl(double x, const double* c, int n) {
    double ans = x + c[0];
    for (int i = 1; i < n; ++i) ans = ans * x + c[i];
    return ans;
}

/// cephes lgam (scipy.special.gammaln) for x > 0.
inline double gammaln(double x) {
    static const double A[] = {8.11614167470508450300E-4, -5.95061904284301438324E-4, 7.93650340457716943945E-4,
                               -2.77777777730099687205E-3, 8.33333333333331927722E-2};
    static const double B[] = {-1.37825152569120859100E3, -3.88016315134637840924E4, -3.31612992738871184744E5,
                               -1.16237097492762307383E6, -1.72173700820839662146E6, -8.53555664245765465627E5};
    static const double C[] = {-3.51815701436523470549E2, -1.70642106651881159223E4, -2.20528590553854454839E5,
                               -1.13933444367982507207E6, -2.53252307177582951285E6, -2.01889141433532773231E6};
    const double LS2PI = 0.91893853320467274178;
    const double MAXLGM = 2.556348e305;
    if (!std::isfinite(x)) return x;
    if (x < -34.0) return std::lgamma(x);  // not reached: arguments here are positive
    if (x < 13.0) {
        double z = 1.0, p = 0.0, u = x;
        while (u >= 3.0) {
            p -= 1.0;
            u = x + p;
            z *= u;
        }
        while (u < 2.0) {
            if (u == 0.0) return std::numeric_limits<double>::infinity();
            z /= u;
            p += 1.0;
            u = x + p;
        }
        if (z < 0.0) z = -z;
        if (u == 2.0) return std::log(z);
        p -= 2.0;
        x = x + p;
        p = x * polevl(x, B, 5) / p1evl(x, C, 6);
        return std::log(z) + p;
    }
    if (x > MAXLGM) return std::numeric_limits<double>::infinity();
    double q = (x - 0.5) * std::log(x) - x + LS2PI;
    if (x > 1.0e8) return q;
    double p = 1.0 / (x * x);
    if (x >= 1000.0)
        q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p + 0.0833333333333333333333) / x;
    else
        q += polevl(p, A, 4) / x;
    return q;
}

/// numba's np.random on one MT19937 stream (seeded like np.random.seed(int)).
struct NumbaRng {
    std::mt19937 mt;
    explicit NumbaRng(uint32_t seed) : mt(seed) {}
    uint32_t next32() { return static_cast<uint32_t>(mt()); }
    double random() {
        uint32_t a = next32() >> 5, b = next32() >> 6;
        return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) / 9007199254740992.0;
    }
    /// np.random.randint(0, n) for int64 n >= 1
    int64_t randint(int64_t n) {
        if (n == 1) return 0;
        uint64_t nm1 = static_cast<uint64_t>(n - 1);
        int nbits = 64 - __builtin_clzll(nm1);
        for (;;) {
            uint64_t r;
            if (nbits <= 32) {
                uint32_t mask = 0xFFFFFFFFu >> (32 - nbits);
                r = next32() & mask;
            } else {
                uint32_t mask = 0xFFFFFFFFu >> (32 - (nbits - 32));
                uint64_t high = next32() & mask;
                uint64_t low = next32();
                r = low + (high << 32);
            }
            if (static_cast<int64_t>(r) < n) return static_cast<int64_t>(r);
        }
    }
};

/// scipy minimize_scalar(method="bounded") (fminbound) with xatol, maxiter 500.
template <typename F>
double fminbound(F&& func, double x1, double x2, double xatol) {
    const int maxfun = 500;
    const double sqrt_eps = std::sqrt(2.2e-16);
    const double golden_mean = 0.5 * (3.0 - std::sqrt(5.0));
    double a = x1, b = x2;
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0.0, e = 0.0;
    double x = xf;
    double fx = func(x);
    int num = 1;
    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
    double tol2 = 2.0 * tol1;
    auto sign = [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0); };
    while (std::fabs(xf - xm) > (tol2 - 0.5 * (b - a))) {
        bool golden = true;
        if (std::fabs(e) > tol1) {
            golden = false;
            double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = std::fabs(q);
            r = e;
            e = rat;
            if ((std::fabs(p) < std::fabs(0.5 * q * r)) && (p > q * (a - xf)) && (p < q * (b - xf))) {
                rat = (p + 0.0) / q;
                x = xf + rat;
                if (((x - a) < tol2) || ((b - x) < tol2)) {
                    double si = sign(xm - xf) + ((xm - xf) == 0 ? 1.0 : 0.0);
                    rat = tol1 * si;
                }
            } else {
                golden = true;
            }
        }
        if (golden) {
            if (xf >= xm)
                e = a - xf;
            else
                e = b - xf;
            rat = golden_mean * e;
        }
        double si = sign(rat) + (rat == 0 ? 1.0 : 0.0);
        x = xf + si * std::max(std::fabs(rat), tol1);
        double fu = func(x);
        num += 1;
        if (fu <= fx) {
            if (x >= xf)
                a = xf;
            else
                b = xf;
            fulc = nfc;
            ffulc = fnfc;
            nfc = xf;
            fnfc = fx;
            xf = x;
            fx = fu;
        } else {
            if (x < xf)
                a = x;
            else
                b = x;
            if ((fu <= fnfc) || (nfc == xf)) {
                fulc = nfc;
                ffulc = fnfc;
                nfc = x;
                fnfc = fu;
            } else if ((fu <= ffulc) || (fulc == xf) || (fulc == nfc)) {
                fulc = x;
                ffulc = fu;
            }
        }
        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * std::fabs(xf) + xatol / 3.0;
        tol2 = 2.0 * tol1;
        if (num >= maxfun) break;
    }
    return xf;
}

// ─────────────────────────────────────────────── FITPACK curfit (scipy UnivariateSpline, iopt = 0, k = 3, w = 1)
struct Spline {
    std::vector<double> t, c;
    int k = 3;
    int ier = 0;
};

inline void fpbspl(const std::vector<double>& t, int k, double x, int l, double* h) {
    // l is 1-based as in FITPACK: t(l) <= x < t(l+1)
    double hh[20];
    h[0] = 1.0;
    for (int j = 1; j <= k; ++j) {
        for (int i = 1; i <= j; ++i) hh[i - 1] = h[i - 1];
        h[0] = 0.0;
        for (int i = 1; i <= j; ++i) {
            int li = l + i, lj = li - j;
            if (t[li - 1] == t[lj - 1]) {
                h[i] = 0.0;
                continue;
            }
            double f = hh[i - 1] / (t[li - 1] - t[lj - 1]);
            h[i - 1] = h[i - 1] + f * (t[li - 1] - x);
            h[i] = f * (x - t[lj - 1]);
        }
    }
}

inline void fpgivs(double piv, double& ww, double& cs, double& sn) {
    double store = std::fabs(piv);
    double dd;
    if (store >= ww)
        dd = store * std::sqrt(1.0 + (ww / piv) * (ww / piv));
    else
        dd = ww * std::sqrt(1.0 + (piv / ww) * (piv / ww));
    cs = ww / dd;
    sn = piv / dd;
    ww = dd;
}

inline void fprota(double cs, double sn, double& a, double& b) {
    double stor1 = a, stor2 = b;
    b = cs * stor2 + sn * stor1;
    a = cs * stor1 - sn * stor2;
}

/// back substitution, a is n x k band (row-major a[i*k + j]), 0-based
inline void fpback(const std::vector<double>& a, const std::vector<double>& z, int n, int k, std::vector<double>& c, int lda) {
    int k1 = k - 1;
    c.assign(n, 0.0);
    c[n - 1] = z[n - 1] / a[(n - 1) * lda + 0];
    int i = n - 1;  // 1-based index of the next row
    if (i == 0) return;
    for (int j = 2; j <= n; ++j) {
        double store = z[i - 1];
        int i1 = k1;
        if (j <= k1) i1 = j - 1;
        int m = i;
        for (int l = 1; l <= i1; ++l) {
            m = m + 1;
            store = store - c[m - 1] * a[(i - 1) * lda + l];
        }
        c[i - 1] = store / a[(i - 1) * lda + 0];
        i = i - 1;
    }
}

inline void fpdisc(const std::vector<double>& t, int n, int k2, std::vector<double>& b, int ldb) {
    int k1 = k2 - 1, k = k1 - 1, nk1 = n - k1, nrint = nk1 - k;
    double an = nrint;
    double fac = an / (t[nk1] - t[k1 - 1]);
    double h[12];
    for (int l = k2; l <= nk1; ++l) {
        int lmk = l - k1;
        for (int j = 1; j <= k1; ++j) {
            int ik = j + k1, lj = l + j, lk = lj - k2;
            h[j - 1] = t[l - 1] - t[lk - 1];
            h[ik - 1] = t[l - 1] - t[lj - 1];
        }
        int lp = lmk;
        for (int j = 1; j <= k2; ++j) {
            int jk = j;
            double prod = h[j - 1];
            for (int i = 1; i <= k; ++i) {
                jk = jk + 1;
                prod = prod * h[jk - 1] * fac;
            }
            int lk = lp + k1;
            b[(lmk - 1) * ldb + (j - 1)] = (t[lk - 1] - t[lp - 1]) / prod;
            lp = lp + 1;
        }
    }
}

inline double fprati(double& p1, double& f1, double p2, double f2, double& p3, double& f3) {
    double p;
    if (p3 > 0.0) {
        double h1 = f1 * (f2 - f3), h2 = f2 * (f3 - f1), h3 = f3 * (f1 - f2);
        p = -(p1 * p2 * h3 + p2 * p3 * h1 + p3 * p1 * h2) / (p1 * h1 + p2 * h2 + p3 * h3);
    } else {
        p = (p1 * (f1 - f3) * f2 - p2 * (f2 - f3) * f1) / ((f1 - f2) * f3);
    }
    if (f2 < 0.0) {
        p3 = p2;
        f3 = f2;
    } else {
        p1 = p2;
        f1 = f2;
    }
    return p;
}

inline void fpknot(const std::vector<double>& x, std::vector<double>& t, int& n, std::vector<double>& fpint,
                   std::vector<int>& nrdata, int& nrint, int istart) {
    int k = (n - nrint - 1) / 2;
    double fpmax = 0.0;
    int number = 0, maxpt = 0, maxbeg = 0;
    int jbegin = istart;
    for (int j = 1; j <= nrint; ++j) {
        int jpoint = nrdata[j - 1];
        if (!(fpmax >= fpint[j - 1] || jpoint == 0)) {
            fpmax = fpint[j - 1];
            number = j;
            maxpt = jpoint;
            maxbeg = jbegin;
        }
        jbegin = jbegin + jpoint + 1;
    }
    int ihalf = maxpt / 2 + 1;
    int nrx = maxbeg + ihalf;
    int next = number + 1;
    if (next <= nrint) {
        for (int j = next; j <= nrint; ++j) {
            int jj = next + nrint - j;
            fpint[jj] = fpint[jj - 1];
            nrdata[jj] = nrdata[jj - 1];
            int jk = jj + k;
            t[jk] = t[jk - 1];
        }
    }
    nrdata[number - 1] = ihalf - 1;
    nrdata[next - 1] = maxpt - ihalf;
    double am = maxpt;
    double an = nrdata[number - 1];
    fpint[number - 1] = fpmax * an / am;
    an = nrdata[next - 1];
    fpint[next - 1] = fpmax * an / am;
    int jk = next + k;
    t[jk - 1] = x[nrx - 1];
    n = n + 1;
    nrint = nrint + 1;
}

/// fpcurf with iopt = 0, w = 1, k = 3, tol = 0.001, maxit = 20, nest = max(m/2, 2k+2) (scipy fpcurf0 for s > 0).
inline Spline curfit(const std::vector<double>& x, const std::vector<double>& y, double s) {
    const int k = 3, k1 = k + 1, k2 = k1 + 1;
    const int m = static_cast<int>(x.size());
    const double tol = 0.001;
    const int maxit = 20;
    const double con1 = 0.1, con9 = 0.9, con4 = 0.04, half = 0.5;
    int nest = std::max(m / 2, 2 * k1);
    const double xb = x[0], xe = x[m - 1];
    Spline out;
    out.k = k;
    const int nmin = 2 * k1;
    const double acc = tol * s;
    int nmax = m + k1;
    std::vector<double> t(std::max(nest, nmax) + 2, 0.0), c(std::max(nest, nmax) + 2, 0.0);
    std::vector<double> fpint(std::max(nest, nmax) + 2, 0.0);
    std::vector<int> nrdata(std::max(nest, nmax) + 2, 0);
    std::vector<double> a(static_cast<size_t>(m + k1 + 2) * k1, 0.0), z(m + k1 + 2, 0.0);
    std::vector<double> q(static_cast<size_t>(m) * k1, 0.0);
    double h[8];
    int ier = 0;
    double fp = 0.0, fp0 = 0.0, fpold = 0.0, fpms = 0.0;
    int nplus = 0;
    int n = nmin;
    nrdata[0] = m - 2;
    int nk1 = 0;
    bool to_part2 = false;
    // s > 0 is required here (s = 0 is interpolation; the knee always passes s > 0)
    for (int iter = 1; iter <= m; ++iter) {
        if (n == nmin) ier = -2;
        int nrint = n - nmin + 1;
        nk1 = n - k1;
        {
            int i = n;
            for (int j = 1; j <= k1; ++j) {
                t[j - 1] = xb;
                t[i - 1] = xe;
                i = i - 1;
            }
        }
        fp = 0.0;
        for (int i = 0; i < nk1; ++i) {
            z[i] = 0.0;
            for (int j = 0; j < k1; ++j) a[i * k1 + j] = 0.0;
        }
        int l = k1;
        for (int it = 1; it <= m; ++it) {
            double xi = x[it - 1];
            double wi = 1.0;
            double yi = y[it - 1] * wi;
            while (!(xi < t[l] || l == nk1)) l = l + 1;
            fpbspl(t, k, xi, l, h);
            for (int i = 1; i <= k1; ++i) {
                q[(it - 1) * k1 + (i - 1)] = h[i - 1];
                h[i - 1] = h[i - 1] * wi;
            }
            int j = l - k1;
            for (int i = 1; i <= k1; ++i) {
                j = j + 1;
                double piv = h[i - 1];
                if (piv == 0.0) continue;
                double cs, sn;
                fpgivs(piv, a[(j - 1) * k1 + 0], cs, sn);
                fprota(cs, sn, yi, z[j - 1]);
                if (i == k1) break;
                int i2 = 1;
                for (int i1 = i + 1; i1 <= k1; ++i1) {
                    i2 = i2 + 1;
                    fprota(cs, sn, h[i1 - 1], a[(j - 1) * k1 + (i2 - 1)]);
                }
            }
            fp = fp + yi * yi;
        }
        if (ier == -2) fp0 = fp;
        std::vector<double> cc;
        fpback(a, z, nk1, k1, cc, k1);
        for (int i = 0; i < nk1; ++i) c[i] = cc[i];
        fpms = fp - s;
        if (std::fabs(fpms) < acc) {
            out.ier = ier;
            out.t.assign(t.begin(), t.begin() + n);
            out.c.assign(c.begin(), c.begin() + n);
            return out;
        }
        if (fpms < 0.0) {
            to_part2 = true;
            break;
        }
        if (n == nmax) {  // interpolating spline (not reached for the knee's s)
            out.ier = -1;
            out.t.assign(t.begin(), t.begin() + n);
            out.c.assign(c.begin(), c.begin() + n);
            return out;
        }
        if (n == nest) {
            if (nest < m + k1) {
                // UnivariateSpline._reset_nest: ier = 1 -> fpcurf1 (iopt = 1) resumes from these knots with
                // nest = m + k + 1, entering with ier = 1 (so the next step adds one knot).
                nest = m + k1;
                ier = 1;
            } else {
                out.ier = 1;
                out.t.assign(t.begin(), t.begin() + n);
                out.c.assign(c.begin(), c.begin() + n);
                return out;
            }
        }
        if (ier != 0) {
            nplus = 1;
            ier = 0;
        } else {
            int npl1 = nplus * 2;
            double rn = nplus;
            if (fpold - fp > acc) npl1 = static_cast<int>(rn * fpms / (fpold - fp));
            nplus = std::min(nplus * 2, std::max(std::max(npl1, nplus / 2), 1));
        }
        fpold = fp;
        double fpart = 0.0;
        int i = 1;
        l = k2;
        int nnew = 0;
        for (int it = 1; it <= m; ++it) {
            if (!(x[it - 1] < t[l - 1] || l > nk1)) {
                nnew = 1;
                l = l + 1;
            }
            double term = 0.0;
            int l0 = l - k2;
            for (int j = 1; j <= k1; ++j) {
                l0 = l0 + 1;
                term = term + c[l0 - 1] * q[(it - 1) * k1 + (j - 1)];
            }
            term = (1.0 * (term - y[it - 1])) * (1.0 * (term - y[it - 1]));
            fpart = fpart + term;
            if (nnew == 0) continue;
            double store = term * half;
            fpint[i - 1] = fpart - store;
            i = i + 1;
            fpart = store;
            nnew = 0;
        }
        fpint[nrint - 1] = fpart;
        bool restart_interp = false, stop_nest = false;
        for (int ll = 1; ll <= nplus; ++ll) {
            fpknot(x, t, n, fpint, nrdata, nrint, 1);
            if (n == nmax) {
                restart_interp = true;
                break;
            }
            if (n == nest) {
                stop_nest = true;
                break;
            }
        }
        (void)stop_nest;
        if (restart_interp) {
            // fpcurf "if(n.eq.nmax) go to 10": place the interior knots as for interpolation (k odd: t(k2..) =
            // x(k/2+2..)) and run the main loop again with them; the least-squares fit then interpolates (fp ~ 0 < s),
            // so part 2 smooths on these knots. Reached by short fits (the curve-bounded knee on a few points).
            n = nmax;
            const int mk1 = m - k1, k3 = k / 2;
            int ti = k2, xj = k3 + 2;
            for (int ll = 1; ll <= mk1; ++ll) t[(ti++) - 1] = x[(xj++) - 1];
            continue;
        }
    }
    if (!to_part2) {  // ran out of trials
        out.ier = ier;
        out.t.assign(t.begin(), t.begin() + n);
        out.c.assign(c.begin(), c.begin() + n);
        return out;
    }
    // part 2
    if (ier == -2) {
        out.ier = -2;
        out.t.assign(t.begin(), t.begin() + n);
        out.c.assign(c.begin(), c.begin() + n);
        return out;
    }
    std::vector<double> b(static_cast<size_t>(nest + 2) * k2, 0.0);
    fpdisc(t, n, k2, b, k2);
    double p1 = 0.0, f1 = fp0 - s, p3 = -1.0, f3 = fpms;
    double p = 0.0;
    for (int i = 0; i < nk1; ++i) p = p + a[i * k1 + 0];
    double rn = nk1;
    p = rn / p;
    int ich1 = 0, ich3 = 0;
    int n8 = n - nmin;
    std::vector<double> g(static_cast<size_t>(nest + 2) * k2, 0.0), cvec(nest + 2, 0.0);
    for (int iter = 1; iter <= maxit; ++iter) {
        double pinv = 1.0 / p;
        for (int i = 0; i < nk1; ++i) {
            cvec[i] = z[i];
            g[i * k2 + (k2 - 1)] = 0.0;
            for (int j = 0; j < k1; ++j) g[i * k2 + j] = a[i * k1 + j];
        }
        for (int it = 1; it <= n8; ++it) {
            for (int i = 1; i <= k2; ++i) h[i - 1] = b[(it - 1) * k2 + (i - 1)] * pinv;
            double yi = 0.0;
            for (int j = it; j <= nk1; ++j) {
                double piv = h[0];
                double cs, sn;
                fpgivs(piv, g[(j - 1) * k2 + 0], cs, sn);
                fprota(cs, sn, yi, cvec[j - 1]);
                if (j == nk1) break;
                int i2 = k1;
                if (j > n8) i2 = nk1 - j;
                for (int i = 1; i <= i2; ++i) {
                    int i1 = i + 1;
                    fprota(cs, sn, h[i1 - 1], g[(j - 1) * k2 + (i1 - 1)]);
                    h[i - 1] = h[i1 - 1];
                }
                h[i2] = 0.0;
            }
        }
        std::vector<double> cc;
        fpback(g, cvec, nk1, k2, cc, k2);
        for (int i = 0; i < nk1; ++i) c[i] = cc[i];
        fp = 0.0;
        int l = k2;
        for (int it = 1; it <= m; ++it) {
            if (!(x[it - 1] < t[l - 1] || l > nk1)) l = l + 1;
            int l0 = l - k2;
            double term = 0.0;
            for (int j = 1; j <= k1; ++j) {
                l0 = l0 + 1;
                term = term + c[l0 - 1] * q[(it - 1) * k1 + (j - 1)];
            }
            double r = 1.0 * (term - y[it - 1]);
            fp = fp + r * r;
        }
        fpms = fp - s;
        if (std::fabs(fpms) < acc) {
            ier = 0;
            break;
        }
        if (iter == maxit) {
            ier = 3;
            break;
        }
        double p2 = p, f2 = fpms;
        bool cont = false;
        if (ich3 == 0) {
            if (!((f2 - f3) > acc)) {
                p3 = p2;
                f3 = f2;
                p = p * con4;
                if (p <= p1) p = p1 * con9 + p2 * con1;
                cont = true;
            } else if (f2 < 0.0) {
                ich3 = 1;
            }
        }
        if (cont) continue;
        if (ich1 == 0) {
            if (!((f1 - f2) > acc)) {
                p1 = p2;
                f1 = f2;
                p = p / con4;
                if (!(p3 < 0.0) && p >= p3) p = p2 * con1 + p3 * con9;
                continue;
            } else if (f2 > 0.0) {
                ich1 = 1;
            }
        }
        if (f2 >= f1 || f2 <= f3) {
            ier = 2;
            break;
        }
        p = fprati(p1, f1, p2, f2, p3, f3);
    }
    out.ier = ier;
    out.t.assign(t.begin(), t.begin() + n);
    out.c.assign(c.begin(), c.begin() + n);
    return out;
}

/// scipy splder: derivative spline (FITPACK convention: c padded to len(t)).
inline Spline splder(const Spline& s) {
    Spline d;
    const int k = s.k;
    const int n = static_cast<int>(s.t.size());
    std::vector<double> c(n, 0.0);
    for (int i = 0; i < n - 2 - k; ++i) {
        double dt = s.t[k + 1 + i] - s.t[1 + i];
        c[i] = (s.c[1 + i] - s.c[i]) * k / dt;
    }
    d.t.assign(s.t.begin() + 1, s.t.end() - 1);
    d.c.assign(c.begin(), c.begin() + (n - 2));
    d.k = k - 1;
    return d;
}

/// FITPACK splev with ext = 0, x ascending and inside [t(k+1), t(n-k)].
inline std::vector<double> splev(const Spline& s, const std::vector<double>& xs) {
    const int k = s.k, k1 = k + 1;
    const int n = static_cast<int>(s.t.size());
    const int nk1 = n - k1;
    std::vector<double> out(xs.size());
    double h[8];
    int l = k1;
    for (size_t i = 0; i < xs.size(); ++i) {
        double arg = xs[i];
        while (l > k1 && arg < s.t[l - 1]) l = l - 1;
        while (!(arg < s.t[l] || l == nk1)) l = l + 1;
        fpbspl(s.t, k, arg, l, h);
        double sp = 0.0;
        int ll = l - k1;
        for (int j = 1; j <= k1; ++j) {
            ll = ll + 1;
            sp = sp + s.c[ll - 1] * h[j - 1];
        }
        out[i] = sp;
    }
    return out;
}

/// numpy quantile (method "linear") of an int64 sample.
inline double quantile_linear(std::vector<int64_t> v, double q) {
    std::sort(v.begin(), v.end());
    const int64_t n = static_cast<int64_t>(v.size());
    double vi = static_cast<double>(n - 1) * q;
    double prevf = std::floor(vi);
    int64_t prev = static_cast<int64_t>(prevf);
    int64_t next = prev + 1;
    if (prev > n - 1) prev = n - 1;
    if (prev < 0) prev = 0;
    if (next > n - 1) next = n - 1;
    if (next < 0) next = 0;
    double gamma = vi - prevf;
    int64_t a = v[prev], b = v[next];
    int64_t diff = b - a;
    if (gamma >= 0.5) return static_cast<double>(b) - static_cast<double>(diff) * (1.0 - gamma);
    return static_cast<double>(a) + static_cast<double>(diff) * gamma;
}

/// Least-squares line through (x, y): (slope, intercept). Stands in for np.polyfit(x, y, 1) (last-bit differences).
inline std::pair<double, double> polyfit1(const std::vector<double>& x, const std::vector<double>& y) {
    const size_t n = x.size();
    double mx = 0, my = 0;
    for (size_t i = 0; i < n; ++i) {
        mx += x[i];
        my += y[i];
    }
    mx /= n;
    my /= n;
    double sxx = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) {
        sxx += (x[i] - mx) * (x[i] - mx);
        sxy += (x[i] - mx) * (y[i] - my);
    }
    double b = sxy / sxx;
    return {b, my - b * mx};
}

}  // namespace detail

// ───────────────────────────────────────────────────────────────────────── pieces of cellcall.py

/// (family, validated)
inline std::pair<std::string, bool> protocol_family(const std::string& protocol) {
    std::string p;
    for (char ch : protocol) {
        if (ch == ' ' || ch == '_' || ch == '-') continue;
        p.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (p.find("parse") != std::string::npos || p.find("splitseq") != std::string::npos ||
        p.find("evercode") != std::string::npos)
        return {"combinatorial", false};
    if (p.find("indrop") != std::string::npos) return {"indrop", false};
    if (p.find("dropseq") != std::string::npos) return {"dropseq", false};
    return {"droplet", true};
}

inline std::string gene_key(const std::string& name) {
    auto p = name.find('_');
    return p == std::string::npos ? name : name.substr(0, p);
}

/// Simple Good-Turing proportions (cellcall.good_turing_proportions).
inline std::vector<double> good_turing_proportions(const std::vector<int64_t>& counts) {
    const size_t G = counts.size();
    int64_t n = 0;
    for (auto v : counts) n += v;
    if (n == 0) return std::vector<double>(G, 1.0 / static_cast<double>(G));
    std::map<int64_t, int64_t> freq;
    int64_t n0 = 0;
    for (auto v : counts) {
        if (v > 0)
            ++freq[v];
        else if (v == 0)
            ++n0;
    }
    std::vector<int64_t> r, nr;
    for (auto& kv : freq) {
        r.push_back(kv.first);
        nr.push_back(kv.second);
    }
    const size_t R = r.size();
    double p0 = r[0] == 1 ? static_cast<double>(nr[0]) / static_cast<double>(n) : 0.0;
    std::vector<double> z(R);
    for (size_t i = 0; i < R; ++i) {
        int64_t q = i > 0 ? r[i - 1] : 0;
        int64_t t = i < R - 1 ? r[i + 1] : 2 * r[i] - q;
        z[i] = static_cast<double>(nr[i]) / (0.5 * static_cast<double>(t - q));
    }
    double a, b;
    if (R > 1) {
        std::vector<double> lr(R), lz(R);
        for (size_t i = 0; i < R; ++i) {
            lr[i] = std::log(static_cast<double>(r[i]));
            lz[i] = std::log(z[i]);
        }
        auto ba = detail::polyfit1(lr, lz);
        b = ba.first;
        a = ba.second;
    } else {
        b = -1.0;
        a = std::log(z[0]);
    }
    std::vector<double> rstar(R);
    bool use_lgt = false;
    for (size_t i = 0; i < R; ++i) {
        double y = static_cast<double>(r[i] + 1) * std::exp(a + b * std::log(static_cast<double>(r[i] + 1))) /
                   std::exp(a + b * std::log(static_cast<double>(r[i])));
        if (!use_lgt && i < R - 1 && r[i + 1] == r[i] + 1) {
            double x = static_cast<double>((r[i] + 1) * nr[i + 1]) / static_cast<double>(nr[i]);
            double sd = 1.96 * std::sqrt(static_cast<double>((r[i] + 1) * (r[i] + 1) * nr[i + 1]) /
                                         static_cast<double>(nr[i] * nr[i]) *
                                         (1.0 + static_cast<double>(nr[i + 1]) / static_cast<double>(nr[i])));
            if (std::fabs(x - y) > sd) {
                rstar[i] = x;
                continue;
            }
        }
        use_lgt = true;
        rstar[i] = y;
    }
    std::vector<double> nrr(R);
    for (size_t i = 0; i < R; ++i) nrr[i] = static_cast<double>(nr[i]) * rstar[i];
    double denom = detail::np_sum(nrr);
    std::map<int64_t, double> lookup;
    for (size_t i = 0; i < R; ++i) lookup[r[i]] = (1 - p0) * rstar[i] / denom;
    std::vector<double> out(G);
    double unseen = n0 ? p0 / static_cast<double>(n0) : 0.0;
    for (size_t g = 0; g < G; ++g) out[g] = counts[g] > 0 ? lookup[counts[g]] : unseen;
    double tot = detail::np_sum(out);
    for (auto& v : out) v = v / tot;
    return out;
}

/// The column subset `cols` of X as (indices, data, per-column totals) in CSC order.
struct ColumnView {
    std::vector<int64_t> indptr;
    std::vector<int32_t> indices;
    std::vector<int64_t> data;
    std::vector<int64_t> totals;
};
inline ColumnView take_columns(const GeneCounts& X, const std::vector<uint32_t>& cols) {
    ColumnView v;
    v.indptr.push_back(0);
    for (uint32_t j : cols) {
        int64_t t = 0;
        for (int64_t p = X.indptr[j]; p < X.indptr[j + 1]; ++p) {
            v.indices.push_back(X.indices[p]);
            v.data.push_back(X.data[p]);
            t += X.data[p];
        }
        v.totals.push_back(t);
        v.indptr.push_back(static_cast<int64_t>(v.indices.size()));
    }
    return v;
}

/// Dirichlet-multinomial alpha by maximum likelihood over the ambient columns (cellcall.dm_alpha_mle).
inline double dm_alpha_mle(const ColumnView& Y, const std::vector<double>& prop) {
    std::vector<double> tk, y, pr;
    std::vector<size_t> colstart;
    size_t nkeep = 0;
    for (size_t j = 0; j + 1 < Y.indptr.size(); ++j) {
        if (Y.totals[j] <= 0) continue;
        ++nkeep;
        tk.push_back(static_cast<double>(Y.totals[j]));
        for (int64_t p = Y.indptr[j]; p < Y.indptr[j + 1]; ++p) {
            y.push_back(static_cast<double>(Y.data[p]));
            pr.push_back(prop[Y.indices[p]]);
        }
    }
    std::vector<double> b1(tk.size()), b2(y.size());
    auto nll = [&](double la) {
        double a = std::exp(la);
        for (size_t i = 0; i < tk.size(); ++i) b1[i] = detail::gammaln(tk[i] + a);
        for (size_t i = 0; i < y.size(); ++i) {
            double ap = a * pr[i];
            b2[i] = detail::gammaln(y[i] + ap) - detail::gammaln(ap);
        }
        double v = static_cast<double>(nkeep) * detail::gammaln(a) - detail::np_sum(b1) + detail::np_sum(b2);
        return -v;
    };
    double x = detail::fminbound(nll, std::log(1e-2), std::log(1e7), 1e-4);
    return std::exp(x);
}

/// Full DM log-probability of every column (cellcall.dm_logprob).
inline std::vector<double> dm_logprob(const ColumnView& Y, const std::vector<double>& prop, double alpha) {
    const size_t nc = Y.totals.size();
    std::vector<double> out(nc);
    const double ga = detail::gammaln(alpha);
    for (size_t j = 0; j < nc; ++j) {
        double s = 0.0;
        for (int64_t p = Y.indptr[j]; p < Y.indptr[j + 1]; ++p) {
            double ap = alpha * prop[Y.indices[p]];
            double yv = static_cast<double>(Y.data[p]);
            double per = detail::gammaln(yv + ap) - detail::gammaln(ap) - detail::gammaln(yv + 1.0);
            s += per;
        }
        double t = static_cast<double>(Y.totals[j]);
        out[j] = detail::gammaln(t + 1.0) + ga - detail::gammaln(t + alpha) + s;
    }
    return out;
}

/// cellcall.alias_table
inline void alias_table(const std::vector<double>& p, std::vector<double>& prob, std::vector<int64_t>& alias) {
    const int64_t K = static_cast<int64_t>(p.size());
    std::vector<double> q(K);
    for (int64_t i = 0; i < K; ++i) q[i] = p[i] * static_cast<double>(K);
    alias.assign(K, 0);
    prob.assign(K, 0.0);
    std::vector<int64_t> small, large;
    for (int64_t i = 0; i < K; ++i)
        if (q[i] < 1.0) small.push_back(i);
    for (int64_t i = 0; i < K; ++i)
        if (q[i] >= 1.0) large.push_back(i);
    while (!small.empty() && !large.empty()) {
        int64_t s = small.back();
        small.pop_back();
        int64_t lg = large.back();
        large.pop_back();
        prob[s] = q[s];
        alias[s] = lg;
        q[lg] = q[lg] + q[s] - 1.0;
        (q[lg] < 1.0 ? small : large).push_back(lg);
    }
    for (int64_t i : large) prob[i] = 1.0;
    for (int64_t i : small) prob[i] = 1.0;
}

/// cellcall.bh
inline std::vector<double> bh(const std::vector<double>& p) {
    const size_t n = p.size();
    std::vector<double> out(n);
    if (n == 0) return out;
    std::vector<size_t> o(n);
    for (size_t i = 0; i < n; ++i) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](size_t a, size_t b) { return p[a] < p[b]; });
    std::vector<double> q(n);
    for (size_t i = 0; i < n; ++i) q[i] = p[o[i]] * static_cast<double>(n) / static_cast<double>(i + 1);
    for (size_t i = n - 1; i-- > 0;) q[i] = std::min(q[i], q[i + 1]);
    for (size_t i = 0; i < n; ++i) out[o[i]] = std::min(q[i], 1.0);
    return out;
}

namespace detail {
/// Descending unique totals > lower and their average (1-based) ranks among all totals > 0; false if < 10 totals > 0.
inline bool rank_curve(const std::vector<int64_t>& totals, int64_t lower, std::vector<double>& u, std::vector<double>& rr) {
    std::vector<int64_t> t;
    for (auto v : totals)
        if (v > 0) t.push_back(v);
    if (t.size() < 10) return false;
    std::sort(t.begin(), t.end(), std::greater<int64_t>());
    for (size_t i = 0; i < t.size();) {
        size_t j = i;
        while (j < t.size() && t[j] == t[i]) ++j;
        if (t[i] > lower) {
            u.push_back(static_cast<double>(t[i]));
            rr.push_back(static_cast<double>(i + (j - 1)) / 2 + 1);
        }
        i = j;
    }
    return true;
}

/// Total at the maximum curvature of the smoothing spline through (xs, ys) = (log10 rank, log10 total); NaN when undefined.
inline double spline_knee(const std::vector<double>& xs, const std::vector<double>& ys) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    // np.var (ddof 0)
    double mean = detail::np_sum(ys) / static_cast<double>(ys.size());
    std::vector<double> dev(ys.size());
    for (size_t i = 0; i < ys.size(); ++i) {
        double d = ys[i] - mean;
        dev[i] = d * d;
    }
    double var = detail::np_sum(dev) / static_cast<double>(ys.size());
    double s = static_cast<double>(xs.size()) * var * 0.002;
    if (!(s > 0)) return nan;
    detail::Spline spl = detail::curfit(xs, ys, s);
    if (spl.t.size() < 8) return nan;  // ier = -1 (interpolating spline) is a valid FITPACK result, as in scipy
    detail::Spline d1s = detail::splder(spl), d2s = detail::splder(d1s);
    std::vector<double> d1 = detail::splev(d1s, xs), d2 = detail::splev(d2s, xs);
    size_t best = 0;
    double bestv = std::numeric_limits<double>::infinity();
    bool any = false;
    for (size_t i = 0; i < xs.size(); ++i) {
        double sq = d1[i] * d1[i];
        double v = d2[i] / std::pow(1 + sq, 1.5);
        if (std::isnan(v)) return nan;
        if (!any || v < bestv) {
            bestv = v;
            best = i;
            any = true;
        }
    }
    return std::pow(10.0, ys[best]);
}
}  // namespace detail

/// DropletUtils barcodeRanks knee without curve bounds (cellcall.knee, VERSION 2.0): the spline is fitted on every total
/// > lower after the top `exclude_from` unique totals. Kept for reference; call() uses knee_bounded() since 2.1.
inline double knee(const std::vector<int64_t>& totals, int64_t lower, int exclude_from = 50) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> u, rr;
    if (!detail::rank_curve(totals, lower, u, rr)) return nan;
    if (static_cast<int>(u.size()) <= exclude_from + 5) return nan;
    std::vector<double> xs, ys;
    for (size_t i = exclude_from; i < u.size(); ++i) {
        xs.push_back(std::log10(rr[i]));
        ys.push_back(std::log10(u[i]));
    }
    return detail::spline_knee(xs, ys);
}

/// The knee called_v2 uses as `retain` (VERSION 2.1; cellcall.knee_bounded, bench/called_v2_truth/common.py): the same
/// spline, fitted only between DropletUtils' curve bounds (barcodeRanks / .find_curve_bounds). On the (log10 rank,
/// log10 total) points with total > lower, d1n = diff(y) / diff(x); skip = number of points with log10 rank <=
/// log10(exclude_from); right = skip + argmin(d1n[skip:]) (the steepest drop), left = skip + argmax(d1n[skip:right+1]).
/// Without the bounds a plateau `lower` of 5-100 puts the ambient shoulder inside the fit and the knee can land just above
/// `lower`, which auto-calls every tested barcode (docs/called-v2-truth.md).
inline double knee_bounded(const std::vector<int64_t>& totals, int64_t lower, int exclude_from = 50) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> u, rr;
    if (!detail::rank_curve(totals, lower, u, rr)) return nan;
    const size_t m = u.size();
    if (m < 10) return nan;
    std::vector<double> x(m), y(m);
    for (size_t i = 0; i < m; ++i) {
        x[i] = std::log10(rr[i]);
        y[i] = std::log10(u[i]);
    }
    std::vector<double> d1n(m - 1);
    for (size_t i = 0; i + 1 < m; ++i) d1n[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    const double xcut = std::log10(static_cast<double>(exclude_from));
    size_t nle = 0;
    for (double v : x) nle += v <= xcut ? 1 : 0;
    const size_t skip = std::min(d1n.size() - 1, nle);
    size_t right = skip;  // np.argmin / np.argmax: first occurrence
    for (size_t i = skip; i < d1n.size(); ++i)
        if (d1n[i] < d1n[right]) right = i;
    size_t left = skip;
    for (size_t i = skip; i <= right; ++i)
        if (d1n[i] > d1n[left]) left = i;
    std::vector<double> xs(x.begin() + left, x.begin() + right + 1), ys(y.begin() + left, y.begin() + right + 1);
    if (xs.size() < 6) return std::pow(10.0, ys[0]);
    return detail::spline_knee(xs, ys);
}

/// scipy.special.ndtr (cephes): standard normal CDF.
inline double ndtr(double a) {
    const double s = 0.70710678118654752440;  // NPY_SQRT1_2
    double x = a * s, z = std::fabs(x);
    if (z < s) return 0.5 + 0.5 * std::erf(x);
    // cephes erfc underflows to 0 once z * z > MAXLOG (libm would return subnormals there)
    double y = z * z > 7.09782712893383996843e2 ? 0.0 : 0.5 * std::erfc(z);
    return x > 0 ? 1.0 - y : y;
}

/// cellcall.plateau_lower
inline int64_t plateau_lower(const std::vector<int64_t>& tot, const std::vector<uint8_t>& called) {
    std::vector<int64_t> tc;
    for (size_t i = 0; i < tot.size(); ++i)
        if (called[i]) tc.push_back(tot[i]);
    double q05 = !tc.empty() ? detail::quantile_linear(tc, 0.05) : 200.0;
    double cut = std::max(4.0, 0.5 * q05);
    std::vector<double> lt;
    for (auto v : tot)
        if (v > 0 && static_cast<double>(v) < cut) lt.push_back(std::log10(static_cast<double>(v)));
    auto clip = [](double v, double lo, double hi) { return std::min(std::max(v, lo), hi); };
    if (lt.size() < 100) return static_cast<int64_t>(clip(0.5 * q05, 5, 100));
    double mx = *std::max_element(lt.begin(), lt.end());
    // bins = np.arange(0, mx + 0.1, 0.1): edges i * 0.1
    double stop = mx + 0.1;
    int64_t nb = static_cast<int64_t>(std::ceil((stop - 0.0) / 0.1));
    if (nb < 2) nb = 2;  // np.arange gives >= 2 edges here (mx >= 0)
    std::vector<double> e(nb);
    e[0] = 0.0;
    if (nb > 1) e[1] = 0.0 + 0.1;
    double delta = e[1] - e[0];
    for (int64_t i = 2; i < nb; ++i) e[i] = e[0] + static_cast<double>(i) * delta;
    std::sort(lt.begin(), lt.end());
    std::vector<int64_t> cum(nb);
    for (int64_t i = 0; i < nb - 1; ++i)
        cum[i] = std::lower_bound(lt.begin(), lt.end(), e[i]) - lt.begin();
    cum[nb - 1] = std::upper_bound(lt.begin(), lt.end(), e[nb - 1]) - lt.begin();
    const int64_t nh = nb - 1;
    std::vector<double> h(nh);
    for (int64_t i = 0; i < nh; ++i) h[i] = static_cast<double>(cum[i + 1] - cum[i]);
    // np.convolve(h, ones(3)/3, mode="same")
    const double w = 1.0 / 3.0;
    std::vector<double> hs(nh);
    for (int64_t i = 0; i < nh; ++i) {
        double acc = 0.0;
        bool first = true;
        for (int64_t j = i - 1; j <= i + 1; ++j) {
            if (j < 0 || j >= nh) continue;
            double term = h[j] * w;
            acc = first ? term : acc + term;
            first = false;
        }
        hs[i] = acc;
    }
    int64_t jbest = 0;
    for (int64_t i = 1; i < nh; ++i)
        if (hs[i] > hs[jbest]) jbest = i;
    double mode = std::pow(10.0, 0.5 * (e[jbest] + e[jbest + 1]));
    double r = std::nearbyint(3 * mode);
    return static_cast<int64_t>(clip(r, 5, 100));
}

/// Monte Carlo counts: for every simulated tested barcode b (with total tt[b] and log-probability lp[b]) the number of
/// Polya-urn paths whose log-probability at that total is <= lp[b] (searchsorted(sorted sims, lp, "right")).
/// With null_mean / null_sd, also the mean and (ddof 0) standard deviation of the niters simulated log-probabilities at
/// tt[b] (sims.mean(axis=0), sims.std(axis=0)), for the tail p-values of VERSION 2.1. They are accumulated as exact
/// fixed-point sums (2^-40 resolution) of the deviations from path 0, so they do not depend on the thread count; they agree
/// with numpy's to about 1e-12 relative (numpy sums the sorted paths in order, which a streaming kernel cannot reproduce).
inline std::vector<int64_t> mc_counts(const std::vector<double>& prop, double alpha, const std::vector<int64_t>& tt,
                                      const std::vector<double>& lp, int64_t niters, int64_t seed, int threads,
                                      std::vector<double>* null_mean = nullptr, std::vector<double>* null_sd = nullptr) {
    const size_t nb = tt.size();
    std::vector<int64_t> ut(tt);
    std::sort(ut.begin(), ut.end());
    ut.erase(std::unique(ut.begin(), ut.end()), ut.end());
    const size_t T = ut.size();
    // per unique total: the observed log-probabilities, sorted, and the barcodes they belong to
    std::vector<size_t> col_start(T + 1, 0);
    std::vector<size_t> col_of(nb);
    for (size_t b = 0; b < nb; ++b) {
        col_of[b] = std::lower_bound(ut.begin(), ut.end(), tt[b]) - ut.begin();
        ++col_start[col_of[b] + 1];
    }
    for (size_t c = 0; c < T; ++c) col_start[c + 1] += col_start[c];
    std::vector<size_t> order(nb);
    {
        std::vector<size_t> fill(col_start.begin(), col_start.end() - 1);
        for (size_t b = 0; b < nb; ++b) order[fill[col_of[b]]++] = b;
        for (size_t c = 0; c < T; ++c)
            std::sort(order.begin() + col_start[c], order.begin() + col_start[c + 1],
                      [&](size_t x, size_t y) { return lp[x] < lp[y]; });
    }
    std::vector<double> obs(nb);
    for (size_t i = 0; i < nb; ++i) obs[i] = lp[order[i]];

    std::vector<double> prob;
    std::vector<int64_t> alias;
    alias_table(prop, prob, alias);
    const int64_t K = static_cast<int64_t>(prop.size());
    const int64_t tmax = T ? ut[T - 1] : 0;
    if (threads < 1) threads = 1;
    const bool moments = null_mean != nullptr || null_sd != nullptr;
    // one Polya-urn path; at(k, v) is called with its log-probability v at each unique total ut[k]
    auto run_path = [&](int64_t it, std::vector<int32_t>& counts, std::vector<int64_t>& draws, auto&& at) {
        detail::NumbaRng rng(static_cast<uint32_t>(seed + it));
        std::fill(counts.begin(), counts.end(), 0);
        double lpv = 0.0;
        size_t k = 0;
        for (int64_t n = 0; n < tmax; ++n) {
            int64_t g;
            if (rng.random() < alpha / (alpha + static_cast<double>(n))) {
                int64_t i = rng.randint(K);
                g = rng.random() < prob[i] ? i : alias[i];
            } else {
                g = draws[rng.randint(n)];
            }
            int32_t c = counts[g];
            double cd = static_cast<double>(c);
            double ap = alpha * prop[g];
            double term = std::log(static_cast<double>(n + 1)) - std::log(static_cast<double>(n) + alpha);
            term = term + std::log(cd + ap);
            term = term - std::log(static_cast<double>(static_cast<int64_t>(c) + 1));
            lpv += term;
            counts[g] = c + 1;
            draws[n] = g;
            while (k < T && ut[k] == n + 1) {
                at(k, lpv);
                ++k;
            }
        }
    };
    // moments: deviations from path 0, quantised to 2^-40 and summed exactly (order-free, hence thread-free)
    const double kScale = 1099511627776.0;  // 2^40
    std::vector<double> shift(moments ? T : 0, 0.0);
    if (moments && niters > 0) {
        std::vector<int32_t> counts(K);
        std::vector<int64_t> draws(tmax);
        run_path(0, counts, draws, [&](size_t k, double v) { shift[k] = v; });
    }
    std::vector<std::vector<__int128>> s1(moments ? threads : 0), s2(moments ? threads : 0);
    std::vector<std::vector<int64_t>> diffs(threads, std::vector<int64_t>(nb + 1, 0));
    auto worker = [&](int tid) {
        std::vector<int32_t> counts(K);
        std::vector<int64_t> draws(tmax);
        auto& diff = diffs[tid];
        if (moments) {
            s1[tid].assign(T, 0);
            s2[tid].assign(T, 0);
        }
        for (int64_t it = tid; it < niters; it += threads) {
            run_path(it, counts, draws, [&](size_t k, double lpv) {
                // paths with log-probability v count for every observed lp >= v
                const double* lo = obs.data() + col_start[k];
                const double* hi = obs.data() + col_start[k + 1];
                size_t pos = std::lower_bound(lo, hi, lpv) - obs.data();
                diff[pos] += 1;
                diff[col_start[k + 1]] -= 1;
                if (moments) {
                    double d = (lpv - shift[k]) * kScale;
                    d = std::min(std::max(d, -9.0e15), 9.0e15);  // |deviation| < ~8000 nats: never binding
                    int64_t qd = std::llround(d);
                    s1[tid][k] += qd;
                    s2[tid][k] += static_cast<__int128>(qd) * qd;
                }
            });
        }
    };
    if (threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; ++t) pool.emplace_back(worker, t);
        for (auto& th : pool) th.join();
    }
    std::vector<int64_t> n_le(nb, 0);
    int64_t run = 0;
    for (size_t i = 0; i < nb; ++i) {
        for (int t = 0; t < threads; ++t) run += diffs[t][i];
        n_le[order[i]] = run;
    }
    if (moments) {
        std::vector<double> mean(T), sd(T);
        const double nd = static_cast<double>(niters);
        for (size_t k = 0; k < T; ++k) {
            __int128 a = 0, b = 0;
            for (int t = 0; t < threads; ++t) {
                a += s1[t][k];
                b += s2[t][k];
            }
            double m1 = static_cast<double>(a) / kScale / nd;
            double m2 = static_cast<double>(b) / kScale / kScale / nd;
            mean[k] = shift[k] + m1;
            sd[k] = std::sqrt(std::max(m2 - m1 * m1, 0.0));
        }
        if (null_mean) null_mean->resize(nb);
        if (null_sd) null_sd->resize(nb);
        for (size_t b = 0; b < nb; ++b) {
            if (null_mean) (*null_mean)[b] = mean[col_of[b]];
            if (null_sd) (*null_sd)[b] = sd[col_of[b]];
        }
    }
    return n_le;
}

// ───────────────────────────────────────────────────────────────────────── the call

/// cellcall.call: called_v2 from genes x barcodes counts and the knee `called` flags.
inline Result call(const GeneCounts& X, const std::vector<uint8_t>& called, const std::string& protocol = "",
                   const Params& P = Params(), int threads = 1) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    Result R;
    R.params = P;
    R.protocol = protocol;
    const size_t n = called.size();
    auto fam = protocol_family(protocol);
    R.protocol_family = fam.first;
    R.validated = fam.second;
    R.fdr.assign(n, nan);
    R.pvalue.assign(n, nan);
    R.log_prob.assign(n, nan);
    R.called_v2.assign(called.begin(), called.end());
    R.totals.assign(n, 0);
    for (size_t j = 0; j < n && j < X.ncols; ++j)
        for (int64_t p = X.indptr[j]; p < X.indptr[j + 1]; ++p) R.totals[j] += X.data[p];
    for (auto c : called) R.n_called += c ? 1 : 0;
    if (fam.first == "combinatorial" || n == 0) {
        R.method = "stored-call";
        R.n_called_v2 = R.n_called;
        return R;
    }
    const auto& tot = R.totals;
    int64_t lower = P.lower_rule == "plateau" ? plateau_lower(tot, called) : std::stoll(P.lower_rule);
    R.lower = lower;
    std::vector<uint32_t> amb, ti;
    for (size_t j = 0; j < n; ++j) {
        if (called[j]) continue;
        if (tot[j] > 0 && tot[j] <= lower) amb.push_back(static_cast<uint32_t>(j));
        if (tot[j] > std::max(lower, P.test_min)) ti.push_back(static_cast<uint32_t>(j));
    }
    double kn = P.retain == "knee" ? knee_bounded(tot, lower) : nan;  // 2.1: curve-bounded knee
    bool has_retain = std::isfinite(kn) && kn > static_cast<double>(lower);
    R.retain_total = has_retain ? kn : nan;
    R.n_ambient = static_cast<int64_t>(amb.size());
    for (auto j : amb) R.ambient_umi += tot[j];
    R.n_tested = static_cast<int64_t>(ti.size());
    if (amb.size() < 10 || R.ambient_umi < 100 || ti.empty()) {
        R.error = !ti.empty() ? "ambient pool too small" : "nothing to test";
        R.n_called_v2 = R.n_called;
        return R;
    }
    ColumnView A = take_columns(X, amb);
    std::vector<int64_t> asum(X.nrows, 0);
    for (size_t p = 0; p < A.indices.size(); ++p) asum[A.indices[p]] += A.data[p];
    std::vector<double> prop = good_turing_proportions(asum);
    double alpha = std::isnan(P.alpha_override) ? dm_alpha_mle(A, prop) : P.alpha_override;
    R.alpha = alpha;
    const size_t nt = ti.size();
    std::vector<double> pv(nt, 1.0 / static_cast<double>(P.niters + 1));
    std::vector<uint32_t> simcols;
    std::vector<size_t> simpos;
    for (size_t i = 0; i < nt; ++i) {
        int64_t t = tot[ti[i]];
        bool sim = t <= kMaxSimulatedTotal;
        if (has_retain) {
            if (static_cast<double>(t) >= kn) pv[i] = 0.0;
            sim = sim && static_cast<double>(t) < kn;
        }
        if (sim) {
            simcols.push_back(ti[i]);
            simpos.push_back(i);
        }
    }
    if (!simcols.empty()) {
        ColumnView S = take_columns(X, simcols);
        std::vector<double> lp = dm_logprob(S, prop, alpha);
        std::vector<double> mu, sd;
        std::vector<int64_t> n_le = mc_counts(prop, alpha, S.totals, lp, P.niters, P.seed, threads, &mu, &sd);
        const double floor_p = 1.0 / static_cast<double>(P.niters + 1);
        for (size_t s = 0; s < simcols.size(); ++s) {
            pv[simpos[s]] = static_cast<double>(n_le[s] + 1) / static_cast<double>(P.niters + 1);
            if (n_le[s] == 0) {
                // 2.1: below every simulated path, extrapolate the null tail (normal at that total), capped at the floor,
                // so that BH can pass more than n_tested * fdr / floor barcodes in small samples
                double z = (lp[s] - mu[s]) / std::max(sd[s], 1e-12);
                pv[simpos[s]] = std::min(ndtr(z), floor_p);
                ++R.n_tail;
            }
            R.log_prob[simcols[s]] = lp[s];
        }
    }
    std::vector<double> q = bh(pv);
    for (size_t i = 0; i < nt; ++i) {
        R.fdr[ti[i]] = q[i];
        R.pvalue[ti[i]] = pv[i];
        if (q[i] <= P.fdr) R.called_v2[ti[i]] = 1;
    }
    for (size_t j = 0; j < n; ++j) {
        R.n_called_v2 += R.called_v2[j] ? 1 : 0;
        R.n_added += (R.called_v2[j] && !called[j]) ? 1 : 0;
    }
    return R;
}

/// STARsolo --soloCellFilter CellRanger2.2 nExpected maxPercentile maxMinRatio (SoloFeature_cellFiltering.cpp) on
/// per-barcode totals; barcodes with total 0 are not STAR "detected" cell barcodes and are never called.
inline std::vector<uint8_t> cellranger22(const std::vector<int64_t>& totals, double n_expected = 3000,
                                         double max_percentile = 0.99, double max_min_ratio = 10,
                                         int64_t* umi_min_out = nullptr) {
    std::vector<int64_t> s;
    for (auto v : totals)
        if (v > 0) s.push_back(v);
    std::vector<uint8_t> out(totals.size(), 0);
    if (s.empty()) return out;
    std::sort(s.begin(), s.end(), std::greater<int64_t>());
    int64_t maxind = static_cast<int64_t>(std::round(n_expected * (1.0 - max_percentile)));
    int64_t nmax = s[std::min<int64_t>(static_cast<int64_t>(s.size()) - 1, maxind)];
    int64_t nmin = static_cast<int64_t>(std::round(static_cast<double>(nmax) / max_min_ratio));
    nmin = std::max<int64_t>(nmin, 1);
    for (size_t i = 0; i < totals.size(); ++i) out[i] = totals[i] >= nmin ? 1 : 0;
    if (umi_min_out) *umi_min_out = nmin;
    return out;
}

/// Gene-level exon + intron counts keyed like cellcall.gene_counts (feature name before the first '_'), genes sorted.
template <typename CSC>
GeneCounts gene_counts(const CSC& exon, const std::vector<std::string>& exon_names, const CSC* intron,
                       const std::vector<std::string>* intron_names) {
    GeneCounts X;
    std::vector<const CSC*> mats;
    std::vector<const std::vector<std::string>*> names;
    if (exon.ncols > 0 && exon_names.size() == exon.nrows) {
        mats.push_back(&exon);
        names.push_back(&exon_names);
    }
    if (intron && intron_names && intron->ncols > 0 && intron_names->size() == intron->nrows &&
        (mats.empty() || intron->ncols == mats[0]->ncols)) {
        mats.push_back(intron);
        names.push_back(intron_names);
    }
    if (mats.empty()) return X;
    std::vector<std::string> keys;
    for (auto* nm : names)
        for (auto& s : *nm) keys.push_back(gene_key(s));
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<std::vector<int32_t>> rowmap(mats.size());
    for (size_t m = 0; m < mats.size(); ++m) {
        rowmap[m].resize(names[m]->size());
        for (size_t r = 0; r < names[m]->size(); ++r)
            rowmap[m][r] = static_cast<int32_t>(
                std::lower_bound(keys.begin(), keys.end(), gene_key((*names[m])[r])) - keys.begin());
    }
    X.nrows = static_cast<uint32_t>(keys.size());
    X.ncols = mats[0]->ncols;
    X.genes = keys;
    X.indptr.assign(static_cast<size_t>(X.ncols) + 1, 0);
    std::vector<int64_t> acc(X.nrows, 0);
    std::vector<int32_t> active;
    for (uint32_t j = 0; j < X.ncols; ++j) {
        for (size_t m = 0; m < mats.size(); ++m) {
            const CSC& M = *mats[m];
            for (auto p = M.indptr[j]; p < M.indptr[j + 1]; ++p) {
                int32_t g = rowmap[m][M.indices[p]];
                if (acc[g] == 0) active.push_back(g);
                acc[g] += static_cast<int64_t>(M.data[p]);
            }
        }
        std::sort(active.begin(), active.end());
        for (int32_t g : active) {
            if (acc[g] != 0) {
                X.indices.push_back(g);
                X.data.push_back(acc[g]);
            }
            acc[g] = 0;
        }
        active.clear();
        X.indptr[j + 1] = static_cast<int64_t>(X.indices.size());
    }
    return X;
}

/// cell_calls.tsv with one row per barcode (matrix column order):
///   barcode, is_cell (= called_v2: the call every downstream output and the .s1 `called` flag use), called_knee (the
///   CellRanger2.2 knee set, the input of the call; it was `is_cell` and `called` before), called_v2 (1/0, the same set
///   as is_cell), umi_exon (knee input), umi_total (exon + intron, the called_v2 input), fdr, pvalue, log_prob
///   (NA = not tested / not simulated), call_method.
template <typename TotT>
inline bool write_cell_calls_tsv(const std::string& path, const std::vector<std::string>& barcodes,
                                 const std::vector<TotT>& exon_totals, const Result& R,
                                 const std::vector<uint8_t>& called) {
    FILE* f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    std::fputs("barcode\tis_cell\tcalled_knee\tcalled_v2\tumi_exon\tumi_total\tfdr\tpvalue\tlog_prob\tcall_method\n", f);
    auto num = [&](double v, const char* fmt) {
        if (std::isnan(v))
            std::fputs("NA", f);
        else
            std::fprintf(f, fmt, v);
    };
    for (size_t j = 0; j < barcodes.size() && j < R.called_v2.size(); ++j) {
        const bool c = j < called.size() && called[j];
        std::fprintf(f, "%s\t%s\t%d\t%d\t%llu\t%lld\t", barcodes[j].c_str(), R.called_v2[j] ? "TRUE" : "FALSE", c ? 1 : 0,
                     R.called_v2[j] ? 1 : 0,
                     static_cast<unsigned long long>(j < exon_totals.size() ? exon_totals[j] : 0),
                     static_cast<long long>(R.totals[j]));
        num(static_cast<double>(R.fdr[j]), "%.6g");
        std::fputc('\t', f);
        num(R.pvalue[j], "%.6g");
        std::fputc('\t', f);
        num(R.log_prob[j], "%.10g");
        std::fprintf(f, "\t%s\n", R.method == kMethod ? "called_v2" : R.method.c_str());
    }
    return std::fclose(f) == 0;
}

/// The call's parameters as a JSON object (the keys of cellcall.call's params dict).
inline std::string params_json(const Result& R, const std::string& indent = "  ") {
    auto num = [](double v) {
        if (!std::isfinite(v)) return std::string("null");
        std::ostringstream o;
        o.precision(17);
        o << v;
        return o.str();
    };
    std::ostringstream o;
    o << "{\n"
      << indent << "  \"method\": \"" << R.method << "\",\n"
      << indent << "  \"version\": \"" << kVersion << "\",\n"
      << indent << "  \"counts\": \"exon+intron\",\n"
      << indent << "  \"protocol\": \"" << R.protocol << "\",\n"
      << indent << "  \"protocol_family\": \"" << R.protocol_family << "\",\n"
      << indent << "  \"validated\": " << (R.validated ? "true" : "false") << ",\n"
      << indent << "  \"lower_rule\": \"" << R.params.lower_rule << "\",\n"
      << indent << "  \"test_min\": " << R.params.test_min << ",\n"
      << indent << "  \"fdr\": " << num(R.params.fdr) << ",\n"
      << indent << "  \"retain\": \"" << R.params.retain << "\",\n"
      << indent << "  \"niters\": " << R.params.niters << ",\n"
      << indent << "  \"seed\": " << R.params.seed << ",\n"
      << indent << "  \"lower\": " << R.lower << ",\n"
      << indent << "  \"retain_total\": " << num(R.retain_total) << ",\n"
      << indent << "  \"n_ambient\": " << R.n_ambient << ",\n"
      << indent << "  \"ambient_umi\": " << R.ambient_umi << ",\n"
      << indent << "  \"n_tested\": " << R.n_tested << ",\n"
      << indent << "  \"alpha\": " << num(R.alpha) << ",\n"
      << indent << "  \"n_tail\": " << R.n_tail << ",\n"
      << indent << "  \"n_called\": " << R.n_called << ",\n"
      << indent << "  \"n_called_v2\": " << R.n_called_v2 << ",\n"
      << indent << "  \"n_added\": " << R.n_added << ",\n"
      << indent << "  \"error\": \"" << R.error << "\",\n"
      << indent << "  \"seconds\": " << num(R.seconds) << "\n"
      << indent << "}";
    return o.str();
}

}  // namespace calledv2
}  // namespace singlet

#if defined(__clang__)
#pragma clang fp contract(on)
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
