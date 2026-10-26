# SPDX-License-Identifier: MIT
"""Write tests/cpp/called_v2_fixture.h: the expected output of the called_v2 reference (cellcall.call, VERSION 2.1) on two
synthetic droplet samples that tests/cpp/test_called_v2.cpp regenerates bit for bit (splitmix64, integer sampling only),
plus a barcode-rank curve on which the unbounded knee falls in the ambient shoulder.

    python -P bench/called_ed/make_cpp_fixture.py        (needs numpy, scipy, numba)

The reference is python/singlet/store/cellcall.py (call_detail, VERSION 2.1). The cases shared with the Python tests are the
golden vectors in tests/data/called_ed_golden/, which test_called_v2.cpp also reads; this fixture adds the pieces and the
synthetic samples that exercise the C++ internals (thread counts, the Monte Carlo kernel's null moments).

The header also pins reference values of the pieces: scipy.special.gammaln, numba's np.random after np.random.seed,
scipy UnivariateSpline knots, numpy quantile and the Good-Turing proportions.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "python"))
from singlet.store import cellcall  # noqa: E402

assert cellcall.VERSION == "2.1", cellcall.VERSION

M64 = (1 << 64) - 1


class SplitMix:
    def __init__(self, seed: int):
        self.s = seed & M64

    def next(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & M64
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
        return z ^ (z >> 31)


def synth(seed=20261024, G=120, n_cells=150, n_low=100, n_amb=2500, n_fat=150):
    """Same generator as test_called_v2.cpp::synth (keep in step)."""
    r = SplitMix(seed)
    amb = [1 + (r.next() % 50) * (r.next() % 50) for _ in range(G)]
    types = [[1 + (r.next() % 40) * (r.next() % 40) * (r.next() % 4) for _ in range(G)] for _ in range(3)]

    def cum(w):
        out, c = [], 0
        for v in w:
            c += v
            out.append(c)
        return out

    camb, ctypes = cum(amb), [cum(w) for w in types]

    def draw(cw):
        u = r.next() % cw[-1]
        lo, hi = 0, len(cw) - 1
        while lo < hi:  # first index with cw[i] > u
            mid = (lo + hi) // 2
            if cw[mid] > u:
                hi = mid
            else:
                lo = mid + 1
        return lo

    cols = []
    # n_fat: empty droplets deep enough to be tested (50-150 UMIs, ambient + a little cell RNA)
    total = n_cells + n_low + n_fat + n_amb
    for i in range(total):
        kind = r.next() % total
        x = [0] * G
        if kind < n_cells:
            T, mix = 300 + r.next() % 3000, 85
        elif kind < n_cells + n_low:
            T, mix = 60 + r.next() % 200, 60
        elif kind < n_cells + n_low + n_fat:
            T, mix = 50 + r.next() % 100, r.next() % 8
        else:
            T, mix = 1 + r.next() % 25, r.next() % 20
        ty = ctypes[r.next() % 3]
        for _ in range(T):
            g = draw(ty) if (r.next() % 100) < mix else draw(camb)
            x[g] += 1
        cols.append(x)
    return np.array(cols, dtype=np.int64).T  # genes x barcodes


def cr22(tot):
    s = np.sort(tot[tot > 0])[::-1]
    nmax = s[min(len(s) - 1, int(round(3000 * (1 - 0.99))))]
    nmin = max(int(round(nmax / 10)), 1)
    return tot >= nmin


def knee_unbounded(totals, lower, exclude_from=50):
    """The 2.0 knee (cellcall.knee before 2.1): the spline over every total > lower after the top exclude_from unique
    totals, without DropletUtils' curve bounds. Pins the C++ calledv2::knee."""
    from scipy.interpolate import UnivariateSpline

    u, rr = cellcall._rank_curve(np.asarray(totals), lower)
    if len(u) <= exclude_from + 5:
        return float("nan")
    xs, ys = np.log10(rr)[exclude_from:], np.log10(u)[exclude_from:]
    spl = UnivariateSpline(xs, ys, k=3, s=len(xs) * np.var(ys) * 0.002)
    d1, d2 = spl.derivative(1)(xs), spl.derivative(2)(xs)
    return float(10 ** ys[np.argmin(d2 / (1 + d1**2) ** 1.5)])


def hexf(v: float) -> str:
    return float(v).hex() if np.isfinite(v) else "NAN"


def main():
    M = synth()
    X = sp.csc_matrix(M)
    X.sort_indices()
    exon_tot = M.sum(0)
    called = cr22(exon_tot)
    cv2, pvalue, fdr, params = cellcall.call_detail(X, called)
    tested = np.nonzero(~np.isnan(fdr))[0]
    pv = pvalue[tested]

    # tail sample: many deep empty droplets, low-RNA cells below the knee all at the Monte Carlo floor, fewer than
    # n_tested / 10 of them, so BH passes none of them without the 2.1 tail p-values
    tail_args = dict(seed=11, n_cells=150, n_low=200, n_fat=3000, n_amb=4000)
    Mt = synth(**tail_args)
    Xt = sp.csc_matrix(Mt)
    Xt.sort_indices()
    called_t = cr22(Mt.sum(0))
    cv2_t, pvalue_t, fdr_t, params_t = cellcall.call_detail(Xt, called_t)
    tested_t = np.nonzero(~np.isnan(fdr_t))[0]
    pv_t = pvalue_t[tested_t]
    floor = 1.0 / (params_t["niters"] + 1)
    tot_t = Mt.sum(0)
    amb_t = (tot_t > 0) & (tot_t <= params_t["lower"]) & ~called_t
    prop_t = cellcall.good_turing_proportions(np.asarray(Xt[:, amb_t].sum(axis=1)).ravel().astype(np.int64))
    n_tail = int(((pv_t > 0) & (pv_t < floor)).sum())

    # kernel check with the reference's own prop / alpha (independent of the last-bit Good-Turing / alpha differences):
    # 30 simulated tested barcodes of the tail sample, their DM log-probability, Monte Carlo count and the null mean / sd
    tt_t = Mt.sum(0)[tested_t]
    simk = np.nonzero((tt_t < params_t["retain_total"]) & (tt_t <= cellcall.MAX_SIMULATED_TOTAL))[0]
    tl = simk[(pv_t[simk] > 0) & (pv_t[simk] < floor)]
    rest = np.setdiff1d(simk, tl)
    pick = np.sort(np.concatenate([tl[np.linspace(0, len(tl) - 1, 15).round().astype(int)],
                                   rest[np.linspace(0, len(rest) - 1, 15).round().astype(int)]]))
    kcols = tested_t[pick]
    k_lp = cellcall.dm_logprob(Xt[:, kcols], prop_t, params_t["alpha"])
    k_tt = Mt.sum(0)[kcols].astype(np.int64)
    k_ut = np.unique(k_tt)
    prob, alias = cellcall.alias_table(prop_t)
    sims = cellcall._mc_kernel()(prob, alias, prop_t, float(params_t["alpha"]), k_ut, int(params_t["niters"]),
                                  int(params_t["seed"]))
    sims.sort(axis=0)
    kc = np.searchsorted(k_ut, k_tt)
    k_nle = np.array([np.searchsorted(sims[:, c], o, side="right") for c, o in zip(kc, k_lp)])
    k_mu, k_sd = sims.mean(axis=0)[kc], sims.std(axis=0)[kc]
    assert (k_nle == 0).sum() >= 5, k_nle
    assert 50 < n_tail and int((pv_t <= floor).sum()) < len(tested_t) / 10, "tail sample does not exercise the tail p"

    # knee sample: a barcode-rank curve (cells, an ambient shoulder of 10-100 UMIs, empties) on which the unbounded knee
    # falls in the shoulder and the bounded knee among the cells
    knee_tot = np.concatenate([np.round(10 ** np.linspace(3.3, 4.0, 2000)), np.round(10 ** np.linspace(1.0, 2.0, 1000)),
                               np.round(10 ** np.linspace(0.0, 1.0, 60000))]).astype(np.int64)
    knee_lower = 8
    kn_old, kn_new = knee_unbounded(knee_tot, knee_lower), cellcall.knee(knee_tot, knee_lower)
    assert kn_old <= 100 < 1000 < kn_new, (kn_old, kn_new)
    kv, kc = np.unique(knee_tot, return_counts=True)

    # pieces
    gl_x = [0.001, 0.5, 1.0, 1.5, 2.0, 2.5, 3.7, 12.9, 13.0, 50.5, 999.0, 1000.0, 12345.678, 3.3e8]
    from scipy.special import gammaln
    from scipy.interpolate import UnivariateSpline
    import numba

    @numba.njit
    def rng_seq(seed):
        np.random.seed(seed)
        out = np.empty(12)
        for i in range(4):
            out[i] = np.random.random()
        for i, n in enumerate((2, 7, 1000, 123456789, 5000000000, 3)):
            out[4 + i] = np.random.randint(0, n)
        out[10] = np.random.random()
        out[11] = np.random.randint(0, 38606)
        return out

    rs = rng_seq(20261023 + 5)
    xs = np.log10(np.arange(1, 400) * 1.0 + 0.5)
    ys = 3.5 - 0.2 * xs - 1.2 / (1 + np.exp(-(xs - 1.7) * 6)) + 0.01 * np.sin(xs * 37)
    s = len(xs) * np.var(ys) * 0.002
    spl = UnivariateSpline(xs, ys, k=3, s=s)
    t = spl._eval_args[0]
    asum = np.asarray(X[:, (exon_tot > 0) & (exon_tot <= 25)].sum(axis=1)).ravel().astype(np.int64)
    gt = cellcall.good_turing_proportions(asum)

    L = []
    L.append("// Generated by bench/called_ed/make_cpp_fixture.py from the called_v2 reference %s (VERSION %s) -- do not edit."
             % ("python/singlet/store/cellcall.py", cellcall.VERSION))
    L.append("#pragma once")
    L.append("#include <cstdint>")
    L.append("#include <cmath>")
    L.append("namespace cv2fix {")
    L.append("constexpr uint64_t kSeed = 20261024;")
    L.append("constexpr int kG = 120, kCells = 150, kLow = 100, kAmb = 2500;")
    L.append("constexpr int64_t kSum = %d;  // all counts" % int(M.sum()))
    L.append("constexpr int64_t kLower = %d;" % params["lower"])
    L.append("constexpr double kRetain = %s;" % hexf(params["retain_total"] if params["retain_total"] is not None else np.nan))
    L.append("constexpr double kAlpha = %s;" % hexf(params.get("alpha", np.nan)))
    L.append("constexpr int64_t kNCalled = %d, kNCalledV2 = %d, kNTested = %d;" % (called.sum(), cv2.sum(), len(tested)))
    L.append("constexpr int kTested[] = {%s};" % ", ".join(str(int(i)) for i in tested))
    L.append("constexpr double kFdr[] = {%s};" % ", ".join(hexf(float(fdr[i])) for i in tested))
    L.append("constexpr unsigned char kCalledV2Tested[] = {%s};" % ", ".join(str(int(cv2[i])) for i in tested))
    L.append("constexpr double kPv[] = {%s};" % ", ".join(hexf(v) for v in pv))
    L.append("// tail sample: synth(seed, G = 120, cells, low, amb, fat)")
    L.append("constexpr uint64_t kTSeed = %d;" % tail_args["seed"])
    L.append("constexpr int kTCells = %d, kTLow = %d, kTAmb = %d, kTFat = %d;"
             % (tail_args["n_cells"], tail_args["n_low"], tail_args["n_amb"], tail_args["n_fat"]))
    L.append("constexpr int64_t kTSum = %d;" % int(Mt.sum()))
    L.append("constexpr int64_t kTLower = %d;" % params_t["lower"])
    L.append("constexpr double kTRetain = %s;" % hexf(params_t["retain_total"] if params_t["retain_total"] is not None else np.nan))
    L.append("constexpr int64_t kTNCalled = %d, kTNCalledV2 = %d, kTNTested = %d;"
             % (called_t.sum(), cv2_t.sum(), len(tested_t)))
    L.append("constexpr int kTTested[] = {%s};" % ", ".join(str(int(i)) for i in tested_t))
    L.append("constexpr unsigned char kTCalledV2Tested[] = {%s};" % ", ".join(str(int(cv2_t[i])) for i in tested_t))
    L.append("constexpr double kTPv[] = {%s};" % ", ".join(hexf(v) for v in pv_t))
    L.append("// kernel check on the tail sample with the reference prop / alpha")
    L.append("constexpr double kTProp[] = {%s};" % ", ".join(hexf(v) for v in prop_t))
    L.append("constexpr double kTAlpha = %s;" % hexf(params_t["alpha"]))
    L.append("constexpr int kKN = %d;" % len(kcols))
    L.append("constexpr int kKCols[] = {%s};" % ", ".join(str(int(v)) for v in kcols))
    L.append("constexpr double kKLp[] = {%s};" % ", ".join(hexf(v) for v in k_lp))
    L.append("constexpr int64_t kKNle[] = {%s};" % ", ".join(str(int(v)) for v in k_nle))
    L.append("constexpr double kKMu[] = {%s};" % ", ".join(hexf(v) for v in k_mu))
    L.append("constexpr double kKSd[] = {%s};" % ", ".join(hexf(v) for v in k_sd))
    L.append("// knee curve: totals as (value, count) runs")
    L.append("constexpr int64_t kKneeLower = %d;" % knee_lower)
    L.append("constexpr int64_t kKneeVal[] = {%s};" % ", ".join(str(int(v)) for v in kv))
    L.append("constexpr int64_t kKneeCnt[] = {%s};" % ", ".join(str(int(v)) for v in kc))
    L.append("constexpr double kKneeUnbounded = %s, kKneeBounded = %s;" % (hexf(kn_old), hexf(kn_new)))
    L.append("constexpr double kGlX[] = {%s};" % ", ".join(hexf(v) for v in gl_x))
    L.append("constexpr double kGlY[] = {%s};" % ", ".join(hexf(v) for v in gammaln(np.array(gl_x))))
    L.append("constexpr double kRng[] = {%s};  // numba after np.random.seed(20261028)" % ", ".join(hexf(v) for v in rs))
    L.append("constexpr int kSplineN = %d;" % len(t))
    L.append("constexpr double kSplineT[] = {%s};" % ", ".join(hexf(v) for v in t))
    L.append("constexpr double kGoodTuringFirst[] = {%s};" % ", ".join(hexf(v) for v in gt[:8]))
    L.append("constexpr double kQuantile = %s;  // np.quantile(exon totals of called, 0.05)" % hexf(np.quantile(exon_tot[called], 0.05)))
    L.append("}  // namespace cv2fix")
    out = os.path.join(ROOT, "tests", "cpp", "called_v2_fixture.h")
    with open(out, "w", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    print("wrote", out, "| called", called.sum(), "called_v2", cv2.sum(), "tested", len(tested), params)
    print("tail sample: called", called_t.sum(), "called_v2", cv2_t.sum(), "tested", len(tested_t), "tail-extrapolated", n_tail, "retain", params_t["retain_total"])
    print("knee curve: unbounded", kn_old, "bounded", kn_new)


if __name__ == "__main__":
    main()
