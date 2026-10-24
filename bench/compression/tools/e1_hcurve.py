#!/usr/bin/env python
"""E1: how biased is the per-cell entropy estimate at the depths we have?  (floor bracket by sub-sampling extrapolation)

    python -P tools/e1_hcurve.py [sample:member ...]   -> results/e1_hcurve.tsv

For called cells with N >= NDEEP UMIs, thin each cell's counts binomially to depth d = N * f (f in 1/16..1, several
replicates) and compute the plug-in and Chao-Shen entropy (bits) of the thinned cell.  If the estimate keeps rising
with depth, the estimator at shallow depth is biased low, and the implied sampling floor
    H(Mult(N, pi)) = N H(pi) - log2 N! + sum log2 x!                   (theory_fable.md sec. 2)
is under-estimated by (N * bias).  The curve H_cs(d) is extrapolated with  H(d) = Hinf - a d^-b  (grid fit over b,
linear least squares for Hinf and a) to give an asymptotic entropy per cell; "per cell" because the extrapolation is a
property of the deep cells, applied to shallower cells of the same file only as a bracket.
Corpus data are untrusted input: parsed as numbers only.
"""
import os, sys, math, csv, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e1_split as E

FRACS = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0]
NDEEP = 1500
MAXC = 150


def fit_inf(d, h):
    """h(d) = Hinf - a d^-b ; scan b, solve (Hinf, a) by least squares; return Hinf (nan if not increasing)."""
    best = None
    for b in np.arange(0.05, 1.5, 0.05):
        X = np.column_stack([np.ones_like(d), -d ** (-b)])
        coef, res, *_ = np.linalg.lstsq(X, h, rcond=None)
        r = np.sum((X @ coef - h) ** 2)
        if best is None or r < best[0]:
            best = (r, coef[0], coef[1], b)
    return best[1], best[3]


def main():
    targets = sys.argv[1:] or [f"{s}:{m}" for s in ["pbmc_v2_typical", "pbmc_v3_lowgenes", "pbmc_v2_shallow", "tcell_5p",
                                                   "brain_sn_v3_deep", "brain_sn_v3_intronic", "tumor_3p"] for m in ["exon", "intron"]]
    rng = np.random.default_rng(5)
    rows = []
    for t in targets:
        sample, member = t.split(":")
        d = E.load_matrix(sample, member)
        ptr, ind, val = d["indptr"], d["indices"], d["data"]
        n = len(ptr) - 1
        col = np.repeat(np.arange(n, dtype=np.int64), np.diff(ptr))
        N = np.bincount(col, weights=val.astype(np.float64), minlength=n)
        del col
        called = d["is_cell"] & (N >= E.N_MIN_CELL)
        nd = NDEEP if np.sum(called & (N >= NDEEP)) >= 30 else float(np.percentile(N[called], 90))
        cand = np.flatnonzero(called & (N >= nd))
        if len(cand) < 20:
            print(t, "too few deep cells", flush=True)
            continue
        cand = rng.permutation(cand)[:MAXC]
        Hc = np.zeros((len(cand), len(FRACS)))
        Hp = np.zeros_like(Hc)
        Nd = np.zeros_like(Hc)
        for i, c in enumerate(cand):
            x = val[ptr[c]:ptr[c + 1]].astype(np.int64)
            for j, f in enumerate(FRACS):
                y = x if f == 1.0 else rng.binomial(x, f)
                y = y[y > 0]
                Nd[i, j] = y.sum()
                Hc[i, j] = E.chao_shen_bits(y)
                Hp[i, j] = E.plugin_bits(y)
        dm = np.median(Nd, 0)
        hc = np.median(Hc, 0)
        hp = np.median(Hp, 0)
        # extrapolate per cell then take the median
        hinf = []
        for i in range(len(cand)):
            try:
                v, b = fit_inf(Nd[i], Hc[i])
                if np.isfinite(v) and Hc[i, -1] <= v <= 25:
                    hinf.append(v)
            except Exception:
                pass
        row = [f"{sample}/{member}", len(cand), f"{np.median(Nd[:, -1]):.0f}"] + [f"{x:.0f}" for x in dm] + [f"{x:.3f}" for x in hp] + \
              [f"{x:.3f}" for x in hc] + [f"{np.median(hinf):.3f}" if hinf else "", len(hinf)]
        rows.append(row)
        print(row, flush=True)
    cols = ["file", "n_cells", "median_N"] + [f"d_f{int(round(1 / f))}" for f in FRACS] + \
           [f"Hplug_f{int(round(1 / f))}" for f in FRACS] + [f"Hcs_f{int(round(1 / f))}" for f in FRACS] + ["Hcs_extrap_inf", "n_fit"]
    with open(os.path.join(E.RES, "e1_hcurve.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(cols)
        w.writerows(rows)


if __name__ == "__main__":
    main()
