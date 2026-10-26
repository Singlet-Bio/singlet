# SPDX-License-Identifier: MIT
"""Score every caller on the independent truths.

    score.py TRUTH_DIR OUT_JSON [DS ...]

Decided barcodes only (truth cell or empty; unknown excluded). Variants: ``lenient`` (any non-ambient single-source barcode is a
cell, damaged included) and ``strict`` (damaged = mito / intronic share outside the intact-cell range counts as not-a-cell).
Precision, recall, F1 overall and per UMI bin; 95% bootstrap CIs (barcodes resampled, B = 1000) and the paired CI of each caller's
F1 minus called_v2's.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CALLERS = ["called_v2", "edcr", "cellbender", "cellranger", "demuxlet_table", "knee"]
B = 1000


def prf(call, truth):
    tp = (call & truth).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = tp / call.sum(-1)
        r = tp / truth.sum(-1)
        f = 2 * p * r / (p + r)
    return p, r, f


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


def score_table(t, variant, seed=3, min_umi=50):
    """Overall metrics over decided barcodes with >= min_umi UMIs (the EmptyDrops test floor; below it no caller calls and the
    truth is mostly unknown); the per-bin table also shows the bins below."""
    col = "truth" if variant == "lenient" else "truth_strict"
    dall = t[t[col].isin(["cell", "empty", "damaged"])].reset_index(drop=True)
    d = dall[dall["umi_total"] >= min_umi].reset_index(drop=True)
    truth = (d[col] == "cell").to_numpy()
    callers = [c for c in CALLERS if c in d]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    res = {"n_decided": int(len(d)), "n_cell": int(truth.sum()), "overall": {}, "by_umi": {}}
    boot = {}
    for c in callers:
        a = d[c].to_numpy().astype(bool)
        p, r, f = prf(a, truth)
        bp, br, bf = prf(a[idx], truth[idx])
        boot[c] = bf
        res["overall"][c] = dict(n_called=int(a.sum()), precision=float(p), recall=float(r), f1=float(f),
                                 precision_ci=ci(bp), recall_ci=ci(br), f1_ci=ci(bf),
                                 tp=int((a & truth).sum()), fp=int((a & ~truth).sum()), fn=int((~a & truth).sum()))
    for c in callers:
        if c != "called_v2" and "called_v2" in boot:
            res["overall"][c]["f1_minus_called_v2_ci"] = ci(boot[c] - boot["called_v2"])
    bins = pd.cut(dall["umi_total"], C.UMI_BINS, right=False)
    for b, g in dall.groupby(bins, observed=True):
        tr = (g[col] == "cell").to_numpy()
        e = {"n": int(len(g)), "n_cell": int(tr.sum())}
        gi = rng.integers(0, len(g), size=(B, len(g)))
        for c in callers:
            a = g[c].to_numpy().astype(bool)
            p, r, f = prf(a, tr)
            bp, br, _ = prf(a[gi], tr[gi])
            e[c] = dict(n_called=int(a.sum()), precision=float(p), recall=float(r), f1=float(f), tp=int((a & tr).sum()),
                        fp=int((a & ~tr).sum()), fn=int((~a & tr).sum()), precision_ci=ci(bp), recall_ci=ci(br))
        res["by_umi"][str(b)] = e
    return res


def _em(f1, f0, iters=200):
    pi = 0.5
    for _ in range(iters):
        post = pi * f1 / (pi * f1 + (1 - pi) * f0)
        new = float(post.mean())
        if abs(new - pi) < 1e-7:
            break
        pi = new
    return pi, pi * f1 / (pi * f1 + (1 - pi) * f0)


def mixture_score(t, min_umi=50, seed=5, nb=300):
    """Cell fraction of each group of barcodes (caller-agreement pattern x UMI bin) by EM on the per-barcode densities f1 / f0
    (common.mixture_densities); posterior P(cell) per barcode; expected precision / recall / F1 of each caller with bootstrap CIs.
    'intact' variant: posterior times 1[not damaged]."""
    callers = [c for c in CALLERS if c in t and c != "knee"]
    callers_all = [c for c in CALLERS if c in t]
    u = t[(t["umi_total"] >= min_umi) & np.isfinite(t["f1"]) & np.isfinite(t["f0"])].reset_index(drop=True)
    pat = u[callers].astype(int).astype(str).agg("".join, axis=1)
    grp = (pat + "|" + pd.cut(u["umi_total"], C.UMI_BINS, right=False).astype(str)).to_numpy()
    f1, f0 = u["f1"].to_numpy(), u["f0"].to_numpy()
    nd = (u["intact_state"] != "damaged").to_numpy()
    calls = {c: u[c].to_numpy().astype(bool) for c in callers_all}

    def run(ix):
        post = np.zeros(len(ix))
        g = grp[ix]
        pis = {}
        for k in np.unique(g):
            m = g == k
            pis[k], post[m] = _em(f1[ix][m], f0[ix][m])
        out = {}
        for v, w in (("lenient", post), ("intact", post * nd[ix])):
            tot = w.sum()
            for c in callers_all:
                a = calls[c][ix]
                tp = w[a].sum()
                p, r = tp / max(a.sum(), 1), tp / max(tot, 1e-9)
                out[(v, c)] = (p, r, 2 * p * r / max(p + r, 1e-12))
        return out, pis, post
    full, pis, post = run(np.arange(len(u)))
    rng = np.random.default_rng(seed)
    boots = [run(rng.integers(0, len(u), len(u)))[0] for _ in range(nb)]
    res = {"n_universe": int(len(u)), "expected_cells": float(post.sum()), "expected_cells_intact": float((post * nd).sum()),
           "groups": {}, "lenient": {}, "intact": {}}
    for k, pi in pis.items():
        m = grp == k
        res["groups"][k] = {"n": int(m.sum()), "cell_fraction": float(pi), "median_umi": float(u.loc[m, "umi_total"].median()),
                            "damaged_frac": float(1 - nd[m].mean())}
    for (v, c), (p, r, f) in full.items():
        bp = np.array([b[(v, c)] for b in boots])
        e = {"n_called": int(calls[c].sum()), "precision": float(p), "recall": float(r), "f1": float(f),
             "precision_ci": ci(bp[:, 0]), "recall_ci": ci(bp[:, 1]), "f1_ci": ci(bp[:, 2])}
        if c != "called_v2":
            bv = np.array([b[(v, "called_v2")] for b in boots])
            e["f1_minus_called_v2_ci"] = ci(bp[:, 2] - bv[:, 2])
            e["precision_minus_called_v2_ci"] = ci(bp[:, 0] - bv[:, 0])
            e["recall_minus_called_v2_ci"] = ci(bp[:, 1] - bv[:, 1])
        res[v][c] = e
    # per-barcode posterior, for the extras analysis
    u = u.assign(post=post)
    return res, u[["barcode", "post"]]


def coverage(t):
    bins = pd.cut(t["umi_total"], C.UMI_BINS, right=False)
    out = {}
    for b, g in t.groupby(bins, observed=True):
        out[str(b)] = {k: int(v) for k, v in g["truth"].value_counts().items()}
        anyc = g[[c for c in CALLERS if c in g]].max(axis=1) == 1
        out[str(b)]["called_by_any"] = int(anyc.sum())
        out[str(b)]["called_by_any_unknown"] = int((anyc & (g["truth"] == "unknown")).sum())
    return out


def main(tdir, out_json, *dss):
    dss = dss or [d[: -len(".truth.tsv.gz")] for d in sorted(os.listdir(tdir)) if d.endswith(".truth.tsv.gz")]
    R = {}
    for ds in dss:
        t = pd.read_csv(os.path.join(tdir, f"{ds}.truth.tsv.gz"), sep="\t", dtype={"barcode": str})
        R[ds] = {"coverage": coverage(t), "lenient": score_table(t, "lenient"), "strict": score_table(t, "strict")}
        if "f1" in t:
            R[ds]["mixture"], pp = mixture_score(t)
            pp.to_csv(os.path.join(tdir, f"{ds}.posterior.tsv.gz"), sep="\t", index=False)
            mx = R[ds]["mixture"]
            print(f"== {ds} [mixture] universe {mx['n_universe']} expected cells {mx['expected_cells']:.0f} (intact {mx['expected_cells_intact']:.0f})")
            for v in ("lenient", "intact"):
                for c, m in mx[v].items():
                    print(f"  {v:8s} {c:15s} n={m['n_called']:6d} P={m['precision']:.4f} {np.round(m['precision_ci'], 4)} "
                          f"R={m['recall']:.4f} {np.round(m['recall_ci'], 4)} F1={m['f1']:.4f} dF1={np.round(m.get('f1_minus_called_v2_ci', [np.nan]), 4)}")
            for k, g in sorted(mx["groups"].items()):
                if g["n"] >= 20:
                    print(f"    group {k:40s} n={g['n']:6d} umi_med={g['median_umi']:8.0f} cell_frac={g['cell_fraction']:.3f} damaged={g['damaged_frac']:.2f}")
        for v in ("lenient", "strict"):
            o = R[ds][v]["overall"]
            print(f"== {ds} [{v}] decided {R[ds][v]['n_decided']} cells {R[ds][v]['n_cell']}")
            for c, m in o.items():
                d = m.get("f1_minus_called_v2_ci", "")
                print(f"  {c:15s} n={m['n_called']:6d} P={m['precision']:.4f} {m['precision_ci']} R={m['recall']:.4f} {m['recall_ci']} "
                      f"F1={m['f1']:.4f} fp={m['fp']} fn={m['fn']} dF1={d}")
    with open(out_json, "w") as fh:
        json.dump(R, fh, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
