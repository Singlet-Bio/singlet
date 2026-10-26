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
