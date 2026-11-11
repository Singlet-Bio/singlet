# SPDX-License-Identifier: MIT
"""Gate scorer for cell calling (bench/calls-sota/gates/cell_calling.md).

    python -P score_tables.py OUT.json NAME=TABLE.tsv[.gz]:CLASS [...]

TABLE has columns barcode, umi_total, truth (cell / empty / unknown / damaged), optionally truth_strict, and 0/1 caller columns among
called_v2 (Singlet), edcr (EmptyDrops), cellbender. CLASS is G, H or S (ground truth; T is not scored here).
Per library: precision, recall, F1 of each caller over decided barcodes with >= 50 UMIs, and recall at a matched number of called
cells (N = truth cells; each tool contributes its called barcodes first, then total UMIs), with a paired bootstrap 95% interval
(B = 1000) of Singlet minus each tool.
"""
import json
import sys

import numpy as np
import pandas as pd

TOOLS = [("singlet", "called_v2"), ("emptydrops", "edcr"), ("cellbender", "cellbender")]
B = 1000


def prf(a, t):
    tp = int((a & t).sum())
    p = tp / max(int(a.sum()), 1)
    r = tp / max(int(t.sum()), 1)
    return p, r, (2 * p * r / (p + r) if p + r > 0 else 0.0)


def topn(called, tot, n):
    o = np.lexsort((-tot, ~called))
    s = np.zeros(len(called), bool)
    s[o[:n]] = True
    return s


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def score(df, col, rng):
    d = df[df[col].isin(["cell", "empty", "damaged"]) & (df["umi_total"] >= 50)].reset_index(drop=True)
    t = (d[col] == "cell").to_numpy()
    tot = d["umi_total"].to_numpy().astype(float)
    calls = {n: d[c].fillna(0).astype(int).to_numpy().astype(bool) for n, c in TOOLS if c in d}
    N = int(t.sum())
    sel = {n: topn(v, tot, N) for n, v in calls.items()}
    out = {"n_decided": int(len(d)), "n_truth_cells": N, "tools": {}, "matched": {}}
    idx = rng.integers(0, len(d), size=(B, len(d)))
    for n, a in calls.items():
        p, r, f = prf(a, t)
        out["tools"][n] = {"n_called": int(a.sum()), "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)}
        out["matched"][n] = {"recall_at_N": round(float((sel[n] & t).sum() / max(N, 1)), 4)}
    for n in calls:
        if n == "singlet":
            continue
        bf = np.array([prf(calls["singlet"][i], t[i])[2] - prf(calls[n][i], t[i])[2] for i in idx[:200]])
        bm = np.array([((sel["singlet"][i] & t[i]).sum() - (sel[n][i] & t[i]).sum()) / max(t[i].sum(), 1) for i in idx])
        out["tools"][n]["f1_singlet_minus_ci"] = ci(bf)
        out["matched"][n]["recall_singlet_minus"] = round(out["matched"]["singlet"]["recall_at_N"] - out["matched"][n]["recall_at_N"], 4)
        out["matched"][n]["recall_singlet_minus_ci"] = ci(bm)
    return out


def main(out_json, *specs):
    rng = np.random.default_rng(11)
    R = {}
    for s in specs:
        nm, rest = s.split("=", 1)
        path, cls = rest.rsplit(":", 1)
        df = pd.read_csv(path, sep="\t", dtype={"barcode": str})
        R[nm] = {"class": cls, "lenient": score(df, "truth", rng)}
        if "truth_strict" in df:
            R[nm]["strict"] = score(df, "truth_strict", rng)
        for v in ("lenient", "strict"):
            if v not in R[nm]:
                continue
            r = R[nm][v]
            print(f"== {nm} [{cls}/{v}] decided {r['n_decided']} truth cells {r['n_truth_cells']}")
            for n, m in r["tools"].items():
                mm = r["matched"][n]
                print(f"  {n:11s} called={m['n_called']:6d} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}  recall@N={mm['recall_at_N']:.4f}"
                      + (f"  dRecall(singlet-tool)={mm['recall_singlet_minus']:+.4f} {mm['recall_singlet_minus_ci']}" if "recall_singlet_minus" in mm else ""))
    json.dump(R, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
