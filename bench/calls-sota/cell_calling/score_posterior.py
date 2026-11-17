# SPDX-License-Identifier: MIT
"""Gate scorer for cell calling on truth tables where most barcodes are unknown (Kang genotype lanes, species mixtures).

    python -P score_posterior.py TRUTH_DIR OUT.json NAME:CLASS [...]

Uses the per-barcode posterior P(cell) that jobs/cellcall_truth/score.py computed from the genotype or species evidence by EM within
caller-agreement x UMI-bin groups (TRUTH_DIR/NAME.posterior.tsv.gz, universe: barcodes with >= 50 UMIs and finite densities). Expected
precision, recall and F1 of each caller = posterior mass of its calls over its calls / over all expected cells. Matched count: N =
round(expected cells); each tool contributes its called barcodes first, then total UMIs; expected recall = posterior mass of the
selected barcodes over all expected cells. Paired bootstrap (B = 300) of Singlet minus each tool.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

TOOLS = [("singlet", "called_v2"), ("emptydrops", "edcr"), ("cellbender", "cellbender")]
B = 300


def topn(called, tot, n):
    o = np.lexsort((-tot, ~called))
    s = np.zeros(len(called), bool)
    s[o[:n]] = True
    return s


def stats(w, a, sel):
    tot = max(w.sum(), 1e-9)
    tp = w[a].sum()
    p, r = tp / max(a.sum(), 1), tp / tot
    return p, r, 2 * p * r / max(p + r, 1e-12), w[sel].sum() / tot


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def main(tdir, out_json, *specs):
    rng = np.random.default_rng(13)
    R = {}
    for s in specs:
        nm, cls = s.split(":")
        t = pd.read_csv(f"{tdir}/{nm}.truth.tsv.gz", sep="\t", dtype={"barcode": str}, low_memory=False)
        if os.path.exists(f"{tdir}/{nm}.posterior.tsv.gz"):
            p = pd.read_csv(f"{tdir}/{nm}.posterior.tsv.gz", sep="\t", dtype={"barcode": str})
        else:  # recompute with the registered EM (jobs/cellcall_truth/score.py)
            sys.path.insert(0, os.environ.get("CCT", "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth"))
            import score as SC
            _, p = SC.mixture_score(t, nb=2)
        u = t.merge(p, on="barcode", how="inner")
        u = u[u["umi_total"] >= 50].reset_index(drop=True)
        w = u["post"].to_numpy()
        tot = u["umi_total"].to_numpy().astype(float)
        N = int(round(w.sum()))
        calls = {n: u[c].fillna(0).astype(int).to_numpy().astype(bool) for n, c in TOOLS if c in u}
        sel = {n: topn(a, tot, N) for n, a in calls.items()}
        res = {"class": cls, "universe": int(len(u)), "expected_cells": round(float(w.sum()), 1), "tools": {}}
        for n, a in calls.items():
            pr, rc, f1, rN = stats(w, a, sel[n])
            res["tools"][n] = {"n_called": int(a.sum()), "precision": round(pr, 4), "recall": round(rc, 4), "f1": round(f1, 4), "recall_at_N": round(rN, 4)}
        idx = rng.integers(0, len(u), size=(B, len(u)))
        for n in calls:
            if n == "singlet":
                continue
            d1, d2 = [], []
            for i in idx:
                a = stats(w[i], calls["singlet"][i], sel["singlet"][i]); b = stats(w[i], calls[n][i], sel[n][i])
                d1.append(a[2] - b[2]); d2.append(a[3] - b[3])
            res["tools"][n]["f1_singlet_minus"] = round(res["tools"]["singlet"]["f1"] - res["tools"][n]["f1"], 4)
            res["tools"][n]["f1_singlet_minus_ci"] = ci(d1)
            res["tools"][n]["recall_at_N_singlet_minus"] = round(res["tools"]["singlet"]["recall_at_N"] - res["tools"][n]["recall_at_N"], 4)
            res["tools"][n]["recall_at_N_singlet_minus_ci"] = ci(d2)
        R[nm] = res
        print(f"== {nm} [{cls}] universe {len(u)} expected cells {w.sum():.0f}")
        for n, m in res["tools"].items():
            print(f"  {n:11s} called={m['n_called']:6d} P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} recall@N={m['recall_at_N']:.4f}"
                  + (f"  dF1={m['f1_singlet_minus']:+.4f} {m['f1_singlet_minus_ci']} dRec@N={m['recall_at_N_singlet_minus']:+.4f} {m['recall_at_N_singlet_minus_ci']}" if "f1_singlet_minus" in m else ""))
    json.dump(R, open(out_json, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
