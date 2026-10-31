# SPDX-License-Identifier: MIT
"""Ambient estimate vs genotype truth on a Kang lane: Singlet (pin), SoupX, DecontX (default / background), CellBender.
Reads the per-cell artifact written by bench/public kang_ambient_artifacts.py. Implied contamination = 1 - UMI after / UMI before.
Writes JSON: weighted mean (weights = truth informative mass), ratio to truth, |log2 ratio|, Spearman and Pearson per cell, by cell type ratio."""
import json, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
tsv, out = sys.argv[1:3]
D = pd.read_csv(tsv, sep="\t")
w = D.weight.to_numpy(); truth = np.average(D.rho_truth, weights=w)
res = {"n_cells": int(len(D)), "truth_mean": float(truth)}
est = {"singlet_rho": D.singlet_rho}
for t in ("singlet", "soupx", "decontx", "decontx_bg", "cellbender", "cellbender_rec"):
    if f"{t}_after" in D:
        est[f"{t}_implied"] = 1 - D[f"{t}_after"] / D.n_umi
for k, v in est.items():
    v = v.to_numpy(float); m = np.isfinite(v)
    mean = float(np.average(v[m], weights=w[m]))
    r = {"mean": mean, "ratio": mean / truth, "abs_log2_ratio": float(abs(np.log2(max(mean, 1e-9) / truth))),
         "spearman": float(spearmanr(v[m], D.rho_truth[m])[0]), "pearson": float(pearsonr(v[m], D.rho_truth[m])[0]), "n": int(m.sum())}
    byct = {}
    for ct, g in D[m].groupby("celltype"):
        if len(g) >= 30 and g.weight.sum() > 0:
            byct[str(ct)] = float(np.average(v[m][(D[m].celltype == ct).to_numpy()], weights=g.weight) / np.average(g.rho_truth, weights=g.weight))
    r["ratio_by_celltype"] = byct
    res[k] = r
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: (round(x["ratio"], 3), round(x["spearman"], 3)) for k, x in res.items() if isinstance(x, dict)}), "truth", round(truth, 4))
