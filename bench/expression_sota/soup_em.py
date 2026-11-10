# SPDX-License-Identifier: MIT
"""Per-read ambient likelihood ratio trained on the hgmm cross-species labels only; per-cell EM for the ambient fraction; scored on Kang genotype truth.
  python soup_em.py HGMM_READS.tsv.gz KANG_READS.tsv.gz KANG_TRUTH.tsv.gz OUT.json
Training: hgmm_soup.reads.tsv.gz (ambient = species-X reads in species-Y pure cells; native reads subsampled 1/16 -> weights restore the prevalence). Features: alen, clip_l, clip_r,
nM, spliced, antisense, log_d3, frac_d3, log(reads in molecule) and each minus its per-gene median (native reads of the sample). A HistGradientBoosting classifier gives p; the
likelihood ratio is odds(p) / odds(prevalence in the training set). Nothing is fitted or tuned on Kang. Per cell: rho <- mean_i rho*LR_i / (rho*LR_i + 1 - rho), 50 EM iterations from 0.01."""
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
hg, kg, tr, out = sys.argv[1:5]
FE = ["alen", "clip_l", "clip_r", "nM", "spliced", "antisense", "log_d3", "frac_d3", "log_dup"]
def prep(D, native_mask):
    D = D.copy(); D["log_dup"] = np.log(D.reads_in_molecule)
    med = D[native_mask].groupby("gene")[FE].median()
    for f in FE: D[f + "_adj"] = D[f] - D.gene.map(med[f]).fillna(D[native_mask][f].median())
    return D
H = pd.read_csv(hg, sep="\t"); H = prep(H, (H.ambient == 0).values)
X = H[FE + [f + "_adj" for f in FE]].values; y = H.ambient.values
w = np.where(y == 1, 1.0, 16.0)   # native reads were subsampled 1/16
prev = (y * 1.0).sum() / w.sum()
clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=4, random_state=0).fit(X, y, sample_weight=w)
print("trained on", len(H), "reads; prevalence", prev, flush=True)
K = pd.read_csv(kg, sep="\t"); K = prep(K, np.ones(len(K), bool))   # Kang reads are ~98% native; per-gene medians over all of them
p = clf.predict_proba(K[FE + [f + "_adj" for f in FE]].values)[:, 1].clip(1e-6, 1 - 1e-6)
K["lr"] = (p / (1 - p)) / (prev / (1 - prev))
T = pd.read_csv(tr, sep="\t"); T["cb"] = T.barcode.map(lambda b: b.split("-")[0]); T = T.set_index("cb")
res = []
for cb, g in K.groupby("cb"):
    lr = g.lr.values; rho = 0.01
    for _ in range(50): rho = float(np.mean(rho * lr / (rho * lr + 1 - rho)))
    res.append((cb, len(g), rho, float(np.mean(lr)), float(np.mean(np.log(lr)))))
R = pd.DataFrame(res, columns=["cb", "n_reads", "rho_em", "mean_lr", "mean_loglr"]).set_index("cb").join(T[["rho_truth", "weight", "singlet_rho", "n_umi", "singlet_after", "celltype"]])
R["singlet_removed"] = 1 - R.singlet_after / R.n_umi
def sp(a, b, w=None):
    m = np.isfinite(a) & np.isfinite(b); return float(spearmanr(a[m], b[m])[0])
o = {"n_cells": int(len(R)), "median_reads": float(R.n_reads.median()), "truth_mean": float(np.average(R.rho_truth, weights=R.weight)),
     "spearman_vs_truth": {k: sp(R[k].values, R.rho_truth.values) for k in ("rho_em", "mean_lr", "mean_loglr", "singlet_rho", "singlet_removed")}}
# cell-type adjusted: remove the per-cell-type mean (the truth is dominated by cell type); within-type Spearman averaged
wt = {}
for k in ("rho_em", "mean_loglr", "singlet_rho", "singlet_removed"):
    s = []; ws = []
    for ct, g in R.groupby("celltype"):
        if len(g) >= 30: s.append(sp(g[k].values, g.rho_truth.values)); ws.append(len(g))
    wt[k] = float(np.average(s, weights=ws))
o["within_celltype_spearman"] = wt
o["mean_em_rho"] = float(np.average(R.rho_em, weights=R.weight)); o["ratio_em_to_truth"] = o["mean_em_rho"] / o["truth_mean"]
json.dump(o, open(out, "w"), indent=1); print(json.dumps(o, indent=1))
