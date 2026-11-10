# SPDX-License-Identifier: MIT
"""AUC per feature (ambient vs native reads of the same gene catalogue), raw and gene-adjusted, within depth strata of the host cell; combined stacker AUC (5-fold, grouped by cell).
  python soup_analyze.py READS.tsv.gz OUT.json"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
D = pd.read_csv(sys.argv[1], sep="\t"); out = sys.argv[2]
FEATS = ["alen", "clip_l", "clip_r", "nM", "spliced", "exonic", "antisense", "log_d3", "frac_d3", "A_after_end", "A_before_start", "mean_q", "reads_in_molecule"]
D["log_dup"] = np.log(D.reads_in_molecule); FEATS = [f if f != "reads_in_molecule" else "log_dup" for f in FEATS]
# the ambient reads of species X sit in cells of species Y; native reads of X in cells of X: compare within the gene's species
D["gene_species"] = np.where(D.gene.str.startswith("ENSMUSG"), "mouse", "human")
nat = D[D.ambient == 0]
med = nat.groupby("gene")[FEATS].median()
for f in FEATS: D[f + "_adj"] = D[f] - D.gene.map(med[f]).fillna(nat[f].median())
def auc(x, y):
    m = np.isfinite(x); x, y = x[m], y[m]
    if y.sum() < 50 or (1 - y).sum() < 50 or len(np.unique(x)) < 2: return np.nan
    return float(roc_auc_score(y, x))
res = {"n_ambient": int(D.ambient.sum()), "n_native": int((1 - D.ambient).sum()), "features": {}}
for f in FEATS:
    r = {}
    for adj in ("", "_adj"):
        per = []; wts = []
        for (gs, q), g in D.groupby(["gene_species", "depth_q"]):
            a = auc(g[f + adj].values.astype(float), g.ambient.values)
            if np.isfinite(a): per.append(a); wts.append(int(g.ambient.sum()))
        r["auc" + (adj or "_raw")] = float(np.average(per, weights=wts)) if per else None
    res["features"][f] = r
    print(f, r, flush=True)
# combined (gene-adjusted features + raw), grouped CV
X = D[[f + "_adj" for f in FEATS] + FEATS].fillna(0).values; y = D.ambient.values; groups = D.cb.values
for name, mk in (("logistic", lambda: LogisticRegression(max_iter=300, C=1.0)), ("trees", lambda: HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_depth=4))):
    p = np.zeros(len(D))
    for tr, te in GroupKFold(5).split(X, y, groups):
        Xt = X[tr]; mu = Xt.mean(0); sd = Xt.std(0) + 1e-9
        m = mk().fit((Xt - mu) / sd, y[tr]); p[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    strata = []
    for (gs, q), g in D.groupby(["gene_species", "depth_q"]):
        idx = g.index.values
        a = auc(p[idx], y[idx])
        if np.isfinite(a): strata.append((a, int(y[idx].sum())))
    res["combined_" + name] = {"auc_overall": float(roc_auc_score(y, p)), "auc_depth_stratified": float(np.average([a for a, w in strata], weights=[w for a, w in strata]))}
    print(name, res["combined_" + name], flush=True)
json.dump(res, open(out, "w"), indent=1)
