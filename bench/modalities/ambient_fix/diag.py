"""Per-cluster diagnostics of a harness run: estimated vs true rho by cluster/species. Usage: python -I diag.py TAG"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
from common import A, WORK  # noqa: E402

tag = sys.argv[1]
d = f"{A}/fix_out/{tag}"
meta = pd.read_csv(f"{WORK}/cells.csv")
meta["rho"] = np.fromfile(f"{d}/rho.f64", dtype=np.float64)
meta["cl"] = np.fromfile(f"{d}/cluster.u32", dtype=np.uint32)
meta["removed"] = np.fromfile(f"{d}/removed.u32", dtype=np.uint32)
pure = meta.species.isin(["human", "mouse"])
g = meta[pure].groupby(["cl", "species"]).agg(n=("rho", "size"), est=("rho", "mean"), truth=("rho_truth", "mean"),
                                              N=("N", "mean"), rem=("removed", "mean"))
g["ratio"] = g.est / g.truth
print(g.round(4).to_string())
for sp in ("human", "mouse"):
    m = meta[meta.species == sp]
    print(sp, "spearman", round(spearmanr(m.rho, m.rho_truth)[0], 3), "mean est", round(m.rho.mean(), 4), "truth", round(m.rho_truth.mean(), 4))
print(meta[~pure][["cl", "species", "rho", "N"]])
p = meta[pure].copy()
p["err"] = p.rho - p.rho_truth
print("worst over-estimates:")
print(p.sort_values("err", ascending=False)[["cl", "species", "N", "rho", "rho_truth", "H", "M"]].head(12).round(4).to_string())
print("worst under-estimates:")
print(p.sort_values("err")[["cl", "species", "N", "rho", "rho_truth", "H", "M"]].head(6).round(4).to_string())
print("est rho quantiles:", p.rho.quantile([.01, .1, .5, .9, .99, 1]).round(4).tolist(), " truth:", p.rho_truth.quantile([.01, .1, .5, .9, .99, 1]).round(4).tolist())
from scipy.stats import pearsonr
q = p[p.rho < 0.05]
print("pearson all", round(pearsonr(p.rho, p.rho_truth)[0], 3), " without rho_est>=0.05:", round(pearsonr(q.rho, q.rho_truth)[0], 3), "n dropped", len(p) - len(q))
