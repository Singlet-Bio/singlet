"""Score a harness run on a barnyard library: implied contamination (UMIs removed over UMIs) against the cross-species truth, by depth.

    python -I score.py IN_DIR OUT_DIR [LABEL]      IN_DIR from dump_run.py, OUT_DIR from harness

Truth (as the gate's ambient-rna capability): a called cell with at most 5% UMIs of its minor species is pure; every other-species UMI in a
pure cell is ambient, so the true rho is the other-species fraction of the cell over the other species' share of the ambient pool
(barcodes with 1 to 100 UMIs, top 50,000 by count). Implied rho of the estimator = n_removed / UMIs of the cell.
Prints, over pure cells: mean implied against mean truth in six depth bins, the lowest-third bias (absolute log2 of the ratio of means,
pooled over species and per species, as in docs/beta-gap-table.md), the shallowest-sixth ratio per species and the Spearman correlation.
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

inp, out = sys.argv[1:3]
label = sys.argv[3] if len(sys.argv) > 3 else out
G2, B, nnz, nc = map(int, open(f"{inp}/dims.txt").read().split())
genes = open(f"{inp}/genes.txt").read().split("\n")[:-1]
G = len(genes)
mouse = np.array(["ENSMUSG" in g for g in genes])
rp = np.fromfile(f"{inp}/raw.indptr", dtype=np.int64)
ri = np.fromfile(f"{inp}/raw.indices", dtype=np.int32)
rd = np.fromfile(f"{inp}/raw.data", dtype=np.uint32)
cidx = np.fromfile(f"{inp}/cells.idx", dtype=np.uint32)
X = sp.csc_matrix((rd.astype(np.float64), ri, rp), shape=(G2, B))
X = (X[:G] + X[G:]).tocsc()  # exon + intron per gene
tot = np.asarray(X.sum(0)).ravel()
mo = np.asarray(X[mouse].sum(0)).ravel()
# ambient pool: 1..100 UMIs, top 50k by count (ties by position)
pool = np.where((tot >= 1) & (tot <= 100))[0]
pool = pool[np.argsort(-tot[pool], kind="stable")][:50000]
ms = mo[pool].sum() / tot[pool].sum()
cells = cidx.astype(int)
n = tot[cells]
mfrac = mo[cells] / np.maximum(n, 1)
pure_h, pure_m = mfrac <= 0.05, mfrac >= 0.95
truth = np.where(pure_h, mfrac / ms, np.where(pure_m, (1 - mfrac) / (1 - ms), np.nan))
rem = np.fromfile(f"{out}/removed.u32", dtype=np.uint32).astype(float)
rho = np.fromfile(f"{out}/rho.f64", dtype=np.float64)
imp = rem / np.maximum(n, 1)
d = pd.DataFrame({"n": n, "truth": truth, "imp": imp, "rho": rho, "sp": np.where(pure_h, "human", np.where(pure_m, "mouse", "mixed"))})
p = d[d.truth.notna()].copy()
p["bin"] = pd.qcut(p.n, 6, labels=False)
res = {"label": label, "mouse_share_pool": round(float(ms), 4), "n_cells": int(nc), "n_pure": int(len(p))}
rows = []
for b, g in p.groupby("bin"):
    rows.append({"bin": int(b), "median_umi": int(g.n.median()), "truth": round(g.truth.mean(), 4), "implied": round(g.imp.mean(), 4),
                 "ratio": round(g.imp.mean() / g.truth.mean(), 3), "human_ratio": round(g[g.sp == "human"].imp.mean() / g[g.sp == "human"].truth.mean(), 3),
                 "mouse_ratio": round(g[g.sp == "mouse"].imp.mean() / g[g.sp == "mouse"].truth.mean(), 3)})
res["bins"] = rows
lo = p[p.n <= p.n.quantile(1 / 3)]


def bias(g):
    return abs(np.log2(max(g.imp.mean(), 1e-9) / g.truth.mean()))


res["low_third_pooled"] = round(bias(lo), 4)
res["low_third_human"] = round(bias(lo[lo.sp == "human"]), 4)
res["low_third_mouse"] = round(bias(lo[lo.sp == "mouse"]), 4)
res["low_third_per_species_mean"] = round((res["low_third_human"] + res["low_third_mouse"]) / 2, 4)
res["overall_pooled"] = round(bias(p), 4)
res["spearman"] = round(float(spearmanr(p.imp, p.truth)[0]), 4)
print(label, "pure", len(p), "pool mouse share", round(ms, 3))
print(pd.DataFrame(rows).to_string(index=False))
print({k: v for k, v in res.items() if k.startswith("low_third") or k in ("overall_pooled", "spearman")})
json.dump(res, open(f"{out}/score.json", "w"), indent=1)
