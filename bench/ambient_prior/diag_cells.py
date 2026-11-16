"""Per-cell diagnosis of the ambient estimator on a barnyard run (D-163): which low-depth cells carry the excess implied contamination.

    python -I diag_cells.py IN_DIR OUT_DIR        IN_DIR from dump_run.py, OUT_DIR from harness (rho.f64, removed.u32)

Prints the lowest-third bias as score.py computes it, then the share of the excess (implied minus truth, summed over the lowest third)
carried by cells grouped by species, depth sixth and ratio of implied to truth.
"""
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

inp, out = sys.argv[1:3]
G2, B, nnz, nc = map(int, open(f"{inp}/dims.txt").read().split())
genes = open(f"{inp}/genes.txt").read().split("\n")[:-1]
G = len(genes)
mouse = np.array(["ENSMUSG" in g for g in genes])
rp = np.fromfile(f"{inp}/raw.indptr", dtype=np.int64)
ri = np.fromfile(f"{inp}/raw.indices", dtype=np.int32)
rd = np.fromfile(f"{inp}/raw.data", dtype=np.uint32)
cidx = np.fromfile(f"{inp}/cells.idx", dtype=np.uint32)
X = sp.csc_matrix((rd.astype(np.float64), ri, rp), shape=(G2, B))
X = (X[:G] + X[G:]).tocsc()
tot = np.asarray(X.sum(0)).ravel()
mo = np.asarray(X[mouse].sum(0)).ravel()
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
d = pd.DataFrame({"n": n, "truth": truth, "imp": rem / np.maximum(n, 1), "rho": rho, "sp": np.where(pure_h, "human", np.where(pure_m, "mouse", "mixed")),
                  "other": np.where(pure_h, mo[cells], tot[cells] - mo[cells]), "own": np.where(pure_h, tot[cells] - mo[cells], mo[cells])})
p = d[d.truth.notna()].copy()
low = p[p.n <= p.n.quantile(1 / 3)].copy()
print("pool mouse share", round(ms, 4), "pure", len(p), "low third", len(low))
print("low third mean truth", round(low.truth.mean(), 5), "mean implied", round(low.imp.mean(), 5), "ratio", round(low.imp.mean() / low.truth.mean(), 3))
low["excess"] = low.imp - low.truth
low["rat"] = pd.cut(low.imp / low.truth.clip(lower=1e-6), [0, 0.5, 1, 1.5, 2, 3, 5, 1e9])
print(low.groupby("rat", observed=True).agg(cells=("n", "size"), median_n=("n", "median"), truth=("truth", "mean"), imp=("imp", "mean"), excess_sum=("excess", "sum"),
                                              other_umis=("other", "mean")).round(4).to_string())
print(low.groupby("sp").agg(cells=("n", "size"), truth=("truth", "mean"), imp=("imp", "mean"), excess_sum=("excess", "sum"), other_umis=("other", "mean")).round(4).to_string())
print("top 15 excess cells")
print(low.sort_values("excess", ascending=False).head(15).round(4).to_string())
print("truth is zero (no other-species UMI): cells", int((low.other == 0).sum()), "implied there", round(low[low.other == 0].imp.mean(), 5))
