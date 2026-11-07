"""The trimmed-MLE map f(c) for each cluster (what cluster_contamination() iterates). Usage: python -I diag3.py TAG"""
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
from common import A, WORK  # noqa: E402

tag = sys.argv[1]
d = f"{A}/fix_out/{tag}"
meta = pd.read_csv(f"{WORK}/cells.csv")
genes = pd.read_csv(f"{WORK}/genes.csv")
gsp = (genes.species == "mouse").values
cl = np.fromfile(f"{d}/cluster.u32", dtype=np.uint32)
a = np.fromfile(f"{d}/ambient.f64", dtype=np.float64)
X = scipy.io.mmread(f"{WORK}/cells.mtx").tocsc()
z = 3.0
for k in np.unique(cl):
    m = cl == k
    sp = meta.species[m].mode()[0]
    off = gsp if sp == "human" else ~gsp
    n = np.asarray(X[:, m].sum(1)).ravel()
    N = n.sum()
    e1 = N * a
    cstar = n[off].sum() / e1[off].sum()
    print(f"cluster {k} {sp} N={N:.0f} cstar(off-species ratio)={cstar:.5f}")
    for c in [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.015, 0.012, 0.01, 0.008, 0.006, 0.005, 0.004]:
        thr = c * e1
        inc = n <= thr + z * np.sqrt(thr) + 1.0
        f = n[inc].sum() / e1[inc].sum()
        print(f"   c={c:.4f} -> f={f:.5f}  included off-mass frac={e1[inc & off].sum() / e1[off].sum():.3f} n_on_included_umis={n[inc & ~off].sum():.0f} off_umis_incl/total={n[inc & off].sum() / n[off].sum():.3f}")
