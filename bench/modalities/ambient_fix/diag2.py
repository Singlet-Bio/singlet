"""Ratio n_g / (N a_g) of off-species genes per cluster (is the ratio flat across genes, or over-dispersed / depth dependent?).
Usage: python -I diag2.py TAG"""
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
truth = meta.rho_truth.values
for k in np.unique(cl):
    m = cl == k
    sp = meta.species[m].mode()[0]
    off = gsp if sp == "human" else ~gsp
    n = np.asarray(X[:, m].sum(1)).ravel()
    N = n.sum()
    e1 = N * a
    sel = off & (e1 >= 20)
    r = n[sel] / e1[sel]
    w = e1[sel]
    o = np.argsort(r)
    cw = np.cumsum(w[o]) / w.sum()
    q = lambda p: r[o][np.searchsorted(cw, p)]
    other_share = a[off].sum()
    print(f"cluster {k} {sp} cells={m.sum()} N={N:.0f} off-genes(e1>=20)={sel.sum()} off-mass={other_share:.3f}")
    print(f"   weighted ratio quantiles 10/25/50/75/90: {q(.1):.4f} {q(.25):.4f} {q(.5):.4f} {q(.75):.4f} {q(.9):.4f};  "
          f"mass-weighted mean {n[off].sum() / e1[off].sum():.4f}; truth-mean(UMI-weighted) {np.average(truth[m], weights=meta.N[m]):.4f}")
    # Poisson expectation for sd of ratio at the median
    print(f"   median e1 of selected genes {np.median(e1[sel]):.1f}; poisson relative sd at median ~ {1 / np.sqrt(np.median(e1[sel]) * q(.5)):.2f}")
