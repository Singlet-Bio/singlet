"""Noise ceilings per library: (1) reliability-based Pearson ceiling, (2) simulated binomial cross-species counts around a smooth
'true' rho (depth-binned truth mean; and the production estimate), correlated with the noisy truth.  Usage: ceiling.py LIB [LIB ...]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402

rows = []
for lib in sys.argv[1:]:
    ds = atlib.load(lib)
    r = dict(lib=lib, n_cells=len(ds.N), n_pure=int(ds.pure.sum()), alpha_m=ds.alpha_m, mean_truth=float(np.mean(ds.rho_truth[ds.pure])))
    r.update(atlib.reliability_ceiling(ds))
    # smooth truth 1: depth-binned mean of truth (20 quantile bins, per species)
    sm = np.full(len(ds.N), np.nan)
    for sp_ in ("human", "mouse"):
        idx = np.where(ds.species == sp_)[0]
        if len(idx) < 40:
            continue
        b = pd.qcut(pd.Series(ds.N[idx]).rank(method="first"), min(20, len(idx) // 20), labels=False).values
        for k in np.unique(b):
            sm[idx[b == k]] = ds.rho_truth[idx[b == k]].mean()
    r["sim_spearman_depthbin"], r["sim_pearson_depthbin"] = atlib.sim_ceiling(ds, sm)
    p = f"{atlib.OUT}/{lib}/rho_full-full_production_t1.npy"
    if os.path.exists(p):
        rho = np.load(p)
        r["sim_spearman_prod"], r["sim_pearson_prod"] = atlib.sim_ceiling(ds, rho)
    rows.append(r)
    print(r, flush=True)
os.makedirs(f"{atlib.OUT}/summary", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{atlib.OUT}/summary/ceiling.csv", index=False)
