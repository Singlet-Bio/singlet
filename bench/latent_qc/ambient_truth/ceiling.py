"""Analysis of TRUTH noise per library (no simulated data): (1) reliability-based Pearson ceiling from the observed variance of the
truth minus its Poisson noise variance; (2) split-half agreement of the truth itself (each cell's real UMIs split at random in two
halves, rho_true from each half, Spearman/Pearson between halves, Spearman-Brown to full depth). Usage: ceiling.py LIB [LIB ...]"""
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
    r.update(atlib.split_half_truth(ds))
    rows.append(r)
    print(r, flush=True)
os.makedirs(f"{atlib.OUT}/summary", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{atlib.OUT}/summary/ceiling.csv", index=False)
