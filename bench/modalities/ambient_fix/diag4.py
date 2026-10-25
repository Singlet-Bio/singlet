"""Per-cell estimate vs truth by depth tertile inside the large human/mouse clusters. Usage: python -I diag4.py TAG"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
from common import A, WORK  # noqa: E402

tag = sys.argv[1]
d = f"{A}/fix_out/{tag}"
meta = pd.read_csv(f"{WORK}/cells.csv")
meta["rho"] = np.fromfile(f"{d}/rho.f64", dtype=np.float64)
meta["cl"] = np.fromfile(f"{d}/cluster.u32", dtype=np.uint32)
meta["removed"] = np.fromfile(f"{d}/removed.u32", dtype=np.uint32)
meta["removed_frac"] = meta.removed / meta.N
meta["cross"] = np.where(meta.species == "human", meta.M, meta.H)
for k in (0, 1):
    m = meta[meta.cl == k].copy()
    m["tert"] = pd.qcut(m.N, 3, labels=["low", "mid", "high"])
    g = m.groupby("tert", observed=True).agg(n=("rho", "size"), N=("N", "mean"), truth=("rho_truth", "mean"), est=("rho", "mean"),
                                           removed_frac=("removed_frac", "mean"), cross=("cross", "mean"))
    print("cluster", k, m.species.iloc[0])
    print(g.round(5).to_string())
    print("   cell-level: corr(est, truth)=", round(np.corrcoef(m.rho, m.rho_truth)[0, 1], 3),
          " slope est~truth:", round(np.polyfit(m.rho_truth, m.rho, 1)[0], 3), " intercept:", round(np.polyfit(m.rho_truth, m.rho, 1)[1], 5))
    print(m[["N", "cross", "rho_truth", "rho", "removed"]].head(12).round(5).to_string())
