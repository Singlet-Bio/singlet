# SPDX-License-Identifier: MIT
"""hgmm1k_v3: the pin's reported rho and its realised removal against species-mixing truth (pure cells: minor-species UMI / pool share of that species).
  python ambient_hgmm_eval.py RUN_DIR CELLS_ARTIFACT POOL_JSON OUT.json"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
run, cells, pool, out = sys.argv[1:5]
C = pd.read_csv(cells, sep="\t"); P = json.load(open(pool))["pools"]["1_100_top50k"]; ms = P["mouse_share"]
C["frac_m"] = C.mouse_umi / C.n_umi
pure_h = C.frac_m < 0.05; pure_m = C.frac_m > 0.95
C = C[pure_h | pure_m].copy()
C["truth"] = np.where(C.frac_m < 0.05, C.frac_m / ms, (1 - C.frac_m) / (1 - ms))
A = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t"); A["barcode"] = A.barcode.astype(str).str.replace("-1$", "", regex=True)
C["barcode"] = C.barcode.astype(str).str.replace("-1$", "", regex=True)
D = C.merge(A, on="barcode", suffixes=("", "_s"))
D["removed_frac"] = D.n_ambient_umi_removed / D.n_umi_s
r = {"n_pure": int(len(D)), "truth_mean": float(D.truth.mean())}
for k in ("rho", "rho_total", "removed_frac"):
    r[k] = {"mean": float(D[k].mean()), "ratio": float(D[k].mean() / D.truth.mean()), "abs_log2": float(abs(np.log2(D[k].mean() / D.truth.mean()))), "spearman": float(spearmanr(D[k], D.truth)[0])}
json.dump(r, open(out, "w"), indent=1); print(json.dumps(r, indent=1))
