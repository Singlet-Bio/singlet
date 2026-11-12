# SPDX-License-Identifier: MIT
"""Score CellBender (0.3.2, cb_run.sbatch) on the held-out libraries and the Kang b1 lanes with the gate metrics; writes cb_scores.json.
python -P cb_score.py OUT.json"""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/cellcall_truth")
S = "/mnt/projects/debruinz_project/debruinz/singlet/bench/calls-sota/cell_calling"; TD = "/mnt/projects/debruinz_project/debruinz/singlet/state/cellcall_truth/truth"
def cbset(n):
    f = f"{S}/cellbender/{n}/cb_cell_barcodes.csv"
    return {l.strip().split(",")[0].split("-")[0] for l in open(f) if l.strip() and not l.startswith("barcode")} if os.path.exists(f) else None
R = {}
for n, tab, cls in (("cellplex10k", "heldout_cellplex10k", "H"), ("nuclei_poc", "heldout_nuc", "H-recall")):
    cb = cbset(n)
    if cb is None: R[n] = "no CellBender output"; continue
    t = pd.read_csv(f"{S}/{tab}/table.tsv.gz", sep="\t", dtype={"barcode": str}); t["cellbender"] = t.barcode.str.split("-").str[0].isin(cb).astype(int)
    if cls == "H":
        d = t[t.truth.isin(["cell", "empty"]) & (t.umi_total >= 50)]; T = (d.truth == "cell").to_numpy()
    else:
        d = t; T = (d.truth == "cell").to_numpy()
    tot = d.umi_total.to_numpy().astype(float); N = int(T.sum()); e = {"n_truth_cells": N}
    for k in ("called_v2", "edcr", "cellbender"):
        a = d[k].to_numpy().astype(bool); tp = (a & T).sum(); P = tp / max(a.sum(), 1); Rr = tp / N
        o = np.lexsort((-tot, ~a)); s = np.zeros(len(a), bool); s[o[:N]] = True
        e[k] = dict(n_called_all=int(t[k].sum()), precision=round(float(P), 4), recall=round(float(Rr), 4), f1=round(float(2 * P * Rr / max(P + Rr, 1e-12)), 4), recall_at_N=round(float((s & T).sum() / N), 4))
    R[n] = e; print(n, json.dumps(e))
import score as SC
for ds in ("kang_b1A", "kang_b1B", "kang_b1C"):
    cb = cbset(ds)
    if cb is None: R[ds] = "no CellBender output"; continue
    t = pd.read_csv(f"{TD}/{ds}.truth.tsv.gz", sep="\t", dtype={"barcode": str}, low_memory=False); t["barcode"] = t.barcode.str.split("-").str[0]
    _, p = SC.mixture_score(t, nb=2); p["barcode"] = p.barcode.str.split("-").str[0]; u = t.merge(p, on="barcode")
    u["cellbender"] = u.barcode.isin(cb).astype(int); w = u.post.to_numpy(); e = {}
    for k in ("called_v2", "edcr", "cellbender"):
        a = u[k].to_numpy().astype(bool); tp = w[a].sum(); P = tp / max(a.sum(), 1); Rr = tp / w.sum(); e[k] = dict(n=int(a.sum()), precision=round(float(P), 4), recall=round(float(Rr), 4), f1=round(float(2 * P * Rr / (P + Rr)), 4))
    R[ds] = e; print(ds, json.dumps(e))
json.dump(R, open(sys.argv[1], "w"), indent=1)
