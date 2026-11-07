# SPDX-License-Identifier: MIT
"""Cell-calling diagnostics on development libraries: recall at a matched number of called cells, and what the missed truth cells are.

  matched_count.py RG4 RUNS OUT.json     RG4: release-gate4 dir (gate2/artifacts/cell-calling/<ds>.barcodes.tsv.gz tables);
                                         RUNS: directory of Singlet runs (<ds>/cell_calls.tsv)

Per library: N = number of truth cells. Each tool contributes its top N barcodes: its called barcodes first (ordered by total UMIs, the
only score the tables carry for the competitors), then the rest by total UMIs. Singlet also gets its own ordering (called, then the EmptyDrops
FDR, then total UMIs). Recall at N is reported with a paired bootstrap 95% interval of Singlet minus each tool (barcodes resampled).
Also: UMI totals of the truth cells Singlet misses, against the smallest called total, and which other tools call them.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

rg4, runs, out = sys.argv[1:4]
rng = np.random.default_rng(7)
res = {}
for f in sorted(glob.glob(f"{rg4}/gate2/artifacts/cell-calling/*.barcodes.tsv.gz")):
    ds = os.path.basename(f).split(".")[0]
    cc = f"{runs}/{ds}/cell_calls.tsv"
    if not os.path.exists(cc):
        continue
    a = pd.read_csv(f, sep="\t")
    a["barcode"] = a.barcode.astype(str).str.replace("-1$", "", regex=True)
    truthcol = "cellranger" if "cellranger" in a else "demuxlet"
    c = pd.read_csv(cc, sep="\t")
    c["barcode"] = c.barcode.astype(str).str.replace("-1$", "", regex=True)
    c = c.set_index("barcode")
    a = a.set_index("barcode")
    u = a.index.union(c.index)
    a = a.reindex(u).fillna(0)
    c = c.reindex(u)
    tot = np.where(a.umi_total > 0, a.umi_total, c.umi_total.fillna(0)).astype(float)
    T = a[truthcol].values.astype(bool)
    N = int(T.sum())
    sing = c["called_v2"].fillna(0).astype(str).isin(["1", "True", "TRUE"]).values
    fdr = c["fdr"].astype(float).fillna(1.0).values
    calls = {"singlet": sing}
    for t in ("starsolo_edcr", "cellbender"):
        if t in a:
            calls[t] = a[t].values.astype(bool)

    def topn(called, key2):
        o = np.lexsort((-key2[0], -key2[1], ~called)) if len(key2) == 2 else np.lexsort((-key2[0], ~called))
        sel = np.zeros(len(called), bool)
        sel[o[:N]] = True
        return sel

    sel = {"singlet": topn(sing, (tot,))}
    sel["singlet_fdr_order"] = np.zeros(len(sing), bool)
    o = np.lexsort((-tot, fdr, ~sing))
    sel["singlet_fdr_order"][o[:N]] = True
    for t, v in calls.items():
        if t != "singlet":
            sel[t] = topn(v, (tot,))
    sel["umi_total"] = np.zeros(len(sing), bool)
    sel["umi_total"][np.argsort(-tot, kind="stable")[:N]] = True
    rec = {k: round(float((v & T).sum() / max(N, 1)), 4) for k, v in sel.items()}
    diffs = {}
    for k in sel:
        if k.startswith("singlet"):
            continue
        d = []
        for _ in range(1000):
            i = rng.integers(0, len(T), len(T))
            d.append(((sel["singlet"][i] & T[i]).sum() - (sel[k][i] & T[i]).sum()) / max(T[i].sum(), 1))
        lo, hi = np.percentile(d, [2.5, 97.5])
        diffs[k] = {"diff": round(rec["singlet"] - rec[k], 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}
    miss = T & ~sing
    called_min = float(tot[sing].min()) if sing.any() else 0.0
    res[ds] = {"n_truth": N, "recall_at_N": rec, "singlet_minus": diffs, "n_called": {k: int(v.sum()) for k, v in calls.items()},
               "missed_truth": {"n": int(miss.sum()), "umi_median": float(np.median(tot[miss])) if miss.any() else None,
                                "umi_max": float(tot[miss].max()) if miss.any() else None, "smallest_called_umi": called_min,
                                "called_by": {t: int((v & miss).sum()) for t, v in calls.items() if t != "singlet"}},
               "extra_calls": {"n": int((sing & ~T).sum()), "umi_median": float(np.median(tot[sing & ~T])) if (sing & ~T).any() else None,
                               "called_by": {t: int((v & sing & ~T).sum()) for t, v in calls.items() if t != "singlet"}}}
json.dump(res, open(out, "w"), indent=1)
for ds, r in res.items():
    print(ds, r["n_truth"], r["recall_at_N"], r["missed_truth"], r["extra_calls"], flush=True)
