# SPDX-License-Identifier: MIT
"""False-split rate of the donor demultiplexer on single-donor libraries (gate: bench/calls-sota/gates/donor_single_split.md).
python score_single_split.py OUT.json PATH:LABEL [PATH:LABEL ...]   (PATH = donor_assignments.tsv)"""
import json, sys
import pandas as pd

def score(path):
    d = pd.read_csv(path, sep="\t")
    n_all = len(d)
    m = (d["cell_status"] == "cell") if "cell_status" in d else (d["donor_id"] == d["donor_id"])
    lab = d[m & d["donor_id"].astype(str).str.startswith("donor")]
    vc = lab["donor_id"].value_counts()
    n = int(vc.sum())
    dom = vc.index[0] if n else None
    minor = vc.iloc[1:]
    split = [k for k, v in minor.items() if v >= 50 and v >= 0.01 * n]
    return {"cells": n_all, "labeled": n, "k_found": int(len(vc)), "counts": {k: int(v) for k, v in vc.items()},
            "false_split_rate": float(minor.sum() / n) if n else None, "split_declared": split,
            "doublet_frac": float((d["donor_id"] == "doublet").mean()) if n_all else None,
            "status": {k: int(v) for k, v in d["cell_status"].value_counts().items()}}

if __name__ == "__main__":
    out, items = sys.argv[1], sys.argv[2:]
    R = {}
    for it in items:
        p, l = it.rsplit(":", 1)
        R[l] = score(p)
        r = R[l]
        print(l, "cells", r["cells"], "labeled", r["labeled"], "K", r["k_found"], "rate", r["false_split_rate"], "split", r["split_declared"], "dbl", r["doublet_frac"], r["counts"])
    tot = sum(r["labeled"] for r in R.values()); mn = sum(r["labeled"] * (r["false_split_rate"] or 0) for r in R.values())
    R["_pooled"] = {"labeled": tot, "false_split_rate": mn / tot if tot else None}
    print("pooled", R["_pooled"])
    json.dump(R, open(out, "w"), indent=1)
