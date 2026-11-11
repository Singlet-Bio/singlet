# SPDX-License-Identifier: MIT
"""Post-hoc diagnostic (not used for the verdict): how the Singlet call depends on the barcode universe of the input matrix.

The campaign stores only barcodes with at least 5 reads, so singleton barcodes are absent from the stored matrix. The deposited matrices of
the held-out libraries keep every barcode with at least one UMI, so the primary scoring (gate) used a universe of total UMIs >= 1. This script
reruns the unchanged call (defaults) with the universe cut at total UMIs >= 1, 2, 3 and 5 and reports the call count, the fitted ambient
bound, the knee, and the truth recall (and, for CellPlex, precision and F1 on the hashtag truth).

    python -P universe_sensitivity.py CELLCALL.py OUT.json HELDOUT_DIR_CELLPLEX HELDOUT_DIR_NUC
"""
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

ccpath, outp, dcp, dnu = sys.argv[1:5]
spec = importlib.util.spec_from_file_location("cellcall", ccpath)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
res = {}
# the primary tables hold exactly the stored matrix columns; rebuild matrices from the binary triplets written by the prep scripts
for name, d in (("cellplex10k", dcp), ("nuclei_poc", dnu)):
    dims = [int(x) for x in open(f"{d}/dims.txt").read().split()]
    i = np.fromfile(f"{d}/i.bin", dtype=np.int32)
    j = np.fromfile(f"{d}/j.bin", dtype=np.int32)
    x = np.fromfile(f"{d}/x.bin", dtype=np.int32)
    G = sp.csc_matrix((x, (i, j)), shape=(dims[0], dims[1]))
    t = pd.read_csv(f"{d}/table.tsv.gz", sep="\t", dtype={"barcode": str})
    tot = t["umi_total"].to_numpy()
    T = (t["truth"] == "cell").to_numpy()
    res[name] = {}
    for mn in (1, 2, 3, 5):
        k = np.nonzero(tot >= mn)[0]
        Gk, tk = G[:, k].tocsc(), tot[k]
        srt = np.sort(tk)[::-1]
        knee_called = tk >= max(srt[int(0.01 * 3000)] / 10.0, 1)
        c2, fdr, p = cc.call(Gk, knee_called)
        full = np.zeros(len(t), bool)
        full[k[c2]] = True
        e = {"min_umi": mn, "n_universe": int(len(k)), "n_called_v2": int(c2.sum()), "lower": p.get("lower"), "retain_total": p.get("retain_total"),
             "n_ambient": p.get("n_ambient"), "n_tested": p.get("n_tested"), "recall_truth_cells": round(float((full & T).sum() / max(T.sum(), 1)), 4)}
        if name == "cellplex10k":
            dec = t["truth"].isin(["cell", "empty"]).to_numpy() & (tot >= 50)
            tp = int((full & T & dec).sum())
            pr, rc = tp / max(int((full & dec).sum()), 1), tp / max(int((T & dec).sum()), 1)
            e.update(precision=round(pr, 4), recall=round(rc, 4), f1=round(2 * pr * rc / max(pr + rc, 1e-12), 4))
        else:
            auth = t["in_authors_set"].to_numpy().astype(bool)
            e["share_calls_outside_authors_set_T"] = round(float((full & ~auth).sum() / max(full.sum(), 1)), 4)
        res[name][str(mn)] = e
        print(name, e, flush=True)
json.dump(res, open(outp, "w"), indent=1, default=float)
