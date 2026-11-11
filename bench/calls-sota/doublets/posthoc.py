# SPDX-License-Identifier: MIT
"""POST HOC sensitivity (written after the pre-registered held-out scores were seen; not verdict-bearing): the pre-registered label rules
let low-depth cells be labeled doublets (tag noise in CellPlex, ambient RNA in the barnyard), so the scores of the first run are rescored
under depth-robust rules on the same stored scores. Usage: python -P posthoc.py OUT_DIR"""
import json
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

out = sys.argv[1]
W = "/mnt/projects/debruinz_project/debruinz/singlet"
res = {}


def score(name, sc, y, ok):
    y, sc = y[ok], sc.loc[ok].reset_index(drop=True)
    r = {m: dict(auprc=float(average_precision_score(y, sc[m])), auroc=float(roc_auc_score(y, sc[m]))) for m in ("v2", "scdblfinder", "prod")}
    rng = np.random.default_rng(0)
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum():
            d.append(average_precision_score(y[i], sc["v2"].to_numpy()[i]) - average_precision_score(y[i], sc["scdblfinder"].to_numpy()[i]))
    r.update(n=int(ok.sum()), n_doublets=int(y.sum()), diff=r["v2"]["auprc"] - r["scdblfinder"]["auprc"], ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    res[name] = r


# CellPlex: tags
base = f"{W}/bench/calls-sota/cell_calling/heldout_cellplex10k"
sc = pd.read_csv(f"{out}/cellplex10k.scores.tsv.gz", sep="\t")
with h5py.File(f"{base}/raw.h5", "r") as f:
    m = f["matrix"]
    M = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=tuple(m["shape"][:]))
    bc = np.array([b.decode() for b in m["barcodes"][:]])
    ft = np.array([s.decode() for s in m["features"]["feature_type"][:]])
idx = pd.Series(np.arange(len(bc)), index=bc).reindex(sc.barcode).to_numpy()
C = np.asarray(M[:, idx][np.flatnonzero(ft == "Multiplexing Capture"), :].todense()).T.astype(float)
s = C.sum(1)
srt = np.sort(C, axis=1)[:, ::-1]
f2 = srt[:, 1] / np.maximum(s, 1)
y = np.where((s >= 500) & (srt[:, 1] >= 100) & (f2 >= 0.25), 1, np.where((s >= 500) & (f2 <= 0.05), 0, -1))
score("cellplex10k_depth_robust", sc, y, y >= 0)

# hgmm5k: species, depth floor
base = f"{W}/state/cellcall_truth/starsolo/hgmm5k_v3/Solo.out/Gene/filtered"
sc = pd.read_csv(f"{out}/hgmm5k_v3.scores.tsv.gz", sep="\t")
Mx = scipy.io.mmread(f"{base}/matrix.mtx").tocsr()
feat = pd.read_csv(f"{base}/features.tsv", header=None, sep="\t")
isH = feat[0].astype(str).str.startswith("ENSG").to_numpy()
h = np.asarray(Mx[np.flatnonzero(isH), :].sum(0)).ravel()
mo = np.asarray(Mx[np.flatnonzero(~isH), :].sum(0)).ravel()
tot = h + mo
minor = np.minimum(h, mo) / np.maximum(tot, 1)
y = np.where((tot >= 5000) & (minor >= 0.20), 1, np.where((tot >= 5000) & (minor <= 0.05), 0, -1))
score("hgmm5k_v3_depth_robust", sc, y, y >= 0)
json.dump(res, open(f"{out}/posthoc.json", "w"), indent=1)
print(json.dumps(res, indent=1))
