# SPDX-License-Identifier: MIT
"""Held-out scoring of doublet/score_v2 against scDblFinder (gate: bench/calls-sota/gates/doublets.md). One library per call:

  python -P run_heldout.py NAME OUT_DIR DBL_RUN RSCRIPT SCDBLFINDER_R

NAME = cellplex10k (hashtag truth) or hgmm5k_v3 (species truth). Writes OUT_DIR/NAME.scores.tsv.gz and OUT_DIR/NAME.result.json."""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, roc_auc_score

name, out, dbl_run, rscript, rfile = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
W = "/mnt/projects/debruinz_project/debruinz/singlet"
THREADS = os.environ.get("SLURM_CPUS_PER_TASK", "4")
from singlet.features import doublet as D  # noqa: E402

t_all = time.perf_counter()
if name == "cellplex10k":
    import h5py

    base = f"{W}/bench/calls-sota/cell_calling/heldout_cellplex10k"
    tab = pd.read_csv(f"{base}/table_nocomp.tsv.gz", sep="\t")
    called = set(tab.barcode[tab.called_v2 == 1])
    with h5py.File(f"{base}/raw.h5", "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        M = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        bc = np.array([b.decode() for b in m["barcodes"][:]])
        ftype = np.array([s.decode() for s in m["features"]["feature_type"][:]])
        gid = np.array([s.decode() for s in m["features"]["id"][:]])
    sel = np.flatnonzero(np.isin(bc, list(called)))
    gex, cmo = ftype == "Gene Expression", ftype == "Multiplexing Capture"
    Ms = M[:, sel]
    X = sp.csr_matrix(Ms[np.flatnonzero(gex), :].T.tocsr(), dtype=np.float32)
    genes = gid[gex]
    C = np.asarray(Ms[np.flatnonzero(cmo), :].todense()).T.astype(float)  # cells x tags
    s = C.sum(1)
    srt = np.sort(C, axis=1)[:, ::-1]
    f2 = np.where(s > 0, srt[:, 1] / np.maximum(s, 1), 0)
    label = np.full(len(sel), -1)
    label[(s >= 100) & (f2 <= 0.10)] = 0
    label[(s >= 100) & (f2 >= 0.25)] = 1
    barcodes = bc[sel]
    meta = dict(n_tags=int(cmo.sum()), n_called=len(sel))
elif name == "hgmm5k_v3":
    base = f"{W}/state/cellcall_truth/starsolo/hgmm5k_v3/Solo.out/Gene/filtered"
    M = scipy.io.mmread(f"{base}/matrix.mtx").tocsr()  # genes x cells
    barcodes = pd.read_csv(f"{base}/barcodes.tsv", header=None)[0].astype(str).to_numpy()
    feat = pd.read_csv(f"{base}/features.tsv", header=None, sep="\t")
    X = sp.csr_matrix(M.T.tocsr(), dtype=np.float32)
    genes = feat[0].astype(str).to_numpy()
    is_h = np.array([g.startswith("ENSG") or g.startswith("GRCh38") or g.startswith("hg") for g in genes])
    if not is_h.any() or is_h.all():
        is_h = np.array(["ENSMUSG" not in g and "mm10" not in g for g in genes])
    h = np.asarray(X[:, np.flatnonzero(is_h)].sum(1)).ravel()
    mo = np.asarray(X[:, np.flatnonzero(~is_h)].sum(1)).ravel()
    minor = np.minimum(h, mo) / np.maximum(h + mo, 1)
    label = np.full(X.shape[0], -1)
    label[minor <= 0.05] = 0
    label[minor >= 0.20] = 1
    meta = dict(n_cells=int(X.shape[0]), human_genes=int(is_h.sum()), mouse_genes=int((~is_h).sum()))
else:
    sys.exit("unknown library")
species = "human"  # as in training: the barnyards and the human libraries all use the human panel
print(name, "cells", X.shape[0], "genes", X.shape[1], "doublets", int((label == 1).sum()), "singlets", int((label == 0).sum()),
      "dropped", int((label < 0).sum()), flush=True)

with tempfile.TemporaryDirectory() as tmp:
    # engine score: dbl_run on the genes x cells CSC (uint16), the call of bench/doublets/legacy_doublets/score.py
    G = sp.csc_matrix(X.T)
    pre = f"{tmp}/g"
    np.array([G.shape[0], G.shape[1], G.nnz], np.uint32).tofile(pre + ".hdr")
    G.indptr.astype(np.int32).tofile(pre + ".indptr")
    G.indices.astype(np.int32).tofile(pre + ".indices")
    np.minimum(G.data, 65535).astype(np.uint16).tofile(pre + ".data")
    t = time.perf_counter()
    r = subprocess.run([dbl_run, pre, f"{tmp}/out.tsv", f"threads={THREADS}", "nomap=1"], capture_output=True, text=True)
    t_prod = time.perf_counter() - t
    if r.returncode:
        sys.exit(f"dbl_run exit {r.returncode}: {r.stderr[-400:]}")
    prod = pd.read_csv(f"{tmp}/out.tsv", sep="\t").sort_values("col")["score"].to_numpy()
    tm = {}
    t = time.perf_counter()
    F = D.compute_inputs(X, list(genes), species, prod, timings=tm)
    v2 = D.score_inputs(F)
    t_pkg = time.perf_counter() - t
    scipy.io.mmwrite(f"{tmp}/c.mtx", sp.csc_matrix(X.T))
    t = time.perf_counter()
    r = subprocess.run([rscript, rfile, f"{tmp}/c.mtx", f"{tmp}/sdf.tsv"], capture_output=True, text=True)
    t_sdf = time.perf_counter() - t
    if r.returncode:
        sys.exit(f"scDblFinder failed: {r.stderr[-600:]}")
    sdf = pd.read_csv(f"{tmp}/sdf.tsv", sep="\t")["score"].to_numpy()

ok = (label >= 0) & np.isfinite(v2)
y = label[ok]
methods = {"v2": v2[ok], "scdblfinder": sdf[ok], "prod": prod[ok], "sdfl_a": np.asarray(F["sdfl"])[ok], "atlas_knn": np.asarray(F["atlas_knn"])[ok]}
res = {m: dict(auprc=float(average_precision_score(y, s)), auroc=float(roc_auc_score(y, s))) for m, s in methods.items()}
rng = np.random.default_rng(0)
d, n = [], len(y)
for _ in range(2000):
    i = rng.integers(0, n, n)
    if y[i].sum() == 0:
        continue
    d.append(average_precision_score(y[i], methods["v2"][i]) - average_precision_score(y[i], methods["scdblfinder"][i]))
diff = res["v2"]["auprc"] - res["scdblfinder"]["auprc"]
core_min = (t_prod + t_pkg) * int(THREADS) / 60 / (X.shape[0] / 1e4)  # upper bound: wall x allocated threads
out_j = dict(library=name, n_cells=int(X.shape[0]), n_scored=int(ok.sum()), n_doublets=int(y.sum()), prevalence=float(y.mean()), methods=res,
             v2_minus_scdblfinder=diff, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
             seconds=dict(prod_dbl_run=t_prod, package=t_pkg, package_stages=tm, scdblfinder=t_sdf, total=time.perf_counter() - t_all),
             core_minutes_per_10k_cells_prod_plus_package_upper_bound=core_min, threads=int(THREADS), meta=meta)
json.dump(out_j, open(f"{out}/{name}.result.json", "w"), indent=1)
pd.DataFrame({"barcode": barcodes, "label": label, "v2": v2, "scdblfinder": sdf, "prod": prod}).to_csv(
    f"{out}/{name}.scores.tsv.gz", sep="\t", index=False)
print(json.dumps(out_j, indent=1))
