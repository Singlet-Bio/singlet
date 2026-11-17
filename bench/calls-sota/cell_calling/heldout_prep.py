# SPDX-License-Identifier: MIT
"""Held-out hashed library (10x CellPlex PBMC 10k, raw feature matrix): Singlet call, hashtag truth, matrix for the EmptyDrops comparator.

    python -P heldout_prep.py RAW.h5 CELLCALL.py OUT_DIR

Truth class H (bench/calls-sota/gates/cell_calling.md): cell = total CMO UMIs >= 100, empty = total CMO UMIs <= 10, else unknown;
alt (robustness only, not the verdict) = >= 50 and <= 20. Singlet call = cellcall.call at defaults on gene counts, with the stored-knee
flag emulating STARsolo CellRanger2.2 3000 0.99 10. Timing is single-threaded (NUMBA_NUM_THREADS=1), first call includes the JIT.
"""
import importlib.util
import json
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

raw, ccpath, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("cellcall", ccpath)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

with h5py.File(raw, "r") as f:
    m = f["matrix"]
    shape = tuple(m["shape"][:])
    X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
    bc = np.array([b.decode() for b in m["barcodes"][:]])
    ftype = np.array([s.decode() for s in m["features"]["feature_type"][:]])
    fname = np.array([s.decode() for s in m["features"]["name"][:]])
print("raw matrix", shape, "nnz", X.nnz, "feature types", dict(zip(*np.unique(ftype, return_counts=True))), flush=True)
gex = ftype == "Gene Expression"
cmo = ftype == "Multiplexing Capture"
G = X[np.nonzero(gex)[0], :].tocsc()
tot = np.asarray(G.sum(axis=0)).ravel().astype(np.int64)
cmo_tot = np.asarray(X[np.nonzero(cmo)[0], :].sum(axis=0)).ravel().astype(np.int64)
keep = np.nonzero(tot >= 1)[0]
G, tot, cmo_tot, bc = G[:, keep].tocsc(), tot[keep], cmo_tot[keep], bc[keep]
print("universe (total gene UMIs >= 1):", len(keep), " CMO features:", list(fname[cmo]), flush=True)

# STARsolo CellRanger2.2 3000 0.99 10
srt = np.sort(tot)[::-1]
robust_max = srt[int(0.01 * 3000)]
knee_called = tot >= max(robust_max / 10.0, 1)
print("stored-knee emulation: robust max", robust_max, "threshold", robust_max / 10.0, "called", int(knee_called.sum()), flush=True)

times = []
for rep in range(2):
    t0 = time.time()
    c2, fdr, params = cc.call(G, knee_called)
    times.append(time.time() - t0)
    print(f"call #{rep + 1}: {times[-1]:.1f} s, called_v2 {int(c2.sum())}", flush=True)
params["seconds_first_call_with_jit"] = times[0]
params["seconds_second_call"] = times[1]
params["n_universe"] = int(len(keep))
params["n_called_v2"] = int(c2.sum())
params["core_minutes_per_10k_called"] = times[1] / 60.0 / max(c2.sum() / 1e4, 1e-9)
json.dump(params, open(f"{out}/singlet_params.json", "w"), indent=1, default=float)

def cls(hi, lo):
    return np.where(cmo_tot >= hi, "cell", np.where(cmo_tot <= lo, "empty", "unknown"))

df = pd.DataFrame({"barcode": bc, "umi_total": tot, "cmo_total": cmo_tot, "truth": cls(100, 10), "truth_strict": cls(50, 20),
                   "knee": knee_called.astype(int), "called_v2": c2.astype(int), "fdr": fdr})
df.to_csv(f"{out}/table_nocomp.tsv.gz", sep="\t", index=False)
# binary triplets for the R comparator
Gc = G.tocoo()
Gc.row.astype(np.int32).tofile(f"{out}/i.bin"); Gc.col.astype(np.int32).tofile(f"{out}/j.bin"); Gc.data.astype(np.int32).tofile(f"{out}/x.bin")
open(f"{out}/dims.txt", "w").write(f"{G.shape[0]} {G.shape[1]} {Gc.nnz}\n")
print("wrote", out)
