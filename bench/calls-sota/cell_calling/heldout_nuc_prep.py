# SPDX-License-Identifier: MIT
"""Held-out hashed nuclei library (McGinnis 2019 POC nuclei, GSE129578): Singlet call and tag-confirmed nuclei (gate amendment A).

    python -P heldout_nuc_prep.py MTX.gz BARCODES.tsv MULTI.csv CELLCALL.py OUT_DIR
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

mtx, bcf, csvf, ccpath, out = sys.argv[1:6]
os.makedirs(out, exist_ok=True)
spec = importlib.util.spec_from_file_location("cellcall", ccpath)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
t0 = time.time()
M = sp.csc_matrix(scipy.io.mmread(mtx))
bc = np.array([ln.strip().split("\t")[0] for ln in open(bcf)])
print("matrix", M.shape, "nnz", M.nnz, "barcodes", len(bc), f"read {time.time() - t0:.0f} s", flush=True)
assert M.shape[1] == len(bc)
tot = np.asarray(M.sum(axis=0)).ravel().astype(np.int64)
keep = np.nonzero(tot >= 1)[0]
G, tot, bc = M[:, keep].tocsc(), tot[keep], bc[keep]
print("universe (total UMIs >= 1):", len(keep), flush=True)
srt = np.sort(tot)[::-1]
robust_max = srt[int(0.01 * 3000)]
knee_called = tot >= max(robust_max / 10.0, 1)  # STARsolo CellRanger2.2 3000 0.99 10
print("stored-knee emulation threshold", robust_max / 10.0, "called", int(knee_called.sum()), flush=True)
times = []
for rep in range(2):
    t1 = time.time()
    c2, fdr, params = cc.call(G, knee_called)
    times.append(time.time() - t1)
    print(f"call #{rep + 1}: {times[-1]:.1f} s, called_v2 {int(c2.sum())}", flush=True)
params.update(seconds_first_call_with_jit=times[0], seconds_second_call=times[1], n_universe=int(len(keep)), n_called_v2=int(c2.sum()),
              core_minutes_per_10k_called=times[1] / 60.0 / max(c2.sum() / 1e4, 1e-9))
json.dump(params, open(f"{out}/singlet_params.json", "w"), indent=1, default=float)
tags = pd.read_csv(csvf, dtype={"CellID": str}, encoding="utf-8-sig")
tag_ok = set(tags.loc[tags["nUMI_Bar"] >= 100, "CellID"])
authors = set(tags["CellID"])
df = pd.DataFrame({"barcode": bc, "umi_total": tot, "knee": knee_called.astype(int), "called_v2": c2.astype(int), "fdr": fdr})
df["truth"] = np.where(df["barcode"].isin(tag_ok), "cell", "unknown")
df["in_authors_set"] = df["barcode"].isin(authors).astype(int)
df.to_csv(f"{out}/table_nocomp.tsv.gz", sep="\t", index=False)
Gc = G.tocoo()
Gc.row.astype(np.int32).tofile(f"{out}/i.bin")
Gc.col.astype(np.int32).tofile(f"{out}/j.bin")
Gc.data.astype(np.int32).tofile(f"{out}/x.bin")
open(f"{out}/dims.txt", "w").write(f"{G.shape[0]} {G.shape[1]} {Gc.nnz}\n")
print("tag-confirmed nuclei in universe:", int((df["truth"] == "cell").sum()), "of", len(tag_ok), "; authors set", len(authors))
