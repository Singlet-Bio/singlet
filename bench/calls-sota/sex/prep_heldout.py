#!/usr/bin/env python3
"""Held-out library: GSE205796 (scHi-HOST), uninfected pool of 1000 Genomes lymphoblastoid lines of both sexes.

Truth per cell = 1000 Genomes sex of the line assigned by the authors' genotype demultiplexing (metadata columns LCL and Sex).
Counts = the authors' deposited Cell Ranger matrix (no Singlet run of this library exists). Run once, after the gate is committed.
Writes evidence/schihost.parquet and work/mtx/schihost/ (for cellxy.R).
"""
import gzip
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps

W = Path("/mnt/projects/debruinz_project/debruinz/singlet")
OUT = W / "bench/calls-sota/sex"
G = W / "data/heldout-launch/schihost_gse205796/geo"
spec = importlib.util.spec_from_file_location("sexmod", OUT / "sexmod.py")
SX = importlib.util.module_from_spec(spec)
sys.modules["sexmod"] = SX
spec.loader.exec_module(SX)

meta = pd.read_csv(G / "GSE205796_Uninfected_SeuratMetadata.csv.gz", index_col=0)
bc = pd.read_csv(G / "GSE205796_Uninfected_barcodes.tsv.gz", header=None)[0].astype(str).tolist()
ft = pd.read_csv(G / "GSE205796_Uninfected_features.tsv.gz", header=None, sep="\t")
names = ft[1].astype(str).tolist() if ft.shape[1] > 1 else ft[0].astype(str).tolist()
print("metadata", meta.shape, "barcodes", len(bc), "features", len(names), flush=True)

# matrix: genes x cells, coordinate text; read in chunks
rows, cols, vals = [], [], []
with gzip.open(G / "GSE205796_Uninfected_matrix.mtx.gz", "rt") as fh:
    line = fh.readline()
    while line.startswith("%"):
        line = fh.readline()
    ng, nc, nnz = map(int, line.split())
    print("matrix", ng, nc, nnz, flush=True)
    for ch in pd.read_csv(fh, sep=" ", header=None, names=["g", "c", "v"], dtype={"g": np.int32, "c": np.int32, "v": np.float32},
                          chunksize=20_000_000):
        rows.append(ch["g"].to_numpy() - 1)
        cols.append(ch["c"].to_numpy() - 1)
        vals.append(ch["v"].to_numpy())
        print("read", sum(len(r) for r in rows), flush=True)
M = sps.csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ng, nc))  # genes x cells
del rows, cols, vals
X = M.T.tocsr()  # cells x genes
bcs = pd.Index(bc)
keep = np.flatnonzero(bcs.isin(meta.index))
print("cells with metadata", len(keep), "of", len(bc), flush=True)
X = X[keep]
bck = [bc[i] for i in keep]
m = meta.loc[bck]
x, y = SX.sex_evidence(X, names, "human")
df = pd.DataFrame({"dataset": "schihost", "tier": 1, "col": np.arange(len(bck)), "barcode": bck, "demux": m["LCL"].astype(str).to_numpy(),
                   "total_umi": np.asarray(X.sum(1)).ravel(), "x": x, "y": y, "unit": m["LCL"].astype(str).to_numpy(),
                   "truth": m["Sex"].str.lower().to_numpy()})
print(df.groupby(["unit", "truth"]).size().to_string(), flush=True)
chk = df.groupby("unit")["truth"].nunique()
print("lines with more than one sex label:", int((chk > 1).sum()), flush=True)
(OUT / "evidence").mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT / "evidence/schihost.parquet", index=False)
d = OUT / "work/mtx/schihost"
d.mkdir(parents=True, exist_ok=True)
import scipy.io
scipy.io.mmwrite(str(d / "matrix.mtx"), sps.csr_matrix(X.T.astype(np.int32)))
os.system(f"gzip -f {d / 'matrix.mtx'}")
(d / "genes.tsv").write_text("\n".join(names) + "\n")
(d / "cells.tsv").write_text("\n".join(str(i) for i in range(len(bck))) + "\n")
(d / "species.txt").write_text("human\n")
print("done")
