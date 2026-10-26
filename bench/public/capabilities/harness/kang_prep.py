# SPDX-License-Identifier: MIT
"""Inputs for the expression-based doublet callers on a genotype-multiplexed lane: Singlet's gene-level counts of the cells it
called (`called_v2`), in the layout scrublet_run.py and barnyard_tools.R read.

    python -I kang_prep.py RUN_DIR WORK_DIR
writes WORK_DIR/cells.mtx (genes x cells), cells.csv (barcode, n_umi), genes.csv (id, name, species=human), barcodes.txt (cellsnp input).
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from barnyard_prep import read_10x  # noqa: E402

run, work = sys.argv[1:3]
os.makedirs(work, exist_ok=True)
X, gid, gname, bcs = read_10x(f"{run}/filtered_feature_bc_matrix")
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t")
called = set(cc.loc[cc.called_v2.astype(int) == 1, "barcode"].astype(str).str.replace(r"-1$", "", regex=True))  # production cell caller
keep = np.array([i for i, b in enumerate(bcs) if b in called])
print(f"filtered matrix {X.shape}; called cells {len(called)}; kept {len(keep)}")
X = X[:, keep]
pd.DataFrame({"barcode": bcs[keep], "n_umi": np.asarray(X.sum(0)).ravel()}).to_csv(f"{work}/cells.csv", index=False)
pd.DataFrame({"id": gid, "name": gname, "species": "human"}).to_csv(f"{work}/genes.csv", index=False)
scipy.io.mmwrite(f"{work}/cells.mtx", X.tocoo())
pd.Series([b + "-1" for b in bcs[keep]]).to_csv(f"{work}/barcodes.txt", index=False, header=False)

# Droplet-level inputs of the ambient tools (SoupX droplet table, CellBender input, DecontX background): every barcode with >= 1 UMI
import gzip  # noqa: E402
raw, rid, rname, rbc = read_10x(f"{run}/raw_feature_bc_matrix")
assert (rid == gid).all()
tot = np.asarray(raw.sum(0)).ravel()
nz = np.where(tot > 0)[0]
os.makedirs(f"{work}/raw", exist_ok=True)
with gzip.open(f"{work}/raw/matrix.mtx.gz", "wb") as fh:
    scipy.io.mmwrite(fh, raw[:, nz].tocoo(), field="integer")
pd.DataFrame({0: rid, 1: rname, 2: "Gene Expression"}).to_csv(f"{work}/raw/features.tsv.gz", sep="\t", header=False, index=False)
pd.DataFrame({0: rbc[nz]}).to_csv(f"{work}/raw/barcodes.tsv.gz", sep="\t", header=False, index=False)
e = np.where((tot >= 10) & (tot <= 100))[0]
e = np.random.default_rng(0).choice(e, min(20000, len(e)), replace=False)
scipy.io.mmwrite(f"{work}/empty.mtx", raw[:, e].tocoo())
import json  # noqa: E402
json.dump({"n_cells": int(len(keep)), "n_raw_barcodes_ge1umi": int(len(nz))}, open(f"{work}/pool.json", "w"))
