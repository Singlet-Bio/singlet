# SPDX-License-Identifier: MIT
"""Common inputs for every tool on a human + mouse mixture library (ambient-rna and doublets capabilities).

Reads one Singlet run directory (`singlet-campaign --pipeline --raw-matrix`, joint GRCh38 + GRCm39 reference) and writes, under
WORK/<id>/, exactly what each competitor needs, so that every tool sees the same droplets and the same cells:

  raw/                 10x-style directory: every barcode with >= 1 UMI (gene level, Singlet's own counts); input of CellBender,
                       the droplet table of SoupX
  cells.mtx, cells.csv, genes.csv    genes x called cells (Singlet's `called_v2` set), their barcodes, gene ids and species
  empty.mtx            20,000 random barcodes with 10-100 UMI (DecontX background)
  pool.json            species composition of the ambient pool under several pool definitions (truth input)

Species is the Ensembl prefix (ENSG human / ENSMUSG mouse). Run inside a Slurm job:  python -I barnyard_prep.py RUN_DIR WORK_DIR
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def read_10x(d):
    """10x-style directory (plain or .gz) -> genes x barcodes csc, feature ids, feature names, barcodes (suffix -1 removed)."""
    def op(n):
        p = os.path.join(d, n)
        return gzip.open(p + ".gz", "rt") if os.path.exists(p + ".gz") else open(p)
    with op("matrix.mtx") as f:
        X = scipy.io.mmread(f).tocsc()
    feats = pd.read_csv(op("features.tsv"), sep="\t", header=None)
    bcs = pd.read_csv(op("barcodes.tsv"), sep="\t", header=None)[0].astype(str).str.replace(r"-1$", "", regex=True).values
    return X, feats[0].astype(str).values, feats[1].astype(str).values, bcs


def main(run, work):
    os.makedirs(f"{work}/raw", exist_ok=True)
    raw, gid, gname, rbc = read_10x(f"{run}/raw_feature_bc_matrix")
    cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t")
    # `called_v2` is the production cell caller (EmptyDrops-style on stored data); it defines filtered_feature_bc_matrix, the
    # doublet scores and the ambient correction. `is_cell` is the older STARsolo call and is not used.
    called = cc.loc[cc.called_v2.astype(int) == 1, "barcode"].astype(str).str.replace(r"-1$", "", regex=True).values
    fil = pd.read_csv(f"{run}/filtered_feature_bc_matrix/barcodes.tsv.gz", header=None)[0].astype(str).str.replace(r"-1$", "", regex=True).values
    print("cell_calls TRUE == filtered_feature_bc_matrix barcodes:", set(called) == set(fil), len(called))
    species_g = np.where(pd.Series(gid).str.startswith("ENSMUSG").values, 1, 0)
    tot = np.asarray(raw.sum(0)).ravel()
    print(f"raw {raw.shape}, barcodes >= 1 UMI: {int((tot > 0).sum())}, nnz {raw.nnz}")
    keep = np.where(tot > 0)[0]
    with gzip.open(f"{work}/raw/matrix.mtx.gz", "wb") as fh:
        scipy.io.mmwrite(fh, raw[:, keep].tocoo(), field="integer")
    pd.DataFrame({0: gid, 1: gname, 2: "Gene Expression"}).to_csv(f"{work}/raw/features.tsv.gz", sep="\t", header=False, index=False)
    pd.DataFrame({0: rbc[keep]}).to_csv(f"{work}/raw/barcodes.tsv.gz", sep="\t", header=False, index=False)

    idx = pd.Series(np.arange(len(rbc)), index=rbc).reindex(called)
    assert idx.notna().all(), "called cells missing from the raw matrix"
    cell_idx = idx.astype(int).values
    Xc = raw[:, cell_idx]

    def species_umis(X):
        return (np.asarray(X[species_g == 0].sum(0)).ravel(), np.asarray(X[species_g == 1].sum(0)).ravel())

    H, M = species_umis(Xc)
    pd.DataFrame({"barcode": called, "n_umi": H + M, "human_umi": H, "mouse_umi": M}).to_csv(f"{work}/cells.csv", index=False)
    pd.DataFrame({"id": gid, "name": gname, "species": np.where(species_g == 1, "mouse", "human")}).to_csv(f"{work}/genes.csv", index=False)
    scipy.io.mmwrite(f"{work}/cells.mtx", Xc.tocoo())

    def pool(lo, hi, cap=None):
        i = np.where((tot >= lo) & (tot <= hi))[0]
        if cap and len(i) > cap:
            i = i[np.argsort(-tot[i], kind="stable")[:cap]]
        h, m = species_umis(raw[:, i])
        return {"mouse_share": float(m.sum() / (h.sum() + m.sum())), "n_barcodes": int(len(i)), "umis": int(h.sum() + m.sum())}

    # The pool definition Singlet itself uses (1-100 UMI, top 50,000 by count) is the primary truth input; the others are
    # sensitivity checks reported by compute().
    pools = {"1_100_top50k": pool(1, 100, 50000), "10_100": pool(10, 100), "20_200": pool(20, 200), "1_20": pool(1, 20)}
    print("ambient pools:", pools)
    rng = np.random.default_rng(0)
    e = np.where((tot >= 10) & (tot <= 100))[0]
    e = rng.choice(e, min(20000, len(e)), replace=False)
    scipy.io.mmwrite(f"{work}/empty.mtx", raw[:, e].tocoo())
    json.dump({"pools": pools, "n_cells": int(len(called)), "n_raw_barcodes_ge1umi": int(len(keep))},
              open(f"{work}/pool.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
