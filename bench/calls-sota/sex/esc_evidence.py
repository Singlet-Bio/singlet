#!/usr/bin/env python3
"""Per-cell X-escape evidence for development libraries: total UMIs, summed UMIs of the escape set, XIST, chrY (Singlet set).

Escape set (ESCAPE below) is written from the literature on genes that escape X inactivation broadly across tissues
(Tukiainen 2017 Nature; Balaton 2015; Carrel and Willard 2005), not from cellXY. Usage: esc_evidence.py dev|heldout
"""
import gzip, importlib.util, sys
from pathlib import Path
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sps

W = Path("/mnt/projects/debruinz_project/debruinz/singlet")
OUT = W / "bench/calls-sota/sex"
spec = importlib.util.spec_from_file_location("sexmod", OUT / "sexmod.py")
SX = importlib.util.module_from_spec(spec); sys.modules["sexmod"] = SX; spec.loader.exec_module(SX)
ESCAPE = ("KDM6A DDX3X EIF2S3 ZFX KDM5C EIF1AX RPS4X TXLNG SMC1A USP9X UBA1 PRKX STS NLGN4X CA5B CD99L2 PLCXD1 "
          "GTPBP6 ZRSR2 HCFC1 BCOR MED14 OFD1 AP1S2 PUDP").split()


def one(d):
    M = scipy.io.mmread(gzip.open(d / "matrix.mtx.gz")).tocsr()          # genes x cells
    genes = (d / "genes.tsv").read_text().split("\n")[:-1] if (d / "genes.tsv").read_text().endswith("\n") else (d / "genes.tsv").read_text().split("\n")
    cols = [int(c) for c in (d / "cells.tsv").read_text().split()]
    X = M.T.tocsr()
    up = np.array([g.upper() for g in genes])
    tot = np.asarray(X.sum(1)).ravel()
    ei = np.flatnonzero(np.isin(up, ESCAPE))
    esc = np.asarray(X[:, ei].sum(1)).ravel() if len(ei) else np.zeros(X.shape[0])
    x, y = SX.sex_evidence(X, genes, "human")
    return pd.DataFrame({"col": cols, "total": tot, "esc": esc, "x": x, "y": y}), len(ei)


def main():
    mode = sys.argv[1]
    names = ["lcl_pool", "osorio"] if mode == "dev" else ["schihost"]
    parts = []
    for n in names:
        df, k = one(OUT / "work/mtx" / n); df["sample"] = n; parts.append(df); print(n, len(df), "escape genes found", k, flush=True)
    if mode == "dev":
        for d in sorted((W / "bpub/work/sex/mtx").iterdir()):
            try:
                df, k = one(d); df["sample"] = d.name; parts.append(df)
            except Exception as e:
                print("skip", d.name, e, flush=True)
        print("tier2 samples", len(parts) - 2, flush=True)
    pd.concat(parts).to_parquet(OUT / f"evidence/esc_{mode}.parquet", index=False)
    print("done")


main()
