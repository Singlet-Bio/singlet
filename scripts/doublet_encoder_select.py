#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Atlas reference cells for the doublet encoder (build time only): up to --cap cells per Cell Ontology term (terms with at least
--min-cells cells), round-robin over datasets, from the celltype_nn CELLxGENE training corpus -> OUT/lite/sel_X.npz, sel_meta.npz.
Input of scripts/doublet_encoder_distill.py. The doublet evaluation datasets are not involved.

  python scripts/doublet_encoder_select.py --out OUT
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

CORPUS = "/mnt/projects/debruinz_project/debruinz/singlet/data/celltype_nn"


def cmd_select(a):
    import pandas as pd

    obs = pd.read_parquet(f"{CORPUS}/corpus_obs.parquet").reset_index(drop=True)
    col = next(c for c in obs.columns if obs[c].astype(str).str.startswith("CL:").mean() > 0.9)
    ds = next(c for c in obs.columns if "dataset" in c.lower())
    rng = np.random.default_rng(0)
    obs["_r"] = rng.random(len(obs))
    counts = obs[col].value_counts()
    terms = counts.index[counts >= a.min_cells].tolist()
    obs["_rank"] = obs.groupby([col, ds])["_r"].rank(method="first")
    sel = obs[obs[col].isin(terms)].sort_values([col, "_rank", "_r"]).groupby(col).head(a.cap)
    idx = np.sort(sel.index.values)
    X = sp.load_npz(f"{CORPUS}/corpus_counts.npz").tocsr()
    Xs = X[idx]
    del X
    o = obs.loc[idx]
    ti = {t: i for i, t in enumerate(terms)}
    vocab = json.load(open(f"{CORPUS}/vocab.json"))
    out = Path(a.out) / "lite"
    out.mkdir(parents=True, exist_ok=True)
    sp.save_npz(out / "sel_X.npz", Xs)
    np.savez(
        out / "sel_meta.npz",
        term=np.array([ti[t] for t in o[col]]),
        dataset=o[ds].astype(str).values,
        terms=np.array(terms),
        genes=np.array(vocab["genes"]),
        symbols=np.array(vocab["symbols"]),
        mouse_ids=np.array(list(vocab["mouse"].keys())),
        mouse_human=np.array(list(vocab["mouse"].values())),
    )
    print("selected", Xs.shape, len(terms), "terms", o[ds].nunique(), "datasets")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--cap", type=int, default=400)
    cmd_select(ap.parse_args())
