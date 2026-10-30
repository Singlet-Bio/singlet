#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Controlled runtime of the fast cell type tools: Singlet's network and CellTypist (the others are 10-300x slower), each
labelling every benchmark sample's called cells in turn, back to back in one job with the same thread count, from counts already
in memory (reading is not timed). Writes ``<compete>/calls/<tool>/timings_controlled.jsonl``, which
``artifacts.py`` prefers over the timings recorded while the comparison ran on shared nodes.

  python scripts/celltype_nn/compete/timing.py --compete $D/compete --model $D/models/m1_notissue/model.pt --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "python"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compete", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = str(a.threads)

    import anndata as ad
    import celltypist
    import pandas as pd
    import torch

    from singlet.celllabels.celltype import lognorm
    from singlet.celltype_nn import CellTypePredictor

    torch.set_num_threads(a.threads)
    pred = CellTypePredictor.load(a.model)
    mdir = a.compete / "models" / "celltypist"
    ct = {sp: celltypist.models.Model.load(str(mdir / f"{sp}.pkl")) for sp in ("human", "mouse")}
    samples = pd.read_csv(a.compete / "query" / "samples.csv")
    outs = {t: open(a.compete / "calls" / d / "timings_controlled.jsonl", "w") for t, d in (("singlet", "singlet_nn_argmax"), ("CellTypist", "celltypist"))}
    for r in samples.itertuples():
        A = ad.read_h5ad(a.compete / "query" / r.gse / f"{r.gsm}.h5ad")
        X = sps.csr_matrix(A.X, dtype=np.float32)
        genes = list(A.var_names)
        t0 = time.perf_counter()
        P = pred.predict(X, genes, r.species, threads=a.threads)
        pred.summarise(P)
        ts = time.perf_counter() - t0
        t0 = time.perf_counter()
        Q = ad.AnnData(X=lognorm(X), var=A.var.copy())
        celltypist.annotate(Q, model=ct[r.species], majority_voting=False)
        tc = time.perf_counter() - t0
        for t, secs in (("singlet", ts), ("CellTypist", tc)):
            outs[t].write(json.dumps(dict(gse=r.gse, gsm=r.gsm, n_cells=int(A.n_obs), seconds=secs, threads=a.threads, device="cpu",
                                          controlled=True)) + "\n")
        print(f"{r.gse}/{r.gsm} {A.n_obs} cells: singlet {ts:.2f}s celltypist {tc:.2f}s", flush=True)
    for f in outs.values():
        f.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
