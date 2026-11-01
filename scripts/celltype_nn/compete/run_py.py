#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Python competitors in the cell type comparison, run as published, trained on the same Census corpus as the network.

  celltypist  CellTypist 1.7 (MIT). ``celltypist.train`` on the corpus training split (log1p CP10k of the 6,000-gene
              vocabulary, SGD with mini-batches as the CellTypist docs recommend for large references), then
              ``celltypist.annotate`` per sample without majority voting. Output: label and its probability.
  scanvi      scvi-tools 1.4 (BSD-3). scVI then scANVI on a stratified corpus subsample (``--per-term`` cells per term,
              batch = dataset), one model per species; each query sample is mapped by scArches surgery
              (``SCANVI.load_query_data`` + 100 epochs) and labelled by the soft prediction's argmax and its probability.

  python scripts/celltype_nn/compete/run_py.py train   --method celltypist --data $D --out $D/compete
  python scripts/celltype_nn/compete/run_py.py predict --method celltypist --out $D/compete [--studies GSE1,GSE2]

Writes ``<out>/calls/<method>/<GSE>/<GSM>.parquet`` (barcode, pred_cl, score, prob) and appends to
``<out>/calls/<method>/timings.jsonl`` (seconds per sample, threads, device). Training time goes to ``models/<method>/train.json``.
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


def _lognorm(X):
    from singlet.celllabels.celltype import lognorm

    return lognorm(sps.csr_matrix(X, dtype=np.float32))


def _queries(out: Path, studies: str):
    import pandas as pd

    s = pd.read_csv(out / "query" / "samples.csv")
    if studies:
        s = s[s["gse"].isin(studies.split(","))]
    return s


def train_celltypist(a) -> int:
    import anndata as ad
    import celltypist
    import pandas as pd

    t0 = time.time()
    vocab = json.loads((a.data / "vocab.json").read_text())
    X = sps.load_npz(a.data / "corpus_counts.npz").tocsr()
    obs = pd.read_parquet(a.data / "corpus_obs.parquet", columns=["organism", "split"])
    cl = pd.read_parquet(a.data / "space_cells.parquet", columns=["author_term"])["author_term"].to_numpy(object)
    mdir = a.out / "models" / "celltypist"
    mdir.mkdir(parents=True, exist_ok=True)
    info = {}
    for sp in ("human", "mouse"):
        idx = np.flatnonzero((obs["organism"].to_numpy() == sp) & (obs["split"].to_numpy() == "train") & pd.notna(cl))
        A = ad.AnnData(X=_lognorm(X[idx]), obs=pd.DataFrame({"cl": cl[idx].astype(str)}),
                       var=pd.DataFrame(index=pd.Index(vocab["genes"])))
        t1 = time.time()
        m = celltypist.train(A, labels="cl", use_SGD=True, mini_batch=True, n_jobs=a.threads, check_expression=False,
                             feature_selection=False, balance_cell_type=True, epochs=a.epochs)
        m.write(str(mdir / f"{sp}.pkl"))
        info[sp] = dict(cells=len(idx), labels=int(len(set(cl[idx]))), seconds=time.time() - t1)
        print(sp, info[sp], flush=True)
    (mdir / "train.json").write_text(json.dumps(dict(info=info, threads=a.threads, seconds=time.time() - t0), indent=1))
    return 0


def predict_celltypist(a) -> int:
    import anndata as ad
    import celltypist
    import pandas as pd

    mdir = a.out / "models" / "celltypist"
    models = {sp: celltypist.models.Model.load(str(mdir / f"{sp}.pkl")) for sp in ("human", "mouse") if (mdir / f"{sp}.pkl").exists()}
    for r in _queries(a.out, a.studies).itertuples():
        A = ad.read_h5ad(a.out / "query" / r.gse / f"{r.gsm}.h5ad")
        t0 = time.time()
        Q = ad.AnnData(X=_lognorm(A.X), var=A.var.copy())
        p = celltypist.annotate(Q, model=models[r.species], majority_voting=False)
        lab = p.predicted_labels["predicted_labels"].astype(str).to_numpy()
        prob = p.probability_matrix.max(1).to_numpy(np.float32)
        _save(a, r, A, lab, prob, prob, time.time() - t0, "cpu")
    return 0


def _ref_subsample(a, sp):
    import anndata as ad

    A = ad.read_h5ad(a.out / "reference" / f"ref_{sp}.h5ad")
    return A


def train_scanvi(a) -> int:
    import anndata as ad
    import pandas as pd
    import scvi
    import torch

    mdir = a.out / "models" / "scanvi"
    mdir.mkdir(parents=True, exist_ok=True)
    info = {}
    torch.set_float32_matmul_precision("high")
    for sp in ("human", "mouse"):
        A = _ref_subsample(a, sp)
        A.layers["counts"] = A.X.copy()
        A.obs["batch"] = A.obs["dataset_id"].astype(str)
        t1 = time.time()
        scvi.model.SCVI.setup_anndata(A, layer="counts", batch_key="batch", labels_key="cl")
        v = scvi.model.SCVI(A, n_latent=30, n_layers=2, gene_likelihood="nb")
        v.train(max_epochs=a.epochs, early_stopping=True, batch_size=1024)
        s = scvi.model.SCANVI.from_scvi_model(v, unlabeled_category="Unknown")
        s.train(max_epochs=20, n_samples_per_label=100, batch_size=1024)
        s.save(str(mdir / sp), overwrite=True, save_anndata=False)
        A.obs[[]].to_csv(mdir / f"{sp}_obs_index.csv")
        info[sp] = dict(cells=A.n_obs, labels=int(A.obs["cl"].nunique()), seconds=time.time() - t1)
        print(sp, info[sp], flush=True)
    (mdir / "train.json").write_text(json.dumps(dict(info=info, device=str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"), indent=1))
    return 0


def predict_scanvi(a) -> int:
    import anndata as ad
    import scvi
    import torch

    mdir = a.out / "models" / "scanvi"
    refs = {}
    for r in _queries(a.out, a.studies).itertuples():
        if r.species not in refs:
            R = _ref_subsample(a, r.species)
            R.layers["counts"] = R.X.copy()
            R.obs["batch"] = R.obs["dataset_id"].astype(str)
            refs[r.species] = scvi.model.SCANVI.load(str(mdir / r.species), adata=R)
        ref = refs[r.species]
        A = ad.read_h5ad(a.out / "query" / r.gse / f"{r.gsm}.h5ad")
        t0 = time.time()
        Q = ad.AnnData(X=sps.csr_matrix(A.X, dtype=np.float32), var=A.var.copy())
        Q.layers["counts"] = Q.X.copy()
        Q.obs["batch"] = f"query_{r.gsm}"
        Q.obs["cl"] = "Unknown"
        scvi.model.SCANVI.prepare_query_anndata(Q, ref)
        q = scvi.model.SCANVI.load_query_data(Q, ref)
        q.train(max_epochs=100, plan_kwargs=dict(weight_decay=0.0), check_val_every_n_epoch=None, batch_size=1024)
        P = q.predict(soft=True)
        lab = P.columns.to_numpy()[P.to_numpy().argmax(1)].astype(str)
        prob = P.to_numpy().max(1).astype(np.float32)
        _save(a, r, A, lab, prob, prob, time.time() - t0, "cuda" if torch.cuda.is_available() else "cpu")
    return 0


def _save(a, r, A, lab, score, prob, secs, device):
    import pandas as pd

    d = a.out / "calls" / a.method / r.gse
    d.mkdir(parents=True, exist_ok=True)
    lab = np.array([x if str(x).startswith("CL:") else None for x in lab], dtype=object)
    pd.DataFrame({"barcode": A.obs["barcode"].astype(str).to_numpy(), "pred_cl": lab, "score": score, "prob": prob}).to_parquet(
        d / f"{r.gsm}.parquet", index=False)
    with open(a.out / "calls" / a.method / "timings.jsonl", "a") as f:
        f.write(json.dumps(dict(gse=r.gse, gsm=r.gsm, n_cells=int(A.n_obs), seconds=secs, threads=a.threads, device=device)) + "\n")
    print(f"{a.method} {r.gse}/{r.gsm}: {A.n_obs} cells {secs:.1f}s", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "predict"])
    ap.add_argument("--method", required=True, choices=["celltypist", "scanvi"])
    ap.add_argument("--data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--studies", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    a = ap.parse_args(argv)
    fn = {("train", "celltypist"): train_celltypist, ("predict", "celltypist"): predict_celltypist,
          ("train", "scanvi"): train_scanvi, ("predict", "scanvi"): predict_scanvi}[(a.cmd, a.method)]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
