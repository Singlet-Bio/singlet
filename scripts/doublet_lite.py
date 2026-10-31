#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build and evaluate the small numpy atlas encoder of the doublet stacker (python/singlet/features/doublet_lite.py).

  select   atlas reference cells from the celltype_nn training corpus (balanced over cell types and datasets) -> OUT/lite/sel.npz
  fit      gene panel + PCA encoder + Gaussian cell-type classifier from those cells only -> OUT/lite/model_<G>_<dim>.npz
  feats    lite features of the 17 evaluation datasets with a model (+ CPU seconds per dataset) -> OUT/lite/feats_<tag>/
  eval     leave-one-study-out AUPRC (stack.py recipe) of one input combination, paired CI against scDblFinder

The 17 evaluation datasets never enter `select` or `fit`."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

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


def lognorm_dense(C, mu=None):
    tot = np.asarray(C.sum(1)).ravel()
    d = C.toarray().astype(np.float32)
    d *= (1e4 / np.maximum(tot, 1.0))[:, None]
    np.log1p(d, out=d)
    return d


def cmd_fit(a):
    out = Path(a.out) / "lite"
    Xs = sp.load_npz(out / "sel_X.npz").tocsr()
    m = np.load(out / "sel_meta.npz", allow_pickle=True)
    term, dsid, terms = m["term"], m["dataset"], m["terms"]
    rng = np.random.default_rng(1)
    uds = np.unique(dsid)
    val_ds = set(rng.choice(uds, max(1, len(uds) // 10), replace=False))
    isval = np.array([d in val_ds for d in dsid])
    tr = np.flatnonzero(~isval)
    va = np.flatnonzero(isval)
    T = len(terms)
    # panel: genes with the largest between-cell-type variance of the mean log profile (training cells)
    tot = np.asarray(Xs[tr].sum(1)).ravel()
    Xl = sp.diags(1e4 / np.maximum(tot, 1)) @ Xs[tr]
    Xl = sp.csr_matrix(Xl)
    Xl.data = np.log1p(Xl.data)
    Gt = sp.csr_matrix((np.ones(len(tr)), (term[tr], np.arange(len(tr)))), shape=(T, len(tr)))
    cnt = np.asarray(Gt.sum(1)).ravel()
    tm = np.asarray((Gt @ Xl).todense()) / np.maximum(cnt, 1)[:, None]
    ok = cnt >= 5
    var = tm[ok].var(0)
    panel = np.sort(np.argsort(-var)[: a.genes])
    del Xl
    # encoder: PCA of the panel's log1p CP10k profiles (normalised by the panel total)
    xt = lognorm_dense(Xs[tr][:, panel])
    mu = xt.mean(0)
    xt -= mu
    cov = (xt.T @ xt) / len(xt)
    ev, evec = np.linalg.eigh(cov.astype(np.float64))
    W = evec[:, ::-1][:, : a.dim].astype(np.float32)
    Z = xt @ W
    del xt
    # Gaussian classes with one shared covariance
    means = np.zeros((T, a.dim))
    for t in range(T):
        sel = term[tr] == t
        if sel.any():
            means[t] = Z[sel].mean(0)
    resid = Z - means[term[tr]]
    S = (resid.T @ resid) / len(resid)
    S += 1e-3 * np.trace(S) / a.dim * np.eye(a.dim)
    C = np.linalg.cholesky(S)
    L = np.linalg.inv(C).T.astype(np.float32)
    present = cnt >= 5
    # temperature: held-out atlas datasets
    Wh, meanh = W.astype(np.float16).astype(np.float32), means.astype(np.float16).astype(np.float32)
    zv = (lognorm_dense(Xs[va][:, panel]) - mu) @ Wh
    A = zv @ L
    B = meanh @ L
    d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T
    d2[:, ~present] = np.inf
    yv = term[va]
    keep = present[yv]
    best = (1e9, None)
    for temp in (0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128):
        lg = -0.5 * d2[keep] / temp
        lg -= lg.max(1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(1, keepdims=True))
        nll = -lp[np.arange(keep.sum()), yv[keep]].mean()
        print("temp", temp, "held-out NLL", round(float(nll), 3), flush=True)
        if nll < best[0]:
            best = (nll, temp)
    acc = (np.argmin(d2[keep], 1) == yv[keep]).mean()
    print("held-out top-1 accuracy", round(float(acc), 3), "temp", best[1], "val cells", int(keep.sum()), "datasets", len(val_ds))
    means_out = means.astype(np.float16)
    means_out[~present] = 1e4  # absent terms: never selected
    mouse = dict(zip(m["mouse_human"], m["mouse_ids"]))
    arrs = dict(
        panel_ids=m["genes"][panel].astype("U15"),
        panel_symbols=m["symbols"][panel].astype("U16"),
        panel_mouse=np.array([mouse.get(g, "") for g in m["genes"][panel]], dtype="U19"),
        mu=mu.astype(np.float32),
        W=W.astype(np.float16),
        means=means_out,
        prec_chol=L,
        temp=np.float32(best[1]),
        terms=terms.astype("U12"),
    )
    f = out / f"model_{a.genes}_{a.dim}.npz"
    np.savez_compressed(f, **arrs)
    print("saved", f, os.path.getsize(f), "bytes; held-out acc", round(float(acc), 3))


def load_counts(name):
    import doublet_v2 as S

    return S.load_counts(name)


def cmd_feats(a):
    from singlet.features import doublet as D
    from singlet.features.doublet_lite import Lite

    model = Lite.load(a.model)
    out = Path(a.out) / "lite" / f"feats_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    cost = {}
    for n in a.names or list(D.STUDY):
        X, y, genes, species = load_counts(n)
        w0, c0 = time.perf_counter(), time.process_time()
        F = model.features(X, genes, species, n_sim=None if a.sim == 0 else a.sim * X.shape[0])
        cost[n] = dict(n_cells=int(X.shape[0]), wall_s=round(time.perf_counter() - w0, 2), cpu_s=round(time.process_time() - c0, 2),
                       peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        np.savez(out / f"{n}.npz", **F)
        print(n, cost[n], flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))


BASE5 = ["umi", "genes", "prod", "sdfl", "sdflc"]
COMBOS = {
    "base5": BASE5,
    "base5+knn+ent": BASE5 + ["atlas_knn", "ct_entropy"],
    "full": BASE5 + ["atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls"],
    "lite_both": BASE5 + ["lite_knn", "lite_entropy"],
    "lite_ent": BASE5 + ["lite_entropy"],
    "lite_knn": BASE5 + ["lite_knn"],
    "knn+lite_ent": BASE5 + ["atlas_knn", "lite_entropy"],
    "lite_knn+ent": BASE5 + ["lite_knn", "ct_entropy"],
    "lite_both+clf": BASE5 + ["lite_knn", "lite_entropy", "atlas_clf"],
}


def cmd_eval(a):
    import pandas as pd
    from sklearn.metrics import average_precision_score

    import doublet_v2 as S
    from singlet.features import doublet as D

    data = S.load_training("stored")
    if a.tag:
        for n in D.STUDY:
            z = np.load(Path(a.out) / "lite" / f"feats_{a.tag}" / f"{n}.npz")
            for k in z.files:
                data[n][0][k] = z[k]
    cols = COMBOS[a.combo]
    ap = {}
    for n in D.STUDY:
        s = D.loso_scores(data, n, cols)
        ap[n] = float(average_precision_score(data[n][1], s))
    res = Path(a.out) / "lite" / "eval"
    res.mkdir(parents=True, exist_ok=True)
    name = f"{a.combo}__{a.tag or 'stored'}"
    pd.Series(ap).to_csv(res / f"{name}.csv")
    ref = pd.read_csv(f"{S.LQC}/dbl/eval/auprc_wide.csv", index_col=0)["scdblfinder"].reindex(list(ap))
    d = pd.Series(ap) - ref
    stu = pd.Series(D.STUDY)
    rng = np.random.default_rng(0)
    b1 = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    g = d.groupby(stu.reindex(d.index)).mean()
    b2 = [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]
    rep = dict(combo=a.combo, tag=a.tag, mean_auprc=round(float(np.mean(list(ap.values()))), 4),
               vs_scdblfinder=round(float(d.mean()), 4), ci95=[round(float(np.percentile(b1, 2.5)), 4), round(float(np.percentile(b1, 97.5)), 4)],
               ci95_study_clustered=[round(float(np.percentile(b2, 2.5)), 4), round(float(np.percentile(b2, 97.5)), 4)])
    (res / f"{name}.json").write_text(json.dumps(rep))
    print(json.dumps(rep))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--out", required=True)
    s.add_argument("--min-cells", type=int, default=50)
    s.add_argument("--cap", type=int, default=400)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--genes", type=int, default=2000)
    f.add_argument("--dim", type=int, default=64)
    t = sub.add_parser("feats")
    t.add_argument("--out", required=True)
    t.add_argument("--model", required=True)
    t.add_argument("--tag", required=True)
    t.add_argument("--sim", type=int, default=0, help="simulated doublets per cell (0 = 2)")
    t.add_argument("--names", nargs="*")
    e = sub.add_parser("eval")
    e.add_argument("--out", required=True)
    e.add_argument("--combo", required=True, choices=list(COMBOS))
    e.add_argument("--tag", default="")
    a = ap.parse_args()
    {"select": cmd_select, "fit": cmd_fit, "feats": cmd_feats, "eval": cmd_eval}[a.cmd](a)


if __name__ == "__main__":
    main()
