#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Second leakage screen (near-duplicate cells at dataset level): for each evaluation dataset, find the corpus datasets whose mean
profile is closest, then compare 1,000 of its cells with every cell of its 5 closest corpus datasets (cosine of log1p CP10k vectors
over the 6,000-gene vocabulary). A cell that is in the corpus (same cell, any pipeline) has a corpus neighbour at cosine ~ 1; the
control is its nearest neighbour among the other cells of its own dataset.   python scripts/doublet_leak2.py --out OUT"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))
from doublet_leak import CORPUS, project  # noqa: E402


def lognorm(X):
    X = sp.csr_matrix(X, dtype=np.float32)
    tot = np.asarray(X.sum(1)).ravel()
    Y = sp.csr_matrix(sp.diags(1e4 / np.maximum(tot, 1)) @ X)
    Y.data = np.log1p(Y.data)
    return Y


def unit(D):
    D = np.asarray(D, np.float32)
    return D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-9)


def main():
    import doublet_v2 as S
    from singlet.features.doublet import STUDY

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    obs = pd.read_parquet(f"{CORPUS}/corpus_obs.parquet").reset_index(drop=True)
    vocab = json.load(open(f"{CORPUS}/vocab.json"))
    X = sp.load_npz(f"{CORPUS}/corpus_counts.npz").tocsr()
    codes, uniq = pd.factorize(obs.dataset_id)
    G = len(uniq)
    sums = np.zeros((G, X.shape[1]), np.float64)
    for s in range(0, X.shape[0], 200000):
        Y = lognorm(X[s : s + 200000])
        M = sp.csr_matrix((np.ones(Y.shape[0]), (codes[s : s + 200000], np.arange(Y.shape[0]))), shape=(G, Y.shape[0]))
        sums += np.asarray((M @ Y).todense())
    n_ds = np.bincount(codes, minlength=G).astype(np.float64)
    means = sums / n_ds[:, None]
    glob = (sums.sum(0) / n_ds.sum())
    print("dataset means", G, f"{time.time() - t0:.0f}s", flush=True)
    title = obs.groupby(codes).dataset_title.first()
    rows_of = {g: np.flatnonzero(codes == g) for g in range(G)}
    res = {}
    rng = np.random.default_rng(0)
    for n in STUDY:
        Xe, y, genes, species = S.load_counts(n)
        V = sp.csr_matrix(sp.csr_matrix(Xe, dtype=np.float32) @ project(genes, vocab))
        Ye = lognorm(V)
        me = np.asarray(Ye.mean(0)).ravel()
        sim = unit(means - glob) @ unit((me - glob)[None, :]).ravel()
        top = np.argsort(-sim)[:5]
        cand = np.concatenate([rows_of[g][:60000] for g in top])
        C = unit(lognorm(X[cand]).toarray())
        sub = rng.choice(Ye.shape[0], min(1000, Ye.shape[0]), replace=False)
        Q = unit(Ye[sub].toarray())
        best = np.empty(len(sub))
        for s in range(0, len(sub), 250):
            best[s : s + 250] = (Q[s : s + 250] @ C.T).max(1)
        # control: nearest neighbour among the other cells of the same dataset
        pool = rng.choice(Ye.shape[0], min(20000, Ye.shape[0]), replace=False)
        P = unit(Ye[np.union1d(pool, sub)].toarray())
        ids = np.union1d(pool, sub)
        pos = {c: i for i, c in enumerate(ids)}
        within = np.empty(len(sub))
        for s in range(0, len(sub), 250):
            d = Q[s : s + 250] @ P.T
            for j, c in enumerate(sub[s : s + 250]):
                d[j, pos[c]] = -1
            within[s : s + 250] = d.max(1)
        res[n] = dict(closest_corpus_datasets=[dict(title=str(title[g])[:70], mean_profile_cosine=round(float(sim[g]), 3), cells=int(n_ds[g])) for g in top[:3]],
                      cells_compared=int(len(cand)), nn_cos_corpus_median=round(float(np.median(best)), 3), nn_cos_corpus_p99=round(float(np.percentile(best, 99)), 3),
                      nn_cos_corpus_max=round(float(best.max()), 4), n_ge_098=int((best >= 0.98).sum()), nn_cos_within_median=round(float(np.median(within)), 3),
                      frac_corpus_nn_above_within_nn=round(float((best > within).mean()), 3))
        print(n, res[n], flush=True)
        del C, P
    out = Path(a.out) / "leak"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report2.json").write_text(json.dumps(res, indent=1))
    print("max cosine to any corpus cell over all datasets:", max(r["nn_cos_corpus_max"] for r in res.values()), "cells >= 0.98:", sum(r["n_ge_098"] for r in res.values()), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
