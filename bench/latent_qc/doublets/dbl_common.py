"""Shared I/O for the doublet track. One dataset = $LQC/dbl/data/<name>/ds.npz:

  X (cells x genes CSR, int counts: X_data/X_indices/X_indptr/X_shape), genes (Ensembl ids or symbols), cells,
  label (1 = doublet by experiment: genotype, hashtag, species), species ('human', 'mouse', 'mixed'),
  optional per-cell extras prefixed 'x_' (e.g. x_vireo_geno, x_prod_geno, x_dbl_type) and 'meta' (JSON string).

Scores go to $LQC/dbl/scores/<name>/<method>.npy (float per cell, higher = more doublet-like) with <method>.json holding
wall time and settings.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import scipy.sparse as sp

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
DATA = f"{LQC}/dbl/data"
SCORES = f"{LQC}/dbl/scores"


def load(name: str) -> dict:
    z = np.load(f"{DATA}/{name}/ds.npz", allow_pickle=False)
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    d = dict(X=X, genes=z["genes"].astype(str), cells=z["cells"].astype(str), label=z["label"].astype(np.int8),
             species=str(z["species"]), name=name)
    d["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    for k in z.files:
        if k.startswith("x_"):
            d[k] = z[k]
    return d


def save(name: str, X, genes, cells, label, species: str, meta: dict | None = None, **extras) -> None:
    X = sp.csr_matrix(X)
    X.sum_duplicates()
    X.sort_indices()
    os.makedirs(f"{DATA}/{name}", exist_ok=True)
    arrs = dict(X_data=X.data.astype(np.int32), X_indices=X.indices.astype(np.int32), X_indptr=X.indptr.astype(np.int64),
                X_shape=np.array(X.shape), genes=np.asarray(genes).astype(str), cells=np.asarray(cells).astype(str),
                label=np.asarray(label).astype(np.int8), species=np.array(species), meta=np.array(json.dumps(meta or {})))
    for k, v in extras.items():
        arrs[k if k.startswith("x_") else "x_" + k] = np.asarray(v)
    np.savez(f"{DATA}/{name}/ds.npz", **arrs)
    print(f"{name}: {X.shape[0]} cells x {X.shape[1]} genes, {int(np.sum(label))} doublets ({np.mean(label):.3f}), {species}",
          flush=True)


def put_score(name: str, method: str, score, seconds: float, **info) -> None:
    os.makedirs(f"{SCORES}/{name}", exist_ok=True)
    np.save(f"{SCORES}/{name}/{method}.npy", np.asarray(score, np.float64))
    json.dump(dict(method=method, seconds=round(seconds, 2), **info), open(f"{SCORES}/{name}/{method}.json", "w"), default=str)
    print(f"[{name}] {method}: {seconds:.1f}s", flush=True)


def has_score(name: str, method: str) -> bool:
    return os.path.exists(f"{SCORES}/{name}/{method}.npy")


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t


def simulate_doublets(X: sp.csr_matrix, n: int, rng, groups=None, same_group: bool | None = None):
    """Sum the raw counts of n random cell pairs (Scrublet / Solo). With ``groups`` and ``same_group`` True / False, pairs are
    drawn within / across groups (homotypic / heterotypic). Returns (CSR, parent index pairs)."""
    N = X.shape[0]
    if groups is None or same_group is None:
        a = rng.integers(0, N, n)
        b = rng.integers(0, N, n)
        b = np.where(a == b, (b + 1) % N, b)
    else:
        g = np.asarray(groups)
        idx = {k: np.flatnonzero(g == k) for k in np.unique(g)}
        a = rng.integers(0, N, 4 * n)
        if same_group:
            a = a[[len(idx[g[i]]) > 1 for i in a]][:n]
            b = np.array([rng.choice(idx[g[i]][idx[g[i]] != i]) for i in a])
        else:
            a = a[[len(idx[g[i]]) < N for i in a]][:n]
            other = {k: np.flatnonzero(g != k) for k in idx}
            b = np.array([rng.choice(other[g[i]]) for i in a])
    S = (X[a] + X[b]).tocsr()
    return S, np.stack([a, b], 1)


def lognorm(X: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    n = np.asarray(X.sum(1)).ravel()
    Y = sp.diags(target / np.maximum(n, 1)) @ X.astype(np.float32)
    Y = sp.csr_matrix(Y)
    Y.data = np.log1p(Y.data)
    return Y


def hvg(X: sp.csr_matrix, n_top: int = 2000) -> np.ndarray:
    """Top genes by variance of log-normalized expression relative to a mean trend (dispersion residual, binned)."""
    Y = lognorm(X)
    mu = np.asarray(Y.mean(0)).ravel()
    var = np.asarray(Y.multiply(Y).mean(0)).ravel() - mu ** 2
    ok = mu > 0
    bins = np.digitize(mu, np.quantile(mu[ok], np.linspace(0, 1, 21)[1:-1]))
    z = np.full(len(mu), -np.inf)
    for b in np.unique(bins[ok]):
        m = ok & (bins == b)
        v = var[m]
        z[m] = (v - np.median(v)) / (np.median(np.abs(v - np.median(v))) * 1.4826 + 1e-9)
    return np.argsort(-z)[: min(n_top, int(ok.sum()))]
