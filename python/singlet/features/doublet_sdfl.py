# SPDX-License-Identifier: MIT
"""sdfl / sdflc: R-free artificial-doublet scorers in the style of scDblFinder (Germain et al. 2021), reimplemented from the
published description; no labels, no R. Fast, linear-in-cells variant of bench/expression_sota/doublets/sdfl.py (branch
claude/expression-sota-31cf6d).

The fit uses at most ``cap`` cells (a random subsample when the sample is larger); every cell is then scored against that fixed
reference, so the cost is O(n) apart from fixed per-sample terms:

1. HVG (2,000) and PCA (20) from the fitting cells; k-means clusters of the fitting cells;
2. artificial doublets (``art_ratio`` x fitting cells) from random pairs of different clusters (variant ``a``) or with one third
   same-cluster pairs (variant ``c``), summed in raw counts and projected;
3. kNN of every cell in PC space against the fixed reference (fitting cells + artificial doublets); features: artificial-neighbour
   fraction at k = 5, 10, 20 and sqrt(n_ref)/2, library size and detected genes relative to the observed neighbours, distance to the
   nearest artificial doublet;
4. gradient-boosted classifier "artificial vs observed" in ``folds`` folds (``rounds`` rounds; round 2 drops the top-rate observed
   cells from the negatives). Fitting cells get their out-of-fold prediction, other cells the mean of the fold models'."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import scipy.sparse as sp

from .doublet_util import hvg, lognorm


@dataclass(frozen=True)
class Params:
    cap: int = 6000  # fitting / reference cells
    art_ratio: float = 1.0  # artificial doublets per fitting cell
    art_min: int = 2000
    n_hvg: int = 2000
    n_pc: int = 20
    folds: int = 2
    rounds: int = 1
    max_iter: int = 60
    learning_rate: float = 0.15
    max_depth: int = 4
    kmax: int = 50


FAST = Params()


def _pairs(rng, cl, N, n_art, homo_frac):
    """Random cell pairs (indices into the fitting cells): different clusters, except ``homo_frac`` same-cluster pairs."""
    a = rng.integers(0, N, n_art)
    b = rng.integers(0, N, n_art)
    homo = rng.random(n_art) < homo_frac
    for _ in range(20):
        bad = (~homo) & (cl[a] == cl[b])
        if not bad.any():
            break
        b[bad] = rng.integers(0, N, int(bad.sum()))
    for c in np.unique(cl[a[homo]]) if homo.any() else []:
        m = homo & (cl[a] == c)
        members = np.flatnonzero(cl == c)
        b[m] = members[rng.integers(0, len(members), int(m.sum()))]
    return a, b


def score(
    X: sp.csr_matrix,
    variants: Sequence[str] = ("a", "c"),
    seed: int = 0,
    p: Params = FAST,
) -> Dict[str, np.ndarray]:
    """``{"sdfl": ..., "sdflc": ...}`` (variant ``a`` is named ``sdfl``), one score per row of ``X`` (raw counts)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.neighbors import NearestNeighbors

    X = sp.csr_matrix(X, dtype=np.float32)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    sub = np.sort(rng.choice(N, p.cap, replace=False)) if N > p.cap else np.arange(N)
    rest = np.setdiff1d(np.arange(N), sub)
    Ns = len(sub)
    Xs = X[sub]
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    ng = np.diff(X.indptr).astype(np.float64)
    h = hvg(Xs, p.n_hvg)
    Ys = lognorm(Xs)[:, h]
    mu = np.asarray(Ys.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Ys.multiply(Ys).mean(0)).ravel() - mu**2, 1e-6))
    D = sp.diags(1.0 / sd)
    Ysc = sp.csr_matrix(Ys @ D)
    svd = TruncatedSVD(p.n_pc, random_state=seed, n_iter=5).fit(Ysc)
    off = svd.components_ @ np.asarray(Ysc.mean(0)).ravel()

    def project(M):
        return svd.transform(sp.csr_matrix(lognorm(M)[:, h] @ D)) - off

    P_sub = project(Xs).astype(np.float32)
    P_rest = project(X[rest]).astype(np.float32) if len(rest) else np.zeros((0, p.n_pc), np.float32)
    k_cl = int(np.clip(round(np.sqrt(Ns) / 4), 4, 25))
    cl = MiniBatchKMeans(k_cl, random_state=seed, n_init=3, batch_size=4096).fit_predict(P_sub)
    n_art = int(max(p.art_min, p.art_ratio * Ns))
    ks = (5, 10, 20, int(np.clip(round(0.5 * np.sqrt(Ns + n_art)), 5, p.kmax)))
    out: Dict[str, np.ndarray] = {}
    for variant in variants:
        r = np.random.default_rng(seed + 1)
        a, b = _pairs(r, cl, Ns, n_art, 1 / 3 if variant == "c" else 0.0)
        S = (Xs[a] + Xs[b]).tocsr()
        lib_a = lib[sub][a] + lib[sub][b]
        ng_a = np.diff(S.indptr).astype(np.float64)
        PA = project(S).astype(np.float32)
        Zref = np.vstack([P_sub, PA])
        isart = np.r_[np.zeros(Ns, bool), np.ones(n_art, bool)]
        libs_ref = np.r_[lib[sub], lib_a]
        nn = NearestNeighbors(n_neighbors=p.kmax + 1, n_jobs=-1).fit(Zref)

        def feats(Zq, lib_q, ng_q, self_in_ref):
            dist, ind = nn.kneighbors(Zq)
            if self_in_ref:
                dist, ind = dist[:, 1:], ind[:, 1:]
            else:
                dist, ind = dist[:, : p.kmax], ind[:, : p.kmax]
            art_nb = isart[ind]
            f = []
            for k in ks:
                fr = art_nb[:, :k].mean(1)
                f.append(np.log((fr + 0.5 / k) / (1 - fr + 0.5 / k)))
            obs_nb = np.where(~art_nb[:, :20], libs_ref[ind[:, :20]], np.nan)
            with np.errstate(all="ignore"):
                med = np.nanmedian(obs_nb, axis=1)
            med = np.where(np.isfinite(med), med, np.median(lib[sub]))
            f += [np.log(lib_q / np.maximum(med, 1)), np.log(ng_q + 1), np.log(lib_q + 1)]
            first = np.where(art_nb.any(1), np.argmax(art_nb, 1), p.kmax - 1)
            f += [dist[np.arange(len(Zq)), first], dist[:, 0], dist[:, 9]]
            return np.nan_to_num(np.column_stack(f).astype(np.float32))

        F_ref = feats(Zref, libs_ref, np.r_[ng[sub], ng_a], True)
        F_rest = feats(P_rest, lib[rest], ng[rest], False) if len(rest) else np.zeros((0, F_ref.shape[1]), np.float32)
        y = isart.astype(int)
        rho = min(0.008 * Ns / 1000, 0.25)
        fold = np.random.default_rng(seed + 2).integers(0, p.folds, len(y))
        sc_sub = np.zeros(Ns)
        sc_rest = np.zeros(len(rest))
        for rnd in range(p.rounds):
            keep = np.ones(len(y), bool)
            if rnd == 1:
                keep[np.argsort(-sc_sub)[: int(1.5 * rho * Ns)]] = False
            oof = np.zeros(len(y))
            pr = np.zeros(len(rest))
            for f_ in range(p.folds):
                tr = np.flatnonzero((fold != f_) & keep)
                clf = HistGradientBoostingClassifier(
                    max_iter=p.max_iter, learning_rate=p.learning_rate, max_depth=p.max_depth, random_state=seed
                ).fit(F_ref[tr], y[tr])
                te = fold == f_
                oof[te] = clf.predict_proba(F_ref[te])[:, 1]
                if len(rest):
                    pr += clf.predict_proba(F_rest)[:, 1] / p.folds
            sc_sub, sc_rest = oof[:Ns], pr
        sc = np.zeros(N)
        sc[sub] = sc_sub
        sc[rest] = sc_rest
        out["sdfl" + ("" if variant == "a" else variant)] = sc
    return out
