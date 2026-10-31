# SPDX-License-Identifier: MIT
"""sdfl / sdflc: R-free artificial-doublet scorers in the style of scDblFinder (Germain et al. 2021), reimplemented from the
published description (port of bench/expression_sota/doublets/sdfl.py, branch claude/expression-sota-31cf6d; no labels, no R).

HVG(2000) -> lognorm -> PCA(20) -> k-means clusters -> artificial doublets from cluster pairs (variant ``a``: heterotypic pairs;
variant ``c``: one third homotypic pairs) summed in raw counts and projected -> kNN over observed + artificial cells in PC space ->
features (artificial-neighbour fraction at k = 5, 10, 20, sqrt(N); library size and genes relative to the neighbours; distance to
the nearest artificial doublet) -> gradient-boosted classifier "artificial vs observed" in two rounds (round 2 drops the top-rate
observed cells from the negatives), observed-cell scores out of fold.

Both variants share the HVG / PCA / clustering prefix; each continues from its own copy of the random stream, so the values equal
those of running the reference script once per variant."""

from __future__ import annotations

import copy
from typing import Dict, Sequence

import numpy as np
import scipy.sparse as sp

from .doublet_util import hvg, lognorm


def score(
    X: sp.csr_matrix, variants: Sequence[str] = ("a", "c"), seed: int = 0
) -> Dict[str, np.ndarray]:
    """``{"sdfl": ..., "sdflc": ...}`` (variant ``a`` is named ``sdfl``), one score per row of ``X`` (raw counts)."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neighbors import NearestNeighbors

    X = sp.csr_matrix(X)
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    h = hvg(X, 2000)
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    ng = np.asarray((X > 0).sum(1)).ravel().astype(np.float64)
    Y = lognorm(X)[:, h]
    mu = np.asarray(Y.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Y.multiply(Y).mean(0)).ravel() - mu**2, 1e-6))
    svd = TruncatedSVD(20, random_state=seed, n_iter=7)
    D = sp.diags(1.0 / sd)
    Ys = sp.csr_matrix(Y @ D)
    svd.fit(Ys)
    off = svd.components_ @ np.asarray(Ys.mean(0)).ravel()
    P = svd.transform(Ys) - off
    k_cl = int(np.clip(round(np.sqrt(N) / 4), 4, 25))
    cl = MiniBatchKMeans(k_cl, random_state=seed, n_init=3, batch_size=4096).fit_predict(P)
    n_art = int(min(max(2 * N, 3000), 30000))
    idx = {c: np.flatnonzero(cl == c) for c in np.unique(cl)}
    a0 = rng.integers(0, N, n_art)
    b0 = rng.integers(0, N, n_art)
    u = rng.random(n_art)
    rng_after = copy.deepcopy(rng)
    out: Dict[str, np.ndarray] = {}
    for variant in variants:
        r = copy.deepcopy(rng_after)
        a, b = a0.copy(), b0.copy()
        homo = u < (1 / 3 if variant == "c" else 0.0)
        for i in np.flatnonzero(~homo):
            tries = 0
            while cl[a[i]] == cl[b[i]] and tries < 20:
                b[i] = r.integers(0, N)
                tries += 1
        for i in np.flatnonzero(homo):
            b[i] = r.choice(idx[cl[a[i]]])
        S = (X[a] + X[b]).tocsr()
        libA = lib[a] + lib[b]
        YA = sp.csr_matrix(lognorm(S)[:, h] @ D)
        PA = svd.transform(YA) - off
        Z = np.vstack([P, PA]).astype(np.float32)
        isart = np.r_[np.zeros(N, bool), np.ones(n_art, bool)]
        nn = NearestNeighbors(n_neighbors=51, algorithm="auto", n_jobs=-1).fit(Z)
        dist, ind = nn.kneighbors(Z)
        dist, ind = dist[:, 1:], ind[:, 1:]
        art_nb = isart[ind]
        feats = []
        for k in (5, 10, 20, int(np.clip(round(0.5 * np.sqrt(N)), 5, 50))):
            f = art_nb[:, :k].mean(1)
            feats.append(np.log((f * 1.0 + 0.5 / k) / (1 - f + 0.5 / k)))
        libs = np.r_[lib, libA]
        ngs = np.r_[ng, np.asarray((S > 0).sum(1)).ravel()]
        obs_nb = np.where(~art_nb[:, :20], libs[ind[:, :20]], np.nan)
        med = np.nanmedian(obs_nb, axis=1)
        med = np.where(np.isfinite(med), med, np.nanmedian(lib))
        feats += [np.log(libs / np.maximum(med, 1)), np.log(ngs + 1), np.log(libs + 1)]
        firstart = np.where(art_nb.any(1), np.argmax(art_nb, 1), 49)
        feats += [dist[np.arange(len(Z)), firstart], dist[:, 0], dist[:, 9]]
        F = np.nan_to_num(np.column_stack(feats).astype(np.float32))
        y = isart.astype(int)
        rho = min(0.008 * N / 1000, 0.25)
        sc = np.zeros(N)
        for rnd in range(2):
            keep = np.ones(len(Z), bool)
            if rnd == 1:
                keep[np.argsort(-sc)[: int(1.5 * rho * N)]] = False
            res = np.zeros(len(Z))
            for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed + rnd).split(F, y):
                tr = tr[keep[tr]]
                clf = HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.1, max_depth=4, random_state=seed
                )
                clf.fit(F[tr], y[tr])
                res[te] = clf.predict_proba(F[te])[:, 1]
            sc = res[:N]
        out["sdfl" + ("" if variant == "a" else variant)] = sc
    return out
