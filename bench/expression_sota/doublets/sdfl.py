# SPDX-License-Identifier: MIT
"""sdfl: an unsupervised artificial-doublet scorer in the style of scDblFinder (Germain et al. 2021), reimplemented from the
published description; no R, no labels. Writes $LQC/dbl/scores/<name>/sdfl<SUFFIX>.npy for the datasets of bench/latent_qc/doublets.

  python sdfl.py NAME [NAME ...]   (env SDFL_VARIANT = a|b|c, see below)

Steps: HVG(2000) -> lognorm -> PCA(20) on the observed cells -> k-means clusters -> artificial doublets from cluster pairs
(heterotypic pairs by default; variant c adds a third same-cluster) summed in raw counts and projected -> kNN over observed +
artificial in PC space -> features (artificial-neighbour fraction at k = 5, 10, 20, sqrt(N); library size and genes relative to
the neighbours; distance to the nearest artificial doublet) -> gradient-boosted classifier "artificial vs observed", two rounds
(round 2 drops the top-rate observed cells from the negatives), observed-cell scores out-of-fold.
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dbl_common import load, lognorm, hvg, put_score, Timer


def score(X, variant="a", seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    h = hvg(X, 2000)
    Xh = X[:, h].astype(np.float32)
    lib = np.asarray(X.sum(1)).ravel().astype(np.float64)
    ng = np.asarray((X > 0).sum(1)).ravel().astype(np.float64)
    Y = lognorm(X)[:, h]
    mu = np.asarray(Y.mean(0)).ravel()
    sd = np.sqrt(np.maximum(np.asarray(Y.multiply(Y).mean(0)).ravel() - mu ** 2, 1e-6))
    svd = TruncatedSVD(20, random_state=seed, n_iter=7)
    # scale genes (divide by sd) then SVD on the sparse matrix; centre in PC space with the mean projection
    D = sp.diags(1.0 / sd)
    Ys = sp.csr_matrix(Y @ D)
    svd.fit(Ys)
    m = np.asarray(Ys.mean(0)).ravel()
    off = svd.components_ @ m
    P = svd.transform(Ys) - off
    k_cl = int(os.environ.get('SDFL_KCL', 0)) or int(np.clip(round(np.sqrt(N) / 4), 4, 25))
    cl = MiniBatchKMeans(k_cl, random_state=seed, n_init=3, batch_size=4096).fit_predict(P)
    n_art = int(min(max(2 * N, 3000), 30000))
    # heterotypic pairs (a, b from different clusters); variant c: 1/3 homotypic pairs as well
    idx = {c: np.flatnonzero(cl == c) for c in np.unique(cl)}
    a = rng.integers(0, N, n_art)
    b = rng.integers(0, N, n_art)
    homo = (rng.random(n_art) < (1 / 3 if variant == "c" else 0.0))
    for i in np.flatnonzero(~homo):
        tries = 0
        while cl[a[i]] == cl[b[i]] and tries < 20:
            b[i] = rng.integers(0, N); tries += 1
    for i in np.flatnonzero(homo):
        b[i] = rng.choice(idx[cl[a[i]]])
    S = (X[a] + X[b]).tocsr()
    libA = lib[a] + lib[b]
    # depth-normalise like the observed cells: lognorm of the raw sum, same scaling and projection
    YA = sp.csr_matrix(lognorm(S)[:, h] @ D)
    PA = svd.transform(YA) - off
    Z = np.vstack([P, PA]).astype(np.float32)
    isart = np.r_[np.zeros(N, bool), np.ones(n_art, bool)]
    nn = NearestNeighbors(n_neighbors=51, algorithm="auto", n_jobs=-1).fit(Z)
    dist, ind = nn.kneighbors(Z)
    dist, ind = dist[:, 1:], ind[:, 1:]
    art_nb = isart[ind]
    feats = []
    r = n_art / N
    for k in (5, 10, 20, int(np.clip(round(0.5 * np.sqrt(N)), 5, 50))):
        f = art_nb[:, :k].mean(1)
        feats.append(np.log((f * 1.0 + 0.5 / k) / (1 - f + 0.5 / k)))
    libs = np.r_[lib, libA]
    ngs = np.r_[ng, np.asarray((S > 0).sum(1)).ravel()]
    obs_nb = np.where(~art_nb[:, :20], libs[ind[:, :20]], np.nan)
    med = np.nanmedian(np.where(np.isnan(obs_nb), np.nan, obs_nb), axis=1)
    med = np.where(np.isfinite(med), med, np.nanmedian(lib))
    feats += [np.log(libs / np.maximum(med, 1)), np.log(ngs + 1), np.log(libs + 1)]
    firstart = np.where(art_nb.any(1), np.argmax(art_nb, 1), 49)
    feats += [dist[np.arange(len(Z)), firstart], dist[:, 0], dist[:, 9]]
    F = np.column_stack(feats).astype(np.float32)
    F = np.nan_to_num(F)
    y = isart.astype(int)
    # out-of-fold classifier; artificial doublets and observed cells both folded so an observed cell never trains its own prediction
    rho = min(0.008 * N / 1000, 0.25)
    sc = np.zeros(N)
    for rnd in range(2):
        keep = np.ones(len(Z), bool)
        if rnd == 1:
            top = np.argsort(-sc)[: int(1.5 * rho * N)]
            keep[top] = False
        out = np.zeros(len(Z))
        skf = StratifiedKFold(3, shuffle=True, random_state=seed + rnd)
        for tr, te in skf.split(F, y):
            tr = tr[keep[tr]]
            clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=4, random_state=seed)
            clf.fit(F[tr], y[tr])
            out[te] = clf.predict_proba(F[te])[:, 1]
        sc = out[:N]
    return sc


def main(names):
    variant = os.environ.get("SDFL_VARIANT", "a")
    for n in names:
        d = load(n)
        with Timer() as t:
            s = score(d["X"], variant)
        put_score(n, "sdfl" + (variant if variant != "a" else "") + os.environ.get("SDFL_TAG", ""), s, t.s)


if __name__ == "__main__":
    main(sys.argv[1:])
