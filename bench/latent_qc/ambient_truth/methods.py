"""Post-hoc ambient estimators on counts: latent-kNN / cluster mixture (new) and oracle-marker (upper bound for marker methods).

Model for every cell c (called cells only):  x_c ~ Multinomial(N_c, (1 - rho_c) q_c + rho_c a)
  a   ambient profile from the empty-droplet pool (atlib.DS.a)
  q_c native profile = decontaminated, pooled counts of the cell's neighbours (itself excluded), smoothed towards the global
      native profile with tau pseudo-UMIs;  rho_c by 1-D maximum likelihood (concave in rho), optional Beta prior shared by cells.
Iterated: neighbours' contamination (rho_j) is removed from their counts before pooling: q_c ~ (sum_j x_j - rho_j N_j a)_+.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def pca_latent(ds, n_hvg=2000, n_pc=20, seed=0):
    X = ds.C.T.tocsr().astype(np.float32)  # cells x genes
    N = np.asarray(X.sum(1)).ravel()
    L = X.multiply(1e4 / np.maximum(N, 1)[:, None]).tocsr()
    L.data = np.log1p(L.data)
    mean = np.asarray(L.mean(0)).ravel()
    var = np.asarray(L.multiply(L).mean(0)).ravel() - mean ** 2
    hv = np.argsort(-var)[:n_hvg]
    D = L[:, hv].toarray()
    D = (D - D.mean(0)) / np.maximum(D.std(0), 1e-3)
    D = np.clip(D, -10, 10)
    Z = TruncatedSVD(n_pc, algorithm="randomized", random_state=seed).fit_transform(D)
    return Z


def knn_weights(Z, k):
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)
    rows, cols = [], []
    for i in range(n):
        j = idx[i][idx[i] != i][:k]
        rows.append(np.full(len(j), i)); cols.append(j)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))


def cluster_weights(Z, seed=0):
    n = Z.shape[0]
    K = int(np.clip(round(np.sqrt(n / 20)), 2, 24))
    lab = KMeans(K, n_init=3, random_state=seed).fit_predict(Z)
    G = sp.csr_matrix((np.ones(n, np.float32), (np.arange(n), lab)), shape=(n, K))
    W = (G @ G.T).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    return W


def _solve_rho(rows, a_v, q_v, x, n, mu, kappa, lo=1e-6, hi=0.6, it=40):
    """Per-row max of sum x log(q + rho (a - q)) + Beta prior; g' is decreasing in rho -> safeguarded bisection/Newton."""
    d = a_v - q_v
    r = np.full(n, 0.01)
    lo_ = np.full(n, lo); hi_ = np.full(n, hi)
    al = max(kappa * mu, 1e-9); be = max(kappa * (1 - mu), 1e-9)

    def grad(rr):
        den = q_v + rr[rows] * d
        g = np.bincount(rows, weights=x * d / den, minlength=n)
        g2 = -np.bincount(rows, weights=x * d * d / den ** 2, minlength=n)
        if kappa > 0:
            g = g + (al - 1) / rr - (be - 1) / (1 - rr)
            g2 = g2 - (al - 1) / rr ** 2 - (be - 1) / (1 - rr) ** 2
        return g, g2

    for _ in range(it):
        g, g2 = grad(r)
        lo_ = np.where(g > 0, r, lo_); hi_ = np.where(g <= 0, r, hi_)
        nr = r - g / np.minimum(g2, -1e-12)
        bad = ~((nr > lo_) & (nr < hi_)) | ~np.isfinite(nr)
        nr = np.where(bad, 0.5 * (lo_ + hi_), nr)
        if np.max(np.abs(nr - r) / r) < 1e-4:
            r = nr
            break
        r = nr
    return r


def mixture(ds, W, tau=10.0, iters=3, kappa=0.0, block=400):
    """Returns rho, removed UMIs per cell from mouse genes / human genes (expected, by posterior ambient responsibility)."""
    n = ds.C.shape[1]
    Xt = ds.C.T.tocsr().astype(np.float32)  # cells x genes
    Nc = ds.N.astype(np.float64)
    a = ds.a.astype(np.float32)
    S = (W @ Xt).tocsr()  # neighbours' summed counts
    mouse = ds.gene_mouse
    rho = np.zeros(n)
    mu = 0.01
    for t in range(iters + 1):
        R = np.asarray(W @ (rho * Nc)).ravel()
        # global native profile
        tot_clip = np.asarray(Xt.sum(0)).ravel() - (rho * Nc).sum() * a
        g = np.maximum(tot_clip, 0); g = (g + 1e-6) / (g + 1e-6).sum()
        g = g.astype(np.float32)
        new_rho = np.zeros(n); rm = np.zeros(n); rh = np.zeros(n)
        for s in range(0, n, block):
            e = min(n, s + block)
            Sb = S[s:e].toarray()
            Sb -= R[s:e, None].astype(np.float32) * a[None, :]
            np.maximum(Sb, 0, out=Sb)
            Z = Sb.sum(1)
            q = (Sb + tau * g[None, :]) / (Z[:, None] + tau)
            Xb = Xt[s:e]
            rows = np.repeat(np.arange(e - s), np.diff(Xb.indptr))
            cols = Xb.indices
            x = Xb.data.astype(np.float64)
            qv = q[rows, cols].astype(np.float64)
            av = a[cols].astype(np.float64)
            r = _solve_rho(rows, av, qv, x, e - s, mu, kappa if t > 0 else 0.0)
            new_rho[s:e] = r
            post = r[rows] * av / ((1 - r[rows]) * qv + r[rows] * av)
            rem = x * post
            rm[s:e] = np.bincount(rows, weights=rem * mouse[cols], minlength=e - s)
            rh[s:e] = np.bincount(rows, weights=rem * ~mouse[cols], minlength=e - s)
        rho = new_rho
        mu = float(np.median(rho))
    return rho, rm, rh


def latent_knn(ds, latent="pca", mode="knn", k=30, tau=10.0, iters=3, kappa=0.0, Z=None):
    if Z is None:
        Z = pca_latent(ds)
    W = cluster_weights(Z) if mode == "cluster" else knn_weights(Z, k)
    return mixture(ds, W, tau=tau, iters=iters, kappa=kappa)


def marker_oracle(ds, m):
    """Oracle species markers: for a cell of species s, the m most abundant ambient genes of the OTHER species.
    rho_hat = reads in the marker set / (N_c * ambient mass of the marker set).  m=None -> all other-species genes."""
    est = np.full(ds.C.shape[1], np.nan)
    Xr = ds.C.tocsr()
    for sp_name, other in (("human", True), ("mouse", False)):
        cells = np.where(ds.species == sp_name)[0]
        genes = np.where(ds.gene_mouse == other)[0]
        if m is not None:
            genes = genes[np.argsort(-ds.a[genes])[:m]]
        reads = np.asarray(Xr[genes].sum(0)).ravel()[cells]
        est[cells] = reads / ds.N[cells] / ds.a[genes].sum()
    return est
