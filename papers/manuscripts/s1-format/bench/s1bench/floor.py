"""Stage `floor` (Q7, Q8): ideal code lengths of the count matrix under count models, cross-validated over cells.

Every model is fitted on one half of the cells and charged -log2 p for the other half (zeros included), then the halves
swap. Results are bits per non-zero of the test half, averaged over the two folds. No model is charged for its own
description, so each number is a lower bound for a decoder that would have to ship that model.

Models
  poisson     x_gc ~ Poisson(s_c mu_g)
  nb          x_gc ~ NB(mean s_c mu_g, dispersion theta_g), theta_g by moments, smoothed over mean expression
  gene_depth  presence per (gene, depth bin), value per (gene, coarse depth bin); KT-smoothed counts. The
              nonparametric model closest to what .s1 does with shared tables.
  cluster     as gene_depth with every context also split by the cell's cluster (Leiden on the training half; test
              cells take the nearest training centroid), backing off to gene_depth for rare (gene, cluster, depth)
              combinations. The cluster label itself is charged at its empirical entropy.
  sampling    H(Multinomial(N_c, pi_c)) for a random sample of test cells, pi_c = the cell's counts shrunk toward its
              cluster profile with a strength fitted on held-out UMIs. The part of the code length no decoder can avoid.
The decomposition for Fig 3c is computed on the sampled cells, where all three nonparametric quantities are per cell.
"""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln, polygamma

from .util import Canon, Config, Unit, load_canon, log, read_tsv, result_path, write_tsv

LN2 = math.log(2.0)


# ----------------------------------------------------------------------------------------------- helpers
def _cols(c: Canon, cells: np.ndarray):
    """Sub-matrix of the given cells as (indptr, genes, values, owner) with owner = position in `cells`."""
    x = c.to_scipy()[:, cells]
    x = sp.csc_matrix(x)
    owner = np.repeat(np.arange(len(cells)), np.diff(x.indptr))
    return x, x.indices.astype(np.int64), x.data.astype(np.int64), owner


def _depth_bins(train_totals: np.ndarray, totals: np.ndarray, n_bins: int) -> np.ndarray:
    edges = np.unique(np.quantile(train_totals, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return np.searchsorted(edges, totals, side="right")


def _keys_hist(keys: np.ndarray):
    u, n = np.unique(keys, return_counts=True)
    return u, n


def _lookup(u: np.ndarray, n: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Counts of `keys` in the histogram (u, n); 0 where absent."""
    if len(u) == 0:
        return np.zeros(len(keys), dtype=np.int64)
    i = np.searchsorted(u, keys)
    i = np.minimum(i, len(u) - 1)
    return np.where(u[i] == keys, n[i], 0)


def _escape_bits(v: np.ndarray, cap: int) -> np.ndarray:
    e = (v - cap + 1).astype(np.float64)
    return 2.0 * np.floor(np.log2(np.maximum(e, 1))) + 1.0


# ----------------------------------------------------------------------------------------------- nonparametric
class Presence:
    """p(gene present | context) = (k + beta * prior) / (n + beta), from training counts k of cells in the context with the
    gene non-zero and n cells in the context. Without a parent the prior is 1/2 with beta = 1 (Krichevsky-Trofimov); with
    a parent the prior is the parent's probability for the parent context, so a fine context backs off to a coarse one.
    Sparse: only (gene, context) pairs seen in training are stored."""

    def __init__(self, G, g_tr, ctx_cell_tr, own_tr, n_ctx, beta=1.0, parent=None, parent_of=None):
        self.G, self.n_ctx, self.beta, self.parent, self.parent_of = G, n_ctx, float(beta), parent, parent_of
        self.ncell = np.bincount(ctx_cell_tr, minlength=n_ctx).astype(np.float64)
        self.k_u, self.k_n = _keys_hist(g_tr * n_ctx + ctx_cell_tr[own_tr])

    def prior(self, g, c):
        return 0.5 if self.parent is None else self.parent.p(g, self.parent_of[c])

    def p(self, g, c):
        k = _lookup(self.k_u, self.k_n, g * self.n_ctx + c).astype(np.float64)
        return (k + self.beta * self.prior(g, c)) / (self.ncell[c] + self.beta)

    def dense(self) -> np.ndarray:
        """G x n_ctx probabilities (only for small n_ctx: used as a parent's prior)."""
        g = np.repeat(np.arange(self.G), self.n_ctx)
        c = np.tile(np.arange(self.n_ctx), self.G)
        return self.p(g, c).reshape(self.G, self.n_ctx)

    def zero_sums(self) -> np.ndarray:
        """Z[c] = sum over all genes of -log2(1 - p(g, c)): the cost of a cell in context c in which no gene is present."""
        if self.parent is None:
            unseen = self.G * -np.log2(1.0 - 0.5 * self.beta / (self.ncell + self.beta))
        else:
            pd = self.parent.dense()
            unseen = np.array([np.sum(-np.log2(1.0 - self.beta * pd[:, self.parent_of[c]] / (self.ncell[c] + self.beta)))
                               for c in range(self.n_ctx)])
        g, c = self.k_u // self.n_ctx, self.k_u % self.n_ctx
        p_seen = (self.k_n + self.beta * self.prior(g, c)) / (self.ncell[c] + self.beta)
        p_unseen = self.beta * self.prior(g, c) / (self.ncell[c] + self.beta)
        delta = -np.log2(1.0 - p_seen) + np.log2(1.0 - p_unseen)
        return unseen + np.bincount(c, weights=delta, minlength=self.n_ctx)


class Values:
    """q(value | gene, context) over 1..cap-1 and an escape symbol (values >= cap, Elias-gamma tail), with the same
    back-off as Presence: KT (prior 1/cap, beta cap/2) without a parent, the parent's q otherwise."""

    def __init__(self, cap, g_tr, v_tr, ctx_cell_tr, own_tr, n_ctx, beta=None, parent=None, parent_of=None):
        self.cap, self.n_ctx, self.parent, self.parent_of = cap, n_ctx, parent, parent_of
        self.beta = 0.5 * cap if beta is None else float(beta)
        key = g_tr * n_ctx + ctx_cell_tr[own_tr]
        self.h_u, self.h_n = _keys_hist(key * (cap + 1) + np.minimum(v_tr, cap))
        self.t_u, self.t_n = _keys_hist(key)

    def q(self, g, c, v):
        vv = np.minimum(v, self.cap)
        key = g * self.n_ctx + c
        h = _lookup(self.h_u, self.h_n, key * (self.cap + 1) + vv).astype(np.float64)
        t = _lookup(self.t_u, self.t_n, key).astype(np.float64)
        prior = 1.0 / self.cap if self.parent is None else self.parent.q(g, self.parent_of[c], v)
        return (h + self.beta * prior) / (t + self.beta)


def cell_costs(G: int, te: tuple, pres: Presence, ctx_te: np.ndarray, vals: Values, vctx_te: np.ndarray, cap: int) -> np.ndarray:
    """Bits for every test cell: presence of all G genes, then the value of each non-zero."""
    _, g_te, v_te, own_te = te
    n = len(ctx_te)
    c_nz = ctx_te[own_te]
    p1 = pres.p(g_te, c_nz)
    cell = pres.zero_sums()[ctx_te]
    cell += np.bincount(own_te, weights=-np.log2(p1) + np.log2(1.0 - p1), minlength=n)
    q = vals.q(g_te, vctx_te[own_te], v_te)
    vbits = -np.log2(q) + np.where(v_te >= cap, _escape_bits(v_te, cap), 0.0)
    return cell + np.bincount(own_te, weights=vbits, minlength=n)


# ----------------------------------------------------------------------------------------------- parametric
def _size_factors(totals: np.ndarray, ref_mean: float) -> np.ndarray:
    return np.maximum(totals, 1) / ref_mean


def poisson_cost(xtr, xte, s_tr, s_te) -> float:
    mu = (np.asarray(xtr.sum(axis=1)).ravel() + 0.5) / (s_tr.sum() + 1.0)
    lam_nz = s_te[np.repeat(np.arange(xte.shape[1]), np.diff(xte.indptr))] * mu[xte.indices]
    x = xte.data.astype(np.float64)
    nats = s_te.sum() * mu.sum() + np.sum(-x * np.log(lam_nz) + gammaln(x + 1))
    return nats / LN2


def _nb_theta(xtr, s_tr, mu) -> np.ndarray:
    xs = xtr.multiply(1.0).tocsr()
    sx = np.asarray(xs @ s_tr).ravel()                  # sum_c s_c x_gc
    x2 = np.asarray(xs.multiply(xs).sum(axis=1)).ravel()
    ss = float((s_tr ** 2).sum())
    resid2 = x2 - 2 * mu * sx + mu ** 2 * ss            # sum_c (x - s mu)^2
    excess = resid2 - mu * s_tr.sum()
    theta = np.where(excess > 0, (mu ** 2) * ss / np.maximum(excess, 1e-12), 1e6)
    theta = np.clip(theta, 1e-3, 1e6)
    # smooth log theta over log mu (binned medians, linear interpolation), as sctransform regularises
    lm = np.log10(np.maximum(mu, 1e-12))
    ok = mu > 0
    if ok.sum() > 100:
        edges = np.quantile(lm[ok], np.linspace(0, 1, 51))
        mids, meds = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sel = ok & (lm >= a) & (lm <= b)
            if sel.sum() >= 5:
                mids.append(np.median(lm[sel]))
                meds.append(np.median(np.log10(theta[sel])))
        theta = 10 ** np.interp(lm, mids, meds)
    return theta


def nb_cost(xtr, xte, s_tr, s_te, n_bins: int, exact: bool = False) -> float:
    mu = (np.asarray(xtr.sum(axis=1)).ravel() + 0.5) / (s_tr.sum() + 1.0)
    th = _nb_theta(xtr, s_tr, mu)
    col = np.repeat(np.arange(xte.shape[1]), np.diff(xte.indptr))
    g = xte.indices
    lam = s_te[col] * mu[g]
    t = th[g]
    x = xte.data.astype(np.float64)
    logp_nz = gammaln(x + t) - gammaln(t) - gammaln(x + 1) + t * np.log(t / (t + lam)) + x * np.log(lam / (t + lam))
    log0_nz = t * np.log(t / (t + lam))
    if exact:
        L = s_te[None, :] * mu[:, None]
        zero_all = float(np.sum(th[:, None] * np.log(th[:, None] / (th[:, None] + L))))
    else:
        b = _depth_bins(s_tr, s_te, n_bins)
        nb_ = np.bincount(b, minlength=n_bins)
        sb = np.bincount(b, weights=s_te, minlength=n_bins) / np.maximum(nb_, 1)
        L = sb[None, :] * mu[:, None]
        zero_all = float(np.sum(nb_[None, :] * th[:, None] * np.log(th[:, None] / (th[:, None] + L))))
    nats = -(zero_all - log0_nz.sum() + logp_nz.sum())
    return nats / LN2


# ----------------------------------------------------------------------------------------------- clusters
def _lognorm(x):
    tot = np.asarray(x.sum(axis=0)).ravel()
    y = sp.csc_matrix(x.multiply(1e4 / np.maximum(tot, 1)))
    y.data = np.log1p(y.data)
    return y


def fit_clusters(xtr, cfg_floor: dict, seed: int):
    """Leiden on (a subsample of) the training cells; returns a function assigning any cells to a cluster id."""
    import anndata as ad
    import scanpy as sc

    rng = np.random.default_rng(seed)
    n = xtr.shape[1]
    cap = int(cfg_floor.get("max_cluster_cells", 200_000))
    sub = np.sort(rng.choice(n, size=min(n, cap), replace=False))
    a = ad.AnnData(X=sp.csr_matrix(xtr[:, sub].T.astype(np.float32)))
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=min(2000, a.n_vars))
    hv = np.flatnonzero(a.var["highly_variable"].to_numpy())
    npc = min(int(cfg_floor["n_pcs"]), len(hv) - 1, a.n_obs - 1)
    sc.pp.pca(a, n_comps=npc, mask_var="highly_variable", random_state=seed)
    sc.pp.neighbors(a, random_state=seed)
    sc.tl.leiden(a, resolution=float(cfg_floor["leiden_resolution"]), random_state=seed, flavor="igraph", n_iterations=2)
    lab = a.obs["leiden"].astype(int).to_numpy()
    load = a.varm["PCs"][hv]                                  # hv genes x pcs
    mean = np.asarray(a.X[:, hv].mean(axis=0)).ravel()
    pcs = a.obsm["X_pca"]
    K = lab.max() + 1
    cent = np.stack([pcs[lab == k].mean(axis=0) for k in range(K)])

    def assign(x) -> np.ndarray:
        out = np.empty(x.shape[1], dtype=np.int64)
        for s in range(0, x.shape[1], 50_000):
            y = _lognorm(x[:, s:s + 50_000])[hv, :].T.toarray() - mean
            z = y @ load
            d = ((z[:, None, :] - cent[None, :, :]) ** 2).sum(-1)
            out[s:s + 50_000] = d.argmin(1)
        return out

    return assign, K


# ----------------------------------------------------------------------------------------------- sampling term
def _elogfact_binom(N: float, p: np.ndarray) -> float:
    """sum_g E[log y_g!] for y_g ~ Binomial(N, p_g): exact pmf sums for small means, a second-order delta method above."""
    from scipy.stats import binom

    lam = N * p
    total = 0.0
    big = lam >= 30
    if big.any():
        m = lam[big]
        v = m * (1 - p[big])
        total += float(np.sum(gammaln(m + 1) + 0.5 * v * polygamma(1, m + 1)))
    small = (~big) & (lam > 1e-10)
    if small.any():
        ps = p[small]
        kmax = int(np.ceil(lam[small].max() + 8 * np.sqrt(lam[small].max()) + 10))
        for k in range(2, kmax + 1):
            total += float(np.sum(binom.pmf(k, N, ps)) * gammaln(k + 1))
    return total


def multinomial_entropy_bits(N: int, p: np.ndarray) -> float:
    p = p[p > 0]
    return (-gammaln(N + 1) - N * float(np.sum(p * np.log(p))) + _elogfact_binom(N, p)) / LN2


def fit_shrinkage(xs, prof, rng) -> float:
    """Shrinkage strength alpha for pi_c = (x_c + alpha prof_k) / (N_c + alpha), by held-out UMIs (binomial thinning)."""
    best, best_ll = None, -np.inf
    a_half = xs.copy()
    a_half.data = rng.binomial(xs.data.astype(np.int64), 0.5)
    b_half = xs - a_half
    b_half = sp.csc_matrix(b_half)
    a_half = sp.csc_matrix(a_half)
    Na = np.asarray(a_half.sum(axis=0)).ravel()
    for alpha in (3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5, 3e5, 1e6):
        ll = 0.0
        for j in range(xs.shape[1]):
            sa, ea = a_half.indptr[j], a_half.indptr[j + 1]
            sb, eb = b_half.indptr[j], b_half.indptr[j + 1]
            gb, vb = b_half.indices[sb:eb], b_half.data[sb:eb]
            num = alpha * prof[j][gb]
            ga, va = a_half.indices[sa:ea], a_half.data[sa:ea]
            own = dict(zip(ga.tolist(), va.tolist()))
            num = num + np.array([own.get(g, 0) for g in gb.tolist()], dtype=np.float64)
            pi = num / (Na[j] + alpha)
            ll += float(np.sum(vb * np.log(np.maximum(pi, 1e-300))))
        if ll > best_ll:
            best, best_ll = alpha, ll
    return float(best)


# ----------------------------------------------------------------------------------------------- driver
def _fold(c: Canon, tr: np.ndarray, te: np.ndarray, cfg_floor: dict, seed: int, want_sampling: bool) -> dict:
    G = c.m
    B = int(cfg_floor["depth_bins"])
    VB = int(cfg_floor.get("value_depth_bins", 4))
    cap = int(cfg_floor["value_cap"])
    TR, TE = _cols(c, tr), _cols(c, te)
    xtr, xte = TR[0], TE[0]
    tot_tr = np.asarray(xtr.sum(axis=0)).ravel()
    tot_te = np.asarray(xte.sum(axis=0)).ravel()
    nnz_te = np.diff(xte.indptr)
    b_tr, b_te = _depth_bins(tot_tr, tot_tr, B), _depth_bins(tot_tr, tot_te, B)
    vb_tr, vb_te = _depth_bins(tot_tr, tot_tr, VB), _depth_bins(tot_tr, tot_te, VB)
    out = {"nnz_test": int(xte.nnz), "cells_test": len(te)}

    _, g_tr, v_tr, own_tr = TR
    pres0 = Presence(G, g_tr, b_tr, own_tr, B)
    vals0 = Values(cap, g_tr, v_tr, vb_tr, own_tr, VB)
    gd_cell = cell_costs(G, TE, pres0, b_te, vals0, vb_te, cap)
    out["gene_depth_bits"] = float(gd_cell.sum())

    assign, K = fit_clusters(xtr, cfg_floor, seed)
    k_tr, k_te = assign(xtr), assign(xte)
    lab_p = (np.bincount(k_tr, minlength=K) + 0.5) / (len(k_tr) + 0.5 * K)
    label_bits = -np.log2(lab_p[k_te])
    beta = float(cfg_floor.get("cluster_backoff", 8.0))
    pres1 = Presence(G, g_tr, k_tr * B + b_tr, own_tr, K * B, beta, pres0, np.arange(K * B) % B)
    vals1 = Values(cap, g_tr, v_tr, k_tr * VB + vb_tr, own_tr, K * VB, beta, vals0, np.arange(K * VB) % VB)
    cl_cell = cell_costs(G, TE, pres1, k_te * B + b_te, vals1, k_te * VB + vb_te, cap)
    cl_cell += label_bits
    out["cluster_bits"] = float(cl_cell.sum())
    out["clusters"] = int(K)

    ref = float(tot_tr.mean())
    s_tr, s_te = _size_factors(tot_tr, ref), _size_factors(tot_te, ref)
    out["poisson_bits"] = poisson_cost(xtr, xte, s_tr, s_te)
    exact = G * len(te) <= 5e7
    out["nb_bits"] = nb_cost(xtr, xte, s_tr, s_te, B, exact=False)
    if exact:
        out["nb_bits_exact"] = nb_cost(xtr, xte, s_tr, s_te, B, exact=True)

    if want_sampling:
        rng = np.random.default_rng(seed + 17)
        m = min(int(cfg_floor["sampling_cells"]), len(te))
        pick = np.sort(rng.choice(len(te), size=m, replace=False))
        prof_k = np.zeros((K, G))
        np.add.at(prof_k, (np.repeat(k_tr, np.diff(xtr.indptr)), xtr.indices), xtr.data)
        prof_k = (prof_k + 1e-3) / (prof_k + 1e-3).sum(axis=1, keepdims=True)
        xs = xte[:, pick]
        profs = [prof_k[k] for k in k_te[pick]]          # views, no copy
        alpha = fit_shrinkage(sp.csc_matrix(xs), profs, rng)
        samp = np.empty(m)
        for j in range(m):
            s0, s1 = xs.indptr[j], xs.indptr[j + 1]
            N = int(xs.data[s0:s1].sum())
            pi = alpha * profs[j]
            pi[xs.indices[s0:s1]] += xs.data[s0:s1]
            pi /= N + alpha
            samp[j] = multinomial_entropy_bits(N, pi)
        nz = nnz_te[pick].sum()
        out.update({
            "sample_cells": m, "sample_nnz": int(nz), "shrinkage_alpha": alpha,
            "sample_sampling_bits": float(samp.sum()),
            "sample_cluster_bits": float(cl_cell[pick].sum()),
            "sample_gene_depth_bits": float(gd_cell[pick].sum()),
        })
    return out


def run(cfg: Config, units: list[Unit], force: bool = False) -> None:
    fc = cfg["floor"]
    for u in units:
        out = result_path(cfg, "floor", u)
        if out.exists() and not force:
            continue
        c = load_canon(cfg, u)
        rng = np.random.default_rng(int(fc["seed"]))
        perm = rng.permutation(c.n)
        a, b = np.sort(perm[: c.n // 2]), np.sort(perm[c.n // 2:])
        folds = [_fold(c, a, b, fc, int(fc["seed"]), True), _fold(c, b, a, fc, int(fc["seed"]) + 1, True)]
        row = {"dataset": u.id, "matrix": u.name, "protocol_class": u.dataset["protocol_class"], "nnz": c.nnz,
               "clusters": folds[0]["clusters"]}
        nnz_te = sum(f["nnz_test"] for f in folds)
        for m in ("poisson", "nb", "gene_depth", "cluster"):
            row[f"{m}_bpn"] = sum(f[f"{m}_bits"] for f in folds) / nnz_te
        if all("nb_bits_exact" in f for f in folds):
            row["nb_exact_bpn"] = sum(f["nb_bits_exact"] for f in folds) / nnz_te
        sn = sum(f["sample_nnz"] for f in folds)
        for m in ("sampling", "cluster", "gene_depth"):
            row[f"sample_{m}_bpn"] = sum(f[f"sample_{m}_bits"] for f in folds) / sn
        row["shrinkage_alpha"] = folds[0]["shrinkage_alpha"]
        # .s1 payload for the same matrix, from the ladder stage
        lad = read_tsv(result_path(cfg, "ladder", u))
        if lad:
            row["s1_payload_bpn"] = float(lad[0]["L4_s1_payload"])
        write_tsv([row], out)
        log(f"{u.key}: floor {row}")
