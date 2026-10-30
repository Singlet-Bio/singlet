"""Check the sparse code-length bookkeeping of floor.py against a dense, cell-by-cell computation on a tiny matrix.

    cd bench && python -m pytest tests/test_floor.py      (or: python tests/test_floor.py)
"""

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from s1bench import floor  # noqa: E402
from s1bench.util import canon_from_scipy  # noqa: E402


def _toy(seed=0, G=40, C=120):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(0.5, 1.0, size=(G, 1)) * rng.lognormal(0, 0.6, size=(1, C))
    x = rng.poisson(lam * 3)
    x[0, :5] = 200  # a few escapes
    return canon_from_scipy(sp.csc_matrix(x), [f"g{i}" for i in range(G)], [f"c{j}" for j in range(C)])


def _dense_presence(G, x_tr, ctx_tr, x_te, ctx_te, n_ctx, beta, prior):
    n = np.bincount(ctx_tr, minlength=n_ctx).astype(float)
    k = np.zeros((G, n_ctx))
    for j, c in enumerate(ctx_tr):
        k[:, c] += x_tr[:, j] > 0
    p = (k + beta * prior) / (n[None, :] + beta)
    bits = np.zeros(x_te.shape[1])
    for j, c in enumerate(ctx_te):
        on = x_te[:, j] > 0
        bits[j] = -np.log2(p[on, c]).sum() - np.log2(1 - p[~on, c]).sum()
    return bits, p


def _dense_values(G, x_tr, vc_tr, x_te, vc_te, n_v, cap, beta, prior_fn):
    h = np.zeros((G, n_v, cap + 1))
    for j, c in enumerate(vc_tr):
        for g in np.flatnonzero(x_tr[:, j]):
            h[g, c, min(x_tr[g, j], cap)] += 1
    t = h.sum(axis=2)
    bits = np.zeros(x_te.shape[1])
    for j, c in enumerate(vc_te):
        for g in np.flatnonzero(x_te[:, j]):
            v = x_te[g, j]
            q = (h[g, c, min(v, cap)] + beta * prior_fn(g, c, v)) / (t[g, c] + beta)
            bits[j] += -np.log2(q) + (floor._escape_bits(np.array([v]), cap)[0] if v >= cap else 0.0)
    return bits


def test_gene_depth_matches_dense():
    c = _toy()
    cap, B, VB = 16, 4, 2
    tr, te = np.arange(0, c.n, 2), np.arange(1, c.n, 2)
    TR, TE = floor._cols(c, tr), floor._cols(c, te)
    tot_tr = np.asarray(TR[0].sum(axis=0)).ravel()
    tot_te = np.asarray(TE[0].sum(axis=0)).ravel()
    b_tr, b_te = floor._depth_bins(tot_tr, tot_tr, B), floor._depth_bins(tot_tr, tot_te, B)
    v_tr, v_te = floor._depth_bins(tot_tr, tot_tr, VB), floor._depth_bins(tot_tr, tot_te, VB)
    _, g_tr, val_tr, own_tr = TR
    pres = floor.Presence(c.m, g_tr, b_tr, own_tr, B)
    vals = floor.Values(cap, g_tr, val_tr, v_tr, own_tr, VB)
    got = floor.cell_costs(c.m, TE, pres, b_te, vals, v_te, cap)
    xd_tr, xd_te = TR[0].toarray(), TE[0].toarray()
    pb, _ = _dense_presence(c.m, xd_tr, b_tr, xd_te, b_te, B, 1.0, 0.5)
    vb = _dense_values(c.m, xd_tr, v_tr, xd_te, v_te, VB, cap, 0.5 * cap, lambda g, k, v: 1.0 / cap)
    assert np.allclose(got, pb + vb), (got[:5], (pb + vb)[:5])

    # cluster level backing off to the gene x depth level
    K = 3
    k_tr, k_te = np.arange(len(tr)) % K, np.arange(len(te)) % K
    beta = 8.0
    pres1 = floor.Presence(c.m, g_tr, k_tr * B + b_tr, own_tr, K * B, beta, pres, np.arange(K * B) % B)
    vals1 = floor.Values(cap, g_tr, val_tr, k_tr * VB + v_tr, own_tr, K * VB, beta, vals, np.arange(K * VB) % VB)
    got1 = floor.cell_costs(c.m, TE, pres1, k_te * B + b_te, vals1, k_te * VB + v_te, cap)
    _, p0 = _dense_presence(c.m, xd_tr, b_tr, xd_te, b_te, B, 1.0, 0.5)
    prior = p0[:, np.arange(K * B) % B]
    pb1, _ = _dense_presence(c.m, xd_tr, k_tr * B + b_tr, xd_te, k_te * B + b_te, K * B, beta, prior)
    q0 = lambda g, kv, v: vals.q(np.array([g]), np.array([kv % VB]), np.array([v]))[0]
    vb1 = _dense_values(c.m, xd_tr, k_tr * VB + v_tr, xd_te, k_te * VB + v_te, K * VB, cap, beta, q0)
    assert np.allclose(got1, pb1 + vb1), (got1[:5], (pb1 + vb1)[:5])


def test_multinomial_entropy_matches_monte_carlo():
    rng = np.random.default_rng(3)
    p = rng.dirichlet(np.full(300, 0.05))
    N = 400
    exact = floor.multinomial_entropy_bits(N, p)
    from scipy.stats import multinomial

    draws = rng.multinomial(N, p, size=4000)
    mc = -multinomial.logpmf(draws, N, p).mean() / np.log(2)
    assert abs(exact - mc) / mc < 0.01, (exact, mc)


if __name__ == "__main__":
    test_gene_depth_matches_dense()
    test_multinomial_entropy_matches_monte_carlo()
    print("ok")
