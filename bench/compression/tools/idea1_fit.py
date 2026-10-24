"""Parametric presence model: cloglog(p_fd) = a_f + c_d (Poisson occupancy with a free depth curve):
   P(f absent | depth bucket d) = exp(-exp(a_f + c_d)).  IRLS / Fisher scoring, alternating a and c."""
import numpy as np

def fit_cloglog(n, N, eps=None, iters=12, verbose=False):
    m, nb = n.shape
    n = n.astype(np.float64).copy()
    w = N * (n.sum(0) / np.maximum(N, 1))
    w = w / max(w.sum(), 1e-9)
    if eps is not None:
        n += eps[:, None] * w[None, :]
    tot = n.sum(1)
    act = np.where(tot > 0)[0]
    na = n[act].astype(np.float32)
    Nf = N.astype(np.float32)
    rho = n.sum(0) / np.maximum(N, 1) / m
    c = np.log(np.maximum(rho, 1e-12))
    a_act = np.log(np.maximum(tot[act], 1e-3) / max(tot.sum(), 1)).astype(np.float64)
    def stats(a, c):
        mu = np.exp(np.clip(a[:, None] + c[None, :], -40, 8)).astype(np.float32)
        p = np.clip(-np.expm1(-mu), 1e-12, 1 - 1e-6)
        return mu, p
    for it in range(iters):
        for _ in range(2):
            mu, p = stats(a_act, c)
            sc = ((na - Nf[None, :] * p) * mu / p).sum(1, dtype=np.float64)
            info = (Nf[None, :] * (1 - p) * mu * mu / p).sum(1, dtype=np.float64)
            a_act = a_act + np.clip(np.where(info > 0, sc / np.maximum(info, 1e-12), 0.0), -2, 2)
        for _ in range(2):
            mu, p = stats(a_act, c)
            sc = ((na - Nf[None, :] * p) * mu / p).sum(0, dtype=np.float64)
            info = (Nf[None, :] * (1 - p) * mu * mu / p).sum(0, dtype=np.float64)
            c = c + np.clip(np.where(info > 0, sc / np.maximum(info, 1e-12), 0.0), -2, 2)
        if verbose: print(it, float(np.abs(sc).max()), flush=True)
    a = np.full(m, -25.0)
    a[act] = a_act
    return a, c
