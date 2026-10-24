"""Multinomial topic model (PLSA == KL-NMF with simplex-normalised columns) at gene level; C++ EM via ctypes.
cell profile pi_c = Phi @ theta_c, Phi (G x k) column stochastic, theta_c on the simplex."""
import os, ctypes, time
import numpy as np, scipy.sparse as sp

_dll = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin', 'idea4_em.dll'))
_dll.em_sweep.restype = ctypes.c_double
_dll.em_sweep.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_void_p] * 3 + [ctypes.c_int, ctypes.c_int]
_dll.cell_ll.restype = None
_dll.cell_ll.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 3 + [ctypes.c_void_p] * 3


_dll.pred_nz.restype = None
_dll.pred_nz.argtypes = [ctypes.c_int64, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 5


def pred_nz(C, Theta, Phi):
    out = np.empty(C.data.shape[0], dtype=np.float32)
    _dll.pred_nz(C.n, C.G, Phi.shape[1], _p(C.indptr), _p(C.indices), _p(Theta), _p(Phi), _p(out))
    return out


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


class Csr:
    def __init__(self, X):
        X = sp.csr_matrix(X)
        X.sort_indices()
        self.n, self.G = X.shape
        self.indptr = np.ascontiguousarray(X.indptr.astype(np.int64))
        self.indices = np.ascontiguousarray(X.indices.astype(np.int32))
        self.data = np.ascontiguousarray(X.data.astype(np.float32))
        self.N = np.asarray(X.sum(1)).ravel().astype(np.float64)
        self.X = X


def sweep(C, Theta, Phi, acc=None, update_theta=True, nth=2):
    k = Phi.shape[1]
    return _dll.em_sweep(C.n, C.G, k, _p(C.indptr), _p(C.indices), _p(C.data), _p(Theta), _p(Phi),
                         _p(acc) if acc is not None else None, int(update_theta), nth)


def cell_ll(C, Theta, Phi):
    out = np.empty(C.n, dtype=np.float64)
    _dll.cell_ll(C.n, C.G, Phi.shape[1], _p(C.indptr), _p(C.indices), _p(C.data), _p(Theta), _p(Phi), _p(out))
    return out


def fit_theta(C, Phi, iters=30, Theta=None, nth=2):
    Phi = np.ascontiguousarray(Phi, dtype=np.float32)
    k = Phi.shape[1]
    if Theta is None:
        Theta = np.full((C.n, k), 1.0 / k, dtype=np.float32)
    Theta = np.ascontiguousarray(Theta, dtype=np.float32)
    for _ in range(iters):
        sweep(C, Theta, Phi, None, True, nth)
    return Theta


def init_phi(C, k, pi0, seed=0):
    """Sparse spherical k-means on the top 2000 genes -> cluster profiles smoothed with pi0."""
    rng = np.random.default_rng(seed)
    X = C.X
    n, G = X.shape
    topg = np.argsort(-np.asarray(X.sum(0)).ravel())[:2000]
    Xt = X[:, topg].tocsr().astype(np.float32)
    Xt.data = np.sqrt(Xt.data)
    nr = np.sqrt(np.asarray(Xt.multiply(Xt).sum(1)).ravel()) + 1e-9
    Xt = sp.diags(1.0 / nr) @ Xt
    cent = np.asarray(Xt[rng.choice(n, k, replace=False)].todense())
    for _ in range(12):
        lab = np.asarray(Xt @ cent.T).argmax(1)
        for j in range(k):
            sel = lab == j
            if sel.any():
                c = np.asarray(Xt[sel].sum(0)).ravel()
                cent[j] = c / (np.linalg.norm(c) + 1e-9)
    Phi = np.empty((G, k), dtype=np.float32)
    for j in range(k):
        sel = np.where(lab == j)[0]
        c = np.asarray(X[sel].sum(0)).ravel() if len(sel) else np.zeros(G)
        Phi[:, j] = (c + 50 * pi0) / (c.sum() + 50)
    return Phi


def train(C, k, iters=40, beta=500.0, pi0=None, init=None, seed=0, nth=2, verbose=True):
    n, G = C.n, C.G
    if pi0 is None:
        pi0 = np.asarray(C.X.sum(0)).ravel() + 1.0
        pi0 = pi0 / pi0.sum()
    Phi = init_phi(C, k, pi0, seed) if init is None else init.astype(np.float32).copy()
    Phi = np.ascontiguousarray(Phi)
    Theta = np.full((n, k), 1.0 / k, dtype=np.float32)
    for _ in range(3):
        sweep(C, Theta, Phi, None, True, nth)
    acc = np.zeros((G, k), dtype=np.float32)
    t0 = time.time()
    for it in range(iters):
        ll = sweep(C, Theta, Phi, acc, True, nth)
        Phin = acc + beta * pi0[:, None].astype(np.float32)
        Phin /= Phin.sum(0, keepdims=True)
        Phi = np.ascontiguousarray(Phin, dtype=np.float32)
        if verbose and (it % 10 == 0 or it == iters - 1):
            print(f'  it {it} ll/UMI {ll / C.data.sum():.4f}  {time.time() - t0:.0f}s', flush=True)
    return Phi, Theta
