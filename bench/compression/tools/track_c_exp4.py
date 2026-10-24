"""Two-part code: side info = per-feature presence-count bins (coded with CPT given train profile), then presence given class (k/n bin) and depth bucket."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp3 import pava_dec

def rbin(k, n, step=0.25, nlev=120):
    """class of count k in n columns: 0 = k==0 ; else 1+level of log2(k/n) from top (rate 1 -> level 0)"""
    c = np.zeros(len(k), dtype=np.int32)
    nz = k > 0
    c[nz] = 1 + np.clip(np.round(-np.log2(k[nz] / n) / step), 0, nlev - 1).astype(np.int32)
    return c

def table_from_counts(A, M, floor=1e-9):
    """A,M [nb,nc] -> monotone-smoothed P (non-decreasing in class index meaning lower rate => decreasing P); class 0 (k==0) has P=0"""
    nb, nc = A.shape
    P = np.zeros((nb, nc))
    for b in range(nb):
        y = A[b, 1:] / np.maximum(M[b, 1:], 1e-9); w = M[b, 1:]
        if w.sum() == 0: P[b, 1:] = floor; continue
        P[b, 1:] = np.maximum(pava_dec(y, w + 1e-6), floor)
    P[:, 0] = 0
    return P

def run(test, mem, nb=32, nz_hi=14, step=0.25, verbose=True):
    org = ORG[test]; g, ng = gene_index(org, mem)
    pool = [k for k in STUDY if ORG[k] == org and STUDY[k] != STUDY[test]]
    names = [f'{k}__{mem}' for k in pool]; m = get(names[0])['m']
    scheme = ('nz', nb, 0, nz_hi)
    nclass = 1 + 120
    Atr = np.zeros((nb, nclass)); Mtr = np.zeros((nb, nclass))
    for nm in names:
        d = get(nm); b = bucketize(d, scheme)
        k = np.bincount(d['indices'], minlength=m); c = rbin(k, d['n'], step)
        A, Nb = counts_AN(d, b, c, nb, nclass); mc = np.bincount(c, minlength=nclass).astype(float)
        Atr += A; Mtr += Nb[:, None] * mc[None, :]
    P = table_from_counts(Atr, Mtr)
    d = get(f'{test}__{mem}'); b = bucketize(d, scheme)
    k = np.bincount(d['indices'], minlength=m); c = rbin(k, d['n'], step)
    A, Nb = counts_AN(d, b, c, nb, nclass); mc = np.bincount(c, minlength=nclass).astype(float)
    nnz = len(d['indices'])
    # class 0 never present; handle: absent cost 0
    pc = presence_cost_bits(A, Nb, mc, np.where(P > 0, P, 0.5))  # P=0 classes: A=0, cost 0 for absent? set P=0.5 -> wrong; fix below
    # recompute excluding class 0
    A1, mc1 = A[:, 1:], mc[1:]
    pc = presence_cost_bits(A1, Nb, mc1, P[:, 1:])
    bb = entropy_bits(np.bincount(b, minlength=nb).astype(float))
    # side info: CPT of file class given train-profile context
    s_all = profile(names, m); mu = s_all * nnz           # expected count in this file
    gs = np.bincount(g, weights=s_all, minlength=ng) / np.maximum(np.bincount(g, minlength=ng), 1)
    mug = gs[g] * nnz
    def ctx(mu, mug):
        cx = np.where(mu > 0, 2 + np.clip(np.round(np.log2(np.maximum(mu, 1e-9)) * 2) + 40, 0, 79), np.where(mug > 0, 1, 0)).astype(np.int32)
        return cx
    ctxT = ctx(mu, mug)
    ncx = 82
    # train CPT: leave-one-study-out pairs: (profile from other-studies-in-pool, file class)
    C = np.zeros((ncx, nclass))
    studies = sorted(set(STUDY[kk] for kk in pool))
    for st in studies:
        inn = [f'{kk}__{mem}' for kk in pool if STUDY[kk] != st]; out = [f'{kk}__{mem}' for kk in pool if STUDY[kk] == st]
        s_in = profile(inn, m); gs_in = np.bincount(g, weights=s_in, minlength=ng) / np.maximum(np.bincount(g, minlength=ng), 1)
        for nm in out:
            dd = get(nm); nn = len(dd['indices']); kk = np.bincount(dd['indices'], minlength=m); cc = rbin(kk, dd['n'], step)
            cx = ctx(s_in * nn, gs_in[g] * nn)
            np.add.at(C, (cx, cc), 1)
    # classes depend on n (relative) -> CPT symbol is relative-rate class: fine only if n similar; refine below by predicting k directly
    marg = C.sum(0) + 1; marg /= marg.sum()
    Pc = (C + 2.0 * marg[None, :]) / (C.sum(1, keepdims=True) + 2.0)
    side = -np.log2(Pc[ctxT, c]).sum()
    if verbose:
        print(f"{test:22s} {mem} presence|class {pc/nnz:.3f} + side-info(k_f) {side/nnz:.3f} + bucket {bb/nnz:.3f} = {(pc+side+bb)/nnz:.3f} bits/nnz (presence only; rows used {np.sum(k>0)})", flush=True)
    return pc / nnz, side / nnz

if __name__ == '__main__':
    mem = sys.argv[1]
    for t in (sys.argv[2:] or TESTS): run(t, mem)
