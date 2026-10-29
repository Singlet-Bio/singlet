"""Idea 4 cross-entropy study at gene level (called cells).

python -P tools/idea4_ce.py <organism> <k,k,...> [--insample] [--cross]

For each test sample of the organism: static per-gene x depth (Poisson rate N*pi0, and the explicit per-gene x
depth-bucket presence table), NMF k models with unquantised / 6-bit / 4-bit / 256-codebook theta, each both as raw
Poisson cross-entropy and with calibrated lambda-bucket presence + count tables (what the coder implements).
Writes results/idea4_ce.tsv (append) and results/idea4_models/<org>_k<k>.npz.
"""
import os, sys, time
import numpy as np, scipy.sparse as sp
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L, idea4_nmf as NM

org = sys.argv[1]
ks = [int(v) for v in sys.argv[2].split(',')]
insample = '--insample' in sys.argv
cross = '--cross' in sys.argv
NTH = 1
NB, CMAX = 128, 24
OUT = os.path.join(L.ROOT, 'results', 'idea4_ce.tsv')
MD = os.path.join(L.ROOT, 'results', 'idea4_models'); os.makedirs(MD, exist_ok=True)
if not os.path.exists(OUT):
    with open(OUT, 'w') as f:
        f.write('org\tsample\tmode\tmodel\tk\ttheta\tcells\tnnz_iv\tnnz_gene\tgene_bits_pois\tgene_bits_cal1\tgene_bits_cal2\t'
                'split_bits\tconst_bits\tside_bits\tbits_per_nnz_pois\tbits_per_nnz_cal1\tbits_per_nnz_cal2\tnet_cal1\tnet_cal2\tnote\n')


def log(*a):
    print(*a, flush=True)


def rows_of(tr, te):
    if org == 'human':
        return [t for t in tr if L.ALL[t][2] == 'human'], [t for t in te if L.ALL[t][2] == 'human']
    return [t for t in tr if L.ALL[t][2] == 'mouse'], [t for t in te if L.ALL[t][2] == 'mouse']


train_ids, test_ids = rows_of(list(L.TRAIN), list(L.TEST))
S0 = L.load_sample(test_ids[0])
go, st, ids, ln = L.gene_structure(S0.rownames); G = len(ids); m = S0.m
genes_sym = [nm.split('_')[1] for nm in [S0.rownames[s] for s in st[:-1]]]


def load_gene(sid, cap=None, seed=1):
    S = L.load_sample(sid)
    X = L.csr_of(S, S.called).tocsr()
    if cap and X.shape[0] > cap:
        sel = np.sort(np.random.default_rng(seed).choice(X.shape[0], cap, replace=False)); X = X[sel]
    Xg = L.gene_matrix(go, G, X.astype(np.float32))
    return S, X, Xg


# ---------------- training pool ----------------
cnt_i = np.zeros(m); Xs = []
for t in train_ids:
    S, X, Xg = load_gene(t, cap=3000)
    cnt_i += np.asarray(L.csr_of(S, S.called).sum(0)).ravel()
    Xs.append(Xg)
Xtr = sp.vstack(Xs).tocsr()
Ctr = NM.Csr(Xtr)
cnt_g = np.asarray(Xtr.sum(0)).ravel()
pi0 = (cnt_g + 0.5) / (cnt_g + 0.5).sum()
a = ln / ln.sum() * 1000
ci = cnt_i + a
q = ci / np.add.reduceat(ci, st[:-1])[go]
log('train cells', Xtr.shape[0], 'gene nnz', Xtr.nnz)


def hist(C, Theta, Phi):
    k = Phi.shape[1]
    oa = np.zeros(NB); oc = np.zeros(NB * CMAX); oa2 = np.zeros(8 * NB); oc2 = np.zeros(8 * NB * CMAX)
    NM._dll.calib_hist(C.n, C.G, k, NM._p(C.indptr), NM._p(C.indices), NM._p(C.data), NM._p(np.ascontiguousarray(Theta, dtype=np.float32)),
                       NM._p(np.ascontiguousarray(Phi, dtype=np.float32)), NM._p(C.N), NB, CMAX, NM._p(oa), NM._p(oc), NM._p(oa2), NM._p(oc2))
    return oa, oc.reshape(NB, CMAX), oa2.reshape(8, NB), oc2.reshape(8, NB, CMAX)


NM._dll.calib_hist.restype = None
NM._dll.calib_hist.argtypes = [NM.ctypes.c_int64, NM.ctypes.c_int, NM.ctypes.c_int] + [NM.ctypes.c_void_p] * 6 + [NM.ctypes.c_int, NM.ctypes.c_int] + [NM.ctypes.c_void_p] * 4
NM._dll.best_code.restype = None
NM._dll.best_code.argtypes = [NM.ctypes.c_int64, NM.ctypes.c_int, NM.ctypes.c_int] + [NM.ctypes.c_void_p] * 5 + [NM.ctypes.c_int, NM.ctypes.c_void_p, NM.ctypes.c_int, NM.ctypes.c_void_p]
NM._dll.gene_depth_hist.restype = None
NM._dll.gene_depth_hist.argtypes = [NM.ctypes.c_int64, NM.ctypes.c_int] + [NM.ctypes.c_void_p] * 5
NM._dll.gene_depth_cost.restype = NM.ctypes.c_double
NM._dll.gene_depth_cost.argtypes = [NM.ctypes.c_int64, NM.ctypes.c_int] + [NM.ctypes.c_void_p] * 4


def tables(oa, oc, oa2, oc2):
    """Smoothed presence prob and count pmf per bucket (and per depth x bucket)."""
    pres = oc.sum(1)
    p1 = (pres + 0.5) / (oa + 1.0)
    t1 = (oc + 0.5) / (oc + 0.5).sum(1, keepdims=True)
    pres2 = oc2.sum(2)
    # shrink 2-D toward 1-D with pseudo-count 50
    p2 = (pres2 + 50 * p1[None]) / (oa2 + 50)
    t2 = (oc2 + 20 * t1[None]) / (oc2 + 20 * t1[None]).sum(2, keepdims=True)
    return p1, t1, p2, t2


def cal_cost(h, tb):
    oa, oc, oa2, oc2 = h; p1, t1, p2, t2 = tb
    pres = oc.sum(1)
    c1 = -(pres * np.log2(p1) + (oa - pres) * np.log2(1 - p1)).sum() - (oc * np.log2(t1)).sum()
    pres2 = oc2.sum(2)
    c2 = -(pres2 * np.log2(p2) + (oa2 - pres2) * np.log2(1 - p2)).sum() - (oc2 * np.log2(t2)).sum()
    return c1, c2


def pois_cost(C, Theta, Phi):
    P = NM.pred_nz(C, np.ascontiguousarray(Theta, dtype=np.float32), np.ascontiguousarray(Phi, dtype=np.float32))
    rows = np.repeat(np.arange(C.n), np.diff(C.indptr))
    return -(C.data.astype(np.float64) * np.log2(np.maximum(C.N[rows] * P.astype(np.float64), 1e-300))).sum()


def quant_theta(Theta, bits):
    lv = (1 << bits) - 1
    Q = np.round(Theta * lv) / lv
    s = Q.sum(1, keepdims=True); s[s == 0] = 1
    return (Q / s).astype(np.float32)


def kmeans_codebook(Theta, nc=256, iters=20, seed=0):
    rng = np.random.default_rng(seed)
    Z = np.sqrt(Theta)
    cent = Z[rng.choice(len(Z), nc, replace=False)].copy()
    for _ in range(iters):
        d = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        for j in range(nc):
            if (lab == j).any(): cent[j] = Z[lab == j].mean(0)
    Cb = cent ** 2
    Cb /= Cb.sum(1, keepdims=True)
    return Cb.astype(np.float32)


def assign_codebook(C, Theta, Phi, Cb, ncand=16):
    Z = np.sqrt(Theta); Zc = np.sqrt(Cb)
    cand = np.empty((C.n, ncand), dtype=np.int32)
    for s in range(0, C.n, 2000):
        d = ((Z[s:s + 2000, None, :] - Zc[None]) ** 2).sum(-1)
        cand[s:s + 2000] = np.argsort(d, axis=1)[:, :ncand]
    out = np.empty(C.n, dtype=np.int32)
    NM._dll.best_code(C.n, C.G, Phi.shape[1], NM._p(C.indptr), NM._p(C.indices), NM._p(C.data),
                      NM._p(np.ascontiguousarray(Phi, dtype=np.float32)), NM._p(np.ascontiguousarray(Cb)), len(Cb), NM._p(np.ascontiguousarray(cand)), ncand, NM._p(out))
    return out


def gene_depth_table(Ctr_list, pcal_fn):
    pres = np.zeros(8 * G); nc = np.zeros(8)
    for C in Ctr_list:
        NM._dll.gene_depth_hist(C.n, C.G, NM._p(C.indptr), NM._p(C.indices), NM._p(C.N), NM._p(pres), NM._p(nc))
    pres = pres.reshape(8, G)
    # shrink toward the calibrated-lambda curve at the depth-bucket centre
    Nc = 2.0 ** (2 * (np.arange(8) + 2) + 1)
    prior = np.stack([pcal_fn(Nc[d] * pi0) for d in range(8)])
    A = 30.0
    return (pres + A * prior) / (nc[:, None] + A)


# -------- evaluate --------
def emit(sample, mode, model, k, theta, S, X, Xg, gb_p, gb_c1, gb_c2, side_bits, note=''):
    x = X.data.astype(np.float64)
    split = -(x * np.log2(q[X.indices])).sum()
    N = np.asarray(X.sum(1)).ravel().astype(np.float64)
    const = (N.sum() + gammaln(x + 1).sum()) * L.LOG2E
    nnz = X.nnz
    bp = (gb_p + split + const) / nnz
    b1 = (gb_c1 + split) / nnz
    b2 = (gb_c2 + split) / nnz
    with open(OUT, 'a') as f:
        f.write(f'{org}\t{sample}\t{mode}\t{model}\t{k}\t{theta}\t{X.shape[0]}\t{nnz}\t{Xg.nnz}\t{gb_p:.1f}\t{gb_c1:.1f}\t{gb_c2:.1f}\t'
                f'{split:.1f}\t{const:.1f}\t{side_bits:.1f}\t{bp:.4f}\t{b1:.4f}\t{b2:.4f}\t{(gb_c1 + split + side_bits) / nnz:.4f}\t{(gb_c2 + split + side_bits) / nnz:.4f}\t{note}\n')
    log(f'  {sample:22s} {mode:9s} {model:8s} k={k:<3d} {theta:8s} pois {bp:.3f}  cal1 {b1:.3f}  cal2 {b2:.3f}  side {side_bits / nnz:.3f}  net1 {(gb_c1 + split + side_bits) / nnz:.3f}  net2 {(gb_c2 + split + side_bits) / nnz:.3f} {note}')


tests = [load_gene(t) for t in test_ids]
Ctests = [NM.Csr(Xg) for (_, _, Xg) in tests]

# static baseline: pi0 as a k=1 model
Phi1 = np.ascontiguousarray(pi0[:, None], dtype=np.float32)
h_tr = hist(Ctr, np.ones((Ctr.n, 1), np.float32), Phi1)
tb_static = tables(*h_tr)
p1s = tb_static[0]
def pcal_static(lam):
    b = np.clip(np.floor(4 * (np.log2(np.maximum(lam, 1e-30)) + 24)).astype(int), 0, NB - 1)
    return p1s[b]
Ptab = gene_depth_table([Ctr], pcal_static)
for (S, X, Xg), C in zip(tests, Ctests):
    Th1 = np.ones((C.n, 1), np.float32)
    h = hist(C, Th1, Phi1)
    c1, c2 = cal_cost(h, tb_static)
    emit(S.id, 'oos', 'static', 1, 'none', S, X, Xg, pois_cost(C, Th1, Phi1), c1, c2, 0.0)
    # explicit per-gene x depth-bucket presence table + static count table
    pres_bits = NM._dll.gene_depth_cost(C.n, C.G, NM._p(C.indptr), NM._p(C.indices), NM._p(C.N), NM._p(np.ascontiguousarray(Ptab)))
    cnt_bits = -(h[1] * np.log2(tb_static[1])).sum()
    emit(S.id, 'oos', 'gxd_tab', 1, 'none', S, X, Xg, float('nan'), pres_bits + cnt_bits, pres_bits + cnt_bits, 0.0, 'per-gene x depth-bucket presence table')
    # oracle calibration (tables fitted on the test file itself; lower bound for the static rate)
    c1o, c2o = cal_cost(h, tables(*h))
    emit(S.id, 'oracle', 'static', 1, 'none', S, X, Xg, float('nan'), c1o, c2o, 0.0, 'calibration tables fitted on test file (in-sample)')

for k in ks:
    mp = os.path.join(MD, f'{org}_k{k}.npz')
    if os.path.exists(mp):
        Phi = np.load(mp)['Phi']; Th_tr = np.load(mp)['Theta']; log('loaded', mp)
    else:
        t0 = time.time()
        Phi, Th_tr = NM.train(Ctr, k, iters=40, beta=500.0, pi0=pi0, nth=NTH)
        log(f'trained k={k} in {time.time() - t0:.0f}s')
        np.savez(mp, Phi=Phi, Theta=Th_tr, pi0=pi0)
    Cb = kmeans_codebook(Th_tr)
    h_tr = hist(Ctr, Th_tr, Phi)
    tb = tables(*h_tr)
    for (S, X, Xg), C in zip(tests, Ctests):
        Th = NM.fit_theta(C, Phi, iters=30, nth=NTH)
        for name, T, side in [('float', Th, 0.0), ('q6', quant_theta(Th, 6), 6.0 * k * C.n), ('q4', quant_theta(Th, 4), 4.0 * k * C.n)]:
            h = hist(C, T, Phi)
            c1, c2 = cal_cost(h, tb)
            emit(S.id, 'oos', 'nmf', k, name, S, X, Xg, pois_cost(C, T, Phi), c1, c2, side)
        code = assign_codebook(C, Th, Phi, Cb)
        T = Cb[code]
        h = hist(C, T, Phi); c1, c2 = cal_cost(h, tb)
        emit(S.id, 'oos', 'nmf', k, 'cb256', S, X, Xg, pois_cost(C, T, Phi), c1, c2, 8.0 * C.n)
    if insample:
        for (S, X, Xg), C in zip(tests, Ctests):
            cap = min(C.n, 3000)
            sel = np.sort(np.random.default_rng(2).choice(C.n, cap, replace=False))
            Cs = NM.Csr(Xg[sel])
            pi_s = np.asarray(Xg.sum(0)).ravel() + 0.5; pi_s /= pi_s.sum()
            Phi_s, Th_s = NM.train(Cs, k, iters=40, beta=500.0, pi0=pi_s, nth=NTH, verbose=False)
            tb_s = tables(*hist(Cs, Th_s, Phi_s))
            Th = NM.fit_theta(C, Phi_s, iters=30, nth=NTH)
            T = quant_theta(Th, 6)
            h = hist(C, T, Phi_s); c1, c2 = cal_cost(h, tb_s)
            emit(S.id, 'insample', 'nmf', k, 'q6', S, X, Xg, pois_cost(C, T, Phi_s), c1, c2, 6.0 * k * C.n, 'Phi trained on the test file (upper bound)')
            # in-sample static for reference
            P1 = np.ascontiguousarray(pi_s[:, None], dtype=np.float32); T1 = np.ones((C.n, 1), np.float32)
            h = hist(C, T1, P1); tb1 = tables(*hist(Cs, np.ones((Cs.n, 1), np.float32), P1)); c1, c2 = cal_cost(h, tb1)
            emit(S.id, 'insample', 'static', 1, 'none', S, X, Xg, pois_cost(C, T1, P1), c1, c2, 0.0, 'pi0 from the test file (upper bound)')
log('done')
