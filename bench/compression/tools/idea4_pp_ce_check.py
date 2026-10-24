"""Independent float64 cross-entropy of the K=0 (static base) pp codec on the called cells of one sample.

python -P tools/idea4_pp_ce_check.py <model_k0.bin> <sample>
Re-implements, with floats, the model that kernels/idea4_pp.h evaluates with integers: Poisson-process survival gap symbols with the
2^28 scale and per-symbol floor, scale class 1/8 octave, count tables bucketed by the interval hazard.  The result is compared with
the harness' bits/nnz (gap, count) for the called cells; they should agree to well under 1%.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import idea4_lib as L, idea4_pp_export as EX

mp, sid = sys.argv[1], sys.argv[2]
d = EX.read_model(mp)
org = L.ALL[sid][2]
ref = EX.get_ref(org); go, st, G, m = ref.go, ref.st, ref.G, ref.m
pi0 = np.exp(-d['code0'].astype(np.float64) / 2048.0)
pi0 = pi0 / pi0.sum()
w = np.exp2(-d['qcode'].astype(np.float64) / 16.0)
qc = np.cumsum(w) - np.concatenate([[0], np.cumsum(w)[st[1:-1] - 1]])[go] * 0   # placeholder, recomputed below
# within-gene inclusive cumulative of q
cs = np.cumsum(w)
base = np.concatenate([[0.0], cs[st[1:-1] - 1]])[go]
tot = np.add.reduceat(w, st[:-1])[go]
qcum = (cs - base) / tot                                   # inclusive within-gene cumulative (last = 1)
pre = np.concatenate([[0.0], np.cumsum(pi0)])[:-1][go]     # prefix of gene mass before gene(j)
F = pre + pi0[go] * qcum                                   # cumulative hazard per unit Nhat, inclusive of interval j
S = L.load_sample(sid)
X = L.csr_of(S, S.called).tocsr(); X.sort_indices()
T = float(1 << 28)
ct = d['ctab0'].astype(np.float64)
NCB, NSYM = ct.shape[0], ct.shape[1] - 1
tot_gap = 0.0; tot_cnt = 0.0; nnz = 0
for r in range(X.shape[0]):
    a, b = X.indptr[r], X.indptr[r + 1]
    idx = X.indices[a:b].astype(np.int64); val = X.data[a:b].astype(np.int64)
    n = len(idx)
    if n == 0:
        continue
    N = val.sum()
    s0 = int(round(8 * np.log2(N))) + 16
    cand = [s for s in (s0 - 1, s0, s0 + 1) if 0 <= s < 160]
    nh = [2.0 ** ((s - 16) / 8.0) for s in cand]
    best = min(nh, key=lambda h: h - N * np.log(h))
    nh = best
    Fj = F[idx]; Fprev = np.concatenate([[0.0], Fj[:-1]])            # F at previous present (i), F(0..)
    prev_idx = np.concatenate([[-1], idx[:-1]])
    L_ = (m - prev_idx).astype(np.float64)
    # e_{j-1} uses F(j-1): for j adjacent to i it equals e(i)=1
    Fjm1 = np.where(idx == prev_idx + 1, Fprev, F[np.maximum(idx - 1, 0)])
    e_j = np.exp(-nh * (Fj - Fprev)); e_jm1 = np.exp(-nh * (Fjm1 - Fprev))
    p = (1.0 + (T - L_) * (e_jm1 - e_j)) / T
    tot_gap += -np.log2(p).sum()
    if idx[-1] != m - 1:
        Li = float(m - idx[-1]); e_end = np.exp(-nh * (F[m - 1] - Fj[-1]))
        tot_gap += -np.log2((1.0 + (T - Li) * e_end) / T)
    # counts
    lam = nh * (Fj - Fjm1)
    lam_q = np.maximum(lam * 4294967296.0, 1.0)
    e = np.floor(np.log2(lam_q)) - 32
    half = (lam_q >= np.exp2(np.floor(np.log2(lam_q))) * 1.5).astype(int)
    bk = np.clip(2 * (e.astype(int) + 10) + half, 0, NCB - 1)
    sym = np.minimum(val - 1, NSYM - 1)
    freq = ct[bk, sym + 1] - ct[bk, sym]
    cb = -np.log2(freq / 16384.0)
    esc = sym == NSYM - 1
    if esc.any():
        v = val[esc] - NSYM + 1
        nb = np.floor(np.log2(v)).astype(int) + 1
        cb[esc] += 5 + (nb - 1)
    tot_cnt += cb.sum()
    nnz += n
print('%s called cells: nnz %d   python-float gap %.4f  cnt %.4f  bits/nnz  (excl. scale-class side info)' % (sid, nnz, tot_gap / nnz, tot_cnt / nnz))
