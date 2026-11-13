# SPDX-License-Identifier: MIT
# D-152 diagnostic (run on Clipper in debug-cpu; see the PR body of ve/non10x-routes for what it measured).
import sys, os, gzip, numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
R = sys.argv[1]
T = "/mnt/projects/debruinz_project/debruinz/singlet/data/non10x/truth/GSM1544798_SpeciesMix_ThousandSTAMPs_%s.digital_expression.txt.gz"
rd = lambda p: pd.read_csv(p, sep=r"\s+", index_col=0, engine="python")
H, M = rd(T % "HUMAN"), rd(T % "MOUSE")
tot = H.sum(axis=0).add(M.sum(axis=0), fill_value=0)
aset = set(tot.sort_values(ascending=False).head(1020).index.astype(str))
Mx = scipy.io.mmread(os.path.join(R, "raw_feature_bc_matrix", "matrix.mtx.gz")).tocsc()
feats = [l.split("\t")[0] for l in gzip.open(os.path.join(R, "raw_feature_bc_matrix", "features.tsv.gz"), "rt")]
bcs = [l.strip().split("-")[0] for l in gzip.open(os.path.join(R, "raw_feature_bc_matrix", "barcodes.tsv.gz"), "rt")]
mouse = np.array([f.startswith("ENSMUSG") for f in feats])
if Mx.shape[0] != len(feats): Mx = Mx.T.tocsc()
n = len(bcs)
u = np.asarray(Mx.sum(axis=0)).ravel()
order = np.argsort(-u)
idx = {b: i for i, b in enumerate(bcs)}
def nbrs(b):
    for i in range(len(b)):
        for c in "ACGT":
            if c != b[i]: yield b[:i] + c + b[i+1:]
def collapse(ratio, minrank=300000):
    parent = np.arange(n)
    for i in order[:minrank]:
        if u[i] == 0: break
        best = -1
        for x in nbrs(bcs[i]):
            j = idx.get(x)
            if j is not None and u[j] >= ratio * u[i] and (best < 0 or u[j] > u[best]): best = j
        if best >= 0: parent[i] = best
    for i in range(n):   # root
        p = i
        while parent[p] != p: p = parent[p]
        parent[i] = p
    return parent
def cr22(totals, nexp=3000, mp=0.99, mr=10):
    s = np.sort(totals[totals > 0])[::-1]
    nmax = s[min(len(s) - 1, int(round(nexp * (1 - mp))))]
    return max(1, int(round(nmax / mr)))
def score(called_idx, label):
    cb = [bcs[i] for i in called_idx]
    X = Mx[:, list(called_idx)]
    h = np.asarray(X[~mouse].sum(axis=0)).ravel(); m = np.asarray(X[mouse].sum(axis=0)).ravel(); t = h + m; ok = t > 0
    mn = np.minimum(h, m)[ok] / t[ok]; mj = np.maximum(h, m)[ok] / t[ok]
    inter = sum(b in aset for b in cb)
    print("%-34s cells %5d (x%.2f) recall %.3f precision %.3f mixed %.3f median_major %.3f pure>=.9 %.3f" % (label, len(cb), len(cb) / 1020, inter / 1020, inter / max(1, len(cb)), (mn > .1).mean(), np.median(mj), (mj >= .9).mean()))
thr = cr22(u); print("uncollapsed CR2.2 knee thr", thr)
score(np.where(u >= thr)[0], "uncollapsed knee")
for ratio in (2, 3, 4, 6, 10):
    par = collapse(ratio)
    roots = (par == np.arange(n))
    # merged totals (sum of children into root)
    mt = np.zeros(n); np.add.at(mt, par, u)
    thr = cr22(mt[roots])
    called = np.where(roots & (mt >= thr))[0]
    # use merged counts for composition: approximate by root column plus children columns
    cb = called
    cols = {}
    for i in np.where(par != np.arange(n))[0]:
        if par[i] in set(called[:0]): pass
    ch = {}
    for i in np.where(par != np.arange(n))[0]:
        ch.setdefault(par[i], []).append(i)
    # build composition
    h = np.zeros(len(called)); m = np.zeros(len(called))
    Mh = Mx[~mouse]; Mm = Mx[mouse]
    hs = np.asarray(Mh.sum(axis=0)).ravel(); ms = np.asarray(Mm.sum(axis=0)).ravel()
    for k, r in enumerate(called):
        h[k] = hs[r] + sum(hs[c] for c in ch.get(r, [])); m[k] = ms[r] + sum(ms[c] for c in ch.get(r, []))
    t = h + m; ok = t > 0; mn = np.minimum(h, m)[ok] / t[ok]; mj = np.maximum(h, m)[ok] / t[ok]
    inter = sum(bcs[r] in aset for r in called)
    print("collapse ratio>=%-3d thr %5d cells %5d (x%.2f) recall %.3f precision %.3f mixed %.3f median_major %.3f pure>=.9 %.3f  n_children %d" % (ratio, thr, len(called), len(called) / 1020, inter / 1020, inter / len(called), (mn > .1).mean(), np.median(mj), (mj >= .9).mean(), int((par != np.arange(n)).sum())))
    # EmptyDrops-free alternative: also remove very mixed? (report mixed of removed)
    # rank curve after collapse
    s = np.sort(mt[roots])[::-1]
    print("   merged rank curve: 500 %d 1000 %d 1500 %d 2000 %d 3000 %d" % (s[500], s[1000], s[1500], s[2000], s[3000]))
