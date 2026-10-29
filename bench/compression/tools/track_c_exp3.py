"""Gene-level (multinomial split) vs independent-exon cross-entropy, out of sample. Presence/value tables by (bucket, class)."""
import sys, os, numpy as np
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *

LN2 = np.log(2)
def smooth_cond(cnt, kappa=8.0):
    """cnt[ctx, sym] training counts -> P[ctx,sym], smoothed toward marginal over ctx (all contexts pooled in rows of same bucket handled by caller)"""
    marg = cnt.sum(0) + 1e-3; marg = marg / marg.sum()
    tot = cnt.sum(1, keepdims=True)
    return (cnt + kappa * marg[None, :]) / (tot + kappa)

def pava_dec(y, w):
    """isotonic regression, non-increasing y (weights w)"""
    y = list(y); w = list(w); idx = []
    vals = []; ws = []; sizes = []
    for yi, wi in zip(y, w):
        vals.append(yi); ws.append(wi); sizes.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:   # violation for non-increasing
            v = (vals[-2]*ws[-2] + vals[-1]*ws[-1]) / max(ws[-2] + ws[-1], 1e-12)
            ws[-2] += ws[-1]; sizes[-2] += sizes[-1]; vals[-2] = v
            vals.pop(); ws.pop(); sizes.pop()
    out = []
    for v, s in zip(vals, sizes): out += [v] * s
    return np.array(out)

def ge_class(share, nlev, step, top):
    c = np.full(len(share), nlev - 1, dtype=np.int32)
    ok = share > 0
    c[ok] = np.clip(np.round((top - np.log2(share[ok])) / step), 0, nlev - 2).astype(np.int32)
    return c

def run(test, mem, nb=32, step=0.5, U=6, kappa=4.0):
    org = ORG[test]; g, ng = gene_index(org, mem)
    pool = [k for k in STUDY if ORG[k] == org and STUDY[k] != STUDY[test]]
    names = [f'{k}__{mem}' for k in pool]
    m = get(names[0])['m']
    dT = get(f'{test}__{mem}'); nnzT = len(dT['indices'])
    scheme = ('nz', nb, 0, 14)
    # ---- exon weights within gene (UMI share), gene share
    Wf = np.zeros(m); 
    for nm in names:
        d = get(nm); Wf += np.bincount(d['indices'], weights=d['data'].astype(np.float64), minlength=m) / d['data'].sum()
    Wf /= len(names)
    Wg = np.bincount(g, weights=Wf, minlength=ng)
    ng_ex = np.bincount(g, minlength=ng)
    beta = 1e-7   # tiny prior mass; unseen exons get w from uniform within gene
    w_e = (Wf + beta / ng_ex[g]) / (Wg[g] + beta)
    # ---- gene class from share of UMIs
    top = np.log2(Wg.max()); nlev = 40
    gcls = ge_class(Wg, nlev, step, top)
    # ---- build gene-level (col,gene) groups for a data dict
    def groups(d):
        col = d['col']; gi = g[d['indices']]
        key = col.astype(np.int64) * ng + gi
        # entries sorted by col then row; gene is monotone in row => keys sorted
        st = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
        u = np.add.reduceat(d['data'].astype(np.int64), st)
        gcol = col[st]; ggene = gi[st]
        return st, u, gcol, ggene
    # ---- training counts for gene-level u tables: [bucket, class, sym(0..U)] with u capped at U
    def u_counts(d, gcl):
        st, u, gcol, ggene = groups(d)
        b = bucketize(d, scheme)
        key = (b[gcol].astype(np.int64) * nlev + gcl[ggene]) * (U + 1) + np.minimum(u, U)
        C = np.bincount(key, minlength=nb * nlev * (U + 1)).reshape(nb, nlev, U + 1).astype(np.float64)
        Nb = np.bincount(b, minlength=nb).astype(np.float64)
        mcg = np.bincount(gcl, minlength=nlev).astype(np.float64)
        C[:, :, 0] = Nb[:, None] * mcg[None, :] - C[:, :, 1:].sum(2)
        return C, b
    # calibrate with leave-one-study-out within pool (class from other-study share)
    Ccal = np.zeros((nb, nlev, U + 1))
    studies = sorted(set(STUDY[k] for k in pool))
    for st in studies:
        inn = [f'{k}__{mem}' for k in pool if STUDY[k] != st]; out = [f'{k}__{mem}' for k in pool if STUDY[k] == st]
        Wf_in = np.zeros(m)
        for nm in inn:
            d = get(nm); Wf_in += np.bincount(d['indices'], weights=d['data'].astype(np.float64), minlength=m) / d['data'].sum()
        Wg_in = np.bincount(g, weights=Wf_in / len(inn), minlength=ng)
        gc_in = ge_class(Wg_in, nlev, step, top)
        for nm in out:
            C, _ = u_counts(get(nm), gc_in); Ccal += C
    # tail-aware smoothing: per-bucket, monotone-in-class P(u>=1), shape of u|u>=1 pooled per bucket
    P = np.zeros_like(Ccal)
    for bi in range(nb):
        tot = Ccal[bi].sum(1); pres = tot - Ccal[bi, :, 0]
        rate = pres / np.maximum(tot, 1)
        seen = np.arange(nlev - 1)
        rate_m = pava_dec(rate[:nlev-1], tot[:nlev-1]) if tot[:nlev-1].sum() > 0 else rate[:nlev-1]
        rate_m = np.maximum(rate_m, 1e-9)
        rr = np.concatenate([rate_m, [max(rate[nlev-1], 1e-9)]])
        shape = Ccal[bi, :, 1:]
        marg = shape.sum(0) + 0.5; marg /= marg.sum()
        shp = (shape + kappa * marg[None, :]) / (shape.sum(1, keepdims=True) + kappa)
        P[bi, :, 0] = 1 - rr; P[bi, :, 1:] = rr[:, None] * shp
    P = np.maximum(P, 1e-9); P /= P.sum(2, keepdims=True)
    CT, b = u_counts(dT, gcls)
    bits_u = -(CT * np.log2(P)).sum()
    # ---- multinomial split cost
    st, u, gcol, ggene = groups(dT)
    cnt = dT['data'].astype(np.float64)
    lg = np.add.reduceat(gammaln(cnt + 1), st)
    wcost = np.add.reduceat(cnt * np.log(w_e[dT['indices']]), st)
    nll = -(gammaln(u + 1) - lg + wcost)
    bits_split = nll.sum() / LN2
    bb = entropy_bits(np.bincount(b, minlength=nb).astype(np.float64))
    print(f"{test:22s} {mem} GENE-MODEL u-tables {bits_u/nnzT:.3f} + split {bits_split/nnzT:.3f} + bucket {bb/nnzT:.3f} = {(bits_u+bits_split+bb)/nnzT:.3f} bits/nnz  (genes present groups {len(u)/nnzT:.3f} per nnz)", flush=True)

if __name__ == '__main__':
    mem = sys.argv[1]
    for t in (sys.argv[2:] or TESTS):
        run(t, mem)
