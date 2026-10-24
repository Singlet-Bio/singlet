# Track B model experiments (entropy estimates, static tables, in-sample unless noted): values, row order, orientation.
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz


def lb2(x):
    x = np.asarray(x, dtype=np.int64) + 2
    l = np.floor(np.log2(x)).astype(np.int64)
    return 2 * l + ((x >> np.maximum(l - 1, 0)) & 1) - 2


def gcls(g):
    g = np.asarray(g, dtype=np.int64)
    b = np.floor(np.log2(np.maximum(g, 1))).astype(np.int64)
    s = np.where(g < 8, g, 8 + (b - 3) * 4 + ((g >> np.maximum(b - 2, 0)) & 3))
    ex = np.where(g < 8, 0, b - 2)
    return s, ex


def ent(ctx, sym, nsym=None):
    """sum over contexts of H(sym|ctx) in bits (static, in-sample) and number of nonzero table cells"""
    ctx = np.asarray(ctx, dtype=np.int64)
    sym = np.asarray(sym, dtype=np.int64)
    if nsym is None:
        nsym = int(sym.max()) + 1
    key = ctx * nsym + sym
    u, c = np.unique(key, return_counts=True)
    cc = u // nsym
    uc, inv = np.unique(cc, return_inverse=True)
    tot = np.bincount(inv, weights=c)
    t = tot[inv]
    return float(-(c * np.log2(c / t)).sum()), len(u)


class Mat:
    def __init__(self, path):
        r = read_1pz(path)
        self.path = path
        self.m, self.n, self.nnz = r['m'], r['n'], r['nnz']
        ip = np.asarray(r['indptr'], dtype=np.int64)
        self.cc = np.diff(ip)
        self.col = np.repeat(np.arange(self.n), self.cc)
        self.ix = np.asarray(r['indices']).astype(np.int64)
        self.val = np.asarray(r['data']).astype(np.int64)
        self.names = r['rownames']
        self.rn = np.bincount(self.ix, minlength=self.m)
        self.kb = np.minimum(np.floor(np.log2(self.cc + 1)).astype(np.int64), 11)

    def rank_by(self, key_desc):
        order = np.lexsort((np.arange(self.m), -key_desc))
        rank = np.empty(self.m, np.int64)
        rank[order] = np.arange(self.m)
        return rank

    def gap_stream(self, rank):
        rk = rank[self.ix]
        o = np.lexsort((rk, self.col))
        rk = rk[o]
        col = self.col[o]
        val = self.val[o]
        first = np.ones(len(rk), bool)
        first[1:] = col[1:] != col[:-1]
        prev = np.where(first, -1, np.roll(rk, 1))
        return rk, col, val, prev, first


def gap_bits(M, rank):
    rk, col, val, prev, first = M.gap_stream(rank)
    g = rk - prev - 1
    s, ex = gcls(g)
    ctx = lb2(prev + 1) * 12 + M.kb[col]
    b, cells = ent(ctx, s, 80)
    return (b + ex.sum()) / M.nnz, cells


def dk_of(cc):
    return np.minimum((np.floor(np.log2(cc + 1)).astype(np.int64) * 11) >> 5, 3)


def value_experiments(M):
    rank = M.rank_by(M.rn)
    rk, col, val, prev, first = M.gap_stream(rank)
    nn = M.nnz
    flag = (val > 1).astype(np.int64)
    g = rk - prev - 1
    s, ex = gcls(g)
    pb_prev = lb2(prev + 1)
    kb = M.kb[col]
    base, _ = ent(pb_prev * 12 + kb, s, 80)
    joint, cj = ent(pb_prev * 12 + kb, s * 2 + flag, 160)
    print(f"  joint-flag cost = H(cls,flag|ctx)-H(cls|ctx) = {(joint - base) / nn:.3f} bits/nnz (cells {cj})")
    pb_own = lb2(rk)
    pf = np.where(first, 0, np.roll(flag, 1))
    for name, ctx in [("own rank bucket x k", pb_own * 12 + kb), ("own rank bucket", pb_own), ("none", np.zeros_like(pb_own)),
                      ("own x k x prev flag", (pb_own * 12 + kb) * 2 + pf)]:
        b, c = ent(ctx, flag, 2)
        print(f"  flag separate [{name}]: {b / nn:.3f} bits/nnz (cells {c})")
    fm = flag == 1
    r = val[fm] - 2
    rs, rex = gcls(r)
    dk = dk_of(M.cc[col[fm]])
    for name, ctx in [("own/2 x dk", pb_own[fm] // 2 * 4 + dk), ("own x kb", pb_own[fm] * 12 + kb[fm]), ("none", np.zeros(fm.sum(), np.int64)), ("dk", dk)]:
        b, c = ent(ctx, rs, 80)
        print(f"  residual [{name}]: {(b + rex.sum()) / nn:.3f} bits/nnz (cells {c})")
    vs, vex = gcls(val - 1)
    for name, ctx in [("own x kb", pb_own * 12 + kb), ("own/2 x dk", pb_own // 2 * 4 + dk_of(M.cc[col]))]:
        b, c = ent(ctx, vs, 80)
        print(f"  single-symbol value class [{name}]: {(b + vex.sum()) / nn:.3f} bits/nnz (cells {c})")
    best_total = 0.0
    v1 = val - 1
    for bkt in np.unique(pb_own):
        x = v1[pb_own == bkt]
        best_total += min((x >> kk).sum() + len(x) * (1 + kk) for kk in range(0, 8))
    print(f"  Rice code on (v-1), best k per own-rank bucket: {best_total / nn:.3f} bits/nnz")
    # group-by-value (VOCSC-style): sets per value, gap-coded in rank space
    order = np.lexsort((rk, val, col))
    rk2 = rk[order]
    col2 = col[order]
    val2 = val[order]
    newgrp = np.ones(len(rk2), bool)
    newgrp[1:] = (col2[1:] != col2[:-1]) | (val2[1:] != val2[:-1])
    prev2 = np.where(newgrp, -1, np.roll(rk2, 1))
    gid = np.cumsum(newgrp) - 1
    gsz = np.bincount(gid)[gid]
    g2 = rk2 - prev2 - 1
    s2, ex2 = gcls(g2)
    kbg = np.minimum(np.floor(np.log2(gsz + 1)).astype(np.int64), 11)
    ctx2 = (lb2(prev2 + 1) * 12 + kbg) * 4 + np.minimum(val2 - 1, 3)
    b, c = ent(ctx2, s2, 80)
    ngroups = int(newgrp.sum())
    hs, hex_ = gcls(gsz[newgrp])
    hb, _ = ent(np.minimum(val2[newgrp] - 1, 7), hs, 80)
    print(f"  group-by-value (VOCSC-style) index {(b + ex2.sum()) / nn:.3f} + headers ~{(hb + hex_.sum() + ngroups * 1.5) / nn:.3f} bits/nnz ({ngroups} groups)"
          f" ; vs joint index+flag {(joint + ex.sum()) / nn:.3f} + residual")


def order_experiments(M):
    gene = np.array([n.split('_')[0] for n in M.names])
    _, gidx = np.unique(gene, return_inverse=True)
    G = gidx.max() + 1
    gnn = np.bincount(gidx, weights=M.rn, minlength=G)
    res = {}
    res["freq (baseline)"] = gap_bits(M, M.rank_by(M.rn))
    res["orig order"] = gap_bits(M, np.arange(M.m))
    res["gene-freq, exons by freq"] = gap_bits(M, M.rank_by(gnn[gidx] * 1e7 + M.rn))
    order = np.lexsort((np.arange(M.m), -gnn[gidx]))
    r = np.empty(M.m, np.int64)
    r[order] = np.arange(M.m)
    res["gene-freq, exons genomic"] = gap_bits(M, r)
    for nm, mult in [("freq octave class, orig inside", 1), ("freq half-octave class, orig inside", 2), ("freq 4/octave class, orig inside", 4)]:
        cls = np.floor(mult * np.log2(M.rn + 1)).astype(np.int64)
        order = np.lexsort((np.arange(M.m), -cls))
        r = np.empty(M.m, np.int64)
        r[order] = np.arange(M.m)
        res[nm] = gap_bits(M, r)
    rng = np.random.default_rng(0)
    res["random"] = gap_bits(M, rng.permutation(M.m))
    for k, (b, c) in res.items():
        print(f"  row order [{k:38s}] gap index {b:.3f} bits/nnz (cells {c})")


def csr_experiment(M):
    corder = np.lexsort((np.arange(M.n), -M.cc))
    crank = np.empty(M.n, np.int64)
    crank[corder] = np.arange(M.n)
    cr = crank[M.col]
    o = np.lexsort((cr, M.ix))
    ixs = M.ix[o]
    crs = cr[o]
    first = np.ones(len(ixs), bool)
    first[1:] = ixs[1:] != ixs[:-1]
    prev = np.where(first, -1, np.roll(crs, 1))
    g = crs - prev - 1
    s, ex = gcls(g)
    rnb = np.minimum(np.floor(np.log2(M.rn[ixs] + 1)).astype(np.int64), 20)
    ctx = lb2(prev + 1) * 24 + rnb
    b, c = ent(ctx, s, 80)
    cs, cex = gcls(M.rn)
    hb, _ = ent(np.zeros(M.m, np.int64), cs, 80)
    print(f"  CSR (cells ordered by depth): gap index [pos x rownnz] {(b + ex.sum()) / M.nnz:.3f} + row counts {(hb + cex.sum()) / M.nnz:.3f} bits/nnz (cells {c})")
    b2, c2 = gap_bits(M, M.rank_by(M.rn))
    cs2, cex2 = gcls(M.cc)
    hb2, _ = ent(np.zeros(M.n, np.int64), cs2, 80)
    print(f"  CSC (rows by freq): gap index [pos x k] {b2:.3f} + col counts {(hb2 + cex2.sum()) / M.nnz:.3f} bits/nnz (cells {c2})")


if __name__ == '__main__':
    what = sys.argv[1]
    for p in sys.argv[2:]:
        M = Mat(p)
        print(f"== {os.path.basename(os.path.dirname(p))}/{os.path.basename(p)} m={M.m} n={M.n} nnz={M.nnz}")
        {'values': value_experiments, 'order': order_experiments, 'csr': csr_experiment}[what](M)
