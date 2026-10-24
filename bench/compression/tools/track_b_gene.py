# Does grouping exon rows by gene (rows are contiguous per gene in original order) cut the index cost?
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from singlet._pz import read_1pz
from track_b_bern import lb2, gsym   # reuse

def H(counts):
    counts = np.asarray(counts, float); t = counts.sum()
    c = counts[counts > 0]
    return -(c * np.log2(c / t)).sum()

def ent_ctx(ctx, sym, nsym):
    # sum over contexts of entropy of sym given ctx
    key = ctx.astype(np.int64) * nsym + sym
    cnt = np.bincount(key)
    cnt = np.pad(cnt, (0, (-len(cnt)) % nsym)).reshape(-1, nsym)
    tot = cnt.sum(1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        l = np.where(cnt > 0, cnt * np.log2(cnt / tot), 0)
    return -l.sum(), int((cnt > 0).sum())

def run(path):
    r = read_1pz(path)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']).astype(np.int64)
    n, m = r['n'], r['m']
    cc = np.diff(ip); col = np.repeat(np.arange(n), cc)
    names = r['rownames']
    gid = []
    prev = None; g = -1
    gidx = np.empty(m, np.int64); first_of = []
    for i, nm in enumerate(names):
        p = nm.split('_')[0]
        if p != prev:
            g += 1; prev = p; first_of.append(i)
        gidx[i] = g
    G = g + 1
    first_of = np.array(first_of); e = np.diff(np.append(first_of, m))
    ii = np.arange(m) - first_of[gidx]
    nnz = r['nnz']
    # gene-level presence per (col, gene)
    gg = gidx[ix]
    key = col * G + gg
    ukey = np.unique(key)
    ucol = ukey // G; ug = ukey % G
    gfreq = np.bincount(ug, minlength=G)
    order = np.lexsort((np.arange(G), -gfreq)); grank = np.empty(G, np.int64); grank[order] = np.arange(G)
    gr = grank[ug]
    o = np.lexsort((gr, ucol)); gr = gr[o]; ucol = ucol[o]; ug = ug[o]
    kg = np.bincount(ucol, minlength=n)
    kb = np.minimum(np.floor(np.log2(kg + 1)).astype(np.int64), 11)
    first = np.ones(len(gr), bool); first[1:] = ucol[1:] != ucol[:-1]
    prev = np.where(first, -1, np.roll(gr, 1))
    gap = gr - prev - 1
    s, ex = gsym(gap)
    pb = np.minimum(lb2(prev + 1), 39)
    ctx = pb * 12 + kb[ucol]
    bits_g, cells = ent_ctx(ctx, s, 80)
    bits_g += ex.sum()
    # counts of genes per column cost ~ small
    # mask bits: for each present (col,gene) an e-bit mask, bit i = exon i hit
    # build hit matrix on the fly: set of (col,row)
    rowkey = col * m + ix
    hitset = np.sort(rowkey)
    # for each present gene unit and each exon i of gene
    ne = e[ug]
    tot = int(ne.sum())
    unit = np.repeat(np.arange(len(ug)), ne)
    off = np.arange(tot) - np.repeat(np.cumsum(ne) - ne, ne)
    rows = first_of[ug][unit] + off
    q = ucol[unit] * m + rows
    pos = np.searchsorted(hitset, q)
    pos[pos >= len(hitset)] = len(hitset) - 1
    bit = (hitset[pos] == q).astype(np.int64)
    # context: (min(e,12), min(off,12), prev bit within unit)
    prevbit = np.zeros(tot, np.int64)
    prevbit[1:] = bit[:-1]
    prevbit[off == 0] = 2
    # ones so far (cap 2)
    cs = np.cumsum(bit); base = np.repeat(cs[np.cumsum(ne) - ne] - bit[np.cumsum(ne) - ne], ne)
    so_far = np.minimum(cs - bit - base, 2)
    # forced last bit: if no ones so far and this is last exon -> bit must be 1 (free)
    lastforced = (off == ne[unit] - 1) & (so_far == 0)
    for name, c in [("e,i", np.minimum(ne[unit], 12) * 13 + np.minimum(off, 12)),
                    ("e,i,prev", (np.minimum(ne[unit], 12) * 13 + np.minimum(off, 12)) * 3 + prevbit),
                    ("e,i,prev,sofar", ((np.minimum(ne[unit], 12) * 13 + np.minimum(off, 12)) * 3 + prevbit) * 3 + so_far),
                    ("e,i,prev,sofar,kb", (((np.minimum(ne[unit], 12) * 13 + np.minimum(off, 12)) * 3 + prevbit) * 3 + so_far) * 12 + kb[ucol[unit]])]:
        mask = ~lastforced
        bm, cl = ent_ctx(c[mask], bit[mask], 2)
        print(f"{os.path.basename(os.path.dirname(path))}: gene units {len(ug)} ({len(ug)/nnz:.2f}/nnz) gene-gap {bits_g/nnz:.3f} + mask[{name}] {bm/nnz:.3f} = {(bits_g+bm)/nnz:.3f} bits/nnz idx (cells g={cells}, m={cl}); cols n={n} G={G}")

for p in sys.argv[1:]:
    run(p)
