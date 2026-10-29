"""Does within-gene dependence help? Independent per-feature presence (class, bucket) vs presence given #earlier present features of the
same gene in the same column (context prevc in {0,1,2+}); in-sample tables, exact accounting of all slots."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin

def tabent(A, M):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.nansum(-(np.where(A > 0, A * np.log2(A / M), 0) + np.where(M - A > 0, (M - A) * np.log2((M - A) / M), 0)))

def run(test, mem, nb=32):
    d = get(f'{test}__{mem}'); org = ORG[test]; g, ng = gene_index(org, mem)
    m, n = d['m'], d['n']; idx, col = d['indices'], d['col']; nnz = len(idx)
    b = bucketize(d, ('nz', nb, 0, 14)); NB = nb + 1
    k = np.bincount(idx, minlength=m); ce = rbin(k, n); nc = 121
    mc = np.bincount(ce, minlength=nc).astype(float); Nb = np.bincount(b, minlength=NB).astype(float)
    # independent
    A = np.bincount(b[col].astype(np.int64) * nc + ce[idx], minlength=NB * nc).reshape(NB, nc).astype(float)
    M = Nb[:, None] * mc[None]; M[:, 0] = 1
    cost_I = tabent(A, M)
    # groups
    gi = g[idx]; key = col.astype(np.int64) * ng + gi
    st = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))
    first_exon = idx[st]; gcol = col[st]; ggene = gi[st]
    # prev=0 slots
    H1 = np.bincount(b[gcol].astype(np.int64) * m + first_exon, minlength=NB * m).reshape(NB, m).astype(float)
    cs = np.cumsum(H1, axis=1)
    gstart_row = np.flatnonzero(np.concatenate([[True], g[1:] != g[:-1]])); gid_row = np.cumsum(np.concatenate([[1], (g[1:] != g[:-1]).astype(int)])) - 1
    cs0 = np.concatenate([np.zeros((NB, 1)), cs], axis=1)           # cs0[:, r] = sum of H1 over rows < r
    C = cs0[:, :-1] - cs0[:, gstart_row][:, gid_row]                # sum over rows of same gene before this row
    N0 = Nb[:, None] - C                                            # slots with prev=0 for each (b,row)
    P0 = H1
    A0 = np.zeros((NB, nc)); M0 = np.zeros((NB, nc))
    for bi in range(NB):
        A0[bi] = np.bincount(ce, weights=P0[bi], minlength=nc); M0[bi] = np.bincount(ce, weights=N0[bi], minlength=nc)
    M0[:, 0] = 1; M0 = np.maximum(M0, A0)
    cost0 = tabent(A0, M0)
    # prev>=1 slots: expand exons after first present, per group
    E = np.bincount(g, minlength=ng); estart = np.concatenate([[0], np.cumsum(E)])
    after = estart[ggene + 1] - first_exon - 1                      # exons after the first present one
    tot = int(after.sum())
    gid = np.repeat(np.arange(len(st)), after)
    off = np.arange(tot) - np.repeat(np.cumsum(after) - after, after)
    ex = first_exon[gid] + 1 + off
    ekey = col.astype(np.int64) * m + idx
    qkey = gcol[gid].astype(np.int64) * m + ex
    pos = np.searchsorted(ekey, qkey); pos[pos >= len(ekey)] = len(ekey) - 1
    bit = (ekey[pos] == qkey).astype(np.int64)
    cb = np.cumsum(bit) - bit; gs = np.repeat(np.cumsum(after) - after, after)
    prev = np.minimum(1 + (cb - cb[gs]), 2)                         # first present counts as 1 earlier
    ctx = (b[gcol[gid]].astype(np.int64) * nc + ce[ex]) * 2 + (prev - 1)
    A1 = np.bincount(ctx, weights=bit, minlength=NB * nc * 2); M1 = np.bincount(ctx, minlength=NB * nc * 2).astype(float)
    cost1 = tabent(A1, np.maximum(M1, 1))
    print(f"{test:20s} {mem}: indep {cost_I/nnz:.3f} | prev-ctx: prev=0 {cost0/nnz:.3f} + prev>=1 {cost1/nnz:.3f} = {(cost0+cost1)/nnz:.3f}   (slots after first: {tot/nnz:.1f}/nnz)", flush=True)

if __name__ == '__main__':
    for t in sys.argv[2:]: run(t, sys.argv[1])
