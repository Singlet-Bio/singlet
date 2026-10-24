"""In-sample ceiling comparison: independent per-exon presence vs gene-hierarchical presence (gene presence + exon presence given previous exons)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_exp4 import rbin

def tab_cost(ctx, bit, nctx):
    """in-sample empirical conditional entropy (bits) of bit given ctx"""
    n1 = np.bincount(ctx, weights=bit, minlength=nctx); n = np.bincount(ctx, minlength=nctx).astype(float)
    n0 = n - n1
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(n1 > 0, n1 * np.log2(n1 / n), 0) + np.where(n0 > 0, n0 * np.log2(n0 / n), 0))
    return h.sum()

def run(test, mem, nb=32):
    d = get(f'{test}__{mem}'); org = ORG[test]; g, ng = gene_index(org, mem)
    m, n = d['m'], d['n']; nnz = len(d['indices'])
    b = bucketize(d, ('nz', nb, 0, 14)); col = d['col']; idx = d['indices']
    k = np.bincount(idx, minlength=m); ce = rbin(k, n)
    # --- I: independent
    A, Nb = counts_AN(d, b, ce, nb, 121); mc = np.bincount(ce, minlength=121).astype(float)
    M = Nb[:, None] * mc[None, :]; M[:, 0] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(A > 0, A * np.log2(A / M), 0) + np.where(M - A > 0, (M - A) * np.log2((M - A) / M), 0))
    cost_I = np.nansum(h)
    # --- G: gene-level presence
    gi = g[idx]; key = col.astype(np.int64) * ng + gi
    st = np.flatnonzero(np.concatenate([[True], key[1:] != key[:-1]]))      # first entry of each (col,gene) group
    gcol = col[st]; ggene = gi[st]
    kg = np.bincount(ggene, minlength=ng); cg = rbin(kg, n)
    Ag, Nbg = counts_AN({'col': gcol, 'indices': ggene}, b, cg, nb, 121); mcg = np.bincount(cg, minlength=121).astype(float)
    Mg = Nbg[:, None] * mcg[None, :]; Mg[:, 0] = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        hg = -(np.where(Ag > 0, Ag * np.log2(Ag / Mg), 0) + np.where(Mg - Ag > 0, (Mg - Ag) * np.log2((Mg - Ag) / Mg), 0))
    cost_gene = np.nansum(hg)
    # --- exon-level given gene present: enumerate all exons of each present (col,gene)
    exon_start = np.concatenate([[0], np.cumsum(np.bincount(g, minlength=ng))])   # exons contiguous per gene
    E = np.diff(exon_start)
    # within-gene order: by k_f descending (static, file-derived => side-info-free in this ideal count)
    order_key = np.lexsort((np.arange(m), -k, g))     # sort by gene, then -k
    rank_in_gene = np.empty(m, dtype=np.int64); rank_in_gene[order_key] = np.arange(m) - exon_start[g[order_key]]
    # for each present group: list of exons (E_g) -> bit present?
    cnt_groups = E[ggene]
    tot_pairs = cnt_groups.sum()
    print(f"  present (col,gene) groups {len(st)}, exon slots {tot_pairs} ({tot_pairs/nnz:.2f} per nnz)")
    gid = np.repeat(np.arange(len(st)), cnt_groups)
    off = np.arange(tot_pairs) - np.repeat(np.cumsum(cnt_groups) - cnt_groups, cnt_groups)   # position within enumeration (rank order)
    exon_by_rank = order_key                                    # exons sorted by gene then k desc; index = exon_start[g]+rank
    exon_id = exon_by_rank[exon_start[ggene][gid] + off]
    # presence bit: is (col, exon) present in d? membership via sorted key
    ekey = col.astype(np.int64) * m + idx       # sorted (col, row)
    qkey = gcol[gid].astype(np.int64) * m + exon_id
    pos = np.searchsorted(ekey, qkey); pos[pos >= len(ekey)] = len(ekey) - 1
    bit = (ekey[pos] == qkey).astype(np.float64)
    # previous-present count within the group (cumulative within group before this slot)
    cb = np.cumsum(bit) - bit
    gstart = np.repeat(np.cumsum(cnt_groups) - cnt_groups, cnt_groups)
    prev = (cb - cb[gstart]).astype(np.int64)
    prevc = np.minimum(prev, 3)
    # exon class: ratio of k_e / k_g  (static, via file)
    ratio = k[exon_id] / np.maximum(kg[g[exon_id]], 1)
    rcls = np.clip(np.round(-np.log2(np.maximum(ratio, 1e-6)) * 2), 0, 15).astype(np.int64)
    last = (off == cnt_groups[gid] - 1)
    forced = last & (prev == 0)
    ctx = ((b[gcol[gid]] // 4) * 16 + rcls) * 4 + prevc          # b8 x 16 x 4
    ctx = ctx * 2 + (np.minimum(cnt_groups[gid], 2) == 1)
    nctx = 8 * 16 * 4 * 2
    keep = ~forced
    cost_exon = tab_cost(ctx[keep], bit[keep], nctx)
    # alt context without prev (to show effect)
    ctx2 = ((b[gcol[gid]] // 4) * 16 + rcls) * 2 + (np.minimum(cnt_groups[gid], 2) == 1)
    cost_exon_noprev = tab_cost(ctx2[keep], bit[keep], 8 * 16 * 2)
    print(f"{test:20s} {mem}: I(indep) {cost_I/nnz:.3f} | G: gene {cost_gene/nnz:.3f} + exon|gene {cost_exon/nnz:.3f} = {(cost_gene+cost_exon)/nnz:.3f}  (exon|gene w/o prev-ctx {cost_exon_noprev/nnz:.3f} => {(cost_gene+cost_exon_noprev)/nnz:.3f})", flush=True)

if __name__ == '__main__':
    mem = sys.argv[1]
    for t in (sys.argv[2:] or TESTS): run(t, mem)
