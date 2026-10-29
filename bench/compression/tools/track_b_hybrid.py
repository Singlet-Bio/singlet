# Hybrid orientation estimate: densest rows stored as raw bitmaps over the called cells (CSR-style), everything else as the
# usual CSC rank-gap stream. Entropy estimates (static tables, in-sample).
import sys, os
sys.path.insert(0, '../../python'); sys.path.insert(0, 'tools')
import numpy as np
from track_b_models import Mat, lb2, gcls, ent


def gap_cost(M, keep_mask_nnz):
    """cost (bits/nnz of ORIGINAL nnz) of the rank-gap stream restricted to the kept entries; ranks recomputed over kept rows"""
    ix = M.ix[keep_mask_nnz]
    col = M.col[keep_mask_nnz]
    rn = np.bincount(ix, minlength=M.m)
    order = np.lexsort((np.arange(M.m), -rn))
    rank = np.empty(M.m, np.int64)
    rank[order] = np.arange(M.m)
    rk = rank[ix]
    o = np.lexsort((rk, col))
    rk = rk[o]
    col = col[o]
    first = np.ones(len(rk), bool)
    first[1:] = col[1:] != col[:-1]
    prev = np.where(first, -1, np.roll(rk, 1))
    s, ex = gcls(rk - prev - 1)
    kcnt = np.bincount(col, minlength=M.n)
    kb = np.minimum(np.floor(np.log2(kcnt + 1)).astype(np.int64), 11)
    b, c = ent(lb2(prev + 1) * 12 + kb[col], s, 80)
    # counts
    cs, cex = gcls(kcnt)
    hb, _ = ent(np.zeros(M.n, np.int64), cs, 80)
    return (b + ex.sum() + hb + cex.sum()) / M.nnz


def run(path, kmin=200):
    M = Mat(path)
    base = gap_cost(M, np.ones(M.nnz, bool))
    cells = M.cc >= kmin
    ncell = int(cells.sum())
    in_cell = cells[M.col]
    rn_cells = np.bincount(M.ix[in_cell], minlength=M.m)
    q = rn_cells / max(ncell, 1)
    print(f"== {os.path.basename(os.path.dirname(path))}/{os.path.basename(path)} nnz={M.nnz} cells(k>={kmin})={ncell} baseline rank-gap+counts {base:.3f} b/nnz")
    for theta in (0.15, 0.25, 0.4, 0.6):
        rows = q >= theta
        nr = int(rows.sum())
        if nr == 0:
            continue
        sel = rows[M.ix] & in_cell      # entries absorbed by bitmaps
        share = sel.sum() / M.nnz
        raw_bits = nr * ncell
        qq = np.clip(q[rows], 1e-9, 1 - 1e-9)
        ent_bits = float((-(qq * np.log2(qq) + (1 - qq) * np.log2(1 - qq))).sum() * ncell)
        rest = gap_cost(M, ~sel)
        # row list (which rows are bitmaps): ~ nr * log2(m/nr)+1.5 bits
        side = nr * (np.log2(M.m / nr) + 1.5)
        print(f"  theta {theta:.2f}: {nr:5d} bitmap rows absorb {100*share:4.1f}% of nnz | raw bitmaps {raw_bits/M.nnz:.3f} (entropy-coded {ent_bits/M.nnz:.3f}) + rest {rest:.3f} + rowlist {side/M.nnz:.3f}"
              f" => hybrid raw {(raw_bits + side)/M.nnz + rest:.3f}, hybrid entropy {(ent_bits + side)/M.nnz + rest:.3f} b/nnz")


if __name__ == '__main__':
    for p in sys.argv[1:]:
        run(p)
