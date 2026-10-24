#!/usr/bin/env python
"""Track D, task 4b: per-chunk code length under *models* with different inductive bias, to join with the
entropy-coder sizes in results/track_d_oracle_chunks.tsv for a transform-level orchestrator oracle.

For every file: 2-fold CV over columns (even / odd), so every column is scored by a model fitted on the other
half (a proxy for a decoder that builds the model from already-decoded columns):
  M2      per-feature x depth-bucket presence + rate-context value table
  mult1   one pooled multinomial profile (all columns)
  mult16  mixture of 16 multinomial profiles (EM, cosine k-means++ seeds)
  amb     one multinomial profile fitted on the training-half EMPTIES only (the ambient model), applied to every column
Each model's per-column bits are summed per 1024-column chunk; side information (order-0 cost of nnz_j and k_j)
is added so the totals are comparable with the coder sizes (whose metadata carries the same information).
Output: results/track_d_oracle_models.tsv  (sample member chunk ncols nnz side_bits M2 mult1 mult16 amb)

    python -P tools/track_d_oracle_models.py [sample:member ...]
"""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import track_d_cells as tc
from track_d_common import corpus_files, load, ROOT
from track_d_lib import *

OUT = os.path.join(ROOT, 'results', 'track_d_oracle_models.tsv')
CHUNK = 1024


def main():
    want = sys.argv[1:]
    new = not os.path.exists(OUT)
    fo = open(OUT, 'a')
    if new:
        fo.write('sample\tmember\tchunk\tncols\tnnz\tside_bits\tM2\tmult1\tmult16\tamb\n')
    for s, mem, p in corpus_files():
        key = f'{s}:{mem}'
        if want and key not in want and s not in want:
            continue
        t0 = time.time()
        d = load(s, mem)
        m = int(d['m']); n = int(d['n'])
        ic = d['is_cell']
        cc = np.diff(d['indptr'])
        col = columns_of(d)
        k_col = np.bincount(col, weights=d['data'], minlength=n).astype(np.int64)
        rn = np.bincount(d['indices'], minlength=m)
        fmap = np.full(m, -1, dtype=np.int64)
        F = int((rn > 0).sum())
        fmap[rn > 0] = np.arange(F)
        base = dict(indptr=d['indptr'], indices=fmap[d['indices']], data=d['data'], m=F)
        # side info: order-0 over the file of nnz_j and k_j (what the coder metadata carries)
        def h0_percol(x):
            c = np.bincount(x); pr = c / c.sum()
            return -np.log2(pr[x])
        side = h0_percol(cc) + h0_percol(k_col)
        cost = {k: np.zeros(n) for k in ('M2', 'mult1', 'mult16', 'amb')}
        even = np.arange(n) % 2 == 0
        for trm in (even, ~even):
            tem = ~trm
            tr_idx = np.flatnonzero(trm & (cc > 0)); te_idx = np.flatnonzero(tem & (cc > 0))
            if len(tr_idx) < 10 or len(te_idx) == 0:
                continue
            tr = tc.take_cols(base, tr_idx); te = tc.take_cols(base, te_idx)
            # M2 (alphas chosen on the training fold by 2-way split inside it would be cleaner; use fixed defaults from the cells runs)
            a1 = 300.0; a2 = 1000.0
            pc2 = presence_percell_featdepth(tr, te, a2, a1) + np.bincount(te['col'], weights=value_cost_entries(tr, te, rate_ctx_factory(tr)), minlength=te['n'])
            cost['M2'][te_idx] = pc2
            q = tc.pooled_profile(tr, 0.3)
            cost['mult1'][te_idx] = tc.mult_cost_cells(te, np.log2(q)[te['indices']])
            C = 16
            if tr['n'] >= 8 * C:
                N, Nc, w, qq = tc.em_mixture(tr, C, 0.3, seed=C)
                cost['mult16'][te_idx] = tc.mixture_cells_cost(te, N, Nc, w, qq, 3000.0)
            else:
                cost['mult16'][te_idx] = cost['mult1'][te_idx]
            # ambient profile from the training-half empties
            emp_tr = np.flatnonzero(trm & (cc > 0) & (~ic))
            if len(emp_tr) >= 10:
                tre = tc.take_cols(base, emp_tr)
                qa = tc.pooled_profile(tre, 0.1)
                cost['amb'][te_idx] = tc.mult_cost_cells(te, np.log2(qa)[te['indices']])
            else:
                cost['amb'][te_idx] = cost['mult1'][te_idx]
        nch = (n + CHUNK - 1) // CHUNK
        for c in range(nch):
            lo, hi = c * CHUNK, min(n, (c + 1) * CHUNK)
            sl = slice(lo, hi)
            fo.write(f'{s}\t{mem}\t{c}\t{hi - lo}\t{int(cc[sl].sum())}\t{side[sl].sum():.1f}\t' + '\t'.join(f'{cost[k][sl].sum():.1f}' for k in ('M2', 'mult1', 'mult16', 'amb')) + '\n')
        fo.flush()
        print(f'{key} done in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
