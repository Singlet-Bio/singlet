"""Rung 4 evidence: non-called ('empty') columns: our actual bits vs multinomial-from-ambient bound (split-half estimated profile)."""
import sys, os, numpy as np
from scipy.special import gammaln
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
def run(sample, mem, colcost_file):
    d = get(f'{sample}__{mem}'); m = d['m']; ip = d['indptr']; n = d['n']
    cols = open(os.path.join(SCR, f'{sample}__{mem}.cols.txt')).read().split('\n')
    calls = set()
    corp = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'corpus', sample, 'cell_calls.tsv')
    for line in open(corp).read().split('\n')[1:]:
        if line: 
            f = line.split('\t'); 
            if f[1] == 'TRUE': calls.add(f[0])
    is_cell = np.array([c in calls for c in cols])
    bits = np.fromfile(colcost_file, dtype=np.float32).astype(np.float64)
    nz = d['nz']; tot = d['tot']
    nnz_all = nz.sum()
    emp = (~is_cell) & (nz > 0)
    print(f"{sample} {mem}: columns {n}, called {is_cell.sum()}, non-called with nnz>0 {emp.sum()}; nnz share non-called {nz[emp].sum()/nnz_all:.3f}; our cost share {bits[emp].sum()/bits.sum():.3f}")
    print(f"   called:     {bits[is_cell].sum()/nz[is_cell].sum():.3f} bits/nnz;  non-called: {bits[emp].sum()/nz[emp].sum():.3f} bits/nnz  (median nnz non-called {np.median(nz[emp]):.0f}, mean UMI {tot[emp].mean():.2f})")
    # multinomial bound on non-called, split half
    ids = np.flatnonzero(emp); rng = np.random.default_rng(1); rng.shuffle(ids)
    A, B = ids[:len(ids)//2], ids[len(ids)//2:]
    def pooled(cs):
        q = np.zeros(m)
        for c in cs:
            s, e = ip[c], ip[c+1]; q[d['indices'][s:e]] += d['data'][s:e]
        return q
    qa = pooled(A); support = np.bincount(d['indices'], minlength=m) > 0
    alpha = 0.05
    q = (qa + alpha * support); q /= q.sum()
    lq = np.log(q)
    tot_bits = 0.0; ntot = 0
    ks = []
    for c in B:
        s, e = ip[c], ip[c+1]; cnt = d['data'][s:e].astype(np.float64); ix = d['indices'][s:e]
        k = cnt.sum(); ks.append(int(k))
        ll = gammaln(k + 1) - gammaln(cnt + 1).sum() + (cnt * lq[ix]).sum()
        tot_bits += -ll / np.log(2); ntot += len(cnt)
    ks = np.array(ks); u, cn = np.unique(ks, return_counts=True); Hk = -(cn * np.log2(cn / cn.sum())).sum()
    ours = bits[B].sum()
    print(f"   non-called held-out half: multinomial-ambient {tot_bits/ntot:.3f} bits/nnz (+ total-UMI cost {Hk/ntot:.3f}) = {(tot_bits+Hk)/ntot:.3f};  ours {ours/ntot:.3f} bits/nnz")
if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])
