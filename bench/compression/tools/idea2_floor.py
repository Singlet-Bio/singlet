"""Exact multinomial entropy floor for the empty droplets of each corpus file.

For droplet j with k_j UMIs drawn from profile p (m features), H(Mult(k_j, p)) =
    -log2 k_j!  -  k_j * sum_f p_f log2 p_f  +  sum_f E[log2 X_f!],   X_f ~ Binomial(k_j, p_f)
(the x_f are the counts; the identity follows from -E[log pmf] with the multinomial coefficient).
E[log2 X_f!] is computed exactly with the binomial pmf, truncated where the pmf is < 1e-12.
Two profiles: p_hat = pooled empties MLE (optimistic, "lower-bound flavour"), p_mix = (empties + cells*gamma)
as a smoothed profile. Per file also the order-0 entropy of k_j (what any code must spend on the depth).

Usage: python -P tools/idea2_floor.py > results/idea2_floor.tsv
"""
import sys, math, time
import numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz

SAMPLES = ['pbmc_v2_typical', 'pbmc_v2_shallow', 'pbmc_v3_lowgenes', 'brain_sn_v3_deep', 'brain_sn_v3_intronic', 'tcell_5p', 'tumor_3p']

def log2fact(n):
    return np.array([math.lgamma(x + 1) for x in np.asarray(n, dtype=np.float64)]) / math.log(2)

def floor_bits(k, p):
    """sum_j H(Mult(k_j, p)) in bits, exact per unique depth."""
    p = p[p > 0]
    lp = np.log2(p)
    H1 = -(p * lp).sum()                       # order-0 entropy of one draw
    uk, cnt = np.unique(k, return_counts=True)
    tot = 0.0
    lf = np.zeros(64)
    for x in range(1, 64): lf[x] = lf[x - 1] + math.log2(x)
    for kk, c in zip(uk, cnt):
        kk = int(kk)
        # sum_f E[log2 X!] with X ~ Bin(kk, p_f): iterate x = 2.. while terms matter
        q = 1.0 - p
        pmf = q ** kk                               # x = 0
        s = np.zeros_like(p)
        x = 0
        while x < min(kk, 63):
            # pmf(x+1) = pmf(x) * (kk - x)/(x+1) * p/q
            pmf = pmf * ((kk - x) / (x + 1)) * (p / q)
            x += 1
            if x >= 2:
                s += pmf * lf[x]
            if x >= 8 and pmf.max() < 1e-13:
                break
        h = -math.lgamma(kk + 1) / math.log(2) + kk * H1 + s.sum()
        tot += c * h
    return tot

def h0(x):
    _, c = np.unique(x, return_counts=True)
    pp = c / c.sum()
    return -(c * np.log2(pp)).sum()

print('sample\tmember\tnempt\tnnz_empt\tumi_empt\tfloor_mle_bits\tfloor_mle_b_nnz\tfloor_mix_bits\tfloor_mix_b_nnz\tH0_k_bits\tH0_k_b_nnz')
for sid in SAMPLES:
    calls = set(l.split('\t')[0] for l in open(f'corpus/{sid}/cell_calls.tsv').read().splitlines()[1:])
    for mem in ['exon', 'intron']:
        t0 = time.time()
        r = read_1pz(f'corpus/{sid}/{mem}_counts.1pz')
        m, n = r['m'], r['n']
        ip = np.asarray(r['indptr']).astype(np.int64); ix = np.asarray(r['indices']).astype(np.int64); d = np.asarray(r['data']).astype(np.int64)
        isc = np.array([c in calls for c in r['colnames']])
        colid = np.repeat(np.arange(n), np.diff(ip))
        em = ~isc[colid]
        k = np.bincount(colid, weights=d, minlength=n).astype(np.int64)
        kemp = k[~isc]; kemp = kemp[kemp > 0]
        T = np.bincount(ix[em], weights=d[em], minlength=m)
        C = np.bincount(ix[~em], weights=d[~em], minlength=m)
        nnz_e = int(em.sum())
        p_mle = T / T.sum()
        gamma = 1e5 * (T.sum() / 5e6)
        p_mix = (T + gamma * (C + 0.1) / (C.sum() + 0.1 * m)) / (T.sum() + gamma)
        f1 = floor_bits(kemp, p_mle)
        f2 = floor_bits(kemp, p_mix)
        hk = h0(kemp)
        print(f'{sid}\t{mem}\t{len(kemp)}\t{nnz_e}\t{int(T.sum())}\t{f1:.0f}\t{f1/max(1,nnz_e):.4f}\t{f2:.0f}\t{f2/max(1,nnz_e):.4f}\t{hk:.0f}\t{hk/max(1,nnz_e):.4f}')
        sys.stdout.flush()
        print(f'  {sid} {mem} {time.time()-t0:.1f}s', file=sys.stderr)
