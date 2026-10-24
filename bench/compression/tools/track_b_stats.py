import sys, glob, os
sys.path.insert(0, '../../python')
import numpy as np
from singlet._pz import read_1pz
for p in sorted(glob.glob('corpus/*/*_counts.1pz')):
    r = read_1pz(p)
    ip = np.asarray(r['indptr'], dtype=np.int64); ix = np.asarray(r['indices']); d = np.asarray(r['data'])
    m, n, nnz = r['m'], r['n'], r['nnz']
    cc = np.diff(ip)
    rn = np.bincount(ix, minlength=m)
    srt = np.sort(rn)[::-1]
    cs = np.cumsum(srt) / nnz
    ones = (d == 1).mean()
    print(f"{os.path.basename(os.path.dirname(p)):22s}{os.path.basename(p)[:6]} m={m} n={n} nnz={nnz} nnz/col mean={nnz/n:.1f} med={np.median(cc):.0f} "
          f"cols<=5:{(cc<=5).mean():.2f} cols>=500:{(cc>=500).mean():.2f} ones={ones:.3f} maxv={d.max()} "
          f"rows_used={int((rn>0).sum())} top256={cs[255]:.2f} top4096={cs[4095]:.2f} top32768={cs[32767]:.2f}")
