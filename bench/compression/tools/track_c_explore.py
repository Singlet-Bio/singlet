import sys, numpy as np, glob, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_common import *
for f in sorted(glob.glob(SCR + '/*.npz')):
    name = os.path.basename(f)[:-4]
    d = load(name); ip = d['indptr']; nnz = np.diff(ip); tot = np.add.reduceat(d['data'].astype(np.int64), ip[:-1][nnz>0]) if True else 0
    v = d['data']
    print(f"{name:34s} m={d['m']} n={d['n']} nnz={len(v)} ones={np.mean(v==1):.3f} max={v.max()} "
          f"cols nnz<=3:{np.mean(nnz<=3):.2f} nnz>=300:{np.sum(nnz>=300)} nnzshare(nnz>=300)={nnz[nnz>=300].sum()/nnz.sum():.2f} "
          f"share(nnz<=5)={nnz[nnz<=5].sum()/nnz.sum():.3f} rows used={len(np.unique(d['indices']))}")
