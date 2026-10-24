import sys; sys.path.insert(0, 'tools')
from e2_common import *
for s in SAMPLES:
    for mem in ('exon', 'intron'):
        d = load(s, mem)
        c = d['is_cell']
        k = d['k']; N = d['N']
        print(f"{s:22s} {mem:6s} m={d['m']} n={d['n']} nnz={k.sum()} cells={c.sum()} nnz_cells={k[c].sum()} nnz_empt={k[~c].sum()} "
              f"cells N med={np.median(N[c]):.0f} k med={np.median(k[c]):.0f}  empt N med={np.median(N[~c]):.1f} k med={np.median(k[~c]):.1f} "
              f"maxcount={d['data'].max()} frac1={np.mean(d['data']==1):.3f} cellpos first/last: {np.flatnonzero(c)[:3]} ")
