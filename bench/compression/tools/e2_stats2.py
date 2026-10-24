import sys; sys.path.insert(0, 'tools')
from e2_common import *
for s in SAMPLES:
    d = load(s, 'exon')
    c = d['is_cell']; N = d['N']; n = d['n']
    nc = c.sum()
    prefix = c[:nc].all()
    # depth monotone?
    mono = np.all(np.diff(N) <= 0)
    inv = np.sum(np.diff(N) > 0)
    print(s, 'cells are prefix:', prefix, ' depth non-increasing:', mono, 'n inversions', inv, ' N[0:5]=', N[:5], 'N[nc-3:nc+3]=', N[nc-3:nc+3], 'N[-3:]=', N[-3:])
