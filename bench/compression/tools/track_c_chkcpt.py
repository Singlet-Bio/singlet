import sys, os, struct, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
from track_c_train import *
org, mem, hold, test = sys.argv[1:5]
f = open(os.path.join(RES, f'track_c_model_{org}_{mem}_{hold}.bin'), 'rb')
magic, ver, m, nh, nb1, nc1, nc2, ncx, nst, ncg1, ncg2, nbg = struct.unpack('<IIIQ8I', f.read(52))
pl = np.frombuffer(f.read(m), np.uint8).astype(int); cls1 = np.frombuffer(f.read(m), np.uint8); gf = np.frombuffer(f.read(m), np.uint8)
f.seek(nb1 * nc1 * 2 + nb1 * nc2 * 2, 1)
cpt = np.frombuffer(f.read(ncx * nst * 127 * 2), np.uint16).reshape(ncx, nst, 127)
d = get(f'{test}__{mem}'); k = np.bincount(d['indices'], minlength=m); c = cabs(k)
nnz = len(d['indices']); sh = int(round(2 * math.log2(nnz)))
cx = cx_of(pl, sh)
used = (c > 0).astype(np.int64); cs = np.cumsum(used); gstart = np.flatnonzero(gf); gid = np.cumsum(gf) - 1
base = np.concatenate([[0], cs])[gstart][gid]; prev_used = (np.concatenate([[0], cs])[:-1] - base) > 0
stt = np.where(gf == 1, 0, np.where(prev_used, 2, 1))
bits = 0.0; per = np.zeros(m)
node = np.ones(m, dtype=np.int64)
for i in range(6, -1, -1):
    bit = (c >> i) & 1
    p1 = cpt[cx, stt, node - 1] / 65536.0
    per += -np.log2(np.where(bit == 1, p1, 1 - p1))
    node = node * 2 + bit
print('side bits', per.sum(), 'per nnz', per.sum() / nnz, 'sh', sh)
print('unused cost', per[c == 0].sum() / nnz, 'used cost', per[c > 0].sum() / nnz, 'used feats', (c > 0).sum())
for lo, hi in ((0, 1), (1, 41), (41, 60), (60, 80), (80, 256)):
    sel = (pl >= lo) & (pl < hi)
    print(f'plevel [{lo},{hi}): feats {sel.sum()} used {(c[sel]>0).sum()} cost/nnz {per[sel].sum()/nnz:.3f}')
