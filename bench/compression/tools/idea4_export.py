"""Export an Idea 4 model file (without vocab/tables; harness/idea4_tables.cpp adds those).

python -P tools/idea4_export.py <org> <k> [out.i4m]
Reads results/idea4_models/<org>_k<k>.npz (Phi, Theta, pi0) from tools/idea4_ce.py and the training files for the
interval split table.
"""
import os, sys, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L

org, k = sys.argv[1], int(sys.argv[2])
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(L.ROOT, 'results', 'idea4_models', f'{org}_k{k}.i4m')
NB, NDEP, CMAX = 128, 8, 24
train_ids = [t for t in L.TRAIN if L.ALL[t][2] == org]
test0 = [t for t in L.TEST if L.ALL[t][2] == org][0]
S0 = L.load_sample(test0)
go, st, ids, ln = L.gene_structure(S0.rownames); G = len(ids); m = S0.m

z = np.load(os.path.join(L.ROOT, 'results', 'idea4_models', f'{org}_k{k}.npz'))
Phi, Theta, pi0 = z['Phi'].astype(np.float64), z['Theta'].astype(np.float64), z['pi0'].astype(np.float64)
assert Phi.shape == (G, k)
pi0q = np.maximum(1, np.round(pi0 * 2.0 ** 32)).astype(np.uint32)
E = np.clip(np.round(4 * np.log2(Phi / pi0[:, None])), -64, 63).astype(np.int8)
# codebook: k-means (sqrt space) on training thetas, quantised to 6 bits
rng = np.random.default_rng(0)
Z = np.sqrt(Theta)
cent = Z[rng.choice(len(Z), 256, replace=False)].copy()
for _ in range(25):
    d = ((Z[:, None, :] - cent[None]) ** 2).sum(-1)
    lab = d.argmin(1)
    for j in range(256):
        if (lab == j).any(): cent[j] = Z[lab == j].mean(0)
Cb = cent ** 2; Cb /= Cb.sum(1, keepdims=True)
cbq = np.clip(np.round(Cb * 63), 0, 63).astype(np.uint8)
for j in range(256):
    if cbq[j].sum() == 0: cbq[j, Cb[j].argmax()] = 1
# split levels from training counts (all columns) + length prior
cnt_i = np.zeros(m)
for t in train_ids:
    T = L.load_sample(t)
    cnt_i += np.bincount(T.indices, weights=T.data, minlength=m)
ci = cnt_i + ln / ln.sum() * 1000
q = ci / np.add.reduceat(ci, st[:-1])[go]
lvl = np.clip(np.round(-4 * np.log2(q)), 0, 255).astype(np.uint8)
with open(out, 'wb') as f:
    f.write(b'I4M3')
    f.write(struct.pack('<7I', G, m, k, NB, NDEP, CMAX, 0))
    f.write(st.astype(np.uint32).tobytes())
    f.write(pi0q.tobytes())
    f.write(E.tobytes())
    f.write(cbq.tobytes())
    f.write(lvl.tobytes())
    f.write(struct.pack('<QII', 0, 0, 0))
print('wrote', out, os.path.getsize(out), 'bytes; E range', E.min(), E.max(), 'lvl max', lvl.max(), 'frac lvl==255', (lvl == 255).mean())
