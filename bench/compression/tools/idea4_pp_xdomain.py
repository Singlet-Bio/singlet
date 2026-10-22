"""Cross-domain robustness: graft the human (PBMC/10x) topic ratios onto the mouse reference by gene symbol.

python -P tools/idea4_pp_xdomain.py <human_model_k16.bin> <mouse_model_k0.bin> <out.bin>
Mouse pi0 / q / count tables come from the mouse k0 model (in-domain, static); the topics are the human ones for genes with a
case-insensitive symbol match and ratio 1 (neutral) elsewhere.  Tests whether human immune topics carry anything to brain.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_pp_export as EX

hm = EX.read_model(sys.argv[1]); mm = EX.read_model(sys.argv[2])
href = EX.get_ref('human'); mref = EX.get_ref('mouse')
sym = lambda ids_rn, ref: [rn.split('_')[1].upper() for rn in (ref.rownames[ref.st[g]] for g in range(ref.G))]
hs = sym(None, href); ms = sym(None, mref)
hidx = {}
for g, s in enumerate(hs):
    hidx.setdefault(s, g)
K = hm['K']; G = mref.G
Rm = np.zeros((K, G), dtype=np.uint8)
match = 0
for k in range(K):
    rmax = hm['tscale'][k] * 65025.0 / 65536.0
    c1 = int(np.clip(round(np.sqrt(1.0 / rmax) * 255.0), 1, 255))
    Rm[k, :] = c1
for g, s in enumerate(ms):
    h = hidx.get(s)
    if h is not None:
        Rm[:, g] = hm['Rcode'][:, h]; match += 1
print('matched genes', match, 'of', G)
par = dict(code0=mm['code0'], qcode=mm['qcode'])
nb = EX.write_model(sys.argv[3], mref, par, K, Rm, hm['tscale'], hm['thprior'], [mm['ctab0'], mm['ctab1']])
print('wrote', sys.argv[3], nb)
