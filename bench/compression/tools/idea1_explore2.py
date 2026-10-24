import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
import idea1_lib as L
EDGES_B = np.array([0,1,2,4,7,12,20,34,58,100,170,290,500,850,1450,2500,10**9])
def run(sid, mem, edges):
    d = load(sid, mem); nnz = len(d['indices'])
    tr = train_set(sid, mem)
    n, N = pool(tr, mem, edges)
    res = {}
    # order 0
    p0 = np.clip(model_order0(n, N), 1e-7, 0.999)
    res['o0'] = ce_presence(d, p0, edges)/nnz
    tot = n.sum(1)
    for a in (0.5, 0.1, 0.02):
        # additive smoothing toward the feature's pooled rate scaled by bucket density
        # bucket-level mean density per feature: prior = global rate f * (bucket mean nnz / global mean nnz)
        dens = (n.sum(0) / np.maximum(N,1))            # expected nnz per column per bucket
        gdens = tot.sum() / N.sum()
        rate = (tot + 0.01) / (tot.sum() + 0.01*len(tot))   # feature share of nnz
        prior = np.clip(rate[:,None] * dens[None,:], 1e-7, 0.99)
        alpha = a * 20  # pseudo-columns
        p = (n + alpha * prior) / (N[None,:] + alpha)
        p = np.clip(p, 1e-7, 0.999)
        res['fxd a=%g'%alpha] = ce_presence(d, p, edges)/nnz
    return res
for sid, mem in [('pbmc_v2_typical','exon'),('tcell_5p','exon'),('brain_sn_v3_deep','exon')]:
    for name, e in (('EDGES16', EDGES16), ('EDGES_B', EDGES_B)):
        print(sid, mem, name, {k: round(v,3) for k,v in run(sid, mem, e).items()}, flush=True)
