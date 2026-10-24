import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
from idea1_data import genes
from idea1_fit import fit_cloglog
EB = np.array([0,1,2,3,5,9,16,30,55,100,180,320,560,1000,1800,3200,10**9])
def run(sid, mem):
    d = load(sid, mem); nnz=len(d['indices'])
    tr = train_set(sid, mem)
    n, N = pool(tr, mem, EB)
    g = genes(sid, mem); ng = g.max()+1
    sz = np.bincount(g, minlength=ng).astype(np.float64)
    tot = n.sum(1)
    G = np.bincount(g, weights=tot, minlength=ng)
    res = {}
    for eps_k in (0.0, 0.02, 0.2):
        eps = eps_k * (G/sz)[g] / (1 + (G/sz)[g]/50)      # pseudo hits from gene mean
        a, c, p = fit_cloglog(n, N, eps=eps if eps_k>0 else None)
        pc = np.clip(p, 1e-7, 0.9985)
        res[f'cloglog eps={eps_k}'] = ce_presence(d, pc, EB)/nnz
        for al in (50, 500):
            pp = np.clip((n + al*p)/(N[None,:]+al), 1e-7, 0.9985)
            res[f'cloglog eps={eps_k} +emp a={al}'] = ce_presence(d, pp, EB)/nnz
    return res
for sid, mem in [('pbmc_v2_typical','exon'),('brain_sn_v3_deep','exon'),('tumor_3p','exon'),('tcell_5p','exon')]:
    print(sid, mem, {k: round(float(v),3) for k,v in run(sid, mem).items()}, flush=True)
