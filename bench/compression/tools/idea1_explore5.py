import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
from idea1_data import genes
EB = np.array([0,1,2,3,5,9,16,30,55,100,180,320,560,1000,1800,3200,10**9])
def run(sid, mem, alphas):
    d = load(sid, mem); nnz=len(d['indices'])
    tr = train_set(sid, mem)
    n, N = pool(tr, mem, EB)
    g = genes(sid, mem); ng = g.max()+1
    sz = np.bincount(g, minlength=ng).astype(np.float64)
    tot = n.sum(1)
    dens = n.sum(0)/np.maximum(N,1)
    rate = (tot+0.01)/(tot.sum()+0.01*len(tot))
    prior_a = np.clip(rate[:,None]*dens[None,:],1e-7,0.99)
    G = np.zeros((ng, NB)); np.add.at(G, g, n)
    Rg = (G/(np.maximum(N,1)[None,:]*sz[:,None]))[g]            # mean per-interval presence rate of f's gene
    res={}
    for a in alphas:
        for nm, prior in (('rate*dens',prior_a),('gene',np.clip(Rg,1e-7,0.99))):
            p = np.clip((n + a*prior)/(N[None,:]+a),1e-7,0.999)
            res[f'{nm} a={a}'] = ce_presence(d, p, EB)/nnz
    return res
for sid, mem in [('pbmc_v2_typical','exon'),('brain_sn_v3_deep','exon'),('brain_sn_v3_deep','intron'),('tumor_3p','exon')]:
    print(sid, mem, {k: round(float(v),3) for k,v in run(sid, mem, (50,200,1000,5000)).items()}, flush=True)
