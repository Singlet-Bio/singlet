import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
for sid, mem in [('brain_sn_v3_deep','exon'),('pbmc_v2_typical','exon')]:
    d = load(sid, mem); nnz=len(d['indices'])
    for ins in (False, True):
        tr = train_set(sid, mem, ins)
        n, N = pool(tr, mem)
        tot = n.sum(1)
        dens = n.sum(0)/np.maximum(N,1)
        rate = (tot+0.01)/(tot.sum()+0.01*len(tot))
        prior = np.clip(rate[:,None]*dens[None,:],1e-7,0.99)
        alpha=10
        p = np.clip((n+alpha*prior)/(N[None,:]+alpha),1e-7,0.999)
        D,k,b = col_info(d); bn = nnz_bucket(d,b)
        pp = p[d['indices'], bn]
        unseen = tot[d['indices']]==0
        absent_bits = (N_ := np.bincount(b,minlength=NB)) @ (-np.log2(1-p)).sum(0)
        print(sid, mem, 'in' if ins else 'out', tr, 'unseen nnz frac %.4f'%unseen.mean(),
              'bits/nnz: total %.3f  absent-part %.3f  present-part %.3f  unseen-present-part %.3f' % (
              (absent_bits + np.log2((1-pp)/pp).sum())/nnz, absent_bits/nnz, np.log2((1-pp)/pp).sum()/nnz, np.log2((1-pp[unseen])/pp[unseen]).sum()/nnz))
