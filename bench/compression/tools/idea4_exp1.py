import sys; sys.path.insert(0,'tools')
import idea4_lib as L, numpy as np, scipy.sparse as sp, time
from scipy.special import gammaln
org='human'
trs=['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d']
S0=L.load_sample('pbmc_v2_typical')
go,st,ids,ln=L.gene_structure(S0.rownames)
cnt=np.zeros(S0.m)
for t in trs:
    T=L.load_sample(t)
    X=L.csr_of(T,T.called)
    cnt+=np.asarray(X.sum(0)).ravel()
print('train UMIs',cnt.sum())
def prof(alpha):
    a=alpha*ln/ln.sum()
    return (cnt+a)/(cnt.sum()+alpha)
for sid in ['pbmc_v2_typical','tcell_5p','tumor_3p','pbmc_v3_lowgenes']:
    S=L.load_sample(sid)
    X=L.csr_of(S,S.called).tocsr()
    N=np.asarray(X.sum(1)).ravel(); nnz=X.nnz
    rows=np.repeat(np.arange(X.shape[0]),np.diff(X.indptr))
    for alpha in [10,100,1000]:
        p=prof(alpha)
        lam=N[rows]*p[X.indices]
        x=X.data.astype(np.float64)
        bits=(N.sum() - (x*np.log(lam)).sum() + gammaln(x+1).sum())*L.LOG2E
        print(sid,'alpha',alpha,'bits/nnz %.3f'%(bits/nnz),'cells',X.shape[0],'nnz/cell %.0f'%(nnz/X.shape[0]),flush=True)
