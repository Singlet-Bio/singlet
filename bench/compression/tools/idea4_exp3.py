# Empty droplets: Poisson-process cost under different ambient profile estimates.
import sys; sys.path.insert(0,'tools')
import idea4_lib as L, numpy as np, scipy.sparse as sp, time
from scipy.special import gammaln
sid=sys.argv[1]
org=L.ALL[sid][2]
trs=['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d'] if org=='human' else ['m_brain_a','m_brain_b','m_brain_c','m_brain_d']
S=L.load_sample(sid)
go,st,ids,ln=L.gene_structure(S.rownames)
cnt=np.zeros(S.m)
for t in trs:
    T=L.load_sample(t); cnt+=np.asarray(L.csr_of(T,T.called).sum(0)).ravel()
pi0=(cnt+ln/ln.sum()*100); pi0/=pi0.sum()
Xc=L.csr_of(S,S.called).tocsr(); Xe=L.csr_of(S,~S.called).tocsr()
Ac=np.asarray(Xc.sum(0)).ravel().astype(np.float64)
Ne=np.asarray(Xe.sum(1)).ravel().astype(np.float64)
keep=Ne>0
Xe=Xe[keep]; Ne=Ne[keep]
print(sid,'empties',Xe.shape[0],'nnz',Xe.nnz,'UMI',Ne.sum(),'median N',np.median(Ne))
rows=np.repeat(np.arange(Xe.shape[0]),np.diff(Xe.indptr)); x=Xe.data.astype(np.float64)
const=(Ne.sum()+gammaln(x+1).sum())*L.LOG2E
def cost(prof_nz): return (const-(x*np.log(Ne[rows]*prof_nz)).sum()*L.LOG2E)
def rep(name,prof):
    print('  %-34s %.3f bits/nnz'%(name,cost(prof[Xe.indices])/Xe.nnz),flush=True)
rep('static pi0 (train called)',pi0)
pc=Ac/Ac.sum()
for kap in [100,1000,10000]:
    p=pc*Ac.sum()/Ac.sum(); p=(Ac+kap*pi0); p/=p.sum()
    rep('called cells + %d*pi0'%kap,p)
# oracle: own empties
Ae=np.asarray(Xe.sum(0)).ravel().astype(np.float64)
for kap in [10,100]:
    p=Ae+kap*pi0; p/=p.sum(); rep('oracle own empties (+%d pi0) IN-SAMPLE'%kap,p)
# sequential adaptive
def adaptive(kc,k0,B0=256,Bmax=4096,decay=1.0):
    n=Xe.shape[0]; tot=0.0; Aemp=np.zeros(S.m); i=0; B=B0
    pc=Ac/Ac.sum()
    while i<n:
        e=min(n,i+B)
        prof=Aemp+kc*pc+k0*pi0; prof=prof/prof.sum()
        sub=Xe[i:e]
        r=np.repeat(np.arange(e-i),np.diff(sub.indptr)); xx=sub.data.astype(np.float64); NN=Ne[i:e]
        tot+=(NN.sum()+gammaln(xx+1).sum())*L.LOG2E-(xx*np.log(NN[r]*prof[sub.indices])).sum()*L.LOG2E
        Aemp+=np.asarray(sub.sum(0)).ravel()
        i=e; B=min(Bmax,B*2)
    return tot
for kc,k0 in [(1000,100),(5000,500),(20000,1000),(50000,2000)]:
    print('  adaptive kc=%d k0=%d  %.3f bits/nnz'%(kc,k0,adaptive(kc,k0)/Xe.nnz),flush=True)
