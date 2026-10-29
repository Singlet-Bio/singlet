import sys; sys.path.insert(0,'tools')
import idea4_lib as L, idea4_nmf as NM, numpy as np, scipy.sparse as sp, time
from scipy.special import gammaln
k=int(sys.argv[1]); beta=float(sys.argv[2]); iters=int(sys.argv[3])
trs=['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d']
S0=L.load_sample('pbmc_v2_typical')
go,st,ids,ln=L.gene_structure(S0.rownames); G=len(ids)
Xs=[];cnt_i=np.zeros(S0.m)
for t in trs:
    T=L.load_sample(t); X=L.csr_of(T,T.called)
    cnt_i+=np.asarray(X.sum(0)).ravel()
    Xg=L.gene_matrix(go,G,X.astype(np.float32))
    # cap cells per sample
    if Xg.shape[0]>3000:
        idx=np.random.default_rng(1).choice(Xg.shape[0],3000,replace=False); Xg=Xg[np.sort(idx)]
    Xs.append(Xg)
Xtr=sp.vstack(Xs).tocsr(); print('train cells',Xtr.shape[0],'nnz',Xtr.nnz,flush=True)
cnt_g=np.asarray(Xtr.sum(0)).ravel()
pi0=(cnt_g+0.5)/(cnt_g+0.5).sum()
t=time.time()
Phi,Th=NM.train(Xtr,k,iters=iters,beta=beta,pi0=pi0)
print('train time',time.time()-t,flush=True)
# interval split q within gene from all train counts (+length prior)
a=ln/ln.sum()*1000
ci=cnt_i+a
gs=np.add.reduceat(ci,st[:-1]); q=ci/gs[go]
np.savez('results/idea4_exp2_model_k%d.npz'%k,Phi=Phi,pi0=pi0,q=q)
for sid in ['pbmc_v2_typical','tcell_5p','pbmc_v3_lowgenes','tumor_3p']:
    S=L.load_sample(sid); X=L.csr_of(S,S.called).tocsr().astype(np.float32)
    Xg=L.gene_matrix(go,G,X)
    N=np.asarray(X.sum(1)).ravel(); nnz=X.nnz
    rows=np.repeat(np.arange(X.shape[0]),np.diff(X.indptr))
    res={}
    # static
    lam=N[rows]*(pi0[go][X.indices]*q[X.indices]) # approx static (gene pi0 x q)
    x=X.data.astype(np.float64)
    split=-(x*np.log(q[X.indices])).sum()*L.LOG2E
    base_const=(N.sum()+gammaln(x+1).sum())*L.LOG2E
    def gene_cost(Pg_nz):  # Pg_nz: pi at gene-level nnz of Xg
        xg=Xg.data.astype(np.float64); Ng=N
        rg=np.repeat(np.arange(Xg.shape[0]),np.diff(Xg.indptr))
        return -(xg*np.log(Ng[rg]*Pg_nz)).sum()*L.LOG2E
    rg=np.repeat(np.arange(Xg.shape[0]),np.diff(Xg.indptr)); cg=Xg.indices
    stat=gene_cost(pi0[cg])
    print(sid,'cells',X.shape[0],'nnz',nnz,'gene-nnz',Xg.nnz)
    print('  static bits/nnz %.3f (const %.3f split %.3f gene %.3f)'%((stat+split+base_const)/nnz,base_const/nnz,split/nnz,stat/nnz))
    for ti in [30]:
        Th=NM.em_theta(Xg,Phi,iters=ti)
        P=NM.pred_nz(Th,Phi,rg,cg)
        nm=gene_cost(P)
        print('  nmf k=%d ti=%d bits/nnz %.3f  delta vs static %.3f bits/nnz  (gene part %.3f)'%(k,ti,(nm+split+base_const)/nnz,(stat-nm)/nnz,nm/nnz),flush=True)
