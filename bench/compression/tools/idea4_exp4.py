# Static model calibration study at interval level: Poisson vs calibrated presence hazard and empirical count tables.
import sys; sys.path.insert(0,'tools')
import idea4_lib as L, numpy as np, scipy.sparse as sp, time
from scipy.special import gammaln
test=sys.argv[1]
trs=['h_pbmc_a','h_pbmc_b','h_b','h_c','h_d']
S0=L.load_sample(test)
go,st,ids,ln=L.gene_structure(S0.rownames); G=len(ids); m=S0.m
cnt=np.zeros(m)
trX=[]
for t in trs:
    T=L.load_sample(t); X=L.csr_of(T,T.called).tocsr()
    cnt+=np.asarray(X.sum(0)).ravel(); trX.append(X)
pi=(cnt+ln/ln.sum()*1000); pi/=pi.sum()
NB=64
def bucket(lam):  # quarter-octave buckets from 2^-24
    b=np.floor(4*(np.log2(np.maximum(lam,1e-30))+24)).astype(np.int64)
    return np.clip(b,0,NB-1)
def collect(X,pi,ncell=400,seed=0):
    rng=np.random.default_rng(seed)
    sel=np.sort(rng.choice(X.shape[0],min(ncell,X.shape[0]),replace=False))
    n_all=np.zeros(NB); n_pres=np.zeros(NB); lam_sum=np.zeros(NB)
    ctab=np.zeros((NB,20))
    Xs=X[sel]; N=np.asarray(Xs.sum(1)).ravel()
    for r in range(Xs.shape[0]):
        lam=N[r]*pi
        b=bucket(lam)
        n_all+=np.bincount(b,minlength=NB); lam_sum+=np.bincount(b,weights=lam,minlength=NB)
        row=Xs[r]; ix=row.indices; c=row.data
        bp=b[ix]; n_pres+=np.bincount(bp,minlength=NB)
        np.add.at(ctab,(bp,np.minimum(c,20)-1),1)
    return n_all,n_pres,lam_sum,ctab
# train calibration on training cells (each sample pooled)
tot=[np.zeros(NB),np.zeros(NB),np.zeros(NB),np.zeros((NB,20))]
for X in trX:
    r=collect(X,pi,300)
    for a,b in zip(tot,r): a+=b
n_all,n_pres,lam_sum,ctab=tot
p_b=(n_pres+0.5)/(n_all+1)
# test
n_all2,n_pres2,lam2,ctab2=collect(L.csr_of(S0,S0.called).tocsr(),pi,600,seed=5)
def H(p,n_all,n_pres):
    return -(n_pres*np.log2(p)+(n_all-n_pres)*np.log2(1-p)).sum()
# Poisson presence
lam_mean=lam2/np.maximum(n_all2,1)
p_pois=1-np.exp(-lam_mean)  # per-bucket mean-lambda approx
p_pois=np.clip(p_pois,1e-12,1-1e-12)
pres_pois=0; # exact per-interval poisson presence cost needs lam; approximate with bucket means
pres_cal=H(np.clip(p_b,1e-12,1-1e-12),n_all2,n_pres2)
pres_pb=H(np.clip((n_pres2+0.5)/(n_all2+1),1e-12,1-1e-12),n_all2,n_pres2)
nnz2=n_pres2.sum()
print('test nnz sampled',nnz2)
print('presence bits/nnz: poisson(bucket-mean) %.3f  calibrated(train) %.3f  oracle(test) %.3f'%(H(p_pois,n_all2,n_pres2)/nnz2,pres_cal/nnz2,pres_pb/nnz2))
# counts given presence: ZTP with bucket mean lambda vs empirical table (train) vs oracle
def ztp_pmf(lam,c):
    return np.exp(c*np.log(lam)-lam-gammaln(c+1))/(-np.expm1(-lam))
cs=np.arange(1,21)
cost_z=0;cost_t=0;cost_o=0
tab=(ctab+0.5)/ (ctab+0.5).sum(1,keepdims=True)
orc=(ctab2+0.5)/(ctab2+0.5).sum(1,keepdims=True)
for b in range(NB):
    if ctab2[b].sum()==0: continue
    lam=max(lam_mean[b],1e-9)
    pm=ztp_pmf(lam,cs); pm=pm/pm.sum()
    cost_z+=-(ctab2[b]*np.log2(pm)).sum(); cost_t+=-(ctab2[b]*np.log2(tab[b])).sum(); cost_o+=-(ctab2[b]*np.log2(orc[b])).sum()
print('count|presence bits/nnz: ZTP(bucket-mean) %.3f  table(train) %.3f  oracle %.3f'%(cost_z/nnz2,cost_t/nnz2,cost_o/nnz2))
