import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_c_model import *
import track_c_exp2 as E
t='pbmc_v3_lowgenes'; mem='exon'
org=ORG[t]; g,ng=gene_index(org,mem)
pool=[k for k in STUDY if ORG[k]==org and STUDY[k]!=STUDY[t]]
names=[f'{k}__{mem}' for k in pool]; print(names)
m=get(names[0])['m']
s=profile(names,m); c,nc=make_classes(s,g,ng)
d=get(f'{t}__{mem}')
kf=np.bincount(d['indices'],minlength=m)
print('nc',nc,'classes of top feats', c[np.argsort(-kf)[:10]], 'train share', s[np.argsort(-kf)[:10]])
# per-class mass
mc=np.bincount(c,minlength=nc); Ac=np.bincount(c[d['indices']],minlength=nc)
for k in range(0,nc,6): print(k, mc[k], Ac[k])
# replicate eval with details
scheme=('nz',16,0,14); nb=16
def bsch(d): return bucketize(d,scheme)
pool2=pool
studies=sorted(set(STUDY[k] for k in pool2))
Acal=np.zeros((nb,nc)); Mcal=np.zeros((nb,nc))
for st in studies:
    inn=[f'{k}__{mem}' for k in pool2 if STUDY[k]!=st]; out=[f'{k}__{mem}' for k in pool2 if STUDY[k]==st]
    s_in=profile(inn,m); c_in,_=make_classes(s_in,g,ng); mc_in=np.bincount(c_in,minlength=nc).astype(float)
    for nm in out:
        dd=get(nm); A,Nb=counts_AN(dd,bsch(dd),c_in,nb,nc); Acal+=A; Mcal+=Nb[:,None]*mc_in[None,:]
P=np.clip((Acal+0.0005)/(Mcal+0.001),1e-7,0.999)
b=bsch(d); A,Nb=counts_AN(d,b,c,nb,nc); mcc=np.bincount(c,minlength=nc).astype(float)
absent=Nb[:,None]*mcc[None,:]-A
cost_p=-(np.where(A>0,A*np.log2(P),0)); cost_a=-(np.where(absent>0,absent*np.log2(1-P),0))
print('present cost bits/nnz',cost_p.sum()/len(d['indices']),'absent',cost_a.sum()/len(d['indices']))
print('per bucket present', (cost_p.sum(1)/len(d['indices'])).round(2))
print('per bucket absent', (cost_a.sum(1)/len(d['indices'])).round(2))
print('Nb',Nb)
print('A.sum(1)',A.sum(1))
i=np.argmax(cost_a.sum(1)); print('worst bucket',i, 'worst classes', np.argsort(-cost_a[i])[:5], cost_a[i][np.argsort(-cost_a[i])[:5]], P[i][np.argsort(-cost_a[i])[:5]])
