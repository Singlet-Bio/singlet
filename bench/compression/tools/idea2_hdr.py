import sys, numpy as np
sys.path.insert(0,'../../python')
from singlet._pz import read_1pz
def H(x):
    _,c=np.unique(x,return_counts=True); p=c/c.sum(); return -(c*np.log2(p)).sum()
def Hcond(x,y):
    # H(x|y) plug-in
    key=y.astype(np.int64)
    tot=0
    for v in np.unique(key):
        tot+=H(x[key==v])
    return tot
for sid in ['pbmc_v2_typical','pbmc_v2_shallow','brain_sn_v3_deep','tumor_3p','tcell_5p']:
  for mem in ['exon','intron']:
    r = read_1pz(f'corpus/{sid}/{mem}_counts.1pz')
    m,n=r['m'],r['n']; ip=np.asarray(r['indptr']).astype(np.int64); d=np.asarray(r['data']).astype(np.int64)
    calls=set(l.split('\t')[0] for l in open(f'corpus/{sid}/cell_calls.tsv').read().splitlines()[1:])
    isc=np.array([c in calls for c in r['colnames']])
    colid=np.repeat(np.arange(n),np.diff(ip))
    k=np.bincount(colid,weights=d,minlength=n).astype(np.int64); nn=np.diff(ip)
    lb=np.floor(np.log2(k+1)*4).astype(np.int64)   # quarter-octave bucket
    lbn=np.floor(np.log2(nn+1)*4).astype(np.int64)
    print(sid,mem,'n',n,'cells',isc.sum())
    print('  H(flag)=%.0f  H(flag|kbucket)=%.0f  H(flag|nnzbucket)=%.0f bits | H(k_all)=%.0f H(nnz_all)=%.0f  cells: H(k)=%.0f H(nnz)=%.0f H(nnz|kb)=%.0f  empties: H(k)=%.0f'%(
      H(isc), Hcond(isc,lb), Hcond(isc,lbn), H(k), H(nn), H(k[isc]), H(nn[isc]), Hcond(nn[isc],lb[isc]), H(k[~isc])))
