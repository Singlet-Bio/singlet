import sys, numpy as np, csv
sys.path.insert(0,'../../python')
from singlet._pz import read_1pz
for sid in ['pbmc_v2_typical','tumor_3p','pbmc_v2_shallow','brain_sn_v3_deep']:
  for mem in ['exon','intron']:
    r = read_1pz(f'corpus/{sid}/{mem}_counts.1pz')
    m,n=r['m'],r['n']; ip=np.asarray(r['indptr']).astype(np.int64); ix=np.asarray(r['indices']).astype(np.int64); d=np.asarray(r['data']).astype(np.int64)
    calls=set(l.split('\t')[0] for l in open(f'corpus/{sid}/cell_calls.tsv').read().splitlines()[1:])
    isc=np.array([c in calls for c in r['colnames']])
    colid=np.repeat(np.arange(n),np.diff(ip))
    cm=isc[colid]
    z=np.bincount(ix[cm],minlength=m)
    order=np.argsort(-z,kind='stable')
    cs=np.cumsum(z[order])/z.sum()
    k=np.bincount(colid[cm],weights=d[cm],minlength=n)[isc]
    nn=np.diff(ip)[isc]
    # depth correlation of the flag
    kall=np.bincount(colid,weights=d,minlength=n)
    print(sid,mem,'m',m,'n',n,'cells',isc.sum(),'nnz_cells',cm.sum(),'supp',int((z>0).sum()),
      'cov T=1024/2048/4096/8192/16384: %.3f %.3f %.3f %.3f %.3f'%tuple(cs[t-1] for t in (1024,2048,4096,8192,16384)),
      'maxval',d.max(),'frac1 %.3f'%((d[cm]==1).mean()), 'min k cell',kall[isc].min(),'max k empty',kall[~isc].max() if (~isc).any() else -1)
    print('   first rownames',r['rownames'][:3],' colname sorted?',r['colnames']==sorted(r['colnames']))
