import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from idea1_lib import *
# depth stats: share of nnz by bucket across test+train exon
for mem in ('exon','intron'):
  print(mem)
  for sid in ALL:
    d = load(sid, mem); D,k,b = col_info(d)
    nn = np.bincount(b, weights=k, minlength=NB)/k.sum()
    cs = np.bincount(b, minlength=NB)/len(b)
    print('%-22s'%sid, 'cols%:', ' '.join('%3d'%round(100*x) for x in cs), '| nnz%:', ' '.join('%3d'%round(100*x) for x in nn))
