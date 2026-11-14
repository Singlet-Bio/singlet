import gzip, importlib.util, sys, numpy as np, pandas as pd, scipy.io
from pathlib import Path
W=Path("/mnt/projects/debruinz_project/debruinz/singlet"); O=W/"bench/calls-sota/sex"
sp=importlib.util.spec_from_file_location("sexnew",O/"sexmod_pool.py"); m=importlib.util.module_from_spec(sp); sys.modules["sexnew"]=m; sp.loader.exec_module(m)
d=O/"work/mtx/schihost"; X=scipy.io.mmread(gzip.open(d/"matrix.mtx.gz")).T.tocsr(); genes=(d/"genes.tsv").read_text().split("\n")[:X.shape[1]]
ev=pd.read_parquet(O/"evidence/schihost.parquet"); r=m.call_sex(X,genes,species="human",groups=ev.demux.tolist())
from collections import Counter
print(Counter(v["call"] for v in r.groups.values()))
print(r.groups["HG00154"]); print(r.groups["HG00479"])
