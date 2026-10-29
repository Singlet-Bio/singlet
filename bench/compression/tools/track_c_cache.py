#!/usr/bin/env python
"""Cache .1pz matrices as npz for training: python -P tools/track_c_cache.py CACHE_DIR file.1pz ...
Output  CACHE_DIR/<parent>__<member>.npz  (indptr,indices,data,m,n) and CACHE_DIR/<parent>__<member>.rows.txt"""
import os, sys, numpy as np
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
cache = sys.argv[1]; os.makedirs(cache, exist_ok=True)
for p in sys.argv[2:]:
    parent = os.path.basename(os.path.dirname(os.path.abspath(p))); mem = os.path.basename(p).replace('_counts.1pz', '')
    out = os.path.join(cache, f"{parent}__{mem}.npz")
    if os.path.exists(out): continue
    r = read_1pz(p)
    np.savez(out, indptr=np.asarray(r['indptr'], dtype=np.int64), indices=np.asarray(r['indices'], dtype=np.int32),
             data=np.asarray(r['data'], dtype=np.uint32), m=len(r['rownames']), n=len(r['colnames']))
    with open(out.replace('.npz', '.rows.txt'), 'w') as f: f.write('\n'.join(r['rownames']))
    with open(out.replace('.npz', '.cols.txt'), 'w') as f: f.write('\n'.join(r['colnames']))
    print(out, len(r['rownames']), len(r['colnames']), len(r['indices']), flush=True)
