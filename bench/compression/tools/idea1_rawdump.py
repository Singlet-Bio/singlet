"""Write raw CSC dumps (for harness/idea1_micro.cpp) from the cached npz: python -P tools/idea1_rawdump.py OUTDIR sample:member ..."""
import sys, os, struct
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from idea1_data import load
out = sys.argv[1]
for spec in sys.argv[2:]:
    sid, mem = spec.split(':')
    d = load(sid, mem)
    with open(os.path.join(out, f'{sid}_{mem}.raw'), 'wb') as f:
        f.write(struct.pack('<IIQB7x', d['m'], d['n'], len(d['indices']), 5))
        f.write(d['indptr'].astype('<u4').tobytes()); f.write(d['indices'].astype('<u4').tobytes()); f.write(d['data'].astype('<u4').tobytes())
    print(sid, mem, d['m'], d['n'], len(d['indices']))
