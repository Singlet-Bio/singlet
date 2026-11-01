# SPDX-License-Identifier: MIT
"""Re-anchor the V segments of a built reference (first of adjacent Cys codons):  python -P reanchor.py IN.tsv.gz OUT.tsv.gz"""
import gzip
import sys

from singlet.vdj import reference as R

src, dst = sys.argv[1:3]
segs = R.load(src)
prov = ""
with gzip.open(src, "rt") as f:
    first = f.readline()
    if first.startswith("#"):
        prov = first[1:].strip()
before = {s.name: s.anchor for s in segs}
n = R.reanchor_v(segs)
R.write(segs, dst, (prov + "; " if prov else "") + "V anchors moved to the first of adjacent Cys codons (bench/expression_sota/vdj/reanchor.py)")
print("moved", n, [s.name for s in segs if s.anchor != before[s.name]])
