# SPDX-License-Identifier: MIT
"""Barcodes of a Cell Ranger h5 matrix (all columns, or those with >= --min-umis Gene Expression UMIs) -> text file."""
import argparse

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("h5")
ap.add_argument("out")
ap.add_argument("--min-umis", type=int, default=0)
a = ap.parse_args()
with h5py.File(a.h5, "r") as f:
    # Cell Ranger >= 3: group "matrix"; Cell Ranger 2.x: one group per genome (no feature_type: all Gene Expression)
    m = f["matrix"] if "matrix" in f else f[next(k for k in f if "barcodes" in f[k])]
    bc = [b.decode() for b in m["barcodes"][:]]
    if a.min_umis:
        ft = np.array([t.decode() for t in m["features"]["feature_type"][:]])
        ind, ptr, dat = m["indices"][:], m["indptr"][:], m["data"][:]
        gex = ft[ind] == "Gene Expression"
        col = np.repeat(np.arange(len(bc)), np.diff(ptr))
        tot = np.bincount(col[gex], weights=dat[gex], minlength=len(bc))
        bc = [b for b, t in zip(bc, tot) if t >= a.min_umis]
with open(a.out, "w") as f:
    f.write("\n".join(bc) + "\n")
print(a.out, len(bc))
