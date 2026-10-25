"""Write the inputs of the external tools for LIB VIEW under $LQC/ambient_truth/work/LIB_VIEW: C++ production harness
(raw trimmed matrix + cell columns) and DecontX (cells and a 20,000-droplet 10-100 UMI background, binary CSC)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atlib  # noqa: E402


def wr(M, d, name):
    M = M.tocsc(); M.sort_indices()
    M.indptr.astype(np.int32).tofile(f"{d}/{name}.p")
    M.indices.astype(np.int32).tofile(f"{d}/{name}.i")
    M.data.astype(np.int32).tofile(f"{d}/{name}.x")
    open(f"{d}/{name}.dim", "w").write(f"{M.shape[0]} {M.shape[1]}\n")


lib, view = sys.argv[1], sys.argv[2]
ds = atlib.load(lib, view)
d = f"{atlib.LQC}/ambient_truth/work/{lib}_{view}"
os.makedirs(d, exist_ok=True)
atlib.dump_harness(ds, d + "/in")
wr(ds.C, d, "cells")
cand = np.where((ds.tot >= 10) & (ds.tot <= 100))[0]
cand = cand[np.argsort(-ds.tot[cand], kind="stable")[:20000]]
wr(ds.X[:, cand], d, "bg")
open(d + "/gene_mouse.txt", "w").write(",".join(map(str, ds.gene_mouse.astype(int))) + "\n")
print("dumped", d, ds.C.shape, len(cand))
