"""Dump the hgmm1k_v3 barnyard inputs for the C++ harness (binary CSC): the trimmed raw matrix (barcodes >= 1 UMI,
genes x barcodes, Singlet's STAR gene counts) and the called-cell columns. Run in a Slurm job (python -I)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
from common import A, WORK, read_mtx_dir  # noqa: E402

OUT = f"{A}/fix_in"
os.makedirs(OUT, exist_ok=True)
meta = pd.read_csv(f"{WORK}/cells.csv")
X, gid, gname, rbc = read_mtx_dir(f"{WORK}/raw")  # genes x barcodes (>= 1 UMI)
X = X.tocsc()
X.sort_indices()
ix = pd.Series(np.arange(len(rbc)), index=rbc).reindex(meta.barcode.values)
assert ix.notna().all()
G, B = X.shape
X.indptr.astype(np.int64).tofile(f"{OUT}/raw.indptr")
X.indices.astype(np.int32).tofile(f"{OUT}/raw.indices")
X.data.astype(np.uint32).tofile(f"{OUT}/raw.data")
ix.astype(np.uint32).values.tofile(f"{OUT}/cells.idx")
open(f"{OUT}/dims.txt", "w").write(f"{G} {B} {X.nnz} {len(meta)}\n")
print("dumped", G, B, X.nnz, len(meta))
