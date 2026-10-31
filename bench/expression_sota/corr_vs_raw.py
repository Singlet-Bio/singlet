# SPDX-License-Identifier: MIT
"""Is ambient_corrected_counts equal to the raw gene-level counts of the called cells (filtered_feature_bc_matrix)? python corr_vs_raw.py RUN_DIR"""
import gzip, sys, numpy as np, pandas as pd, scipy.io as sio, scipy.sparse as sp, json
run = sys.argv[1]; bare = lambda b: b.split("-")[0]
def rd(prefix, genes_first=True):
    M = sio.mmread(gzip.open(f"{prefix}.mtx.gz", "rb")).tocsr(); return M
Cc = rd(f"{run}/ambient_corrected_counts")
cf = [l.rstrip("\n").split("\t")[0].split("_")[0] for l in gzip.open(f"{run}/ambient_corrected_counts_features.tsv.gz", "rt")]
cb = [bare(l.split("\t")[0].strip()) for l in gzip.open(f"{run}/ambient_corrected_counts_barcodes.tsv.gz", "rt")]
F = rd(f"{run}/filtered_feature_bc_matrix")
ff = [l.rstrip("\n").split("\t")[0] for l in gzip.open(f"{run}/filtered_feature_bc_matrix/features.tsv.gz", "rt")]
fb = [bare(l.strip()) for l in gzip.open(f"{run}/filtered_feature_bc_matrix/barcodes.tsv.gz", "rt")]
Cc = sp.csr_matrix(Cc); F = sp.csr_matrix(F)
cells = [b for b in fb if b in set(cb)]
ci = {b: i for i, b in enumerate(cb)}; fi = {b: i for i, b in enumerate(fb)}
A = Cc[:, [ci[b] for b in cells]]; B = F[:, [fi[b] for b in cells]]
# aggregate duplicate gene ids on both sides
def agg(M, ids):
    u = {g: i for i, g in enumerate(sorted(set(ids)))}; P = sp.csr_matrix((np.ones(len(ids)), ([u[g] for g in ids], range(len(ids)))), shape=(len(u), len(ids))); return P @ M, u
A2, ua = agg(A, cf); B2, ub = agg(B, ff)
genes = sorted(set(ua) & set(ub)); A3 = A2[[ua[g] for g in genes]]; B3 = B2[[ub[g] for g in genes]]
d = abs(A3 - B3)
print(json.dumps({"cells": len(cells), "n_called_in_corrected": len(cb), "genes_compared": len(genes), "total_corrected": float(A.sum()), "total_raw": float(B.sum()), "max_abs_entry_diff": float(d.max()) if d.nnz else 0.0, "entries_differing": int((d > 0).sum())}))
