"""Dump a pipeline run's exon and intron counts as the gene-level split input of harness.cpp (as export.h feeds the estimator).

    python -I dump_run.py RUN_DIR OUT_DIR

Needs exon_counts*, intron_counts*, ambient_corrected_counts_features.tsv.gz (the gene index) and cell_calls.tsv of the run.
Cells = is_cell TRUE of cell_calls.tsv, in barcode-sequence order (export.h sorts the called cells by sequence rank).
"""
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

run, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
pat = re.compile(r"^(.+?)(?:_intron)?_GRC[hm]\d+___")


def lines(p):
    with gzip.open(p, "rt") as f:
        return [l.rstrip("\n") for l in f]


gfeats = lines(f"{run}/ambient_corrected_counts_features.tsv.gz")
gindex = {g: i for i, g in enumerate(gfeats)}


def collapse(prefix):
    X = scipy.io.mmread(f"{run}/{prefix}.mtx.gz").tocsc()
    feats = lines(f"{run}/{prefix}_features.tsv.gz")
    keys = [(pat.match(f).group(1) if pat.match(f) else f) for f in feats]
    gi = np.array([gindex.get(k, -1) for k in keys])
    ok = gi >= 0
    M = sp.csr_matrix((np.ones(ok.sum()), (gi[ok], np.where(ok)[0])), shape=(len(gindex), len(feats)))
    return (M @ X).tocsc()


E, I = collapse("exon_counts"), collapse("intron_counts")
G = sp.vstack([E, I]).tocsc()
G.sort_indices()
bcs = lines(f"{run}/exon_counts_barcodes.tsv.gz")
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t")
called = cc[cc.is_cell.astype(str).str.upper() == "TRUE"].barcode.astype(str).values
pos = pd.Series(np.arange(len(bcs)), index=bcs)
ix = pos.reindex(called)
assert ix.notna().all()
order = np.argsort(np.array(called, dtype=object), kind="stable")  # barcode-sequence order
ix = ix.values[order].astype(np.uint32)
called = called[order]
print("genes x barcodes", G.shape, "nnz", G.nnz, "cells", len(ix))
G.indptr.astype(np.int64).tofile(f"{out}/raw.indptr")
G.indices.astype(np.int32).tofile(f"{out}/raw.indices")
G.data.astype(np.uint32).tofile(f"{out}/raw.data")
ix.tofile(f"{out}/cells.idx")
open(f"{out}/dims.txt", "w").write(f"{G.shape[0]} {G.shape[1]} {G.nnz} {len(ix)}\n")
open(f"{out}/barcodes.txt", "w").write("\n".join(bcs) + "\n")
open(f"{out}/cell_barcodes.txt", "w").write("\n".join(called) + "\n")
open(f"{out}/genes.txt", "w").write("\n".join(gfeats) + "\n")
