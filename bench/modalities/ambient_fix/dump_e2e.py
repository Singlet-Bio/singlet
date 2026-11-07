"""Dump a pipeline run directory (exon_counts / intron_counts mtx.gz, cell_calls.tsv) as the gene-level binary CSC the harness reads,
the way export.h builds it (exon -> gene, + intron -> gene). Usage: python -I dump_e2e.py RUN_DIR OUT_DIR ex|ei
OUT_DIR gets raw.* / cells.idx / dims.txt (genes x all barcodes, uint32) and genes.txt (gene feature names), barcodes.txt."""
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

run, out, mode = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
pat = re.compile(r"^(.+?)(?:_intron)?_GRC[hm]\d+___")


def lines(p):
    with gzip.open(p, "rt") as f:
        return [l.rstrip("\n") for l in f]


def gene_map(feats):
    keys = []
    for f in feats:
        m = pat.match(f)
        keys.append(m.group(1) if m else f)
    return keys


def collapse(prefix, genes_index):
    X = scipy.io.mmread(f"{run}/{prefix}.mtx.gz").tocsc()
    feats = lines(f"{run}/{prefix}_features.tsv.gz")
    keys = gene_map(feats)
    gi = np.array([genes_index.get(k, -1) for k in keys])
    ok = gi >= 0
    M = sp.csr_matrix((np.ones(ok.sum()), (gi[ok], np.where(ok)[0])), shape=(len(genes_index), len(feats)))
    return (M @ X).tocsc()


gfeats = lines(f"{run}/ambient_corrected_counts_features.tsv.gz")
gindex = {g: i for i, g in enumerate(gfeats)}
G = collapse("exon_counts", gindex)
if mode == "ei":
    G = G + collapse("intron_counts", gindex)
elif mode == "sp":  # exon and intron genes as separate features (rows 0..G-1 exon, G..2G-1 intron)
    G = sp.vstack([G, collapse("intron_counts", gindex)]).tocsc()
G = G.tocsc()
G.sort_indices()
bcs = lines(f"{run}/exon_counts_barcodes.tsv.gz")
cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t")
called = cc[cc.called_v2 == 1].barcode.values
ix = pd.Series(np.arange(len(bcs)), index=bcs).reindex(called)
assert ix.notna().all()
print(mode, "genes x barcodes", G.shape, "nnz", G.nnz, "cells", len(ix))
G.indptr.astype(np.int64).tofile(f"{out}/raw.indptr")
G.indices.astype(np.int32).tofile(f"{out}/raw.indices")
G.data.astype(np.uint32).tofile(f"{out}/raw.data")
ix.astype(np.uint32).values.tofile(f"{out}/cells.idx")
open(f"{out}/dims.txt", "w").write(f"{G.shape[0]} {G.shape[1]} {G.nnz} {len(ix)}\n")
open(f"{out}/genes.txt", "w").write("\n".join(gfeats) + "\n")
open(f"{out}/cell_barcodes.txt", "w").write("\n".join(called) + "\n")
