"""Dump a Singlet run's stored exon / intron counts (all barcodes of exon_counts.mtx.gz) as gene-level binary CSC for harness.cpp.

usage: dump_run.py RUN_DIR OUT_DIR
Writes OUT_DIR/{exon,intron}.{indptr,indices,data}, dims.txt ("G B nnzE nnzI n_cells 100"), and two cell sets:
  cells.idx / cells.txt   the run's final caller cells (ambient_contamination.tsv = cc_result.cell_indices)
  knee.idx  / knee.txt    the knee-filter cells (cell_calls.tsv called == 1; STARsolo CellRanger2.2)
plus n_umi.f64 (gene-level exon + intron UMIs of each barcode).
"""
import gzip
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp


def mtx(path):
    with gzip.open(path, "rb") as f:
        return scipy.io.mmread(f).tocsc()


def lines(path):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n").split("\t")[0] for l in f]


def main(run, out):
    os.makedirs(out, exist_ok=True)
    E, I = mtx(f"{run}/exon_counts.mtx.gz"), mtx(f"{run}/intron_counts.mtx.gz")
    fe, fi = lines(f"{run}/exon_counts_features.tsv.gz"), lines(f"{run}/intron_counts_features.tsv.gz")
    be, bi = lines(f"{run}/exon_counts_barcodes.tsv.gz"), lines(f"{run}/intron_counts_barcodes.tsv.gz")
    assert be == bi
    ge, gi = [f.split("_", 1)[0] for f in fe], [f.split("_", 1)[0] for f in fi]
    genes = sorted(set(ge) | set(gi))
    ix = {g: i for i, g in enumerate(genes)}

    def agg(M, g):
        r = np.array([ix[x] for x in g])
        A = sp.csr_matrix((np.ones(len(r)), (r, np.arange(len(r)))), shape=(len(genes), len(r)))
        o = (A @ M.astype(np.float64)).tocsc()
        o.data = np.rint(o.data)
        o.eliminate_zeros()
        o.sort_indices()
        return o

    Eg, Ig = agg(E, ge), agg(I, gi)
    for tag, M in (("exon", Eg), ("intron", Ig)):
        M.indptr.astype(np.int64).tofile(f"{out}/{tag}.indptr")
        M.indices.astype(np.int32).tofile(f"{out}/{tag}.indices")
        M.data.astype(np.uint32).tofile(f"{out}/{tag}.data")
    bc = np.array(be)
    pos = {b: i for i, b in enumerate(bc)}
    caller = pd.read_csv(f"{run}/ambient_contamination.tsv", sep="\t").barcode.astype(str).values
    cc = pd.read_csv(f"{run}/cell_calls.tsv", sep="\t")
    knee = cc.loc[cc["called"].astype(str).isin(["1", "TRUE", "True", "true"]), "barcode"].astype(str).values
    for name, s in (("cells", caller), ("knee", knee)):
        cols = np.array([pos[b] for b in s if b in pos], dtype=np.uint32)
        cols.tofile(f"{out}/{name}.idx")
        pd.Series(bc[cols]).to_csv(f"{out}/{name}.txt", index=False, header=False)
        print(name, len(s), "->", len(cols), flush=True)
    tot = np.asarray(Eg.sum(0)).ravel() + np.asarray(Ig.sum(0)).ravel()
    tot.astype(np.float64).tofile(f"{out}/n_umi.f64")
    n_caller = len(np.fromfile(f"{out}/cells.idx", np.uint32))
    open(f"{out}/dims.txt", "w").write(f"{len(genes)} {len(bc)} {Eg.nnz} {Ig.nnz} {n_caller} 100\n")
    print("genes", len(genes), "barcodes", len(bc), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
