"""Diff the exon/intron count matrices of shuffled-order replays (shuffle_diag.sbatch).

Usage: python -I exon_order_diff.py <keep dir> <variant> [<variant> ...]
Each <keep dir>/<variant>/ holds {exon,intron}_counts.mtx.gz with _features/_barcodes.
For every pair against the first variant: entries that differ, net count change, and how
the differing entries split across genes (one exon of a gene up, another exon of the same
gene down = the molecule moved between exons; a gene total change = molecules gained/lost).
"""
import gzip, json, os, sys
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp


def load(d, kind):
    m = scipy.io.mmread(f"{d}/{kind}_counts.mtx.gz").tocsr().astype(np.int64)
    f = pd.read_csv(f"{d}/{kind}_counts_features.tsv.gz", sep="\t", header=None)[0].astype(str).values
    b = pd.read_csv(f"{d}/{kind}_counts_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).values
    return m, f, b


def compare(da, db, kind):
    ma, fa, ba = load(da, kind)
    mb, fb, bb = load(db, kind)
    if not (np.array_equal(fa, fb) and np.array_equal(ba, bb)):
        return {"axes_equal": False, "shape_a": ma.shape, "shape_b": mb.shape}
    d = (mb - ma).tocoo()
    nz = d.data != 0
    rows, cols, vals = d.row[nz], d.col[nz], d.data[nz]
    out = {"axes_equal": True, "entries_differ": int(nz.sum()), "net": int(vals.sum()),
           "abs": int(np.abs(vals).sum()), "cells_touched": int(len(np.unique(cols)))}
    if len(vals):
        gene = np.array([s.split("_")[0] for s in fa])
        g = gene[rows]
        df = pd.DataFrame({"gene": g, "cell": cols, "feat": fa[rows], "d": vals})
        per_gc = df.groupby(["gene", "cell"])["d"].sum()
        out["gene_cell_pairs"] = int(len(per_gc))
        out["gene_cell_pairs_net_zero"] = int((per_gc == 0).sum())  # moved between exons of one gene
        out["genes_touched"] = int(df.gene.nunique())
        out["top_genes"] = df.assign(a=df.d.abs()).groupby("gene")["a"].sum().nlargest(15).to_dict()
        out["examples"] = df.head(12).to_dict("records")
    return out


def main():
    keep, variants = sys.argv[1], sys.argv[2:]
    res = {}
    base = variants[0]
    for v in variants[1:]:
        for kind in ("exon", "intron"):
            res[f"{base}_vs_{v}/{kind}"] = compare(f"{keep}/{base}", f"{keep}/{v}", kind)
    if len(variants) > 2:
        for kind in ("exon",):
            res[f"{variants[1]}_vs_{variants[2]}/{kind}"] = compare(f"{keep}/{variants[1]}", f"{keep}/{variants[2]}", kind)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
