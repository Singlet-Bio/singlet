"""Classify Singlet's wrong-species and chrY molecules by read class, from a SINGLET_MOLECULE_DUMP table.

Truth is physical (bench/exact_vs_pseudo/score.py): in a pure human cell every mouse molecule is wrong and vice versa,
and HEK293T (female) carries no Y chromosome, so every non-PAR chrY molecule in a pure human cell is wrong.
Cells are the published Cell Ranger filtered barcodes; purity uses Cell Ranger's own species fraction (>= 500 UMI, 90%).

Classes come from the molecule's diag bits (OR over its reads, DirectionalUmiStore::kDiag*) and its label:
  multi     some read had NH > 1                         other   a read also lay in another same-strand gene (exon priority)
  anti      a read overlapped an opposite-strand gene    outside some aligned bases outside the gene body
  label     S spliced/exonic, U unspliced, A ambiguous, D dropped by the multi-gene UMI rule

  python diagnose.py --dump molecules.tsv.gz --cr CR.h5 --gtf genes.gtf --out OUT.json [--starsolo DIR]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "exact_vs_pseudo"))
import loaders as L  # noqa: E402

BITS = [(1, "multi"), (2, "spliced"), (4, "flank"), (8, "other"), (16, "anti"), (32, "outside")]


def cell_species(cr_h5):
    cr = L.load_cr_h5(cr_h5)
    human = np.array([g.startswith("ENSG") for g in cr.genes])
    tot = np.asarray(cr.X.sum(1)).ravel()
    fh = np.asarray(cr.X[:, human].sum(1)).ravel() / np.maximum(tot, 1)
    ok = tot >= 500
    sp = {}
    for b, t, f in zip(cr.barcodes, ok, fh):
        if t and f >= 0.9:
            sp[b] = "human"
        elif t and f <= 0.1:
            sp[b] = "mouse"
    return sp


def cls(diag, label):
    names = [n for bit, n in BITS if diag & bit and n in ("multi", "other", "anti", "outside")]
    return label + ":" + ("+".join(names) if names else "plain")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--cr", required=True)
    ap.add_argument("--gtf", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp = cell_species(a.cr)
    chrom = L.gene_chrom(a.gtf)
    ycols = {g for g, (c, n, t) in chrom.items() if c.endswith("chrY") and g.startswith("ENSG") and "_PAR_Y" not in g}

    tot = collections.Counter()           # (cell species, label) -> molecules kept (label != D)
    wrong = collections.Counter()         # class -> wrong-species molecules
    wrong_gene = collections.Counter()    # gene -> wrong-species molecules (kept)
    wrong_gene_cls = collections.defaultdict(collections.Counter)
    y_cls = collections.Counter(); y_gene = collections.Counter(); y_gene_cls = collections.defaultdict(collections.Counter)
    human_kept = 0
    with gzip.open(a.dump, "rt") as f:
        next(f)
        for ln in f:
            bc, gene, umi, lab, flags, diag, reads = ln.rstrip("\n").split("\t")
            s = sp.get(bc.split("-")[0])
            if s is None:
                continue
            g = gene.split(".")[0]
            gh = g.startswith("ENSG")
            d = int(diag)
            c = cls(d, lab)
            if lab != "D":
                tot[(s, lab)] += 1
            if (s == "human") != gh:   # wrong species
                wrong[c] += 1
                if lab != "D":
                    wrong_gene[g] += 1; wrong_gene_cls[g][c] += 1
            if s == "human" and gh:
                if lab != "D":
                    human_kept += 1
                if g in ycols:
                    y_cls[c] += 1
                    if lab != "D":
                        y_gene[g] += 1; y_gene_cls[g][c] += 1

    kept_total = sum(tot.values())
    kept_wrong = sum(v for k, v in wrong.items() if not k.startswith("D:"))
    name = lambda g: chrom.get(g, ("", g, ""))[1] or g  # noqa: E731
    res = {
        "n_pure_cells": {"human": sum(1 for v in sp.values() if v == "human"), "mouse": sum(1 for v in sp.values() if v == "mouse")},
        "molecules_kept": kept_total, "wrong_kept": kept_wrong, "wrong_fraction": kept_wrong / max(kept_total, 1),
        "molecules_by_label": {f"{k[0]}:{k[1]}": v for k, v in sorted(tot.items())},
        "wrong_by_class": dict(wrong.most_common()),
        "wrong_top_genes": [(name(g), g, v, dict(wrong_gene_cls[g].most_common(4))) for g, v in wrong_gene.most_common(40)],
        "chrY": {"human_kept": human_kept, "chrY_kept": sum(y_gene.values()),
                 "per_million": 1e6 * sum(y_gene.values()) / max(human_kept, 1),
                 "by_class": dict(y_cls.most_common()),
                 "top_genes": [(name(g), g, v, dict(y_gene_cls[g].most_common(4))) for g, v in y_gene.most_common(25)]},
    }
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("molecules_kept", "wrong_kept", "wrong_fraction", "wrong_by_class")}, indent=1))
    print("chrY", json.dumps(res["chrY"]["by_class"]), res["chrY"]["per_million"])


if __name__ == "__main__":
    main()
