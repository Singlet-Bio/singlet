"""Correct-species UMIs that STARsolo GeneFull counts and Singlet's X does not, per gene, in the pure barnyard cells.

A wrong-species fraction can be worse because of extra wrong molecules or because of fewer correct ones. This lists the
genes where Singlet counts fewer correct-species UMIs than STARsolo (and the reverse), with the gene's chromosome.

  python deficit.py --singlet RUN --starsolo SOLO --cr CR.h5 --gtf genes.gtf --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "exact_vs_pseudo"))
import loaders as L  # noqa: E402
from parity_score import align, pure_cells  # noqa: E402


def correct_by_gene(m, bcs, ph, pm):
    X = align(m, bcs)
    ish = np.array([g.startswith("ENSG") for g in m.genes])
    pg = np.zeros(len(m.genes))
    pg[ish] += X[ph][:, ish].sum(0).A1
    pg[~ish] += X[pm][:, ~ish].sum(0).A1
    return dict(zip(m.genes, pg))


def main():
    ap = argparse.ArgumentParser()
    for k in ("singlet", "starsolo", "cr", "gtf", "out"):
        ap.add_argument("--" + k, required=True)
    ap.add_argument("--feature", default="GeneFull")
    a = ap.parse_args()
    cr = L.load_cr_h5(a.cr); ph, pm = pure_cells(cr); chrom = L.gene_chrom(a.gtf)
    ga = correct_by_gene(L.load_singlet(a.singlet), cr.barcodes, ph, pm)
    gb = correct_by_gene(L.load_starsolo(a.starsolo, a.feature), cr.barcodes, ph, pm)
    genes = set(ga) | set(gb)
    d = {g: gb.get(g, 0.0) - ga.get(g, 0.0) for g in genes}
    nm = lambda g: chrom.get(g, ("", g, ""))  # noqa: E731
    row = lambda g: (nm(g)[1] or g, g, nm(g)[0], float(d[g]), float(ga.get(g, 0)), float(gb.get(g, 0)))  # noqa: E731
    by_chr = {}
    for g, v in d.items():
        c = nm(g)[0] or "?"
        by_chr[c] = by_chr.get(c, 0.0) + v
    out = {"singlet_total": float(sum(ga.values())), "starsolo_total": float(sum(gb.values())),
           "deficit_positive": float(sum(v for v in d.values() if v > 0)), "deficit_negative": float(sum(v for v in d.values() if v < 0)),
           "n_genes_deficit_gt10": int(sum(v > 10 for v in d.values())),
           "by_chrom": dict(sorted(by_chr.items(), key=lambda kv: -abs(kv[1]))[:20]),
           "top_deficit": [row(g) for g in sorted(genes, key=lambda g: -d[g])[:60]],
           "top_surplus": [row(g) for g in sorted(genes, key=lambda g: d[g])[:30]]}
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("top")}, indent=1))
    for r in out["top_deficit"][:40]:
        print("deficit", r)
    for r in out["top_surplus"][:15]:
        print("surplus", r)


if __name__ == "__main__":
    main()
