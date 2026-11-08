"""Where do the cross-species UMIs come from? Like-for-like feature layers on the barnyard (hgmm1k_v3).

Layers (all cells x genes, gene ids):
  Singlet   exon   = exon_counts.mtx.gz rows collapsed by gene       (STARsolo `Gene` analogue)
            intron = intron_counts.mtx.gz rows collapsed by gene
            ambiguous = ambiguous_counts.mtx.gz (molecules with both exonic and intronic reads; part of exon in the fixed engine)
            X      = raw_feature_bc_matrix (what score.py uses)   (STARsolo `GeneFull` analogue)
  STARsolo  Gene, GeneFull, Velocyto spliced / unspliced / ambiguous
  alevin-fry S, U, A, S+A (exonic analogue), S+U+A

For each layer: UMIs of the wrong species in pure cells / all UMIs, and the share of the wrong-species UMIs carried by the
top loci. Pure cells and species labels come from the published Cell Ranger matrix (as in score.py).

  python breakdown.py --runs RUNS --layout evp|legacy --data DATA --gtf GENES_ONLY.gtf --out OUT.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import scipy.io
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaders as L  # noqa: E402


def collapse_by_gene(mtx, feats, barcodes_file):
    """Sum feature rows (exon / intron features 'ENSG..._NAME_...') into genes. Returns Mat (cells x genes)."""
    M = scipy.io.mmread(mtx).tocsr()
    with gzip.open(feats, "rt") as f:
        ids = [ln.split("\t")[0].split("_")[0] for ln in f]
    uniq, inv = np.unique(ids, return_inverse=True)
    G = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uniq), len(inv)))
    X = (G @ M).T.tocsr().astype(np.float64)
    with gzip.open(barcodes_file, "rt") as f:
        bcs = np.array([ln.strip().split("-")[0] for ln in f])
    return L.Mat(bcs, uniq, X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--data", required=True); ap.add_argument("--gtf", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--layout", default="evp"); ap.add_argument("--ds", default="hgmm1k_v3")
    a = ap.parse_args()
    b = os.path.join(a.runs, a.ds)
    sd, od, ad = (os.path.join(b, "singlet"), os.path.join(b, "starsolo"), os.path.join(b, "alevinfry")) if a.layout == "evp" else \
                 (os.path.join(b, "singlet-campaign-t8"), os.path.join(b, "starsolo"), os.path.join(b, "alevinfry"))
    h5 = [os.path.join(a.data, a.ds, f) for f in os.listdir(os.path.join(a.data, a.ds)) if f.endswith(".h5") and "filtered" in f][0]
    cr = L.load_cr_h5(h5)
    chrom = L.gene_chrom(a.gtf)
    isH_cr = np.array([g.startswith("ENSG") for g in cr.genes])
    tot = np.asarray(cr.X.sum(1)).ravel()
    fh = np.asarray(cr.X[:, isH_cr].sum(1)).ravel() / np.maximum(tot, 1)
    pure_h, pure_m = (tot >= 500) & (fh >= 0.9), (tot >= 500) & (fh <= 0.1)
    bc = cr.barcodes

    layers = {
        "singlet.exon": lambda: collapse_by_gene(os.path.join(sd, "exon_counts.mtx.gz"), os.path.join(sd, "exon_counts_features.tsv.gz"), os.path.join(sd, "exon_counts_barcodes.tsv.gz")),
        "singlet.intron": lambda: collapse_by_gene(os.path.join(sd, "intron_counts.mtx.gz"), os.path.join(sd, "intron_counts_features.tsv.gz"), os.path.join(sd, "intron_counts_barcodes.tsv.gz")),
        "singlet.X(exon+intron)": lambda: L.load_singlet(sd),
        "starsolo.Gene": lambda: L.load_starsolo(od, "Gene"),
        "starsolo.GeneFull": lambda: L.load_starsolo(od, "GeneFull"),
        "starsolo.velocyto.spliced": lambda: L.load_starsolo_velocyto(od, "spliced"),
        "starsolo.velocyto.unspliced": lambda: L.load_starsolo_velocyto(od, "unspliced"),
        "starsolo.velocyto.ambiguous": lambda: L.load_starsolo_velocyto(od, "ambiguous"),
        "alevinfry.S": lambda: L.load_alevinfry(ad, "af_quant", "S"),
        "alevinfry.U": lambda: L.load_alevinfry(ad, "af_quant", "U"),
        "alevinfry.A": lambda: L.load_alevinfry(ad, "af_quant", "A"),
        "alevinfry.S+A": lambda: L.load_alevinfry(ad, "af_quant", "sa"),
        "alevinfry.S+U+A": lambda: L.load_alevinfry(ad, "af_quant", "usa"),
    }
    if os.path.exists(os.path.join(sd, "ambiguous_counts.mtx.gz")):
        layers["singlet.ambiguous"] = lambda: collapse_by_gene(os.path.join(sd, "ambiguous_counts.mtx.gz"), os.path.join(sd, "ambiguous_counts_features.tsv.gz"), os.path.join(sd, "ambiguous_counts_barcodes.tsv.gz"))
    watch = {n: g for g, (c, n, t) in chrom.items() if n in ("LINC00486", "Rn18s-rs5", "mt-Rnr2", "MT-RNR2", "MT-RNR1", "mt-Rnr1")}
    out = {}
    pgw = {}
    for name, load in layers.items():
        try:
            m = load()
        except Exception as e:  # layer absent in this run
            out[name] = {"error": repr(e)[:200]}; continue
        idx = {x: i for i, x in enumerate(m.barcodes)}
        rows = np.array([idx.get(x, -1) for x in bc])
        X = m.X[np.where(rows < 0, 0, rows)]
        X = sp.diags((rows >= 0).astype(float)) @ X
        X = sp.csr_matrix(X)
        ish = np.array([g.startswith("ENSG") for g in m.genes])
        h = np.asarray(X[:, ish].sum(1)).ravel(); mo = np.asarray(X[:, ~ish].sum(1)).ravel()
        sel = pure_h | pure_m
        wrong = np.where(pure_h, mo, np.where(pure_m, h, 0.0))
        t = h + mo
        per_gene_wrong = np.zeros(len(m.genes))
        if pure_m.any():
            per_gene_wrong[ish] += X[pure_m][:, ish].sum(0).A1
        if pure_h.any():
            per_gene_wrong[~ish] += X[pure_h][:, ~ish].sum(0).A1
        pgw[name] = dict(zip(m.genes, per_gene_wrong))
        o = np.argsort(-per_gene_wrong)[:10]
        gi = {g: i for i, g in enumerate(m.genes)}
        out[name] = {
            "umi_pure": float(t[sel].sum()), "wrong_umi": float(wrong[sel].sum()),
            "wrong_fraction": float(wrong[sel].sum() / max(t[sel].sum(), 1)),
            "top_wrong": [(chrom.get(m.genes[i], ("", m.genes[i], ""))[1], float(per_gene_wrong[i])) for i in o],
            "watch": {}}
        for nme, gid in watch.items():
            if gid in gi:
                j = gi[gid]
                col = X[:, j].toarray().ravel()
                out[name]["watch"][nme] = {"in_human_cells": float(col[pure_h].sum()), "in_mouse_cells": float(col[pure_m].sum())}
        print(name, round(out[name]["wrong_fraction"], 5), flush=True)
    pairs = [("singlet.X(exon+intron)", "starsolo.GeneFull"), ("singlet.exon", "starsolo.Gene"), ("singlet.intron", "starsolo.velocyto.unspliced"),
             ("alevinfry.S+U+A", "starsolo.GeneFull"), ("alevinfry.S+A", "starsolo.Gene")]
    out["_excess"] = {}
    for x, y in pairs:
        if x in pgw and y in pgw:
            genes = set(pgw[x]) | set(pgw[y])
            d = {g: pgw[x].get(g, 0.0) - pgw[y].get(g, 0.0) for g in genes}
            top = sorted(d.items(), key=lambda kv: -kv[1])[:12]
            pos = sum(v for v in d.values() if v > 0); neg = sum(v for v in d.values() if v < 0)
            out["_excess"][f"{x} minus {y}"] = {"net_wrong_umi_difference": pos + neg, "sum_positive": pos, "sum_negative": neg,
                "top_positive": [(chrom.get(g, ("", g, ""))[1] or g, v, pgw[x].get(g, 0.0), pgw[y].get(g, 0.0)) for g, v in top]}
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
