# SPDX-License-Identifier: MIT
"""Extractors for gex-counts and spliced-unspliced: per-cell tables from the tool outputs (Clipper side).

Cell set: the barcodes STARsolo (EmptyDrops_CR, Cell Ranger 4+ emulation) calls on the dataset. It does not depend on
Singlet's own cell caller. A family of tools is compared on the cells every member of the family has an output for, and on
the genes every member annotates. Output per dataset and capability:

  <ds>.cells.tsv.gz   barcode, <family>:n:<tool> (total UMI), <family>:r:<a>~<b> (per-cell log1p gene-vector correlation)
  <ds>.genes.tsv.gz   gene, <family>:<tool> (UMI summed over the family's cells), for the pseudobulk Spearman
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import runio as R
from pairstats import pair_corr


class Ctx:
    def __init__(self, pb: Path, ds: str):
        self.pb, self.ds = Path(pb), ds
        self.runs = self.pb / "runs" / ds
        self.data = self.pb / "data" / ds
        self.singlet = self.runs / "singlet-public"
        self.star_cr = self.runs / "starsolo-public-cr"
        self.star_default = self.runs / "starsolo-public-default"
        # 5' libraries: alevin-fry's expected orientation for 10xv2-5p is fw (simpleaf's own chemistry default)
        self.af = self.runs / ("alevinfry-fw" if ds == "pbmc5p_v1" else "alevinfry")


def _restrict(m: R.GeneMat, genes: np.ndarray) -> sp.csr_matrix:
    pos = {g: i for i, g in enumerate(m.genes)}
    return m.X[:, np.array([pos[g] for g in genes], dtype=np.int64)].tocsr()


def family_tables(name: str, tools: dict[str, R.GeneMat], cells: list[str], corr_pairs: bool = True):
    """Per-cell totals and pairwise correlations plus gene pseudobulk for one family of tools."""
    members = list(tools)
    genes = tools[members[0]].genes
    for t in members[1:]:
        genes = np.intersect1d(genes, tools[t].genes)
    mask = np.ones(len(cells), bool)
    for t in members:
        mask &= tools[t].present
    sub = {t: R.GeneMat(genes, _restrict(tools[t], genes), cells, tools[t].present) for t in members}
    cols = {}
    for t in members:
        n = np.asarray(sub[t].X.sum(axis=1)).ravel().astype(float)
        n[~mask] = np.nan
        cols[f"{name}:n:{t}"] = n
    if corr_pairs:
        for a, b in itertools.combinations(members, 2):
            cols[f"{name}:r:{a}~{b}"] = pair_corr(sub[a], sub[b], mask)
    gcols = {}
    idx = np.nonzero(mask)[0]
    for t in members:
        gcols[f"{name}:{t}"] = np.asarray(sub[t].X[idx].sum(axis=0)).ravel()
    return cols, pd.DataFrame(gcols, index=genes), mask


def extract_gex_counts(ctx: Ctx, out: Path) -> dict:
    cells = R.starsolo_cells(ctx.star_cr)
    s = R.load_singlet(ctx.singlet, cells)
    exon = s["E"]
    total = exon.add(s["U"])
    gene = R.load_starsolo(ctx.star_cr, "Gene", cells)
    genefull = R.load_starsolo(ctx.star_cr, "GeneFull", cells)
    af = R.load_alevinfry(ctx.af, cells)
    af_total = af["S"].add(af["U"]).add(af["A"])
    exonic = {"singlet_exon": exon, "starsolo_gene": gene}
    note = {}
    # the published Cell Ranger matrix of the 5' dataset comes from the full 448 M read pairs; ours is the first 60 M
    if ctx.ds != "pbmc5p_v1":
        cr, _ = R.load_cellranger_published(ctx.data, cells)
        if cr is not None:
            exonic["cellranger"] = cr
    else:
        note["cellranger"] = "excluded: published matrix is from the full-depth run, this dataset uses the first 60 M read pairs"
    cols, genes_tables = {}, []
    masks = {}
    for fam, tools in (("exonic", exonic), ("total", {"singlet_total": total, "starsolo_genefull": genefull,
                                                      "alevinfry_total": af_total})):
        c, g, mask = family_tables(fam, tools, cells)
        cols.update(c)
        genes_tables.append(g)
        masks[fam] = mask
    df = pd.DataFrame(cols, index=pd.Index(cells, name="barcode"))
    df.insert(0, "in_total", masks["total"].astype(int))
    df.insert(0, "in_exonic", masks["exonic"].astype(int))
    df.round(6).to_csv(out / f"{ctx.ds}.cells.tsv.gz", sep="\t", float_format="%.6g")
    g = pd.concat(genes_tables, axis=1)
    g.index.name = "gene"
    g.to_csv(out / f"{ctx.ds}.genes.tsv.gz", sep="\t", float_format="%.8g")
    if ctx.ds == "pbmc1k_cite_v3":
        note["adt_pairs"] = extract_adt_pairs(ctx, out, cells, {"singlet_exon": exon, "singlet_total": total, "starsolo_gene": gene,
                                                              "starsolo_genefull": genefull, "alevinfry_total": af_total})
    return {"n_cells": len(cells), "n_exonic": int(masks["exonic"].sum()), "n_total": int(masks["total"].sum()), **note}


# surface protein (10x TotalSeq-B panel name, lower case) -> gene symbol of the transcript that codes for it
ADT_GENE = {"cd3": "CD3E", "cd4": "CD4", "cd8a": "CD8A", "cd14": "CD14", "cd16": "FCGR3A", "cd56": "NCAM1", "cd19": "CD19",
            "cd25": "IL2RA", "pd-1": "PDCD1", "cd127": "IL7R", "tigit": "TIGIT", "cd15": "FUT4"}


def extract_adt_pairs(ctx: Ctx, out: Path, cells: list[str], tools: dict) -> int:
    """Orthogonal truth for gene counts: antibody counts (CITE-seq, Cell Ranger Antibody Capture) of the same cells next to each
    tool's UMI for the gene that codes for the protein. Writes <ds>.adt.tsv.gz: adt:<protein> (raw antibody UMI, every protein
    including isotype controls), tot:<tool> (the tool's total UMI over the cells) and rna:<tool>:<symbol>."""
    import h5py
    import scipy.sparse as sp

    h5 = next(iter(sorted(ctx.data.glob("cellranger_*.h5"))))
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        bcs = [R.bare(b.decode()) for b in g["barcodes"][:]]
        ftype = np.array([x.decode() for x in g["features"]["feature_type"][:]])
        names = [x.decode() for x in g["features"]["name"][:]]
    sel = np.nonzero(ftype == "Antibody Capture")[0]
    pos = {b: i for i, b in enumerate(bcs)}
    keep = [i for i, c in enumerate(cells) if c in pos]
    adt = M[sel][:, [pos[cells[i]] for i in keep]].toarray().T.astype(float)
    feats = pd.read_csv(ctx.star_cr / "Solo.out" / "Gene" / "raw" / "features.tsv", sep="\t", header=None)
    sym2id = {s: R.canon_gene(i) for i, s in zip(feats[0], feats[1])}
    df = pd.DataFrame(adt, index=[cells[i] for i in keep], columns=[f"adt:{names[i]}" for i in sel])
    for t, m in tools.items():
        df[f"tot:{t}"] = np.asarray(m.X[keep].sum(axis=1)).ravel()
        gpos = {gg: j for j, gg in enumerate(m.genes)}
        for prot in [names[i] for i in sel]:
            sym = ADT_GENE.get(prot.split("_")[0].lower())
            if sym and sym in sym2id and sym2id[sym] in gpos:
                df[f"rna:{t}:{sym}"] = np.asarray(m.X[keep][:, gpos[sym2id[sym]]].todense()).ravel()
    df.index.name = "barcode"
    df.to_csv(out / f"{ctx.ds}.adt.tsv.gz", sep="\t", float_format="%.6g")
    return len(keep)


def extract_spliced_unspliced(ctx: Ctx, out: Path) -> dict:
    cells = R.starsolo_cells(ctx.star_cr)
    s = R.load_singlet(ctx.singlet, cells)
    # one gene axis for Singlet's layers: genes without intron (or ambiguous) features count zero there
    universe = sorted({R.canon_gene(g) for g in R.singlet_known_genes(ctx.singlet)})
    E, U, A = (s[k].pad(universe) for k in ("E", "U", "A"))
    # spliced = exon_counts - ambiguous_counts (exon_counts is spliced + ambiguous)
    S = R.GeneMat(universe, E.X - A.X, cells, E.present)
    S.X.data = np.maximum(S.X.data, 0)
    S.X.eliminate_zeros()
    vel = {k: R.load_starsolo(ctx.star_cr, "Velocyto", cells, layer=k) for k in ("spliced", "unspliced", "ambiguous")}
    af = R.load_alevinfry(ctx.af, cells)
    layers = {
        "singlet": {"S": S, "U": U, "A": A},
        "starsolo": {"S": vel["spliced"], "U": vel["unspliced"], "A": vel["ambiguous"]},
        "alevinfry": {"S": af["S"], "U": af["U"], "A": af["A"]},
    }
    cols, genes_tables = {}, []
    # layer-level families: unspliced alone, and spliced + ambiguous
    fams = {
        "unspliced": {f"{t}_U": layers[t]["U"] for t in layers},
        "spliced_ambiguous": {f"{t}_SA": layers[t]["S"].add(layers[t]["A"]) for t in layers},
    }
    masks = {}
    for fam, tools in fams.items():
        c, g, mask = family_tables(fam, tools, cells)
        cols.update(c)
        genes_tables.append(g)
        masks[fam] = mask
    # per-cell totals of every layer (for the S / U / A shares), on the cells all three tools have
    allmask = masks["unspliced"] & masks["spliced_ambiguous"]
    G = np.intersect1d(np.intersect1d(S.genes, vel["spliced"].genes), af["S"].genes)
    for t in layers:
        for k in ("S", "U", "A"):
            m = layers[t][k].pad(list(G))
            n = np.asarray(m.X.sum(axis=1)).ravel().astype(float)
            n[~allmask] = np.nan
            cols[f"layers:n:{t}_{k}"] = n
    df = pd.DataFrame(cols, index=pd.Index(cells, name="barcode"))
    df.insert(0, "in_layers", allmask.astype(int))
    df.to_csv(out / f"{ctx.ds}.cells.tsv.gz", sep="\t", float_format="%.6g")
    g = pd.concat(genes_tables, axis=1)
    g.index.name = "gene"
    g.to_csv(out / f"{ctx.ds}.genes.tsv.gz", sep="\t", float_format="%.8g")
    return {"n_cells": len(cells), "n_layers": int(allmask.sum())}
