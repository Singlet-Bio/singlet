# SPDX-License-Identifier: MIT
"""Extractor for cell-calling: one row per candidate barcode with every tool's call and the truth (Clipper side).

Candidates are the union of every tool's called barcodes and the truth set. Output: <ds>.barcodes.tsv.gz

  barcode, umi_total (Singlet exon+intron UMI), umi_gene (STARsolo Gene UMI),
  singlet (called_v2, the Singlet 1.0.0 call), singlet_knee (the stored knee call, Cell Ranger 2.2 style; Singlet's own earlier call),
  starsolo_edcr (STARsolo EmptyDrops_CR), cellbender (CellBender 0.3.2 cell calls), cellranger (10x-published filtered barcodes),
  demuxlet (Kang: rows of the authors' demuxlet table, control lane), demuxlet_class (singlet / doublet / ambs),
  n_human, n_mouse, species_class (barnyard only: pure_human / pure_mouse / mixed / ambient_like)

Species classes follow bench/called_v2_truth/datasets.py: ambient pool = barcodes with 1-100 UMI that no tool calls; a barcode is
ambient-like when its human fraction is not significantly different (two-sided binomial, p >= 0.001) from the pool's; otherwise pure
(>= 90% one species) or mixed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import runio as R
from counts import Ctx

KANG_TRUTH = Path("/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz")


def _read_calls(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype={"barcode": str})
    df["barcode"] = df["barcode"].map(R.bare)
    return df.set_index("barcode")


def extract_cell_calling(ctx: Ctx, out: Path) -> dict:
    sc = _read_calls(ctx.singlet / "cell_calls.tsv")
    singlet_cells = set(sc.index[sc["called_v2"].astype(int) == 1])
    knee_cells = set(sc.index[sc["called_knee" if "called_knee" in sc.columns else "called"].astype(int) == 1])   # Singlet renamed the column called -> called_knee
    edcr = set(R.starsolo_cells(ctx.star_cr))
    cb_file = ctx.runs / "cellbender-public" / "cb_cell_barcodes.csv"
    cellbender = None
    if cb_file.exists():
        cellbender = {R.bare(ln.strip().split(",")[0]) for ln in cb_file.read_text().splitlines() if ln.strip() and not ln.startswith("barcode")}
    cellranger = None
    kang = None
    if ctx.ds == "kang2018_ctrl":
        t = pd.read_csv(KANG_TRUTH, sep="\t", index_col=0)
        t = t[t["stim"] == "ctrl"]
        t.index = [R.bare(str(b)) for b in t.index]
        kang = t
    else:
        cellranger = set(R.cellranger_published_barcodes(ctx.data))
    cand = set(singlet_cells) | knee_cells | edcr
    for s in (cellbender, cellranger, set(kang.index) if kang is not None else None):
        if s:
            cand |= s
    cand = sorted(cand)
    df = pd.DataFrame(index=pd.Index(cand, name="barcode"))
    df["umi_total"] = sc["umi_total"].reindex(cand).fillna(0).astype(np.int64)
    # STARsolo exonic UMI per barcode from the raw matrix
    d = ctx.star_cr / "Solo.out" / "Gene" / "raw"
    bcs = [R.bare(b) for b in R.read_lines(d / "barcodes.tsv")]
    tot = pd.Series(np.asarray(R.read_mtx(d / "matrix.mtx").sum(axis=0)).ravel().astype(np.int64), index=bcs)
    df["umi_gene"] = tot.reindex(cand).fillna(0).astype(np.int64)
    df["singlet"] = [int(b in singlet_cells) for b in cand]
    df["singlet_knee"] = [int(b in knee_cells) for b in cand]
    df["starsolo_edcr"] = [int(b in edcr) for b in cand]
    if cellbender is not None:
        df["cellbender"] = [int(b in cellbender) for b in cand]
    if cellranger is not None:
        df["cellranger"] = [int(b in cellranger) for b in cand]
    if kang is not None:
        df["demuxlet"] = [int(b in kang.index) for b in cand]
        df["demuxlet_class"] = kang["multiplets"].reindex(cand).fillna("").astype(str).to_numpy()
    info = {"n_candidates": len(cand), "singlet": len(singlet_cells), "singlet_knee": len(knee_cells), "starsolo_edcr": len(edcr),
            "cellbender": None if cellbender is None else len(cellbender),
            "truth": len(kang.index) if kang is not None else len(cellranger)}
    # barnyard species classes
    if ctx.ds == "hgmm1k_v3":
        genes, rbc, M = R.load_singlet_raw_total(ctx.singlet)
        human = np.array([g.startswith("ENSG") for g in genes])
        mouse = np.array([g.startswith("ENSMUSG") for g in genes])
        hu = pd.Series(np.asarray(M[human].sum(axis=0)).ravel(), index=rbc)
        mo = pd.Series(np.asarray(M[mouse].sum(axis=0)).ravel(), index=rbc)
        allcalled = singlet_cells | knee_cells | edcr | (cellbender or set()) | (cellranger or set())
        n_all = hu + mo
        pool = (n_all >= 1) & (n_all <= 100) & ~n_all.index.isin(allcalled)
        h_amb = float(hu[pool].sum() / max(1.0, n_all[pool].sum()))
        from scipy.stats import binom

        h = hu.reindex(cand).fillna(0).to_numpy()
        m = mo.reindex(cand).fillna(0).to_numpy()
        n = h + m
        p = np.minimum(1.0, 2 * np.minimum(binom.cdf(h, n, h_amb), binom.sf(h - 1, n, h_amb)))
        unlike = (p < 0.001) & (n > 0)
        hf = h / np.maximum(n, 1)
        cls = np.where(~unlike, "ambient_like", np.where(hf >= 0.9, "pure_human", np.where(hf <= 0.1, "pure_mouse", "mixed")))
        df["n_human"], df["n_mouse"], df["species_class"] = h.astype(np.int64), m.astype(np.int64), cls
        info["h_amb"] = h_amb
    df.to_csv(out / f"{ctx.ds}.barcodes.tsv.gz", sep="\t")
    return info


REVALIDATE = Path("/mnt/projects/debruinz_project/debruinz/singlet/state/called_v2_truth/report/revalidate.json")
VARIANT = "bounded knee + tail p, fdr .001"  # called_v2 2.1, what the campaign binary implements


def write_validation42(out: Path) -> dict:
    """Study-weighted Jaccard / recall / precision of called_v2 against the authors' Cell Ranger calls in the 42-sample validation
    (docs/called-v2.md, re-scored in docs/called-v2-truth.md by bench/called_v2_truth/revalidate.py)."""
    import json

    d = json.loads(REVALIDATE.read_text())
    v = d["variants"][VARIANT]
    rec = {"source": "bench/called_v2_truth/revalidate.py aggregate (state/called_v2_truth/report/revalidate.json)", "variant": VARIANT,
           "n_samples": d["n_samples"], "n_usable": d["n_usable"], "n_author_samples": d["n_author_samples"],
           "jaccard": v["jaccard"], "recall": v["recall"], "precision": v["precision"], "per_study_jaccard": v["per_study_jaccard"]}
    (out / "validation42.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return {"validation42": True}
