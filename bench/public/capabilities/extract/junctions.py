# SPDX-License-Identifier: MIT
"""Extractor for junctions-psi: per-junction, per-cell and per-cassette-exon tables (Clipper side).

Tools: Singlet (sj_counts, molecules per junction and barcode), STARsolo at STAR/STARsolo defaults (the reference: Solo.out/SJ, UMIs)
and STARsolo in Cell Ranger 4+ emulation (the competitor; its MultiGeneUMI_CR filter removes UMIs of junctions that lie in
overlapping genes, so it is not the natural reference for junction counts). Junctions are keyed by STAR's convention (chromosome, first intron base, last intron base).
Singlet's *filtered* junction set is the set of junctions its own STAR alignment passes through STAR's outSJfilter* rules
(``star_sj_pass`` in sj_features.tsv); STARsolo's set is the feature list of its SJ matrix, which STAR filters the same way.

Output per dataset:
  <ds>.junctions.tsv.gz  chrom start end annotated, f:<tool> (in the filtered set), n:<tool> (UMI over cells), c:<tool> (cells)
  <ds>.cells.tsv.gz      barcode, t:<tool> (UMI on the reference junctions), d:<tool> (distinct junctions),
                         r:<tool>~starsolo_default (per-cell log1p correlation over junctions non-zero in either tool)
  <ds>.cassette.tsv.gz   annotated cassette exons with all three junctions seen: inclusion / skipping UMI per tool (pseudobulk)
  <ds>.psi_cells.tsv.gz  (cell, event) entries with coverage >= 5 in the reference: inclusion / skipping per tool
"""
from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

import runio as R
from counts import Ctx
from pairstats import percell_log_corr_union

TOOLS = ["singlet", "starsolo_cr", "starsolo_default"]
MAX_PSI_ENTRIES = 600_000


def load_starsolo_sj(run: Path, cells: list[str]):
    d = run / "Solo.out" / "SJ" / "raw"
    feats = pd.read_csv(d / "features.tsv", sep="\t", header=None)
    keys = list(zip(feats[0].astype(str), feats[1].astype(int), feats[2].astype(int)))
    bcs = [R.bare(b) for b in R.read_lines(d / "barcodes.tsv")]
    M = R.read_mtx(d / "matrix.mtx")
    return keys, bcs, M


def load_singlet_sj(run: Path):
    names = R.read_lines(run / "sj_counts_features.tsv.gz")
    bcs = [R.bare(b) for b in R.read_lines(run / "sj_counts_barcodes.tsv.gz")]
    M = R.read_mtx(run / "sj_counts.mtx.gz")
    feat = pd.read_csv(run / "sj_features.tsv.gz", sep="\t")
    info = feat.set_index("junction")
    keys = []
    for n in names:
        r = info.loc[n]
        keys.append((str(r["chrom"]), int(r["intron_start"]), int(r["intron_end"])))
    passed = {k for k, p in zip(zip(feat["chrom"].astype(str), feat["intron_start"].astype(int), feat["intron_end"].astype(int)),
                                 feat["star_sj_pass"]) if str(p) in ("1", "1.0", "True", "true")}
    annotated = {k for k, a in zip(zip(feat["chrom"].astype(str), feat["intron_start"].astype(int), feat["intron_end"].astype(int)),
                                    feat["annotated"]) if str(a) in ("1", "1.0")}
    return keys, bcs, M, passed, annotated


def cassette_triples(gtf: Path):
    """Annotated cassette exons: consecutive exons a, b, c of a transcript whose junctions a-b, b-c and a-c are all annotated
    (b shorter than 500 bp). Junctions in STAR coordinates (first and last intron base)."""
    tx = collections.defaultdict(list)
    with open(gtf) as fh:
        for ln in fh:
            if ln.startswith("#"):
                continue
            f = ln.rstrip("\n").split("\t")
            if f[2] != "exon":
                continue
            t = f[8].split('transcript_id "')[1].split('"')[0]
            tx[t].append((f[0], int(f[3]), int(f[4])))
    jset = set()
    for ex in tx.values():
        ex.sort(key=lambda x: x[1])
        for a, b in zip(ex[:-1], ex[1:]):
            if b[1] > a[2] + 1:
                jset.add((a[0], a[2] + 1, b[1] - 1))
    trip = set()
    for ex in tx.values():
        for a, b, c in zip(ex[:-2], ex[1:-1], ex[2:]):
            j3 = (a[0], a[2] + 1, c[1] - 1)
            j1 = (a[0], a[2] + 1, b[1] - 1)
            j2 = (b[0], b[2] + 1, c[1] - 1)
            if j3 in jset and j1 in jset and j2 in jset and j3 != j1 and (b[2] - b[1]) < 500:
                trip.add((j1, j2, j3))
    return sorted(trip)


def extract_junctions(ctx: Ctx, out: Path, gtf: Path) -> dict:
    cells = R.starsolo_cells(ctx.star_cr)
    cpos = {c: i for i, c in enumerate(cells)}
    # ---- load the three tools
    s_keys, s_bcs, s_M, s_pass, s_ann = load_singlet_sj(ctx.singlet)
    cr_keys, cr_bcs, cr_M = load_starsolo_sj(ctx.star_cr, cells)
    ref_run = ctx.star_reference if (ctx.star_reference / "Solo.out" / "SJ" / "raw").exists() else ctx.star_default   # see counts.Ctx
    df_keys, df_bcs, df_M = load_starsolo_sj(ref_run, cells)
    raw = {"singlet": (s_keys, s_bcs, s_M), "starsolo_cr": (cr_keys, cr_bcs, cr_M), "starsolo_default": (df_keys, df_bcs, df_M)}
    filt = {"singlet": s_pass, "starsolo_cr": set(cr_keys), "starsolo_default": set(df_keys)}
    # annotated flag from STAR's own SJ.out.tab of the reference run (column 6) plus Singlet's flag
    ann = set(s_ann)
    sj_tab = ctx.star_cr / "SJ.out.tab"
    if sj_tab.exists():
        t = pd.read_csv(sj_tab, sep="\t", header=None)
        ann |= {(str(a), int(b), int(c)) for a, b, c, f in zip(t[0], t[1], t[2], t[5]) if f == 1}
    # ---- universe and matrices (cells x junctions)
    uni: dict = {}
    for keys in (s_keys, cr_keys, df_keys):
        for k in keys:
            uni.setdefault(k, len(uni))
    nj = len(uni)
    mats = {}
    present = {}
    for tool, (keys, bcs, M) in raw.items():
        pos = {b: i for i, b in enumerate(bcs)}
        take = [pos.get(c, -1) for c in cells]
        present[tool] = np.array([t >= 0 for t in take])
        sel = [t for t in take if t >= 0]
        sub = M[:, sel].tocoo()                                   # junctions x selected cells
        rows = np.array([uni[k] for k in keys], dtype=np.int64)[sub.row]
        cols = np.nonzero(present[tool])[0][sub.col]
        mats[tool] = sp.csr_matrix((sub.data.astype(np.float64), (cols, rows)), shape=(len(cells), nj))
    inv = [None] * nj
    for k, i in uni.items():
        inv[i] = k
    # ---- junction table
    jt = pd.DataFrame({"chrom": [k[0] for k in inv], "start": [k[1] for k in inv], "end": [k[2] for k in inv]})
    jt["annotated"] = [int(k in ann) for k in inv]
    for tool in TOOLS:
        jt[f"f:{tool}"] = [int(k in filt[tool]) for k in inv]
        jt[f"n:{tool}"] = np.asarray(mats[tool].sum(axis=0)).ravel().astype(np.int64)
        jt[f"c:{tool}"] = np.diff(mats[tool].tocsc().indptr).astype(np.int64)
    keep = (jt[[f"f:{t}" for t in TOOLS]].sum(axis=1) > 0) | (jt[[f"n:{t}" for t in TOOLS]].sum(axis=1) > 0)
    jt[keep].to_csv(out / f"{ctx.ds}.junctions.tsv.gz", sep="\t", index=False)
    # ---- per cell, on the reference (STARsolo at defaults) junction list
    ref_cols = np.array([uni[k] for k in df_keys], dtype=np.int64)  # reference junction list: STARsolo at defaults
    ref = {t: mats[t][:, ref_cols].tocsr() for t in TOOLS}
    allp = np.ones(len(cells), bool)
    for t in TOOLS:
        allp &= present[t]
    cd = {"in_cells": allp.astype(int)}
    for t in TOOLS:
        tot = np.asarray(ref[t].sum(axis=1)).ravel().astype(float)
        dist = np.diff(ref[t].indptr).astype(float)
        tot[~allp], dist[~allp] = np.nan, np.nan
        cd[f"t:{t}"], cd[f"d:{t}"] = tot, dist
    idx = np.nonzero(allp)[0]
    for t in ("singlet", "starsolo_cr"):
        r = np.full(len(cells), np.nan)
        r[idx] = percell_log_corr_union(ref[t][idx], ref["starsolo_default"][idx])
        cd[f"r:{t}~starsolo_default"] = r
    pd.DataFrame(cd, index=pd.Index(cells, name="barcode")).to_csv(out / f"{ctx.ds}.cells.tsv.gz", sep="\t", float_format="%.6g")
    # ---- cassette exons
    trip = cassette_triples(gtf)
    T = np.array([[uni.get(j, -1) for j in t] for t in trip], dtype=np.int64)
    seen = (T >= 0).all(axis=1)
    trip = [t for t, s in zip(trip, seen) if s]
    T = T[seen]
    rows = {"event": [f"{t[0][0]}:{t[0][1]}-{t[0][2]}|{t[1][1]}-{t[1][2]}|{t[2][1]}-{t[2][2]}" for t in trip]}
    colsum = {t: np.asarray(mats[t][idx].sum(axis=0)).ravel() for t in TOOLS}
    for t in TOOLS:
        rows[f"i:{t}"] = (colsum[t][T[:, 0]] + colsum[t][T[:, 1]]) / 2.0
        rows[f"s:{t}"] = colsum[t][T[:, 2]]
    pd.DataFrame(rows).to_csv(out / f"{ctx.ds}.cassette.tsv.gz", sep="\t", index=False, float_format="%.6g")
    # ---- per-cell PSI entries (coverage >= 5 in the reference)
    I, S_, COV = {}, {}, {}
    for t in TOOLS:
        m = mats[t][idx].tocsc()
        i_ = (m[:, T[:, 0]] + m[:, T[:, 1]]) / 2.0
        s_ = m[:, T[:, 2]]
        I[t], S_[t] = sp.csr_matrix(i_), sp.csr_matrix(s_)
        COV[t] = sp.csr_matrix(i_ + s_)
    cov = COV["starsolo_default"].tocoo()
    sel = cov.data >= 5
    ci_, ei_ = cov.row[sel], cov.col[sel]
    if len(ci_) > MAX_PSI_ENTRIES:
        keep = np.random.default_rng(20261025).choice(len(ci_), MAX_PSI_ENTRIES, replace=False)
        ci_, ei_ = ci_[keep], ei_[keep]
    ent = {"cell": ci_.astype(np.int32), "event": ei_.astype(np.int32)}
    for t in TOOLS:
        ent[f"i:{t}"] = np.asarray(I[t][ci_, ei_]).ravel()
        ent[f"s:{t}"] = np.asarray(S_[t][ci_, ei_]).ravel()
    pd.DataFrame(ent).to_csv(out / f"{ctx.ds}.psi_cells.tsv.gz", sep="\t", index=False, float_format="%.6g")
    return {"n_cells": len(cells), "n_cells_all_tools": int(allp.sum()), "n_junctions_union": nj,
            "n_filtered": {t: len(filt[t]) for t in TOOLS}, "n_cassette": len(trip), "n_psi_entries": int(len(ci_))}
