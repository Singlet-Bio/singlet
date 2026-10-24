# SPDX-License-Identifier: MIT
"""QC of one ADT/HTO library against its paired GEX sample. Nothing is published unless the library passes.

Library metrics (``library_qc``) and the rule each enters (``THRESHOLDS``; ``fail`` blocks publication, ``warn`` is
reported with the data):

| metric | meaning | fail | warn |
|---|---|---|---|
| ``frac_valid_barcode`` | reads whose cell barcode is on the whitelist after correction | < 0.50 | < 0.75 |
| ``frac_feature`` | reads whose feature barcode is in the panel (exact or Hamming 1) | < 0.20 | < 0.50 |
| ``frac_usable_in_cells`` | reads in called cells with a valid barcode and feature (Cell Ranger "usable") | < 0.05 | < 0.25 |
| ``cell_empty_ratio`` | median ADT UMIs of called cells / median of empty droplets: the pairing check (an ADT library that does not hold these cells has ratio ~1) | < 3 | < 10 |
| ``frac_cells_with_adt`` | called cells with at least ``MIN_CELL_UMI`` ADT UMIs | < 0.50 | < 0.80 |
| ``median_umi_cells`` | median ADT UMIs per called cell | < 20 | < 100 |
| ``rna_concordance`` | median Spearman, over antibodies whose gene is measured, of CLR(ADT) vs log RNA in called cells; canonical markers listed separately | < 0.05 (when >= 3 pairs) | < 0.15 |
| ``panel_unexplained`` | share of feature-like reads on abundant sequences that are not in the panel | > 0.20 | > 0.05 |
| ``panel_identity`` | panels identified from the reads: share of panel reads on catalog barcodes whose reactivity excludes the organism or that a published panel used for another antibody | > 0.10 | > 0.02 |
| ``saturation`` | 1 - molecules / usable reads | - | < 0.20 |
| ``isotype_share_median`` | median per-cell share of isotype-control UMIs | - | > 0.10 |
| ``hto_singlet_share`` | hashtag libraries: called cells assigned to one hashtag | < 0.30 | < 0.50 |

Panel-level ``panel_detected``: antibodies whose mean in cells is at least twice their ambient-scaled mean in empty
droplets (reported, not a rule: a tissue may simply lack an antigen).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

MIN_CELL_UMI = 10
THRESHOLDS = {
    "frac_valid_barcode": ("<", 0.50, 0.75),
    "frac_feature": ("<", 0.20, 0.50),
    "frac_usable_in_cells": ("<", 0.05, 0.25),
    "cell_empty_ratio": ("<", 3.0, 10.0),
    "frac_cells_with_adt": ("<", 0.50, 0.80),
    "median_umi_cells": ("<", 20, 100),
    "rna_concordance": ("<", 0.05, 0.15),
    "panel_unexplained": (">", 0.20, 0.05),
    "panel_identity": (">", 0.10, 0.02),
    "saturation": ("<", None, 0.20),
    "isotype_share_median": (">", None, 0.10),
    "hto_singlet_share": ("<", 0.30, 0.50),
}

# antibody name -> gene symbol, for panels without an Ensembl id (study references) and as a cross-check
CANONICAL = {"CD3": "CD3E", "CD4": "CD4", "CD8": "CD8A", "CD8a": "CD8A", "CD14": "CD14", "CD19": "CD19",
             "CD20": "MS4A1", "CD56": "NCAM1", "CD16": "FCGR3A", "CD11c": "ITGAX", "CD11b": "ITGAM",
             "CD25": "IL2RA", "CD127": "IL7R", "CD279": "PDCD1", "PD-1": "PDCD1", "CD45": "PTPRC",
             "CD34": "CD34", "CD38": "CD38", "CD27": "CD27", "CD28": "CD28", "CD69": "CD69", "HLA-DR": "HLA-DRA",
             "CD197": "CCR7", "CCR7": "CCR7", "CD161": "KLRB1", "CD1c": "CD1C", "CD123": "IL3RA", "CD141": "THBD",
             "CD86": "CD86", "CD2": "CD2", "CD5": "CD5", "CD7": "CD7", "CD33": "CD33", "CD36": "CD36",
             "CD44": "CD44", "CD62L": "SELL", "CD24": "CD24", "CD9": "CD9", "CD10": "MME", "CD15": "FUT4",
             "CD26": "DPP4", "CD39": "ENTPD1", "CD57": "B3GAT1", "CD71": "TFRC", "CD95": "FAS", "CD103": "ITGAE",
             "CD137": "TNFRSF9", "CD152": "CTLA4", "CD223": "LAG3", "TIGIT": "TIGIT", "CD366": "HAVCR2"}
CANONICAL_MARKERS = ("CD3E", "CD4", "CD8A", "MS4A1", "CD14", "CD19", "NCAM1", "FCGR3A")


def _rule(metric, value):
    if metric not in THRESHOLDS or value is None or (isinstance(value, float) and np.isnan(value)):
        return "na"
    op, fail, warn = THRESHOLDS[metric]
    bad = (lambda t: value < t) if op == "<" else (lambda t: value > t)
    if fail is not None and bad(fail):
        return "fail"
    if warn is not None and bad(warn):
        return "warn"
    return "pass"


def gene_lookup(genes: List[str]):
    """Rows of each gene by Ensembl id (version stripped) and by symbol.

    Row names may be genes (``ENSG..``, ``SYMBOL``, ``ENSG..|SYMBOL``) or the legacy exon rows
    ``ENSG..._SYMBOL_chr:start-end``; a gene's rows are summed for concordance. -> (by_id, by_sym): name -> list of rows."""
    by_id, by_sym = {}, {}
    for i, g in enumerate(genes):
        g = str(g)
        if g.startswith(("ENSG", "ENSMUSG")) and g.count("_") >= 2:
            parts = g.split("_")[:2]  # exon row: id, symbol
        else:
            parts = g.split("|")
        for part in parts:
            p = part.strip()
            if p.startswith(("ENSG", "ENSMUSG")):
                by_id.setdefault(p.split(".")[0], []).append(i)
            elif p:
                by_sym.setdefault(p.upper(), []).append(i)
    return by_id, by_sym


def _spearman(a, b):
    from scipy.stats import spearmanr

    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(spearmanr(a, b)[0])


def rna_concordance(adt_clr: np.ndarray, panel: pd.DataFrame, exon, genes: List[str], cell_cols: np.ndarray,
                    min_frac: float = 0.05) -> pd.DataFrame:
    """Per antibody with a measured gene: Spearman(CLR ADT, log1p RNA per 10k) over called cells, plus a permutation null."""
    by_id, by_sym = gene_lookup(genes)
    E = exon[:, cell_cols]
    lib = np.asarray(E.sum(axis=0)).ravel()
    lib[lib == 0] = 1
    rng = np.random.default_rng(0)
    rows = []
    for i, r in enumerate(panel.itertuples(index=False)):
        if getattr(r, "kind", "antibody") != "antibody":
            continue
        gi = None
        ens = str(getattr(r, "ensembl", "") or "")
        for e in ens.replace(",", "|").split("|"):
            if e.split(".")[0] in by_id:
                gi = by_id[e.split(".")[0]]
                break
        if gi is None:
            nm = str(r.name).replace("Hu.", "").replace("Ms.", "").replace("Mm.", "").split(" (")[0].strip()
            nm = nm.split("_")[0]  # 'CD4_RPA.T4' -> 'CD4'
            sym = CANONICAL.get(nm, CANONICAL.get(nm.upper(), nm))
            gi = by_sym.get(sym.upper())
        if gi is None:
            continue
        g = np.asarray(E[gi].sum(axis=0)).ravel().astype(float)
        if (g > 0).mean() < min_frac:
            continue
        lg = np.log1p(g / lib * 1e4)
        a = adt_clr[i]
        gname = "_".join(str(genes[gi[0]]).split("_")[:2])
        rows.append({"feature": r.name, "gene_row": gname, "rho": _spearman(a, lg),
                     "rho_null": _spearman(rng.permutation(a), lg), "frac_rna_detected": float((g > 0).mean())})
    return pd.DataFrame(rows)


def library_qc(M, panel: pd.DataFrame, called: np.ndarray, stats: dict, ambient: np.ndarray,
               exon=None, genes: Optional[List[str]] = None, discovery: Optional[dict] = None) -> Dict:
    """M: features x GEX columns (scipy sparse), panel rows in the same order, called: bool per column."""
    from . import hto as H
    from . import normalize as N

    called = np.asarray(called, dtype=bool)
    X = M.tocsc()
    totals = np.asarray(X.sum(axis=0)).ravel()
    cells = np.flatnonzero(called)
    emp_mask = N.empty_droplets(totals, called)
    empties = np.flatnonzero(emp_mask)
    kinds = panel.kind.values
    iso = np.flatnonzero(kinds == "isotype")
    hto_rows = np.flatnonzero(kinds == "hashtag")
    ab_rows = np.flatnonzero(kinds == "antibody")
    reads = max(stats.get("reads", 0), 1)
    out = {
        "n_columns": int(X.shape[1]), "n_called": int(len(cells)), "n_empty_droplets": int(len(empties)),
        "n_features": int(X.shape[0]), "n_antibodies": int(len(ab_rows)), "n_isotypes": int(len(iso)),
        "n_hashtags": int(len(hto_rows)), "reads": int(stats.get("reads", 0)),
        "frac_valid_barcode": stats.get("reads_valid_barcode", 0) / reads,
        "frac_feature": (stats.get("reads_feature_exact", 0) + stats.get("reads_feature_corrected", 0)) / reads,
        "saturation": stats.get("saturation"),
    }
    ct = totals[cells]
    et = totals[empties]
    # usable reads in cells: molecules in cells weighted by reads is not kept per column; use molecules share x usable
    in_cells = X[:, cells].sum()
    out["umis_in_cells"] = int(in_cells)
    out["frac_umis_in_cells"] = float(in_cells / max(stats.get("molecules", 1), 1))
    out["frac_usable_in_cells"] = float(out["frac_umis_in_cells"] * stats.get("reads_usable", 0) / reads)
    out["median_umi_cells"] = float(np.median(ct)) if len(ct) else 0.0
    out["median_umi_empty"] = float(np.median(et)) if len(et) else float("nan")
    out["cell_empty_ratio"] = float(out["median_umi_cells"] / max(out["median_umi_empty"], 1.0)) if len(et) else float("nan")
    out["frac_cells_with_adt"] = float((ct >= MIN_CELL_UMI).mean()) if len(ct) else 0.0
    # uncalled columns as rich in ADT as a typical called cell: cells the GEX calling may have missed (reported only)
    if len(ct):
        rich = (~called) & (totals >= max(np.median(ct), MIN_CELL_UMI))
        out["n_uncalled_adt_rich"] = int(rich.sum())
        out["frac_umis_uncalled_adt_rich"] = float(totals[rich].sum() / max(totals.sum(), 1))
    out["ambient_umis"] = int(np.asarray(ambient).sum())

    Xc = X[:, cells].toarray().astype(float) if len(cells) else np.zeros((X.shape[0], 0))
    Xe = X[:, empties].toarray().astype(float) if len(empties) else np.zeros((X.shape[0], 0))
    if len(iso) and len(cells):
        share = Xc[iso].sum(axis=0) / np.maximum(Xc.sum(axis=0), 1)
        out["isotype_share_median"] = float(np.median(share))
    else:
        out["isotype_share_median"] = None
    # detection above background (scaled by library size of cells vs empties)
    if len(cells) and len(empties):
        sc = Xc.sum() / max(Xc.shape[1], 1)
        se = Xe.sum() / max(Xe.shape[1], 1)
        mc = Xc.mean(axis=1) / max(sc, 1e-9)
        me = Xe.mean(axis=1) / max(se, 1e-9)
        det = (Xc.mean(axis=1) >= 2 * Xe.mean(axis=1)) & (Xc.mean(axis=1) > 0.1)
        out["panel_detected"] = float(det[ab_rows].mean()) if len(ab_rows) else None
        out["panel_detected_list"] = [panel.name.iloc[i] for i in ab_rows if det[i]]
        out["enrichment_vs_empty_median"] = float(np.median((mc / np.maximum(me, 1e-9))[ab_rows])) if len(ab_rows) else None
    if discovery:
        out["panel_unexplained"] = discovery.get("unexplained_share")
        if (panel.get("source", pd.Series(["study"])) == "catalog").any():
            out["panel_identity"] = max(discovery.get("reactivity_mismatch_share", 0.0), discovery.get("reused_share", 0.0))
            out["panel_identity_detail"] = {"reactivity_mismatch": discovery.get("reactivity_mismatch", []),
                                            "reused": discovery.get("reused", [])}
    # RNA concordance
    conc = None
    if exon is not None and genes and len(ab_rows) and len(cells) > 20:
        clr = N.clr(Xc)
        conc = rna_concordance(clr, panel, exon, genes, cells)
        if len(conc):
            out["rna_concordance"] = float(np.nanmedian(conc.rho))
            out["rna_concordance_null"] = float(np.nanmedian(conc.rho_null))
            out["rna_pairs"] = int(conc.rho.notna().sum())
            canon = conc[conc.gene_row.str.upper().isin(CANONICAL_MARKERS) |
                         conc.gene_row.str.contains("|".join(CANONICAL_MARKERS), regex=True)]
            out["rna_canonical"] = {r.feature: round(r.rho, 3) for r in canon.itertuples() if not np.isnan(r.rho)}
            if out["rna_pairs"] < 3:
                out["rna_concordance_rule"] = "not assessable (< 3 pairs)"
    # HTO
    hto_calls = None
    if len(hto_rows) >= 2 and len(cells):
        amb = Xe[hto_rows].sum(axis=1) if len(empties) else np.asarray(ambient)[hto_rows]
        call, lt, ls = H.hashed_drops(Xc[hto_rows], amb)
        hto_calls = (call, lt, ls)
        out["hto_singlet_share"] = float((call >= 0).mean())
        out["hto_doublet_share"] = float((call == -2).mean())
        out["hto_negative_share"] = float((call == -1).mean())
        out["hto_counts"] = {panel.name.iloc[hto_rows[i]]: int((call == i).sum()) for i in range(len(hto_rows))}
    # rules
    rules = {}
    for k in THRESHOLDS:
        v = out.get(k)
        if k == "rna_concordance" and out.get("rna_pairs", 0) < 3:
            v = None
        if k == "hto_singlet_share" and len(hto_rows) < 2:
            v = None
        if k in ("frac_cells_with_adt", "median_umi_cells", "rna_concordance") and len(ab_rows) == 0:
            v = None  # hashtag-only library: judged by the hashtag rules
        rules[k] = _rule(k, v)
    out["rules"] = rules
    out["verdict"] = "fail" if "fail" in rules.values() else ("warn" if "warn" in rules.values() else "pass")
    out["failed"] = [k for k, v in rules.items() if v == "fail"]
    out["warned"] = [k for k, v in rules.items() if v == "warn"]
    return {"library": out, "concordance": conc, "hto": hto_calls, "cells": cells, "empties": empties}


def cell_table(M, panel: pd.DataFrame, cells: np.ndarray, hto=None) -> pd.DataFrame:
    """Per called cell: col, adt_total, n_adt_detected, isotype_share (+ hto_call, hto_lfc_top, hto_lfc_second)."""
    X = M.tocsc()[:, cells]
    kinds = panel.kind.values
    ab = np.flatnonzero(kinds == "antibody")
    iso = np.flatnonzero(kinds == "isotype")
    A = X[np.concatenate([ab, iso])] if len(ab) + len(iso) else None
    tot = np.asarray(A.sum(axis=0)).ravel() if A is not None else np.zeros(len(cells))
    df = pd.DataFrame({"col": cells.astype(np.int64), "adt_total": tot.astype(np.int64),
                       "n_adt_detected": (np.asarray((X[ab] > 0).sum(axis=0)).ravel() if len(ab) else 0)})
    if len(iso):
        df["isotype_share"] = (np.asarray(X[iso].sum(axis=0)).ravel() / np.maximum(tot, 1)).astype(np.float32)
    if hto is not None:
        df["hto_call"], df["hto_lfc_top"], df["hto_lfc_second"] = hto[0], hto[1].astype(np.float32), hto[2].astype(np.float32)
    return df
