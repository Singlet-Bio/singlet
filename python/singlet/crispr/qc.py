# SPDX-License-Identifier: MIT
"""QC of one CRISPR guide-capture library against its paired GEX sample. A library that fails is not attached.

| metric | meaning | fail | warn |
|---|---|---|---|
| ``frac_valid_barcode`` | reads whose cell barcode is on the whitelist after correction | < 0.50 | < 0.75 |
| ``frac_protospacer`` | reads holding the protospacer window (anchor found / fixed offset in range) | < 0.20 | < 0.50 |
| ``frac_guide`` | reads whose window is a guide of the library (exact or unique Hamming 1) | < 0.15 | < 0.40 |
| ``guide_unexplained`` | reads with a window that matches no guide, among reads with a window (an incomplete or wrong guide list) | > 0.50 | > 0.20 |
| ``frac_usable_in_cells`` | reads of molecules in called cells | < 0.05 | < 0.25 |
| ``cell_empty_ratio`` | median guide UMIs of called cells / of empty droplets (pairing check) | < 3 | < 10 |
| ``frac_cells_assigned`` | called cells with >= 1 guide (Singlet caller) | < 0.20 | < 0.50 |
| ``multiple_rate`` | assigned cells with > 1 guide (high-MOI designs warn by design) | - | > 0.40 |

Also reported: guides detected (assigned in >= 1 cell), median guide UMIs per assigned cell, and, with GEX, the target
knockdown (median over targeting guides of 1 - mean target expression / non-targeting mean; reported, not a rule:
CRISPRa raises the target and knockout often leaves the transcript).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from singlet.vdj.qc import verdict

THRESHOLDS = {
    "frac_valid_barcode": ("<", 0.50, 0.75),
    "frac_protospacer": ("<", 0.20, 0.50),
    "frac_guide": ("<", 0.15, 0.40),
    "guide_unexplained": (">", 0.50, 0.20),
    "frac_usable_in_cells": ("<", 0.05, 0.25),
    "cell_empty_ratio": ("<", 3.0, 10.0),
    "frac_cells_assigned": ("<", 0.20, 0.50),
    "multiple_rate": (">", None, 0.40),
}


def library_qc(lc, called_mask: np.ndarray, labels) -> Dict:
    """``lc``: :class:`singlet.adt.count.LibraryCounts` of a guide library (columns = GEX columns); ``called_mask``:
    bool per column; ``labels``: :func:`singlet.crispr.assign.summarize` of the called cells."""
    st = lc.stats
    reads = max(1, st.get("reads", 0))
    m: Dict = {}
    m["frac_valid_barcode"] = st.get("reads_valid_barcode", 0) / reads
    with_win = reads - st.get("reads_without_protospacer", 0)
    m["frac_protospacer"] = with_win / reads
    feat = st.get("reads_feature_exact", 0) + st.get("reads_feature_corrected", 0)
    m["frac_guide"] = feat / reads
    m["guide_unexplained"] = max(0.0, 1 - feat / max(1, with_win))
    M = lc.matrix
    called_tot = np.asarray(M[:, np.flatnonzero(called_mask)].sum(axis=0)).ravel()
    reads_cells = st.get("reads_in_gex_columns", 0) * (called_tot.sum() / max(1, st.get("molecules_in_gex_columns", 1)))
    m["frac_usable_in_cells"] = float(reads_cells / reads)
    empties = lc.outside_totals
    unc = np.asarray(M[:, np.flatnonzero(~called_mask)].sum(axis=0)).ravel()
    bg = np.concatenate([unc, empties]) if len(unc) + len(empties) else np.zeros(1)
    m["median_umi_cells"] = float(np.median(called_tot)) if len(called_tot) else 0.0
    m["cell_empty_ratio"] = float(m["median_umi_cells"] / max(1.0, np.median(bg[bg > 0]) if (bg > 0).any() else 1.0))
    cls = labels["class"].to_numpy()
    m["frac_cells_assigned"] = float((cls != "unassigned").mean()) if len(cls) else 0.0
    m["multiple_rate"] = float((cls == "multiple").sum() / max(1, (cls != "unassigned").sum()))
    ids = labels.loc[labels["class"] != "unassigned", "guide_ids"].str.split("|").explode()
    m["guides_detected"] = int(ids.nunique()) if len(ids) else 0
    m["guides_in_library"] = int(M.shape[0])
    out = dict(m)
    out.update(verdict(THRESHOLDS, m))
    return out


def knockdown(expr_target: Dict[str, np.ndarray], labels, guide_targets: Dict[str, str], control_mask: np.ndarray) -> Dict:
    """Per guide: 1 - mean target expression in its single-guide cells / in non-targeting cells.

    ``expr_target``: gene -> normalised expression per cell (same cells as ``labels``); ``guide_targets``: guide id ->
    target gene; ``control_mask``: cells whose single guide is non-targeting."""
    out = {}
    single = labels["class"].to_numpy() == "single"
    gids = labels["guide_ids"].to_numpy()
    for g, t in guide_targets.items():
        if t not in expr_target:
            continue
        e = expr_target[t]
        cm = e[control_mask]
        sel = single & (gids == g)
        if sel.sum() < 5 or cm.mean() <= 0:
            continue
        out[g] = {"target": t, "cells": int(sel.sum()), "knockdown": float(1 - e[sel].mean() / cm.mean())}
    return out
