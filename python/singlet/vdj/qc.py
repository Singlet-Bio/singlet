# SPDX-License-Identifier: MIT
"""QC of one V(D)J library against its paired GEX sample. A library that fails is not attached.

Library metrics (:func:`library_qc`) and their rules (``THRESHOLDS``; ``fail`` blocks publication, ``warn`` is published
with the flag). The cuts sit below what every Cell Ranger reference library in the benchmark reaches
(``docs/vdj-crispr-modalities.md``), so a library failing them is broken or mislabelled rather than merely shallow.

| metric | meaning | fail | warn |
|---|---|---|---|
| ``frac_on_target`` | mates with a V, J or C segment hit / all mates used | < 0.20 | < 0.50 |
| ``frac_valid_barcode`` | on-target mates whose barcode is on the whitelist after correction | < 0.50 | < 0.75 |
| ``frac_vdj_cells_called`` | V(D)J cells (>= 1 productive kept contig with >= 3 UMIs) that are called GEX cells: the pairing check | < 0.30 | < 0.60 |
| ``called_vs_uncalled`` | share of called GEX cells with a productive contig / share of uncalled GEX columns with one | < 3 | < 10 |
| ``frac_called_with_contig`` | called GEX cells with a productive contig (low in tissues with few lymphocytes) | < 0.01 | < 0.05 |
| ``pairing_rate`` | V(D)J cells with a productive pair (TRA+TRB, TRG+TRD, IGH+IGK/L) | < 0.20 | < 0.40 |
| ``productive_frac`` | kept contigs that are productive | < 0.50 | < 0.70 |
| ``excess_chain_rate`` | V(D)J cells with > 2 productive contigs of one chain (doublets, ambient) | > 0.20 | > 0.10 |

Also reported: the receptor composition (TCR / BCR share of productive contigs: a TCR-enriched library is mostly TCR),
UMIs per contig, the number of clonotypes and the largest clonotype's share.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

THRESHOLDS = {
    "frac_on_target": ("<", 0.20, 0.50),
    "frac_valid_barcode": ("<", 0.50, 0.75),
    "frac_vdj_cells_called": ("<", 0.30, 0.60),
    "called_vs_uncalled": ("<", 3.0, 10.0),
    "frac_called_with_contig": ("<", 0.01, 0.05),
    "pairing_rate": ("<", 0.20, 0.40),
    "productive_frac": ("<", 0.50, 0.70),
    "excess_chain_rate": (">", 0.20, 0.10),
}
VDJ_CELL_MIN_UMIS = 3
TCR = ("TRA", "TRB", "TRG", "TRD")


def rule(thresholds, metric, value) -> str:
    if metric not in thresholds or value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "na"
    op, fail, warn = thresholds[metric]
    bad = (lambda t: value < t) if op == "<" else (lambda t: value > t)
    if fail is not None and bad(fail):
        return "fail"
    if warn is not None and bad(warn):
        return "warn"
    return "pass"


def verdict(thresholds, metrics: Dict) -> Dict:
    rules = {k: rule(thresholds, k, v) for k, v in metrics.items() if k in thresholds}
    failed = sorted(k for k, r in rules.items() if r == "fail")
    warned = sorted(k for k, r in rules.items() if r == "warn")
    return {"verdict": "fail" if failed else ("warn" if warned else "pass"), "failed": failed, "warned": warned,
            "rules": rules}


def library_qc(contigs, cells, stats: Dict, gex_called: Optional[Sequence[str]] = None,
               gex_columns: Optional[Sequence[str]] = None) -> Dict:
    """``contigs``: :data:`singlet.vdj.assemble.CONTIG_COLUMNS` frame; ``cells``: :func:`singlet.vdj.clonotype.cell_table`
    (+ clonotypes); ``stats``: :class:`singlet.vdj.assemble.LibraryResult` stats."""
    mates = stats.get("read_pairs", 0) * (2 if stats.get("orientation", {}).get("r1_use") else 1)
    m: Dict = {}
    m["frac_on_target"] = stats.get("mates_with_segment", 0) / max(1, mates)
    m["frac_valid_barcode"] = stats.get("mates_valid_barcode", 0) / max(1, stats.get("mates_with_segment", 0))
    pk = contigs[contigs["kept"] & contigs["productive"]]
    m["productive_frac"] = float(len(pk) / max(1, int(contigs["kept"].sum())))
    strong = set(pk.loc[pk["umis"] >= VDJ_CELL_MIN_UMIS, "barcode"])
    m["vdj_cells"] = len(strong)
    with_contig = set(pk["barcode"])
    if gex_called is not None:
        called = set(gex_called)
        m["frac_vdj_cells_called"] = len(strong & called) / max(1, len(strong))
        m["frac_called_with_contig"] = len(with_contig & called) / max(1, len(called))
        if gex_columns is not None:
            unc = set(gex_columns) - called
            r_unc = len(with_contig & unc) / max(1, len(unc))
            m["called_vs_uncalled"] = m["frac_called_with_contig"] / max(r_unc, 1e-4)
    if len(cells):
        c = cells[cells["barcode"].isin(strong)]
        m["pairing_rate"] = float(c["paired"].mean()) if len(c) else 0.0
        m["excess_chain_rate"] = float((c["excess_chains"] > 0).mean()) if len(c) else 0.0
        if "clonotype_id" in c and len(c):
            vc = c["clonotype_id"].value_counts()
            m["clonotypes"] = int(len(vc))
            m["top_clonotype_share"] = float(vc.iloc[0] / len(c))
    tcr = pk["chain"].isin(TCR).mean() if len(pk) else float("nan")
    m["tcr_share"] = float(tcr)
    m["receptor"] = "TCR" if tcr >= 0.8 else ("BCR" if tcr <= 0.2 else "mixed")
    m["median_umis_per_contig"] = float(pk["umis"].median()) if len(pk) else 0.0
    out = dict(m)
    out.update(verdict(THRESHOLDS, m))
    return out
