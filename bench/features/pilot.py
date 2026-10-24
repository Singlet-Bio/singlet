#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pilot of singlet.features on a few real samples: cost per 10k cells and score sanity.

    python bench/features/pilot.py --out OUT --sample GSE145370:v2:GSM4317423 --sample GSE138867:path/GSE138867.singlet:GSM4120733 ...

Writes the sidecars and summary rows (scripts/features_run.py layout) under OUT and ``OUT/pilot.json``: per sample the timings,
per coarse marker-defined cell group the median of selected scores, the share of cells per cell-cycle phase with MKI67/TOP2A
detection per phase, and the Spearman correlation of each score with the cell's detected-gene count (raw rank AUC vs the
null-adjusted score that is written). Groups come from a handful of canonical markers, for this check only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

GROUPS = {
    "T": ["CD3E", "CD3D", "CD2"],
    "NK": ["NKG7", "GNLY", "KLRD1"],
    "B": ["MS4A1", "CD79A", "CD19"],
    "plasma": ["JCHAIN", "MZB1", "IGHG1"],
    "myeloid": ["LYZ", "CD68", "CD14", "C1QA", "FCGR3A"],
    "epithelial": ["EPCAM", "KRT8", "KRT18", "KRT5", "KRT19"],
    "ccRCC_tumour": ["CA9", "NDUFA4L2", "PAX8"],
    "endothelial": ["PECAM1", "VWF", "CDH5"],
    "fibroblast": ["COL1A1", "DCN", "LUM"],
    "erythroid": ["HBB", "HBA1", "HBA2"],
}
SHOW = [
    "hypoxia",
    "interferon_alpha_response",
    "interferon_gamma_response",
    "inflammatory_response",
    "tnfa_signaling_via_nfkb",
    "oxidative_phosphorylation",
    "glycolysis",
    "e2f_targets",
    "g2m_checkpoint",
    "epithelial_mesenchymal_transition",
    "myc_targets_v1",
    "heme_metabolism",
    "allograft_rejection",
]


def groups(X, gene_names, species):
    gi = {}
    for j, g in enumerate(gene_names):
        gi.setdefault(str(g).upper(), j)
    tot = np.asarray(X.sum(1)).ravel()
    names, mat = [], []
    for k, ms in GROUPS.items():
        idx = [gi[m.upper()] for m in ms if m.upper() in gi]
        if not idx:
            continue
        v = np.asarray(X[:, idx].sum(1)).ravel() / np.maximum(tot, 1) * 1e4
        names.append(k)
        mat.append(np.log1p(v))
    M = np.vstack(mat).T
    best = M.argmax(1)
    srt = np.sort(M, 1)
    ok = (srt[:, -1] >= 1.5) & (srt[:, -1] - srt[:, -2] >= 0.7)
    lab = np.array(names, dtype=object)[best]
    lab[~ok] = "unassigned"
    return lab


def spearman(a, b):
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.nanstd(a[m]) == 0:
        return None
    return round(float(spearmanr(a[m], b[m]).statistic), 3)


def analyse(s, res, info):
    X = sps.csr_matrix(s.X)
    lab = groups(X, s.gene_names, s.species)
    ng = np.log(np.maximum(np.diff(X.indptr), 1))
    hm = res["hallmark"]
    out = {
        "gsm": s.gsm,
        "gse": s.gse,
        "n_cells": info["n_cells"],
        "timings": info["timings"],
        "median_genes": info["n_genes_detected_median"],
        "characteristics": s.meta.get("sample_characteristics", "")[:300],
    }
    gtab = {}
    for g in sorted(set(lab)):
        m = lab == g
        if m.sum() < 20:
            continue
        gtab[g] = {
            "n": int(m.sum()),
            **{k: round(float(np.nanmedian(hm.values[k][m])), 3) for k in SHOW},
        }
    out["groups"] = gtab
    names = [f.name for f in hm.features]
    raw = hm.extra["auc"]
    out["depth_spearman"] = {
        "raw_auc_median_abs": float(
            np.nanmedian(
                [
                    abs(x)
                    for x in (spearman(raw[:, j], ng) for j in range(len(names)))
                    if x is not None
                ]
            )
        ),
        "adjusted_median_abs": float(
            np.nanmedian(
                [abs(x) for x in (spearman(hm.values[n], ng) for n in names) if x is not None]
            )
        ),
        "adjusted_per_feature": {n: spearman(hm.values[n], ng) for n in SHOW},
    }
    cc = res["cell_cycle"]
    ph = cc.values["phase"]
    gi = {str(g).upper(): j for j, g in reversed(list(enumerate(s.gene_names)))}
    prolif = np.zeros(X.shape[0], bool)
    for m in ("MKI67", "TOP2A"):
        if m in gi:
            prolif |= X[:, gi[m]].toarray().ravel() > 0
    out["cell_cycle"] = {
        p: {
            "share": round(float((ph == i).mean()), 4),
            "mki67_or_top2a_detected": round(float(prolif[ph == i].mean()), 3)
            if (ph == i).any()
            else None,
        }
        for i, p in enumerate(("G1", "S", "G2M"))
    }
    out["cell_cycle"]["NA"] = round(float((ph == 65535).mean()), 4)
    hi = {}
    for k in SHOW:
        f = next(f for f in hm.features if f.name == k)
        v = hm.values[k]
        hi[k] = [round(float(np.nanmean(v >= t)), 4) for t in f.thresholds]
    out["share_at_thresholds"] = hi
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--sample", action="append", required=True, help="GSE:v2:GSM or GSE:<bundle path>:GSM"
    )
    a = ap.parse_args(argv)
    from features_run import run_sample
    from singlet.celllabels import io as A
    from singlet.features import pipeline as P

    out = Path(a.out)
    report = []
    for spec in a.sample:
        gse, rest = spec.split(":", 1)
        src, gsm = rest.rsplit(":", 1)
        t = time.perf_counter()
        s = A.read_v2(gse, gsm) if src == "v2" else A.read_legacy(src, gsm)
        read_s = time.perf_counter() - t
        info = run_sample(s, out, ("hallmark", "cell_cycle"), read_s)
        res = P.score_sample(s)  # again in memory for the diagnostics (scores are deterministic)
        r = analyse(s, res, info)
        report.append(r)
        print(json.dumps({k: r[k] for k in ("gsm", "n_cells", "timings")}), flush=True)
    (out / "pilot.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
