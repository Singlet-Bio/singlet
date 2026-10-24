#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which per-cell statistic to write: raw rank AUC vs null-calibrated versions, on pilot samples.

    python bench/features/contrasts.py BUNDLE_DIR > results.json

For each statistic: median |Spearman| with log detected genes over the 50 Hallmark sets (depth confounding, lower is
better) and the AUROC of expected contrasts (marker-defined groups from pilot.py; COVID-19 vs healthy donors of GSE293707).
Bundles in BUNDLE_DIR: GSE138867, GSE293707, GSE294109, GSE178941 (data.singlet.bio/data/<GSE>/<GSE>.singlet).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot import groups  # noqa: E402

IMMUNE = ["T", "NK", "myeloid", "B", "plasma"]
# (name, (gse, gsm), program, group a, group b, other sample for b or None); groups: a marker group, "cycling" or "G1"
CONTRASTS = [
    (
        "heme: erythroid vs other (PBMC)",
        ("GSE138867", "GSM4120733"),
        "heme_metabolism",
        "erythroid",
        "!erythroid",
        None,
    ),
    (
        "heme: erythroid vs other (COVID)",
        ("GSE293707", "GSM8888965"),
        "heme_metabolism",
        "erythroid",
        "!erythroid",
        None,
    ),
    ("e2f: S+G2M vs G1", ("GSE178941", "GSM5401596"), "e2f_targets", "cycling", "G1", None),
    (
        "emt: endothelial vs immune",
        ("GSE294109", "GSM8898802"),
        "epithelial_mesenchymal_transition",
        "endothelial",
        "immune",
        None,
    ),
    (
        "ifna: NK covid(65) vs healthy(63)",
        ("GSE293707", "GSM8888965"),
        "interferon_alpha_response",
        "NK",
        "NK",
        ("GSE293707", "GSM8888963"),
    ),
    (
        "ifna: NK covid(64) vs healthy(63)",
        ("GSE293707", "GSM8888964"),
        "interferon_alpha_response",
        "NK",
        "NK",
        ("GSE293707", "GSM8888963"),
    ),
    (
        "ifna: myeloid covid(64) vs healthy(63)",
        ("GSE293707", "GSM8888964"),
        "interferon_alpha_response",
        "myeloid",
        "myeloid",
        ("GSE293707", "GSM8888963"),
    ),
    (
        "ifng: myeloid vs T (PBMC)",
        ("GSE138867", "GSM4120733"),
        "interferon_gamma_response",
        "myeloid",
        "T",
        None,
    ),
    (
        "oxphos: myeloid vs T (PBMC)",
        ("GSE138867", "GSM4120733"),
        "oxidative_phosphorylation",
        "myeloid",
        "T",
        None,
    ),
    (
        "inflam: myeloid vs T (PBMC)",
        ("GSE138867", "GSM4120733"),
        "inflammatory_response",
        "myeloid",
        "T",
        None,
    ),
]


def auroc(a, b):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) <= 10 or len(b) <= 10:
        return None
    return round(float(mannwhitneyu(a, b).statistic / (len(a) * len(b))), 3)


def select(lab, phase, g):
    if g == "cycling":
        return (phase == 1) | (phase == 2)
    if g == "G1":
        return phase == 0
    if g == "immune":
        return np.isin(lab, IMMUNE)
    if g.startswith("!"):
        return (lab != g[1:]) & (lab != "unassigned")
    return lab == g


def main(argv=None):
    from singlet.celllabels import io as A
    from singlet.features import pipeline as P

    bundles = Path((argv or sys.argv[1:])[0])
    cache = {}

    def get(key):
        if key not in cache:
            gse, gsm = key
            s = A.read_legacy(bundles / f"{gse}.singlet", gsm)
            res = P.score_sample(s)
            h = res["hallmark"]
            names = [f.name for f in h.features]
            auc, nm, z = h.extra["auc"], h.extra["null_mean"], h.extra["z"]
            stats = {
                "score": np.column_stack([h.values[n] for n in names]),
                "z": z,
                "log2_fold_over_null": np.log2((auc + 1e-3) / (nm + 1e-3)),
                "adjusted": np.clip((auc - nm) / np.maximum(1 - nm, 1e-6), 0, 1),
                "raw_auc": auc,
            }
            X = s.X.tocsr()
            lab = groups(X, s.gene_names, s.species)
            ng = np.log(np.diff(X.indptr) + 1)
            cache[key] = (stats, names, lab, ng, res["cell_cycle"].values["phase"])
        return cache[key]

    out = {}
    for key in {c[1] for c in CONTRASTS}:
        stats, names, lab, ng, _ = get(key)
        for k, v in stats.items():
            rho = [
                abs(spearmanr(v[:, j], ng, nan_policy="omit").statistic) for j in range(v.shape[1])
            ]
            out.setdefault(k, {"depth_abs_spearman": {}, "auroc": {}})["depth_abs_spearman"][
                "/".join(key)
            ] = round(float(np.nanmedian(rho)), 3)
    for name, key, prog, ga, gb, other in CONTRASTS:
        stats, names, lab, _, ph = get(key)
        j = names.index(prog)
        stats_b, _, lab_b, _, ph_b = get(other) if other else (stats, names, lab, None, ph)
        for k in stats:
            a = stats[k][:, j][select(lab, ph, ga)]
            b = stats_b[k][:, j][select(lab_b, ph_b, gb)]
            out[k]["auroc"][name] = auroc(a, b)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
