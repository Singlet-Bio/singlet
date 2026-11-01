# SPDX-License-Identifier: MIT
"""Kang b2 ctrl genotype check of the 2.2 cell calls (dev lane only; the b1 lanes are held out).

    kang_genotype.py CALLS_TSV TRUTH_TSV_GZ TRUTH_JSON

CALLS_TSV is the cv2_stored output (barcode, is_cell, called_lowrna, umi_total, ...). TRUTH_* come from
bench/cellcall-independent-truth (genotruth.py): per barcode llr = log-likelihood of one donor (or a donor pair, with ambient
contamination) minus that of the ambient donor mixture, snp_umis = UMIs at the kept SNP sites, and the calibration table (per
informative depth m: t_cell = LLR cut at which ambient (empty) barcodes pass 0.1%, pos_cell = fraction of real cells passing).

Per group and UMI bin, a barcode "hits" when llr >= t_cell(m) at its informative depth m (largest calibration m <= snp_umis).
Real-cell fraction = sum(hit - neg_cell) / sum(pos_cell - neg_cell) over barcodes with pos_cell >= 0.05 (m >= 25): the share
of real cells that explains the hits (1 = every barcode is a real cell, 0 = ambient only), a ratio estimator that gives weak
barcodes little weight. Pools: is_cell, called_lowrna, and empties = no caller (knee, called_v2, edcr, cellbender,
cell ranger) calls them, at the same UMI bins. Bootstrap 95% CI over barcodes.
"""
import json
import sys

import numpy as np
import pandas as pd

calls = pd.read_csv(sys.argv[1], sep="\t", dtype={"barcode": str}).set_index("barcode")
tr = pd.read_csv(sys.argv[2], sep="\t", dtype={"barcode": str}).set_index("barcode")
cal = pd.DataFrame(json.load(open(sys.argv[3]))["calibration"]).sort_values("m")
ms = cal["m"].to_numpy()

d = tr.join(calls[["is_cell", "called_lowrna", "umi_total"]].rename(columns={"umi_total": "umi_new"}), how="left")
d[["is_cell", "called_lowrna"]] = d[["is_cell", "called_lowrna"]].fillna(0).astype(int)
callers = [c for c in ["knee", "called_v2", "edcr", "cellbender", "cellranger"] if c in d]
d["nocaller"] = (d[callers].fillna(0).sum(axis=1) == 0) & (d.is_cell == 0) & (d.called_lowrna == 0)

k = np.searchsorted(ms, d["snp_umis"].fillna(0).to_numpy(), side="right") - 1
ok = k >= 0
k = np.clip(k, 0, len(ms) - 1)
d["m"] = ms[k]
d["decidable"] = ok & (cal["pos_cell"].to_numpy()[k] >= 0.05)   # m >= 25: weak but unbiased in the ratio estimator
d["hit"] = (d["llr"] >= cal["t_cell"].to_numpy()[k]).astype(float)
d["single"] = (d["model"] == "single")
neg, pos = cal["neg_cell"].to_numpy()[k], cal["pos_cell"].to_numpy()[k]
d["num"], d["den"] = d["hit"] - neg, pos - neg

rng = np.random.default_rng(1)


def est(g):
    return g["num"].sum() / g["den"].sum() if len(g) and g["den"].sum() > 0 else float("nan")


def ci(g):
    if len(g) < 5:
        return float("nan"), float("nan")
    num, den = g["num"].to_numpy(), g["den"].to_numpy()
    b = []
    for _ in range(2000):
        i = rng.integers(0, len(g), len(g))
        b.append(num[i].sum() / max(den[i].sum(), 1e-9))
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


bins = [(50, 500), (500, 1000), (1000, 3000), (3000, 1e9)]
print(f"Kang b2 ctrl: {len(d)} barcodes in the truth universe; {int(d.is_cell.sum())} is_cell, {int(d.called_lowrna.sum())} called_lowrna")
print("group           UMI bin     n  decidable  hit-rate  real-fraction [95% CI]  single-donor among hits")
for name, mask in [("is_cell", d.is_cell == 1), ("called_lowrna", d.called_lowrna == 1), ("empties", d.nocaller)]:
    for lo, hi in bins:
        g = d[mask & (d.umi_total >= lo) & (d.umi_total < hi)]
        gd = g[g.decidable]
        if len(g) == 0:
            continue
        r = est(gd)
        lo_ci, hi_ci = ci(gd)
        h = gd[gd.hit == 1]
        sg = h["single"].mean() if len(h) else float("nan")
        print(f"{name:14s} {lo:>5.0f}-{hi if hi < 1e8 else 'inf':<6} {len(g):5d} {len(gd):9d}  {gd['hit'].mean() if len(gd) else float('nan'):8.3f}  {r:6.3f} [{lo_ci:.3f}, {hi_ci:.3f}]  {sg:6.3f}")
