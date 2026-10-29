"""Atlas-wide per-sample comparison on every shared sample: cells called and median UMIs per cell, Singlet vs scBaseCount.

Usage: python figs_atlaswide.py COVERAGE_DIR CAMPAIGN_SAMPLES_CSV OUT_DIR
Inputs in COVERAGE_DIR: singlet_atlas_joined.parquet, sbc_joined.parquet, sbc_obs_summary.parquet, catalog_map.parquet,
bundle_bpc.tsv (bytes per cell per study, for the empty-bundle flag).
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COV, CAMP, OUT = sys.argv[1:4]
SINGLET, SBC, INK, INK2, LIGHT = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
a = pd.read_parquet(os.path.join(COV, "singlet_atlas_joined.parquet"))
A = a[a.status.isin(["DONE", "DONE_QC_WARN"])]
s = pd.read_parquet(os.path.join(COV, "sbc_joined.parquet"))
o = pd.read_parquet(os.path.join(COV, "sbc_obs_summary.parquet")).set_index("srx")
camp = pd.read_csv(CAMP, low_memory=False)
camp["ws"] = camp.wrong_strand / (camp.wrong_strand + camp.exon_hits)
cat = pd.read_parquet(os.path.join(COV, "catalog_map.parquet")).drop_duplicates("gsm_id").set_index("gsm_id")
bpc = pd.read_csv(os.path.join(COV, "bundle_bpc.tsv"), sep="\t", index_col=0).bpc

b = A.merge(s[["srx_accession", "tech_10x", "cell_prep", "obs_count"]], left_on="srx", right_on="srx_accession")
b = b.merge(camp[["gsm", "ws", "total_reads"]], left_on="gsm_id", right_on="gsm", how="left")
b["sra_reads"] = b.gsm_id.map(cat.read_count)
b["depth_frac"] = (b.total_reads / b.sra_reads).clip(upper=1)
b["sbc_cells"] = b.srx.map(o.n)
b["sbc_umi_med"] = b.srx.map(o.umi_med)
b["sbc_genes_med"] = b.srx.map(o.genes_med)
b["empty_bundle"] = b.gse_id.map(bpc) < 150
b["cell_ratio"] = b.n_cells / b.sbc_cells
b["umi_ratio"] = b.median_umis / b.sbc_umi_med
b["strand_ok"] = b.ws < 0.5
good = b[~b.empty_bundle & b.strand_ok & b.sbc_cells.notna() & (b.n_cells > 0) & b.tech_10x.isin(["3_prime_gex", "5_prime_gex", "multiome"])]

res = {"shared": len(b), "with_sbc_obs": int(b.sbc_cells.notna().sum()), "empty_bundle": int(b.empty_bundle.sum()),
       "wrong_strand_gt50": int((b.ws > 0.5).sum()), "good_gex": len(good),
       "cells_singlet_sum_good": float(good.n_cells.sum()), "cells_sbc_sum_good": float(good.sbc_cells.sum()),
       "cell_ratio_median": float(good.cell_ratio.median()),
       "cell_ratio_p10_p90": [float(good.cell_ratio.quantile(0.1)), float(good.cell_ratio.quantile(0.9))],
       "share_singlet_fewer_cells": float((good.cell_ratio < 1).mean()),
       "umi_ratio_median": float(good.umi_ratio.median()),
       "umi_ratio_p10_p90": [float(good.umi_ratio.quantile(0.1)), float(good.umi_ratio.quantile(0.9))],
       "depth_frac_median_good": float(good.depth_frac.median()),
       "umi_ratio_vs_depth_spearman": float(good[["umi_ratio", "depth_frac"]].corr("spearman").iloc[0, 1]),
       "umi_ratio_median_full_depth": float(good[good.depth_frac > 0.9].umi_ratio.median()),
       "n_full_depth": int((good.depth_frac > 0.9).sum()),
       "cell_ratio_median_full_depth": float(good[good.depth_frac > 0.9].cell_ratio.median()),
       "median_umi_singlet": float(good.median_umis.median()), "median_umi_sbc": float(good.sbc_umi_med.median()),
       "median_genes_singlet": float(good.median_genes.median()), "median_genes_sbc": float(good.sbc_genes_med.median())}
for lab in ("3_prime_gex", "5_prime_gex"):
    g = good[good.tech_10x == lab]
    res[f"{lab}_n"] = len(g)
    res[f"{lab}_cell_ratio_median"] = float(g.cell_ratio.median())
    res[f"{lab}_umi_ratio_median"] = float(g.umi_ratio.median())
json.dump(res, open(os.path.join(OUT, "atlaswide.json"), "w"), indent=1)
print(json.dumps(res, indent=1))

# ---- Fig 9: cells per sample and median UMI per cell, all shared GEX samples
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
ax = axs[0]
ax.scatter(good.sbc_cells, good.n_cells, s=2, alpha=0.25, color=SINGLET, linewidths=0, rasterized=True)
lim = [30, 1e5]
ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("scBaseCount cells (EmptyDrops_CR)")
ax.set_ylabel("Singlet cells (knee filter)")
ax.set_title(f"Cells per sample (n={len(good):,})\nSinglet = {res['cell_ratio_median']:.2f} x scBaseCount (median)",
             loc="left", color=INK, fontsize=9.5)
ax = axs[1]
bins = [0, 0.1, 0.2, 0.35, 0.5, 0.75, 0.9, 1.01]
good = good.assign(dbin=pd.cut(good.depth_frac, bins))
gg = good.dropna(subset=["dbin"]).groupby("dbin", observed=True).umi_ratio
med = gg.median()
q1, q3 = gg.quantile(0.25), gg.quantile(0.75)
n = gg.size()
xs = np.arange(len(med))
ax.vlines(xs, q1, q3, color="#9ec3ef", lw=6)
ax.scatter(xs, med, color=SINGLET, s=30, zorder=3)
ax.axhline(1, color=INK2, lw=0.8, ls="--")
ax.set_xticks(xs, [f"{iv.left:.0%}-{min(iv.right, 1):.0%}\nn={k:,}" for iv, k in zip(med.index, n)], fontsize=7)
ax.set_xlabel("share of the sample's SRA reads Singlet processed")
ax.set_ylabel("median UMIs/cell, Singlet / scBaseCount")
ax.set_ylim(0, 1.4)
ax.set_title("Per-cell depth tracks the read cap", loc="left", color=INK, fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig9_atlaswide.png"), dpi=200)
plt.close(fig)
