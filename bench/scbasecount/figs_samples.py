"""Figures and the per-sample table for the 20 matched samples (inputs: compare.py outputs, pulled back).

Usage: python figs_samples.py COMPARE_DIR PICKS_TSV OUT_DIR
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CMP, PICKS, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
SINGLET, SBC, AUTHOR, INK, INK2, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": LIGHT, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb"})
picks = pd.read_csv(PICKS, sep="\t").set_index("gsm")
CAT = {"h_3v3": "human 3' v3", "h_3v2": "human 3' v2", "m_3v3": "mouse 3' v3", "h_5p_ok": "human 5'",
       "h_5p_wrongstrand": "5', Singlet wrong strand", "h_3p_label_5p_sbc": "5' labelled 3' by Singlet",
       "nuclei_h": "human nuclei", "nuclei_m": "mouse nuclei", "multiome": "Multiome GEX", "citeseq": "CITE-seq GEX",
       "zebrafish": "zebrafish", "macaque": "macaque", "rat": "rat", "vdj_lib": "V(D)J library",
       "m_3p_wrongstrand": "mouse, Singlet wrong strand", "h_3p_wrongstrand_sbc3p": "human 3', Singlet wrong strand"}


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


rows = []
for f in sorted(glob.glob(os.path.join(CMP, "GSM*.json"))):
    d = json.load(open(f))
    gsm = d["gsm"]
    p = picks.loc[gsm] if gsm in picks.index else {}
    rows.append({
        "gsm": gsm, "gse": d["gse"], "srx": d["srx"], "category": CAT.get(p.get("cat"), p.get("cat")),
        "organism": d["organism_dir"].replace("_", " "),
        "singlet_protocol": g(d, "singlet", "protocol"), "sbc_label": p.get("tech"), "prep": p.get("prep"),
        "wrong_strand": g(d, "singlet", "wrong_strand_frac"),
        "singlet_reads": g(d, "singlet", "total_reads"),
        "cells_singlet": g(d, "calls", "singlet"), "cells_sbc": g(d, "calls", "sbc"), "cells_author": g(d, "calls", "author"),
        "jacc_singlet_sbc": g(d, "calls", "jaccard_singlet_sbc"),
        "author_recovered_singlet": g(d, "calls", "author_recovered_by_singlet"),
        "author_recovered_sbc": g(d, "calls", "author_recovered_by_sbc"),
        "singlet_in_author": g(d, "calls", "singlet_in_author"), "sbc_in_author": g(d, "calls", "sbc_in_author"),
        "umi_ratio_total_vs_ex50": g(d, "totals", "singlet_total_vs_sbc_genefull_ex50", "median_ratio"),
        "umi_ratio_exon_vs_gene": g(d, "totals", "singlet_exon_vs_sbc_gene", "median_ratio"),
        "umi_logr_total_vs_ex50": g(d, "totals", "singlet_total_vs_sbc_genefull_ex50", "log_pearson"),
        "sbc_gene_vs_author": g(d, "totals", "sbc_gene_vs_author", "median_ratio"),
        "singlet_exon_vs_author": g(d, "totals", "singlet_exon_vs_author", "median_ratio"),
        "median_umi_singlet": g(d, "totals", "median_umi", "singlet_total"),
        "median_umi_sbc": g(d, "totals", "median_umi", "sbc_genefull_ex50"),
        "corr": g(d, "gene_level", "median_cell_corr_total_vs_ex50"),
        "corr_depth_matched": g(d, "gene_level", "median_cell_corr_total_vs_ex50_depth_matched"),
        "corr_sbc_self": g(d, "gene_level", "median_cell_corr_sbc_self_at_singlet_depth"),
        "pb_rho": g(d, "gene_level", "pseudobulk_spearman_total_vs_ex50"),
        "corr_exon_vs_author_dm": g(d, "gene_level", "median_cell_corr_singlet_exon_vs_author_depth_matched"),
        "unspliced_singlet": g(d, "unspliced", "singlet_median"), "unspliced_sbc": g(d, "unspliced", "sbc_median"),
        "sbc_h5ad_mb": sum((g(d, "sbc", "h5ad_bytes") or {}).values()) / 1e6 if g(d, "sbc", "h5ad_bytes") else None,
        "author_source": g(d, "author", "source"), "errors": "; ".join(d.get("errors") or []),
    })
T = pd.DataFrame(rows)
order = list(CAT.values())
T["_o"] = T.category.map({c: i for i, c in enumerate(order)})
T = T.sort_values(["_o", "gsm"]).drop(columns="_o").reset_index(drop=True)
T.to_csv(os.path.join(OUT, "samples.tsv"), sep="\t", index=False)
T["empty_bundle"] = T.cells_singlet.fillna(0).eq(0) | T.umi_ratio_total_vs_ex50.isna() & T["corr"].isna()
T["label"] = T.category + "  " + T.gsm + np.where(T.empty_bundle, "  (empty Singlet matrix)", "")

# ---- Fig 6: per-cell UMI ratio and depth-matched gene correlation per sample
fig, axs = plt.subplots(1, 2, figsize=(9.5, 6.2), sharey=True, gridspec_kw={"width_ratios": [1.1, 1]})
y = np.arange(len(T))[::-1]
ax = axs[0]
ax.scatter(T.umi_ratio_total_vs_ex50, y, color=SINGLET, s=36, zorder=3, label="exon + intron vs GeneFull_Ex50pAS")
ax.scatter(T.umi_ratio_exon_vs_gene, y, color="#9ec3ef", s=22, zorder=3, label="exon vs Gene")
ax.axvline(1, color=INK2, lw=1, ls="--")
ax.set_xscale("log")
ax.set_xlim(0.002, 3)
ax.set_xticks([0.003, 0.01, 0.03, 0.1, 0.3, 1], ["0.003", "0.01", "0.03", "0.1", "0.3", "1"])
ax.set_yticks(y, T.label, fontsize=8)
ax.set_xlabel("Singlet / scBaseCount, median per-cell UMIs")
ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, -0.2))
ax.set_title("Counts per cell (shared cells)", loc="left", color=INK, fontsize=10)
ax = axs[1]
ax.scatter(T["corr"], y, color=LIGHT, s=26, zorder=3, label="as published")
ax.scatter(T.corr_depth_matched, y, color=SINGLET, s=36, zorder=4, label="scBaseCount thinned to Singlet's depth")
ax.scatter(T.corr_sbc_self, y, marker="|", color=SBC, s=120, zorder=5, label="scBaseCount vs itself, same thinning")
ax.set_xlim(0, 1)
ax.set_xlabel("median per-cell gene correlation (log1p)")
ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, -0.2))
ax.set_title("Same cells, same genes", loc="left", color=INK, fontsize=10)
fig.suptitle("Singlet vs scBaseCount on 20 shared samples", x=0.01, ha="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_samples_counts_corr.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Fig 7: cell calls per sample (Singlet, scBaseCount, authors)
fig, ax = plt.subplots(figsize=(9.5, 6.2))
h = 0.26
ax.barh(y + h, T.cells_singlet, h, color=SINGLET, label="Singlet (knee filter)")
ax.barh(y, T.cells_sbc, h, color=SBC, label="scBaseCount (EmptyDrops_CR)")
ax.barh(y - h, T.cells_author.fillna(0), h, color=AUTHOR, label="authors' deposited matrix (Cell Ranger)")
for yi, (a_, r) in zip(y, T.iterrows()):
    if pd.notna(r.cells_author) and r.cells_author:
        ax.text(max(r.cells_singlet, r.cells_sbc, r.cells_author) * 1.03, yi,
                f"authors' cells found: Singlet {r.author_recovered_singlet:.0%}, scBaseCount {r.author_recovered_sbc:.0%}",
                va="center", fontsize=7, color=INK2)
ax.set_yticks(y, T.label, fontsize=8)
ax.set_xlabel("cells called")
ax.set_xlim(0, T[["cells_singlet", "cells_sbc", "cells_author"]].max().max() * 1.9)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=8)
ax.set_title("Cells called per sample", loc="left", color=INK, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_cell_calls.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# ---- Fig 8: per-cell totals scatter, four example samples
ex = [c for c in ["human 3' v3", "mouse 3' v3", "mouse nuclei", "5' labelled 3' by Singlet"]
      if c in set(T[T.umi_ratio_total_vs_ex50.notna()].category)]
fig, axs = plt.subplots(1, len(ex), figsize=(3.0 * len(ex), 3.2))
for ax, cat in zip(np.atleast_1d(axs), ex):
    gsm = T[(T.category == cat) & T.umi_ratio_total_vs_ex50.notna()].gsm.iloc[0]
    C = pd.read_parquet(os.path.join(CMP, f"{gsm}.cells.parquet"))
    sc = C[C.singlet_in_matrix & C.sbc_called]
    x_, y_ = sc.sbc_genefull_ex50, sc.singlet_exon + sc.singlet_intron
    ax.scatter(x_, y_, s=3, color=SINGLET, alpha=0.35, linewidths=0)
    lim = [max(1, min(x_.min(), y_.min())), max(x_.max(), y_.max()) * 1.2]
    ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("scBaseCount UMIs / cell")
    r_ = float(np.median(y_ / x_.clip(lower=1)))
    ax.set_title(f"{cat}, {gsm}\nSinglet = {r_:.2f} x scBaseCount", loc="left", fontsize=8.5, color=INK)
np.atleast_1d(axs)[0].set_ylabel("Singlet UMIs / cell")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig8_cell_totals_scatter.png"), dpi=200)
plt.close(fig)

with open(os.path.join(OUT, "samples_table.md"), "w", encoding="utf-8") as fh:
    cols = ["category", "gsm", "singlet_protocol", "sbc_label", "wrong_strand", "singlet_reads", "cells_singlet", "cells_sbc",
            "cells_author", "jacc_singlet_sbc", "author_recovered_singlet", "author_recovered_sbc", "umi_ratio_total_vs_ex50",
            "sbc_gene_vs_author", "corr", "corr_depth_matched", "corr_sbc_self", "pb_rho", "unspliced_singlet", "unspliced_sbc"]
    fh.write(T[cols].to_markdown(index=False, floatfmt=".3g"))
print(T.drop(columns=["label"]).to_string())
