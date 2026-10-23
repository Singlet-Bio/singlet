"""Aggregate results/<dataset>/metrics.json into figures (PNG) and markdown tables.

    python -I make_report.py <PB> <out_dir>

Figures use the reference palette's categorical slots in a fixed tool order, so a tool
keeps its colour in every figure; text stays in neutral ink.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TOOL_COLOR = {  # fixed slot per tool (never by rank)
    "singlet-fixed": "#2a78d6",
    "singlet-baseline": "#eb6834",
    "singlet-baseline-t4": "#eb6834",
    "starsolo": "#1baf7a",
    "alevinfry": "#eda100",
    "cellranger_published": "#e87ba4",
}
LABEL_COLOR = {"spliced": "#2a78d6", "unspliced": "#eb6834", "ambiguous": "#1baf7a"}
NICE = {
    "singlet-fixed": "Singlet (fixed)", "singlet-baseline": "Singlet (atlas engine, 16 workers)",
    "singlet-baseline-t4": "Singlet (atlas engine)", "singlet-fixed-t4": "Singlet (fixed, 4 workers)",
    "starsolo_GeneFull": "STARsolo GeneFull", "starsolo_Gene": "STARsolo Gene",
    "starsolo_Velocyto": "STARsolo Velocyto", "starsolo": "STARsolo", "alevinfry": "alevin-fry (USA)",
    "cellranger_published": "Cell Ranger (10x published)",
}
DS_ORDER = ["pbmc1k_v3", "pbmc1k_v2", "pbmc5p_v1", "pbmc1k_cite_v3", "neuron1k_v3", "nuclei2k_v2", "hgmm1k_v3"]
DS_NICE = {
    "pbmc1k_v3": "PBMC 3' v3", "pbmc1k_v2": "PBMC 3' v2", "pbmc5p_v1": "PBMC 5' v1",
    "pbmc1k_cite_v3": "PBMC CITE 3' v3", "neuron1k_v3": "Mouse neurons 3' v3",
    "nuclei2k_v2": "Mouse brain nuclei 3' v2", "hgmm1k_v3": "Human+mouse barnyard 3' v3",
}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fmt_pct(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}%"


def fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(out)


def main():
    pb, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    M = {}
    for p in sorted((pb / "results").glob("*/metrics.json")):
        M[p.parent.name] = json.loads(p.read_text())
    ds = [d for d in DS_ORDER if d in M]
    bams = sorted(d for d in M if d.startswith("bam_"))
    T = {}

    # ── Table 1: double count ──
    rows = []
    for d in ds + bams:
        dc = M[d].get("double_count")
        if not dc:
            continue
        rows.append({
            "dataset": DS_NICE.get(d, d), "cells": dc["n_cells"],
            "per-cell overcount, median": fmt_pct(dc["cell_overcount_median"]),
            "per-cell overcount, p10-p90": f"{fmt_pct(dc['cell_overcount_p10_p90'][0])} to {fmt_pct(dc['cell_overcount_p10_p90'][1])}",
            "total overcount": fmt_pct(dc["total_overcount"]),
            "total overcount, 16 workers": fmt_pct(M[d].get("double_count_16w", {}).get("total_overcount")),
            "exon / intron matrix ratio old:new": f"{fmt(dc['exon_old_vs_new_ratio'], 3)} / {fmt(dc['intron_old_vs_new_ratio'], 3)}",
            "same-gene exon+intron doubles": fmt_pct(dc.get("legacy_double_frac")),
            "cross-worker multi-mapper molecules": fmt_pct(dc.get("old_cross_worker_frac")),
            "gene overcount, median (>=50 UMI)": fmt_pct(dc["gene_overcount_median_expressed"]),
            "gene overcount, p99": fmt_pct(dc["gene_overcount_p99_expressed"]),
            "cells called old / new": f"{dc['called_cells_old']} / {dc['called_cells_new']}",
        })
    T["double_count"] = pd.DataFrame(rows)

    rows = []
    for d in ds:
        for eng, v in M[d].get("worker_invariance", {}).items():
            rows.append({"dataset": DS_NICE[d], "engine": NICE.get(eng, eng),
                         "total UMIs, 16 / 4 workers": fmt(v["total_16w_over_4w"], 4),
                         "matrix entries that differ": v["entries_differing"],
                         "sum of |differences|": int(v["abs_diff_total"])})
    T["worker_invariance"] = pd.DataFrame(rows)

    # Figure 1: per-cell overcount distributions
    fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=160)
    data, labels = [], []
    for d in ds + bams:
        f = pb / "results" / d / "double_count_cells.csv"
        if f.exists():
            data.append(100 * pd.read_csv(f)["overcount"].values)
            labels.append(DS_NICE.get(d, d.replace("bam_", "")))
    if data:
        bp = ax.boxplot(data, vert=True, widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2, "lw": 0.8},
                        capprops={"color": INK2, "lw": 0.8})
        for b in bp["boxes"]:
            b.set(facecolor=TOOL_COLOR["singlet-baseline"], edgecolor=SURF, lw=2, alpha=0.9)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
        ax.set_ylabel("per-cell overcount of exon + intron (%)")
        ax.set_title("Atlas engine vs fixed engine, same reads, same cells", fontsize=9, color=INK, loc="left")
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "fig1_double_count.png", facecolor=SURF)
    plt.close(fig)

    # ── Table 2: spliced / unspliced / ambiguous split ──
    rows = []
    for d in ds:
        for t, s in M[d].get("suf_split", {}).items():
            rows.append({"dataset": DS_NICE[d], "tool": NICE.get(t, t), "spliced": fmt_pct(s.get("spliced")),
                         "unspliced": fmt_pct(s.get("unspliced")), "ambiguous": fmt_pct(s.get("ambiguous"))})
    T["suf"] = pd.DataFrame(rows)

    # Figure 2: S/U/A stacked bars
    tools_suf = ["singlet-baseline-t4", "singlet-fixed", "starsolo_Velocyto", "alevinfry"]
    if ds:
        fig, axes = plt.subplots(1, len(ds), figsize=(1.6 * len(ds) + 1, 3.2), dpi=160, sharey=True)
        axes = np.atleast_1d(axes)
        for ax, d in zip(axes, ds):
            ss = M[d].get("suf_split", {})
            x = 0
            ticks = []
            for t in tools_suf:
                if t not in ss:
                    continue
                bottom = 0
                for k in ("spliced", "unspliced", "ambiguous"):
                    v = ss[t].get(k) or 0
                    if t.startswith("singlet-baseline") and k == "spliced":
                        v = 1 - (ss[t].get("unspliced") or 0)
                    ax.bar(x, 100 * v, bottom=100 * bottom, width=0.7, color=LABEL_COLOR[k],
                           edgecolor=SURF, linewidth=2)
                    bottom += v
                ticks.append((x, NICE.get(t, t).replace(" (", "\n(")))
                x += 1
            ax.set_xticks([p for p, _ in ticks], [lab for _, lab in ticks], rotation=90, fontsize=6)
            ax.set_title(DS_NICE[d], fontsize=7, color=INK)
            style(ax)
        axes[0].set_ylabel("share of molecules in shared cells (%)")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in LABEL_COLOR.values()]
        fig.legend(handles, list(LABEL_COLOR), loc="upper right", fontsize=7, frameon=False, ncol=3)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(out / "fig2_suf_split.png", facecolor=SURF)
        plt.close(fig)

    # ── Table 3: count concordance ──
    rows = []
    for d in ds:
        for k, c in M[d].get("concordance", {}).items():
            if "error" in c:
                continue
            rows.append({"dataset": DS_NICE[d], "comparison": k, "what": c["what"], "cells": c["n_cells"],
                         "per-cell total ratio (median)": fmt(c["cell_total_ratio_median"]),
                         "per-cell total r (log)": fmt(c["cell_total_pearson_log"]),
                         "per-cell gene r (log, median)": fmt(c["per_cell_gene_pearson_log_median"]),
                         "pseudobulk rho": fmt(c["gene_pseudobulk_spearman"]),
                         "genes/cell ratio": fmt(c["genes_detected_ratio_median"])})
    T["concordance"] = pd.DataFrame(rows)

    # Figure 3: per-cell totals, Singlet fixed & baseline vs STARsolo GeneFull (all datasets)
    if ds:
        n = len(ds)
        fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6), dpi=160)
        axes = np.atleast_1d(axes)
        for ax, d in zip(axes, ds):
            f = pb / "results" / d / "cell_totals.csv"
            if not f.exists():
                continue
            ct = pd.read_csv(f, index_col=0)
            if "starsolo_GeneFull" not in ct:
                continue
            ref = ct["starsolo_GeneFull"]
            for t in ("singlet-baseline-t4", "singlet-fixed", "alevinfry"):
                if t in ct:
                    ax.scatter(ref, ct[t], s=3, alpha=0.6, color=TOOL_COLOR[t], linewidths=0, label=NICE[t])
            lim = [max(ref.min(), 1), ref.max() * 1.2]
            ax.plot(lim, lim, color=INK2, lw=0.8, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(DS_NICE[d], fontsize=7, color=INK)
            ax.set_xlabel("STARsolo GeneFull UMIs/cell", fontsize=7)
            style(ax)
        axes[0].set_ylabel("tool UMIs/cell", fontsize=7)
        h, lab = axes[0].get_legend_handles_labels()
        fig.legend(h, lab, loc="upper right", fontsize=7, frameon=False, ncol=3, markerscale=3)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(out / "fig3_cell_totals.png", facecolor=SURF)
        plt.close(fig)

    # ── Table 4: cell calls ──
    rows = []
    for d in ds:
        cc = M[d].get("cell_calls", {})
        ov = M[d].get("cell_call_overlap", {})

        def j(a, b):
            k = f"{a}|{b}" if a < b else f"{b}|{a}"
            return fmt(ov[k]["jaccard"], 3) if k in ov else "-"
        rows.append({"dataset": DS_NICE[d],
                     "Singlet (fixed)": cc.get("singlet-fixed", "-"), "Singlet (atlas)": cc.get("singlet-baseline-t4", "-"),
                     "STARsolo EmptyDrops_CR": cc.get("starsolo_GeneFull", "-"),
                     "knee CR2.2 on STARsolo GeneFull": cc.get("starsolo_GeneFull__knee_cr22", "-"),
                     "knee CR2.2 on Singlet (fixed)": cc.get("singlet-fixed__knee_cr22", "-"),
                     "Cell Ranger (published)": cc.get("cellranger_published", "-"),
                     "J(Singlet fixed, STARsolo)": j("singlet-fixed", "starsolo_GeneFull"),
                     "J(Singlet fixed, Cell Ranger)": j("singlet-fixed", "cellranger_published"),
                     "J(STARsolo, Cell Ranger)": j("cellranger_published", "starsolo_GeneFull"),
                     "J(knee CR2.2, Cell Ranger)": j("cellranger_published", "starsolo_GeneFull__knee_cr22"),
                     "J(Singlet atlas, fixed)": j("singlet-baseline-t4", "singlet-fixed")})
    T["cell_calls"] = pd.DataFrame(rows)

    # ── Table 5: clustering / markers / ground truth ──
    rows = []
    for d in ds:
        a = M[d].get("ari", {})
        mk = next((v for k, v in M[d].items() if k.startswith("marker_jaccard_vs_")), {})
        adt = M[d].get("ari_vs_adt", {})
        for t in ("singlet-fixed", "singlet-baseline-t4", "alevinfry", "starsolo_Gene", "cellranger_published"):
            k = f"{t}|starsolo_GeneFull" if t < "starsolo_GeneFull" else f"starsolo_GeneFull|{t}"
            if k not in a and t not in adt:
                continue
            rows.append({"dataset": DS_NICE[d], "tool": NICE.get(t, t),
                         "ARI vs STARsolo GeneFull clusters": fmt(a.get(k)),
                         "top-50 marker Jaccard (same labels)": fmt(mk.get(t)),
                         "ARI vs ADT clusters": fmt(adt.get(t)) if adt else "-"})
        if adt and "starsolo_GeneFull" in adt:
            rows.append({"dataset": DS_NICE[d], "tool": NICE["starsolo_GeneFull"],
                         "ARI vs STARsolo GeneFull clusters": "1.000", "top-50 marker Jaccard (same labels)": "1.000",
                         "ARI vs ADT clusters": fmt(adt["starsolo_GeneFull"])})
    T["clusters"] = pd.DataFrame(rows)

    # ── Table 6: barnyard ──
    if "hgmm1k_v3" in M and "barnyard" in M["hgmm1k_v3"]:
        rows = []
        for t, b in M["hgmm1k_v3"]["barnyard"].items():
            rows.append({"tool": NICE.get(t, t), "cells": b["n_cells"], "human": b["human"], "mouse": b["mouse"],
                         "mixed (doublet)": b["doublet"], "doublet rate": fmt_pct(b["doublet_rate"]),
                         "cross-species UMIs in singlets (median)": fmt_pct(b["cross_species_frac_median"], 2),
                         "(mean)": fmt_pct(b["cross_species_frac_mean"], 2)})
        T["barnyard"] = pd.DataFrame(rows)

    # ── Table 7: resources ──
    rows = []
    for d in ds:
        for t, r in M[d].get("resources", {}).items():
            if not r:
                continue
            rows.append({"dataset": DS_NICE[d], "tool": NICE.get(t, t),
                         "wall (min)": fmt(r.get("wall_s", np.nan) / 60, 1),
                         "+ .1fq encode (min)": fmt(r["encode_wall_s"] / 60, 1) if r.get("encode_wall_s") else "-",
                         "CPU (h)": fmt(r.get("cpu_s", np.nan) / 3600, 2),
                         "peak RSS (GB)": fmt(r.get("max_rss_gb"), 1)})
    T["resources"] = pd.DataFrame(rows)

    md = []
    for k, df in T.items():
        md.append(f"## {k}\n\n" + (md_table(df) if len(df) else "(no data)") + "\n")
    (out / "tables.md").write_text("\n".join(md))
    (out / "all_metrics.json").write_text(json.dumps(M, indent=1, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
