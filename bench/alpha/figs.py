"""Static figures (PNG + SVG) for the Research Alpha benchmarks.

    python -P figs.py OUT_DIR      # reads alpha_samples.csv, alpha_census.json, alpha-engine-campaign.json when present
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = Path(sys.argv[1])
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # categorical slots 1-4, fixed order
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "legend.frameon": False})
ORDER = [f"{s}|{c}|{z}" for s in ("human", "mouse") for c in ("3p_v2", "3p_v3", "5p", "nuclei") for z in ("small", "large")]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=130)
    fig.savefig(OUT / f"{name}.svg")
    plt.close(fig)


def strata_strip(S, cols, names, fname, xlabel, xlim=None, ref=None):
    S = S[S.status == "scored"]
    cats = [c for c in ORDER if (S.stratum == c).any()]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 0.32 * len(cats) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    rng = np.random.default_rng(0)
    for ax, col, nm in zip(axes, cols, names):
        for i, c in enumerate(cats):
            v = S.loc[S.stratum == c, col].dropna().values
            color = SER[0] if c.startswith("human") else SER[1]
            ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=color, alpha=0.75, linewidths=0)
            if len(v):
                ax.plot([np.median(v)] * 2, [i - 0.32, i + 0.32], color=INK, lw=2)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.set_title(nm, color=INK, fontsize=10, loc="left")
        ax.set_xlabel(xlabel)
        if xlim:
            ax.set_xlim(*xlim)
    axes[0].set_yticks(range(len(cats)))
    axes[0].set_yticklabels([c.replace("|", "  ") for c in cats])
    axes[0].invert_yaxis()
    axes[0].scatter([], [], color=SER[0], s=12, label="human")
    axes[0].scatter([], [], color=SER[1], s=12, label="mouse")
    axes[0].plot([], [], color=INK, lw=2, label="median")
    axes[0].legend(loc="lower left", fontsize=8)
    save(fig, fname)


def main():
    p = OUT / "alpha_samples.csv"
    if p.exists():
        S = pd.read_csv(p)
        strata_strip(S, ["recovery", "precision"], ["Authors' cells recovered by Singlet", "Singlet calls that are authors' cells"],
                     "fig_cell_calls_by_stratum", "fraction", (0, 1.02), 0.95)
        strata_strip(S, ["umi_ratio_matched"], ["UMIs per cell, Singlet / authors (shared cells)"], "fig_umi_ratio_by_stratum",
                     "ratio (1 = same depth)", None, 1.0)
        sc = S[S.status == "scored"]
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        for sp, col in (("human", SER[0]), ("mouse", SER[1])):
            s = sc[sc.species == sp].dropna(subset=["frac_reads_used", "umi_ratio_matched"])
            ax.scatter(s.frac_reads_used, s.umi_ratio_matched, s=14, color=col, alpha=0.8, linewidths=0, label=sp)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axhline(1, color=INK2, lw=1, ls="--")
        ax.plot([0.005, 1], [0.005, 1], color=GRID, lw=1)
        ax.set_xlabel("reads used / reads available in SRA")
        ax.set_ylabel("UMIs per cell, Singlet / authors")
        ax.set_title("Lower counts track the read cap", loc="left", fontsize=10)
        ax.legend()
        save(fig, "fig_umi_ratio_vs_reads_used")
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        for sp, col in (("human", SER[0]), ("mouse", SER[1])):
            s = sc[sc.species == sp]
            ax.scatter(s.noise_floor, s.cell_corr_depth_matched, s=14, color=col, alpha=0.8, linewidths=0, label=sp)
        lo = np.nanmin(sc[["noise_floor", "cell_corr_depth_matched"]].values) - 0.02
        ax.plot([lo, 1], [lo, 1], color=INK2, lw=1, ls="--", label="equal to the noise floor")
        ax.set_xlabel("noise floor (authors vs authors, thinned)")
        ax.set_ylabel("Singlet vs authors at equal depth")
        ax.set_title("Per-cell profiles at equal depth", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        save(fig, "fig_depth_matched_vs_floor")
    p = OUT / "alpha_census.json"
    if p.exists():
        c = json.load(open(p))
        h = c["reads"]["frac_reads_used_histogram"]
        labels = ["<5%", "5-10%", "10-25%", "25-50%", "50-75%", "75-90%", ">90%"]
        fig, ax = plt.subplots(figsize=(5.4, 3.2))
        ax.bar(labels, h["counts"], color=SER[0], width=0.8)
        ax.set_ylabel("samples")
        ax.set_xlabel("share of the sample's SRA reads used by the atlas")
        ax.set_title(f"Research Alpha: reads used per sample (n={sum(h['counts']):,}, human campaign)", loc="left", fontsize=10)
        ax.grid(axis="x", visible=False)
        save(fig, "fig_census_reads_used")
    p = OUT / "alpha-engine-campaign.json"
    if p.exists():
        d = json.load(open(p))
        tab = d.get("_table") or []
        if tab:
            T = pd.DataFrame(tab).fillna(0.0)
            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            x = np.arange(len(T))
            w = 0.26
            for k, (col, nm) in enumerate((("alpha30m", "Alpha engine, 30M reads (as published)"),
                                           ("baseline_full", "same engine, all reads"),
                                           ("starsolo", "STARsolo EmptyDrops_CR, all reads"))):
                ax.bar(x + (k - 1) * w, T[col], width=w - 0.02, color=SER[k], label=nm)
            ax.set_xticks(x)
            ax.set_xticklabels(T.dataset, rotation=20, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("recovery of 10x's published cells")
            ax.axhline(0.95, color=INK2, lw=1, ls="--")
            ax.legend(fontsize=7, loc="lower left")
            ax.grid(axis="x", visible=False)
            save(fig, "fig_engine_cell_recovery")


if __name__ == "__main__":
    main()
