# SPDX-License-Identifier: MIT
"""Figures and tables for docs/adt-modality.md, from the benchmark and the assembled library table (runs on Clipper).

    python -P bench/adt/report.py --bench $WORK/state/adt/benchmark/benchmark.json \
        --libs $WORK/state/adt/assembled/libraries.tsv --data $WORK/data/adt --out $WORK/state/adt/report [--example GSM]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b", "na": "#a3a29c"}


def style(plt):
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def fig_benchmark(bench, out, plt):
    rows = [b for b in bench if "ratio" in b]
    names = [b["method"] for b in rows]
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, (key, title, ref) in enumerate([("ratio", "UMIs / Cell Ranger", 1.0),
                                           ("pearson_log_entries", "Pearson r, log1p entries", 1.0),
                                           ("median_per_cell_pearson", "median per-cell Pearson r", 1.0)]):
        v = [b.get(key) or 0 for b in rows]
        ax[k].barh(range(len(v)), v, color=SERIES[0], height=0.6)
        for i, x in enumerate(v):
            ax[k].text(x, i, f" {x:.3f}", va="center", color=INK, fontsize=8)
        ax[k].axvline(ref, color=INK2, lw=0.8, ls="--")
        ax[k].set_yticks(range(len(v)))
        ax[k].set_yticklabels(names if k == 0 else [""] * len(v))
        ax[k].set_title(title)
        ax[k].set_xlim(0, max(1.15, max(v) * 1.2))
        ax[k].invert_yaxis()
    fig.suptitle("ADT counting vs Cell Ranger 3.0 (10x pbmc_1k_protein_v3, TotalSeq-B, 713 cells, 17 antibodies)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "adt_benchmark.png"), dpi=150)
    plt.close(fig)


METRICS = [("cell_empty_ratio", "cells / empty droplets (median ADT UMIs)", True),
           ("median_umi_cells", "median ADT UMIs per called cell", True),
           ("frac_feature", "reads with a panel barcode", False),
           ("frac_valid_barcode", "reads with a valid cell barcode", False),
           ("rna_concordance", "protein vs RNA (median Spearman)", False),
           ("saturation", "sequencing saturation", False)]


def fig_qc(libs, out, plt):
    from singlet.adt.qc import THRESHOLDS

    ok = libs[libs.status == "ok"].copy()
    fig, ax = plt.subplots(2, 3, figsize=(11, 5.6))
    for a, (m, title, logx) in zip(ax.ravel(), METRICS):
        v = pd.to_numeric(ok[m], errors="coerce")
        y = np.arange(len(ok))
        cols = [STATUS.get(x, STATUS["na"]) for x in ok.verdict.fillna("na")]
        a.scatter(v, np.random.default_rng(0).uniform(-0.3, 0.3, len(v)), c=cols, s=22, edgecolors=SURFACE,
                  linewidths=0.8)
        _, fail, warn = THRESHOLDS.get(m, (None, None, None))
        for t, lab in ((fail, "fail"), (warn, "warn")):
            if t is not None:
                a.axvline(t, color=STATUS[lab], lw=1, ls="--")
                a.text(t, 0.42, f" {lab} {t:g}", color=INK2, fontsize=7, va="bottom")
        if logx:
            a.set_xscale("log")
        a.set_yticks([])
        a.set_ylim(-0.5, 0.6)
        a.set_title(title)
    hand = [plt.Line2D([], [], marker="o", ls="", color=STATUS[k], label=k) for k in ("pass", "warn", "fail")]
    fig.legend(handles=hand, loc="upper right", ncol=3)
    fig.suptitle(f"Library QC, {len(ok)} processed libraries (dot colour = verdict)", fontsize=10, color=INK, x=0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out, "adt_qc_libraries.png"), dpi=150)
    plt.close(fig)


def fig_concordance(libs, data, out, plt):
    recs = []
    for r in libs[libs.status == "ok"].itertuples():
        p = os.path.join(data, r.gse, r.gsm, "concordance.tsv")
        if os.path.exists(p):
            c = pd.read_csv(p, sep="\t")
            c["gsm"] = r.gsm
            recs.append(c)
    if not recs:
        return
    c = pd.concat(recs)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].hist(c.rho.dropna(), bins=30, color=SERIES[0], alpha=0.9, label="antibody vs its gene")
    ax[0].hist(c.rho_null.dropna(), bins=30, color=SERIES[1], alpha=0.6, label="permuted cells (null)")
    ax[0].set_xlabel("Spearman rho over called cells")
    ax[0].set_title(f"Protein vs RNA, {len(c)} antibody-library pairs")
    ax[0].legend()
    g = c.assign(gene=c.gene_row.str.split("|").str[-1]).groupby("gene").rho.agg(["median", "size"])
    g = g[g["size"] >= 3].sort_values("median", ascending=False).head(25)
    ax[1].barh(range(len(g)), g["median"], color=SERIES[0], height=0.6)
    ax[1].set_yticks(range(len(g)))
    ax[1].set_yticklabels([f"{i} (n={n})" for i, n in zip(g.index, g["size"])], fontsize=7)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("median Spearman rho across libraries")
    ax[1].set_title("Best-measured antigens (>= 3 libraries)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "adt_rna_concordance.png"), dpi=150)
    plt.close(fig)
    c.to_csv(os.path.join(out, "concordance_all.tsv"), sep="\t", index=False)


def fig_example(gsm, libs, data, out, plt):
    from singlet.adt import count as C
    from singlet.adt import gex as G
    from singlet.adt import normalize as N

    r = libs[libs.gsm == gsm].iloc[0]
    d = os.path.join(data, r.gse, gsm)
    M, feats, amb, outside, meta = C.load_npz(os.path.join(d, "counts.npz"))
    pan = pd.read_csv(os.path.join(d, "panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
    gx = G.read_legacy(r.gse, meta["gex_gsm"], with_counts=False)
    tot = np.asarray(M.sum(axis=0)).ravel()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    o = np.argsort(-tot)
    ax[0].loglog(np.arange(1, len(o) + 1), np.maximum(tot[o], 0.8), color=INK2, lw=1)
    called_rank = np.flatnonzero(gx.called[o]) + 1
    ax[0].scatter(called_rank, np.maximum(tot[o][called_rank - 1], 0.8), s=3, color=SERIES[0], label="GEX-called cell")
    ax[0].set_xlabel("GEX column rank by ADT UMIs")
    ax[0].set_ylabel("ADT UMIs")
    ax[0].set_title("ADT totals: called cells vs empty droplets")
    ax[0].legend(markerscale=4)
    cells = np.flatnonzero(gx.called)
    emp = N.empty_droplets(tot, gx.called)
    X = M[:, cells].toarray()
    E = M[:, emp].toarray()
    iso = np.flatnonzero(pan.kind.values == "isotype")
    D = N.dsb(X, E, iso if len(iso) else None)
    names = list(pan.name)

    def pick(*cands):
        for c in cands:
            for i, n in enumerate(names):
                if n.replace("Hu.", "").split(" ")[0].upper() == c:
                    return i
        return None

    pairs = [(pick("CD4"), pick("CD8", "CD8A")), (pick("CD3"), pick("CD19", "CD20"))]
    for k, (i, j) in enumerate(pairs):
        a = ax[k + 1]
        if i is None or j is None:
            a.set_axis_off()
            continue
        a.scatter(D[i], D[j], s=3, color=SERIES[0], alpha=0.5)
        a.set_xlabel(f"{names[i]} (DSB)")
        a.set_ylabel(f"{names[j]} (DSB)")
        a.set_title("DSB-normalised protein")
    fig.suptitle(f"{r.gse} / {gsm} (GEX {meta['gex_gsm']})", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "adt_example.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench")
    ap.add_argument("--libs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--example")
    a = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    os.makedirs(a.out, exist_ok=True)
    if a.bench and os.path.exists(a.bench):
        fig_benchmark(json.load(open(a.bench)), a.out, plt)
    libs = pd.read_csv(a.libs, sep="\t")
    fig_qc(libs, a.out, plt)
    fig_concordance(libs, a.data, a.out, plt)
    if a.example:
        fig_example(a.example, libs, a.data, a.out, plt)
    print("figures in", a.out, os.listdir(a.out))


if __name__ == "__main__":
    main()
