# SPDX-License-Identifier: MIT
"""Cross-study summary of the empty-droplet study: tables (summary.json, tables.md) and figures (PNG) for docs/.

Runs where the per-sample results live (Clipper): ``python -P aggregate.py RESULTS --out REPORT_DIR``.
Reads ``<RESULTS>/<GSE>/<GSM>.json`` (+ ``.bc.npz``, ``.author.json``, ``.cellbender.json`` when present).
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference palette (dataviz skill, light mode): slots 1-3 validated all-pairs; gray for "not significant / other"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6"
GRAY, INK, INK2, GRID = "#a3a29d", "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})
BIN_EDGES = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 5000]
FLOORS = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500]


def load_all(res):
    S = []
    for f in sorted(glob.glob(os.path.join(res, "GSE*", "GSM*.json"))):
        if f.endswith((".author.json", ".cellbender.json")):
            continue
        R = json.load(open(f))
        base = f[:-5]
        for ext in ("author", "cellbender"):
            p = f"{base}.{ext}.json"
            if os.path.exists(p):
                R[ext] = json.load(open(p))
        R["_bc"] = base + ".bc.npz"
        S.append(R)
    return S


def is_mixture(R):
    b = R.get("barnyard") or {}
    ch = b.get("called_human_frac") or {}
    return bool(ch) and ch["0.05"] < 0.2 and ch["0.95"] > 0.8


def label(R):
    return f"{R['gse']}/{R['gsm']}"


def usable(R):
    """Samples whose called set is a real cell call: >= 200 called, median called UMI >= 100 (excludes HTO/ADT/guide
    libraries processed as RNA, which the atlas QC already flags)."""
    uc = R.get("umi_called") or {}
    return R["n_called"] >= 200 and (uc.get("0.5") or 0) >= 100


def bins_series(rows, key, xkey=None):
    x, y, n = [], [], []
    for r in rows:
        if r.get(key) is None or r["n"] < 20:
            continue
        hi = r["hi"] if r["hi"] is not None else r["lo"] * 2
        if xkey:
            if not r.get(xkey):
                continue
            x.append(r[xkey])
        else:
            x.append(np.sqrt(max(r["lo"], 1) * hi))
        y.append(r[key])
        n.append(r["n"])
    return np.array(x), np.array(y), np.array(n)


def fig_ranks(S, out, picks):
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for ax, R in zip(axes.ravel(), picks):
        z = np.load(R["_bc"])
        t = z["total"].astype(float)
        c = z["called"].astype(bool)
        sig = (~c) & (np.nan_to_num(z["fdr_cell"], nan=1) < 0.01) if "fdr_cell" in z.files else np.zeros(len(t), bool)
        o = np.argsort(-t)
        r = np.empty(len(t))
        r[o] = np.arange(1, len(t) + 1)
        keep = t > 0
        col = np.where(c, BLUE, np.where(sig, ORANGE, GRAY))
        idx = np.nonzero(keep)[0]
        idx = idx[np.random.default_rng(0).permutation(len(idx))]  # random draw order: neither class hides the other
        ax.scatter(r[idx], t[idx], s=2, c=col[idx], lw=0, rasterized=True)
        for cc, lab in ((GRAY, "uncalled, ambient-like"), (ORANGE, "uncalled, cell-like (FDR 1%)"), (BLUE, "called")):
            ax.scatter([], [], s=8, c=cc, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        k = (R.get("callers") or {}).get("knee", {}).get("threshold")
        if k and np.isfinite(k):
            ax.axhline(k, color=INK2, lw=0.8, ls="--")
            ax.text(1.2, k * 1.1, "knee", color=INK2, fontsize=7)
        ax.set_title(f"{label(R)}\n{R.get('protocol') or ''}", fontsize=8)
        ax.set_xlabel("barcode rank")
        ax.set_ylabel("UMIs (exon + intron)")
    axes.ravel()[0].legend(loc="lower left", fontsize=7, markerscale=4)
    fig.suptitle("Barcode-rank curves: called cells, and uncalled barcodes split by the profile test", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_rank_plots.png"))
    plt.close(fig)


def fig_profile(S, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for R in S:
        pt = R.get("profile_test")
        if not pt:
            continue
        x, y, _ = bins_series(pt["by_bin"], "frac_sig_fdr01", "median_test_umi")
        axes[0].plot(x, y, color=ORANGE if usable(R) else GRAY, alpha=0.55, lw=1)
        x, y, _ = bins_series(pt["by_bin"], "frac_sig_fdr01_alpha_div4", "median_test_umi")
        axes[1].plot(x, y, color=ORANGE if usable(R) else GRAY, alpha=0.55, lw=1)
    # power of the same test on thinned held-out called cells (median over samples); depth is exon UMIs
    P = {}
    for R in S:
        for p in R.get("power_thinned_called") or []:
            P.setdefault(p["depth"], []).append(p["power_p01"])
    if P:
        d = sorted(P)
        axes[0].plot(d, [np.median(P[k]) for k in d], color=BLUE, lw=2.2, label="power on thinned cells (median, p<0.01)")
        axes[0].legend(loc="lower right", fontsize=7)
    axes[0].set_title("Uncalled barcodes rejecting 'ambient only' (FDR 1%)", fontsize=9)
    axes[1].set_title("Same, null 4x more overdispersed", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("exon UMIs of the barcode (median of its total-UMI bin)")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("fraction of uncalled barcodes in bin")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_profile_test.png"))
    plt.close(fig)


def fig_split(S, out):
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    lab_done = set()
    for R in S:
        sh = R.get("split_half")
        if not sh or "uncalled_test_umi" not in sh or not usable(R):
            continue
        for key, col, lab in (("uncalled_test_umi", ORANGE, "uncalled barcodes"), ("null_alpha_div4_test_umi", GRAY, "ambient null (alpha/4)")):
            x, y, _ = bins_series(sh[key], "agree")
            ax.plot(x, y, color=col, alpha=0.5, lw=1, label=None if lab in lab_done else lab)
            lab_done.add(lab)
        c = sh.get("thinned_cells_heldout") or []
        if c:
            ax.plot([r["depth"] for r in c], [r["agree"] for r in c], color=BLUE, alpha=0.5, lw=1,
                    label=None if "cells" in lab_done else "thinned held-out cells")
            lab_done.add("cells")
    ax.set_xscale("log")
    ax.set_xlabel("exon UMIs")
    ax.set_ylabel("halves pick the same cluster")
    ax.set_ylim(0, 1.02)
    ax.set_title("Split-half cluster agreement", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_split_half.png"))
    plt.close(fig)


def fig_barnyard(S, out):
    B = [R for R in S if is_mixture(R) and usable(R)]
    if not B:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, R in enumerate(B):
        # from 30 UMIs: below that a few counts (often one molecule counted twice) pass as "pure"
        x, y, _ = bins_series([r for r in R["barnyard"]["by_bin_uncalled"] if r["lo"] >= 30], "frac_pure")
        ax.plot(x, y, color=ORANGE, alpha=0.6, lw=1, label="uncalled" if i == 0 else None)
        x, y, _ = bins_series(R["barnyard"]["by_bin_called"], "frac_pure")
        ax.plot(x, y, color=BLUE, alpha=0.6, lw=1, label="called" if i == 0 else None)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("barcode UMIs")
    ax.set_ylabel("fraction single-species")
    ax.set_title("Species mixing: >= 90% of UMIs from one genome, binomial p < 0.001 vs ambient mix", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_barnyard.png"))
    plt.close(fig)


def fig_storage(S, out):
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for R in S:
        st = R["storage"]
        tot = sum(st["nnz_total"].get(k, 0) for k in ("exon", "intron"))
        if not tot:
            continue
        y = [sum(st["keep_called_plus_umi_ge"][str(f)].get(f"nnz_{k}", 0) for k in ("exon", "intron")) / tot for f in FLOORS]
        c0 = sum(st["nnz_called"].get(k, 0) for k in ("exon", "intron")) / tot
        ax.plot(FLOORS, y, color=BLUE if usable(R) else GRAY, alpha=0.5, lw=1)
        ax.scatter([FLOORS[-1] * 1.25], [c0], s=8, color=INK2)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("keep called cells + uncalled barcodes with >= X UMIs")
    ax.set_ylabel("fraction of exon+intron entries kept")
    ax.set_title("Entries kept vs UMI floor (dot: called only)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_storage.png"))
    plt.close(fig)


def fig_groundtruth(S, out):
    # the pilot's TCR libraries are compared with the GEX call of the same wells: plotted apart, they are a different assay
    A = [R for R in S if ((R.get("author") or {}).get("author") or {}).get("uncalled_by_bin") and R["gse"] not in ("GSE145370", "GSE213957")]  # GSE213957 deposits QC-passed cells, not a call
    H = [R for R in S if (R.get("author") or {}).get("hashing")]
    if not A and not H:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, R in enumerate(A):
        x, y, _ = bins_series(R["author"]["author"]["uncalled_by_bin"], "frac_author_called")
        axes[0].plot(x, y, color=AQUA, alpha=0.6, lw=1, label="authors' call (Cell Ranger etc.)" if i == 0 else None)
        x, y, _ = bins_series(R["author"]["author"]["uncalled_by_bin"], "frac_profile_sig")
        axes[0].plot(x, y, color=ORANGE, alpha=0.4, lw=1, ls="--", label="profile test" if i == 0 else None)
    axes[0].set_title(f"Uncalled barcodes the authors called ({len(A)} samples)", fontsize=9)
    for R in H:
        h = R["author"]["hashing"]
        x, y, _ = bins_series(h["uncalled_by_bin"], "frac_hto_pos")
        axes[1].plot(x, y, color=VIOLET, lw=1.6, label="hashtag-positive")
        x, y, _ = bins_series(h["uncalled_by_bin"], "frac_profile_sig")
        axes[1].plot(x, y, color=ORANGE, lw=1.2, ls="--", label="profile test")
        x, y, _ = bins_series(h["called_by_bin"], "frac_hto_pos")
        axes[1].plot(x, y, color=BLUE, lw=1.2, label="called cells, hashtag-positive")
    axes[1].set_title("Cell hashing (GSE108313 PBMC): uncalled barcodes", fontsize=9)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("barcode UMIs (bin centre)")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("fraction of uncalled barcodes in bin")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_ground_truth.png"))
    plt.close(fig)


def fig_rules(out):
    p = os.path.join(out, "rules.json")
    if not os.path.exists(p):
        return None
    RJ = json.load(open(p))
    names = list(next(iter(RJ.values()))["rules"].keys())
    tab = {}
    for nm in names:
        saved, cl, cl_t, sp, sp_t, ed, ed_t = [], 0, 0, 0, 0, 0, 0
        for v in RJ.values():
            e = v["rules"][nm]
            if e["bytes_total"]:
                saved.append(1 - e["bytes_kept_est"] / e["bytes_total"])
            cl += e["cell_like_lost"]
            cl_t += e["cell_like_total"]
            sp += e.get("species_pure_lost", 0)
            sp_t += e.get("species_pure_total", 0)
            ed += e.get("emptydrops_extra_lost", 0)
            ed_t += e.get("emptydrops_extra_total", 0)
        tab[nm] = {"median_bytes_saved": float(np.median(saved)), "p10_bytes_saved": float(np.quantile(saved, 0.1)),
                   "p90_bytes_saved": float(np.quantile(saved, 0.9)), "cell_like_lost": cl / max(1, cl_t),
                   "species_pure_lost": sp / max(1, sp_t), "emptydrops_extra_lost": ed / max(1, ed_t),
                   "n_samples": len(saved)}
    fig, ax = plt.subplots(figsize=(6.2, 4))
    for nm, t in tab.items():
        if "q05" in nm:  # depth-relative variants coincide with the fixed floors here; in the table only
            continue
        ax.scatter(t["median_bytes_saved"], t["cell_like_lost"], s=36, color=ORANGE, zorder=3)
        ax.scatter(t["median_bytes_saved"], t["emptydrops_extra_lost"], s=36, color=AQUA, marker="s", zorder=3)
        ax.annotate(nm.replace("keep_", ""), (t["median_bytes_saved"], t["cell_like_lost"]), fontsize=7, color=INK2,
                    xytext=(4, 3), textcoords="offset points")
    ax.scatter([], [], s=36, color=ORANGE, label="profile-test cell-like barcodes lost")
    ax.scatter([], [], s=36, color=AQUA, marker="s", label="extra EmptyDrops cells lost")
    ax.set_xlabel("median matrix bytes saved per sample (estimate)")
    ax.set_ylabel("fraction lost (pooled over samples)")
    ax.set_title("Keep rules: saving vs what is lost", fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig7_rules.png"))
    plt.close(fig)
    return tab


def summary_rows(S):
    rows = []
    for R in S:
        pt = R.get("profile_test") or {}
        st = R["storage"]
        tot = {k: v for k, v in st["nnz_total"].items()}
        cal = st["nnz_called"]
        sh = R.get("split_half") or {}
        cb = R.get("cellbender") or {}
        au = (R.get("author") or {}).get("author") or {}
        hs = (R.get("author") or {}).get("hashing") or {}
        row = {
            "sample": label(R), "protocol": R.get("protocol"), "ref": R.get("reference_build"), "usable": usable(R),
            "n_barcodes": R["n_barcodes"], "n_called": R["n_called"],
            "called_umi_min": R.get("min_called_umi"), "called_umi_q05": (R.get("umi_called") or {}).get("0.05"),
            "called_umi_med": (R.get("umi_called") or {}).get("0.5"),
            "uncalled_umi_q99": (R.get("umi_uncalled") or {}).get("0.99"), "uncalled_umi_max": R.get("max_uncalled_umi"),
            "single_threshold_mismatch": (R.get("best_single_threshold") or {}).get("frac_mismatch"),
            "uncalled_entry_share": {k: 1 - cal.get(k, 0) / max(1, tot[k]) for k in tot},
            "uncalled_sig": pt.get("n_sig_fdr01"), "uncalled_sig_div4": pt.get("n_sig_fdr01_alpha_div4"),
            "sig_entry_share_of_uncalled": pt.get("nnz_share_of_sig"),
            "sig_umi_q05": (pt.get("umi_of_sig") or {}).get("0.05"), "sig_umi_med": (pt.get("umi_of_sig") or {}).get("0.5"),
            "split_agree_uncalled": sh.get("uncalled_overall"), "split_agree_null_div4": sh.get("null_alpha_div4_overall"),
            "called_look_empty": (R.get("called_look_empty") or {}).get("n"),
            "dup2_single_gene": R.get("dup2_single_gene"), "ambient_alpha": (R.get("ambient") or {}).get("alpha"),
            "barnyard_uncalled_pure": (R.get("barnyard") or {}).get("n_uncalled_pure") if is_mixture(R) else None,
            "barnyard_ambient_pool_pure": (R.get("barnyard") or {}).get("frac_pure_ambient_pool"),
            "cellbender_not_ours": cb.get("theirs_not_ours"), "cellbender_ours_not": cb.get("ours_not_theirs"),
            "author_not_ours": au.get("author_not_ours"), "ours_not_author": au.get("ours_not_author"),
            "author_jaccard": au.get("jaccard"),
            "hto_uncalled_pos": hs.get("n_uncalled_hto_pos"),
        }
        for name, c in (R.get("callers") or {}).items():
            cc = c.get("called") if isinstance(c.get("called"), dict) else None
            if cc:
                row[f"{name}_not_ours"] = cc["theirs_not_ours"]
                row[f"{name}_ours_not"] = cc["ours_not_theirs"]
        rows.append(row)
    return rows


def rule_eval(S):
    """Entries kept and cell-like barcodes lost under candidate rules, per usable sample and pooled."""
    out = {}
    for R in S:
        if not usable(R):
            continue
        z = np.load(R["_bc"])
        t = z["total"]
        c = z["called"].astype(bool)
        sig = (~c) & (np.nan_to_num(z["fdr_cell"], nan=1) < 0.01) if "fdr_cell" in z.files else np.zeros(len(t), bool)
        q05 = float(np.quantile(t[c], 0.05))
        st = R["storage"]
        rules = {f"umi>={f}": t >= f for f in (20, 50, 100, 200)}
        rules.update({f"umi>={k}x_q05_called": t >= k * q05 for k in (0.25, 0.5, 1.0)})
        rules["profile_sig"] = sig
        e = {}
        for name, m in rules.items():
            keep = c | m
            e[name] = {"barcodes": int(keep.sum()), "sig_lost": int((sig & ~keep).sum()), "sig_total": int(sig.sum())}
        out[label(R)] = {"q05_called": q05, "rules": e}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = load_all(a.results)
    picks = []
    want = ["GSE145370", "GSE163793", "GSE157827", "GSE137539", "GSE274381", "GSE266546"]
    for g in want:
        c = [R for R in S if R["gse"] == g and usable(R) and os.path.exists(R["_bc"])]
        if c:
            picks.append(c[0])
    for R in S:
        if len(picks) >= 6:
            break
        if R not in picks and usable(R) and os.path.exists(R["_bc"]):
            picks.append(R)
    fig_ranks(S, a.out, picks[:6])
    fig_profile(S, a.out)
    fig_split(S, a.out)
    fig_barnyard(S, a.out)
    fig_storage(S, a.out)
    fig_groundtruth(S, a.out)
    rule_tab = fig_rules(a.out)
    rows = summary_rows(S)
    summ = {"n_samples": len(S), "n_usable": sum(r["usable"] for r in rows), "rows": rows, "rules": rule_eval(S),
            "rule_table": rule_tab}
    for R in S:  # keep the per-bin tables of every sample for the report
        summ.setdefault("per_sample", {})[label(R)] = {k: R.get(k) for k in (
            "profile_test", "split_half", "power_thinned_called", "barnyard", "qc_fractions", "qc_by_bin_uncalled",
            "correlation", "called_look_empty", "ambient", "clusters", "callers", "storage", "author", "cellbender", "umi_called",
            "umi_uncalled", "dup2_single_gene", "dup2_exon1_intron1_same_gene", "best_single_threshold", "call_method", "summary")}
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=str)
    print(f"{len(S)} samples, {summ['n_usable']} usable -> {a.out}")


if __name__ == "__main__":
    main()
