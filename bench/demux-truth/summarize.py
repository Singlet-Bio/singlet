#!/usr/bin/env python3
"""Aggregate the scored mixtures into summary.json and summary.md.

usage: summarize.py RESULTS_DIR      (RESULTS_DIR holds one subdirectory per mixture with metrics.json)
"""
import json
import sys
from pathlib import Path


def f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main(root):
    root = Path(root)
    mixes = {}
    for p in sorted(root.glob("*/metrics.json")):
        mixes[p.parent.name] = json.load(open(p))
    if not mixes:
        print("no metrics found")
        return 1
    S = {"mixtures": mixes}

    # observed variation, K-given runs only (thread counts, repeats, seeds), per kind
    var = {}
    for kind in ("repeat", "threads", "seed"):
        ag, ari, dp99, dpm = [], [], [], []
        for m in mixes.values():
            for pr in m["pairs"].values():
                if pr["kind"] == kind and pr["a"].startswith("k_"):
                    ag.append(pr["agreement"]); ari.append(pr["ari"]); dp99.append(pr["dprob_p99"]); dpm.append(pr["dprob_mean"])
        if ag:
            var[kind] = {"min_agreement": min(ag), "min_pair_ari": min(ari), "max_dprob_p99": max(dp99), "max_dprob_mean": max(dpm), "n_pairs": len(ag)}
    S["observed_variation"] = var

    # spread of truth metrics across the K-given runs of one mixture
    spread = {k: 0.0 for k in ("accuracy_singlets", "ari_singlets", "unassigned_frac", "doublet_auroc_1_minus_pmax")}
    for m in mixes.values():
        ks = [r for t, r in m["runs"].items() if t.startswith("k_")]
        for k in spread:
            v = [r[k] for r in ks if r[k] is not None]
            if v:
                spread[k] = max(spread[k], max(v) - min(v))
    S["truth_metric_spread_k_given"] = spread

    # agreement and probability bands use repeats and thread counts (same optimum, different float order).
    # Seed changes can reach another optimum, so seed variation only widens the bands against truth.
    rt = [v for k, v in var.items() if k in ("repeat", "threads")]
    worst_dis = 1.0 - min([v["min_agreement"] for v in rt] or [1.0])
    worst_dp99 = max([v["max_dprob_p99"] for v in rt] or [0.0])
    worst_dpm = max([v["max_dprob_mean"] for v in rt] or [0.0])
    S["tolerance_proposal"] = {
        "basis": "2x the largest variation seen across repeats and thread counts (4/8/16) for agreement and probabilities, "
                 "and across repeats, threads and seeds (42 vs 7) for metrics against truth, with a floor",
        "seed_assignment_agreement_min_observed": var.get("seed", {}).get("min_agreement"),
        "k_given_assignment_agreement_min": round(1.0 - max(0.01, 2 * worst_dis), 4),
        "ari_vs_truth_drop_max": round(max(0.01, 2 * spread["ari_singlets"]), 4),
        "accuracy_vs_truth_drop_max": round(max(0.01, 2 * spread["accuracy_singlets"]), 4),
        "unassigned_fraction_abs_diff_max": round(max(0.01, 2 * spread["unassigned_frac"]), 4),
        "doublet_auroc_drop_max": round(max(0.02, 2 * spread["doublet_auroc_1_minus_pmax"]), 4),
        "prob_max_abs_diff_mean_max": round(max(0.005, 2 * worst_dpm), 4),
        "prob_max_abs_diff_p99_max": round(max(0.05, 2 * worst_dp99), 4),
        "k_found": "equal to the CPU value on every mixture when K is given",
        "doublet_f1": "CPU engine makes no doublet calls (prob_doublet is 0). Judge a port by AUROC of 1 - prob_max until a doublet term exists.",
    }
    json.dump(S, open(root / "summary.json", "w"), indent=1)

    L = ["# Donor demux ground-truth benchmark", ""]
    L.append("Mixtures are built from real single-donor samples (same chemistry, same SNP panel). "
             "Truth: donor of origin per cell, planted cross-donor doublets. Reference run: K given, 16 threads, seed 42.")
    L.append("")
    L.append("## Accuracy against truth (K given, 16 threads, repeat 1)")
    L.append("")
    L.append("| mixture | donors | cells | doublets | acc singlets | ARI | unassigned | dbl AUROC | dbl called | wall s | core-s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n, m in mixes.items():
        r = m["runs"].get("k_t16_r1")
        if r:
            L.append(f"| {n} | {m['mix']['n_donors']} | {r['n_cells']} | {r['n_doublets']} | {f(r['accuracy_singlets'])} | {f(r['ari_singlets'])} | "
                     f"{f(r['unassigned_frac'])} | {f(r['doublet_auroc_1_minus_pmax'])} | {r['doublet_called_n']} | {f(r['wall_s'],1)} | {f(r['cpu_s'],1)} |")
    L.append("")
    L.append("## Auto K (16 threads)")
    L.append("")
    L.append("| mixture | true K | K found | ARI | unassigned | wall s | core-s |")
    L.append("|---|---|---|---|---|---|---|")
    for n, m in mixes.items():
        r = m["runs"].get("auto_t16_r1")
        if r:
            L.append(f"| {n} | {m['mix']['n_donors']} | {r['k_found']} | {f(r['ari_singlets'])} | {f(r['unassigned_frac'])} | {f(r['wall_s'],1)} | {f(r['cpu_s'],1)} |")
    L.append("")
    L.append("## Demux time by thread count (K given, wall s / core-s)")
    L.append("")
    L.append("| mixture | t4 | t8 | t16 |")
    L.append("|---|---|---|---|")
    for n, m in mixes.items():
        cells = []
        for t in ("k_t4_r1", "k_t8_r1", "k_t16_r1"):
            r = m["runs"].get(t)
            cells.append(f"{f(r['wall_s'],1)} / {f(r['cpu_s'],1)}" if r else "n/a")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Run-to-run variation (worst case over mixtures, K given)")
    L.append("")
    L.append("| source | pairs | min assignment agreement | min pair ARI | max mean abs d prob_max | max p99 abs d prob_max |")
    L.append("|---|---|---|---|---|---|")
    for k, v in var.items():
        L.append(f"| {k} | {v['n_pairs']} | {f(v['min_agreement'],4)} | {f(v['min_pair_ari'],4)} | {f(v['max_dprob_mean'],4)} | {f(v['max_dprob_p99'],4)} |")
    L.append("")
    L.append("## Proposed tolerance for a GPU port")
    L.append("")
    L.append(S["tolerance_proposal"]["basis"] + ".")
    L.append("")
    for k, v in S["tolerance_proposal"].items():
        if k != "basis":
            L.append(f"- **{k}:** {v}")
    L.append("")
    open(root / "summary.md", "w").write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
