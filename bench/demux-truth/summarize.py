#!/usr/bin/env python3
"""Score every mixture and write summary.json and summary.md.

usage: summarize.py RESULTS_DIR
RESULTS_DIR holds one subdirectory per mixture (truth.tsv, runs*.jsonl, diag.jsonl) and qc/qc_<id>.json.
Only aggregate metrics are written. Genotypes and per-SNP counts are never read here.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score  # noqa: E402

TOOLS = [("singlet", "our engine"), ("vireo", "Vireo"), ("souporcell", "souporcell + troublet")]
SEEDS = (42, 7, 123)
SAMPLE_INFO = {
    "S1": "GSM7873657 H1, GSE246624, 10x 3' v3 PBMC",
    "S2": "GSM7873658 H2, GSE246624, 10x 3' v3 PBMC",
    "S5": "GSM5102900 HC1, GSE167363, 10x 3' v3 PBMC",
    "S6": "GSM5102901 HC2, GSE167363, 10x 3' v3 PBMC",
    "S3": "GSM7901264, GSE247824, 10x 5' CD45+ cells (earlier panel)",
    "S4": "pbmc1k_v2, 10x 3' v2 PBMC (earlier panel)",
}


def f(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def rng(vals, nd=3):
    v = [x for x in vals if x is not None]
    if not v:
        return "n/a"
    return f"{np.median(v):.{nd}f} [{min(v):.{nd}f}, {max(v):.{nd}f}]" if len(v) > 1 else f"{v[0]:.{nd}f}"


def runs_of(m, tool, mode, t=None):
    out = []
    for tag, r in m["runs"].items():
        p = tag.split("_")
        if len(p) == 4 and p[0] == tool and p[1] == mode and (t is None or p[2] == f"t{t}"):
            out.append(r)
    return out


def main(root):
    root = Path(root)
    mixes = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "truth.tsv").exists()):
        try:
            score.main(d)
            mixes[d.name] = json.load(open(d / "metrics.json"))
        except Exception as e:  # a mixture that failed upstream must not stop the summary
            print("scoring failed", d.name, type(e).__name__, e)
    qc = {}
    for p in sorted((root / "qc").glob("qc_*.json")):
        q = json.load(open(p))
        qc[q["id"]] = q
    S = {"mixtures": mixes, "qc": qc}
    json.dump(S, open(root / "summary.json", "w"), indent=1)

    L = ["# Donor demux ground-truth benchmark, version 2", ""]
    L.append("Mixtures are built from single-donor 10x PBMC libraries run through the production pipeline. "
             "Truth is the donor of origin per cell. Planted doublets sum two cells of different donors, each thinned to a random 60 to 80 percent of its reads. "
             "Our engine is the October donor_demux.h. Vireo (vireoSNP) and souporcell with troublet read the same mixed AD and DP counts "
             "after the cellSNP-style site filter (at least 20 reads, minor allele fraction at least 0.1). "
             "All numbers are the median over 3 seeds with the range in brackets. Doublets are scored with prob_doublet.")
    L.append("")

    L.append("## Sample QC")
    L.append("")
    L.append("A single donor shows alt fractions at 0, 0.5 and 1 in the pseudobulk. Mass between the modes points to a pooled sample or contamination. "
             "Informative SNPs are covered SNPs with both alleles seen in at least 10 cells.")
    L.append("")
    L.append("| sample | library | cells | median covered SNPs per cell | median informative SNPs per cell | informative SNPs | SNPs with 20+ reads | near 0 | near 0.5 | near 1 | between | modes |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for sid in ("S1", "S2", "S5", "S6", "S3", "S4"):
        q = qc.get(sid)
        if not q:
            L.append(f"| {sid} | {SAMPLE_INFO[sid]} | n/a |  |  |  |  |  |  |  |  | no QC |")
            continue
        tri = q["frac_between"] < 0.15 and min(q["frac_near0"], q["frac_near_half"], q["frac_near1"]) > 0.05
        L.append(f"| {sid} | {SAMPLE_INFO[sid]} | {q['n_cells']} | {q['median_covered_per_cell']:.0f} | {q['median_informative_per_cell']:.0f} | "
                 f"{q['informative_snps']} | {q['snps_ge20_reads']} | {f(q['frac_near0'], 2)} | {f(q['frac_near_half'], 2)} | {f(q['frac_near1'], 2)} | "
                 f"{f(q['frac_between'], 2)} | {'trimodal' if tri else 'check for pooling'} |")
    L.append("")
    L.append("Pseudobulk alt fraction histogram, 20 bins from 0 to 1 (SNP counts):")
    L.append("")
    for sid, q in qc.items():
        L.append(f"- **{sid}:** " + " ".join(str(x) for x in q["hist20"]))
    L.append("")

    L.append("## Accuracy against truth, K given (16 threads)")
    L.append("")
    L.append("| mixture | donors | cells | doublets | tool | singlet accuracy | ARI | unassigned | doublet AUROC | doublet F1 | K found | wall s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n, m in mixes.items():
        for tool, name in TOOLS:
            rs = runs_of(m, tool, "k", 16)
            if not rs:
                L.append(f"| {n} | {m['mix']['n_donors']} | {m['mix']['n_cells']} | {m['mix']['n_doublets']} | {name} | no result |  |  |  |  |  |  |")
                continue
            L.append(f"| {n} | {m['mix']['n_donors']} | {m['mix']['n_cells']} | {m['mix']['n_doublets']} | {name} | "
                     f"{rng([r['accuracy_singlets'] for r in rs])} | {rng([r['ari_singlets'] for r in rs])} | "
                     f"{rng([r['unassigned_frac'] for r in rs])} | {rng([r['doublet_auroc'] for r in rs])} | "
                     f"{rng([r['doublet_f1'] for r in rs])} | {rng([r['k_found'] for r in rs], 0)} | {rng([r['wall_s'] for r in rs], 1)} |")
    L.append("")

    L.append("## Auto K (16 threads)")
    L.append("")
    L.append("Vireo picks K by its own ELBO over K = 2 to 6. souporcell has no K selection, so K is the maximum of BIC over K = 2 to 6.")
    L.append("")
    L.append("| mixture | true K | tool | K found per seed (42, 7, 123) | singlet accuracy | ARI | doublet AUROC | doublet F1 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for n, m in mixes.items():
        for tool, name in TOOLS:
            rs = {r["seed"]: r for r in runs_of(m, tool, "auto", 16)}
            if not rs:
                L.append(f"| {n} | {m['mix']['n_donors']} | {name} | no result |  |  |  |  |")
                continue
            ks = ", ".join(str(rs[s]["k_found"]) if s in rs else "n/a" for s in SEEDS)
            v = list(rs.values())
            L.append(f"| {n} | {m['mix']['n_donors']} | {name} | {ks} | {rng([r['accuracy_singlets'] for r in v])} | "
                     f"{rng([r['ari_singlets'] for r in v])} | {rng([r['doublet_auroc'] for r in v])} | {rng([r['doublet_f1'] for r in v])} |")
    L.append("")

    L.append("## ELBO diagnostic on the m2 mixtures")
    L.append("")
    L.append("VB is started from the truth labels (doublets split over their two donors) and run to convergence. "
             "Its ELBO is compared with the ELBO at the optimum the engine finds by itself. "
             "A higher found ELBO means the model prefers the wrong partition (misspecification). "
             "A higher truth ELBO means the model prefers the truth and the seeding misses it.")
    L.append("")
    L.append("| mixture | seed | ELBO found | ELBO truth start | ELBO truth converged | truth minus found | ARI found | ARI truth converged | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    any_diag = False
    for n, m in mixes.items():
        for d in m.get("diag", []):
            any_diag = True
            delta = d["elbo_truth_converged"] - d["elbo_found"]
            tol = 1e-4 * abs(d["elbo_found"])
            verdict = ("equal: same optimum" if abs(delta) <= tol else
                       "truth higher: seeding at fault" if delta > 0 else "found higher: model misspecified")
            L.append(f"| {n} | {d['seed']} | {d['elbo_found']:.1f} | {d['elbo_truth_start']:.1f} | {d['elbo_truth_converged']:.1f} | "
                     f"{delta:.1f} | {f(d['ari_found_vs_truth'])} | {f(d['ari_truth_converged_vs_truth'])} | {verdict} |")
    if not any_diag:
        L.append("| no diagnostic output |  |  |  |  |  |  |  |  |")
    L.append("")

    L.append("## Stability across thread counts and seeds (K given)")
    L.append("")
    L.append("| tool | thread pairs | min agreement | seed pairs | min agreement | min pair ARI | widest accuracy spread over seeds | widest ARI spread over seeds |")
    L.append("|---|---|---|---|---|---|---|---|")
    for tool, name in TOOLS:
        th, sd, acc, ar, rs = [], [], 0.0, 0.0, []
        for m in mixes.values():
            for pr in m["pairs"].values():
                if pr["tool"] == tool and pr["mode"] == "k":
                    (th if pr["kind"] == "threads" else sd).append(pr)
            rs = runs_of(m, tool, "k", 16)
            if len(rs) > 1:
                acc = max(acc, max(r["accuracy_singlets"] for r in rs) - min(r["accuracy_singlets"] for r in rs))
                ar = max(ar, max(r["ari_singlets"] for r in rs) - min(r["ari_singlets"] for r in rs))
        L.append(f"| {name} | {len(th)} | {f(min([p['agreement'] for p in th]) if th else None, 4)} | {len(sd)} | "
                 f"{f(min([p['agreement'] for p in sd]) if sd else None, 4)} | {f(min([p['ari'] for p in sd]) if sd else None, 4)} | "
                 f"{f(acc if rs else None)} | {f(ar if rs else None)} |")
    L.append("")

    L.append("## Run time by thread count (K given, median wall s / core-s)")
    L.append("")
    L.append("| mixture | tool | 4 threads | 16 threads |")
    L.append("|---|---|---|---|")
    for n, m in mixes.items():
        for tool, name in TOOLS:
            cells = []
            for t in (4, 16):
                rs = runs_of(m, tool, "k", t)
                cells.append(f"{np.median([r['wall_s'] for r in rs]):.1f} / {np.median([r['cpu_s'] for r in rs]):.1f}" if rs else "n/a")
            L.append(f"| {n} | {name} | {cells[0]} | {cells[1]} |")
    L.append("")

    fails = []
    for d in sorted(root.iterdir()):
        for p in d.glob("failures-*.txt") if d.is_dir() else []:
            txt = p.read_text().strip()
            if txt:
                fails.append(f"- **{d.name} {p.stem.replace('failures-', '')}:** " + txt.replace("\n", " | ")[:600])
    L.append("## Failed or skipped runs")
    L.append("")
    L += fails or ["None."]
    L.append("")
    open(root / "summary.md", "w").write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
