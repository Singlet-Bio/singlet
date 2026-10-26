"""donor-demux: genotype-free donor demultiplexing, including the number of donors (K) when it is not given.

compute(artifact_dir) reads only the small per-cell label tables written by bench/public/genotype/lib/export_demux.py
and returns the capability document (same function the harness used to write results/donor-demux.json). No genotype or
allele count is in the artifacts.

    python bench/public/capabilities/donor_demux.py ARTIFACT_DIR OUT.json [--final]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent / "genotype" / "lib"))
import contract  # noqa: E402
import demux_eval as DE  # noqa: E402
from _gcommon import (SINGLET_BINARY, SINGLET_BINARY_SHA256, SINGLET_COMMIT, REPO, ari, artifact_entry, boot, code_url,  # noqa: E402
                     now, r6, wilson)

CAP = "donor-demux"
ART_BASE = "https://data.singlet.bio/benchmarking/1.0.0/donor-demux"
K_THRESHOLD = 0.9
HELD_OUT_NOTE = "kang2018_stim"

DATASETS = {
    "kang2018_ctrl": dict(title="Kang 2018 batch 2 control lane (8 lupus donors), 10x 3' v1", n=8,
                          accessions={"gse": "GSE96583", "gsm": ["GSM2560248"], "srr": ["SRR5398238"]},
                          truth="demuxlet donor and doublet calls with donor genotypes (authors; GSE96583_batch2.total.tsne.df)"),
    "kang2018_stim": dict(title="Kang 2018 batch 2 IFN-beta lane (the same 8 donors)", n=8,
                          accessions={"gse": "GSE96583", "gsm": ["GSM2560249"], "srr": ["SRR5398239"]},
                          truth="demuxlet donor and doublet calls with donor genotypes (authors)"),
    "kang2018_b1_a": dict(title="Kang 2018 batch 1 lane A (4 donors)", n=4,
                          accessions={"gse": "GSE96583", "gsm": ["GSM2560245"], "srr": ["SRR5398235"]},
                          truth="demuxlet donor and doublet calls with donor genotypes (authors; GSE96583_batch1.total.tsne.df)"),
    "kang2018_b1_b": dict(title="Kang 2018 batch 1 lane B (4 donors)", n=4,
                          accessions={"gse": "GSE96583", "gsm": ["GSM2560246"], "srr": ["SRR5398236"]},
                          truth="demuxlet donor and doublet calls with donor genotypes (authors)"),
    "kang2018_b1_c": dict(title="Kang 2018 batch 1 lane C (8 donors, the donors of A and B)", n=8,
                          accessions={"gse": "GSE96583", "gsm": ["GSM2560247"], "srr": ["SRR5398237"]},
                          truth="demuxlet donor and doublet calls with donor genotypes (authors)"),
    "maester_k562_bt142_3p": dict(title="MAESTER cell-line mixture (K562 + BT142), 10x 3' v3", n=2,
                                  accessions={"gse": "GSE182685"},
                                  truth="two real cell lines; the authors' cell labels (expression and mt-variant labels agree)"),
    "maester_bpdcn712_3p": dict(title="MAESTER BPDCN712 bone marrow, 10x 3' v3 (one patient)", n=1,
                                accessions={"gse": "GSE182685"},
                                truth="a single patient: every called cell is one donor, K = 1"),
}
DEV = "kang2018_ctrl"


def _load(art: Path):
    cells = pd.read_csv(art / "donor_demux_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    pools = pd.read_csv(art / "donor_demux_pools.tsv", sep="\t")
    res = pd.read_csv(art / "donor_demux_resources.tsv", sep="\t")
    prov = json.loads((art / "provenance.json").read_text())
    return cells, pools, res, prov


def _truth(g: pd.DataFrame) -> pd.DataFrame:
    return g.set_index("barcode")[["truth_donor", "truth_class"]].rename(columns={"truth_donor": "ind", "truth_class": "multiplets"})


def _score_tool(g: pd.DataFrame, col: str):
    if col not in g.columns or (g[col] == "").all():
        return None
    return DE.score(_truth(g), g.set_index("barcode")[col])


def _ari_ci(g: pd.DataFrame, col: str, B: int = 200):
    t = g[(g["truth_class"] == "singlet") & ~g["truth_donor"].isin(DE.NO_DONOR)]
    t = t[t["truth_donor"].map(t["truth_donor"].value_counts()) >= DE.MIN_TRUTH_DONOR_CELLS]
    y, p = t["truth_donor"].values, t[col].values
    return boot(lambda idx: ari(y[idx], p[idx]), len(t), B=B)


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    cells, pools, res, prov = _load(art)
    pools = pools.set_index("pool")
    order = [p for p in DATASETS if p in pools.index]
    held = [p for p in order if p != DEV]
    S = {}   # pool -> tool -> score dict
    ARI_CI = {}
    for p in order:
        g = cells[cells["pool"] == p]
        S[p] = {}
        S[p]["singlet"] = _score_tool(g, "singlet_auto")
        S[p]["singlet_told"] = _score_tool(g, "singlet_told")
        k_pool = int(pools.loc[p, "k_true"])
        # vireo and souporcell need at least two donors; with K = 1 they have nothing to separate and are not scored
        seeds = [] if k_pool == 1 else [s for s in (_score_tool(g, f"vireo_told_s{i}") for i in (1, 2, 3)) if s]
        S[p]["vireo_seeds"] = seeds
        S[p]["vireo"] = {k: float(np.mean([s[k] for s in seeds])) for k in ("ari", "ari_assigned", "accuracy_assigned", "coverage", "min_donor_recall")} if seeds else None
        S[p]["souporcell"] = None if k_pool == 1 else _score_tool(g, "souporcell")
        ARI_CI[p] = _ari_ci(g, "singlet_auto")

    # tolerance for ARI: seed-to-seed standard deviation of vireo (the competitor's own run-to-run variation)
    sds = [np.std([s["ari"] for s in S[p]["vireo_seeds"]], ddof=1) for p in order if len(S[p]["vireo_seeds"]) > 1]
    tol_ari = float(max(np.sqrt(np.mean(np.square(sds))) if sds else 0.0, 0.0))
    tol_ari = round(max(tol_ari, 0.002), 4)

    held2 = [p for p in held if int(pools.loc[p, "k_true"]) > 1]
    metrics = []

    def M(id_, name, unit, cat, ds, direction, sing, comps, rule, ci=None, notes=None):
        m = {"id": id_, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
             "singlet": {"value": r6(sing)}, "rule": rule}
        if ci:
            m["singlet"]["ci"] = [r6(ci[0]), r6(ci[1])]
        if comps:
            m["competitors"] = [{"tool": k, "value": r6(v)} for k, v in comps.items() if v is not None]
        if notes:
            m["notes"] = notes
        metrics.append(m)

    # ---- headline: unknown K
    def khat_s(p): return int(pools.loc[p, "singlet_k_selected"]) if not pd.isna(pools.loc[p, "singlet_k_selected"]) else int(pools.loc[p, "singlet_k_labels"])
    def khat_v(p): return int(pools.loc[p, "vireo_n12_khat_1pct"])
    kt = {p: int(pools.loc[p, "k_true"]) for p in order}
    ok_s = [khat_s(p) == kt[p] for p in held]; ok_v = [khat_v(p) == kt[p] for p in held]
    n = len(held)
    M("k_correct_fraction", "Fraction of held-out pools with the right donor count (K estimated, not given)", "fraction", "accuracy",
      "pooled", "higher", sum(ok_s) / n, {"vireo": sum(ok_v) / n},
      {"type": "threshold", "value": K_THRESHOLD,
       "rationale": f"at most 1 miss in 10 pools; with {n} held-out pools this means every pool must be right. 'Highly accurate' was the requirement for the unknown-K claim"},
      ci=wilson(sum(ok_s), n),
      notes=f"{n} real pools, true K = " + ", ".join(f"{p}: {kt[p]}" for p in held) + ". Singlet estimates K with --n-donors -1; vireo has no K search, so K is the number of donors holding >= 1% of cells when it is run with -N 12.")
    M("k_correct_vs_vireo", "Right donor count vs vireo (N = 12, surplus donors pruned)", "fraction", "accuracy", "pooled", "higher",
      sum(ok_s) / n, {"vireo": sum(ok_v) / n}, {"type": "match_or_beat", "against": "best"}, ci=wilson(sum(ok_s), n))
    mae_s = float(np.mean([abs(khat_s(p) - kt[p]) for p in held])); mae_v = float(np.mean([abs(khat_v(p) - kt[p]) for p in held]))
    M("k_abs_error", "Mean absolute error of the estimated donor count", "donors", "accuracy", "pooled", "lower", mae_s,
      {"vireo": mae_v}, {"type": "match_or_beat", "against": "best"})
    s_dev, v_dev = khat_s(DEV), khat_v(DEV)
    M("k_correct_dev", "Right donor count on the development lane (K = 8)", "fraction", "accuracy", DEV, "higher",
      float(s_dev == kt[DEV]), {"vireo": float(v_dev == kt[DEV])}, {"type": "match_or_beat", "against": "best"},
      notes=f"Singlet K = {s_dev}, vireo K = {v_dev}. Development data: Singlet's auto-K was tuned on this lane (PR #36).")

    # ---- per pool accuracy of the assignments
    for p in order:
        s, v, so = S[p]["singlet"], S[p]["vireo"], S[p]["souporcell"]
        comps = {"vireo": v["ari"] if v else None, "souporcell": so["ari"] if so else None}
        if kt[p] == 1:
            M("ari", "Adjusted Rand index of donor labels (one true donor: 1 only if every cell is in one cluster)", "ARI", "accuracy", p, "higher",
              s["ari"], {}, {"type": "threshold", "value": 0.95,
                             "rationale": "a one-donor pool has a single true class, so ARI is 1 when all cells share one label and falls with every cell split off or called doublet; 0.95 allows a few percent"},
              ci=ARI_CI[p], notes="vireo and souporcell need at least two donors and are not scored.")
            continue
        M("ari", "Adjusted Rand index of donor labels (truth singlets)", "ARI", "accuracy", p, "higher", s["ari"], comps,
          {"type": "match_or_beat", "against": "best", "tolerance": {"abs": tol_ari},
           "rationale": f"{tol_ari} = pooled seed-to-seed standard deviation of vireo's ARI (3 seeds per pool), the competitor's own run-to-run variation (floor 0.002)"},
          ci=ARI_CI[p], notes="Singlet K estimated; competitors given the true K. Unassigned and doublet calls count as their own labels.")
    for key, nm, hi in (("ari", "Mean ARI over pools", True), ("accuracy_assigned", "Mean accuracy of assigned cells over pools", True),
                        ("coverage", "Mean fraction of truth singlets given a donor", True)):
        def mean_of(tool):
            vals = []
            for p in held2:
                t = S[p][tool]
                if t is None:
                    return None
                vals.append(t[key])
            return float(np.mean(vals))
        sv = mean_of("singlet")
        ci = None
        if key == "ari":
            per = np.array([S[p]["singlet"]["ari"] for p in held2]); ci = boot(lambda i: float(per[i].mean()), len(per))
        M(f"mean_{key}_heldout", nm + " (held-out pools with K > 1)", "fraction" if key != "ari" else "ARI", "accuracy" if key != "coverage" else "coverage",
          "pooled", "higher", sv, {"vireo": mean_of("vireo"), "souporcell": mean_of("souporcell")},
          {"type": "match_or_beat", "against": "best", "tolerance": {"abs": tol_ari},
           "rationale": f"{tol_ari} = pooled seed-to-seed standard deviation of vireo's ARI"}, ci=ci)
    mr = lambda tool: float(min(S[p][tool]["min_donor_recall"] for p in held2)) if all(S[p][tool] for p in held2) else None
    M("min_donor_recall", "Recall of the worst-recalled donor in any held-out pool", "fraction", "robustness", "pooled", "higher",
      mr("singlet"), {"vireo": mr("vireo"), "souporcell": mr("souporcell")},
      {"type": "match_or_beat", "against": "best"}, notes="A donor merged into another or never found gives 0.")

    # ---- figures
    tools = [("singlet", "Singlet (K estimated)"), ("vireo", "vireo (K given)"), ("souporcell", "souporcell (K given)")]
    fig_ari = {"id": "ari_by_pool", "title": "Donor-label ARI by pool", "kind": "grouped_bar",
               "caption": "Singlet estimates K; vireo and souporcell are given the true K.",
               "x": {"label": "pool", "categories": order}, "y": {"label": "ARI", "unit": "ARI", "min": 0, "max": 1},
               "series": []}
    for key, nm in tools:
        ys = [(S[p][key]["ari"] if S[p][key] else None) for p in order]
        fig_ari["series"].append({"name": nm, "tool": key, "x": order, "y": [r6(y) for y in ys]})
    k_fig = {"id": "k_estimated_vs_true", "title": "Estimated vs true donor count", "kind": "scatter",
             "caption": "One point per pool; the diagonal is correct.",
             "x": {"label": "true K", "unit": "donors"}, "y": {"label": "estimated K", "unit": "donors"},
             "reference_lines": [], "series": [
                 {"name": "Singlet", "tool": "singlet", "x": [kt[p] for p in order], "y": [khat_s(p) for p in order]},
                 {"name": "vireo (N = 12)", "tool": "vireo", "x": [kt[p] for p in order], "y": [khat_v(p) for p in order]}]}
    table = {"id": "per_pool", "title": "Per-pool results", "kind": "table",
             "columns": ["pool", "true K", "Singlet K", "vireo K (1%)", "Singlet ARI", "vireo ARI", "souporcell ARI", "Singlet coverage", "Singlet min recall"],
             "rows": [[p, kt[p], khat_s(p), khat_v(p), r6(S[p]["singlet"]["ari"]), r6(S[p]["vireo"]["ari"]) if S[p]["vireo"] else None,
                       r6(S[p]["souporcell"]["ari"]) if S[p]["souporcell"] else None, r6(S[p]["singlet"]["coverage"]),
                       r6(S[p]["singlet"]["min_donor_recall"])] for p in order]}

    resources = []
    for _, r in res[res["tool"] != "mgatk"].iterrows():
        tool = {"singlet": "singlet", "singlet_told": "singlet", "cellsnp-lite": "cellsnp-lite", "vireo_told": "vireo",
                "vireo_n12": "vireo", "souporcell": "souporcell"}[r["tool"]]
        resources.append({"tool": tool, "dataset": r["pool"], "wall_s": r6(r["wall_s"]), "max_rss_gb": r6(r["max_rss_gb"]),
                          "threads": int(r["threads"]), "cpu_h": r6(r["wall_s"] * r["threads"] / 3600.0),
                          "notes": {"singlet": "end to end from reads incl. alignment, K estimated",
                                    "singlet_told": "K given", "cellsnp-lite": "pileup on the shared BAM and panel (vireo input)",
                                    "vireo_told": "K given, one seed, clustering only", "vireo_n12": "N = 12, clustering only",
                                    "souporcell": "remap + freebayes + vartrix + clustering, k given"}[r["tool"]]})

    conclusion = _conclusion(order, held, S, khat_s, khat_v, kt, sum(ok_s), n, sum(ok_v))
    doc = {
        "schema": contract.SCHEMA_CAPABILITY, "schema_version": contract.SCHEMA_VERSION, "capability": CAP,
        "title": "Genotype-free donor demultiplexing and donor count",
        "conclusion": conclusion, "methods": METHODS, "status": "final", "generated_at": now(), "owner": "pipeline-benchmark",
        "singlet": {"version": "1.0.0", "commit": SINGLET_COMMIT, "branch": "prod/campaign", "binary": SINGLET_BINARY,
                    "binary_sha256": SINGLET_BINARY_SHA256,
                    "build_flags": prov.get("singlet_command", "--pipeline --n-donors -1 (default 1000G SNP panel)")},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": prov["singlet_command"]},
            {"name": "vireo", "version": "0.5.9", "role": "competitor", "url": "https://github.com/single-cell-genetics/vireo",
             "citation": "Huang et al. 2019, Genome Biol 20:273", "params": prov["vireo_command"]},
            {"name": "cellsnp-lite", "version": "1.2.3", "role": "helper", "url": "https://github.com/single-cell-genetics/cellsnp-lite",
             "params": prov["cellsnp_command"]},
            {"name": "souporcell", "version": "2.5", "role": "competitor", "url": "https://github.com/wheaton5/souporcell",
             "citation": "Heaton et al. 2020, Nat Methods 17:615", "params": prov["souporcell_command"]},
            {"name": "demuxlet-authors", "version": "n/a", "role": "truth",
             "params": "donor calls published with the data (Kang 2018, GSE96583), made with donor genotypes"},
            {"name": "maester-authors", "version": "n/a", "role": "truth", "params": "cell-line labels and single patient of Miller 2022"},
        ],
        "datasets": [{"id": p, "title": DATASETS[p]["title"], "accessions": DATASETS[p]["accessions"], "truth": DATASETS[p]["truth"], "data_kind": "real",
                      "n_cells": int(pools.loc[p, "truth_cells"]) if not pd.isna(pools.loc[p, "truth_cells"]) else None,
                      **({"notes": "development data: Singlet's auto-K and mt caller were developed on this lane (PR #36)"} if p == DEV else
                         {"notes": "same donors and chemistry as the development lane; stimulated with IFN-beta"} if p == HELD_OUT_NOTE else {})}
                     for p in order],
        "metrics": metrics, "resources": resources, "figures": [k_fig, fig_ari, table],
        "reach": [
            {"tool": "singlet", "output": "donor count estimated from the reads", "supported": "yes", "note": "--n-donors -1 (default): held-out likelihood search over K"},
            {"tool": "vireo", "output": "donor count estimated from the reads", "supported": "no", "note": "N must be given; over-specifying N leaves surplus donors, which is not a selection rule"},
            {"tool": "souporcell", "output": "donor count estimated from the reads", "supported": "no", "note": "k must be given"},
            {"tool": "singlet", "output": "donor labels without reference genotypes", "supported": "yes", "note": "also labels doublets"},
            {"tool": "vireo", "output": "donor labels without reference genotypes", "supported": "yes", "note": "needs a cellsnp-lite pileup first"},
            {"tool": "souporcell", "output": "donor labels without reference genotypes", "supported": "yes", "note": "remaps reads; slowest"},
            {"tool": "singlet", "output": "one command from reads to labels", "supported": "yes", "note": "alignment, cell calling, pileup and demultiplexing in one run"},
        ],
        "artifacts": [artifact_entry(art / f, f, "derived" if f.endswith("tsv") else "singlet_output", ART_BASE)
                      for f in ("donor_demux_cells.tsv.gz", "donor_demux_pools.tsv", "donor_demux_resources.tsv", "provenance.json")],
        "code": [
            {"path": "include/singlet/pileup/donor_demux.h", "role": "implementation", "url": code_url("include/singlet/pileup/donor_demux.h", SINGLET_COMMIT)},
            {"path": "bench/public/capabilities/donor_demux.py", "role": "metric", "url": code_url("bench/public/capabilities/donor_demux.py", prov["code_commit"])},
            {"path": "bench/public/genotype", "role": "harness", "url": code_url("bench/public/genotype", prov["code_commit"], tree=True)},
        ],
        "reproduce": {"module": "bench/public/capabilities/donor_demux.py", "function": "compute", "laptop_minutes": 2,
                      "clipper_command": "sbatch bench/public/genotype/hpc/export.sbatch demux"},
        "caveats": _caveats(order, held, S, khat_s, khat_v, kt),
    }
    return contract.finalize(doc)


METHODS = (
    "Real pools only, each with truth from the data's authors: five Kang 2018 lupus PBMC lanes (true K = 8, 8, 8, 4, 4; labels are "
    "demuxlet calls made with donor genotypes), the MAESTER K562 + BT142 cell-line mixture (K = 2) and a single-patient bone marrow "
    "(K = 1). Reads were rebuilt from the submitted Cell Ranger BAMs (Kang) or downloaded (MAESTER), and every tool sees the same "
    "tagged BAM and the same called cells: Singlet is run once end to end (STAR alignment, cell calling, demultiplexing, K estimated "
    "by the held-out-likelihood search), and cellsnp-lite, vireo and souporcell are run on its BAM and cell list with the same 1000 "
    "Genomes SNP panel. Vireo has no K search, so its K is the number of donors holding at least 1% of cells when it is given N = 12 "
    "(the documented way to over-specify N); the rule was fixed before the runs. Vireo (3 seeds) and souporcell get the true K for the "
    "accuracy metrics. Scores are on truth singlets: ARI over all of them, with unassigned and doublet calls as their own labels; "
    "coverage is the fraction given a donor; per-donor recall uses Hungarian matching. The ARI tolerance is vireo's seed-to-seed "
    "standard deviation. The K metric is the fraction of held-out pools with the exact true K, gated at 0.9. The Kang batch 2 control "
    "lane is development data and is reported separately; it is excluded from the pooled numbers."
)


def _conclusion(order, held, S, khat_s, khat_v, kt, ok_s, n, ok_v):
    wrong = [f"{p} (K = {kt[p]}, called {khat_s(p)})" for p in held if khat_s(p) != kt[p]]
    mean_s = np.mean([S[p]["singlet"]["ari"] for p in held])
    vs = [S[p]["vireo"]["ari"] for p in held if S[p]["vireo"]]
    mean_v = np.mean(vs) if vs else float("nan")
    txt = (f"Without being told the donor count, Singlet finds the exact K in {ok_s} of {n} held-out real pools "
           f"(vireo with N = 12: {ok_v} of {n}); mean ARI {mean_s:.3f} against {mean_v:.3f} for vireo given the true K.")
    if wrong:
        txt += " Wrong: " + "; ".join(wrong) + "."
    return txt


def _caveats(order, held, S, khat_s, khat_v, kt):
    c = [
        "Truth for the Kang lanes is demuxlet's call, not an independent array; its own errors are in every number.",
        "The Kang batch 2 control lane is development data (auto-K and the mt caller were developed on it in PR #36). The batch 2 "
        "IFN-beta lane has the same donors and chemistry and is therefore only a weak hold-out; the strict hold-outs are the "
        "batch 1 lanes and the MAESTER pools.",
        "True K takes the values 1, 2, 4 and 8 only; K = 3, 5, 6, 7 and K > 8 have no real pool with truth in the cohort. No "
        "simulated or in-silico pools are used.",
        "Vireo has no K-selection procedure. Its K here is a prune-by-size rule on an N = 12 run (donors with >= 1% of cells); "
        "other rules would give other numbers. Souporcell cannot estimate K and is only scored with the true K.",
        "The BPDCN pool has one patient; any donor label other than one cluster counts against the tool, and homotypic doublets "
        "cannot be told from singlets by genotype.",
        "Kang FASTQs were rebuilt from mapped primary alignments of Cell Ranger BAMs, so unmapped reads are missing.",
    ]
    if khat_s(DEV) != kt[DEV]:
        c.append(
            f"Failure on the development lane: Singlet's automatic K is {khat_s(DEV)} for {kt[DEV]} donors (k_correct_dev). The extra "
            "component is a ~105-cell duplicate of one donor (its genotypes agree with that donor's at 0.88 of shared sites at >= 10 reads, "
            "against 0.60-0.68 between any two different donors and >= 0.96 for the same donor in two lanes). The held-out K search "
            "accepts it because select_k_heldout (include/singlet/pileup/donor_demux.h:1879-1880) takes any K whose count of 'valid' "
            "components rises with paired z > 4, and count_valid_components (line 1757) declares a component valid when held-out cells "
            "are better explained by it than by any other single donor or donor pair (mean log-likelihood ratio > 1 nat, z > 5); "
            "neither test compares genotypes, so a cell subpopulation of one donor that differs in coverage or ambient RNA passes.")
    bad = [p for p in held if khat_s(p) != kt[p]]
    if bad:
        c.append("Singlet's donor count is wrong in " + ", ".join(bad) + "; see the metric k_correct_fraction. This is a failing result, "
                 "not a data problem: the held-out K search (select_k_heldout, include/singlet/pileup/donor_demux.h) accepts a component "
                 "as a donor when it passes the distinctness test and raises held-out likelihood.")
    return c


def main(argv=None):
    a = argv or sys.argv[1:]
    doc = compute(Path(a[0]))
    Path(a[1]).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    st = contract.capability_status(doc)
    print(a[1], st)


if __name__ == "__main__":
    main()
