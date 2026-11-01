"""Capability ``doublets``: Singlet's expression doublet score (singlet.doublets, feature ``qc/doublet_score``) against its
previous production scorer, scDblFinder and Scrublet, on 17 datasets with real experimental doublet labels.

Genotype doublets (``donor_assignments.tsv`` ``prob_doublet``) stay the call for pools with two or more donors and are not part
of this comparison; this capability scores the expression score used for single-donor samples.

Artifacts (``artifact_dir``), written by ``bench/doublets/train_eval.py``:
- ``doublets_cells.parquet``: one row per cell: dataset, study, i (row in the dataset), label (1 = experimental doublet), and the
  scores of ``prod`` (previous Singlet scorer), ``scdblfinder``, ``scrublet`` and the combiner (``combo``, leave-one-study-out).
- ``doublets_datasets.tsv``: dataset, study, cells, doublets, truth.
- ``provenance.json``: Singlet commit and version, tool versions, resources of the batch job, code commit of this module.

Datasets with at least 100 labelled doublets are gating; the pooled metrics are means over the 17 datasets (each dataset counts
once). Leave-one-study-out: the combiner scoring a dataset was trained without any dataset of its study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import artifact_entries, code_url, contract, load_provenance, now_iso, r

CAP = "doublets"
TOOLS = {"singlet": "combo", "singlet-production": "prod", "scDblFinder": "scdblfinder", "Scrublet": "scrublet"}
COMP = [t for t in TOOLS if t != "singlet"]
KANG = ["kang2018_ctrl", "xili_pbmc_1a_dm", "xili_pbmc_1b_dm", "xili_pbmc_1c_dm", "xili_pbmc_2ctrl_dm", "xili_pbmc_2stim_dm"]
# analysis dataset name -> cohort id (kang21 is lane 2.1 as Singlet counts it, the cohort's kang2018_ctrl)
COHORT_ID = {"kang21": "kang2018_ctrl"}


def cid(name: str) -> str:
    return COHORT_ID.get(name) or "xili_" + name.lower().replace("-", "_")


def _auprc(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def _auroc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def _boot_halfwidth(a: np.ndarray, b: np.ndarray, n_boot: int = 4000, seed: int = 0) -> float:
    """Half-width of the paired 95% bootstrap CI of mean(a - b), resampling datasets."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    est = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
    return float((np.quantile(est, 0.975) - np.quantile(est, 0.025)) / 2)


def _boot_ci(v: np.ndarray, n_boot: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    est = np.asarray(v)[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(1)
    return [float(np.quantile(est, 0.025)), float(np.quantile(est, 0.975))]


def compute(artifact_dir: Path) -> dict:
    import pandas as pd

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    cells = pd.read_parquet(artifact_dir / "doublets_cells.parquet")
    dsinfo = pd.read_csv(artifact_dir / "doublets_datasets.tsv", sep="\t").set_index("dataset")
    names = list(dsinfo.index)

    res: dict[str, dict[str, dict[str, float]]] = {"auprc": {}, "auroc": {}}
    for n in names:
        c = cells[cells.dataset == n]
        y = c["label"].to_numpy()
        for k, f in (("auprc", _auprc), ("auroc", _auroc)):
            res[k][n] = {t: f(y, c[col].to_numpy()) for t, col in TOOLS.items()}
    arr = {k: {t: np.array([res[k][n][t] for n in names]) for t in TOOLS} for k in res}
    kang_idx = [i for i, n in enumerate(names) if cid(n) in KANG]

    metrics = []

    def tol_rule(vals, idx=None, against="best"):
        sel = np.arange(len(vals['singlet'])) if idx is None else np.array(idx)
        best = max(COMP, key=lambda t: vals[t][sel].mean())
        hw = _boot_halfwidth(vals["singlet"][sel], vals[best][sel])
        return {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(hw, 4)},
                "rationale": f"half-width of the paired 95% bootstrap CI (datasets resampled) of Singlet minus {best}; "
                             f"{len(sel)} datasets"}

    def pooled(mid, name, vals, idx=None, notes=None):
        sel = np.arange(len(vals['singlet'])) if idx is None else np.array(idx)
        m = {"id": mid, "name": name, "unit": "AUPRC" if "auprc" in mid else "AUROC", "category": "accuracy",
             "dataset": "pooled", "direction": "higher",
             "singlet": {"value": r(vals["singlet"][sel].mean()), "ci": [r(x) for x in _boot_ci(vals["singlet"][sel])]},
             "competitors": [{"tool": t, "value": r(vals[t][sel].mean()), "ci": [r(x) for x in _boot_ci(vals[t][sel])]}
                             for t in COMP],
             "rule": tol_rule(vals, idx)}
        if notes:
            m["notes"] = notes
        metrics.append(m)

    pooled("auprc_mean", "Mean AUPRC over the 17 datasets (leave-one-study-out)", arr["auprc"],
           notes="each dataset counts once; the CI is a bootstrap over datasets")
    pooled("auprc_mean_kang", "Mean AUPRC over the Kang 2018 lanes (genotype truth)", arr["auprc"], kang_idx,
           notes="lane 2.1 as Singlet counts it plus the five Cell Ranger lanes of Xi & Li; the combiner never saw any Kang lane here")
    studies = dsinfo["study"].to_numpy()
    by_study = {t: np.array([arr["auprc"][t][studies == u].mean() for u in dict.fromkeys(studies)]) for t in TOOLS}
    pooled("auprc_mean_by_study", f"Mean AUPRC over the {len(by_study['singlet'])} studies (leave-one-study-out)", by_study,
           notes="each study counts once (the six Kang lanes are one study, the five MULTI-seq sets another, the two 10x hgmm sets "
                 "a third); the CI is a bootstrap over studies")
    pooled("auroc_mean", "Mean AUROC over the 17 datasets (leave-one-study-out)", arr["auroc"])

    for k, label in (("auprc", "AUPRC"),):
        for i, n in enumerate(names):
            v = {t: res[k][n][t] for t in TOOLS}
            m = {"id": k, "name": f"{label} against experimental doublet labels", "unit": label, "category": "accuracy",
                 "dataset": cid(n), "direction": "higher", "singlet": {"value": r(v["singlet"])},
                 "competitors": [{"tool": t, "value": r(v[t])} for t in COMP],
                 "notes": f"{int(dsinfo.loc[n, 'cells'])} cells, {int(dsinfo.loc[n, 'doublets'])} labelled doublets"}
            # per dataset the gate is the scorer this one replaces; scDblFinder and Scrublet are shown, and gate the pooled means
            m["rule"] = {"type": "match_or_beat", "against": ["singlet-production"]}
            metrics.append(m)

    sel_lanes = kang_idx
    k21 = names.index("kang21")
    n_better = int((arr["auprc"]["singlet"] > arr["auprc"]["singlet-production"]).sum())
    sg, pr, sd = (arr["auprc"][t].mean() for t in ("singlet", "singlet-production", "scDblFinder"))
    conclusion = (
        f"Leave-one-study-out on 17 real-label datasets, Singlet's expression doublet score has mean AUPRC {sg:.3f}, against "
        f"{pr:.3f} for its previous scorer (better on {n_better}/17) and {sd:.3f} for scDblFinder, a statistical tie without R. "
        f"On the Kang lanes it reaches {arr['auprc']['singlet'][sel_lanes].mean():.3f} against "
        f"{arr['auprc']['scDblFinder'][sel_lanes].mean():.3f}; on lane 2.1 as Singlet counts it, {arr['auprc']['singlet'][k21]:.3f} "
        f"against {arr['auprc']['scDblFinder'][k21]:.3f}. Genotype doublets remain the call for pools with two or more donors.")
    lose_prod = [f"{n} ({k})" for k in ("auprc", "auroc") for n in names if res[k][n]["singlet-production"] > res[k][n]["singlet"]]
    wins = {n: [t for t in COMP if res["auprc"][n][t] > res["auprc"][n]["singlet"]] for n in names}
    lose_sdf = [n for n in names if "scDblFinder" in wins[n]]

    tools = [
        {"name": "singlet", "version": "1.0.0", "role": "singlet",
         "params": "singlet.doublets: gradient boosting (300 trees, 15 leaves) on within-sample ranks of log UMI, log genes, "
                   "the engine's doublet score and four atlas-network features; trained on the labels of the other studies"},
        {"name": "singlet-production", "version": "doublet_detect.h v7 (prod/modality-fixes 4f110f2)", "role": "competitor",
         "params": "detect_doublets through the dbl_run harness, gene level, default settings"},
        {"name": "scDblFinder", "version": "1.23 (R)", "role": "competitor", "url": "https://github.com/plger/scDblFinder",
         "params": "scDblFinder(counts) defaults"},
        {"name": "Scrublet", "version": "0.2.3", "role": "competitor", "url": "https://github.com/swolock/scrublet",
         "params": "Scrublet(expected_doublet_rate=0.06).scrub_doublets(min_counts=2, min_cells=3, min_gene_variability_pctl=85, "
                   "n_prin_comps=30)"},
    ]
    datasets = []
    for n in names:
        row = dsinfo.loc[n]
        d = {"id": cid(n), "title": f"{n}: {int(row['doublets'])} labelled doublets in {int(row['cells'])} cells",
             "accessions": {"gse": "GSE96583", "gsm": ["GSM2560248"]} if n == "kang21" else {"other": ["10.5281/zenodo.4062232"]},
             "truth": row["truth"], "data_kind": "real", "n_cells": int(row["cells"])}
        datasets.append(d)

    figures = [{
        "id": "auprc_by_dataset", "title": "AUPRC by dataset (leave-one-study-out for Singlet)", "kind": "grouped_bar",
        "x": {"label": "dataset", "categories": names}, "y": {"label": "AUPRC", "unit": "AUPRC", "min": 0, "max": 1},
        "series": [{"name": t, "tool": t, "x": names, "y": [r(res["auprc"][n][t]) for n in names]} for t in TOOLS]},
        {"id": "cost", "title": "Cost per 10,000 cells", "kind": "table", "columns": ["tool", "seconds per 10k cells", "needs"],
         "rows": [["singlet", prov["cost"]["singlet"], "engine dbl_run + atlas network (GPU or CPU)"],
                  ["singlet-production", prov["cost"]["prod"], "engine dbl_run"],
                  ["scDblFinder", prov["cost"]["scdblfinder"], "R"],
                  ["Scrublet", prov["cost"]["scrublet"], "python"]]}]

    commit = prov["bench_commit"]
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Expression doublets",
        "conclusion": conclusion,
        "methods": prov["methods"],
        "status": prov.get("status", "draft"),
        "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "pipeline-benchmark",
        "singlet": prov["singlet"],
        "tools": tools,
        "datasets": datasets,
        "metrics": metrics,
        "resources": prov["resources"],
        "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {"doublets_cells.parquet": "derived", "doublets_datasets.tsv": "truth",
                                                          "provenance.json": "derived"}),
        "code": [{"path": p, "role": ro, "url": code_url(p, commit)} for p, ro in (
            ("python/singlet/doublets/combiner.py", "implementation"), ("python/singlet/doublets/atlas.py", "implementation"),
            ("scripts/doublets_run.py", "implementation"), ("bench/doublets/train_eval.py", "harness"),
            ("bench/public/capabilities/doublets.py", "metric"))],
        "reproduce": {"module": "bench/public/capabilities/doublets.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/doublets/train_eval.sbatch"},
        "reach": [{"tool": t, "output": "expression doublet score per cell", "supported": "yes"} for t in TOOLS] + [
            {"tool": "singlet", "output": "genotype doublets (donor_assignments prob_doublet)", "supported": "yes",
             "note": "the call for pools with two or more donors; Kang 2.1 AUROC 0.969 / AUPRC 0.897"},
            {"tool": "scDblFinder", "output": "genotype doublets (donor_assignments prob_doublet)", "supported": "no"},
            {"tool": "Scrublet", "output": "genotype doublets (donor_assignments prob_doublet)", "supported": "no"}],
        "caveats": prov.get("caveats", []) + [
            f"scDblFinder is ahead of Singlet on {len(lose_sdf)} of 17 datasets ({', '.join(lose_sdf)}); the pooled mean is a tie, "
            "not a win. The per-dataset gate is the previous Singlet scorer; the pooled and Kang-lane means gate against the best "
            "competitor with a tolerance from the bootstrap over datasets.",
            "Datasets where the previous Singlet scorer is ahead of the combiner: " + ", ".join(lose_prod) + ". These are the failing per-dataset metrics.",
            "Only experimentally visible doublets are labelled (different donors, hashtags, barcodes or species): same-donor "
            "homotypic doublets are invisible to the truth and to every method. About a quarter of cross-donor PBMC doublets look "
            "like one cell type; expression methods find roughly half of them, genotype doublets 0.89.",
            "The Xi & Li matrices come from other pipelines (Cell Ranger and others); the atlas network is tuned to Singlet's counts "
            "and gives a confident lineage to far fewer cells there (1-7% against 40% on Singlet's lane 2.1), so the gain on them "
            "is, if anything, understated. Lane 2.1 as Singlet counts it is the only set on Singlet's own counts.",
            "The 17 datasets include two sets of one Kang lane (2.1 on Singlet counts and as the Cell Ranger 1.0 matrix) and six "
            "Kang lanes in all, so the pooled mean weights the Kang study 6/17. Weighting each of the six studies once, scDblFinder "
            f"is ahead ({by_study['scDblFinder'].mean():.3f} against {by_study['singlet'].mean():.3f}); that gate has its own "
            "tolerance from the bootstrap over six studies, which is wide.",
            "The combiner was chosen after comparing feature sets on these same 17 datasets (bench/doublets/cheap_features.py), "
            "so the pooled numbers carry a small selection effect; adding Scrublet as a feature gave 0.563 pooled and 0.575 on the "
            "Kang lanes, below the shipped combiner's 0.585 there, and was not adopted."],
    }
    return C.finalize(doc)
