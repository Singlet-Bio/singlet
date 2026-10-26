# SPDX-License-Identifier: MIT
"""ambient-rna capability of the Singlet 1.0.0 public benchmark: contamination estimate and correction vs SoupX, DecontX, CellBender.

    python bench/public/capabilities/ambient_rna.py ARTIFACT_DIR [--out bench/public/results/ambient-rna.json]

`compute(artifact_dir)` is both the writer of the published file and the function the notebook re-runs. It reads only the
small per-cell tables in ARTIFACT_DIR (built on Clipper by capabilities/harness/barnyard_artifacts.py) and returns the document.

Truth (human + mouse mixtures of HEK293T and NIH3T3 cells): in a cell of one species every UMI of the other species is ambient,
so the cell's true contamination fraction is rho = (other-species fraction of its UMIs) / (other species' share of the ambient pool).
The pool is the barcodes with 1-100 UMI (top 50,000), Singlet's own definition; 3 other definitions are reported as sensitivity.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _adcommon as C  # noqa: E402

CAP = "ambient-rna"
PURE = 0.05  # a cell is a pure human / pure mouse cell when the other species supplies at most 5% of its UMIs

# (cohort id, development data?, title)
DATASETS = [
    ("hgmm1k_v3", True, "10x HEK293T + NIH3T3, 1k cells, 3' v3"),
    ("hgmm10k_v3", False, "10x HEK293T + NIH3T3, 11k cells, 3' v3"),
    ("hgmm1k_v2", False, "10x HEK293T + NIH3T3, 1k cells, 3' v2"),
]
TISSUE = [  # (cohort id, development data?, title, accessions): real-tissue truth from donor genotypes
    ("kang2018_ctrl", True, "Kang 2018 PBMC pool, batch 2 control lane (8 donors, 10x 3' v1)", {"gse": "GSE96583", "gsm": ["GSM2560248"], "srr": ["SRR5398238"]}),
]
ACCESSIONS = {
    "hgmm1k_v3": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v3/hgmm_1k_v3_fastqs.tar",
    "hgmm10k_v3": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.0/hgmm_10k_v3/hgmm_10k_v3_fastqs.tar",
    "hgmm1k_v2": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/hgmm_1k_v2/hgmm_1k_v2_fastqs.tar",
}
# internal column prefix -> published tool name
TOOLS = {"singlet": "singlet", "soupx": "SoupX", "decontx": "DecontX", "decontx_bg": "DecontX-background", "cellbender": "CellBender",
         "cellbender_rec": "CellBender-recommended"}
METRICS = [  # id, name, unit, direction, category
    ("rho_spearman", "Per-cell contamination rank correlation (Spearman) with truth", "correlation", "higher", "accuracy"),
    ("bias_abs_log2", "Mean contamination removed vs truth, |log2 ratio|", "log2", "lower", "accuracy"),
    ("bias_low_depth_abs_log2", "Same, lowest-depth third of cells", "log2", "lower", "robustness"),
    ("cross_species_residual", "Cross-species UMIs left after correction", "fraction", "lower", "accuracy"),
    ("own_removal_gap", "Own-species removal vs expected ambient, |ratio - 1|", "fraction", "lower", "accuracy"),
]


# ------------------------------------------------------------------------------------------------ data
def _load(art: Path, ds: str):
    df = pd.read_csv(art / f"{ds}_cells.tsv.gz", sep="\t")
    pool = json.loads((art / f"{ds}_pool.json").read_text())
    return df, pool


def _arrays(df: pd.DataFrame, pool: dict, alpha: float | None = None):
    """Pure-cell arrays for every tool present with a value for every pure cell, plus the truth."""
    alpha = pool["pools"]["1_100_top50k"]["mouse_share"] if alpha is None else alpha
    H, M = df.human_umi.values.astype(float), df.mouse_umi.values.astype(float)
    N = H + M
    fm = M / np.maximum(N, 1)
    human, mouse = fm <= PURE, fm >= 1 - PURE
    pure = human | mouse
    tools = [t for t in TOOLS if f"{t}_human" in df and df.loc[pure, f"{t}_human"].notna().all()]
    missing = [t for t in TOOLS if f"{t}_human" in df and t not in tools]
    if missing:  # a tool lacking some cells is scored on the cells every tool has
        ok = pure.copy()
        for t in TOOLS:
            if f"{t}_human" in df:
                ok &= df[f"{t}_human"].notna().values
        pure, tools = ok, [t for t in TOOLS if f"{t}_human" in df]
    other_b = np.where(human, M, H)
    own_b = np.where(human, H, M)
    other_share = np.where(human, alpha, 1 - alpha)
    truth = (other_b / np.maximum(N, 1)) / other_share
    d = {"n": N[pure], "other_b": other_b[pure], "own_b": own_b[pure], "truth": truth[pure],
         "expected_own": (other_b / other_share * (1 - other_share))[pure]}
    for t in tools:
        ha, mo = df[f"{t}_human"].values.astype(float), df[f"{t}_mouse"].values.astype(float)
        other_a, own_a = np.where(human, mo, ha)[pure], np.where(human, ha, mo)[pure]
        d[t] = {"removed": d["n"] - (ha + mo)[pure], "other_a": other_a, "own_a": own_a}
        d[t]["imp"] = d[t]["removed"] / np.maximum(d["n"], 1)
    d["low"] = d["n"] <= np.quantile(d["n"], 1 / 3)
    d["tools"], d["missing"] = tools, missing
    d["n_human"], d["n_mouse"] = int(human[pure].sum()), int(mouse[pure].sum())
    return d


def _spearman_boot(x: np.ndarray, y: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.empty(len(B))
    for i, ix in enumerate(B):
        a, b = rankdata(x[ix]), rankdata(y[ix])
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt(float((a * a).sum() * (b * b).sum()))
        out[i] = float((a * b).sum() / den) if den > 0 else np.nan
    return out


def _values(d: dict, t: str, B: np.ndarray) -> dict[str, np.ndarray]:
    """All five metrics for tool t over the index matrix B (row 0 may be the identity for the point estimate)."""
    a = d[t]
    S = lambda arr: arr[B].sum(1)  # noqa: E731
    mean_imp = S(a["imp"]) / B.shape[1]
    mean_tr = S(d["truth"]) / B.shape[1]
    out = {"rho_spearman": _spearman_boot(a["imp"], d["truth"], B),
           "bias_abs_log2": np.abs(np.log2(np.maximum(mean_imp, 1e-12) / mean_tr)),
           "cross_species_residual": S(a["other_a"]) / S(d["other_b"]),
           "own_removal_gap": np.abs(S(d["own_b"] - a["own_a"]) / S(d["expected_own"]) - 1)}
    low = d["low"].astype(float)
    nl = S(low)
    out["bias_low_depth_abs_log2"] = np.abs(np.log2(np.maximum(S(a["imp"] * low) / nl, 1e-12) / (S(d["truth"] * low) / nl)))
    return out


# ------------------------------------------------------------------------------------------- document
def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    datasets, metrics, figures, caveats = [], [], [], []
    present = [(ds, dev, title) for ds, dev, title in DATASETS if (art / f"{ds}_cells.tsv.gz").exists()]
    if not present and not any((art / f"{x[0]}_tissue.tsv.gz").exists() for x in TISSUE):
        raise FileNotFoundError(f"no <dataset>_cells.tsv.gz in {art}")
    fig_sp, fig_res, fig_bias = {}, {}, {}
    sens, scatter_rows, summary = [], None, {}
    for ds, dev, title in present:
        df, pool = _load(art, ds)
        d = _arrays(df, pool)
        n = len(d["n"])
        pt = np.arange(n)[None, :]
        B = C.boot_indices(n, (CAP, ds))
        stats = {}
        for t in d["tools"]:
            point = {k: float(v[0]) for k, v in _values(d, t, pt).items()}
            boot = _values(d, t, B)
            stats[t] = (point, boot)
        competitors = [t for t in d["tools"] if t != "singlet"]
        for t in d["missing"]:
            caveats.append(f"{TOOLS[t]} returned no value for some called cells of {ds}; every tool is scored on the cells all tools have.")
        for t in TOOLS:
            if t not in d["tools"] and t not in d["missing"] and t != "singlet":
                caveats.append(f"{TOOLS[t]} produced no output on {ds} (the tool failed or could not estimate); it is absent from that dataset's comparison.")
        datasets.append({
            "id": ds, "data_kind": "real", "title": title, "accessions": {"url": ACCESSIONS[ds]},
            "truth": "Mouse and human UMIs: in a pure cell all other-species UMIs are ambient, so rho = other-species fraction of the cell / other species' share of the ambient pool (barcodes with 1-100 UMI).",
            "n_cells": int(len(df)),
            "notes": ("DEVELOPMENT DATA: Singlet's ambient estimator was developed while looking at this library (PR #36)."
                      if dev else "Held out: not looked at while developing Singlet's ambient estimator.")
                     + f" Scored on {n} pure cells ({d['n_human']} human, {d['n_mouse']} mouse) of the {len(df)} called cells."})
        for mid, name, unit, direction, cat in METRICS:
            s_pt, s_boot = stats["singlet"][0][mid], stats["singlet"][1][mid]
            comps = []
            for t in competitors:
                comps.append({"tool": TOOLS[t], "value": C.r6(stats[t][0][mid]), "ci": C.ci95(stats[t][1][mid])})
            if not comps:
                continue
            vals = {c["tool"]: c["value"] for c in comps}
            best_name = (max if direction == "higher" else min)(vals, key=vals.get)
            best_t = next(t for t in competitors if TOOLS[t] == best_name)
            tol = C.half_width(s_boot - stats[best_t][1][mid])
            metrics.append({
                "id": mid, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
                "singlet": {"value": C.r6(s_pt), "ci": C.ci95(s_boot)}, "competitors": comps,
                "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(tol, 6)},
                         "rationale": f"half-width of the paired bootstrap 95% CI (1000 resamples of the {n} cells) of the difference between Singlet and the best competitor ({best_name}) on this metric: {tol:.4f}"},
                "notes": ("development data. " if dev else "held-out. ") + f"Best competitor: {best_name}."})
        metrics.append({"id": "n_pure_cells", "name": "Pure cells scored", "unit": "cells", "category": "coverage", "dataset": ds,
                        "direction": "none", "singlet": {"value": n}, "rule": {"type": "informational"}})
        for t in d["tools"]:
            fig_sp.setdefault(t, {})[ds] = stats[t][0]["rho_spearman"]
            fig_res.setdefault(t, {})[ds] = stats[t][0]["cross_species_residual"]
            fig_bias.setdefault(t, {})[ds] = stats[t][0]["bias_abs_log2"]
        # sensitivity of the truth to the ambient pool definition: bias of each tool's mean removal under 4 pool definitions
        for pname, pinfo in pool["pools"].items():
            dd = _arrays(df, pool, pinfo["mouse_share"])
            sens.append([ds, pname, round(pinfo["mouse_share"], 4)] + [
                round(float(dd[t]["imp"].mean() / dd["truth"].mean()), 3) if t in dd["tools"] else None for t in TOOLS])
        summary[ds] = {"truth_mean": float(d["truth"].mean()), "n": n}
        if not dev and scatter_rows is None:
            sub = np.random.default_rng(1).choice(n, min(n, 1200), replace=False)
            scatter_rows = (ds, d["truth"][sub], {t: d[t]["imp"][sub] for t in d["tools"]})
        del df

    tissue_rows = []
    for ds, dev, title, acc in TISSUE:
        if not (art / f"{ds}_tissue.tsv.gz").exists():
            continue
        tinfo = json.loads((art / f"{ds}_tissue.json").read_text())
        T = pd.read_csv(art / f"{ds}_tissue.tsv.gz", sep="\t")
        tools = [t for t in TOOLS if f"{t}_after" in T and T[f"{t}_after"].notna().all()]
        for t in TOOLS:
            if t not in tools and t != "singlet":
                caveats.append(f"{TOOLS[t]} produced no usable output on {ds}; it is absent from that dataset's comparison.")
        T = T[T[[f"{t}_after" for t in tools]].notna().all(axis=1)].reset_index(drop=True)
        n = len(T)
        w, truth = T.weight.values.astype(float), T.rho_truth.values.astype(float)
        imp = {t: 1 - T[f"{t}_after"].values / np.maximum(T.n_umi.values, 1) for t in tools}
        ctype = T.celltype.fillna("unlabelled").values
        groups = [g for g, c in pd.Series(ctype).value_counts().items() if c >= 100]
        gidx = {g: (ctype == g) for g in groups}

        def tvals(t, B):
            x = imp[t]
            out = {"tissue_rho_spearman": _spearman_boot(x, truth, B)}
            W = w[B]
            tm = (W * truth[B]).sum(1) / W.sum(1)
            em = (W * x[B]).sum(1) / W.sum(1)
            out["tissue_bias_abs_log2"] = np.abs(np.log2(np.maximum(em, 1e-12) / tm))
            gs = []
            for g in groups:
                m = gidx[g][B]  # cells of this type in each resample
                wg = W * m
                den = np.maximum(wg.sum(1), 1e-12)
                gt = (wg * truth[B]).sum(1) / den
                ge = (wg * x[B]).sum(1) / den
                gs.append(np.abs(np.log2(np.maximum(ge, 1e-12) / np.maximum(gt, 1e-12))))
            out["tissue_celltype_calibration"] = np.mean(gs, axis=0)
            return out
        Bt = C.boot_indices(n, (CAP, ds))
        ident = np.arange(n)[None, :]
        st = {t: ({k: float(v[0]) for k, v in tvals(t, ident).items()}, tvals(t, Bt)) for t in tools}
        datasets.append({
            "id": ds, "data_kind": "real", "title": title, "accessions": acc, "n_cells": int(n),
            "truth": "Donor genotypes: in a confident demuxlet singlet, reads carrying an allele the cell's donor lacks came from other donors' RNA via the ambient pool; rho = (foreign reads - error*reads) / (reads weighted by the pool's allele frequency).",
            "notes": ("DEVELOPMENT DATA: Singlet's ambient estimator was developed while looking at this lane (PR #36). " if dev else "")
                     + f"{n} confident singlets; sequencing-error rate {tinfo['error_rate']:.4f}; median {tinfo['median_informative_reads']:.0f} informative reads per cell."})
        competitors = [t for t in tools if t != "singlet"]
        names = {"tissue_rho_spearman": ("Real-tissue per-cell contamination rank correlation (Spearman) with genotype truth", "correlation", "higher", "accuracy"),
                 "tissue_bias_abs_log2": ("Real-tissue mean contamination removed vs genotype truth, |log2 ratio|", "log2", "lower", "accuracy"),
                 "tissue_celltype_calibration": ("Real-tissue calibration by cell type (mean |log2 ratio| over cell types with >= 100 cells)", "log2", "lower", "robustness")}
        for mid, (name, unit, direction, cat) in names.items():
            comps = [{"tool": TOOLS[t], "value": C.r6(st[t][0][mid]), "ci": C.ci95(st[t][1][mid])} for t in competitors]
            vals = {c["tool"]: c["value"] for c in comps}
            best_name = (max if direction == "higher" else min)(vals, key=vals.get)
            best_t = next(t for t in competitors if TOOLS[t] == best_name)
            tol = C.half_width(st["singlet"][1][mid] - st[best_t][1][mid])
            metrics.append({
                "id": mid, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
                "singlet": {"value": C.r6(st["singlet"][0][mid]), "ci": C.ci95(st["singlet"][1][mid])}, "competitors": comps,
                "rule": {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(tol, 6)},
                         "rationale": f"half-width of the paired bootstrap 95% CI (1000 resamples of the {n} singlets) of the difference between Singlet and the best competitor ({best_name}) on this metric: {tol:.4f}"},
                "notes": "development data. Per-cell genotype truth is noisy (about 170 informative reads per cell), so the grouped metrics are the more robust ones."})
        metrics.append({"id": "n_truth_singlets", "name": "Genotype-truth singlets scored", "unit": "cells", "category": "coverage", "dataset": ds,
                        "direction": "none", "singlet": {"value": n}, "rule": {"type": "informational"}})
        for g in ["all"] + groups:
            m = np.ones(n, bool) if g == "all" else gidx[g]
            gt = float(np.average(truth[m], weights=w[m]))
            tissue_rows.append([ds, g, int(m.sum()), round(100 * gt, 2)] + [
                round(100 * float(np.average(imp[t][m], weights=w[m])), 2) if t in tools else None for t in TOOLS])
        figures.append({"id": "tissue_calibration", "title": f"Real-tissue calibration by cell type, {ds} (% of UMIs)", "kind": "table",
                        "caption": "Weighted mean contamination from genotype truth against the mean fraction of UMIs each tool removed, per cell type.",
                        "columns": ["dataset", "group", "cells", "genotype truth %"] + [TOOLS[t] for t in TOOLS], "rows": tissue_rows})
        caveats.append(f"{ds}: the genotype truth sees only contamination from other donors' RNA, not from cell types of the same donor, so it underestimates total ambient RNA; and the pool's allele frequencies are measured on the pool, so truth and estimates share the same droplets. Per-cell genotype truth has split-half reliability around 0.4, which caps per-cell correlations.")

    tool_names =[TOOLS[t] for t in TOOLS if any(m for m in metrics if any(c["tool"] == TOOLS[t] for c in m.get("competitors", [])))]
    held = [ds for ds, dev, _ in present if not dev]
    figures.append({"id": "spearman_by_dataset", "title": "Per-cell contamination estimate: Spearman correlation with truth", "kind": "grouped_bar",
                    "caption": "Implied contamination (fraction of a cell's UMIs removed by the tool) against truth, pure cells. hgmm1k_v3 is development data.",
                    "x": {"label": "dataset", "categories": [p[0] for p in present]},
                    "y": {"label": "Spearman", "unit": "correlation", "min": 0, "max": 1},
                    "series": [{"name": TOOLS[t], "tool": TOOLS[t], "x": [p[0] for p in present if p[0] in v], "y": [round(v[p[0]], 4) for p in present if p[0] in v]}
                               for t, v in fig_sp.items()]})
    figures.append({"id": "residual_by_dataset", "title": "Cross-species UMIs left after correction", "kind": "grouped_bar",
                    "caption": "Fraction of the other species' UMIs remaining in pure cells; lower is better. Some of the residual is real cross-species misassignment, so no tool reaches zero.",
                    "x": {"label": "dataset", "categories": [p[0] for p in present]},
                    "y": {"label": "fraction left", "unit": "fraction", "min": 0, "max": 1},
                    "series": [{"name": TOOLS[t], "tool": TOOLS[t], "x": [p[0] for p in present if p[0] in v], "y": [round(v[p[0]], 4) for p in present if p[0] in v]}
                               for t, v in fig_res.items()]})
    figures.append({"id": "bias_by_dataset", "title": "Mean contamination removed vs truth", "kind": "grouped_bar",
                    "caption": "|log2(mean removed fraction / mean true contamination)|; 0 is unbiased, 1 is a factor of two off.",
                    "x": {"label": "dataset", "categories": [p[0] for p in present]},
                    "y": {"label": "|log2 ratio|", "unit": "log2", "min": 0},
                    "series": [{"name": TOOLS[t], "tool": TOOLS[t], "x": [p[0] for p in present if p[0] in v], "y": [round(v[p[0]], 4) for p in present if p[0] in v]}
                               for t, v in fig_bias.items()]})
    if scatter_rows:
        ds_s, tr, imps = scatter_rows
        figures.append({"id": "estimate_vs_truth", "title": f"Contamination removed vs truth, {ds_s} (held out)", "kind": "scatter",
                        "caption": "One point per pure cell (random 1,200); both axes are fractions of the cell's UMIs.",
                        "x": {"label": "true contamination (rho)", "unit": "fraction"}, "y": {"label": "fraction of UMIs removed", "unit": "fraction"},
                        "series": [{"name": TOOLS[t], "tool": TOOLS[t], "x": [round(float(v), 5) for v in tr], "y": [round(float(v), 5) for v in imp]}
                                   for t, imp in imps.items()]})
    figures.append({"id": "truth_pool_sensitivity", "title": "Mean removal / mean truth under four ambient-pool definitions", "kind": "table",
                    "caption": "Truth depends on the mouse share of the ambient pool; 1.0 means the tool removes as much as the truth says. Rows differ only in how the pool of empty droplets is defined.",
                    "columns": ["dataset", "pool (UMIs per barcode)", "mouse share"] + [TOOLS[t] for t in TOOLS], "rows": sens})

    resources = []
    for p in sorted(art.glob("resources_*.json")):
        resources += json.loads(p.read_text())
    resources = [r for r in resources if r["dataset"] in {x[0] for x in present}]
    names = TOOLS
    for r in resources:
        r["tool"] = names.get(r["tool"], r["tool"])

    caveats = _caveats(present, sorted(set(caveats)), summary)
    ids = {x[0] for x in present} | {x[0] for x in TISSUE}
    artifacts = []
    for p in sorted(art.iterdir()):
        if not p.is_file() or p.name.startswith("."):
            continue
        ds = next((i for i in ids if p.name.startswith(i + "_") or p.name == f"resources_{i}.json"), None)
        a = {"name": p.name, "url": f"https://data.singlet.bio/benchmarking/1.0.0/{CAP}/{p.name}", "sha256": C.sha256_file(p),
             "bytes": p.stat().st_size, "kind": "derived" if p.suffix == ".json" else "singlet_output", "format": "json" if p.suffix == ".json" else "tsv"}
        if ds:
            a["dataset"] = ds
        artifacts.append(a)
    doc = {
        "schema": contract_schema(), "schema_version": C.contract.SCHEMA_VERSION, "capability": CAP,
        "title": "Ambient RNA",
        "conclusion": _conclusion(metrics, held),
        "methods": METHODS,
        "status": _status(art, present), "generated_at": C.now(), "owner": "pipeline-benchmark",
        "singlet": C.SINGLET, "tools": _tools(), "datasets": datasets, "metrics": metrics, "resources": resources,
        "figures": figures, "artifacts": artifacts, "code": _code(),
        "reproduce": {"module": "bench/public/capabilities/ambient_rna.py", "function": "compute", "laptop_minutes": 3,
                      "clipper_command": "per library ID on Clipper (jobs from bench/public/capabilities): clipper/fetch_10x.sbatch ID TAR_URL; clipper/run_singlet_10x.sbatch ID gex_fastqs.txt; harness/barnyard_tools.sbatch ID HARNESS_DIR (also submits harness/cellbender.sbatch twice); harness/artifacts.sbatch barnyard ID; then python bench/public/capabilities/ambient_rna.py $WORK/bench-public/artifacts/ambient-rna"},
        "caveats": caveats,
    }
    return C.finish(doc)


def _status(art: Path, present) -> str:
    """`final` only when every listed library (barnyard and tissue) has artifacts; otherwise `draft`."""
    have = {x[0] for x in present} | {x[0] for x in TISSUE if (art / f"{x[0]}_tissue.tsv.gz").exists()}
    return "final" if {x[0] for x in DATASETS} | {x[0] for x in TISSUE} <= have else "draft"


def contract_schema() -> str:
    return C.contract.SCHEMA_CAPABILITY


def _conclusion(metrics: list[dict], held: list[str]) -> str:
    g = [m for m in metrics if m["rule"]["type"] != "informational"]
    ev = [(m, C.contract.evaluate_metric(m)) for m in g]
    npass = sum(1 for _, r in ev if r == "pass")

    def pick(mid, ds):
        return next((m for m in g if m["id"] == mid and m["dataset"] == ds), None)

    ds = held[0] if held else g[0]["dataset"]
    sp, rs = pick("rho_spearman", ds), pick("cross_species_residual", ds)
    best = lambda m: max(m["competitors"], key=lambda c: c["value"]) if m["direction"] == "higher" else min(m["competitors"], key=lambda c: c["value"])  # noqa: E731
    parts = [f"Held out on {', '.join(held) if held else 'no dataset'}, Singlet passes {sum(1 for m, r in ev if r == 'pass' and m['dataset'] in held)} of "
             f"{sum(1 for m, r in ev if m['dataset'] in held)} gating ambient-RNA metrics ({npass} of {len(g)} over all libraries)."]
    if sp:
        b = best(sp)
        parts.append(f"On {ds} its per-cell contamination ranking has Spearman {sp['singlet']['value']:.2f} against {b['value']:.2f} for {b['tool']}.")
    if rs:
        b = best(rs)
        parts.append(f"After correction {rs['singlet']['value']*100:.0f}% of cross-species UMIs remain, against {b['value']*100:.0f}% for {b['tool']}.")
    return " ".join(parts)


def _tools() -> list[dict]:
    return [
        {"name": "singlet", "version": "1.0.0", "role": "singlet", "url": "https://github.com/Singlet-Bio/singlet",
         "params": "singlet-campaign s.1fq --ref-base REF --genome-dir GRCh38-GRCm39-barnyard/star_2.7.11b --exons genes.gtf --threads 16 --pipeline --output-format mtx --raw-matrix; ambient_corrected_counts.mtx.gz and ambient_contamination.tsv (method decontx_cluster_offgene_v1); no ambient flag"},
        {"name": "SoupX", "version": "1.6.2", "role": "competitor", "url": "https://github.com/constantAmateur/SoupX",
         "citation": "Young & Bhatt 2020, GigaScience 9:giaa151",
         "params": "SoupChannel(tod, toc); setClusters(sc, Seurat default clusters); autoEstCont(sc) (contaminationRange c(0.01, 0.8), all else default); adjustCounts(sc) (subtraction, not rounded). tod = all barcodes with >= 1 UMI from Singlet's raw matrix, toc = the called cells"},
        {"name": "DecontX", "version": "celda 1.22.0", "role": "competitor", "url": "https://bioconductor.org/packages/celda",
         "citation": "Yang et al. 2020, Genome Biology 21:57", "params": "decontX(toc) with its own clustering, set.seed(1); no background"},
        {"name": "DecontX-background", "version": "celda 1.22.0", "role": "competitor", "url": "https://bioconductor.org/packages/celda",
         "citation": "Yang et al. 2020, Genome Biology 21:57",
         "params": "decontX(toc, background = empty) with empty = 20,000 random barcodes of 10-100 UMI, set.seed(1)"},
        {"name": "CellBender", "version": "0.3.2", "role": "competitor", "url": "https://github.com/broadinstitute/CellBender",
         "citation": "Fleming et al. 2023, Nature Methods 20:1323",
         "params": "cellbender remove-background --input RAW_DIR --output cb.h5 --cuda (documented minimal call: expected cells and droplets estimated by the tool, 150 epochs, fpr 0.01), apptainer image cellbender-0.3.2.sif, one V100S GPU"},
        {"name": "CellBender-recommended", "version": "0.3.2", "role": "competitor", "url": "https://github.com/broadinstitute/CellBender",
         "citation": "Fleming et al. 2023, Nature Methods 20:1323",
         "params": "cellbender remove-background --input RAW_DIR --output cb.h5 --cuda --expected-cells N --total-droplets-included 6N, N = number of cells Singlet called (the authors' tutorial asks for both); otherwise defaults"},
        {"name": "Seurat", "version": "5.5.1", "role": "helper", "url": "https://satijalab.org/seurat/",
         "params": "NormalizeData, FindVariableFeatures, ScaleData, RunPCA, FindNeighbors, FindClusters at defaults: the clusters SoupX needs"},
        {"name": "barnyard-species-truth", "version": "n/a", "role": "truth",
         "params": "rho = other-species UMI fraction / other species' share of the ambient pool (barcodes with 1-100 UMI, top 50,000), pure cells only (other species <= 5%)"},
    ]


def _code() -> list[dict]:
    sha = _code_commit()
    base = f"{C.REPO_URL}/blob/{sha}/bench/public/capabilities"
    rows = [("ambient_rna.py", "metric"), ("_adcommon.py", "metric"), ("harness/barnyard_prep.py", "harness"), ("harness/barnyard_tools.R", "harness"),
            ("harness/barnyard_tools.sbatch", "harness"), ("harness/cellbender.sbatch", "harness"), ("harness/barnyard_artifacts.py", "harness"),
            ("clipper/run_singlet_10x.sbatch", "harness"), ("clipper/fetch_10x.sbatch", "harness")]
    out = [{"path": f"bench/public/capabilities/{p}", "url": f"{base}/{p}", "role": r} for p, r in rows]
    out.append({"path": "include/singlet/pileup/ambient_correction.h", "role": "implementation",
                "url": f"{C.REPO_URL}/blob/{C.SINGLET['commit']}/include/singlet/pileup/ambient_correction.h"})
    return out


def _code_commit() -> str:
    p = Path(__file__).resolve().parent / "CODE_COMMIT"
    return p.read_text().strip() if p.exists() else C.SINGLET["commit"]


def _caveats(present, extra, summary) -> list[str]:
    dev = [ds for ds, d, _ in present if d]
    c = [
        f"{', '.join(dev)} is development data: Singlet's gene-level ambient estimator (PR #36) was built and tuned while looking at it. The headline numbers are the held-out libraries.",
        "Truth assumes every other-species UMI in a pure cell is ambient and that the ambient pool has the same species composition for every droplet. Cross-species doublet-like barcodes and index hopping would also contribute. The pool definition moves truth by up to about 20%; see the truth_pool_sensitivity table.",
        "Per-cell truth is Poisson-noisy at 1% contamination, so per-cell correlations have a ceiling well below 1; the bias and residual metrics are the more robust columns.",
        "All three libraries are mixtures of two cell lines. They say nothing about tissues where ambient RNA resembles the cell types themselves, or about contamination between members of the same species.",
        "Every tool is scored on Singlet's called cells (called_v2) and Singlet's raw matrix; tools with their own cell calling or counts were not given them. CellBender at its documented defaults chooses its own droplet range from the same raw matrix.",
        "SoupX uses its default contaminationRange (0.01 to 0.8) and Seurat default clusters; its single global rho is applied to every cell, so its per-cell ranking comes only from how the subtraction distributes over cells.",
        "Competitors' own estimates are compared through the same observable: the fraction of a cell's UMIs they removed (implied rho). Singlet's native rho (exon scale) is in the artifact tables but is not the scored quantity.",
        "Resource rows compare Singlet's whole pipeline (alignment, counting and everything else) with a single competitor step; they are not a like-for-like speed comparison and are not gating.",
    ]
    return c + extra


METHODS = """**Data.** Three public 10x libraries of 1:1 mixed HEK293T (human) and NIH3T3 (mouse) cells: hgmm1k_v3 (development data), and hgmm10k_v3 and hgmm1k_v2 (held out, the latter in 3' v2 chemistry). All tools see Singlet's counts: the raw gene-by-droplet matrix from the joint GRCh38-2024-A + GRCm39 reference, and the cells Singlet called (called_v2).

**Truth.** A cell whose reads are at most 5% from the minor species is pure. In a pure cell all other-species UMIs are ambient, so the true contamination is rho = (other-species fraction of the cell) / (other species' share of the ambient pool). The pool is the barcodes with 1-100 UMI (top 50,000 by count); three other pool definitions are reported as sensitivity.

**Tools.** Singlet 1.0.0 at pipeline defaults (gene-level, cluster-aware off-gene estimator with a Gamma prior; corrected matrix `ambient_corrected_counts`). SoupX, DecontX (with and without the empty-droplet background) and CellBender at their documented defaults; exact commands are in `tools[].params`.

**Metrics.** Every tool is compared through what it removed: implied rho = 1 - (UMIs after / UMIs before) per cell. (1) Spearman correlation between implied rho and truth over pure cells. (2) Bias: |log2| of the ratio of mean removed fraction to mean truth, overall and in the lowest-depth third of cells (a depth-only estimator fails here). (3) Cross-species residual: other-species UMIs after correction over before, summed over pure cells. (4) Own-species gap: |removed own-species UMIs / expected own-species ambient - 1|, where the expectation is inferred from each cell's other-species UMIs and the pool composition; it flags over- or under-correction.

**Pass rule.** Every metric must match or beat the best competitor on that dataset and metric, within the half-width of the paired bootstrap 95% CI (1000 resamples of the pure cells) of the difference between Singlet and that competitor. Nothing is gated on speed. A metric that fails is reported as failing.""".strip()


if __name__ == "__main__":
    sys.exit(C.write_main(compute, __doc__, str(Path(__file__).resolve().parents[1] / "results" / "ambient-rna.json")))
