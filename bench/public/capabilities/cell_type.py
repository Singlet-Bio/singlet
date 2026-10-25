"""Capability ``cell-type``: Singlet's per-cell cell type probabilities (``singlet.celltype_nn``; the published call is the
most specific Cell Ontology term with rolled-up probability >= 0.5) against CellTypist, scANVI, Seurat reference mapping
(the Azimuth method) and SingleR, every tool trained on the same CELLxGENE Census reference, scored against curated author
labels on Singlet's counts of the same cells.

Artifacts (``artifact_dir``):
- ``celltype_calls.parquet``: one row per tool and cell: tool, gse, gsm, barcode, pred_cl (CL id; null = no call), score
  (the tool's confidence), prob (a probability, when the tool gives one), pruned (SingleR's pruned label).
- ``celltype_truth.parquet``: gse, gsm, barcode, species, truth (CL id from the CELLxGENE curators, transferred by barcode).
- ``provenance.json``: Singlet commit, tool versions and settings, datasets, resources, the reference's tissue coverage per
  study, methods text, caveats.

Scoring is :mod:`singlet.celltype_nn.benchmark` (the same code as the harness's ``scripts/celltype_nn/compete/score.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import artifact_entries, code_url, contract, load_provenance, now_iso, r

CAP = "cell-type"
DATASET = "cxg16"
SINGLET = "singlet"
COMPETITORS = ["CellTypist", "scANVI", "Seurat", "SingleR"]
HELPERS = ["singlet-guide", "singlet-argmax", "singlet-threshold-0.3"]  # shown in figures, not compared by the rules
PROB_TOOLS = ["CellTypist", "scANVI", "Seurat"]  # tools whose score is a probability (SingleR's delta is not)


def compute(artifact_dir: Path) -> dict:
    import pandas as pd

    from singlet.celltype_nn.benchmark import bootstrap_studies, mean_over_studies, score_tool

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    calls = pd.read_parquet(artifact_dir / "celltype_calls.parquet")
    truth = pd.read_parquet(artifact_dir / "celltype_truth.parquet")
    tools = [t for t in [SINGLET] + COMPETITORS + HELPERS if (calls["tool"] == t).any()]
    res = {t: score_tool(calls[calls["tool"] == t].drop(columns="tool"), truth) for t in tools}
    per = {t: res[t]["per_study"] for t in tools}
    seen = prov["reference_tissue"]
    unseen = [g for g, v in seen.items() if not v["seen"]]
    comps = [t for t in COMPETITORS if t in res]

    metrics = []

    def add(mid, name, unit, category, direction, key=None, values=None, studies=None, against=None, notes=None, with_ci=True):
        against = [t for t in (against or comps) if t in res]
        if values is None:
            values = {t: mean_over_studies(per[t], key, studies) for t in [SINGLET] + against}
        m = {"id": mid, "name": name, "unit": unit, "category": category, "dataset": DATASET, "direction": direction,
             "singlet": {"value": r(values[SINGLET])},
             "competitors": [{"tool": t, "value": r(values[t])} for t in against],
             "rule": {"type": "match_or_beat", "against": against}}
        if key is not None and with_ci and studies is None:
            sub = {t: per[t] for t in [SINGLET] + against}
            ci, diff = bootstrap_studies(sub, key, SINGLET)
            if ci:
                m["singlet"]["ci"] = [r(x) for x in ci[SINGLET]]
                for c in m["competitors"]:
                    c["ci"] = [r(x) for x in ci[c["tool"]]]
                vals = [values[t] for t in against if values[t] is not None]
                if vals:
                    best = max(against, key=lambda t: values[t] if direction == "higher" else -values[t])
                    lo, hi = diff[best]  # CI of (best - singlet) as computed by bootstrap_studies(ref=singlet)
                    half = (hi - lo) / 2
                    m["rule"]["tolerance"] = {"abs": r(half)}
                    m["rule"]["rationale"] = (f"{r(half)} = half-width of the 95% bootstrap CI (16 studies resampled) of the "
                                              f"difference between Singlet and {best}")
        if notes:
            m["notes"] = notes
        metrics.append(m)

    for d, label in zip((1, 2, 3, 4), ("compartment", "lineage", "cell type", "subtype")):
        add(f"accuracy_depth{d}", f"Accuracy at Cell Ontology depth {d} ({label})", "fraction", "accuracy", "higher", key=f"d{d}",
            notes="mean over studies of the fraction of cells whose depth-d ancestor matches the author's; stopping early counts as wrong")
    add("consistent", "Calls consistent with the author's label (same term, an ancestor or a descendant)", "fraction",
        "accuracy", "higher", key="consistent")
    add("wrong_rate", "Calls on another branch than the author's label", "fraction", "accuracy", "lower", key="wrong")
    add("abstention_aurc", "Wrong-call rate integrated over coverage (area under the risk-coverage curve)", "fraction",
        "accuracy", "lower", values={t: res[t]["aurc"] for t in [SINGLET] + comps},
        notes="cells ranked by each tool's own confidence; lower means its confidence separates right from wrong calls better")
    add("calibration_ece", "Expected calibration error of the call's probability", "fraction", "accuracy", "lower",
        values={t: res[t].get("ece") for t in [SINGLET] + PROB_TOOLS if t in res}, against=PROB_TOOLS,
        notes="10 bins; a call is right when the truth lies in its subtree; SingleR gives no probability")
    if unseen:
        add("unseen_tissue_depth2", "Accuracy at depth 2 on studies whose tissue is thin in the reference", "fraction",
            "robustness", "higher", key="d2", studies=unseen,
            notes=f"{len(unseen)} studies with fewer than {prov['seen_cells_threshold']} reference cells in their tissue group: "
                  + ", ".join(sorted(unseen)))
    rt = prov.get("runtime", {})
    if rt.get(SINGLET):
        add("cpu_seconds_per_million_cells", "CPU time to label a million cells (prediction, reference prepared)", "s",
            "resources", "lower", values={t: rt.get(t) for t in [SINGLET] + comps}, against=[t for t in comps if rt.get(t) is not None],
            notes="wall time x threads on Clipper CPU nodes; scANVI's query mapping runs on one H100 GPU and is counted as wall time x 1")

    # ---- figures
    figs = []
    shown = [t for t in [SINGLET] + comps + HELPERS if t in res]
    cats = ["d1", "d2", "d3", "d4"]
    figs.append({"id": "accuracy_by_depth", "title": "Accuracy by Cell Ontology depth (mean over 16 studies)", "kind": "grouped_bar",
                 "x": {"label": "depth", "categories": cats}, "y": {"label": "accuracy", "unit": "fraction", "min": 0, "max": 1},
                 "series": [{"name": t, "tool": t, "x": cats, "y": [r(mean_over_studies(per[t], k)) for k in cats]} for t in shown]})
    figs.append({"id": "accuracy_vs_wrong", "title": "Lineage accuracy against wrong calls", "kind": "scatter",
                 "x": {"label": "wrong calls", "unit": "fraction"}, "y": {"label": "accuracy at depth 2", "unit": "fraction"},
                 "series": [{"name": t, "tool": t, "x": [r(mean_over_studies(per[t], "wrong"))], "y": [r(mean_over_studies(per[t], "d2"))]}
                            for t in shown]})
    figs.append({"id": "reliability", "title": "Reliability of the call's probability", "kind": "line",
                 "x": {"label": "stated probability", "unit": "fraction", "min": 0, "max": 1},
                 "y": {"label": "observed accuracy", "unit": "fraction", "min": 0, "max": 1},
                 "series": [{"name": t, "tool": t, "x": [r(b["mean_prob"]) for b in res[t]["reliability"]],
                             "y": [r(b["accuracy"]) for b in res[t]["reliability"]]} for t in [SINGLET] + PROB_TOOLS if t in res and res[t].get("reliability")]})
    studies = sorted(per[SINGLET])
    rows = []
    for g in studies:
        row = [g, seen.get(g, {}).get("tissue", ""), per[SINGLET][g]["n"]]
        for t in [SINGLET] + comps + ["singlet-guide"]:
            if t in per and g in per[t]:
                p = per[t][g]
                row.append("/".join("-" if p[f"d{d}"] is None else f"{p[f'd{d}']:.2f}" for d in (1, 2, 3)) + f" ({p['wrong']:.2f})")
            else:
                row.append(None)
        rows.append(row)
    figs.append({"id": "per_study", "title": "Per study: accuracy at depth 1/2/3 (wrong calls)", "kind": "table",
                 "columns": ["study", "tissue", "cells"] + [t for t in [SINGLET] + comps + ["singlet-guide"] if t in per], "rows": rows})

    commit = prov["bench_commit"]
    s, best = metrics[1], None
    d2 = {c["tool"]: c["value"] for c in s["competitors"]}
    conclusion = prov.get("conclusion") or (
        f"On 16 held-out studies with curated cell types ({len(truth):,} cells), Singlet's network gets "
        f"{s['singlet']['value']:.1%} of cells to the right lineage against " + ", ".join(f"{v:.1%} for {k}" for k, v in d2.items())
        + f", with {metrics[5]['singlet']['value']:.1%} wrong calls; every tool learned from the same CELLxGENE reference.")
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Cell type labels",
        "conclusion": conclusion,
        "methods": prov["methods"],
        "status": prov.get("status", "draft"),
        "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "ensemble-classifier",
        "singlet": prov["singlet"],
        "tools": prov["tools"],
        "datasets": prov["datasets"],
        "metrics": metrics,
        "resources": prov["resources"],
        "figures": figs,
        "artifacts": artifact_entries(artifact_dir, CAP, {"celltype_calls.parquet": "derived", "celltype_truth.parquet": "truth",
                                                          "provenance.json": "derived"}),
        "code": [
            {"path": "python/singlet/celltype_nn/predict.py", "role": "implementation", "url": code_url("python/singlet/celltype_nn/predict.py", commit)},
            {"path": "python/singlet/celltype_nn/model.py", "role": "implementation", "url": code_url("python/singlet/celltype_nn/model.py", commit)},
            {"path": "scripts/celltype_nn/train.py", "role": "implementation", "url": code_url("scripts/celltype_nn/train.py", commit)},
            {"path": "python/singlet/celllabels/celltype.py", "role": "implementation", "url": code_url("python/singlet/celllabels/celltype.py", commit)},
            {"path": "scripts/celltype_nn/compete/run_py.py", "role": "harness", "url": code_url("scripts/celltype_nn/compete/run_py.py", commit)},
            {"path": "scripts/celltype_nn/compete/run_r.R", "role": "harness", "url": code_url("scripts/celltype_nn/compete/run_r.R", commit)},
            {"path": "scripts/celltype_nn/compete/export.py", "role": "harness", "url": code_url("scripts/celltype_nn/compete/export.py", commit)},
            {"path": "python/singlet/celltype_nn/benchmark.py", "role": "metric", "url": code_url("python/singlet/celltype_nn/benchmark.py", commit)},
            {"path": "bench/public/capabilities/cell_type.py", "role": "metric", "url": code_url("bench/public/capabilities/cell_type.py", commit)},
        ],
        "reproduce": {"module": "bench/public/capabilities/cell_type.py", "function": "compute", "laptop_minutes": 10,
                      "clipper_command": "see docs/celltype_nn.md section 9 (scripts/celltype_nn/compete/*.sbatch)"},
        "caveats": prov.get("caveats", []),
    }
    return C.finalize(doc)
