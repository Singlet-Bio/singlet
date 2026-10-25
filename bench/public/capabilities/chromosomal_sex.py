"""Capability ``chromosomal-sex``: Singlet's per-cell chromosomal sex (singlet.celllabels.sex) against cellXY and the
common XIST/chrY rule, on the same counts, scored against sex recorded in GEO metadata and against known donors.

Artifacts (``artifact_dir``):
- ``sex_cells.parquet``: one row per cell. Columns: dataset, gse, gsm, species, col, donor, truth_donor_sex (when a cell's
  donor is known, e.g. in-silico pools), total_umi, n_genes, xist, y_singlet, super_x, super_y, singlet_label,
  cellxy_qc, cellxy_noqc. Labels are female, male, ambiguous, or null for no call.
- ``sex_samples.tsv``: dataset, gse, gsm, species, truth (GEO metadata sex, or 'mixed' for a pool), n_donors.
- ``provenance.json``: Singlet commit and binary, tool versions, resources per tool, the commit of this code.

The XIST/chrY rule is computed here from the counts: female when XIST > 0 and no MSY UMI, male when MSY > 0 and XIST = 0,
otherwise no call. It uses cellXY's MSY gene set (``super_y``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import artifact_entries, bootstrap_ci, code_url, contract, load_provenance, now_iso, r

CAP = "chromosomal-sex"
TOOLS = {"singlet": "singlet_label", "cellXY": "cellxy_qc", "xist-chrY-rule": "rule"}


def _rule(df):
    x, y = df["xist"].to_numpy(), df["super_y"].to_numpy()
    out = np.full(len(df), None, dtype=object)
    out[(x > 0) & (y == 0)] = "female"
    out[(y > 0) & (x == 0)] = "male"
    return out


def _norm(v):
    """Map each tool's vocabulary to female / male / None (no call)."""
    v = np.asarray(v, dtype=object)
    out = np.full(len(v), None, dtype=object)
    s = np.array([str(x).lower() if x is not None else "" for x in v])
    out[s == "female"] = "female"
    out[s == "male"] = "male"
    return out


def _sample_call(labels: np.ndarray) -> str | None:
    """Majority of a sample's labelled cells (the way a user summarises per-cell calls); None if none labelled or tied."""
    f, m = int((labels == "female").sum()), int((labels == "male").sum())
    return None if f == m else ("female" if f > m else "male")


def compute(artifact_dir: Path) -> dict:
    import pandas as pd

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    cells = pd.read_parquet(artifact_dir / "sex_cells.parquet")
    samples = pd.read_csv(artifact_dir / "sex_samples.tsv", sep="\t")
    cells["rule"] = _rule(cells)
    calls = {t: _norm(cells[c].to_numpy(object)) for t, c in TOOLS.items()}
    # Singlet's own sample call is the donor group's call (the label most of its cells inherit), i.e. the same majority.

    metrics, figures = [], []
    comp_names = [t for t in TOOLS if t != "singlet"]

    def metric(mid, name, unit, category, dataset, vals, cis=None, rule=None, notes=None):
        m = {"id": mid, "name": name, "unit": unit, "category": category, "dataset": dataset, "direction": "higher",
             "singlet": {"value": r(vals["singlet"])},
             "competitors": [{"tool": t, "value": r(vals[t])} for t in comp_names],
             "rule": rule or {"type": "match_or_beat", "against": "best"}}
        if cis:
            m["singlet"]["ci"] = [r(x) for x in cis["singlet"]]
            for c in m["competitors"]:
                c["ci"] = [r(x) for x in cis[c["tool"]]]
        if notes:
            m["notes"] = notes
        metrics.append(m)

    # ---- GEO-metadata cohort: single-sex samples, truth = the sample's recorded sex
    geo = samples[samples["truth"].isin(["female", "male"])]
    for ds in sorted(geo["dataset"].unique()):
        sm = geo[geo["dataset"] == ds]
        sel = cells["gsm"].isin(sm["gsm"]).to_numpy()
        truth_cell = cells.loc[sel, "gsm"].map(dict(zip(sm["gsm"], sm["truth"]))).to_numpy(object)
        gsm_cell = cells.loc[sel, "gsm"].to_numpy(object)
        acc_s, cov_s, acc_all, prec, cov = {}, {}, {}, {}, {}
        ci_all, ci_prec, ci_cov = {}, {}, {}
        sample_rows = []
        for t in TOOLS:
            lab = calls[t][sel]
            labelled = lab != None  # noqa: E711
            right = (lab == truth_cell)
            per = pd.DataFrame({"gsm": gsm_cell, "lab": lab})
            sc = per.groupby("gsm")["lab"].apply(lambda v: _sample_call(v.to_numpy(object)))
            st = sm.set_index("gsm")["truth"]
            sc = sc.reindex(st.index)
            acc_s[t] = float((sc == st).mean())
            cov_s[t] = float(sc.notna().mean())
            acc_all[t] = float(right.mean())
            prec[t] = float(right[labelled].mean()) if labelled.any() else float("nan")
            cov[t] = float(labelled.mean())
            ci_all[t] = bootstrap_ci(gsm_cell, right.astype(float), np.ones(len(lab)))
            ci_prec[t] = bootstrap_ci(gsm_cell, (right & labelled).astype(float), labelled.astype(float))
            ci_cov[t] = bootstrap_ci(gsm_cell, labelled.astype(float), np.ones(len(lab)))
            sample_rows.append((t, sc))
        n_s, n_c = len(sm), int(sel.sum())
        metric("sample_accuracy", "Samples whose call matches the GEO metadata", "fraction", "accuracy", ds, acc_s,
               notes=f"{n_s} samples; a tool's sample call is the majority of its labelled cells (Singlet: the donor call)")
        metric("sample_called", "Samples given a call", "fraction", "coverage", ds, cov_s)
        metric("cell_accuracy", "Cells labelled with their sample's recorded sex (no call counts as wrong)", "fraction",
               "accuracy", ds, acc_all, ci_all, notes=f"{n_c} cells; 95% CI: bootstrap over samples")
        metric("cell_precision", "Labelled cells that match their sample's recorded sex", "fraction", "accuracy", ds,
               prec, ci_prec)
        metric("cell_coverage", "Cells given a label", "fraction", "coverage", ds, cov, ci_cov)
        figures.append({
            "id": f"{ds}_cells", "title": "Per-cell accuracy and coverage, GEO-metadata samples", "kind": "grouped_bar",
            "x": {"label": "metric", "categories": ["cell_accuracy", "cell_precision", "cell_coverage"]},
            "y": {"label": "fraction", "unit": "fraction", "min": 0, "max": 1},
            "series": [{"name": t, "tool": t, "x": ["cell_accuracy", "cell_precision", "cell_coverage"],
                        "y": [r(acc_all[t]), r(prec[t]), r(cov[t])],
                        "err_lo": [r(ci_all[t][0]), r(ci_prec[t][0]), r(ci_cov[t][0])],
                        "err_hi": [r(ci_all[t][1]), r(ci_prec[t][1]), r(ci_cov[t][1])]} for t in TOOLS]})
        # per-sample table: each tool's sample call and the fraction of cells it labels correctly
        rows = []
        for g, tr in sm[["gsm", "truth"]].itertuples(index=False):
            k = gsm_cell == g
            row = [g, sm.loc[sm["gsm"] == g, "gse"].iloc[0], tr, int(k.sum())]
            for t, sc in sample_rows:
                row += [sc.get(g), r(float((calls[t][sel][k] == tr).mean()) if k.any() else None, 3)]
            rows.append(row)
        figures.append({"id": f"{ds}_samples", "title": "Per-sample calls", "kind": "table",
                        "columns": ["gsm", "gse", "GEO sex", "cells"] + sum(([f"{t} call", f"{t} cell acc."] for t in TOOLS), []),
                        "rows": rows})

    # ---- pools with per-cell donor truth (in-silico mixes of known-sex donors)
    pool = cells[cells["truth_donor_sex"].notna()] if "truth_donor_sex" in cells else cells.iloc[:0]
    for ds in sorted(pool["dataset"].unique()) if len(pool) else []:
        sel = (cells["dataset"] == ds).to_numpy() & cells["truth_donor_sex"].notna().to_numpy()
        tr = cells.loc[sel, "truth_donor_sex"].to_numpy(object)
        acc, prec, cov = {}, {}, {}
        for t in TOOLS:
            lab = calls[t][sel]
            lab_ok = lab != None  # noqa: E711
            acc[t] = float((lab == tr).mean())
            prec[t] = float((lab == tr)[lab_ok].mean()) if lab_ok.any() else float("nan")
            cov[t] = float(lab_ok.mean())
        metric("cell_accuracy", "Cells labelled with their donor's sex (no call counts as wrong)", "fraction", "accuracy",
               ds, acc, notes=f"{int(sel.sum())} cells of known donors")
        metric("cell_precision", "Labelled cells that match their donor's sex", "fraction", "accuracy", ds, prec)
        metric("cell_coverage", "Cells given a label", "fraction", "coverage", ds, cov)

    # ---- descriptive
    metrics.append({"id": "n_cells", "name": "Cells scored", "unit": "cells", "category": "coverage", "dataset": "pooled",
                    "direction": "none", "singlet": {"value": int(len(cells))}, "rule": {"type": "informational"}})
    metrics.append({"id": "n_samples", "name": "Samples scored", "unit": "samples", "category": "coverage",
                    "dataset": "pooled", "direction": "none", "singlet": {"value": int(len(samples))},
                    "rule": {"type": "informational"}})

    commit = prov["bench_commit"]
    acc_best = {m["dataset"]: m for m in metrics if m["id"] == "cell_accuracy"}
    first = next(iter(acc_best.values()), None)
    conclusion = prov.get("conclusion") or (
        f"On {len(samples)} samples with sex in their GEO metadata ({len(cells):,} cells), Singlet labels "
        f"{first['singlet']['value']:.1%} of cells with their sample's recorded sex, against "
        + ", ".join(f"{c['value']:.1%} for {c['tool']}" for c in first["competitors"]) + "."
        if first else "No data.")
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Chromosomal sex per cell",
        "conclusion": conclusion,
        "methods": prov["methods"],
        "status": prov.get("status", "draft"),
        "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "bench-public",
        "singlet": prov["singlet"],
        "tools": prov["tools"],
        "datasets": prov["datasets"],
        "metrics": metrics,
        "resources": prov["resources"],
        "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {"sex_cells.parquet": "derived", "sex_samples.tsv": "truth",
                                                          "provenance.json": "derived"}),
        "code": [
            {"path": "python/singlet/celllabels/sex.py", "role": "implementation",
             "url": code_url("python/singlet/celllabels/sex.py", commit)},
            {"path": "bench/public/sexancestry/sex_extract.py", "role": "harness",
             "url": code_url("bench/public/sexancestry/sex_extract.py", commit)},
            {"path": "bench/public/sexancestry/cellxy.R", "role": "harness",
             "url": code_url("bench/public/sexancestry/cellxy.R", commit)},
            {"path": "bench/public/capabilities/chromosomal_sex.py", "role": "metric",
             "url": code_url("bench/public/capabilities/chromosomal_sex.py", commit)},
        ],
        "reproduce": {"module": "bench/public/capabilities/chromosomal_sex.py", "function": "compute",
                      "laptop_minutes": 2,
                      "clipper_command": "bash bench/public/sexancestry/submit_sex.sh"},
        "caveats": prov.get("caveats", []),
    }
    return C.finalize(doc)
