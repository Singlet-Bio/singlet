"""Capability ``genetic-ancestry``: Singlet's per-donor genetic ancestry (singlet.celllabels.ancestry) against somalier and
peddy on the same alignments, scored against the 1000 Genomes super-population of known individuals. Also scores donor-level
chromosomal sex from the same RNA (bulk), which the ``chromosomal-sex`` document cites.

Artifacts (``artifact_dir``):
- ``ancestry_calls.tsv``: dataset, individual, level (million read pairs), tool, call (the tool's published label: a super-
  population, 'admixed' for Singlet below q = 0.8, or empty), top (the most likely super-population), q_max, n_sites,
  population, super_population (truth), sex (truth).
- ``sex_bulk_calls.tsv``: dataset, individual, tool, call, sex_truth.
- ``provenance.json``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from _common import artifact_entries, code_url, contract, load_provenance, now_iso, r

CAP = "genetic-ancestry"
SUPERS = ["AFR", "AMR", "EAS", "EUR", "SAS"]


def _wilson(k: int, n: int):
    if n == 0:
        return [None, None]
    z, p = 1.959964, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [c - h, c + h]


def compute(artifact_dir: Path) -> dict:
    import pandas as pd

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    calls = pd.read_csv(artifact_dir / "ancestry_calls.tsv", sep="\t", dtype={"level": str})
    tools = [t for t in ["singlet", "somalier", "peddy"] if t in set(calls["tool"])]
    comps = [t for t in tools if t != "singlet"]
    metrics, figures = [], []

    def add(mid, name, unit, category, dataset, vals, cis=None, notes=None, rule=None):
        m = {"id": mid, "name": name, "unit": unit, "category": category, "dataset": dataset, "direction": "higher",
             "singlet": {"value": r(vals.get("singlet"))},
             "competitors": [{"tool": t, "value": r(vals.get(t))} for t in comps],
             "rule": rule or {"type": "match_or_beat", "against": "best"}}
        if cis:
            m["singlet"]["ci"] = [r(x) for x in cis["singlet"]]
            for c in m["competitors"]:
                c["ci"] = [r(x) for x in cis[c["tool"]]]
        if notes:
            m["notes"] = notes
        metrics.append(m)

    for ds in sorted(calls["dataset"].unique()):
        d0 = calls[calls["dataset"] == ds]
        levels = sorted(d0["level"].unique(), key=float)
        fig_series = {t: [] for t in tools}
        for L in levels:
            d = d0[d0["level"] == L]
            inds = sorted(set(d["individual"]))
            n = len(inds)
            acc_pub, acc_top, prec, cov, cis_pub, cis_top = {}, {}, {}, {}, {}, {}
            for t in tools:
                x = d[d["tool"] == t].set_index("individual").reindex(inds)
                truth = d.drop_duplicates("individual").set_index("individual").reindex(inds)["super_population"]
                pub_ok = (x["call"] == truth).to_numpy()
                top_ok = (x["top"] == truth).to_numpy()
                named = x["call"].isin(SUPERS).to_numpy()
                acc_pub[t], acc_top[t] = pub_ok.mean(), top_ok.mean()
                prec[t] = pub_ok[named].mean() if named.any() else float("nan")
                cov[t] = named.mean()
                cis_pub[t], cis_top[t] = _wilson(int(pub_ok.sum()), n), _wilson(int(top_ok.sum()), n)
                fig_series[t].append(r(acc_pub[t]))
            tag = f"{L}m"
            add(f"accuracy.{tag}", f"Individuals whose published label is their 1000 Genomes super-population ({L}M read pairs)",
                "fraction", "accuracy", ds, acc_pub, cis_pub,
                notes=f"{n} individuals; Singlet's label is 'admixed' below q = 0.8, which counts as wrong here; 95% Wilson CI")
            add(f"accuracy_top.{tag}", f"Most likely super-population correct ({L}M read pairs)", "fraction", "accuracy", ds,
                acc_top, cis_top, notes="argmax of each tool's ancestry proportions or probabilities; no abstention")
            add(f"named_precision.{tag}", f"Named calls that are correct ({L}M read pairs)", "fraction", "accuracy", ds, prec)
            # per super-population accuracy at the deepest level
            if L == levels[-1]:
                rows = []
                for sp in SUPERS:
                    row = [sp]
                    for t in tools:
                        x = d[(d["tool"] == t) & (d["super_population"] == sp)]
                        row.append(r((x["call"] == sp).mean(), 3) if len(x) else None)
                    rows.append(row)
                figures.append({"id": f"{ds}_by_superpop", "title": f"Accuracy by super-population ({L}M read pairs)",
                                "kind": "table", "columns": ["super-population"] + tools, "rows": rows})
                cm = d[d["tool"] == "singlet"]
                cols = SUPERS + ["admixed", "no call"]
                z = [[int(((cm["super_population"] == sp) & (cm["call"].fillna("no call").replace({"": "no call"}) == c)).sum())
                      for c in cols] for sp in SUPERS]
                figures.append({"id": f"{ds}_confusion_singlet", "title": f"Singlet confusion ({L}M read pairs; rows truth)",
                                "kind": "heatmap", "rows": SUPERS, "cols": cols, "z": z})
        figures.insert(0, {"id": f"{ds}_accuracy_by_depth", "title": "Accuracy of the published label by depth", "kind": "line",
                           "x": {"label": "read pairs (millions)", "scale": "log"}, "y": {"label": "accuracy", "min": 0, "max": 1},
                           "series": [{"name": t, "tool": t, "x": [float(L) for L in levels], "y": fig_series[t]} for t in tools]})

    # ---- end-to-end single-cell pools of 1000 Genomes cell lines (Singlet demultiplexing -> per-donor ancestry)
    lp = artifact_dir / "lcl_ancestry_calls.tsv"
    if lp.exists():
        lc = pd.read_csv(lp, sep="\t")
        for ds, d in lc.groupby("dataset"):
            vals = {}
            for t in tools:
                x = d[d["tool"] == t]
                if not len(x):
                    vals[t] = None  # tool not run on this dataset
                elif ds == "osorio2019_lcl_mix":  # no per-cell truth: the two donors must carry EUR and AFR
                    want = {"EUR", "AFR"}
                    vals[t] = len(want & set(x["call"].dropna())) / 2
                else:
                    vals[t] = float((x["call"] == x["super_population"]).mean())
            if vals.get("singlet") is None:
                continue
            add("donor_accuracy", "Pool donors given their individual's super-population (single-cell, end to end)",
                "fraction", "accuracy", ds, vals,
                notes="Singlet 1.0.0 demultiplexes the pool (--n-donors 2); each tool then calls ancestry on each donor's reads")

    sx_path = artifact_dir / "sex_bulk_calls.tsv"
    if sx_path.exists():
        sx = pd.read_csv(sx_path, sep="\t")
        sx = sx[sx["tool"] != "_counts"]
        st = [t for t in ["singlet", "peddy", "xist-chrY-rule"] if t in set(sx["tool"])]
        vals = {t: float((sx[sx["tool"] == t]["call"] == sx[sx["tool"] == t]["sex_truth"]).mean()) for t in st}
        m = {"id": "donor_sex_accuracy", "name": "Donor sex from RNA reads matches the 1000 Genomes record (8M read pairs)",
             "unit": "fraction", "category": "accuracy", "dataset": "mage150", "direction": "higher",
             "singlet": {"value": r(vals.get("singlet"))},
             "competitors": [{"tool": t, "value": r(vals[t])} for t in st if t != "singlet"],
             "rule": {"type": "match_or_beat", "against": "best"},
             "notes": "bulk RNA of clonal LCLs: X-chromosome heterozygosity (peddy) is weak because of X inactivation"}
        metrics.append(m)

    metrics.append({"id": "n_individuals", "name": "Individuals scored", "unit": "individuals", "category": "coverage",
                    "dataset": "pooled", "direction": "none", "singlet": {"value": int(calls["individual"].nunique())},
                    "rule": {"type": "informational"}})
    commit = prov["bench_commit"]
    head = [m for m in metrics if m["id"].startswith("accuracy.")]
    deep = head[-1] if head else None
    conclusion = prov.get("conclusion") or (
        f"On {calls['individual'].nunique()} 1000 Genomes individuals from 26 populations (MAGE RNA-seq thinned to single-cell "
        f"donor depth, leave-out reference), Singlet's published ancestry label is correct for "
        f"{deep['singlet']['value']:.1%} at {deep['id'].split('.', 1)[1]} read pairs, against "
        + ", ".join(f"{c['value']:.1%} for {c['tool']}" for c in deep["competitors"] if c["value"] is not None) + "."
        if deep else "No data.")
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Genetic ancestry per donor",
        "conclusion": conclusion, "methods": prov["methods"], "status": prov.get("status", "draft"),
        "generated_at": prov.get("generated_at") or now_iso(), "owner": "bench-public",
        "singlet": prov["singlet"], "tools": prov["tools"], "datasets": prov["datasets"],
        "metrics": metrics, "resources": prov["resources"], "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {"ancestry_calls.tsv": "derived", "sex_bulk_calls.tsv": "derived", "lcl_ancestry_calls.tsv": "derived",
                                                          "provenance.json": "derived"}),
        "code": [
            {"path": "python/singlet/celllabels/ancestry.py", "role": "implementation",
             "url": code_url("python/singlet/celllabels/ancestry.py", commit)},
            {"path": "bench/public/sexancestry/mage.sbatch", "role": "harness", "url": code_url("bench/public/sexancestry/mage.sbatch", commit)},
            {"path": "bench/public/sexancestry/ancestry_af.sbatch", "role": "harness",
             "url": code_url("bench/public/sexancestry/ancestry_af.sbatch", commit)},
            {"path": "bench/public/sexancestry/ancestry_score.py", "role": "harness",
             "url": code_url("bench/public/sexancestry/ancestry_score.py", commit)},
            {"path": "bench/public/capabilities/genetic_ancestry.py", "role": "metric",
             "url": code_url("bench/public/capabilities/genetic_ancestry.py", commit)},
        ],
        "reproduce": {"module": "bench/public/capabilities/genetic_ancestry.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "bash bench/public/sexancestry/submit_ancestry.sh"},
        "caveats": prov.get("caveats", []),
    }
    return C.finalize(doc)
