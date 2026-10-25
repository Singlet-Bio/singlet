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


_SELF = [  # self-reported group (substring, lower case) -> super-populations consistent with it
    (("hispanic", "latin"), {"AMR", "EUR"}),
    (("white", "caucasian", "caucacian", "european"), {"EUR", "AMR"}),
    (("black", "african"), {"AFR"}),
    (("chinese", "japanese", "korean", "east asian"), {"EAS"}),
    (("asian",), {"EAS", "SAS"}),
]


def _consistent(reported, call):
    """True / False when the self-report maps to a group, None when it does not (unknown, mixed, ...) or no named call."""
    if not isinstance(call, str) or call not in SUPERS:
        return None
    rep = str(reported).lower()
    if "not hispanic" in rep:
        return None
    for keys, ok in _SELF:
        if any(k in rep for k in keys):
            return call in ok
    return None


def compute(artifact_dir: Path) -> dict:
    import pandas as pd

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    calls = pd.read_csv(artifact_dir / "ancestry_calls.tsv", sep="\t", dtype={"level": str})
    tools = [t for t in ["singlet", "somalier", "peddy"] if t in set(calls["tool"])]
    comps = [t for t in tools if t != "singlet"]
    metrics, figures = [], []

    lp0 = artifact_dir / "lcl_ancestry_calls.tsv"
    sp0 = artifact_dir / "selfreport_calls.tsv"
    real_refs = []  # metrics on real data that corroborate the derived (thinned bulk) results
    if lp0.exists():
        lc0 = pd.read_csv(lp0, sep="\t")
        real_refs = [f"donor_accuracy@{d}" for d in sorted(set(lc0["dataset"])) if d == "osorio2019_lcl_mix"]
    if sp0.exists():
        real_refs.append("selfreport_consistency@atlas_selfreport")

    def add(mid, name, unit, category, dataset, vals, cis=None, notes=None, rule=None, corroborated_by=None):
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
        if corroborated_by:
            m["corroborated_by"] = corroborated_by
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
                notes=f"{n} individuals; Singlet's label is 'admixed' below q = 0.8, which counts as wrong here; 95% Wilson CI",
                corroborated_by=real_refs)
            add(f"accuracy_top.{tag}", f"Most likely super-population correct ({L}M read pairs)", "fraction", "accuracy", ds,
                acc_top, cis_top, notes="argmax of each tool's ancestry proportions or probabilities; no abstention",
                corroborated_by=real_refs)
            add(f"named_precision.{tag}", f"Named calls that are correct ({L}M read pairs)", "fraction", "accuracy", ds, prec,
                corroborated_by=real_refs)
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
                notes="Singlet 1.0.0 demultiplexes the pool (--n-donors 2); each tool then calls ancestry on each donor's reads",
                corroborated_by=None if ds == "osorio2019_lcl_mix" else real_refs)

    # ---- real atlas donors with self-reported race / ethnicity in GEO (Singlet only: no reads or BAMs for other tools)
    if sp0.exists():
        sr = pd.read_csv(sp0, sep="\t")
        named = sr[sr["call"].isin(SUPERS)]
        ok = [_consistent(rep, call) for rep, call in zip(named["self_reported"], named["call"])]
        known = [o for o in ok if o is not None]
        metrics.append({
            "id": "selfreport_consistency", "name": "Named ancestry calls consistent with the donor's self-reported group",
            "unit": "fraction", "category": "agreement", "dataset": "atlas_selfreport", "direction": "higher",
            "singlet": {"value": r(np.mean(known)) if known else None},
            "rule": {"type": "threshold", "value": 0.9,
                     "rationale": "self-report is a social category, not genetic ancestry, so a minority of donors may "
                                  "legitimately differ; 0.9 is a judgement bar, stated before scoring"},
            "notes": f"{len(sr)} donors with self-report, {len(named)} named calls, {len(sr) - len(named)} admixed or no call. "
                     "Consistent: white/caucasian/european -> EUR or AMR; hispanic/latino -> AMR or EUR; black/african "
                     "american -> AFR; asian -> EAS or SAS; chinese/japanese/korean -> EAS."})
        figures.append({"id": "selfreport_table", "title": "Atlas donors with self-reported ancestry", "kind": "table",
                        "columns": ["self-reported", "Singlet call", "q", "covered sites", "consistent"],
                        "rows": [[a, b if isinstance(b, str) else "no call", r(c, 2), int(d) if d == d else None, _consistent(a, b)]
                                 for a, b, c, d in zip(sr["self_reported"], sr["call"], sr["q_max"], sr["n_sites"])]})

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
             "notes": "bulk RNA of clonal LCLs: X-chromosome heterozygosity (peddy) is weak because of X inactivation",
             "corroborated_by": ["chromosomal-sex/sample_accuracy@geo_sex21"]}
        metrics.append(m)

    metrics.append({"id": "n_individuals", "name": "Individuals scored", "unit": "individuals", "category": "coverage",
                    "dataset": "pooled", "direction": "none", "singlet": {"value": int(calls["individual"].nunique())},
                    "rule": {"type": "informational"}})
    commit = prov["bench_commit"]
    datasets = [dict(d) for d in prov["datasets"]]
    for d in datasets:
        if d["id"] == "mage150":
            d["data_kind"] = "derived"  # real reads thinned to single-cell depth
    if lp0.exists():
        have = {d["id"] for d in datasets}
        lcl_ds = {"lcl_gse230239": {"id": "lcl_gse230239", "data_kind": "derived", "accessions": {"gse": "GSE230239",
                  "gsm": ["GSM7208744", "GSM7208745"]}, "title": "In-silico pool of HG00362 (FIN) and HG01277 (CLM) LCL libraries",
                  "truth": "1000 Genomes super-population of the library each cell came from"},
                  "osorio2019_lcl_mix": {"id": "osorio2019_lcl_mix", "data_kind": "real", "accessions": {"gse": "GSE126321",
                  "gsm": ["GSM3596322"], "srr": ["SRR8551678"]}, "title": "Real 1:1 pool of GM12878 (CEU) and GM18502 (YRI)",
                  "truth": "1000 Genomes super-populations of the two pooled individuals (EUR, AFR)"}}
        datasets += [v for k, v in lcl_ds.items() if k in set(lc0["dataset"]) and k not in have]
    if sp0.exists():
        sr0 = pd.read_csv(sp0, sep="	")
        datasets.append({"id": "atlas_selfreport", "data_kind": "real",
                         "title": f"Atlas donors with self-reported race or ethnicity in GEO ({len(sr0)} donors)",
                         "accessions": {"gse": ";".join(sorted(set(sr0["gse"]))), "gsm": sorted(set(sr0["gsm"]))},
                         "truth": "self-reported race or ethnicity in GEO sample characteristics (a social category, not genotype)"})
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
        "singlet": prov["singlet"], "tools": prov["tools"], "datasets": datasets,
        "metrics": metrics, "resources": prov["resources"], "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {"ancestry_calls.tsv": "derived", "sex_bulk_calls.tsv": "derived", "lcl_ancestry_calls.tsv": "derived", "selfreport_calls.tsv": "derived",
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
        "reach": [
            {"tool": "singlet", "output": "ancestry per donor of a pooled single-cell library (after demultiplexing)", "supported": "yes"},
            {"tool": "somalier", "output": "ancestry per donor of a pooled single-cell library (after demultiplexing)", "supported": "partial",
             "note": "needs a per-donor BAM, which another tool must produce"},
            {"tool": "peddy", "output": "ancestry per donor of a pooled single-cell library (after demultiplexing)", "supported": "partial",
             "note": "needs per-donor genotype calls"},
            {"tool": "singlet", "output": "admixture proportions and an explicit 'admixed' label", "supported": "yes"},
            {"tool": "somalier", "output": "admixture proportions and an explicit 'admixed' label", "supported": "no"},
            {"tool": "peddy", "output": "admixture proportions and an explicit 'admixed' label", "supported": "no"},
            {"tool": "singlet", "output": "published labels with no genotype in any public file", "supported": "yes"},
        ],
        "reproduce": {"module": "bench/public/capabilities/genetic_ancestry.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "bash bench/public/sexancestry/submit_ancestry.sh"},
        "caveats": prov.get("caveats", []),
    }
    return C.finalize(doc)
