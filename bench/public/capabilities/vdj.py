"""Capability ``vdj``: Singlet's V(D)J contigs, chains and clonotypes against Cell Ranger vdj (10x's and the authors'
published outputs, the reference) and TRUST4 (run here on the same FASTQs and threads).

Artifacts (``artifact_dir``, written by ``bench/vdj/make_artifacts.py``), per library ``<lib>``:
- ``<lib>.singlet.csv.gz``, ``<lib>.trust4.csv.gz``: productive contigs (barcode, chain, cdr3_nt, cdr3, v, j, c, umis);
- ``<lib>.cellranger.csv.gz`` and ``<lib>.cellranger_cells.txt``: Cell Ranger's productive high-confidence contigs and
  its V(D)J cells (the reference);
- ``<lib>.gex_cells.txt`` / ``<lib>.gex_lineage.csv`` when the library has a GEX matrix: called GEX cells and their
  T / B / other lineage from marker genes;
- ``provenance.json``: commit, library -> dataset, wall time and memory per run, V gene equivalence classes.

Libraries of one dataset and receptor are pooled (barcodes prefixed by library). Metric definitions:
``bench/vdj/compare_vdj.py`` (the same function computes them here and in the harness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from _common import (
    REPO_URL,
    artifact_entries,
    code_url,
    contract,
    load_provenance,
    now_iso,
    r,
)

CAP = "vdj"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "vdj"))
TOOLS = ("singlet", "trust4")
METRICS = [
    # (key in compare(), id suffix, name, unit, category, direction)
    ("contig_f1", "contig_f1", "Productive contigs (cell, chain, junction nt) against Cell Ranger's: F1 of recovery "
     "and precision", "fraction", "agreement", "higher"),
    ("cdr3_nt_accuracy", "cdr3_nt_accuracy", "Top junction per cell and chain identical to Cell Ranger's (nt)",
     "fraction", "agreement", "higher"),
    ("correct_pairing", "correct_pairing", "Cell Ranger paired cells whose top junctions of both chains are reproduced",
     "fraction", "agreement", "higher"),
    ("v_gene_agreement_equivalent", "v_gene_agreement", "V gene as Cell Ranger's on identical junctions (indistinguishable genes as one)",
     "fraction", "agreement", "higher"),
    ("j_gene_agreement", "j_gene_agreement", "J gene as Cell Ranger's on identical junctions", "fraction", "agreement",
     "higher"),
    ("clonotype_ari", "clonotype_ari", "Clonotypes (exact chain sets) vs Cell Ranger's, adjusted Rand index",
     "ARI", "agreement", "higher"),
    ("wrong_lineage_method", "wrong_lineage", "Contigs of the wrong lineage (TCR in GEX B cells, BCR in GEX T cells)",
     "fraction", "accuracy", "lower"),
]


def _load(artifact_dir: Path, lib: str, name: str):
    import pandas as pd

    p = artifact_dir / f"{lib}.{name}"
    if not p.exists():
        return None
    if p.suffix == ".txt":
        return set(x.strip() for x in p.read_text().splitlines() if x.strip())
    return pd.read_csv(p, keep_default_na=False, na_values=[""])


def _pool(parts):
    import pandas as pd

    out = {}
    for key in parts[0]:
        vals = [p[key] for p in parts]
        if any(v is None for v in vals):
            out[key] = None
        elif isinstance(vals[0], set):
            out[key] = set().union(*vals)
        else:
            out[key] = pd.concat(vals, ignore_index=True)
    return out


def _prefixed(lib, d):
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
        elif isinstance(v, set):
            out[k] = {f"{lib}:{x}" for x in v}
        else:
            v = v.copy()
            v["barcode"] = f"{lib}:" + v["barcode"].astype(str)
            out[k] = v
    return out


NOISE_KEYS = ("contig_f1", "cdr3_nt_accuracy", "correct_pairing", "v_gene_agreement_equivalent", "j_gene_agreement")


def _noise(CV, P, equiv, n=40, seed=0):
    """95% half-width of the Singlet - TRUST4 difference per metric, by half-sampling the reference cells: the
    difference is recomputed on random halves, and a half sample's variance is twice the full sample's for these
    averages, so the half-width is 1.96 * sd / sqrt(2)."""
    rng = np.random.default_rng(seed)
    cells = np.array(sorted(P["cells"]))
    diffs = {k: [] for k in NOISE_KEYS}
    for _ in range(n):
        half = set(rng.choice(cells, len(cells) // 2, replace=False))
        def sub(t, half=half):
            return t[t["barcode"].isin(half)] if t is not None else None
        rs = {t: CV.compare(sub(P["ref"]), half, sub(P[t]), None, P["gex_cells"], equiv) for t in TOOLS}
        for k in NOISE_KEYS:
            diffs[k].append(rs["singlet"][k] - rs["trust4"][k])
    return {k: float(1.96 * np.std(v, ddof=1) / np.sqrt(2)) for k, v in diffs.items()}


def compute(artifact_dir: Path) -> dict:
    import compare_vdj as CV

    artifact_dir = Path(artifact_dir)
    C = contract()
    prov = load_provenance(artifact_dir)
    equiv = prov.get("equivalence", {})
    groups = {}
    for lib, info in prov["libraries"].items():
        groups.setdefault((info["dataset"], info["chains"]), []).append(lib)
    metrics, figures, results = [], [], {}
    for (ds, chains), libs in sorted(groups.items()):
        parts = []
        for lib in libs:
            parts.append(_prefixed(lib, {
                "ref": _load(artifact_dir, lib, "cellranger.csv.gz"), "cells": _load(artifact_dir, lib, "cellranger_cells.txt"),
                "gex_cells": _load(artifact_dir, lib, "gex_cells.txt"), "lineage": _load(artifact_dir, lib, "gex_lineage.csv"),
                **{t: _load(artifact_dir, lib, f"{t}.csv.gz") for t in TOOLS}}))
        P = _pool(parts)
        if P["singlet"] is None or P["ref"] is None:
            continue  # a library not run yet
        res = {t: CV.compare(P["ref"], P["cells"], P[t], P["lineage"], P["gex_cells"], equiv) for t in TOOLS if P[t] is not None}
        results[(ds, chains)] = res
        noise = _noise(CV, P, equiv) if all(P[t] is not None for t in TOOLS) else {}
        pre = chains.lower()
        for key, mid, name, unit, cat, direc in METRICS:
            if key not in res["singlet"]:
                continue
            m = {"id": f"{pre}_{mid}", "name": f"{chains}: {name}", "unit": unit, "category": cat, "dataset": ds,
                 "direction": direc, "singlet": {"value": r(res["singlet"][key])},
                 "competitors": [{"tool": t, "value": r(res[t][key])} for t in res if t != "singlet" and key in res[t]],
                 "rule": {"type": "match_or_beat", "against": "best"}}
            if key in noise:
                hw = r(noise[key])
                if hw and hw > 0:
                    m["rule"]["tolerance"] = {"abs": hw}
                    m["rule"]["rationale"] = ("95% half-width of the Singlet - TRUST4 difference, from 40 random halves of "
                                              "the reference cells (1.96 sd / sqrt 2): a smaller gap is sampling noise")
            if key == "contig_f1":
                m["notes"] = "recovery / precision / extra contigs per Cell Ranger contig: " + "; ".join(
                    f"{t} {res[t]['contig_recovery']:.3f} / {res[t]['contig_precision']:.3f} / {res[t]['extra_contigs']:.3f}"
                    for t in res)
            if key == "wrong_lineage_method":
                # Cell Ranger's own value is a competitor here: these metrics do not use Cell Ranger as truth
                ref_key = key.replace("_method", "_reference")
                if ref_key in res["singlet"]:
                    m["competitors"].append({"tool": "cellranger-vdj", "value": r(res["singlet"][ref_key])})
            metrics.append(m)
        figures.append({
            "id": f"{ds}_{pre}_agreement", "title": f"{chains} agreement with Cell Ranger, {ds}", "kind": "grouped_bar",
            "x": {"label": "metric", "categories": ["contig_recovery", "cdr3_nt_accuracy", "v_gene_agreement_equivalent",
                                                      "clonotype_ari"]},
            "y": {"label": "fraction", "unit": "fraction", "min": 0, "max": 1},
            "series": [{"name": t, "tool": t, "x": ["contig_recovery", "cdr3_nt_accuracy", "v_gene_agreement_equivalent",
                                                     "clonotype_ari"],
                        "y": [r(res[t].get(k)) for k in ("contig_recovery", "cdr3_nt_accuracy",
                                                         "v_gene_agreement_equivalent", "clonotype_ari")]} for t in res]})
    resources = []
    by = {}
    for x in prov["resources"]:
        k = (x["tool"], x["dataset"])
        a = by.setdefault(k, {"tool": x["tool"], "dataset": x["dataset"], "wall_s": 0.0, "max_rss_gb": 0.0,
                              "threads": x.get("threads", 8)})
        a["wall_s"] += x.get("wall_s", 0.0)
        a["max_rss_gb"] = max(a["max_rss_gb"], x.get("max_rss_gb", 0.0))
    for a in by.values():
        a["wall_s"] = r(a["wall_s"], 1)
        a["max_rss_gb"] = r(a["max_rss_gb"], 2)
        a["cpu_h"] = r(a["wall_s"] * a["threads"] / 3600, 3)
        resources.append(a)
    for ds in sorted({d for d, _ in groups}):
        w = {t: sum(x["wall_s"] for x in resources if x["tool"] == t and x["dataset"] == ds) for t in TOOLS}
        if w["singlet"] and w["trust4"]:
            metrics.append({"id": "wall_time", "name": "Wall time, all libraries of the dataset (8 threads)", "unit": "s",
                            "category": "resources", "dataset": ds, "direction": "lower",
                            "singlet": {"value": r(w["singlet"], 1)}, "competitors": [{"tool": "trust4", "value": r(w["trust4"], 1)}],
                            "rule": {"type": "match_or_beat", "against": "best"},
                            "notes": "Cell Ranger vdj was not run (its outputs are 10x's and the authors'); no time for it"})
    commit = prov["commit"]
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAP,
        "title": "V(D)J clonotypes (TCR / BCR)",
        "conclusion": "",
        "methods": (
            "Singlet assembles V(D)J libraries itself (singlet.vdj: segment reference from the GENCODE genome the pipeline uses, "
            "15-mer read classification, per-UMI-consensus junction assembly between codon-checked Cys104 and Phe/Trp118, "
            "Cell Ranger barcode and UMI correction). Truth is Cell Ranger vdj's productive high-confidence contigs as "
            "published by 10x (pbmc5p_v1: Cell Ranger 3.0.0; pbmc10k_5p_v2: 5.0.0) or by the authors (GSE145370: 3.0.1); "
            "Cell Ranger was not re-run. TRUST4 ran on the same FASTQs with 8 threads, its IMGT reference and the 10x "
            "whitelist. Metrics are on Cell Ranger's V(D)J cells; V(D)J cells of Singlet and TRUST4 are restricted to the "
            "called GEX cells where the dataset has a GEX matrix. V genes that are identical (or 1 nt apart) up to Cys104 "
            "count as one (IGKV1-39 / IGKV1D-39 ...). Wrong lineage uses T / B labels from GEX markers (CD3D/E/G, CD2 vs "
            "MS4A1, CD79A/B), independent of every V(D)J caller."),
        "status": "draft", "generated_at": now_iso(), "owner": "vdj-crispr",
        "singlet": {"version": "1.0.0", "commit": commit, "branch": "claude/vdj-crispr-benchmark-721d3a"},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": "bench/vdj/run_vdj.py --threads 8 (singlet.vdj defaults)"},
            {"name": "trust4", "version": "1.1.5", "role": "competitor", "url": "https://github.com/liulab-dfci/TRUST4",
             "params": "run-trust4 -f hg38_bcrtcr.fa --ref human_IMGT+C.fa (-1 R1 -2 R2 --readFormat r1:39:-1,bc:0:15,um:16:25 "
                       "for 150-nt R1, else -u R2 --readFormat bc:0:15,um:16:25) --barcode R1 --UMI R1 "
                       "--barcodeWhitelist 737K-august-2016.txt -t 8"},
            {"name": "cellranger-vdj", "version": "3.0.0 / 5.0.0 / 3.0.1", "role": "truth",
             "url": "https://cf.10xgenomics.com/samples/cell-vdj/",
             "params": "published filtered_contig_annotations.csv (10x datasets, GEO GSE145370 supplementary)"},
        ],
        "datasets": [
            {"id": "pbmc5p_v1", "title": "10x vdj_v1_hs_pbmc2 T and B libraries (5' v1, 150x150)",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-vdj/3.0.0/vdj_v1_hs_pbmc2_t/"},
             "truth": "10x Cell Ranger 3.0.0 filtered contigs", "data_kind": "real", "notes": "development data: assembler parameters were set while looking at it"},
            {"id": "pbmc10k_5p_v2", "title": "10x sc5p_v2_hs_PBMC_10k T and B libraries (5' v2, 26x90)",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/"},
             "truth": "10x Cell Ranger 5.0.0 filtered contigs", "data_kind": "real"},
            {"id": "gse145370_tcr", "title": "GSE145370 ESCC TCR libraries held out from development (10x 5' V(D)J)",
             "accessions": {"gse": "GSE145370", "other": sorted(x for x in prov["libraries"] if x.startswith("gse145370_"))},
             "truth": "authors' Cell Ranger 3.0.1 filtered contigs (GEO supplementary)", "data_kind": "real"},
            {"id": "tenx_pbmc10k_5pv2_controller", "title": "10x 10k PBMC 5' v2 (Chromium Controller) T and B libraries",
             "data_kind": "real", "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-vdj/6.1.0/"
                                                        "10k_PBMC_5pv2_nextgem_Chromium_Controller_Multiplex/"},
             "truth": "10x Cell Ranger 6.1.0 filtered contigs", "notes": "held out: downloaded after the code was frozen"},
        ],
        "metrics": metrics, "resources": resources, "figures": figures,
        "artifacts": artifact_entries(artifact_dir, CAP, {p.name: ("truth" if "cellranger" in p.name or "gex" in p.name
                                                                     else "competitor_output" if "trust4" in p.name
                                                                     else "singlet_output" if "singlet" in p.name else "derived")
                                                         for p in sorted(artifact_dir.iterdir())}),
        "code": [{"path": "python/singlet/vdj", "url": f"{REPO_URL}/tree/{commit}/python/singlet/vdj", "role": "implementation"},
                 {"path": "bench/vdj", "url": f"{REPO_URL}/tree/{commit}/bench/vdj", "role": "harness"},
                 {"path": "bench/vdj/compare_vdj.py", "url": code_url("bench/vdj/compare_vdj.py", commit), "role": "metric"}],
        "reproduce": {"module": "bench/public/capabilities/vdj.py", "function": "compute", "laptop_minutes": 1,
                      "clipper_command": "sbatch bench/vdj/hpc/run.sbatch (per library), then bench/vdj/make_artifacts.py"},
        "caveats": [
            "Cell Ranger vdj was not run here (no licence accepted); its published outputs are the reference, so agreement "
            "metrics measure agreement with Cell Ranger, not with ground truth.",
            "pbmc5p_v1 and pbmc10k_5p_v2 are development data: the assembler's filters and fixes were made while comparing "
            "with their Cell Ranger contigs (and with GSE145370 S133A / S158A, which are therefore left out). The GSE145370 "
            "libraries scored here and the Chromium Controller set were downloaded after the code was frozen.",
            "Neither GENCODE nor RefSeq annotates IGHV1-8 and IGHV4-30-4 on GRCh38 (Cell Ranger and TRUST4 have them from "
            "IMGT): their rearrangements are not assembled or named, which costs BCR pairing and V agreement.",
            "MiXCR was not run: it needs a licence key.",
        ],
    }
    for (ds, chains) in sorted(groups):  # quality against cost (contract 1.1.0)
        try:
            doc["figures"].append(C.pareto_figure(doc, f"{chains.lower()}_contig_f1", ds, "wall_s",
                                                  fig_id=f"pareto_{chains.lower()}_contig_f1_{ds}"))
        except (StopIteration, KeyError):
            pass
    C.finalize(doc)
    doc["conclusion"] = _conclusion(doc)
    return doc


def _conclusion(doc) -> str:
    """Numbers-only summary (at most 100 words), from the metrics."""
    ms = doc["metrics"]
    agree = [m for m in ms if m["category"] != "resources"]
    n_pass = sum(m["pass"] == "pass" for m in agree)

    def rng(mid):
        v = [m["singlet"]["value"] for m in ms if m["id"].endswith(mid) and m["singlet"]["value"] is not None]
        t = [c["value"] for m in ms if m["id"].endswith(mid) for c in m["competitors"] if c["tool"] == "trust4"]
        return (min(v), max(v), min(t), max(t)) if v and t else None

    f1, cdr3 = rng("contig_f1"), rng("cdr3_nt_accuracy")
    wt = [(m["singlet"]["value"], m["competitors"][0]["value"]) for m in ms if m["id"] == "wall_time"]
    speed = min(t / s for s, t in wt) if wt else None
    parts = [f"Against Cell Ranger's published contigs, Singlet's contig F1 is {f1[0]:.2f}-{f1[1]:.2f} "
             f"(TRUST4 {f1[2]:.2f}-{f1[3]:.2f}) and its CDR3 accuracy {cdr3[0]:.3f}-{cdr3[1]:.3f} "
             f"(TRUST4 {cdr3[2]:.3f}-{cdr3[3]:.3f})." if f1 and cdr3 else ""]
    if speed:
        parts.append(f"It is at least {speed:.0f}x faster than TRUST4 at 8 threads.")
    parts.append(f"{n_pass} of {len(agree)} quality metrics match or beat every competitor; the rest are listed "
                 f"as failures (mostly BCR, where GRCh38 lacks IGHV1-8 and IGHV4-30-4).")
    return " ".join(p for p in parts if p)
