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

from _common import REPO_URL, artifact_entries, bootstrap_ci, code_url, contract, load_provenance, now_iso, r

CAP = "vdj"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "bench" / "vdj"))
TOOLS = ("singlet", "trust4")
METRICS = [
    # (key in compare(), id suffix, name, unit, category, direction)
    ("contig_recovery", "contig_recovery", "Cell Ranger productive contigs found exactly (cell, chain, junction nt)",
     "fraction", "agreement", "higher"),
    ("cdr3_nt_accuracy", "cdr3_nt_accuracy", "Top junction per cell and chain identical to Cell Ranger's (nt)",
     "fraction", "agreement", "higher"),
    ("cell_chain_recovery", "cell_chain_recovery", "Cell Ranger cell-chains with a productive contig", "fraction",
     "coverage", "higher"),
    ("v_gene_agreement_equivalent", "v_gene_agreement", "V gene as Cell Ranger's on identical junctions (indistinguishable genes as one)",
     "fraction", "agreement", "higher"),
    ("j_gene_agreement", "j_gene_agreement", "J gene as Cell Ranger's on identical junctions", "fraction", "agreement",
     "higher"),
    ("extra_contigs", "extra_contigs", "Contigs in Cell Ranger cells that Cell Ranger does not report, per its contig",
     "fraction", "agreement", "lower"),
    ("pairing_rate_method", "pairing_rate", "Cell Ranger cells with a productive pair (TRA+TRB / IGH+IGK|L)", "fraction",
     "coverage", "higher"),
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
        res = {t: CV.compare(P["ref"], P["cells"], P[t], P["lineage"], P["gex_cells"], equiv) for t in TOOLS if P[t] is not None}
        results[(ds, chains)] = res
        pre = chains.lower()
        for key, mid, name, unit, cat, direc in METRICS:
            if key not in res["singlet"]:
                continue
            m = {"id": f"{pre}_{mid}", "name": f"{chains}: {name}", "unit": unit, "category": cat, "dataset": ds,
                 "direction": direc, "singlet": {"value": r(res["singlet"][key])},
                 "competitors": [{"tool": t, "value": r(res[t][key])} for t in res if t != "singlet" and key in res[t]],
                 "rule": {"type": "match_or_beat", "against": "best"}}
            if key in ("pairing_rate_method", "wrong_lineage_method"):
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
             "params": "published filtered_contig_annotations.csv (10x datasets, GEO GSE145370 supplementary)"},
        ],
        "datasets": [
            {"id": "pbmc5p_v1", "title": "10x vdj_v1_hs_pbmc2 T and B libraries (5' v1, 150x150)",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-vdj/3.0.0/vdj_v1_hs_pbmc2_t/"},
             "truth": "10x Cell Ranger 3.0.0 filtered contigs", "notes": "development data: assembler parameters were set while looking at it"},
            {"id": "pbmc10k_5p_v2", "title": "10x sc5p_v2_hs_PBMC_10k T and B libraries (5' v2, 26x90)",
             "accessions": {"url": "https://cf.10xgenomics.com/samples/cell-vdj/5.0.0/sc5p_v2_hs_PBMC_10k/"},
             "truth": "10x Cell Ranger 5.0.0 filtered contigs"},
            {"id": "gse145370_tcr", "title": "GSE145370 ESCC TCR libraries S133A, S158A (10x 5' V(D)J)",
             "accessions": {"gse": "GSE145370", "gsm": ["GSM4317423", "GSM4317428"], "srr": ["SRR11094260", "SRR11094265"]},
             "truth": "authors' Cell Ranger 3.0.1 filtered contigs (GEO supplementary)"},
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
            "pbmc5p_v1 is development data (the assembler's error handling was tuned on it); pbmc10k_5p_v2 and GSE145370 are held out.",
            "GRCh38 lacks TRBV4-3, TRBV6-3, TRBV7-8 and TRBV5-8 (Cell Ranger's reference has them from its alt-haplotype "
            "curation); Singlet names those rearrangements by the closest GRCh38 gene, which lowers V gene agreement.",
            "MiXCR was not run: it needs a licence key.",
        ],
    }
    C.finalize(doc)
    return doc
