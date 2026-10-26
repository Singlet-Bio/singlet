"""Capability `atac`: scATAC / Multiome ATAC fragments, peaks, QC and cells vs Cell Ranger ATAC / ARC.

Artifacts (bench/atac_spatial/make_artifacts.py, one set per dataset DS):
- ``atac_<DS>_barcodes.tsv.gz``: per barcode, Cell Ranger (tenx_*), Singlet (singlet_*) and Chromap + SnapATAC2
  (chromap_*) values: cell call, fragments, TSS fragments, peak-region fragments, TSS enrichment, nucleosome signal.
- ``atac_<DS>_peaks_{tenx,singlet,chromap_macs3}.bed.gz``: peak sets.
- ``atac_<DS>_fragment_concordance.json``: exact fragment matches with Cell Ranger on one chromosome, 10x cells (derived:
  needs the multi-GB fragment files).
- ``provenance.json``: commit, tools, datasets, methods, resources, status, caveats.

Competitor for every metric: Chromap 0.2 (``--preset atac``, same FASTQs, same whitelist, same GRCh38-2024-A), then the
Signac MACS recipe for peaks and SnapATAC2 + ArchR's default filter for cells, i.e. the standard open-source route from
FASTQs to a Signac/ArchR object. Truth: Cell Ranger's published outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import artifact_entries, code_url, contract, load_provenance, now_iso  # noqa: E402
from _atacsp import (archr_default_cells, bootstrap_metric, diff_tolerance, jaccard, metric, peak_f1_by_chrom,  # noqa: E402
                     peak_scores, read_bed, read_json, read_tsv, spearman)

CAP = "atac"
COMP = "chromap+snapatac2"
MATCH = {"type": "match_or_beat", "against": "best"}


def dataset_metrics(art: Path, ds: str, gating: bool) -> tuple[list, dict]:
    b = read_tsv(art / f"atac_{ds}_barcodes.tsv.gz")
    tenx = set(b.index[b.tenx_is_cell == 1])
    sing = set(b.index[b.singlet_is_cell == 1])
    comp = set(b.index[archr_default_cells(b.get("chromap_fragments"), b.get("chromap_tsse"))]) \
        if "chromap_fragments" in b else set()
    cells = b.loc[b.index.isin(tenx)]
    out = []
    # 1. cell calls (bootstrap over the barcodes any tool calls)
    idx = np.array(sorted(tenx | sing | comp))
    def jac(sel, other):
        s = set(idx[sel]); return jaccard(s & tenx, s & other)
    rule = (diff_tolerance(lambda i: jac(i, sing), lambda i: jac(i, comp), len(idx), unit="called barcodes")
            if comp else MATCH) or MATCH
    out.append(metric("cell_jaccard", "Cell calls: Jaccard with Cell Ranger", "fraction", "agreement", ds, "higher",
                      jaccard(tenx, sing), [(COMP, jaccard(tenx, comp) if comp else None)], rule,
                      ci=bootstrap_metric(lambda i: jac(i, sing), len(idx), 500),
                      notes=f"Cell Ranger {len(tenx)}, Singlet {len(sing)}, {COMP} (ArchR default filter) {len(comp)} cells"))
    # 2. fragments per cell (bootstrap over Cell Ranger's cells)
    cs, ct = cells.singlet_fragments.to_numpy(float), cells.tenx_fragments.to_numpy(float)
    cc = cells.chromap_fragments.to_numpy(float) if "chromap_fragments" in cells else None
    rule = (diff_tolerance(lambda i: spearman(cs[i], ct[i]), lambda i: spearman(cc[i], ct[i]), len(cells))
            if cc is not None else MATCH) or MATCH
    out.append(metric("fragments_per_cell_spearman", "Fragments per cell: Spearman with Cell Ranger", "rho",
                      "accuracy", ds, "higher", spearman(cs, ct), [(COMP, spearman(cc, ct) if cc is not None else None)],
                      rule,
                      notes="over Cell Ranger's cells; chrM excluded"))
    # 3. exact fragments
    fc = read_json(art / f"atac_{ds}_fragment_concordance.json")
    rec = lambda t: fc[t]["exact_shared"] / max(1, fc["tenx"]) if t in fc else None
    rule = MATCH
    if rec("singlet") is not None and rec("chromap") is not None:
        n = fc["tenx"]; p1, p2 = rec("singlet"), rec("chromap")
        half = 1.96 * float(np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n))
        rule = {"type": "match_or_beat", "against": "best", "tolerance": {"abs": round(half, 4)},
                "rationale": f"1.96 x the binomial standard error of the difference of two proportions over the {n:,} "
                             "Cell Ranger fragments (conservative: the two tools are scored on the same fragments)"}
    out.append(metric("fragment_exact_recall", f"Cell Ranger fragments reproduced exactly ({fc['chromosome']})",
                      "fraction", "accuracy", ds, "higher", rec("singlet"), [(COMP, rec("chromap"))], rule,
                      notes=f"{fc['tenx']:,} Cell Ranger fragments of its {fc['cells']} cells"))
    # 4. peaks
    P10 = read_bed(art / f"atac_{ds}_peaks_tenx.bed.gz")
    PS = read_bed(art / f"atac_{ds}_peaks_singlet.bed.gz")
    ps = peak_scores(PS, P10)
    pc_path = art / f"atac_{ds}_peaks_chromap_macs3.bed.gz"
    PC = read_bed(pc_path) if pc_path.exists() else None
    pc = peak_scores(PC, P10) if PC is not None else None
    chroms = np.array(sorted(set(P10.chrom)))
    rule = (diff_tolerance(lambda i: peak_f1_by_chrom(PS, P10, chroms[i]), lambda i: peak_f1_by_chrom(PC, P10, chroms[i]),
                           len(chroms), n_boot=100, unit="chromosomes") if PC is not None else MATCH) or MATCH
    out.append(metric("peak_f1", "Peaks: F1 of overlap with Cell Ranger peaks", "fraction", "agreement", ds, "higher",
                      ps["f1"], [(COMP, pc["f1"] if pc else None)], rule,
                      notes=f"recall/precision Singlet {ps['recall']:.3f}/{ps['precision']:.3f} ({ps['n']:,} peaks); "
                            + (f"{COMP} {pc['recall']:.3f}/{pc['precision']:.3f} ({pc['n']:,} peaks)" if pc else "")))
    # 5. TSS fragments per cell (Cell Ranger definition)
    if "tenx_tss_fragments" in cells and cells.tenx_tss_fragments.notna().any():
        out.append(metric("tss_fragments_spearman", "TSS fragments per cell: Spearman with Cell Ranger", "rho",
                          "accuracy", ds, "higher", spearman(cells.singlet_tss_fragments, cells.tenx_tss_fragments),
                          [], {"type": "threshold", "value": 0.9,
                               "rationale": "Cell Ranger reports TSS fragments but no open tool reproduces its TSS set; "
                                            "0.9 is the rank agreement of the fragments-per-cell metric of the "
                                            "open-source competitor on the same cells (metric 2), the bar a correct "
                                            "implementation of the same definition should clear"}))
    summary = {"ds": ds, "singlet": ps, "competitor": pc, "cells": (len(tenx), len(sing), len(comp))}
    return out, summary


def compute(artifact_dir: Path) -> dict:
    art = Path(artifact_dir)
    C = contract()
    prov = load_provenance(art)
    commit = prov["singlet"]["commit"]
    metrics, summ = [], []
    for ds in prov["test_datasets"]:
        m, s = dataset_metrics(art, ds, True); metrics += m; summ.append(s)
    figures = [{
        "id": "peak_recall_precision", "title": "Peak overlap with Cell Ranger", "kind": "grouped_bar",
        "caption": "Fraction of Cell Ranger peaks recovered (recall) and of each tool's peaks that Cell Ranger also calls "
                   "(precision).",
        "x": {"label": "dataset / measure", "categories": [f"{s['ds']} {k}" for s in summ for k in ("recall", "precision")]},
        "y": {"label": "fraction", "unit": "fraction", "scale": "linear", "min": 0, "max": 1},
        "series": [{"name": "Singlet", "tool": "singlet", "x": [f"{s['ds']} {k}" for s in summ for k in ("recall", "precision")],
                    "y": [round(s["singlet"][k], 4) for s in summ for k in ("recall", "precision")]}] +
                  ([{"name": COMP, "tool": COMP, "x": [f"{s['ds']} {k}" for s in summ for k in ("recall", "precision")],
                     "y": [round(s["competitor"][k], 4) if s["competitor"] else None for s in summ
                           for k in ("recall", "precision")]}] if all(s["competitor"] for s in summ) else []),
    }]
    lead = metrics[0]
    conclusion = prov.get("conclusion") or (
        "From the same FASTQs, Singlet's ATAC fragments, peaks and cells agree with Cell Ranger "
        + "; ".join(f"{m['dataset']}: {m['name'].split(':')[0].lower()} {m['singlet']['value']} vs "
                    f"{m['competitors'][0]['value']} for {COMP}" for m in metrics if m["id"] == "cell_jaccard") + ".")
    kinds = {}
    for ds in prov["test_datasets"] + prov.get("dev_datasets", []):
        kinds.update({f"atac_{ds}_barcodes.tsv.gz": "derived", f"atac_{ds}_peaks_tenx.bed.gz": "truth",
                      f"atac_{ds}_peaks_singlet.bed.gz": "singlet_output",
                      f"atac_{ds}_peaks_chromap_macs3.bed.gz": "competitor_output",
                      f"atac_{ds}_fragment_concordance.json": "derived"})
    kinds["provenance.json"] = "derived"
    files = ["include/singlet/pileup/atac_pipeline.h", "include/singlet/pileup/atac_barcodes.h",
             "include/singlet/pileup/atac_fragment.h", "include/singlet/pileup/atac_peak_caller.h",
             "include/singlet/pileup/atac_metrics.h", "bench/atac_spatial/make_artifacts.py",
             "bench/atac_spatial/run_chromap.sbatch", "bench/atac_spatial/competitors_atac.py",
             "bench/public/capabilities/atac.py", "bench/public/capabilities/_atacsp.py"]
    doc = {
        "schema": C.SCHEMA_CAPABILITY, "schema_version": C.SCHEMA_VERSION, "capability": CAP,
        "title": "Chromatin accessibility (scATAC)", "conclusion": conclusion, "methods": prov["methods"],
        "status": prov.get("status", "draft"), "generated_at": prov.get("generated_at") or now_iso(),
        "owner": "atac-multiome-spatial", "singlet": prov["singlet"], "tools": prov["tools"],
        "datasets": prov["datasets"], "metrics": metrics, "resources": prov["resources"], "figures": figures,
        "artifacts": artifact_entries(art, CAP, kinds),
        "code": [{"path": f, "url": code_url(f, commit),
                  "role": "metric" if "capabilities" in f else "harness" if "bench/" in f else "implementation"}
                 for f in files],
        "reproduce": {"module": "bench/public/capabilities/atac.py", "function": "compute", "laptop_minutes": 6,
                      "clipper_command": "bash bench/atac_spatial/submit.sh"},
        "caveats": prov.get("caveats", []),
    }
    for m in metrics:
        if m["id"] in ("fragments_per_cell_spearman", "peak_f1"):
            doc["figures"].append(C.pareto_figure(doc, m["id"], m["dataset"], "cpu_h"))
    doc["reach"] = [
        {"tool": t, "output": o, "supported": s, "note": n} for t, o, s, n in (
            ("singlet", "fragments, peaks, peak matrix, per-cell QC, cells", "yes", "from FASTQs, one pipeline run"),
            ("singlet", "gene activity", "yes", "Signac GeneActivity definition"),
            ("singlet", "Multiome ATAC-GEX pairing and joint cells", "yes", "--paired-gex"),
            (COMP, "fragments, peaks, per-cell QC, cells", "yes", "three tools; read format chosen by the user"),
            (COMP, "Multiome ATAC-GEX pairing", "partial", "Chromap's --barcode-translate refused the ARC pair table"),
            ("cellranger-atac/arc", "fragments, peaks, peak matrix, cells", "yes", "10x EULA; published outputs used"))]
    return C.finalize(doc)
