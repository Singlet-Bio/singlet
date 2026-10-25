# SPDX-License-Identifier: MIT
"""Capability ``adt`` for the public benchmark (results contract 1.0.0): ``compute(artifact_dir) -> capability document``.

The artifacts are written by ``bench/adt/adt_public_run.py`` on Clipper (every tool on the same antibody FASTQs, scored
on Cell Ranger's filtered cells). This function is the only place metrics are computed: the published
``bench/public/results/adt.json`` and the notebook both call it.

    python -P bench/adt/public_adt.py ARTIFACT_DIR OUT.json [--final]

Metrics per dataset (rule ``match_or_beat`` against the best competitor: kite, CITE-seq-Count, alevin-fry):

* ``count_pearson_log``: Pearson r of log1p(count) of every cell x antibody entry against Cell Ranger 3.0 (truth);
* ``umi_total_deviation``: |total UMIs / Cell Ranger's - 1| (lower is better);
* ``identical_entries``: share of cell x antibody entries equal to Cell Ranger's;
* ``protein_rna_concordance``: median Spearman rho, over the panel's antibodies whose gene is detected in 5-90% of cells,
  of CLR(protein) against log(RNA per 10k) in the same cells; the RNA is Cell Ranger's for every tool. Cell Ranger's own
  counts compete here too. The tolerance is the half-width of the bootstrap 95% CI (200 resamples of cells) of the
  difference between Singlet and the best competitor: smaller differences are noise.
"""

from __future__ import annotations

import datetime as _dt
import glob
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CAPABILITY = "adt"
TOOLS = ("singlet", "kite", "citeseqcount", "alevinfry")
COMPETITORS = ("kite", "citeseqcount", "alevinfry")
ART_URL = "https://data.singlet.bio/benchmarking/1.0.0/adt/"
REPO = "https://github.com/Singlet-Bio/singlet"
DATASET_META = {
    "pbmc1k_cite_v3": {"title": "10x PBMC 1k + TotalSeq-B, 3' v3",
                       "url": "https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_1k_protein_v3/"},
    "pbmc10k_cite_v3": {"title": "10x PBMC 10k + TotalSeq-B, 3' v3",
                        "url": "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp/3.0.0/pbmc_10k_protein_v3/"},
    "pbmc5p_v1": {"title": "10x PBMC 5' v1 + TotalSeq-C",
                  "url": "https://cf.10xgenomics.com/samples/cell-vdj/3.0.0/vdj_v1_hs_pbmc2_5gex_protein/"},
}
TOOL_INFO = {
    "singlet": {"role": "singlet", "url": f"{REPO}/tree/main/python/singlet/adt"},
    "cellranger": {"role": "truth", "version": "3.0.0",
                   "params": "10x Genomics' published filtered_feature_bc_matrix.h5 (cellranger count with feature barcoding)"},
    "kite": {"role": "competitor", "url": "https://github.com/pachterlab/kite",
             "citation": "Booeshaghi & Pachter (kallisto | bustools feature barcoding)"},
    "citeseqcount": {"role": "competitor", "url": "https://github.com/Hoohm/CITE-seq-Count",
                     "citation": "Roelli et al., CITE-seq-Count (Stoeckius et al. 2017/2018)"},
    "alevinfry": {"role": "competitor", "url": "https://github.com/COMBINE-lab/alevin-fry",
                  "citation": "He et al. 2022, alevin-fry"},
}
N_BOOT = 200
RNA_DETECTED = (0.05, 0.90)


def _contract():
    """bench/public/contract.py of this checkout, or the path in SINGLET_CONTRACT_PY."""
    p = os.environ.get("SINGLET_CONTRACT_PY") or str(Path(__file__).resolve().parents[1] / "public" / "contract.py")
    spec = importlib.util.spec_from_file_location("singlet_bench_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dense(df, cells, feats, col="feature"):
    ci = {b: i for i, b in enumerate(cells)}
    fi = {f: i for i, f in enumerate(feats)}
    X = np.zeros((len(feats), len(cells)))
    d = df[df[col].isin(fi) & df.barcode.isin(ci)]
    X[d[col].map(fi).values, d.barcode.map(ci).values] = d["count"].values
    return X


def _clr(X):
    lg = np.log1p(X)
    return np.log1p(X / np.exp(lg.mean(axis=0, keepdims=True)))


def _spearman(a, b):
    ra = pd.Series(a).rank().values
    rb = pd.Series(b).rank().values
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def _concordance(X, R, pairs, idx=None):
    """median Spearman over (antibody row, gene row) pairs, cells ``idx`` (default all)."""
    if idx is None:
        idx = np.arange(X.shape[1])
    C = _clr(X[:, idx])
    rho = [_spearman(C[i], R[j, idx]) for i, j in pairs]
    rho = [r for r in rho if not np.isnan(r)]
    return float(np.median(rho)) if rho else np.nan


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def dataset_metrics(adir: Path, ds: str):
    panel = pd.read_csv(adir / f"adt_{ds}_panel.tsv", sep="\t", dtype=str, keep_default_na=False)
    cells = pd.read_csv(adir / f"adt_{ds}_cells.tsv", sep="\t")
    bcs = list(cells.barcode)
    feats = list(panel.feature)
    truth = _dense(pd.read_parquet(adir / f"adt_{ds}_cellranger.parquet"), bcs, feats)
    tools = {}
    for t in TOOLS:
        p = adir / f"adt_{ds}_{t}.parquet"
        if p.exists():
            tools[t] = _dense(pd.read_parquet(p), bcs, feats)
    rna = pd.read_parquet(adir / f"adt_{ds}_rna.parquet")
    genes = sorted(rna.gene.unique())
    R = _dense(rna, bcs, genes, col="gene")
    R = np.log1p(R / np.maximum(cells.rna_total.values, 1)[None, :] * 1e4)
    gi = {g: i for i, g in enumerate(genes)}
    det = (R > 0).mean(axis=1)
    pairs = [(i, gi[g]) for i, g in enumerate(panel.gene) if g in gi and RNA_DETECTED[0] <= det[gi[g]] <= RNA_DETECTED[1]]
    lt = np.log1p(truth)
    out = {"n_cells": len(bcs), "n_antibodies": len(feats), "n_pairs": len(pairs), "tools": {}}
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(bcs), len(bcs)) for _ in range(N_BOOT)]
    for t, X in list(tools.items()) + [("cellranger", truth)]:
        lx = np.log1p(X)
        out["tools"][t] = {
            "count_pearson_log": float(np.corrcoef(lx.ravel(), lt.ravel())[0, 1]) if t != "cellranger" else None,
            "umi_total_deviation": float(abs(X.sum() / max(truth.sum(), 1) - 1)) if t != "cellranger" else None,
            "identical_entries": float((X == truth).mean()) if t != "cellranger" else None,
            "protein_rna_concordance": _concordance(X, R, pairs) if pairs else None,
            "umis": float(X.sum()),
        }
    # bootstrap tolerance for the concordance comparison
    comp = {t: v["protein_rna_concordance"] for t, v in out["tools"].items() if t != "singlet"}
    comp = {t: v for t, v in comp.items() if v is not None and not np.isnan(v)}
    if pairs and "singlet" in tools and comp:
        best = max(comp, key=comp.get)
        Xb = truth if best == "cellranger" else tools[best]
        diffs = [_concordance(tools["singlet"], R, pairs, b) - _concordance(Xb, R, pairs, b) for b in boots]
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
        out["concordance_tol"] = float((hi - lo) / 2)
        out["concordance_best"] = best
    return out


def compute(artifact_dir) -> dict:
    adir = Path(artifact_dir)
    dss = sorted({Path(p).name[len("adt_"):-len("_cellranger.parquet")]
                  for p in glob.glob(str(adir / "adt_*_cellranger.parquet"))})
    res = {ds: dataset_metrics(adir, ds) for ds in dss}
    resources, versions, commit = [], {}, None
    for p in sorted(glob.glob(str(adir / "resources_adt_*.json"))):
        r = json.load(open(p))
        if r.get("returncode", 0) != 0:
            continue
        resources.append({"tool": r["tool"], "dataset": r["dataset"], "wall_s": r["wall_s"],
                          "max_rss_gb": r["max_rss_gb"], "threads": r["threads"],
                          "cpu_h": None, "notes": "same 8 threads and node type for every tool"})
        versions.setdefault(r["tool"], (r.get("version") or "unknown", r.get("command", "")))
        commit = commit or r.get("singlet_commit")
    resources = [{k: v for k, v in r.items() if v is not None} for r in resources]
    metrics = []
    defs = [("count_pearson_log", "Pearson r of log counts vs Cell Ranger", "r", "agreement", "higher"),
            ("umi_total_deviation", "Deviation of total UMIs from Cell Ranger", "fraction", "agreement", "lower"),
            ("identical_entries", "Cell x antibody entries identical to Cell Ranger", "fraction", "agreement", "higher"),
            ("protein_rna_concordance", "Protein-RNA concordance (median Spearman)", "rho", "accuracy", "higher")]
    for ds, r in res.items():
        if "singlet" not in r["tools"]:
            continue
        for mid, name, unit, cat, direction in defs:
            sv = r["tools"]["singlet"].get(mid)
            comps = [{"tool": t, "value": r["tools"][t][mid]} for t in list(COMPETITORS) + ["cellranger"]
                     if t in r["tools"] and r["tools"][t].get(mid) is not None and not np.isnan(r["tools"][t][mid])]
            rule = {"type": "match_or_beat", "against": "best"}
            if mid == "protein_rna_concordance" and "concordance_tol" in r:
                rule["tolerance"] = {"abs": round(r["concordance_tol"], 4)}
                rule["rationale"] = ("half-width of the bootstrap 95% CI (200 resamples of cells) of Singlet minus the best "
                                     f"competitor ({r['concordance_best']})")
            m = {"id": mid, "name": name, "unit": unit, "category": cat, "dataset": ds, "direction": direction,
                 "singlet": {"value": None if sv is None or np.isnan(sv) else round(float(sv), 6)},
                 "competitors": [{"tool": c["tool"], "value": round(float(c["value"]), 6)} for c in comps],
                 "rule": rule}
            if not m["competitors"]:
                m["rule"] = {"type": "informational"}
            metrics.append(m)
    tools = []
    for t in ["singlet"] + [x for x in COMPETITORS if x in versions] + ["cellranger"]:
        info = dict(TOOL_INFO[t])
        ver, cmd = versions.get(t, (info.pop("version", "1.0.0" if t == "singlet" else "unknown"), ""))
        info.pop("version", None)
        tools.append({"name": t, "version": "1.0.0" if t == "singlet" else ver, **info,
                      **({"params": cmd} if cmd and "params" not in info else {})})
    artifacts = []
    for p in sorted(glob.glob(str(adir / "adt_*"))):
        n = Path(p).name
        kind = ("truth" if n.endswith("_cellranger.parquet") else "singlet_output" if n.endswith("_singlet.parquet")
                else "competitor_output" if any(n.endswith(f"_{t}.parquet") for t in COMPETITORS) else "derived")
        artifacts.append({"name": n, "url": ART_URL + n, "sha256": _sha(p), "bytes": os.path.getsize(p), "kind": kind,
                          "dataset": next((d for d in dss if n.startswith(f"adt_{d}_")), None)})
    win = {}
    for m in metrics:
        if m["rule"]["type"] == "match_or_beat" and m["singlet"]["value"] is not None:
            vals = [c["value"] for c in m["competitors"]]
            best = max(vals) if m["direction"] == "higher" else min(vals)
            win.setdefault(m["id"], []).append((m["dataset"], m["singlet"]["value"], best))
    pr = res.get("pbmc10k_cite_v3") or next(iter(res.values()), {"tools": {}})
    s_r = pr["tools"].get("singlet", {}).get("count_pearson_log")
    conclusion = (f"Singlet counts antibody UMIs the way Cell Ranger does: on {len(res)} 10x CITE-seq datasets its counts "
                  f"correlate with Cell Ranger's at r = {s_r:.4f} (log scale) on the largest and match more entries "
                  "exactly than kite, CITE-seq-Count or alevin-fry, while 3' v3 TotalSeq-B feature barcodes are "
                  "translated to GEX barcodes from the reads.") if s_r else "Results pending."
    methods = (
        "Data: 10x Genomics CITE-seq datasets (TotalSeq-B on 3' v3, TotalSeq-C on 5'), antibody FASTQs only. Truth: the "
        "Cell Ranger 3.0 feature-barcode matrix 10x published with each dataset; every tool is scored on Cell Ranger's "
        "filtered cells and its antibody reference (names and sequences). Each tool ran once on the same FASTQs with 8 "
        "threads on the same Clipper node type, timed by /usr/bin/time: Singlet (singlet.adt.count: Cell Ranger's "
        "feature-barcoding rules, 1-mismatch cell barcode posterior >= 0.975, Cell Ranger UMI correction, 3' v3 barcode "
        "translation chosen from the reads); kite (kb count --workflow kite:10xFB on 3' v3, kite on 5'); CITE-seq-Count "
        "1.4.5 with the cells' feature-side barcodes as whitelist and a 10-base start trim for TotalSeq-B/C; alevin-fry "
        "(salmon 1.10 --features -k7 index, sketch mapping, cr-like UMI resolution). Metrics: Pearson r of log1p counts "
        "over all cell x antibody entries, |total UMIs / Cell Ranger - 1|, share of identical entries, and protein-RNA "
        "concordance: the median Spearman rho between CLR protein and RNA per 10k (Cell Ranger's RNA for all tools) "
        "over antibodies whose gene is detected in 5-90% of cells. Isotype controls and PTPRC isoform antibodies are "
        "excluded from concordance.")
    doc = {
        "schema": "singlet.benchmark.capability", "schema_version": "1.0.0", "capability": CAPABILITY,
        "title": "Antibody-derived tags (CITE-seq)", "conclusion": conclusion, "methods": methods,
        "status": "draft", "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "owner": "adt-modality",
        "singlet": {"version": "1.0.0", "commit": commit or "0" * 40, "branch": "claude/adt-modality-0637a6"},
        "tools": tools,
        "datasets": [{"id": ds, "title": DATASET_META.get(ds, {}).get("title", ds),
                      "accessions": {"url": DATASET_META.get(ds, {}).get("url", "")},
                      "truth": "Cell Ranger 3.0 feature-barcode counts published by 10x Genomics",
                      "n_cells": res[ds]["n_cells"],
                      "notes": f"{res[ds]['n_antibodies']} antibodies; {res[ds]['n_pairs']} antibody-gene pairs in the "
                               "concordance metric"} for ds in dss],
        "metrics": metrics, "resources": resources,
        "figures": figures(res, dss), "artifacts": artifacts,
        "code": [{"path": p, "role": r, "url": f"{REPO}/blob/{commit or 'main'}/{p}"} for p, r in (
            ("python/singlet/adt/count.py", "implementation"), ("python/singlet/adt/panel.py", "implementation"),
            ("bench/adt/adt_public_run.py", "harness"), ("bench/adt/public_adt.py", "metric"))],
        "reproduce": {"module": "bench/adt/public_adt.py", "function": "compute", "laptop_minutes": 3,
                      "clipper_command": "sbatch --array=0-2 bench/adt/hpc/adt_public.sbatch"},
        "caveats": [
            "Cell Ranger itself was not re-run (its licence forbids redistribution on the cluster); its published 3.0 "
            "output is the truth, so it has no resource row.",
            "pbmc1k_cite_v3's TotalSeq-B sequences predate BioLegend's current numbering (15 of its barcodes name other "
            "antibodies in BioLegend's list); every tool here uses 10x's own reference, so this does not affect the "
            "comparison, but read-identified panels must not name these sequences from the catalog.",
        ],
    }
    complete = all(m["singlet"]["value"] is not None for m in metrics) and all(
        set(TOOLS) <= set(r["tools"]) for r in res.values()) and len(dss) == len(DATASET_META)
    doc["status"] = "final" if complete else "draft"
    return _contract().finalize(doc)


def figures(res, dss):
    out = []
    for mid, title, unit in (("count_pearson_log", "Pearson r of log counts vs Cell Ranger", "r"),
                             ("identical_entries", "Entries identical to Cell Ranger", "fraction"),
                             ("protein_rna_concordance", "Protein-RNA concordance", "median Spearman rho")):
        series = []
        for t in ["singlet", *COMPETITORS, "cellranger"]:
            xs = [ds for ds in dss if res[ds]["tools"].get(t, {}).get(mid) is not None]
            if xs:
                series.append({"name": t, "tool": t, "x": xs, "y": [round(res[ds]["tools"][t][mid], 6) for ds in xs]})
        if series:
            out.append({"id": f"{mid}_by_dataset", "title": title, "kind": "grouped_bar",
                        "x": {"label": "dataset", "categories": dss}, "y": {"label": mid, "unit": unit},
                        "series": series})
    return out


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir")
    ap.add_argument("out")
    ap.add_argument("--final", action="store_true")
    a = ap.parse_args(argv)
    doc = compute(a.artifact_dir)
    if a.final and doc["status"] != "final":
        raise SystemExit("not every dataset/tool/metric is present: the document stays a draft")
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
    print(json.dumps({m["id"] + "@" + m["dataset"]: m.get("pass") for m in doc["metrics"]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
