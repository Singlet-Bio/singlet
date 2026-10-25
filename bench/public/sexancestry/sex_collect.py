#!/usr/bin/env python3
"""Chromosomal-sex benchmark, step 3: assemble the public artifacts and the provenance from the extraction and cellXY runs.

    python sex_collect.py --work $BP/work/sex --labels sex_labels.json --artifacts $BP/artifacts/chromosomal-sex \
        --singlet-binary $BP/singlet_binary.json --bench-commit <sha>

Writes sex_cells.parquet (per cell: counts of a few public sex genes, every tool's label), sex_samples.tsv (truth), and
provenance.json. No genotype-like data is involved.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

METHODS = (
    "Data: the geo_sex21 cohort row, 21 studies whose GEO sample characteristics record donor sex, with both sexes in each "
    "study. Counts are Singlet's served gene counts (exon + intron) of the called cells. Every tool sees the same cells and "
    "the same counts. Truth: the sex recorded in GEO. Singlet: singlet.celllabels.sex. It uses the split of "
    "sex-informative UMIs (XIST against 10 non-PAR chrY genes) as a per-cell likelihood ratio. Cells are pooled per donor "
    "(per sample when genotype demultiplexing found one donor), and the male fraction is fitted by EM. "
    "cellXY::classifySex (phipsonlab, GitHub 0.99.0) runs with defaults (qc = TRUE) on the full count matrix, in random "
    "blocks of at least 5,000 cells because it densifies the matrix. The XIST/chrY rule calls female when XIST > 0 and no "
    "MSY UMI, male when MSY > 0 and XIST = 0, and makes no call otherwise. A tool's sample call is the majority of its "
    "labelled cells; Singlet's is the donor call. Per-cell metrics count a missing label as wrong (accuracy) or leave the "
    "cell out (precision). 95% CIs are bootstrap over samples."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--singlet-binary", type=Path)
    ap.add_argument("--bench-commit", required=True)
    ap.add_argument("--dataset", default="geo_sex21")
    a = ap.parse_args(argv)
    a.artifacts.mkdir(parents=True, exist_ok=True)
    truth = json.loads(a.labels.read_text())
    samples = [json.loads(l) for l in open(a.work / "samples.jsonl")]
    samples = list({s["gsm"]: s for s in samples}.values())  # a re-run study appends again; keep the last record
    parts = []
    for s in samples:
        p = a.work / "cells" / f"{s['gsm']}.parquet"
        x = a.work / "cellxy" / f"{s['gsm']}.tsv"
        if not p.exists() or not x.exists():
            print(f"{s['gsm']}: missing {'cells' if not p.exists() else 'cellXY'} output, left out")
            continue
        df = pd.read_parquet(p)
        cx = pd.read_csv(x, sep="\t", dtype={"col": np.int64}, keep_default_na=False, na_values=["NA"])
        df = df.merge(cx.rename(columns={"pred_qc": "cellxy_qc", "pred_noqc": "cellxy_noqc"}), on="col", how="left")
        df.insert(0, "dataset", a.dataset)
        parts.append(df)
    cells = pd.concat(parts, ignore_index=True)
    for c in ("cellxy_qc", "cellxy_noqc"):
        cells[c] = cells[c].where(cells[c].isin(["Female", "Male"])).str.lower()
    cells["truth_donor_sex"] = pd.array([None] * len(cells), dtype="string")
    t0 = time.perf_counter()
    x, y = cells["xist"].to_numpy(), cells["super_y"].to_numpy()
    _ = np.where((x > 0) & (y == 0), "female", np.where((y > 0) & (x == 0), "male", None))
    rule_s = time.perf_counter() - t0
    cells.to_parquet(a.artifacts / "sex_cells.parquet", index=False)
    kept = set(cells["gsm"])
    st = pd.DataFrame([{"dataset": a.dataset, "gse": s["gse"], "gsm": s["gsm"], "species": s["species"],
                        "truth": truth[s["gsm"]], "n_donors": s.get("n_donors", 0), "n_cells": s["n_cells"]}
                       for s in samples if s["gsm"] in kept])
    st.to_csv(a.artifacts / "sex_samples.tsv", sep="\t", index=False)

    cx_t = [json.loads(l) for l in open(a.work / "cellxy_timing.jsonl")] if (a.work / "cellxy_timing.jsonl").exists() else []
    n = len(cells)
    resources = [
        {"tool": "singlet", "dataset": a.dataset, "wall_s": round(sum(s["singlet_sex_s"] for s in samples if s["gsm"] in kept), 2),
         "max_rss_gb": 0.0, "threads": 1, "notes": f"sex model only (counts already in memory), {n:,} cells"},
        {"tool": "cellXY", "dataset": a.dataset, "wall_s": round(sum(t["cellxy_s"] for t in cx_t if t["gsm"] in kept), 2),
         "max_rss_gb": 0.0, "threads": 1, "notes": "classifySex(qc = TRUE) and (qc = FALSE); dense blocks of >= 5,000 cells"},
        {"tool": "xist-chrY-rule", "dataset": a.dataset, "wall_s": round(rule_s, 4), "max_rss_gb": 0.0, "threads": 1},
    ]
    tf = a.work / "cellxy.time.txt"
    if tf.exists():
        import re
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tf.read_text())
        if m:
            resources[1]["max_rss_gb"] = round(int(m.group(1)) / 1048576, 2)
    sb = json.loads(a.singlet_binary.read_text()) if a.singlet_binary and a.singlet_binary.exists() else {}
    gses = sorted(set(st["gse"]))
    prov = {
        "bench_commit": a.bench_commit,
        "status": "draft",
        "methods": METHODS,
        "singlet": {"version": "1.0.0", "commit": a.bench_commit, "branch": "claude/bench-public-cohort-485cf7",
                    "binary": "python: singlet.celllabels (sex model xist-chrY-split/2)"},
        "tools": [
            {"name": "singlet", "version": "1.0.0", "role": "singlet", "params": "singlet.celllabels.sex.call_sex(X, genes, species, groups=donor)"},
            {"name": "cellXY", "version": "0.99.0 (phipsonlab/cellXY, GitHub)", "role": "competitor",
             "url": "https://github.com/phipsonlab/cellXY", "params": "classifySex(counts, genome = 'Hs' | 'Mm', qc = TRUE)"},
            {"name": "xist-chrY-rule", "version": "n/a", "role": "competitor",
             "params": "female if XIST > 0 and MSY = 0; male if MSY > 0 and XIST = 0; MSY = cellXY's 34-gene set"},
            {"name": "geo-metadata", "version": "GEO sample characteristics", "role": "truth"},
        ],
        "datasets": [{"id": a.dataset, "title": f"{len(gses)} studies, {len(st)} samples with sex in GEO metadata",
                      "accessions": {"gse": ";".join(gses), "gsm": list(st["gsm"])},
                      "truth": "sex in GEO sample characteristics", "n_cells": int(n)}],
        "resources": resources,
        "caveats": ["Counts are the atlas counts served today (pre-1.0 counting engine). The sex model only reads XIST and "
                    "chrY gene counts, which the 1.0 double-count fix changes by a few percent; re-run on 1.0 counts after "
                    "the atlas re-run."],
        "singlet_binary_seen": sb,
    }
    (a.artifacts / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"{len(st)} samples, {n:,} cells -> {a.artifacts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
