# SPDX-License-Identifier: MIT
"""Benchmark artifacts of the ``vdj`` capability (bench/public CONTRACT.md section 5) from the Clipper runs.

    python -P make_artifacts.py RUNS_DIR DATA_DIR OUT_DIR --commit SHA

Per library (``LIBRARIES``): the productive contigs of every tool as one small table each (``<lib>.<tool>.csv.gz``:
barcode, chain, cdr3_nt, cdr3, v, j, c, umis), the reference (Cell Ranger's filtered contigs, ``<lib>.cellranger.csv.gz``
with its cells in ``<lib>.cellranger_cells.txt``), the called GEX cells and their T/B lineage when the library has a
GEX matrix. ``provenance.json``: commit, tools, wall time and peak memory of each run, the V gene equivalence classes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vdj as CV  # noqa: E402

# library id -> (cohort dataset id, chains). pbmc5p_v1 and pbmc10k_5p_v2 are development data (the assembler was
# debugged on them); GSE145370 S133A / S158A too, so they stay out of the contract. The GSE145370 libraries downloaded
# after the code was frozen (bench/vdj/hpc/download_holdout.sbatch) and 10x's Chromium Controller set are held out.
LIBRARIES = {
    "pbmc2_t": ("pbmc5p_v1", "TCR"),
    "pbmc2_b": ("pbmc5p_v1", "BCR"),
    "pbmc10k_t": ("pbmc10k_5p_v2", "TCR"),
    "pbmc10k_b": ("pbmc10k_5p_v2", "BCR"),
    "ctrl10k_t": ("tenx_pbmc10k_5pv2_controller", "TCR"),
    "ctrl10k_b": ("tenx_pbmc10k_5pv2_controller", "BCR"),
}
DEVELOPMENT_GSE = {"gse145370_S133A", "gse145370_S158A"}


def libraries(data_dir):
    out = dict(LIBRARIES)
    for d in sorted(os.listdir(data_dir)):
        if d.startswith("gse145370_") and d not in DEVELOPMENT_GSE and os.path.isdir(os.path.join(data_dir, d)):
            out[d] = ("gse145370_tcr", "TCR")
    return out


def gnu_time(path):
    out = {}
    if not os.path.exists(path):
        return out
    for ln in open(path, errors="replace"):
        if "Elapsed (wall clock)" in ln:
            t = ln.rsplit(" ", 1)[-1].strip().split(":")
            sec = 0.0
            for x in t:
                sec = sec * 60 + float(x)
            out["wall_s"] = sec
        elif "Maximum resident set size" in ln:
            out["max_rss_gb"] = int(ln.rsplit(" ", 1)[-1]) / 1e6
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--vdj-ref", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    from singlet.vdj import reference as VR

    prov = {"commit": a.commit, "libraries": {}, "resources": [],
            "equivalence": {CV.norm_gene(k): v for k, v in VR.equivalence(VR.load(a.vdj_ref)).items() if "|" in v}}
    cols = ["barcode", "chain", "cdr3_nt", "cdr3", "v", "j", "c", "umis"]
    for lib, (ds, chains) in libraries(a.data).items():
        R, D = os.path.join(a.runs, lib), os.path.join(a.data, lib)
        ref_csv = os.path.join(D, "filtered_contig_annotations.csv")
        if not os.path.exists(ref_csv) or not os.path.exists(os.path.join(R, "singlet", "contigs.csv")):
            continue  # not downloaded or not run yet
        keep = CV.TCR if chains == "TCR" else CV.BCR
        ref = CV.load_cellranger(ref_csv)
        ref[ref.chain.isin(keep)][cols].to_csv(os.path.join(a.out, f"{lib}.cellranger.csv.gz"), index=False)
        with open(os.path.join(a.out, f"{lib}.cellranger_cells.txt"), "w") as f:
            f.write("\n".join(sorted(CV.cellranger_cells(ref_csv))) + "\n")
        info = {"dataset": ds, "chains": chains, "tools": []}
        sc = os.path.join(R, "singlet", "contigs.csv")
        if os.path.exists(sc):
            t = CV.load_singlet(sc)
            t[t.chain.isin(keep)][cols].to_csv(os.path.join(a.out, f"{lib}.singlet.csv.gz"), index=False)
            info["tools"].append("singlet")
            run = json.load(open(os.path.join(R, "singlet", "run.json")))
            info["read_pairs"] = run["stats"]["read_pairs"]
            prov["resources"].append({"tool": "singlet", "dataset": ds, "library": lib, **gnu_time(os.path.join(R, "singlet.time")),
                                      "threads": a.threads})
        tr = os.path.join(R, "trust4", "TRUST_barcode_report.tsv")
        if os.path.exists(tr):
            t = CV.load_trust4(tr)
            t[t.chain.isin(keep)][cols].to_csv(os.path.join(a.out, f"{lib}.trust4.csv.gz"), index=False)
            info["tools"].append("trust4")
            prov["resources"].append({"tool": "trust4", "dataset": ds, "library": lib, **gnu_time(os.path.join(R, "trust4.time")),
                                      "threads": a.threads})
        for name in ("gex_cells.txt", "gex_lineage.csv"):
            p = os.path.join(R, name)
            if os.path.exists(p):
                os.system(f"cp {p} {os.path.join(a.out, f'{lib}.{name}')}")
        prov["libraries"][lib] = info
    with open(os.path.join(a.out, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)
    print(json.dumps({k: v for k, v in prov.items() if k != "equivalence"}, indent=1))


if __name__ == "__main__":
    main()
