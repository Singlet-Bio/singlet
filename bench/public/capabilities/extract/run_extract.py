# SPDX-License-Identifier: MIT
"""Write the artifacts of the public-benchmark capabilities from the tool outputs on Clipper.

    python -P bench/public/capabilities/extract/run_extract.py CAPABILITY DATASET --pb $WORK/pipeline-benchmark --out DIR

CAPABILITY: gex-counts | spliced-unspliced | junctions-psi | cell-calling | meta. ``meta`` writes resources.json and provenance.json
for one dataset (merged into any existing file in DIR). Only counts, junction tables and cell calls are read; Singlet's private/
directory (donor genotypes, allele counts) is never opened and nothing derived from it is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import runio as R  # noqa: E402
from counts import Ctx, extract_gex_counts, extract_spliced_unspliced  # noqa: E402

REF = Path("/mnt/projects/debruinz_project/cellarium/reference")
SPECIES = {"hgmm1k_v3": "GRCh38-GRCm39-barnyard", "neuron1k_v3": "GRCm39-2024-A", "nuclei2k_v2": "GRCm39-2024-A"}


def species(ds: str) -> str:
    return SPECIES.get(ds, "GRCh38-2024-A")


def _merge_json(path: Path, ds: str, value: dict) -> None:
    cur = json.loads(path.read_text()) if path.exists() else {}
    cur[ds] = value
    path.write_text(json.dumps(cur, indent=1, sort_keys=True) + "\n")


def meta(ctx: Ctx, out: Path) -> dict:
    res: dict = {}
    for key, run, threads in (("singlet", ctx.singlet, 8), ("starsolo", ctx.star_cr, 16), ("starsolo_default", ctx.star_default, 16),
                              ("alevinfry", ctx.af, 16), ("cellbender", ctx.runs / "cellbender-public", 8)):
        t = R.load_time(run / "time.txt")
        if not t:
            continue
        r = {"wall_s": t["wall_s"], "max_rss_gb": t["max_rss_gb"], "cpu_h": t.get("cpu_h"), "threads": threads}
        if key == "singlet":
            e = R.load_time(run / "encode.time.txt")
            if e:  # `singlet encode` is part of Singlet's cost
                r["wall_s"] += e["wall_s"]
                r["max_rss_gb"] = max(r["max_rss_gb"], e["max_rss_gb"])
                r["cpu_h"] = (r["cpu_h"] or 0) + (e.get("cpu_h") or 0)
                r["notes"] = ("includes singlet encode (FASTQ to .1fq); the run also does donor demultiplexing, mtDNA, ambient and "
                              "doublet steps that no competitor does")
        if key == "cellbender":
            r["notes"] = "GPU job"
        node = run / "node.txt"
        if node.exists():
            n = node.read_text().split(";")[0].replace("node:", "").strip()
            if n:
                r["node"] = n
        res[key] = r
    prov = {"reference": species(ctx.ds)}
    for key, run in (("singlet", ctx.singlet), ("starsolo_cr", ctx.star_cr), ("starsolo_default", ctx.star_default),
                     ("cellbender", ctx.runs / "cellbender-public")):
        for name in ("command.txt", "params.txt"):
            p = run / name
            if p.exists():
                prov[f"{key}_command"] = p.read_text().strip()
        v = run / "star_version.txt"
        if v.exists():
            prov[f"{key}_version"] = v.read_text().strip()
    summ = ctx.singlet / "summary.json"
    if summ.exists():
        s = json.loads(summ.read_text())
        prov["singlet_summary"] = {k: s.get(k) for k in ("n_input_reads", "n_cells_called", "median_umi_per_cell", "protocol_name",
                                                           "uniquely_mapped_pct")}
    binp = ctx.pb / "bin" / "singlet-campaign"
    if binp.exists():
        prov["singlet_binary_sha256"] = hashlib.sha256(binp.read_bytes()).hexdigest()
    _merge_json(out / "resources.json", ctx.ds, res)
    _merge_json(out / "provenance.json", ctx.ds, prov)
    return {"resources": sorted(res)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capability", choices=["gex-counts", "spliced-unspliced", "junctions-psi", "cell-calling", "meta"])
    ap.add_argument("dataset")
    ap.add_argument("--pb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ctx = Ctx(Path(a.pb), a.dataset)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.capability == "gex-counts":
        info = extract_gex_counts(ctx, out)
    elif a.capability == "spliced-unspliced":
        info = extract_spliced_unspliced(ctx, out)
    elif a.capability == "junctions-psi":
        from junctions import extract_junctions
        info = extract_junctions(ctx, out, REF / species(a.dataset) / "genes" / "genes.gtf")
    elif a.capability == "cell-calling":
        from cellcalling import extract_cell_calling, write_validation42
        info = extract_cell_calling(ctx, out)
        if a.dataset == "pbmc1k_v3":  # the pooled 42-sample table is written once
            info.update(write_validation42(out))
    else:
        info = meta(ctx, out)
    print(json.dumps({"capability": a.capability, "dataset": a.dataset, **info}, default=str))


if __name__ == "__main__":
    main()
