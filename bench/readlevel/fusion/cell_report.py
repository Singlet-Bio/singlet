# SPDX-License-Identifier: MIT
"""B6 held-out: per-caller BCR-ABL1 cell counts under each label set (clause 2's inputs in full) and each caller's BCR-ABL1 cells
against the PacBio BCR::ABL1 cells over all genotyped barcodes (the secondary check).

    python cell_report.py --engine chimeric_junctions.tsv.gz --starfusion sf.tsv --arriba fusions.tsv --readmap readmap.tsv \
        --labels pacbio=labels_pacbio.tsv,kit=labels_kit.tsv,hashtag=labels.tsv --pacbio pacbio.csv.gz --out report.json
"""
import argparse, csv, gzip, json
import fusion_eval as fe

P = fe.pair("BCR", "ABL1")
ap = argparse.ArgumentParser()
for a in ("engine", "starfusion", "arriba", "readmap", "labels", "pacbio", "out"):
    ap.add_argument("--" + a.replace("_", "-"), required=True)
x = ap.parse_args()
sf, sf_reads = fe.starfusion_calls(x.starfusion)
ar, ar_reads = fe.arriba_calls(x.arriba)
rm = fe.load_readmap(x.readmap)
rows = list(csv.DictReader(gzip.open(x.pacbio, "rt")))
genotyped = {r["Cell_barcode"].split("-")[0] for r in rows}
pos = {r["Cell_barcode"].split("-")[0] for r in rows if float(r["BCR_ABL1"] or 0) > 0}
res = {"labels": {}}
for spec in x.labels.split(","):
    name, path = spec.split("=")
    labels = fe.load_labels(path); cells = set(labels)
    per = {"engine": fe.engine_calls(x.engine, cells)[1], "starfusion": fe.comparator_per_cell(sf_reads, rm, cells),
           "arriba": fe.comparator_per_cell(ar_reads, rm, cells)}
    res["labels"][name] = {t: fe.cell_metrics(pc, labels, {"Ku812": [P]})[P] for t, pc in per.items()}
# every PacBio-genotyped barcode: every caller's BCR-ABL1 cells, and how many have a PacBio BCR::ABL1 read
per = {"engine": fe.engine_calls(x.engine, genotyped)[1], "starfusion": fe.comparator_per_cell(sf_reads, rm, genotyped), "arriba": fe.comparator_per_cell(ar_reads, rm, genotyped)}
res["genotyped_cells"] = {}
for t, pc in per.items():
    c = sorted(b for b, d in pc.items() if d.get(P, 0) > 0)
    res["genotyped_cells"][t] = {"cells": len(c), "with_pacbio_bcr_abl1": sum(b in pos for b in c), "barcodes": c}
res["pacbio_genotyped_cells"] = len(genotyped); res["pacbio_bcr_abl1_cells"] = len(pos)
json.dump(res, open(x.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items()}, indent=1)[:6000])
