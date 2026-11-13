#!/usr/bin/env python3
"""Turn heldout_candidates.tsv into the scoring files: panel_heldout.tsv (labels in the registry vocabulary) and sources_heldout.tsv (what
`singlet identify --list` reads: id, source, hint, genome index).

Species and the single-cell hint come from ENA run metadata only (scientific_name, library_source), never from reads or from the labels:
the hint is single_cell=1 when library_source names a single-cell library or the archive text quoted in label_source names a single-cell protocol (never for the bulk rows). Genome index: GRCh38 for Homo sapiens, GRCm39 for Mus musculus, none otherwise.
Label mapping: 10x-5p-vdj rows are labeled vdj/tcr and 10x-flex rows flex/10x-flex (their verdict family differs from 10x 3'/5' by design);
v3.1 is v3; probe Visium variants are visium/10x-visium with the variant in label_note.

usage: build_heldout_panel.py HELDOUT_CANDIDATES.tsv OUTDIR
"""
import csv, subprocess, sys, time

src, out = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(src, newline=''), delimiter='\t'))
REG = {"10x-atac": "scatac", "hto": "cite-hashtag", "adt": "cite-adt", "marsseq": "marsseq2", "celseq2-u6b8": "celseq2", "10x-3p-v3.1": "10x-3p-v3",
       "10x-visium-cytassist-probe": "10x-visium", "10x-visium-ffpe-probe": "10x-visium", "10x-5p-vdj": "tcr", "10x-flex": "10x-flex"}


def ena(run):
    url = "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&fields=run_accession,scientific_name,library_source,library_strategy&format=tsv" % run
    for _ in range(3):
        p = subprocess.run(["curl", "-s", "--max-time", "60", url], capture_output=True)
        t = p.stdout.decode("utf-8", "replace").replace("\r", "").splitlines()
        if len(t) >= 2:
            h = t[0].split("\t"); v = t[1].split("\t")
            return dict(zip(h, v))
        time.sleep(2)
    return {}


panel, sources = [], []
for r in rows:
    run = r["accession"]
    m = ena(run)
    sp = m.get("scientific_name", "")
    txt = r["label_source"].lower()
    named = any(k in txt for k in ("single cell", "single-cell", "chromium", "drop-seq", "split-pool", "sci-rna", "smart-seq", "cel-seq", "mars-seq", "scrb-seq",
                                   "cite-seq", "hto", "multiome", "visium", "fixed rna profiling", "scatac", "10x"))
    sc = "single_cell=1" if (not r["family"] == "bulk_rna") and ("SINGLE CELL" in m.get("library_source", "").upper() or named) else ""
    fam, proto = r["family"], r["protocol"]
    if proto == "10x-5p-vdj": fam = "vdj"
    if proto == "10x-flex": fam = "flex"
    if proto.startswith("bulk"): proto = "generic-negative"
    label_note = r["protocol"] if r["protocol"] != proto and r["protocol"] not in REG else ""
    if r["protocol"] in ("10x-visium-cytassist-probe", "10x-visium-ffpe-probe"): label_note = r["protocol"]
    proto = REG.get(proto, proto)
    genome = "$REF/GRCh38-2024-A/star_2.7.11b" if sp == "Homo sapiens" else ("$REF/GRCm39-2024-A/star_2.7.11b" if sp == "Mus musculus" else "-")
    panel.append([run, run, fam, proto, sp, r["label_source"], "H", "1" if sc else "0", label_note, r["note"]])
    sources.append([run, run, sc, genome])
    print(run, fam, proto, sp or "?", m.get("library_source", "?"), file=sys.stderr)
with open(out + "/panel_heldout.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerow(["id", "source", "family", "protocol", "species", "label_source", "tier", "single_cell_hint", "label_note", "layout_note"])
    w.writerows(panel)
with open(out + "/sources_heldout.tsv", "w", newline="") as f:
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    w.writerows(sources)
print(len(panel), "rows")
