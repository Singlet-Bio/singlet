#!/usr/bin/env python3
"""datasets.tsv for the benchmark store: bench/public/cohort.tsv (prod/campaign) + the 13 pilot samples. Run on the laptop; the result is put to $WORK/benchmark-store/."""
import csv, io, subprocess, sys
raw = subprocess.run(["git", "show", "origin/prod/campaign:bench/public/cohort.tsv"], capture_output=True, text=True, check=True).stdout
rows = list(csv.DictReader(io.StringIO("\n".join(l for l in raw.splitlines() if not l.startswith("#"))), delimiter="\t"))
DOI = {"kang2018": "10.1038/nbt.4042", "stoeckius2018": "10.1186/s13059-018-1603-1", "xili": "10.1016/j.cels.2020.11.008", "ding2020": "10.1038/s41587-020-0465-8",
       "norman2019": "10.1126/science.aax4438", "replogle2022": "10.1016/j.cell.2022.05.013", "macosko2015": "10.1016/j.cell.2015.05.002", "ravindra2021": "10.1371/journal.pbio.3001143",
       "segerstolpe2016": "10.1016/j.cmet.2016.08.020", "tabula_muris": "10.1038/s41586-018-0590-4", "howitt2023": "", "osorio2019": "10.1016/j.celrep.2019.11.017", "gse145370": "", "lcl_gse230239": ""}
def doi(i):
    for k, v in DOI.items():
        if i.startswith(k): return v
    return "10x Genomics dataset page" if rows and any(r["id"] == i and r["source"] == "10x" for r in rows) else ""
def wave(r):
    ch = r["chemistry"]
    if not ch.startswith("10x") or ch in ("10x-visium", "10x-visium-ffpe", "10x-atac", "10x-arc"): return "later"
    if r["srr"] and ";" not in r["gsm"] and "see" not in r["srr"]: return "sra"
    if r["url"].endswith(".tar"): return "url"
    return "later"
out = []
for r in rows:
    out.append(dict(dataset_id=r["id"], accessions=";".join(x for x in (r["gse"], r["gsm"], r["srr"], r["url"]) if x), source_paper=r["group"], doi=doi(r["id"]), tissue_title=r["title"],
                    chemistry=r["chemistry"], modalities=r["capabilities"], truth_kind=r["truth"][:90], truth_where=r["truth"], species=r["species"], pairs_m=r["read_pairs_m"], cohort_tier=r["tier"], fetch=wave(r)))
PILOT = [("GSM5682562", "hashed 5p CD8 T cells", "10x-5p", "hto", 69), ("GSM8141311", "BAL, 3p", "10x-3p", "gex", 88), ("GSM9038586", "MDA-MB-231 xenograft", "10x-3p", "gex,human+mouse", 109),
         ("GSM5130155", "T cells 5p F", "10x-5p", "gex", 236), ("GSM5130163", "T cells 5p M", "10x-5p", "gex", 262), ("GSM8183378", "gastric epithelium", "10x-3p", "gex", 266),
         ("GSM4987144", "cell-line mix, 5p v1", "10x-5p-v1", "gex,donor-demux", 312), ("GSM8164511", "HIO organoid", "10x-3p", "gex", 355), ("GSM4987148", "cell-line mix, 3p v3 deep", "10x-3p-v3", "gex,donor-demux", 836),
         ("GSM5171610", "T cells 5p TCR, deep", "10x-5p", "gex,vdj", 851), ("GSM7064234", "retina snMultiome M", "10x-3p", "gex,snRNA", 1198), ("GSM7884577", "retina snMultiome F", "10x-3p", "gex,snRNA", 1404),
         ("GSM8085472", "10-donor hashed PBMC pool, lane 1", "10x-5p", "gex,hto,donor-demux", 1416)]
for g, t, c, m, p in PILOT:
    out.append(dict(dataset_id="pilot_" + g, accessions=g, source_paper="s15 pilot (held out: not used in dev/test/gates)", doi="", tissue_title=t, chemistry=c, modalities=m, truth_kind="see beta-pilot.md",
                    truth_where="docs/research-beta/beta-pilot.md", species="human", pairs_m=str(p), cohort_tier="heldout", fetch="pilot"))
with open(sys.argv[1] if len(sys.argv) > 1 else "datasets.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter="\t"); w.writeheader(); w.writerows(out)
from collections import Counter
print(len(out), Counter(o["fetch"] for o in out))
