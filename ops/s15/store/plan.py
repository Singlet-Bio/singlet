#!/usr/bin/env python3
"""S15 store planner for the node drivers.
  plan.py STORE queue            ids (one per line) that still miss a CPU-side tool, smallest first
  plan.py STORE info ID          shell assignments for one dataset: DIR KIND ACC CHEM SP MODS FM PM FIVE K MISSING NEED_READS NEED_BAM NEED_STAR
  plan.py STORE gpu-queue        ids whose STARsolo output exists and CellBender is missing
Applicable tools: STARsolo, rtools, edrops, kraken_pf for everything; kb and alevin for human non-v1; mgatk for human; vireo and souporcell for known donor counts (human);
trust4 for human 5' with V(D)J. CellBender runs on the GPU node (gpu-queue). A tool counts as done when index.tsv has a row for it."""
import csv, sys, collections, os

store, cmd = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(store + "/datasets.tsv", encoding="utf-8"), delimiter="\t"))
have = collections.defaultdict(set)
for r in csv.DictReader(open(store + "/index.tsv", encoding="utf-8"), delimiter="\t"):
    have[r["dataset"]].add(r["tool"])
K = {"kang2018_b1_a": 4, "kang2018_b1_b": 4, "kang2018_b1_c": 8, "kang2018_ctrl": 8, "kang2018_stim": 8, "pilot_GSM4987148": 4, "pilot_GSM4987144": 4, "pilot_GSM8085472": 10}
W = os.path.dirname(os.path.abspath(store))
byid = {r["dataset_id"]: r for r in rows if r["fetch"] in ("sra", "url", "bam", "pilot")}


def info(i):
    d = byid[i]; sp = d["species"]; chem = d["chemistry"]; mods = d["modalities"]; human = sp == "human"
    five = "5p" in chem
    app = ["rtools", "edrops", "kraken_pf"]
    if human and "v1" not in chem: app += ["kb", "alevin"]
    if human: app += ["mgatk"]
    if human and i in K: app += ["vireo", "souporcell"]
    if human and five and "vdj" in mods: app += ["trust4"]
    if i in ("hgmm1k_v3", "hgmm1k_v2", "hgmm10k_v3"): app += ["kb_barnyard"]   # species-mixture verdict: kb on the human+mouse index
    h = have.get(i, set())
    missing = [t for t in app if t not in h]
    need_star = "STARsolo" not in h or any(t in missing for t in ("mgatk", "vireo", "souporcell"))
    need_reads = need_star or any(t in missing for t in ("kb", "alevin", "trust4", "kraken_pf", "kb_barnyard"))
    kind = "pilot" if i.startswith("pilot_") else "store"
    return dict(DIR=i[6:] if kind == "pilot" else i, KIND=kind, ACC=d["accessions"], CHEM=chem, SP=sp, MODS=mods, FM=d["fetch"], PM=d["pairs_m"] or "100",
                FIVE="1" if five else "0", K=str(K.get(i, "")), MISSING=" ".join(missing), NEED_READS="1" if (missing or "STARsolo" not in h) and need_reads else "0",
                NEED_BAM="1" if any(t in missing for t in ("mgatk", "vireo", "souporcell")) else "0", NEED_STAR="1" if need_star and (missing or "STARsolo" not in h) else "0")


if cmd == "queue":
    out = []
    for i in byid:
        x = info(i)
        if x["MISSING"] or x["NEED_STAR"] == "1":
            try: pm = float(x["PM"])
            except ValueError: pm = 100.0
            out.append((pm, i))
    for _, i in sorted(out): print(i)
elif cmd == "info":
    for k, v in info(sys.argv[3]).items(): print("%s='%s'" % (k, v.replace("'", "")))
elif cmd == "gpu-queue":
    for i in byid:
        if "STARsolo" in have.get(i, set()) and "cellbender" not in have.get(i, set()): print(i)
