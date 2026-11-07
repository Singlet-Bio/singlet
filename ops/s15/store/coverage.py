#!/usr/bin/env python3
"""Dataset x tool coverage of the benchmark store: applicable tools per dataset (species, chemistry, modalities) against index.tsv.
usage: coverage.py STORE [--queued-ids file-with-dataset-ids]   writes STORE/coverage.tsv and prints the gaps per tool"""
import csv, sys, collections
store = sys.argv[1]
queued = set(open(sys.argv[3]).read().split()) if len(sys.argv) > 3 and sys.argv[2] == "--queued-ids" else set()
rows = list(csv.DictReader(open(store + "/datasets.tsv", encoding="utf-8"), delimiter="\t"))
have = collections.defaultdict(set)
for r in csv.DictReader(open(store + "/index.tsv", encoding="utf-8"), delimiter="\t"):
    have[r["dataset"]].add(r["tool"])
K = {"kang2018_b1_a", "kang2018_b1_b", "kang2018_b1_c", "kang2018_ctrl", "kang2018_stim", "pilot_GSM4987148", "pilot_GSM4987144", "pilot_GSM8085472"}
tools = ["STARsolo", "rtools", "edrops", "cellbender", "kraken", "alevin", "kb", "mgatk", "vireo", "souporcell", "trust4"]


def applicable(d):
    sp = d["species"]; chem = d["chemistry"]; mods = d["modalities"]; human = sp == "human"
    app = ["STARsolo", "rtools", "edrops", "cellbender", "kraken"]
    if human and "v1" not in chem: app += ["alevin", "kb"]
    if human: app += ["mgatk"]
    if human and d["dataset_id"] in K: app += ["vireo", "souporcell"]
    if human and "5p" in chem and "vdj" in mods: app += ["trust4"]
    return app


out = []; gaps = collections.defaultdict(list)
for d in rows:
    if d["fetch"] not in ("sra", "url", "bam", "pilot"): continue
    ds = d["dataset_id"]; hv = have.get(ds, set())
    for t in applicable(d):
        st = "done" if t in hv else ("queued" if ds in queued else "missing")
        out.append((ds, t, st))
        if st == "missing": gaps[t].append(ds)
w = csv.writer(open(store + "/coverage.tsv", "w", newline="", encoding="utf-8"), delimiter="\t", lineterminator="\n"); w.writerow(["dataset", "tool", "status"]); w.writerows(out)
c = collections.Counter(s for _, _, s in out); print(dict(c))
for t, l in gaps.items(): print(t, len(l), " ".join(l))
