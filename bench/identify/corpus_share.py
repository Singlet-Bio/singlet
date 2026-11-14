#!/usr/bin/env python3
"""Share of the Beta corpus runs labeled 10x whose SRA layout has no read of 26 to 28 bases (tier 0 layout table).

usage: corpus_share.py TIER0_DIR      (shard-*.jsonl, one record per run: read_lengths, text_family, text_version, spots, sample)
A run is 10x-labeled when text_family is 10x (the depositor's text). The barcode read of every 10x 3' and 5' kit that identify can
name is 26 (v1 deposits aside, v2) or 28 (v3, v4, 5' v3) bases. Index reads (8 or fewer) do not count.
"""
import glob, json, os, sys
from collections import Counter, defaultdict

d = sys.argv[1]
runs = []
for f in sorted(glob.glob(os.path.join(d, "shard-*.jsonl"))):
    for line in open(f):
        r = json.loads(line)
        if r.get("text_family") == "10x":
            runs.append(r)

def has_bc(r):
    return any(26 <= l <= 28 for l in (r.get("read_lengths") or []))

def row(label, rs):
    n = len(rs)
    miss = [r for r in rs if not has_bc(r)]
    sp = sum(r.get("spots") or 0 for r in rs)
    spm = sum(r.get("spots") or 0 for r in miss)
    smp = defaultdict(list)
    for r in rs:
        smp[r.get("sample") or r["run"]].append(has_bc(r))
    s_all_missing = sum(1 for v in smp.values() if not any(v))
    s_some_missing = sum(1 for v in smp.values() if not all(v))
    return (label, n, len(miss), len(miss) / max(1, n), len(smp), s_all_missing, s_some_missing, s_all_missing / max(1, len(smp)), sp, spm, spm / max(1, sp))

groups = defaultdict(list)
for r in runs:
    groups[r.get("text_version") or "unstated"].append(r)
print("label\truns\tno_26_28\tshare_runs\tsamples\tsamples_all_runs_no_26_28\tsamples_some_run_no_26_28\tshare_samples\tspots\tspots_no_26_28\tshare_spots")
rows = [row("all 10x-labeled", runs)] + [row(k, v) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
for x in rows:
    print("\t".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in x))
print()
lens = Counter(tuple(r.get("read_lengths") or []) for r in runs if not has_bc(r))
print("most common layouts among the 10x-labeled runs without a 26 to 28 read:")
for k, v in lens.most_common(12):
    print(v, list(k))
print("t0_class of those runs:", dict(Counter(r.get("t0_class") for r in runs if not has_bc(r)).most_common(8)))
print("with spots missing:", sum(1 for r in runs if not r.get("spots")), "of", len(runs))
