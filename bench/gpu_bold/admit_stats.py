#!/usr/bin/env python3
"""Admission statistics of one tp point (gpu/bold-admit). Usage: admit_stats.py POINT_DIR
Reads c*/run.log.ts. Prints per phase: requests, mean and max wait, mean estimate; per run: total wait; OOM retries; fallback lines."""
import glob
import re
import statistics as st
import sys

D = sys.argv[1].rstrip("/")
ph = {}
per_run = []
oom = []
fb = 0
runs = 0
for f in sorted(glob.glob(D + "/c*/run.log.ts")):
    runs += 1
    tot = 0.0
    bad = False
    for l in open(f, errors="replace"):
        m = re.search(r"\[admit\] pid \d+ (\w+) est ([\d.]+) GB, waited ([\d.e+-]+) s, others reserved ([\d.]+) GB in (\d+) entries", l)
        if m:
            p = ph.setdefault(m.group(1), {"w": [], "est": [], "others": [], "users": []})
            p["w"].append(float(m.group(3))); p["est"].append(float(m.group(2))); p["others"].append(float(m.group(4))); p["users"].append(int(m.group(5)))
            tot += float(m.group(3))
        m = re.search(r"\[admit\] cudaMalloc of ([\d.]+) GB hit device OOM, waited ([\d.]+) s for room \((.*?)\)", l)
        if m:
            oom.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        if re.search(r"ENGINE_GPU: not used|engine_gpu: .*(out of memory)|FELL BACK|fallback: ", l):
            bad = True
    fb += bad
    per_run.append(tot)
print(f"runs {runs}; runs with a fallback line {fb}; OOM retries {len(oom)}" + (f", waited {sum(x[1] for x in oom):.1f} s in total, gave up {sum(1 for x in oom if x[2] == 'gave up')}" if oom else ""))
print("| phase | requests | mean wait s | max wait s | mean est GB | mean others reserved GB at arrival | mean entries at arrival |")
print("|---|---|---|---|---|---|---|")
for k, v in ph.items():
    print(f"| {k} | {len(v['w'])} | {st.mean(v['w']):.2f} | {max(v['w']):.1f} | {st.mean(v['est']):.1f} | {st.mean(v['others']):.1f} | {st.mean(v['users']):.1f} |")
if per_run:
    print(f"total admission wait per run: mean {st.mean(per_run):.1f} s, max {max(per_run):.1f} s")
