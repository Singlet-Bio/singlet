#!/usr/bin/env python3
"""Phase table of one nsys trace of a singlet run (PROFILE.md): NVTX ranges per process, kernels and copies per process.

STAR runs in a child that the pipeline forks without exec and that leaves with _exit(). nsys (--trace-fork-before-exec=true) records the
child's NVTX ranges (mapDistinct's phases, the device lock) but none of its kernels or copies, so the alignment kernels come from the
replay harness (bench/gpu/replay) and this table gives the child's phases by wall time only.

  nvtx_phases.py prof.sqlite [--label NAME] [--tsv OUT.tsv]
"""
import argparse, collections, sqlite3, sys

ap = argparse.ArgumentParser()
ap.add_argument("sqlite")
ap.add_argument("--label", default="")
ap.add_argument("--tsv")
a = ap.parse_args()
c = sqlite3.connect(a.sqlite)
q = lambda s: c.execute(s).fetchall()
tabs = {r[0] for r in q("select name from sqlite_master where type='table'")}
t0 = q("select min(start) from (select start from NVTX_EVENTS union all select start from CUPTI_ACTIVITY_KIND_RUNTIME)")[0][0]
pid = lambda col: f"(({col} >> 24) & 16777215)"
names = dict(q("select id, value from StringIds")) if "StringIds" in tabs else {}
cols = [r[1] for r in q("pragma table_info(NVTX_EVENTS)")]
textid = "textId" if "textId" in cols else None
rows = q(f"select {pid('globalTid')}, start, end, text{', textId' if textid else ''} from NVTX_EVENTS where end is not null and end > start")
by = collections.defaultdict(lambda: [0, 0.0, None, 0.0])   # (pid, name) -> count, summed s, first start s, last end s
procs = set()
for r in rows:
    p, s, e, txt = r[0], r[1], r[2], r[3]
    if txt is None and textid: txt = names.get(r[4], "?")
    k = (p, txt or "?"); procs.add(p)
    b = by[k]; b[0] += 1; b[1] += (e - s) / 1e9
    b[2] = (s - t0) / 1e9 if b[2] is None else min(b[2], (s - t0) / 1e9); b[3] = max(b[3], (e - t0) / 1e9)
star = {p for (p, n) in by if n.startswith("mapDistinct")}
kern = {r[0]: r[1:] for r in q(f"select {pid('globalPid')}, count(*), sum(end-start)/1e9, (min(start)-{t0})/1e9, (max(end)-{t0})/1e9 from CUPTI_ACTIVITY_KIND_KERNEL group by 1")} if "CUPTI_ACTIVITY_KIND_KERNEL" in tabs else {}
cp = {r[0]: r[1:] for r in q(f"select {pid('globalPid')}, count(*), sum(bytes)/1e9, sum(end-start)/1e9 from CUPTI_ACTIVITY_KIND_MEMCPY group by 1")} if "CUPTI_ACTIVITY_KIND_MEMCPY" in tabs else {}
print(f"## {a.label or a.sqlite}")
for p in sorted(procs | set(kern)):
    role = "STAR child (fork, no exec)" if p in star else "pipeline"
    k = kern.get(p); m = cp.get(p)
    print(f"process {p} ({role}): kernels {k[0] if k else 0} launches, {k[1] if k else 0:.2f} s summed"
          + (f", {k[2]:.1f} to {k[3]:.1f} s" if k else "") + (f"; copies {m[0]}, {m[1]:.2f} GB, {m[2]:.2f} s" if m else "; copies 0"))
print("| process | range | count | summed s | first start s | last end s |")
print("|---|---|---|---|---|---|")
out = []
for (p, n), b in sorted(by.items(), key=lambda kv: (kv[0][0] not in star, kv[1][2])):
    role = "STAR" if p in star else "pipeline"
    print(f"| {role} | {n} | {b[0]} | {b[1]:.2f} | {b[2]:.2f} | {b[3]:.2f} |")
    out.append((a.label, role, n, b[0], round(b[1], 3), round(b[2], 3), round(b[3], 3)))
if a.tsv:
    with open(a.tsv, "w") as f:
        f.write("label\tprocess\trange\tcount\tsummed_s\tfirst_start_s\tlast_end_s\n")
        for r in out: f.write("\t".join(map(str, r)) + "\n")
