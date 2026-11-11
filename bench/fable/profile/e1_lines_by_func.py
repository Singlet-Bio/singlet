#!/usr/bin/env python3
"""E1 profile: fold the per-line stall table (e1_ncu_lines.csv from ncu_summary.py) into functions of stitch.cuh.

Usage: e1_lines_by_func.py STITCH_CUH LINES_CSV [LINES_CSV ...]
A line belongs to the last function that starts at or before it (function starts: lines of stitch.cuh that open an HD
function or method). Prints, per label and kernel, the share of warp-stall samples per function, the average active
threads per executed instruction and the stall mix of the function; writes e1_ncu_by_func.csv next to the first CSV.
"""
import csv, os, re, sys
from collections import defaultdict

src = open(sys.argv[1], errors="replace").read().splitlines()
start = re.compile(r"^(template <[^>]*>\s*)?(HD|__device__)[^;]*?\b([A-Za-z_]\w*)\s*\(|^\s+HD [\w:<>, ]+?\b([A-Za-z_]\w*)\s*\(")
starts = []
for i, l in enumerate(src, 1):
    m = start.match(l)
    if m:
        starts.append((i, m.group(3) or m.group(4)))
starts.sort()


def func(file, line):
    if file != "stitch.cuh":
        return file
    name = "?"
    for s, n in starts:
        if s <= line:
            name = n
        else:
            break
    return name


rows = []
for p in sys.argv[2:]:
    rows += list(csv.DictReader(open(p)))
agg = defaultdict(lambda: defaultdict(float))
tot = defaultdict(float)
for r in rows:
    key = (r["label"], r["kernel"])
    f = func(r["file"], int(r["line"]))
    a = agg[key + (f,)]
    s = float(r["samples"] or 0)
    a["samples"] += s
    a["inst"] += float(r["inst_executed"] or 0)
    a["tinst"] += float(r["inst_executed"] or 0) * float(r["threads_per_inst"] or 0)
    a["l2g"] += float(r["l2_sectors_global"] or 0)
    a["l2l"] += float(r["l2_sectors_local"] or 0)
    for st in (r["stall1"], r["stall2"]):
        if st:
            name, pct = st.rsplit(" ", 1)
            a["st_" + name] += s * float(pct.rstrip("%")) / 100
    tot[key] += s
out = os.path.join(os.path.dirname(sys.argv[2]), "e1_ncu_by_func.csv")
with open(out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["label", "kernel", "function", "share_pct", "threads_per_inst", "l2_sectors_global", "l2_sectors_local", "stall_mix"])
    for key in sorted(tot):
        fs = sorted(((k[2], v) for k, v in agg.items() if k[:2] == key), key=lambda x: -x[1]["samples"])
        print("== %s %s (samples %d)" % (key[0], key[1], tot[key]))
        for f, a in fs:
            sh = 100 * a["samples"] / tot[key] if tot[key] else 0
            if sh < 0.5:
                continue
            stl = sorted(((k[3:], v) for k, v in a.items() if k.startswith("st_")), key=lambda x: -x[1])[:3]
            mix = ", ".join("%s %.0f%%" % (n, 100 * v / a["samples"]) for n, v in stl if a["samples"])
            tpi = a["tinst"] / a["inst"] if a["inst"] else 0
            print("   %5.1f%%  %-34s thr/inst %5.2f  L2 sectors global %.3g local %.3g  [%s]" % (sh, f, tpi, a["l2g"], a["l2l"], mix))
            w.writerow([key[0], key[1], f, round(sh, 2), round(tpi, 2), int(a["l2g"]), int(a["l2l"]), mix])
print("wrote", out)
