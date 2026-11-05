#!/usr/bin/env python3
"""Whole-run measures of one throughput point, as one results.tsv row.

Usage: tp_point.py POINT_DIR
Run after bench/fable/throughput/e18_point.py (point.json) and the identity step (ident.tsv).
Provenance comes from the environment: PROV_JOB, PROV_NODE, PROV_BIN, PROV_SHA256, PROV_COMMIT, PROV_SCRIPTS.
Writes POINT_DIR/results.tsv (header and one row), POINT_DIR/fallback.txt, prints the windowed rate and the point verdict.

Windowed rate. Window start is the time the k-th run completes. Window end is the start time of the last run.
All k workers are busy in the window, and the cold-start first wave is excluded.
Rate = pairs of the runs that complete inside the window / window length. Completions are the runs with start < end inside (ws, we].
"""
import bisect
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fallback  # noqa: E402

D = sys.argv[1].rstrip("/")
P = json.load(open(f"{D}/point.json"))
runs = P["runs"]
k = len({r["worker"] for r in runs})
ok = [r for r in runs if r["rc"] == 0]
first, last = P["first_start"], P["last_end"]
span = last - first
pairs = {r["item"]: (r["pairs"] or 0) / 1e6 for r in runs}

# samples: cgroup CPU over time
cg = []
if os.path.exists(f"{D}/samples.jsonl"):
    for l in open(f"{D}/samples.jsonl"):
        try:
            s = json.loads(l)
        except ValueError:
            continue
        if s.get("cpu") is not None:
            cg.append((s["t"], s["cpu"]))
ts = [t for t, _ in cg]


def cpu_at(t):
    if not cg:
        return None
    i = bisect.bisect_left(ts, t)
    c = [j for j in (i - 1, i) if 0 <= j < len(cg)]
    return cg[min(c, key=lambda j: abs(ts[j] - t))][1]


def na(v, f="{:.3f}"):
    return "NA" if v is None else f.format(v)


# span rate and the old steady number (the formula of e18_steady.py)
tot = sum(pairs[r["item"]] for r in runs)
span_rate = tot / span if span > 0 else None
ends = sorted(r["end"] for r in runs)
steady = None
later = [r for r in runs if r["start"] >= ends[0] - 1.0]
if later and len(runs) > k:
    mw = sum(r["end"] - r["start"] for r in later) / len(later)
    steady = k * sum(pairs[r["item"]] for r in later) / len(later) / mw

# windowed rate
win_rate = win_n = win_s = cores_win = None
okends = sorted(r["end"] for r in ok)
if len(okends) >= k and runs:
    ws, we = okends[k - 1], max(r["start"] for r in runs)
    win_s = we - ws
    if win_s > 0:
        inside = [r for r in ok if ws < r["end"] <= we + 1e-6]
        win_n = len(inside)
        win_rate = sum(pairs[r["item"]] for r in inside) / win_s
        a, b = cpu_at(ws), cpu_at(we)
        cores_win = (b - a) / win_s if a is not None and b is not None else None
    else:
        win_n = 0
cs = (cpu_at(last) - cpu_at(first)) if cg else None
cores_span = cs / span if cs is not None and span > 0 else None
core_s_per_M = P.get("cgroup_core_s_per_M") or None

# lock wait and hold (sampler, per run)
n = len(runs)
mean_wall = sum(r["wall"] for r in runs) / n
lock_wait = sum(r["sampler_wait_s"] for r in runs) / n
lock_held = sum(r["sampler_hold_s"] for r in runs) / n

# fallbacks
fb_lines = []
fb_runs = 0
for r in sorted(runs, key=lambda r: r["item"]):
    why = []
    if r["rc"] != 0:
        why.append(f"exit {r['rc']}")
    if not r.get("level5"):
        why.append("level 5 NO")
    if not r.get("gpu_pileup_ok"):
        why.append("pileup NO")
    bad, allowed = fallback.scan(f"{D}/{r['copy']}")
    why += [f"{src}: {l[:140]}" for src, l in bad[:3]]
    if why:
        fb_runs += 1
    fb_lines.append(f"{r['copy']}\t{'FALLBACK' if why else 'ok'}\t{' | '.join(why)}" + (f"\t(allowed info lines: {len(allowed)})" if allowed else ""))
open(f"{D}/fallback.txt", "w").write("\n".join(fb_lines) + "\n")

# identity
idn = {"alone": [0, 0], "cpu": [0, 0]}
if os.path.exists(f"{D}/ident.tsv"):
    for l in open(f"{D}/ident.tsv"):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 3 and f[1] in idn:
            idn[f[1]][1] += 1
            idn[f[1]][0] += 1 if f[2] == "1" else 0

# co-tenants: the other jobs on the node, the most in any minute
other = 0
cur = 0
mine = os.environ.get("PROV_JOB", "")
if os.path.exists(f"{D}/cotenants.log"):
    for l in open(f"{D}/cotenants.log", errors="replace"):
        if l.startswith("##"):
            other = max(other, cur)
            cur = 0
        elif l.strip() and not l.startswith("JOBID") and l.split()[0].split("_")[0] != mine:
            cur += 1
    other = max(other, cur)

cols = [
    ("job", os.environ.get("PROV_JOB", "?")), ("node", os.environ.get("PROV_NODE", P.get("node", "?"))), ("bin", os.environ.get("PROV_BIN", "?")),
    ("bin_sha256", os.environ.get("PROV_SHA256", "?")), ("build_commit", os.environ.get("PROV_COMMIT", "?")),
    ("scripts_commit", os.environ.get("PROV_SCRIPTS", "?")), ("T", P["T"]),
    ("point", P["tag"]), ("k", P["k"]), ("slots", P["slots"]), ("runs", n), ("failed_runs", n - len(ok)),
    ("span_s", na(span, "{:.1f}")), ("span_rate", na(span_rate)), ("steady_rate", na(steady)), ("win_rate", na(win_rate)),
    ("win_n", "NA" if win_n is None else win_n), ("win_s", na(win_s, "{:.1f}")),
    ("core_s_per_M", na(core_s_per_M, "{:.2f}")), ("cores_span", na(cores_span, "{:.1f}")), ("cores_win", na(cores_win, "{:.1f}")),
    ("lock_wait_s", na(lock_wait, "{:.1f}")), ("lock_wait_frac", na(lock_wait / mean_wall if mean_wall else None, "{:.3f}")),
    ("lock_held_s", na(lock_held, "{:.1f}")), ("gpu_util", na(P.get("gpu_util_mean"), "{:.0f}")),
    ("host_peak_gb", na(P.get("host_peak_gb"), "{:.0f}")), ("dev_peak_gb", na(P.get("dev_peak_gb"), "{:.1f}")),
    ("fallback_runs", fb_runs), ("ident_alone", f"{idn['alone'][0]}/{idn['alone'][1]}"), ("ident_cpu", f"{idn['cpu'][0]}/{idn['cpu'][1]}"),
    ("other_jobs_max", other), ("mean_wall_s", na(mean_wall, "{:.1f}")),
]
open(f"{D}/results.tsv", "w").write("\t".join(c for c, _ in cols) + "\n" + "\t".join(str(v) for _, v in cols) + "\n")

print(f"== results {P['tag']}: span {na(span, '{:.1f}')} s, span rate {na(span_rate)} M pairs/s, steady {na(steady)}")
if win_rate is not None:
    print(f"   windowed: ({ws - first:.1f}, {we - first:.1f}] s = {win_s:.1f} s, {win_n} completions, {win_rate:.3f} M pairs/s; "
          f"cores busy {na(cores_win, '{:.1f}')} in the window against {na(cores_span, '{:.1f}')} over the span")
else:
    print(f"   windowed: NA ({'the window is empty, the last run starts at or before the k-th completion' if win_s is not None else 'fewer than k completions'}; "
          f"use more runs than 2k for a window)")
print(f"   lock wait {lock_wait:.1f} s per run = {100 * lock_wait / mean_wall:.1f} % of the mean wall {mean_wall:.1f} s, held {lock_held:.1f} s; "
      f"other jobs on the node, most in any minute: {other}")
print(f"   identity: alone {idn['alone'][0]}/{idn['alone'][1]}, cpu {idn['cpu'][0]}/{idn['cpu'][1]} (identical / checked)")
for l in fb_lines:
    if "\tFALLBACK\t" in l:
        print("   fallback run " + l.replace("\t", " "))
if fb_runs:
    print(f"POINT FAILED: {fb_runs} fallback runs")
