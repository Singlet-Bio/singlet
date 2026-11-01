#!/usr/bin/env python3
"""Core-seconds per pipeline stage from a stage_profile.sbatch run.

Inputs per engine E: log_E.txt (every engine log line prefixed with the wall clock) and cpu_E.txt (wall clock, the job cgroup's
cumulative CPU in microseconds, sampled every second). A stage starts at the first log line matching its marker and ends where the
next stage starts; its core-seconds are the cgroup counter's increase over that interval (linear between samples).

usage: stage_profile.py RESULT_DIR [read_pairs_millions]
"""
import bisect
import json
import re
import sys

MARKERS = [  # (regex on the log line, stage). Stream path: STAR (decode in-process), the serial pileup and the callers overlap between "Pileup: streaming" and "Pileup: N reads".
    (r"\[1fq\] ?\S*: \d+ reads|\.1fq: \d+ reads", "1fq header, pass 1 (CB histogram)"),
    (r"\[singlet\] \.1fq decoded", "barcode dictionary, references"),
    (r"\[singlet\] References loaded", "STAR start, genome load"),
    (r"\[singlet\] Pileup: streaming from", "STAR map + serial pileup (concurrent)"),
    (r"\[singlet\] Pileup: \d+ reads", "STAR/pileup drain"),
    (r"\[singlet\] STAR completed", "non-host screen"),
    (r"\[singlet\] Exporting feature matrices", "export"),
]


def cpu_at(ts, us, t):
    i = bisect.bisect_left(ts, t)
    if i <= 0:
        return us[0]
    if i >= len(ts):
        return us[-1]
    f = (t - ts[i - 1]) / (ts[i] - ts[i - 1])
    return us[i - 1] + f * (us[i] - us[i - 1])


def profile(d, eng):
    ts, us = [], []
    for line in open(f"{d}/cpu.txt"):
        p = line.split()
        if len(p) == 2:
            ts.append(float(p[0])); us.append(float(p[1]) / 1e6)
    marks, first, last = [], None, None
    for line in open(f"{d}/log_{eng}.txt", errors="replace"):
        p = line.split(" ", 1)
        try:
            t = float(p[0])
        except ValueError:
            continue
        first = first if first is not None else t
        last = t
        for rx, name in MARKERS:
            if re.search(rx, p[1] if len(p) > 1 else ""):
                if not marks or marks[-1][1] != name:
                    marks.append((t, name))
                break
    bounds = [(first, "start (arguments, probe)")] + marks + [(last, None)]
    rows = []
    for (t0, name), (t1, _) in zip(bounds, bounds[1:]):
        rows.append({"stage": name, "wall_s": round(t1 - t0, 1), "core_s": round(cpu_at(ts, us, t1) - cpu_at(ts, us, t0), 1)})
    return rows


def main():
    d = sys.argv[1]
    m = float(sys.argv[2]) if len(sys.argv) > 2 else None
    out = {}
    for eng in ("1",):
        try:
            rows = profile(d, eng)
        except FileNotFoundError:
            continue
        out[eng] = rows
        tot_c = sum(r["core_s"] for r in rows); tot_w = sum(r["wall_s"] for r in rows)
        print(f"== --engine {eng}: wall {tot_w:.0f} s, {tot_c:.0f} core-s" + (f" ({tot_c / m:.1f} core-s per M read pairs)" if m else ""))
        for r in rows:
            per = f"{r['core_s'] / m:6.2f}" if m else ""
            print(f"  {r['stage']:<52} wall {r['wall_s']:7.1f}  core-s {r['core_s']:8.1f} {per}  ({100 * r['core_s'] / tot_c:4.1f}%)")
    json.dump(out, open(f"{d}/stages.json", "w"), indent=1)


if __name__ == "__main__":
    main()
