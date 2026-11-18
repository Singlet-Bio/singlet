#!/usr/bin/env python3
"""CPU by thread name over a window of an rt-node run, from tsnap.sh's samples (every 2 s: pid tid comm ticks for each thread).

A thread present in the first sample contributes last - first ticks; a thread first seen later contributes its last ticks (all its
CPU: it was born inside the window). Threads that live less than one interval are never seen; the last line gives what the
per-process totals of procs.snap show beyond the threads seen (their share). Only the job's processes count (pids of procs.snap).
usage: thread_ledger.py <run dir> [core-s per item to scale the shares to]
"""
import collections, os, re, sys

run = sys.argv[1]
scale = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
HZ = 100.0
jobpids = set(); pcomm = {}
proc = collections.defaultdict(list)   # pid -> [(ts, ticks)]
ts = None
for ln in open(os.path.join(run, "procs.snap"), errors="replace"):
    c = ln.split()
    if not c: continue
    if c[0] == "S": ts = float(c[1]); continue
    if len(c) >= 4 and c[0].isdigit():
        jobpids.add(int(c[0])); proc[int(c[0])].append((ts, int(c[3])))

first, last, born0, name, pid_of = {}, {}, {}, {}, {}
si = -1; t0 = t1 = None
for ln in open(os.path.join(run, "threads2.snap"), errors="replace"):
    c = ln.split()
    if not c: continue
    if c[0] == "S":
        si += 1; t = float(c[1]); t0 = t if t0 is None else t0; t1 = t; continue
    if len(c) < 4 or not c[0].isdigit(): continue
    p, tid, tk = int(c[0]), int(c[1]), int(c[3])
    if p not in jobpids: continue
    if tid not in first: first[tid] = tk; born0[tid] = (si == 0)
    last[tid] = tk; pid_of[tid] = p
    name[tid] = re.sub(r"[-_.:/]?\d+$", "", c[2])   # the name at its last sighting

by = collections.defaultdict(float); tot = 0.0
for tid in first:
    d = (last[tid] - first[tid]) if born0[tid] else last[tid]
    if d <= 0: continue
    by[name[tid]] += d / HZ; tot += d / HZ
# the processes' own totals over the window
ptot = 0.0
for p, v in proc.items():
    a = [tk for (t, tk) in v if t <= t0]; b = [tk for (t, tk) in v if t <= t1]
    if not b: continue
    lo = a[-1] if a else 0
    ptot += max(0, b[-1] - lo) / HZ
print("window %.0f s, %d samples; threads seen: %.0f core-s (%.1f cores); processes: %.0f core-s (%.1f cores); not seen at thread level: %.0f%%"
      % (t1 - t0, si + 1, tot, tot / (t1 - t0), ptot, ptot / (t1 - t0), 100 * (1 - tot / ptot) if ptot else 0))
k = scale / ptot if (scale and ptot) else 0
print("%-18s %10s %7s%s" % ("thread name", "core-s", "share", "   per item" if k else ""))
for n, v in sorted(by.items(), key=lambda x: -x[1]):
    if v / ptot < 0.004: continue
    print("%-18s %10.0f %6.1f%%%s" % (n, v, 100 * v / ptot, "   %7.0f" % (v * k) if k else ""))
print("%-18s %10.0f %6.1f%%%s" % ("(short-lived)", ptot - tot, 100 * (ptot - tot) / ptot, "   %7.0f" % ((ptot - tot) * k) if k else ""))
