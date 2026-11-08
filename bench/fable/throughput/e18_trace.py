"""E18: one run's 0.2 s trace from the sampler (point tag f): per tick the time after the run's start, H when a device lock is held,
the device utilisation and memory (device 0), and the run's CPU cores. Usage: e18_trace.py <point dir> <from s> <to s> [item]"""
import json, sys
d = sys.argv[1]; lo = float(sys.argv[2]); hi = float(sys.argv[3])
t0 = None
for l in open(d + "/runs.tsv"):
    f = l.split("\t")
    if len(f) > 3 and f[0] == (sys.argv[4] if len(sys.argv) > 4 else "0"):
        t0 = float(f[3])
prev = None
for l in open(d + "/samples.jsonl"):
    s = json.loads(l); t = s["t"] - t0
    c = s.get("copies", {}).get("c" + (sys.argv[4] if len(sys.argv) > 4 else "0"), {}).get("cpu")
    if lo <= t <= hi:
        g = s.get("gpu") or [[None, None]]
        lk = "H" if any(v.get("h") for v in s.get("locks", {}).values()) else "-"
        dc = None if (prev is None or c is None) else (c - prev[1]) / (t - prev[0])
        print(f"{t:6.1f} {lk} util {g[0][0]} mem {g[0][1]} cores {dc if dc is None else round(dc,1)}")
    if c is not None: prev = (t, c)
