#!/usr/bin/env python3
"""E18 (docs/gpu-engine/fable/E18-throughput.md): the device-lock intervals of each run of a point, from the sampler (samples.jsonl).

Usage: e18_locks.py POINT_DIR

For every run (copy) the intervals in which it held an engine lock (consecutive sampler ticks with the copy as the holder of a
/tmp/singlet_engine_gpu_*.lock), in time order, labelled by their order in a level-5 run (prefill, Solo, hand-off pass, pileup,
non-host) when there are five, and per interval: start (s after the run's start), length, the device's utilisation while held
(nvidia-smi: mean, and the time with utilisation > 0, i.e. some kernel ran during the sample period), the device memory in use (MB:
at the first tick, peak), and the run's own CPU rate while held (cores, from the sampler's per-run CPU). Use a fine sampler
(the run script's tag suffix f: 0.2 s ticks, nvidia-smi every tick) for k = 1.
"""
import sys, json, os

D = sys.argv[1]
S = []
for l in open(f"{D}/samples.jsonl"):
    try:
        S.append(json.loads(l))
    except ValueError:
        pass
runs = {}
for l in open(f"{D}/runs.tsv"):
    f = l.rstrip("\n").split("\t")
    runs[f"c{f[0]}"] = float(f[3])
LBL = ["prefill", "solo", "handoff", "pileup", "nonhost"]
gl = None   # last GPU reading carried forward
for s in S:
    if s.get("gpu"):
        gl = s["gpu"]
    s["_g"] = gl
out = {}
for c, t0 in sorted(runs.items(), key=lambda x: int(x[0][1:])):
    iv = []
    cur = None
    for i, s in enumerate(S):
        h = any(e.get("h") == c for e in s.get("locks", {}).values())
        if h and cur is None:
            cur = [i, i]
        elif h:
            cur[1] = i
        elif cur is not None:
            iv.append(cur); cur = None
    if cur is not None:
        iv.append(cur)
    rows = []
    for a, b in iv:
        ta = S[a]["t"]; tb = S[b + 1]["t"] if b + 1 < len(S) else S[b]["t"]
        ticks = S[a:b + 1]
        g = [x["_g"] for x in ticks if x["_g"]]
        util = [max(u for u, m in gg) for gg in g]
        mem = [max(m for u, m in gg) for gg in g]
        dts = [(S[j + 1]["t"] - S[j]["t"]) if j + 1 < len(S) else 0 for j in range(a, b + 1)]
        busy = sum(dt for dt, x in zip(dts, ticks) if x["_g"] and max(u for u, m in x["_g"]) > 0)
        ca = (S[a].get("copies", {}).get(c) or {}).get("cpu"); cb = (S[min(b + 1, len(S) - 1)].get("copies", {}).get(c) or {}).get("cpu")
        rate = (cb - ca) / (tb - ta) if ca is not None and cb is not None and tb > ta else None
        rows.append({"start": round(ta - t0, 1), "len": round(tb - ta, 2), "util_mean": round(sum(util) / len(util), 1) if util else None,
                     "util_gt0_s": round(busy, 2), "mem_first": mem[0] if mem else None, "mem_peak": max(mem) if mem else None,
                     "cpu_cores": round(rate, 2) if rate is not None else None})
    if len(rows) == len(LBL):
        for r, n in zip(rows, LBL):
            r["phase"] = n
    out[c] = rows
json.dump(out, open(f"{D}/locks.json", "w"), indent=1)
print(f"== lock intervals per run (sampler; tick {S[1]['t'] - S[0]['t']:.2f} s)" if len(S) > 1 else "== no samples")
for c, rows in out.items():
    print(f"   {c}: {len(rows)} intervals, held {sum(r['len'] for r in rows):.1f} s, device util > 0 for {sum(r['util_gt0_s'] for r in rows):.1f} s of it")
    for r in rows:
        print(f"      {r.get('phase', '?'):8s} at {r['start']:6.1f} s: held {r['len']:6.2f} s, util mean {r['util_mean']} %, util > 0 {r['util_gt0_s']:5.2f} s, "
              f"device MB first {r['mem_first']} peak {r['mem_peak']}, run CPU {r['cpu_cores']} cores")
