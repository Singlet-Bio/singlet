"""E18: steady-state view of a point: completions between the first and the last run's end, the lock held (any holder, per lock
file) and the cgroup's CPU in that window. Usage: e18_steady.py <point dir> [pairs per run (M), default from point.json]"""
import json, sys, os
d = sys.argv[1]
runs = []
for l in open(os.path.join(d, "runs.tsv")):
    f = l.rstrip("\n").split("\t")
    if len(f) < 6 or f[0] == "item": continue
    runs.append((int(f[0]), int(f[1]), f[2], float(f[3]), float(f[4]), int(f[5])))
try: pj = json.load(open(os.path.join(d, "point.json")))
except Exception: pj = {}
pairs = {}
for r in pj.get("runs", []):
    pairs[str(r.get("item", r.get("c", "")))] = r.get("pairs")
ends = sorted(r[4] for r in runs); starts = sorted(r[3] for r in runs)
k = len(set(r[1] for r in runs))
t0, t1 = starts[0], ends[-1]
def pr(i):
    v = pairs.get(str(i)) or pairs.get("c%d" % i)
    return (v / 1e6 if v and v > 1e4 else v) if v else 66.6017
tot = sum(pr(r[0]) for r in runs)
print(f"span {t1 - t0:.1f} s, {len(runs)} runs, {tot:.1f} M pairs: {tot / (t1 - t0):.3f} M pairs/s")
a, b = ends[0], (ends[len(ends) - k] if len(ends) > k else ends[-1])   # every worker busy: the first completion to the first completion after the last start
later = [r for r in runs if r[3] >= ends[0] - 1.0]
if later and b > a:
    mw = sum(r[4] - r[3] for r in later) / len(later)
    print(f"steady (runs started after the first completion: {len(later)}): mean wall {mw:.1f} s, k = {k}: {k * sum(pr(r[0]) for r in later) / len(later) / mw:.3f} M pairs/s; window ({a - t0:.1f}, {b - t0:.1f}] s")
else:
    print("no steady window (one round)"); a, b = t0, t1
held = {}; n = 0; cpu = []; util = []; prev = None
for l in open(os.path.join(d, "samples.jsonl")):
    s = json.loads(l); t = s["t"]
    if not (a <= t <= b): continue
    n += 1
    for k, v in s.get("locks", {}).items():
        held[k] = held.get(k, 0) + (1 if v.get("h") else 0)
    if "cpu" in s:
        if prev: cpu.append((s["cpu"] - prev[1]) / (t - prev[0]))
        prev = (t, s["cpu"])
    g = s.get("gpu")
    if g: util.append(sum(x[0] for x in g) / len(g))
if n:
    print("lock held in the window: " + ", ".join(f"{os.path.basename(k)} {100.0 * v / n:.0f} %" for k, v in sorted(held.items())))
    if cpu: print(f"cgroup CPU in the window: {sum(cpu) / len(cpu):.1f} cores mean")
    if util: print(f"GPU util in the window: {sum(util) / len(util):.0f} % mean")
