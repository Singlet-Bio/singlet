#!/usr/bin/env python3
"""CPU ledger of an rt-node run: core-seconds per sample by pipeline phase and process role.

Reads <run>/procs.snap (every 5 s: pid ppid comm ticks rss for each process of the job), <run>/pool{0,1}/samples.tsv (the items'
pids and times) and <run>/pool{0,1}/<item>.log (epoch-stamped pipeline lines, for the phase boundaries).
Roles: main = the item's pipeline process; star = its child with the most CPU (STAR with the GPU prefill); other children by name.
Phases (by the item's own log): scan (start to reads decoded), map (to the stream drained), pileup (to run_pipeline done),
allsite (to the all-site line), qc (to cells called), tail (to the end). A process's CPU between two snapshots goes to the phase its
item is in at the later snapshot. usage: cpu_ledger.py <run dir> [first item index to include, default K from results.txt]
"""
import collections, os, re, sys

run = sys.argv[1]
HZ = 100.0
marks = [("scan", r"decoded into the packed"), ("map", r"stream drained"), ("pileup", r"step run_pipeline"),
         ("allsite", r"\[allsite-gpu\]"), ("qc", r"\[cell_calling\] n_cells")]
K = 0
m = re.search(r" K=(\d+) ", open(os.path.join(run, "results.txt")).readline())
if m: K = int(m.group(1))
first = int(sys.argv[2]) if len(sys.argv) > 2 else K

items = {}   # pid -> dict
for d in (0, 1):
    f = os.path.join(run, "pool%d" % d, "samples.tsv")
    if not os.path.exists(f): continue
    for ln in open(f).read().splitlines()[1:]:
        c = ln.split("\t")
        it = dict(tag="c%d.%s" % (d, c[0]), idx=int(c[0]), start=float(c[4]), end=float(c[5]), wall=float(c[6]), core=float(c[9]), pid=int(c[14]), bounds=[])
        t = {}
        for l in open(os.path.join(run, "pool%d" % d, c[0] + ".log"), errors="replace"):
            sp = l.split(" ", 1)
            if len(sp) < 2 or not sp[0].isdigit(): continue
            for name, pat in marks:
                if name not in t and re.search(pat, sp[1]): t[name] = int(sp[0])
        prev = it["start"]
        for name, _ in marks:
            b = t.get(name, prev); b = max(b, prev); it["bounds"].append((name, b)); prev = b
        it["bounds"].append(("tail", it["end"] + 6))
        items[it["pid"]] = it

def phase(it, ts):
    for name, b in it["bounds"]:
        if ts <= b: return name
    return "tail"

parent, comm, last = {}, {}, {}
owner_cache = {}
def owner(p):
    seen = []
    q = p
    while q not in items and q in parent and q not in seen:
        if q in owner_cache: q = owner_cache[q]; break
        seen.append(q); q = parent[q]
    r = q if q in items else None
    for s in seen: owner_cache[s] = r if r is not None else -1
    return r

led = collections.defaultdict(float)      # (item pid, proc pid, phase) -> core-s
tot = collections.defaultdict(float)      # proc pid -> core-s
ts = None
for ln in open(os.path.join(run, "procs.snap"), errors="replace"):
    c = ln.split()
    if not c: continue
    if c[0] == "S": ts = int(c[1]); continue
    if len(c) < 5 or ts is None: continue
    try: p, pp, tk = int(c[0]), int(c[1]), int(c[3])
    except ValueError: continue
    if p not in parent: parent[p] = pp
    comm[p] = re.sub(r"[-_.:/]?\d+$", "", c[2])
    d = tk - last.get(p, 0) if p in last else tk   # a process first seen: its CPU so far (it started within the last 5 s, or before the sampler)
    last[p] = tk
    if d <= 0: continue
    o = owner(p) if p not in items else p
    if o is None or o == -1: continue
    it = items[o]
    led[(o, p, phase(it, ts))] += d / HZ; tot[p] += d / HZ

star = {}
for o, it in items.items():
    kids = [p for p in parent if parent[p] == o and p not in items]
    star[o] = max(kids, key=lambda p: tot.get(p, 0)) if kids else None

phases = [n for n, _ in marks] + ["tail"]
agg = collections.defaultdict(float); n = 0; walls = collections.defaultdict(float); core = 0.0; seen = 0.0
for o, it in items.items():
    if it["idx"] < first: continue
    n += 1; core += it["core"]
    prev = it["start"]
    for name, b in it["bounds"]:
        walls[name] += min(b, it["end"]) - min(prev, it["end"]); prev = b
    for (oo, p, ph), v in led.items():
        if oo != o: continue
        role = "main" if p == o else "star" if p == star[o] else "child:" + comm.get(p, "?")
        agg[(role, ph)] += v; seen += v
if n == 0: sys.exit("no items")
roles = sorted({r for r, _ in agg}, key=lambda r: -sum(agg[(r, ph)] for ph in phases))
print("items %d (index >= %d); core-s per item: %.0f by the pool's accounting, %.0f seen by the sampler" % (n, first, core / n, seen / n))
print("%-22s" % "role \\ phase" + "".join("%9s" % ph for ph in phases) + "%9s" % "total")
print("%-22s" % "wall s (mean)" + "".join("%9.0f" % (walls[ph] / n) for ph in phases) + "%9.0f" % (sum(walls.values()) / n))
for r in roles:
    row = [agg[(r, ph)] / n for ph in phases]
    if sum(row) < 2: continue
    print("%-22s" % r[:22] + "".join("%9.0f" % v for v in row) + "%9.0f" % sum(row))
col = [sum(agg[(r, ph)] for r in roles) / n for ph in phases]
print("%-22s" % "all" + "".join("%9.0f" % v for v in col) + "%9.0f" % sum(col))
print("%-22s" % "cores while in phase" + "".join("%9.1f" % (col[i] / max(1e-9, walls[ph] / n)) for i, ph in enumerate(phases)))
