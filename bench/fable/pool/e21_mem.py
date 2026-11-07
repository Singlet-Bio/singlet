#!/usr/bin/env python3
"""E21 (docs/gpu-engine/fable/E21-pool.md): host memory of a point run by `singlet gpu-pool`, from e21_sampler.py's samples.jsonl and the
pools' reports (pool<D>/pool.log, pool<D>/nhsrv.log).

Usage: e21_mem.py POINT_DIR     -> summary on stdout, POINT_DIR/mem.json

  cgroup     the job's memory.current, anon, shmem, file, anon_thp: peak, and the mean over the steady window (the first run's end to the
             last run's start, as e18_steady.py), with the number of samples in flight then
  fixed      what each pool holds once: its genomes (pool.log "genome ...: resident, ... N MB") and its non-host server's page-locked
             index (nhsrv.log "resident: N MB"); the pool's and the server's own peak RSS / PSS / private memory (sampler)
  per run    each sample's processes (its process, the STAR child, the probes, ...): peak private memory (Private_Clean + Private_Dirty:
             what the run alone holds; the genome pages it shares with the pool are not in it), peak PSS, peak RSS
  in flight  per PSS sweep, the sum of the runs' private memory and the number of runs; and (cgroup anon + shmem - fixed) / runs in
             flight at the cgroup's peak and over the steady window: host memory per sample in flight
"""
import sys, os, re, json, glob, statistics

D = sys.argv[1]
S = []
for l in open(f"{D}/samples.jsonl"):
    try:
        S.append(json.loads(l))
    except ValueError:
        pass
runs = []
for l in open(f"{D}/runs.tsv"):
    f = l.rstrip("\n").split("\t")
    if len(f) >= 6 and f[0] != "item":
        runs.append((int(f[0]), float(f[3]), float(f[4])))
if not S or not runs:
    print("e21_mem: no samples or runs"); sys.exit(0)

# fixed per pool
fixed = {}
for pd in sorted(glob.glob(f"{D}/pool*/")):
    name = os.path.basename(pd.rstrip("/"))
    g = 0.0
    pl = open(f"{pd}/pool.log", errors="replace").read() if os.path.exists(f"{pd}/pool.log") else ""
    for m in re.finditer(r"genome \S+: resident, (\d+) MB in", pl):
        g += int(m.group(1)) / 1024
    nl = open(f"{pd}/nhsrv.log", errors="replace").read() if os.path.exists(f"{pd}/nhsrv.log") else ""
    m = re.search(r"resident(?: \(first request\))?: (\d+) MB", nl)
    fixed[name] = {"genome_gb": round(g, 2), "nonhost_pinned_gb": round(int(m.group(1)) / 1024, 2) if m else 0.0}

# the pools' and servers' own processes, and per run
peak = {}
for s in S:
    for c, e in s.get("copies", {}).items():
        p = peak.setdefault(c, {"rss": 0.0, "pss": 0.0, "priv": 0.0, "panon": 0.0})
        for k in p:
            p[k] = max(p[k], e.get(k, 0.0) or 0.0)
own = {c: v for c, v in peak.items() if not re.match(r"^c\d+$", c)}
per = {c: v for c, v in peak.items() if re.match(r"^c\d+$", c)}

def inflight(t):
    return sum(1 for _, a, b in runs if a <= t <= b)

ends = sorted(b for _, a, b in runs); starts = sorted(a for _, a, b in runs)
workers = None
try:
    workers = len(set(l.split("\t")[1] for l in open(f"{D}/runs.tsv") if not l.startswith("item")))
except Exception:
    pass
wa = ends[0]
wb = ends[len(ends) - workers] if workers and len(ends) > workers else ends[-1]
fixed_tot = sum(v["genome_gb"] + v["nonhost_pinned_gb"] for v in fixed.values())

def nonrecl(s):
    m = s.get("mstat") or {}
    return (m.get("anon", 0) or 0) + (m.get("shmem", 0) or 0)

pk = max(S, key=lambda s: s.get("mem") or 0)
pk2 = max(S, key=nonrecl)
win = [s for s in S if wa <= s["t"] <= wb]
sweeps = []
for s in S:
    if "pss_s" not in s:
        continue
    cs = [e for c, e in s.get("copies", {}).items() if re.match(r"^c\d+$", c) and "priv" in e]
    sweeps.append({"t": s["t"], "n": len(cs), "priv": sum(e["priv"] for e in cs), "pss": sum(e.get("pss", 0) for e in cs),
                   "own_pss": sum(e.get("pss", 0) for c, e in s.get("copies", {}).items() if not re.match(r"^c\d+$", c)),
                   "cg": nonrecl(s)})
R = {"fixed": fixed, "fixed_gb": round(fixed_tot, 1), "own": own,
     "cgroup_peak": {"mem": pk.get("mem"), **(pk.get("mstat") or {}), "in_flight": inflight(pk["t"])},
     "cgroup_nonreclaimable_peak": {"anon_shmem": round(nonrecl(pk2), 1), "in_flight": inflight(pk2["t"]), **(pk2.get("mstat") or {})},
     "window": [wa, wb]}
if win:
    R["window_mean"] = {"mem": round(statistics.mean(s.get("mem") or 0 for s in win), 1),
                        "anon_shmem": round(statistics.mean(nonrecl(s) for s in win), 1),
                        "in_flight": round(statistics.mean(inflight(s["t"]) for s in win), 2)}
if per:
    pv = sorted(v["priv"] for v in per.values()); ps = sorted(v["pss"] for v in per.values()); rs = sorted(v["rss"] for v in per.values())
    R["per_run"] = {"n": len(per), "priv_median": round(statistics.median(pv), 1), "priv_max": round(pv[-1], 1),
                    "pss_median": round(statistics.median(ps), 1), "pss_max": round(ps[-1], 1),
                    "rss_median": round(statistics.median(rs), 1), "rss_max": round(rs[-1], 1)}
if sweeps:
    mx = max(sweeps, key=lambda w: w["priv"])
    R["sweep_max"] = {k: round(v, 1) if isinstance(v, float) else v for k, v in mx.items()}
k_pk = R["cgroup_nonreclaimable_peak"]["in_flight"]
if k_pk:
    R["per_sample_at_peak_gb"] = round((nonrecl(pk2) - fixed_tot) / k_pk, 1)
if win and R["window_mean"]["in_flight"]:
    R["per_sample_window_gb"] = round((R["window_mean"]["anon_shmem"] - fixed_tot) / R["window_mean"]["in_flight"], 1)
json.dump(R, open(f"{D}/mem.json", "w"), indent=1)

print(f"== host memory ({os.path.basename(D.rstrip('/'))}): fixed per pool " + "; ".join(
    f"{k} genome {v['genome_gb']:.1f} GB + non-host index {v['nonhost_pinned_gb']:.1f} GB page-locked" for k, v in fixed.items())
      + f" = {fixed_tot:.1f} GB")
cp = R["cgroup_peak"]
print(f"   cgroup peak {cp['mem']} GB (anon {cp.get('anon')}, shmem {cp.get('shmem')}, file {cp.get('file')}, anon_thp {cp.get('anon_thp')}) "
      f"with {cp['in_flight']} runs in flight; anon + shmem peak {R['cgroup_nonreclaimable_peak']['anon_shmem']} GB with "
      f"{k_pk} in flight -> {R.get('per_sample_at_peak_gb')} GB per sample in flight beyond the fixed {fixed_tot:.1f} GB")
if "window_mean" in R:
    w = R["window_mean"]
    print(f"   steady window ({wa - starts[0]:.0f}, {wb - starts[0]:.0f}] s: mean {w['mem']} GB, anon + shmem {w['anon_shmem']} GB, "
          f"{w['in_flight']} in flight -> {R.get('per_sample_window_gb')} GB per sample in flight beyond the fixed")
if "per_run" in R:
    p = R["per_run"]
    print(f"   per run ({p['n']}): peak private {p['priv_median']} GB median / {p['priv_max']} max; peak PSS {p['pss_median']} / {p['pss_max']}; "
          f"peak RSS {p['rss_median']} / {p['rss_max']} (RSS counts the shared genome)")
if "sweep_max" in R:
    m = R["sweep_max"]
    print(f"   largest sweep: {m['n']} runs, private {m['priv']} GB in all ({m['priv'] / max(1, m['n']):.1f} per run), PSS {m['pss']}, "
          f"pools + servers PSS {m['own_pss']}, cgroup anon + shmem {m['cg']}")
for c, v in sorted(own.items()):
    print(f"   {c}: peak RSS {v['rss']:.1f} GB, PSS {v['pss']:.1f}, private {v['priv']:.1f}")
