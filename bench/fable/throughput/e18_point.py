#!/usr/bin/env python3
"""E18 (docs/gpu-engine/fable/E18-throughput.md): one point of the throughput curve, from the run script's files.

Usage: e18_point.py POINT_DIR            -> summary on stdout, POINT_DIR/point.json
       e18_point.py --table RES_DIR      -> one markdown row per point (every RES_DIR/*/point.json, in run order)

POINT_DIR holds runs.tsv (item, worker, sample, start, end, rc; one line per sample run), per run c<item>/ with time.txt, run.log.ts,
star_Log.out, star_Log.final.out, and samples.jsonl (bench/fable/throughput/e18_sampler.py).

Per run: wall, host core-s (user + sys of /usr/bin/time: the run's whole process tree), core-s per M pairs, maxRSS, and the device locks:
  prefill   STAR's "ENGINE_GPU: device D (...), waited W s for it, held it H s" (W in whole seconds)
  handoff   the pipeline's hand-off pass on the device (e2ePass): "hand-off pass: ... total X ms" (held; its wait is not logged)
  solo      E11's "waited W s for the device, soloRunKept H s"
  pileup    "[gpu_pileup] device lock: waited W s, held H s"
  nonhost   "[nonhost] engine_gpu: device D (waited W s), index ... loaded in L s, ... classified in C s" (held ~ L + C)
and from the sampler (0.5 s ticks): seconds holding a lock, seconds blocked on one, peak RSS / PSS of the run's processes, peak device MB.
Per point: the span (first start .. last end), pairs, aggregate pairs/s, the cgroup's core-s over the span and per M pairs, GPU
utilisation (nvidia-smi) and the fraction of the span in which some run held a device lock, host and device peaks.
"""
import sys, os, re, json, glob


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def table(res):
    rows = []
    for f in glob.glob(f"{res}/*/point.json"):
        p = json.load(open(f))
        rows.append((p["first_start"], p))
    rows.sort(key=lambda r: r[0])
    print("| point | runs | k | T | slots | span s | M pairs | **pairs/s (M)** | wall s (min / mean / max) | core-s/M (cgroup) | "
          "lock wait s (mean per run) | lock held s (mean per run) | GPU held % of span | GPU util % | host peak GB | device peak GB |")
    print("|" + "---|" * 16)
    for _, p in rows:
        w = [r["wall"] for r in p["runs"]]
        print(f"| {p['tag']} | {len(p['runs'])} | {p['k']} | {p['T']} | {p['slots']} | {p['span']:.1f} | {p['pairs_M']:.1f} | "
              f"**{p['pairs_per_s'] / 1e6:.3f}** | {min(w):.1f} / {sum(w) / len(w):.1f} / {max(w):.1f} | {p['cgroup_core_s_per_M']:.2f} | "
              f"{p['lock_wait_mean']:.1f} | {p['lock_held_mean']:.1f} | {p['gpu_held_pct']:.0f} | {p['gpu_util_mean']:.0f} | "
              f"{p['host_peak_gb']:.0f} | {p['dev_peak_gb']:.1f} |")


if sys.argv[1] == "--table":
    table(sys.argv[2])
    sys.exit(0)

D = sys.argv[1]
tag = os.path.basename(D.rstrip("/"))
meta = json.load(open(f"{D}/meta.json")) if os.path.exists(f"{D}/meta.json") else {}
runs = []
for l in open(f"{D}/runs.tsv"):
    f = l.rstrip("\n").split("\t")
    if f[0] == "item":
        continue
    runs.append({"item": int(f[0]), "worker": int(f[1]), "sample": f[2], "start": float(f[3]), "end": float(f[4]), "rc": int(f[5])})

samples = []
if os.path.exists(f"{D}/samples.jsonl"):
    for l in open(f"{D}/samples.jsonl"):
        try:
            samples.append(json.loads(l))
        except ValueError:
            pass

for r in runs:
    c = f"c{r['item']}"
    rd = f"{D}/{c}"
    r["copy"] = c
    r["wall"] = r["end"] - r["start"]
    tt = open(f"{rd}/time.txt").read() if os.path.exists(f"{rd}/time.txt") else ""
    u = re.search(r"User time \(seconds\): ([\d.]+)", tt)
    s = re.search(r"System time \(seconds\): ([\d.]+)", tt)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", tt)
    r["core_s"] = (float(u.group(1)) + float(s.group(1))) if u and s else None
    r["maxrss_gb"] = int(m.group(1)) / 2**20 if m else None
    lf = open(f"{rd}/star_Log.final.out", errors="replace").read() if os.path.exists(f"{rd}/star_Log.final.out") else ""
    n = re.search(r"Number of input reads \|\s+(\d+)", lf)
    r["pairs"] = int(n.group(1)) if n else None
    sl = open(f"{rd}/star_Log.out", errors="replace").read() if os.path.exists(f"{rd}/star_Log.out") else ""
    lg = open(f"{rd}/run.log.ts", errors="replace").read() if os.path.exists(f"{rd}/run.log.ts") else ""
    L = {}
    x = re.search(r"ENGINE_GPU: device (\d+) \(([^)]*)\), waited (\d+) s for it, held it ([\d.e+-]+) s", sl)
    if x:
        L["prefill"] = {"dev": int(x.group(1)), "wait": float(x.group(3)), "held": float(x.group(4))}
    x = re.search(r"waited ([\d.e+-]+) s for the device, soloRunKept ([\d.e+-]+) s", sl)
    if x:
        L["solo"] = {"wait": float(x.group(1)), "held": float(x.group(2))}
    x = re.search(r"hand-off pass: .*total (\d+) ms\)", lg)
    if x:
        L["handoff"] = {"wait": None, "held": int(x.group(1)) / 1000}
    x = re.search(r"\[gpu_pileup\] device lock: waited ([\d.e+-]+) s, held ([\d.e+-]+) s", lg)
    if x:
        L["pileup"] = {"wait": float(x.group(1)), "held": float(x.group(2))}
    x = re.search(r"\[nonhost\] engine_gpu: device (\d+) \(waited ([\d.e+-]+) s\), index (\d+) MB loaded in ([\d.e+-]+) s, \d+ reads classified in ([\d.e+-]+) s", lg)
    if x:
        L["nonhost"] = {"dev": int(x.group(1)), "wait": float(x.group(2)), "held": float(x.group(4)) + float(x.group(5))}
    r["locks"] = L
    x = re.search(r"engine_gpu: device memory \(GB in use\): (.*)", sl)
    r["engine_mem"] = x.group(1).strip() if x else None
    x = re.search(r"E12 lanes \(new\): batches (\d+) / (\d+) / (\d+)", sl)
    r["lanes"] = x.group(0) if x else None
    x = re.search(r"engine_gpu: steps \(s\): .*; total ([\d.]+)", sl)
    r["prefill_device_s"] = float(x.group(1)) if x else None
    x = re.search(r"E6 dump of \d+ records -> \S+ \(([\d.]+) s\)", sl)
    r["dump_s"] = float(x.group(1)) if x else None
    x = re.search(r"\[gpu_pileup\] on: GPU ([\d.]+) ms, total ([\d.]+) ms, peak ([\d.]+) GB", lg)
    r["pileup_gpu_ms"], r["pileup_peak_gb"] = (float(x.group(1)), float(x.group(3))) if x else (None, None)
    r["level5"] = "E11: Solo on the device" in sl
    r["gpu_pileup_ok"] = "[gpu_pileup] on:" in lg and not re.search(r"\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored", lg)
    r["engine_failed"] = bool(re.search(r"ENGINE_GPU: .*(failed|not used)|engine_gpu: .*(error|out of memory)", sl, re.I))
    r["fallback_lines"] = [l[:200] for l in sl.splitlines() if re.search(r"out of memory|cudaMalloc|not used|failed", l, re.I)][:6]
    # stages from the run's own log (first line with each marker, as bench/fable/ingest/s8_stages_e13.py)
    marks = [("start", None), ("scan", ("Decoding ", "[1fq-ingest] Scanning")), ("refs", (".1fq decoded to temp files", ".1fq scanned:")),
             ("STAR", ("STAR pid=",)), ("stream", ("SNPs to BAM chromosome",)), ("handoff", ("hand-off pass",)), ("pileup", ("[gpu_pileup] on",)),
             ("pileupfin", ("Pileup: ",)), ("nonhost", ("STAR completed",)), ("export", ("Exporting feature matrices",)), ("demux", ("[demux] Starting",))]
    ts = [(int(l[:10]), l[11:]) for l in lg.splitlines() if l[:10].isdigit()]
    pos = [r["start"]]
    for name, pats in marks[1:]:
        t = next((a for a, b in ts if any(p in b for p in pats) and a >= pos[-1] - 1), None)
        pos.append(t if t is not None else pos[-1])
    pos.append(r["end"])
    r["stages"] = {marks[i][0]: round(pos[i + 1] - pos[i], 1) for i in range(len(marks))}
    r["_pos"] = pos

# sampler: per run (copy) seconds holding / blocked on a lock, peaks; per point GPU busy and memory
hold = {}; wait = {}; peak = {}; gpeak = {}
held_any = 0.0; span_s = 0.0; util = []; devtot = 0
mem_peak = 0.0; anon_peak = 0.0; shmem_peak = 0.0
for i in range(1, len(samples)):
    a, b = samples[i - 1], samples[i]
    dt = b["t"] - a["t"]
    if dt <= 0 or dt > 5:
        continue
    span_s += dt
    any_h = False
    for name, e in a.get("locks", {}).items():
        if e.get("h"):
            hold[e["h"]] = hold.get(e["h"], 0) + dt
            any_h = True
        for w in e.get("w", []):
            wait[w] = wait.get(w, 0) + dt
    if any_h:
        held_any += dt
for s in samples:
    mem_peak = max(mem_peak, s.get("mem") or 0)
    anon_peak = max(anon_peak, (s.get("mstat") or {}).get("anon", 0))
    shmem_peak = max(shmem_peak, (s.get("mstat") or {}).get("shmem", 0))
    for c, e in s.get("copies", {}).items():
        p = peak.setdefault(c, {"rss": 0, "pss": 0})
        p["rss"] = max(p["rss"], e.get("rss", 0))
        p["pss"] = max(p["pss"], e.get("pss", 0))
    if s.get("gpu"):
        util.append(sum(g[0] for g in s["gpu"]) / len(s["gpu"]))
        devtot = max(devtot, max(g[1] for g in s["gpu"]))
        for c, d in (s.get("gproc") or {}).items():
            if isinstance(d, dict):
                gpeak[c] = max(gpeak.get(c, 0), max(d.values()) if d else 0)

first = min(r["start"] for r in runs); last = max(r["end"] for r in runs)
span = last - first
pairs = sum(r["pairs"] or 0 for r in runs)
# the cgroup's CPU over the span
cg = [(s["t"], s["cpu"]) for s in samples if s.get("cpu") is not None]
def cpu_at(t):
    if not cg:
        return None
    best = min(cg, key=lambda x: abs(x[0] - t))
    return best[1]
cgs = (cpu_at(last) - cpu_at(first)) if cg else None
def copy_cpu_at(c, t):   # the run's own CPU time (sampler) at time t: last tick at or before t
    v = 0.0
    for s in samples:
        if s["t"] > t:
            break
        e = s.get("copies", {}).get(c)
        if e:
            v = max(v, e["cpu"])
    return v
for r in runs:
    c = r["copy"]
    pos = r.pop("_pos")
    cc = [copy_cpu_at(c, t) for t in pos]
    r["stage_core_s"] = {k: round(cc[i + 1] - cc[i], 1) for i, k in enumerate(r["stages"])}
    r["sampler_hold_s"] = round(hold.get(c, 0), 1)
    r["sampler_wait_s"] = round(wait.get(c, 0), 1)
    r["peak_rss_gb"] = round(peak.get(c, {}).get("rss", 0), 1)
    r["peak_pss_gb"] = round(peak.get(c, {}).get("pss", 0), 1)
    r["peak_dev_mb"] = gpeak.get(c, 0)
    r["log_wait_s"] = round(sum((v.get("wait") or 0) for v in r["locks"].values()), 1)
    r["log_held_s"] = round(sum((v.get("held") or 0) for v in r["locks"].values()), 1)

P = {"tag": tag, "k": meta.get("k"), "T": meta.get("T"), "slots": meta.get("slots"), "env": meta.get("env"), "node": meta.get("node"),
     "first_start": first, "last_end": last, "span": span, "pairs_M": pairs / 1e6, "pairs_per_s": pairs / span if span > 0 else 0,
     "cgroup_core_s": cgs, "cgroup_core_s_per_M": (cgs / (pairs / 1e6)) if cgs and pairs else 0,
     "runs_core_s_per_M": sum(r["core_s"] or 0 for r in runs) / (pairs / 1e6) if pairs else 0,
     "lock_wait_mean": sum(r["sampler_wait_s"] for r in runs) / len(runs), "lock_held_mean": sum(r["sampler_hold_s"] for r in runs) / len(runs),
     "gpu_held_pct": 100 * held_any / span_s if span_s else 0, "gpu_util_mean": sum(util) / len(util) if util else 0,
     "host_peak_gb": mem_peak, "anon_peak_gb": anon_peak, "shmem_peak_gb": shmem_peak, "dev_peak_gb": devtot / 1024,
     "runs": runs}
json.dump(P, open(f"{D}/point.json", "w"), indent=1)

print(f"== point {tag}: k={P['k']} T={P['T']} slots={P['slots']} env={P['env']} node={P['node']}")
print(f"   span {span:.1f} s ({len(runs)} runs, {pairs / 1e6:.2f} M pairs): {P['pairs_per_s'] / 1e6:.3f} M pairs/s; cgroup {cgs or 0:.0f} core-s = "
      f"{P['cgroup_core_s_per_M']:.2f} core-s/M (runs' own: {P['runs_core_s_per_M']:.2f}); GPU util {P['gpu_util_mean']:.0f} %, a lock held "
      f"{P['gpu_held_pct']:.0f} % of the span; host peak {mem_peak:.0f} GB (anon {anon_peak:.0f}, shmem {shmem_peak:.0f}); device peak {devtot / 1024:.1f} GB")
print(f"   {'run':4s} {'w':>2s} {'sample':12s} {'start':>6s} {'wall':>6s} {'core-s':>7s} {'c-s/M':>6s} {'maxRSS':>6s} {'pkRSS':>6s} {'pkPSS':>6s} {'devMB':>6s} "
      f"{'wait':>5s} {'held':>5s} | prefill w/h | handoff h | solo w/h | pileup w/h | nonhost w/h | L5 pileup | stages (s)")
for r in sorted(runs, key=lambda r: r["item"]):
    L = r["locks"]
    def wh(k):
        v = L.get(k)
        if not v:
            return "-"
        return (f"{v['wait']:.1f}/" if v.get("wait") is not None else "") + f"{v['held']:.1f}"
    cs = r["core_s"] or 0
    print(f"   {r['copy']:4s} {r['worker']:2d} {r['sample']:12s} {r['start'] - first:6.1f} {r['wall']:6.1f} {cs:7.0f} "
          f"{cs / (r['pairs'] / 1e6) if r['pairs'] else 0:6.2f} {r['maxrss_gb'] or 0:6.1f} {r['peak_rss_gb']:6.1f} {r['peak_pss_gb']:6.1f} {r['peak_dev_mb']:6d} "
          f"{r['sampler_wait_s']:5.1f} {r['sampler_hold_s']:5.1f} | {wh('prefill')} | {wh('handoff')} | {wh('solo')} | {wh('pileup')} | {wh('nonhost')} | "
          f"{'yes' if r['level5'] else 'NO'} {'ok' if r['gpu_pileup_ok'] else 'NO'} | "
          + " ".join(f"{k} {v:.0f}" for k, v in r["stages"].items()))
    print("        stage core-s (sampler): " + " ".join(f"{k} {v:.0f}" for k, v in r["stage_core_s"].items()))
    if r["rc"] != 0:
        print(f"   !! {r['copy']} exit code {r['rc']}")
    if r["prefill_device_s"] is not None:
        print(f"        prefill device phase {r['prefill_device_s']:.2f} s; {r['lanes'] or ''}; dump {r['dump_s']} s; pileup GPU {r['pileup_gpu_ms']} ms, "
              f"peak {r['pileup_peak_gb']} GB")
        print(f"        engine memory marks (device-wide): {r['engine_mem']}")
    for l in r["fallback_lines"]:
        print(f"        log: {l}")
