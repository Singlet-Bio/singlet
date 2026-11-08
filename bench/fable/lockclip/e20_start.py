#!/usr/bin/env python3
"""E20 (docs/gpu-engine/fable/E20-lockclip.md): the start stage of one gate run (mode gpuT, SINGLET_E16_TRACE=1), from the STAR fork to the
prefill taking the device, with its host core-s, the CR4 probe's wall and CPU, and E13's clip.

Usage: e20_start.py RUN_DIR   (run.log.ts, monitor.tsv, star_Log.out of bench/fable/lockclip/jobs/gate.sbatch)

  fork      "[singlet] STAR pid=" (whole seconds)
  probe     "[singlet] CR4 probe (" (the decision, whole seconds); the probes' wall and CPU from the "CR4 probe timing" line
  lock      the prefill takes the device: the "[e16] ... prefill: device N locked" mark (E20), else "mapDistinct returned" less the held time
  returned  "[e16] ... prefill: mapDistinct returned (device held X s)"
core-s: the job's cgroup CPU (monitor.tsv, 1 s samples, interpolated) between the marks."""
import sys, re, bisect
d = sys.argv[1]
log = [(int(l[:10]), l[11:].rstrip("\n")) for l in open(f"{d}/run.log.ts", errors="replace") if l[:10].isdigit()]
mon = [tuple(float(x) for x in l.split("\t")[:2]) for l in list(open(f"{d}/monitor.tsv"))[1:] if l.count("\t") == 3 and l.split("\t")[1] != ""]
T = [m[0] for m in mon]
def at(t):
    i = bisect.bisect_left(T, t); i = min(max(i, 1), len(mon) - 1); a, b = mon[i - 1], mon[i]
    return a[1] + (b[1] - a[1]) * (t - a[0]) / max(b[0] - a[0], 1e-9)
def first(p):
    return next(((t, s) for t, s in log if p in s), (None, ""))
def e16(p):
    t, s = first(p)
    m = re.search(r"\[e16\] ([0-9.]+) ", s)
    return (float(m.group(1)), s) if m else (None, s)
tf, _ = first("[singlet] STAR pid=")
tp, _ = first("[singlet] CR4 probe (")
_, tim = first("[singlet] CR4 probe timing:")
tr, rs = e16("prefill: mapDistinct returned")
tl = next((float(re.search(r"\[e16\] ([0-9.]+) ", s).group(1)) for t, s in log if re.search(r"\[e16\] [0-9.]+ \d+ prefill: device \d+ locked", s)), None)
if tl is None:
    m = re.search(r"device held ([0-9.]+) s", rs)
    sl = open(f"{d}/star_Log.out", errors="replace").read()
    ms = re.search(r"E20 split: ([0-9.]+) s to the lanes' end, released ([0-9.]+) s", sl)
    if tr and m: tl = tr - float(m.group(1)) - (float(ms.group(2)) if ms else 0)
sl = open(f"{d}/star_Log.out", errors="replace").read()
print(f"run {d}")
rows = [("fork -> probe decision", tf, tp), ("probe decision -> prefill locks the device", tp, tl), ("fork -> prefill locks the device (start stage)", tf, tl)]
print(f"{'interval':48s} {'wall s':>7s} {'core-s':>8s}")
for n, a, b in rows:
    if a is None or b is None: print(f"{n:48s} {'-':>7s}"); continue
    print(f"{n:48s} {b - a:7.1f} {at(b) - at(a):8.1f}")
m = re.search(r"plain ([0-9.]+) s, cr4 ([0-9.]+) s, \w+ on (\d+)\+(\d+) threads, total ([0-9.]+) s", tim)
mc = re.search(r"CPU \(user \+ system\) plain ([0-9.e+-]+) s, cr4 ([0-9.e+-]+) s", tim)
if m: print(f"CR4 probes: plain {m.group(1)} s, cr4 {m.group(2)} s on {m.group(3)}+{m.group(4)} threads, total {m.group(5)} s"
            + (f"; CPU plain {float(mc.group(1)):.1f} s, cr4 {float(mc.group(2)):.1f} s, sum {float(mc.group(1)) + float(mc.group(2)):.1f} s" if mc else ""))
for pat in (r"ENGINE_GPU: E13[^\n]*clip[^\n]*", r"engine_gpu: E20 CellRanger4 clip[^\n]*"):
    for x in re.findall(pat, sl)[:3]: print("  " + x[:400])
