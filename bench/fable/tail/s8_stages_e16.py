#!/usr/bin/env python3
"""S8: per-stage wall, host core-s and GPU busy % from one s8_profile.sbatch run dir (run.log.ts + monitor.tsv).
Usage: s8_stages.py RUN_DIR READS_M   (stage boundaries are the first log line containing each marker)
E11 copy (bench/fable/solowire): the prefill's end is where the device memory falls below 4000 MB (bench/engine_gpu_gate's 1000 MB, but
level 5 keeps ~1.6 GB of Solo inputs on the device through STAR's loop).
E16 copy (bench/fable/tail): the "[e16]" marks are not stage markers; the stream stage starts at the first "SNPs to BAM chromosome" line after the
prefill (E16's hdr cut resolves the references during the prefill), else at the "[e16] ... STAR: threads joined" mark (the end of STAR's loop). The prefill's end is the "[e16] ... prefill: mapDistinct returned" mark
when there is one (at level 5 with E16 the parent's early hand-off pass takes the device within a second, so the memory never drops)."""
import sys, re, bisect
d, readsM = sys.argv[1], float(sys.argv[2])
logAll = [(int(l[:10]), l[11:]) for l in open(f"{d}/run.log.ts", errors="replace") if l[:10].isdigit()]
log = [x for x in logAll if "[e16] " not in x[1]]
def e16mark(txt, after=0.0):   # the epoch of the first "[e16] <epoch> <pid> <txt...>" mark at or after `after`
    for _, s in logAll:
        i = s.find("[e16] ")
        if i >= 0 and txt in s:
            try: t = float(s[i + 6:].split(" ", 1)[0])
            except ValueError: continue
            if t >= after: return t
    return None
e16pre = e16mark("prefill: mapDistinct returned")   # the prefill's device release (the parent's early pass keeps the memory up after it)
# the main run's loop end (the CR4 probe's STAR runs, before the prefill, print the same mark)
e16join = e16mark("STAR: threads joined", e16pre) if e16pre is not None else None
mon = [tuple(float(x) for x in l.split("\t")) for l in list(open(f"{d}/monitor.tsv"))[1:] if l.count("\t") == 3 and l.split("\t")[1] != ""]
T = [m[0] for m in mon]
def at(t):  # cumulative cgroup cpu at time t (linear interpolation)
    i = bisect.bisect_left(T, t); i = min(max(i, 1), len(mon) - 1); a, b = mon[i - 1], mon[i]
    return a[1] + (b[1] - a[1]) * (t - a[0]) / max(b[0] - a[0], 1e-9)
import os, re
sl = open(f"{d}/starlog_engine.txt", errors="replace").read() if os.path.exists(f"{d}/starlog_engine.txt") else ""
m_ = re.search(r"cache filled with \d+ sequence results in (\d+) s", sl)
pre_s = float(m_.group(1)) if m_ else None   # GPU prefill duration from the engine log
MEMT = 4000
MARK = [("startup", None), ("decode .1fq -> FASTQ + R11 histogram", "Decoding "), ("probe/ref load", ".1fq decoded to temp files"),
        ("STAR prefill+map (GPU seed/stitch, host map)", "STAR pid="), ("STAR stream into pileup", "SNPs to BAM chromosome"),
        ("GPU-pileup hand-off", "hand-off pass"), ("GPU pileup + collapse", "[gpu_pileup] on"), ("pileup finish", "Pileup: "),
        ("non-host", "STAR completed"), ("export, QC, callers", "Exporting feature matrices"), ("demux/ancestry/writers", "[demux] Starting")]
t0, t1 = mon[0][0], mon[-1][0]
# the prefill's end (device memory below MEMT after it was above), for the stream stage's start
gA = next((m[0] for m in mon if m[3] > MEMT), None)
gEnd = e16pre if e16pre is not None else next((m[0] for m in mon if gA and m[0] > gA and m[3] < MEMT), None)
pos = [t0]
for name, pat in MARK[1:]:
    lo = pos[-1]
    if pat == "SNPs to BAM chromosome" and gEnd is not None: lo = max(lo, gEnd - 1)
    t = next((ts for ts, s in log if pat in s and ts >= lo), None)
    if t is None and pat == "SNPs to BAM chromosome" and e16join is not None: t = e16join
    pos.append(t if t is not None else pos[-1])
pos.append(t1)
if pre_s:  # split "STAR prefill+map" at the prefill end: the device frees its memory when the prefill is done
    g0 = next((m[0] for m in mon if m[3] > MEMT), None); i = [n for n, _ in MARK].index("STAR prefill+map (GPU seed/stitch, host map)")
    g1 = e16pre if e16pre is not None else next((m[0] for m in mon if g0 and m[0] > g0 and m[3] < MEMT), None)
    if g1 and pos[i] <= g1 - pre_s and g1 <= pos[i + 1]:
        MARK[i:i + 1] = [("STAR load, CR4 probe, start (before prefill)", None), ("GPU prefill (read, clip, seed, stitch, cache)", None),
                         ("STAR per-read host pass (readLoad, map, output)", None)]
        pos[i + 1:i + 1] = [g1 - pre_s, g1]
c0 = at(t0)
print(f"{'stage':46s} {'wall s':>7s} {'core-s':>8s} {'core-s/M':>9s} {'gpu util %':>10s}")
tot = 0
for i, (name, _) in enumerate(MARK):
    a, b = pos[i], pos[i + 1]
    if b <= a: continue
    cs = at(b) - at(a); tot += cs
    u = [m[2] for m in mon if a <= m[0] < b]
    print(f"{name:46s} {b-a:7.0f} {cs:8.1f} {cs/readsM:9.2f} {sum(u)/max(len(u),1):10.1f}")
u = [m[2] for m in mon]
print(f"{'TOTAL':46s} {t1-t0:7.0f} {at(t1)-c0:8.1f} {(at(t1)-c0)/readsM:9.2f} {sum(u)/len(u):10.1f}  gpu busy-s {sum(u)/100*1.0:.0f}")
