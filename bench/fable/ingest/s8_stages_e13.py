#!/usr/bin/env python3
"""S8: per-stage wall, host core-s and GPU busy % from one s8_profile.sbatch run dir (run.log.ts + monitor.tsv).
Usage: s8_stages.py RUN_DIR READS_M   (stage boundaries are the first log line containing each marker)
E11 copy (bench/fable/solowire): the prefill's end is where the device memory falls below 4000 MB (bench/engine_gpu_gate's 1000 MB, but
level 5 keeps ~1.6 GB of Solo inputs on the device through STAR's loop).
E13 copy (bench/fable/ingest): a marker may be a tuple of alternatives (the first log line containing any of them). With
SINGLET_ENGINE_GPU_INGEST=1 the pipeline's ".1fq -> FASTQ" stage is a head scan ("[1fq-ingest] Scanning" .. ".1fq scanned:"), and the
prefill's read-in is the .1fq decode (its seconds from star_Log.out's "ENGINE_GPU: E13: ... decoded ... in X s" are printed under the
prefill); the file decode's markers ("Decoding " .. ".1fq decoded to temp files") are kept for the other modes."""
import sys, re, bisect
d, readsM = sys.argv[1], float(sys.argv[2])
log = [(int(l[:10]), l[11:]) for l in open(f"{d}/run.log.ts", errors="replace") if l[:10].isdigit()]
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
MARK = [("startup", None), ("decode .1fq -> FASTQ + R11 | E13 head scan", ("Decoding ", "[1fq-ingest] Scanning")),
        ("probe/ref load", (".1fq decoded to temp files", ".1fq scanned:")),
        ("STAR prefill+map (GPU seed/stitch, host map)", "STAR pid="), ("STAR stream into pileup", "SNPs to BAM chromosome"),
        ("GPU-pileup hand-off", "hand-off pass"), ("GPU pileup + collapse", "[gpu_pileup] on"), ("pileup finish", "Pileup: "),
        ("non-host", "STAR completed"), ("export, QC, callers", "Exporting feature matrices"), ("demux/ancestry/writers", "[demux] Starting")]
t0, t1 = mon[0][0], mon[-1][0]
pos = [t0]
for name, pat in MARK[1:]:
    pats = pat if isinstance(pat, tuple) else (pat,)
    t = next((ts for ts, s in log if any(p in s for p in pats) and ts >= pos[-1]), None)
    pos.append(t if t is not None else pos[-1])
pos.append(t1)
if pre_s:  # split "STAR prefill+map" at the prefill end: the device frees its memory when the prefill is done
    g0 = next((m[0] for m in mon if m[3] > MEMT), None); i = [n for n, _ in MARK].index("STAR prefill+map (GPU seed/stitch, host map)")
    g1 = next((m[0] for m in mon if g0 and m[0] > g0 and m[3] < MEMT), None)
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
    if name.startswith("GPU prefill"):
        m13 = re.search(r"E13: (\d+) reads decoded from the \.1fq into the packed buffers in ([0-9.]+) s", sl)
        if m13: print(f"{'  of which the .1fq decode (E13, star_Log.out)':46s} {float(m13.group(2)):7.1f}")
        mr = re.search(r"cache filled with \d+ sequence results in \d+ s \(reading reads (\d+) s, GPU (\d+) s", sl)
        if mr: print(f"{'  reading reads / GPU (star_Log.out, whole s)':46s} {float(mr.group(1)):7.0f} {float(mr.group(2)):8.0f}")
u = [m[2] for m in mon]
print(f"{'TOTAL':46s} {t1-t0:7.0f} {at(t1)-c0:8.1f} {(at(t1)-c0)/readsM:9.2f} {sum(u)/len(u):10.1f}  gpu busy-s {sum(u)/100*1.0:.0f}")
