#!/usr/bin/env python3
"""E14 (docs/gpu-engine/fable/E14-probe.md): the "STAR load, CR4 probe, start" stage of one gate run in detail.
Usage: s8_probe_e14.py RUN_DIR   (RUN_DIR: run.log.ts, star_Log.out, star_Log.final.out, out/cr4_probe/*_Log.out, as gate.sbatch keeps them)
Times: run.log.ts has whole epoch seconds, STAR's logs whole local seconds (converted with this node's time zone), the probe timing line
milliseconds; so the rows are +-1 s. The stage runs from "[singlet] STAR pid=" to the main STAR's "Started mapping on" (after its
genome load, before the GPU prefill or the CPU mapping)."""
import os, re, sys, time

d = sys.argv[1]
log = [(int(l[:10]), l[11:].rstrip("\n")) for l in open(f"{d}/run.log.ts", errors="replace") if l[:10].isdigit()]
year = time.localtime(log[0][0]).tm_year if log else 2026


def first(pat):
    return next(((t, s) for t, s in log if pat in s), (None, None))


def star_t(s):  # "Nov 01 03:39:14" or "Sun Nov  1 03:39:15 2026"
    s = s.strip()
    for fmt in ("%b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Y"):
        try:
            return time.mktime(time.strptime(s if fmt.startswith("%a") else f"{s} {year}", fmt))
        except ValueError:
            pass
    return None


def star_marks(logout, final):
    m = {}
    if os.path.exists(final):
        for l in open(final, errors="replace"):
            if "|" in l:
                k, v = l.split("|", 1)
                k = k.strip()
                if k in ("Started job on", "Started mapping on", "Finished on"):
                    m[k] = star_t(v)
    if os.path.exists(logout):
        for l in open(logout, errors="replace"):
            for k in ("Started loading the genome:", "Finished loading the genome:"):
                if l.startswith(k):
                    m[k[:-1]] = star_t(l[len(k):])
            if "Finished reading, sorting and deduplicating CB whitelist" in l:
                m["whitelist done"] = star_t(l[:15])
            if "Genome preload:" in l:
                m["preload"] = l.strip()
    return m


def span(m, a, b):
    return f"{m[b] - m[a]:4.0f}" if m.get(a) is not None and m.get(b) is not None else "   -"


t_fork, _ = first("[singlet] STAR pid=")
t_dec, dec = first("[singlet] CR4 probe (")
_, tim = first("[singlet] CR4 probe timing:")
_, pre = first("[singlet] Genome preload:")
_, pf = first("[singlet] Genome prefetch:")
main = star_marks(f"{d}/star_Log.out", f"{d}/star_Log.final.out")
print(f"genome: {pre or pf or '-'}")
print(f"probe: {dec or '(no probe line)'}")
if tim:
    print(f"probe timing: {tim}")
rows = []
for tag in ("plain", "cr4"):
    p = star_marks(f"{d}/out/cr4_probe/{tag}_Log.out", f"{d}/out/cr4_probe/{tag}_Log.final.out")
    if p:
        rows.append((tag, p))
print(f"{'process':8s} {'start':>6s} {'params+WL':>9s} {'genome':>6s} {'g->map':>6s} {'map+Solo':>8s} {'end':>6s}   (s after STAR pid; spans in s)")
for tag, p in rows:
    st = p.get("Started job on")
    print(f"{tag:8s} {st - t_fork if st and t_fork else float('nan'):6.0f} {span(p, 'Started job on', 'Started loading the genome'):>9s} "
          f"{span(p, 'Started loading the genome', 'Finished loading the genome'):>6s} {span(p, 'Finished loading the genome', 'Started mapping on'):>6s} "
          f"{span(p, 'Started mapping on', 'Finished on'):>8s} {p['Finished on'] - t_fork if p.get('Finished on') and t_fork else float('nan'):6.0f}"
          f"   {p.get('preload', '')[:90]}")
if main:
    st = main.get("Started job on")
    print(f"{'main':8s} {st - t_fork if st and t_fork else float('nan'):6.0f} {span(main, 'Started job on', 'Started loading the genome'):>9s} "
          f"{span(main, 'Started loading the genome', 'Finished loading the genome'):>6s} {span(main, 'Finished loading the genome', 'Started mapping on'):>6s}"
          f"{'':>9s} {'':>6s}   {main.get('preload', '')[:90]}")
if t_fork and main.get("Started mapping on"):
    sm = main["Started mapping on"]
    print(f"stage (STAR pid -> main STAR starts mapping): {sm - t_fork:.0f} s;"
          f" probes {t_dec - t_fork if t_dec else float('nan'):.0f} s, main STAR start {sm - (t_dec or t_fork):.0f} s")
