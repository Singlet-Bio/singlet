#!/usr/bin/env python3
"""Summary of the gpu/bold-split overnight chain (build, identity, paths corpus, timing arms).
Usage: summarize.py OUT.md  (job ids and paths from the environment: BUILD_JOB IDENT_JOBS PATHS_JOB TP_JOB W ANALYZE)
Reads logs and small result files only."""
import glob
import os
import re
import statistics as st
import subprocess
import sys

W = os.environ.get("W", "/mnt/projects/debruinz_project/debruinz/singlet")
LOGS = W + "/logs"
out = []


def p(s=""):
    out.append(s)


def rd(path):
    try:
        return open(path, errors="replace").read()
    except OSError:
        return None


def state(job):
    r = subprocess.run(["sacct", "-j", str(job), "-X", "-n", "-o", "JobID,JobName%20,State,ExitCode,Elapsed,NodeList"], capture_output=True, text=True)
    return r.stdout.strip() or "no sacct record"


p("# gpu/bold-split overnight summary")
p()
b = os.environ.get("BUILD_JOB", "")
p("## Build")
t = rd(f"{LOGS}/gbold-build-{b}.out") if b else None
if t:
    for l in t.splitlines():
        if l.startswith("COMMIT=") or re.match(r"^[0-9a-f]{64}  ", l) or l.startswith("[build]"):
            p("    " + l)
else:
    p(f"    no build log for job {b}")
if b:
    p("    " + state(b))
p()

p("## Identity against b0 (pbmc1k_v3, T=14, one slot)")
for j in os.environ.get("IDENT_JOBS", "").split():
    t = rd(f"{LOGS}/gbold-ident-{j}.out")
    p(f"Job {j}: {state(j)}")
    if not t:
        p("    no log")
        continue
    for l in t.splitlines():
        if any(k in l for k in ("PROVENANCE", "VERDICT", "DIFF", "DIFFERS", "wall A", "STEPS_B", "REFUSED", "exit")):
            p("    " + l[:300])
p()

pj = os.environ.get("PATHS_JOB", "")
p("## Code-path corpus with BHOST=1")
p(f"Job {pj}: {state(pj)}" if pj else "no job")
for f in ("summary.txt", "verdicts.tsv"):
    t = rd(f"{W}/jobs/gpu-paths/results/paths-{pj}/{f}")
    if t:
        p(f"{f}:")
        p("```")
        p("\n".join(t.splitlines()[:60]))
        p("```")
p()

tj = os.environ.get("TP_JOB", "")
p("## Timing on g050 (k=8, T=7, 2 slots)")
p(f"Job {tj}: {state(tj)}" if tj else "no job")
R = f"{W}/jobs/gpu-bold/results/tp-{tj}"
ap = os.environ.get("ANALYZE", "")
if ap and os.path.isdir(R):
    r = subprocess.run([sys.executable, ap, R], capture_output=True, text=True)
    p(r.stdout[-6000:] if r.stdout else r.stderr[-1500:])
p()
p("Per arm, means over the runs after the first 8 (from the engine log). Seconds are per run.")
p()
p("| arm | runs | lock held s | lock waited s | lane A s | lane M s | lane B s | lanes.host s | hostlane wait s | pool core-s | pool core-s per M reads |")
p("|---|---|---|---|---|---|---|---|---|---|---|")
num = r"([0-9.]+)"
for d in sorted(glob.glob(R + "/k*")):
    if not os.path.isdir(d):
        continue
    rows = []
    runs = sorted(glob.glob(d + "/c*/run.log.ts"), key=lambda x: int(re.search(r"/c(\d+)/", x.replace("\\", "/")).group(1)))
    for f in runs:
        i = int(re.search(r"/c(\d+)/", f.replace("\\", "/")).group(1))
        if i < 8:
            continue
        t = rd(f) or ""
        v = {}
        m = re.search(r"waited " + num + r" s for it, held it " + num + " s", t)
        if m:
            v["wait"], v["held"] = float(m[1]), float(m[2])
        m = re.search(r"engine_gpu: (\d+) reads, ", t)
        nreads = int(m[1]) if m else None
        m = re.search(r"E12 host lane during the device lanes: (\d+) reads on (\d+) threads, " + num + r" core-s; waited " + num + " s", t)
        if m:
            v["core"], v["hwait"] = float(m[3]), float(m[4])
            if nreads:
                v["coreM"] = float(m[3]) / (nreads / 1e6)
        m = re.search(r"engine_gpu: steps \(s\): (.*?); total", t)
        if m:
            kv = dict((k, float(x)) for k, x in re.findall(r"([\w.]+) ([0-9.]+)", m[1]))
            v["A"] = sum(x for k, x in kv.items() if k.startswith("A."))
            v["M"] = sum(x for k, x in kv.items() if k.startswith("M."))
            v["B"] = sum(x for k, x in kv.items() if k.startswith("B."))
            v["lh"] = kv.get("lanes.host")
        rows.append(v)

    def mean(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return f"{st.mean(xs):.2f}" if xs else "n/a"

    p(f"| {os.path.basename(d)} | {len(rows)} | {mean('held')} | {mean('wait')} | {mean('A')} | {mean('M')} | {mean('B')} | {mean('lh')} | {mean('hwait')} | {mean('core')} | {mean('coreM')} |")
p()
p("Lane seconds sum the engine's step keys that start with A., M. or B. Pool core-s is the host pool's busy time during the device lanes.")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
