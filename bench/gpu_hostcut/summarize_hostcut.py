#!/usr/bin/env python3
"""gpu/bold-hostcut summarizer (CPU only). Usage: summarize_hostcut.py "kind=jobid;kind=jobid;..." OUT_DIR
kinds: sync, build, buildprof, ident, kang, prof, tp. Writes OUT_DIR/summary.md."""
import csv, glob, os, re, statistics as st, subprocess, sys

W = "/mnt/projects/debruinz_project/debruinz/singlet"
RES = W + "/jobs/gpu-hostcut/results"
LOGS = W + "/logs"
jobs = {}
for part in sys.argv[1].split(";"):
    if "=" in part:
        k, v = part.split("=", 1)
        jobs[k.strip()] = v.strip()
out = sys.argv[2]
os.makedirs(out, exist_ok=True)


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def sacct(j):
    try:
        o = subprocess.run(["sacct", "-j", str(j), "-n", "-X", "-P", "-o", "JobName,State,ExitCode,Elapsed,NodeList"],
                           capture_output=True, text=True, timeout=60).stdout.strip()
        return o.splitlines()[0].replace("|", " ") if o else "unknown"
    except Exception as e:
        return f"sacct failed: {e}"


def logtext(name, j):
    for p in glob.glob(f"{LOGS}/{name}-{j}.out"):
        return read(p)
    return ""


L = ["# gpu/bold-hostcut summary", "", "## Jobs", ""]
for k, j in jobs.items():
    L.append(f"- {k}: job {j}, {sacct(j)}")
L.append("")
for k in ("build", "buildprof"):
    if k in jobs:
        m = re.search(r"COMMIT=(\w+) NAME=(\S+)", logtext("gbold-build", jobs[k]))
        L.append(f"- {k}: " + (f"commit {m.group(1)[:8]} name {m.group(2)}" if m else "commit not found in the log"))
L.append("")

if "ident" in jobs:
    d = f"{RES}/ident-{jobs['ident']}"
    L += ["## Identity (screen on against off, and against the CPU engine)", "", "```"]
    L += [l for l in read(d + "/verdicts.txt").splitlines() if re.search(r"VERDICT|DIFFER|OVERALL", l)][:60]
    L += ["```", "", "Demux lines (did the screen fire):", "", "```"]
    for f in sorted(glob.glob(d + "/*.demux-lines.txt")):
        L.append(os.path.basename(f))
        L += read(f).splitlines()[:30]
    L += ["```", ""]

if "kang" in jobs:
    d = f"{RES}/kang-{jobs['kang']}"
    L += ["## Kang 8-donor pool", "", "```"] + read(d + "/summary.txt").splitlines()[:40] + ["```", ""]

if "prof" in jobs:
    d = f"{RES}/prof-{jobs['prof']}"
    L += ["## Profile of one pbmc1k_v3 run (refs and stream)", "", "```"] + read(d + "/summary.txt").splitlines()[:140] + ["```", ""]
    L += [f"Full reports: {d}/perf-*.txt (symbols per stage, callers), {d}/threads.tsv, {d}/windows.tsv", ""]

if "tp" in jobs:
    d = f"{RES}/tp-{jobs['tp']}"
    rows = list(csv.DictReader(open(d + "/results.tsv"), delimiter="\t")) if os.path.exists(d + "/results.tsv") else []
    L += ["## Timing, k=8 T=7 2 slots, order off, on, on, off", ""]
    cols = ["point", "runs", "failed_runs", "win_rate", "core_s_per_M", "lock_held_s", "lock_wait_s", "cores_win",
            "host_peak_gb", "fallback_runs", "mean_wall_s", "node"]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "---|" * len(cols))
    for r in rows:
        L.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    L.append("")

    def arm(r):
        return "on" if "scron" in r["point"] else ("off" if "scroff" in r["point"] else None)

    def num(r, c):
        try:
            return float(r[c])
        except Exception:
            return None

    g = {"on": [], "off": []}
    for r in rows:
        a = arm(r)
        if a:
            g[a].append(r)
    res = {}
    for c in ("core_s_per_M", "lock_held_s", "win_rate", "span_rate", "cores_win", "mean_wall_s"):
        on = [num(r, c) for r in g["on"] if num(r, c) is not None]
        off = [num(r, c) for r in g["off"] if num(r, c) is not None]
        if on and off:
            res[c] = (st.mean(off), st.mean(on), st.mean(on) / st.mean(off) - 1)
    if res:
        L += ["| metric | off (mean of 2) | on (mean of 2) | on vs off |", "|---|---|---|---|"]
        for c, (a, b, dlt) in res.items():
            L.append(f"| {c} | {a:.3f} | {b:.3f} | {100 * dlt:+.1f}% |")
        L.append("")
        if "core_s_per_M" in res:
            dlt = res["core_s_per_M"][2]
            L.append(f"**Go rule (core-s/M falls 15% or more): {'MET' if dlt <= -0.15 else 'NOT MET'} ({100 * dlt:+.1f}%).**")
            offs = [num(r, "core_s_per_M") for r in g["off"]]
            ons = [num(r, "core_s_per_M") for r in g["on"]]
            L.append(f"Leg values of core-s/M: off {offs}, on {ons}.")
        L.append("")
    L += [f"Details: {d}/ (point.txt per point).", ""]

L += ["## Reading", "",
      "- The screen is a gate only when identity is IDENTICAL on both samples and Kang auto-K is 8 with assignments equal to screen-off.",
      "- The screen defaults to on in code. SINGLET_DEMUX_SCREEN=0 turns it off. Flip `screen_single_donor` in include/singlet/pileup/donor_demux.h if identity or Kang fail.",
      "- The refs fix (mmap or shared memory for the genome) was not implemented before the profile. See the profile section for the share of copy and decode.", ""]
open(out + "/summary.md", "w").write("\n".join(L))
print("\n".join(L)[:6000])
