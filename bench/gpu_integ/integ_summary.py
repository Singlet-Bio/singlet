#!/usr/bin/env python3
"""gpu/bold-integ chain summary (reads results on Clipper, CPU only).

  integ_summary.py "build=ID;ident=ID;paths=ID;tp=ID" OUTDIR     writes OUTDIR/summary.md

Primary timing metrics: host core-s per M pairs and device-lock seconds held per run, each with its ratio to the b0 arm of the same job
(both b0 points pooled). Rate is secondary. Runs that start before the first completion of a point are excluded (ramp-up).
"""
import glob
import json
import os
import random
import re
import statistics as st
import subprocess
import sys

W = "/mnt/projects/debruinz_project/debruinz/singlet"
LOGS = W + "/logs"
RES = W + "/jobs/gpu-integ/results"


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def mean(x):
    x = [v for v in x if v is not None]
    return st.mean(x) if x else None


def f(v, n=2):
    return "NA" if v is None else f"{v:.{n}f}"


def ratio(a, b):
    return None if a is None or not b else a / b


def sacct(j):
    try:
        o = subprocess.run(["sacct", "-j", str(j), "-n", "-X", "-P", "-o", "State,ExitCode,Elapsed,NodeList"], capture_output=True, text=True, timeout=60).stdout.strip()
        return o.splitlines()[0].replace("|", " ") if o else "unknown"
    except Exception as e:  # noqa
        return f"sacct failed: {e}"


def boot_ci(x, n=2000):
    if len(x) < 3:
        return None, None
    rnd = random.Random(1)
    m = sorted(mean([rnd.choice(x) for _ in x]) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def point_stats(pd):
    P = json.load(open(pd + "/point.json"))
    meta = json.load(open(pd + "/meta.json")) if os.path.exists(pd + "/meta.json") else {}
    runs = [r for r in P["runs"] if r.get("pairs") and r.get("rc", 0) == 0]
    first_done = min((r["end"] for r in runs), default=0)
    steady = [r for r in runs if r["start"] >= first_done] or runs
    cs = [r["core_s"] / (r["pairs"] / 1e6) for r in steady if r.get("core_s")]
    held = [sum(v.get("held", 0) or 0 for v in (r.get("locks") or {}).values()) for r in steady]
    rate = [r["pairs"] / 1e6 / r["wall"] for r in steady if r.get("wall")]
    lo, hi = boot_ci(rate)
    k = P["k"]
    return {"cfg": meta.get("cfg", "base"), "slots": P["slots"], "k": k, "node": meta.get("node"), "runs": len(runs), "steady": len(steady),
            "cs": mean(cs), "held": mean(held), "node_rate": P.get("pairs_per_s", 0) / 1e6, "run_rate": mean(rate) * k if rate else None,
            "rate_lo": lo * k if lo else None, "rate_hi": hi * k if hi else None, "pt_cs": P.get("cgroup_core_s_per_M"), "_cs": cs, "_held": held}


def tp_table(jid):
    pts = {os.path.basename(os.path.dirname(p)): point_stats(os.path.dirname(p)) for p in sorted(glob.glob(f"{RES}/tp-{jid}/*/point.json"))}
    b0 = [v for v in pts.values() if v["cfg"] == "b0"]
    bcs = mean([x for v in b0 for x in v["_cs"]])
    bheld = mean([x for v in b0 for x in v["_held"]])
    L = ["| point | arm | slots | node | runs (steady) | core-s/M | ratio to b0 | lock held s/run | ratio to b0 | point core-s/M (all runs) | rate Mpairs/s (steady, 95% CI) | rate (whole point) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in pts.items():
        ci = f"{f(v['run_rate'], 3)} ({f(v['rate_lo'], 3)} to {f(v['rate_hi'], 3)})"
        L.append(f"| {n} | {v['cfg']} | {v['slots']} | {(v['node'] or '').split('.')[0]} | {v['runs']} ({v['steady']}) | {f(v['cs'], 1)} | {f(ratio(v['cs'], bcs), 3)} | "
                 f"{f(v['held'], 1)} | {f(ratio(v['held'], bheld), 3)} | {f(v['pt_cs'], 1)} | {ci} | {f(v['node_rate'], 3)} |")
    L += ["", f"b0 reference (both b0 points pooled): core-s/M {f(bcs, 1)}, lock held {f(bheld, 1)} s per run."]
    return L


def ident_verdicts(jid):
    t = read(f"{LOGS}/gk2-ident-{jid}.out")
    return t, re.findall(r"^(VERDICT .*|UNEXPECTED DIFF.*|.*DIFFERS.*|wall .*|arm .* exit .*|REFUSED.*)$", t, re.M)


def main(spec, outdir):
    os.makedirs(outdir, exist_ok=True)
    jobs = [x.split("=", 1) for x in spec.split(";") if x]
    d = dict(jobs)
    L = ["# gpu/bold-integ summary", "", "## Jobs", "", "| step | job | state exit elapsed node |", "|---|---|---|"]
    for kind, jid in jobs:
        L.append(f"| {kind} | {jid} | {sacct(jid.split(',')[0])} |")
    if "build" in d:
        t = read(f"{LOGS}/gbold-build-{d['build']}.out")
        c = re.search(r"COMMIT=(\w+)", t)
        s = re.search(r"^([0-9a-f]{64})\s", t, re.M)
        L += ["", "## Build", "", f"commit {c.group(1) if c else 'NA'}", f"sha256 {s.group(1) if s else 'NA'}", f"result: {'ok' if '[build]' in t and ' ok in ' in t else 'FAILED, see ' + LOGS + '/gbold-build-' + d['build'] + '.out'}"]
    if "ident" in d:
        t, rows = ident_verdicts(d["ident"])
        L += ["", "## Identity against b0 (CPU engine oracle in the same run)", "", "Arms: ref (b0), base (WSHRINK off), wshrink (on).", ""]
        L += ["- " + r for r in rows] or ["- no verdict lines, see the log"]
    if "paths" in d:
        t = read(f"{RES.replace('gpu-integ', 'gpu-paths')}/paths-{d['paths']}/summary.txt")
        v = read(f"{RES.replace('gpu-integ', 'gpu-paths')}/paths-{d['paths']}/verdicts.tsv")
        L += ["", "## 16-input corpus (GPU engine against the CPU engine of the same build)", ""]
        rows = [l.split("\t") for l in v.splitlines() if l.strip()]
        if rows:
            L += ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
            L += ["| " + " | ".join(c[:60] for c in r) + " |" for r in rows[1:]]
        else:
            L += ["no verdicts.tsv", "", "    " + "\n    ".join(t.splitlines()[-12:])]
        L += ["", [l for l in t.splitlines() if l.startswith("PATHS CORPUS")][-1:] and [l for l in t.splitlines() if l.startswith("PATHS CORPUS")][-1] or "no final line"]
    if "tp" in d:
        t = read(f"{RES}/tp-{d['tp']}/summary.txt")
        L += ["", "## Timing (exclusive node, pbmc1k_v3, k=8, T=7)", "", "Primary: core-s/M and lock held per run, with ratios to the b0 arm. Rate is secondary.", ""]
        L += tp_table(d["tp"]) if glob.glob(f"{RES}/tp-{d['tp']}/*/point.json") else ["no point.json files, see the log"]
        L += ["", "Provenance:", "", "    " + "\n    ".join(l[:400] for l in t.splitlines() if l.startswith("PROVENANCE"))]
    open(outdir + "/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L)[:8000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
