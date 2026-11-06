#!/usr/bin/env python3
"""gpu/bold-kern2 campaign: summary and decider (reads results on Clipper, CPU only).

  kern2_summary.py summary "kind=jobid=label;..." OUTDIR     writes OUTDIR/summary.md
  kern2_summary.py decide IDENT_ID NCU_ID TP_ID               prints a JSON decision: which of wshrink / bstream to combine

kinds: build, ident, ncu, paths, tp, decider, other. Rate: progress-weighted as in bench/gpu-batch1 trunc.py.
Winner rule (the coordinator's): an arm that is IDENTICAL against b0 and at least 10% faster on lane B's phase 3 kernel (ncu device seconds) or on the
prefill hold (mean seconds the device lock was held per run) than the base arm.
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

W = "/mnt/projects/debruinz_project/debruinz/singlet"
RES = W + "/jobs/gpu-kern2/results"
LOGS = W + "/logs"


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


def sacct(j):
    try:
        o = subprocess.run(["sacct", "-j", str(j), "-n", "-X", "-P", "-o", "State,ExitCode,Elapsed,NodeList"], capture_output=True, text=True, timeout=60).stdout.strip()
        return o.splitlines()[0].replace("|", " ") if o else "unknown"
    except Exception as e:  # noqa
        return f"sacct failed: {e}"


def progress(runs, ws, we):
    tot = 0.0
    for r in runs:
        ov = min(r["end"], we) - max(r["start"], ws)
        if ov > 0 and r["end"] > r["start"] and r.get("pairs"):
            tot += r["pairs"] / 1e6 * ov / (r["end"] - r["start"])
    return tot / (we - ws) if we > ws else None


def steps_of(t):
    m = re.search(r"engine_gpu: steps \(s\): (.*); total ([\d.]+)", t)
    if not m:
        return None
    d = {}
    for kv in m.group(1).split(", "):
        if " " in kv:
            k, v = kv.rsplit(" ", 1)
            try:
                d[k.strip()] = float(v)
            except ValueError:
                pass
    h = re.search(r"waited (\d+) s for it, held it ([\d.e+-]+) s", t)
    d["_held"] = float(h.group(2)) if h else None
    d["_waited"] = float(h.group(1)) if h else None
    return d


def tp_job(jid):
    d = f"{RES}/tp-{jid}"
    out = {}
    for pj in sorted(glob.glob(d + "/*/point.json")):
        pd = os.path.dirname(pj)
        P = json.load(open(pj))
        meta = json.load(open(pd + "/meta.json")) if os.path.exists(pd + "/meta.json") else {}
        runs = [r for r in P["runs"] if r.get("pairs")]
        k = P["k"]
        ends = sorted(r["end"] for r in runs)
        starts = sorted(r["start"] for r in runs)
        prog = progress(runs, ends[k - 1], starts[-1]) if len(ends) >= k and len(runs) > 2 * k else None
        t0 = min(r["start"] for r in runs) if runs else 0
        later = [r for r in runs if r["start"] > t0 + 1] or runs
        sv = [s for s in (steps_of(read(f"{pd}/{r['copy']}/star_Log.out")) for r in later) if s]
        g = lambda key: mean([s.get(key) for s in sv])
        out[os.path.basename(pd)] = {"cfg": meta.get("cfg", "base"), "slots": P["slots"], "k": k, "runs": len(runs), "rate": prog, "held": g("_held"), "waited": g("_waited"),
                                     "A": g("laneA.kernels"), "M": g("laneM.kernels"), "B": g("laneB.kernels"), "wall": mean([r["wall"] for r in later]),
                                     "node": meta.get("node")}
    return out


def tp_by_arm(pts):
    arms = {}
    for name, v in pts.items():
        arms.setdefault((v["cfg"], v["slots"]), []).append(v)
    return {a: {key: mean([x[key] for x in vs]) for key in ("rate", "held", "waited", "A", "M", "B", "wall")} | {"points": len(vs)} for a, vs in arms.items()}


def ident_verdicts(jid):
    t = read(f"{LOGS}/gk2-ident-{jid}.out")
    return {m.group(2): m.group(1) == "IDENTICAL" for m in re.finditer(r"VERDICT (NOT IDENTICAL|IDENTICAL) \(\S+, arm (\w+), against b0\)", t)}


def ncu_rows(jid):
    rows = []
    for l in read(f"{RES}/ncukern-{jid}/summary.txt").splitlines():
        m = re.match(r"\| (\w+) \| (\w+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| (\d+) \| (\S+) \| (\S+) \|", l)
        if m:
            rows.append({"arm": m.group(1), "kernel": m.group(2), "launches": int(m.group(3)), "device_s": float(m.group(4)), "grid": int(m.group(6)),
                         "batch": m.group(8), "per_read": m.group(9)})
    return rows


def ncu_p3(rows):
    t = {}
    for r in rows:
        if r["kernel"] in ("k_p3e", "k_p3c"):
            t[r["arm"]] = t.get(r["arm"], 0) + r["device_s"]
    return t


def decide(ident, ncu, tp):
    ver = ident_verdicts(ident)
    p3 = ncu_p3(ncu_rows(ncu))
    arms = tp_by_arm(tp_job(tp))
    base = next((v for (c, s), v in arms.items() if c == "base"), {})
    res = {"verdicts": ver, "p3_device_s": p3, "base_hold": base.get("held"), "winners": [], "why": {}}
    for arm in ("wshrink", "bstream"):
        a = next((v for (c, s), v in arms.items() if c == arm), {})
        # an arm is also judged through the combined arm's identity when only 'both' ran identical? No: its own verdict is required.
        ident_ok = ver.get(arm, False)
        k_gain = (1 - p3[arm] / p3["base"]) if p3.get(arm) and p3.get("base") else None
        h_gain = (1 - a["held"] / base["held"]) if a.get("held") and base.get("held") else None
        ok = ident_ok and ((k_gain is not None and k_gain >= 0.10) or (h_gain is not None and h_gain >= 0.10))
        res["why"][arm] = {"identical": ident_ok, "p3_gain": k_gain, "hold_gain": h_gain}
        if ok:
            res["winners"].append(arm)
    res["arm"] = "both" if len(res["winners"]) == 2 else (res["winners"][0] if res["winners"] else None)
    if res["arm"] == "both" and not ver.get("both", False):
        res["why"]["both"] = "the combined arm was not IDENTICAL; submitting only the faster single winner"
        best = max(res["winners"], key=lambda w: (res["why"][w]["hold_gain"] or 0) + (res["why"][w]["p3_gain"] or 0))
        res["arm"] = best
    return res


def summary(spec, outdir):
    os.makedirs(outdir, exist_ok=True)
    L = ["# gpu/bold-kern2 overnight summary", ""]
    jobs = [x.split("=", 2) for x in spec.split(";") if x]
    L += ["## Jobs", "", "| kind | job | label | state exit elapsed node |", "|---|---|---|---|"]
    for kind, jid, label in jobs:
        L.append(f"| {kind} | {jid} | {label} | {sacct(jid)} |")
    for kind, jid, label in jobs:
        if kind == "build":
            t = read(f"{LOGS}/gbold-build-{jid}.out")
            c = re.search(r"COMMIT=(\w+)", t); s = re.search(r"^([0-9a-f]{64})\s", t, re.M)
            L += ["", f"## Build {jid} ({label})", "", f"commit {c.group(1) if c else 'NA'}, sha256 {s.group(1) if s else 'NA'}"]
        elif kind == "ident":
            t = read(f"{LOGS}/gk2-ident-{jid}.out")
            L += ["", f"## Identity {jid} ({label})", ""]
            L += ["- " + l for l in re.findall(r"^(VERDICT .*|UNEXPECTED DIFF.*|.*DIFFERS.*|wall .*|arm .* exit .*)$", t, re.M)]
            L += ["- " + l[:600] for l in re.findall(r"(kern2 wshrink=.*)", t)[:6]]
        elif kind == "ncu":
            t = read(f"{RES}/ncukern-{jid}/summary.txt")
            L += ["", f"## ncu lane B phase 3 kernel {jid} ({label})", "", t]
            for arm in ("base", "wshrink", "bstream", "both"):
                m = read(f"{RES}/ncukern-{jid}/{arm}.marks")
                for x in re.findall(r"\[wshrink: [^\]]*\]|\[B to host [^\]]*\]", m)[:8]:
                    L.append(f"- {arm}: {x}")
        elif kind == "paths":
            t = read(f"{LOGS}/gpu-paths-{jid}.out")
            L += ["", f"## 16-input corpus {jid} ({label})", ""]
            L += ["    " + l[:200] for l in t.splitlines()[-24:]]
        elif kind == "tp":
            pts = tp_job(jid)
            L += ["", f"## Timing {jid} ({label})", "", "| point | cfg | slots | node | runs | rate (progress-weighted, Mpairs/s) | hold s | wait s | laneA s | laneM s | laneB s |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for n, v in pts.items():
                L.append(f"| {n} | {v['cfg']} | {v['slots']} | {v['node']} | {v['runs']} | {f(v['rate'], 4)} | {f(v['held'])} | {f(v['waited'], 1)} | {f(v['A'])} | {f(v['M'])} | {f(v['B'])} |")
            L += ["", "Mean per arm (the repeats of an arm averaged):", "", "| cfg | slots | points | rate | hold s | laneB s |", "|---|---|---|---|---|---|"]
            for (c, s), v in sorted(tp_by_arm(pts).items()):
                L.append(f"| {c} | {s} | {v['points']} | {f(v['rate'], 4)} | {f(v['held'])} | {f(v['B'])} |")
        elif kind == "decider":
            L += ["", f"## Decider {jid} ({label})", "", "    " + read(f"{RES}/decider-{jid}/decision.json")[:1500].replace("\n", "\n    ")]
            nxt = read(f"{RES}/decider-{jid}/next_jobs.txt").strip()
            if nxt:
                L.append("")
                L.append("Follow-up jobs: " + nxt)
                m = re.search(r"tp=(\d+)", nxt)
                if m and os.path.isdir(f"{RES}/tp-{m.group(1)}"):
                    pts = tp_job(m.group(1))
                    L += ["", f"### Follow-up timing {m.group(1)}", "", "| point | cfg | slots | rate | hold s | laneB s |", "|---|---|---|---|---|---|"]
                    for n, v in pts.items():
                        L.append(f"| {n} | {v['cfg']} | {v['slots']} | {f(v['rate'], 4)} | {f(v['held'])} | {f(v['B'])} |")
                    L.append(f"State of {m.group(1)}: {sacct(m.group(1))}")
    open(outdir + "/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L)[:6000])


if __name__ == "__main__":
    if sys.argv[1] == "summary":
        summary(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "decide":
        print(json.dumps(decide(sys.argv[2], sys.argv[3], sys.argv[4])))
