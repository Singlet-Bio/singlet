#!/usr/bin/env python3
"""Overnight summary of the gpu/bold campaign (reads results on Clipper, writes summary.md and summary.json).

Usage: overnight_summary.py "kind=jobid=label;kind=jobid=label;..." [OUTDIR]
kinds: ident (ident.sbatch job), tp (tp harness job), ncukern (ncu_kern.sbatch job), other (state only).
Reads $W/jobs/gpu-bold/results/{ident,tp,ncukern}-<id>/ and $W/logs/*.out. Never writes outside OUTDIR (default $W/jobs/gpu-bold/results/overnight).
Per job: Slurm state, identity verdicts, per point progress-weighted rate, prefill hold split, lane A/M/B kernel seconds, ncu duration of lane B phase 3
per toggle, the Work sizeof table, MPS errors. Then the advisor's go rules, evaluated.

Rate: progress-weighted (bench/gpu-batch1 trunc.py): each run adds pairs * overlap(run, window) / wall, window = k-th completion to the last start.
Hold split and lane kernel seconds: engine step timers of runs that started after the first start (star_Log.out 'steps (s)' line).
"""
import glob
import json
import os
import re
import statistics as st
import subprocess
import sys

W = "/mnt/projects/debruinz_project/debruinz/singlet"
RES = W + "/jobs/gpu-bold/results"
LOGS = W + "/logs"
LANE_B_READS = 705895   # distinct reads of pbmc1k_v3 that end in lane B (engine log of solo run 800349)


STATES = {}


def sacct(j):
    if str(j) in STATES:
        return STATES[str(j)]
    try:
        o = subprocess.run(["sacct", "-j", str(j), "-n", "-X", "-P", "-o", "State,ExitCode,Elapsed,NodeList"], capture_output=True, text=True, timeout=60).stdout.strip().splitlines()
        return o[0].replace("|", " ") if o else "unknown"
    except Exception as e:
        return f"unknown ({e})"


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


def progress(runs, ws, we):
    tot = 0.0
    for r in runs:
        ov = min(r["end"], we) - max(r["start"], ws)
        if ov > 0 and r["end"] > r["start"] and r.get("pairs"):
            tot += r["pairs"] / 1e6 * ov / (r["end"] - r["start"])
    return tot / (we - ws) if we > ws else None


def steps_of(logtxt):
    m = re.search(r"engine_gpu: steps \(s\): (.*); total ([\d.]+)", logtxt)
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
    h = re.search(r"waited (\d+) s for it, held it ([\d.e+-]+) s", logtxt)
    d["_total"] = float(m.group(2))
    d["_held"] = float(h.group(2)) if h else None
    d["_waited"] = float(h.group(1)) if h else None
    return d


BENIGN = re.compile(r"assuming client .* exit|Client process \d+ disconnected")


def mps_errors(d):
    errs = []
    for f_ in glob.glob(d + "/mps-log/*.log") + glob.glob(d + "/mps-log/*/*.log"):
        for l in read(f_).splitlines():
            if re.search(r"error|fail|fatal|crash|abort", l, re.I) and not BENIGN.search(l):
                errs.append(os.path.basename(f_) + ": " + l.strip()[:200])
    return errs


def tp_job(jid):
    d = f"{RES}/tp-{jid}"
    out = {"points": {}, "mps_errors": mps_errors(d), "run_errors": 0}
    for pj in sorted(glob.glob(d + "/*/point.json")):
        pd = os.path.dirname(pj)
        name = os.path.basename(pd)
        P = json.load(open(pj))
        meta = json.load(open(pd + "/meta.json")) if os.path.exists(pd + "/meta.json") else {}
        res = {}
        rt = pd + "/results.tsv"
        if os.path.exists(rt):
            L = open(rt).read().splitlines()
            res = dict(zip(L[0].split("\t"), L[1].split("\t")))
        runs = [r for r in P["runs"] if r.get("pairs")]
        k = P["k"]
        ends = sorted(r["end"] for r in runs)
        starts = sorted(r["start"] for r in runs)
        prog = progress(runs, ends[k - 1], starts[-1]) if len(ends) >= k and len(runs) > 2 * k else None
        t0 = min(r["start"] for r in runs) if runs else 0
        later = [r for r in runs if r["start"] > t0 + 1] or runs
        sv = []
        for r in later:
            s = steps_of(read(f"{pd}/{r['copy']}/star_Log.out"))
            if s:
                sv.append(s)
            if re.search(r"CUDA error|cudaError", read(f"{pd}/{r['copy']}/run.log.ts")[:2_000_000]):
                out["run_errors"] += 1
        g = lambda key: mean([s.get(key) for s in sv])
        kern = {l: g(f"lane{l}.kernels") for l in "AMB"}
        h2d = mean([sum(s.get(x, 0) for x in ("index.h2d", "tables.h2d", "reads.h2d", "e10.aux.h2d")) for s in sv])
        d2h = mean([sum(s.get(x, 0) for x in ("clip.d2h", "dedup.d2h", "laneA.d2h", "laneM.d2h", "laneB.d2h")) for s in sv])
        host = mean([sum(s.get(x, 0) for x in ("clip.host", "dedup.order", "lanes.host", "hostlane", "records.prep", "records.copy", "readRecord")) for s in sv])
        out["points"][name] = {
            "cfg": meta.get("cfg", "base"), "k": k, "T": P["T"], "slots": P["slots"], "runs": len(runs), "failed": res.get("failed_runs"),
            "progress_rate": prog, "span_rate": res.get("span_rate"), "core_s_per_M": res.get("core_s_per_M"), "lock_held_s": res.get("lock_held_s"),
            "lock_wait_s": res.get("lock_wait_s"), "wall_s": mean([r["wall"] for r in later]),
            "prefill_held_s": g("_held"), "prefill_waited_s": g("_waited"), "laneA_kernels_s": kern["A"], "laneM_kernels_s": kern["M"], "laneB_kernels_s": kern["B"],
            "h2d_s": h2d, "d2h_s": d2h, "host_under_lock_s": host, "n_logs": len(sv),
            "laneB_ms_per_read": (kern["B"] * 1000 / LANE_B_READS) if kern["B"] is not None else None,
        }
    return out


def ident_job(jid):
    t = read(f"{LOGS}/gbold-ident-{jid}.out")
    prov = re.search(r"PROVENANCE .*?commit=(\w+) sample=(\S+) switch=\[(.*?)\]", t)
    ver = re.findall(r"VERDICT (NOT IDENTICAL|IDENTICAL)", t)
    diffs = re.findall(r"^(UNEXPECTED DIFF.*|.*DIFFERS.*)$", t, re.M)
    held = re.findall(r"held it ([\d.e+-]+) s", t)
    d = f"{RES}/ident-{jid}"
    verdict = ver[-1] if ver else None
    note = ""
    if verdict == "NOT IDENTICAL" and diffs and all(re.search(r"nonhost_reads.fq|star_Unmapped", x) for x in diffs) and not any("DIFFERS" in x for x in diffs):
        note = "only the three read files were flagged by out_equiv (record order differs by design) and the sorted-hash comparison found no difference: treated as IDENTICAL (first version of ident.sbatch)"
        verdict = "IDENTICAL"
    return {"verdict_note": note, "commit": prov.group(1) if prov else None, "sample": prov.group(2) if prov else None, "switch": prov.group(3) if prov else None,
            "verdict": verdict, "differs": diffs[:6], "prefill_held_s": [float(x) for x in held[:2]],
            "mps_arm": "MPS daemon started" in t, "mps_errors": mps_errors(d)}


def ncukern_job(jid):
    d = f"{RES}/ncukern-{jid}"
    rows = []
    for l in read(d + "/summary.txt").splitlines():
        m = re.match(r"\| (\w+) \| (\w+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| (\d+) \|", l)
        if m:
            rows.append({"arm": m.group(1), "kernel": m.group(2), "launches": int(m.group(3)), "device_s": float(m.group(4)), "mean_ms": float(m.group(5)),
                         "regs": int(m.group(6)), "grid": int(m.group(7))})
    sizes = ""
    for l in read(d + "/base.marks").splitlines():
        if "Work sizes" in l:
            sizes = l.strip()
    lanek = {}
    for arm in ("base", "bsort", "b64", "bsortb64"):
        m = re.findall(r"lane([ABM])\.kernels ([\d.]+)", read(f"{d}/{arm}.marks"))
        lanek[arm] = {a: float(b) for a, b in m}
    return {"rows": rows, "work_sizes": sizes, "engine_lane_kernels_s": lanek}


def rules(J):
    R = []
    # MPS rule
    for jid, label, kind in J:
        if kind != "tp":
            continue
        pts = TP.get(jid, {}).get("points", {})
        base = [p for p in pts.values() if p["cfg"] == "base"]
        for name, p in pts.items():
            if "mps" in p["cfg"] and base:
                bh = mean([b["prefill_held_s"] for b in base]); bb = mean([b["laneB_kernels_s"] for b in base]); br = mean([b["progress_rate"] for b in base])
                if bh and p["prefill_held_s"] is not None:
                    drop = 1 - p["prefill_held_s"] / bh
                    go = drop >= 0.20 and p["laneB_kernels_s"] is not None and p["laneB_kernels_s"] < 6
                    R.append(f"MPS {label} point {name} (k{p['k']} s{p['slots']}, {p['cfg']}): prefill hold {f(p['prefill_held_s'], 1)} s against {f(bh, 1)} s for the base arms "
                             f"({100 * drop:+.0f}% drop), lane B kernels {f(p['laneB_kernels_s'], 1)} s against {f(bb, 1)} s, rate {f(p['progress_rate'], 3)} against {f(br, 3)} M pairs/s. "
                             f"Rule (hold down 20% or more and lane B under 6 s): **{'GO' if go else 'NO GO'}**.")
    for jid, label, kind in J:
        if kind != "tp":
            continue
        pts = TP.get(jid, {}).get("points", {})
        m1 = [p for p in pts.values() if p["cfg"] == "mps"]
        m2 = [p for p in pts.values() if p["cfg"] == "mpsbsort"]
        if m1 and m2:
            r1 = mean([x["progress_rate"] for x in m1])
            r2 = mean([x["progress_rate"] for x in m2])
            b1 = mean([x["laneB_kernels_s"] for x in m1])
            b2 = mean([x["laneB_kernels_s"] for x in m2])
            iv = {j: IDENT[j]["verdict"] for j in IDENT if IDENT[j].get("switch") == "SINGLET_BOLD_BSORT=1"}
            ok = any(v == "IDENTICAL" for v in iv.values())
            chg = 100 * (r2 / r1 - 1) if r1 and r2 else float("nan")
            word = "counts" if ok else "does not count"
            R.append(f"MPS plus cost sort {label}: rate {f(r2, 3)} against {f(r1, 3)} for MPS alone ({chg:+.0f}%), lane B kernels {f(b2, 1)} s against {f(b1, 1)} s. "
                     f"Cost sort identity verdicts: {iv or 'none'}. The result **{word}**.")
    # budget rule
    for jid, label, kind in J:
        if kind != "tp":
            continue
        pts = TP.get(jid, {}).get("points", {})
        a = [p for p in pts.values() if p["cfg"] == "m45" and p["laneB_ms_per_read"]]
        b = [p for p in pts.values() if p["cfg"] == "m91" and p["laneB_ms_per_read"]]
        if a and b:
            r = mean([x["laneB_ms_per_read"] for x in a]) / mean([x["laneB_ms_per_read"] for x in b])
            verdict = "the Work shrink is worth building" if r >= 2 else ("latency rules, the cost sort comes first" if r < 1.3 else "inconclusive (between 1.3x and 2x)")
            R.append(f"Budget ceiling {label}: lane B kernel time per read at 45 GB is {r:.2f}x that at 91 GB ({f(mean([x['laneB_ms_per_read'] for x in a]), 4)} against "
                     f"{f(mean([x['laneB_ms_per_read'] for x in b]), 4)} ms per read). Rule (2x or more): **{verdict}**.")
    # toggle rule
    nk = [(jid, label) for jid, label, kind in J if kind == "ncukern"]
    idv = {}
    for jid, label, kind in J:
        if kind == "ident" and IDENT.get(jid, {}).get("switch"):
            sw = IDENT[jid]["switch"]
            idv[sw] = IDENT[jid]["verdict"]
    for jid, label in nk:
        rows = NCU.get(jid, {}).get("rows", [])
        tot = {}
        for r in rows:
            if r["kernel"] in ("k_p3e", "k_p3es", "k_p3lb", "k_p3lbs"):
                tot[r["arm"]] = tot.get(r["arm"], 0) + r["device_s"]
        base = tot.get("base")
        for arm, sw in (("bsort", "SINGLET_BOLD_BSORT=1"), ("b64", "SINGLET_BOLD_B64=1"), ("bsortb64", None)):
            if base and tot.get(arm):
                gain = 1 - tot[arm] / base
                iv = idv.get(sw) if sw else ("IDENTICAL" if idv.get("SINGLET_BOLD_BSORT=1") == "IDENTICAL" and idv.get("SINGLET_BOLD_B64=1") == "IDENTICAL" else None)
                go = gain >= 0.15 and iv == "IDENTICAL"
                R.append(f"Toggle {arm} ({label}): lane B phase 3 device time {tot[arm]:.3f} s against {base:.3f} s ({100 * gain:+.0f}%), identity verdict {iv}. "
                         f"Rule (15% on k_p3e and IDENTICAL): **{'GO, run the 16-input corpus on b5' if go else 'NO GO'}**.")
    return R


def main():
    spec = sys.argv[1]
    outd = sys.argv[2] if len(sys.argv) > 2 else RES + "/overnight"
    os.makedirs(outd, exist_ok=True)
    for l in read(outd + "/states.txt").splitlines():   # written by summarize.sbatch with sacct outside the container
        x = l.split("|")
        if len(x) >= 5:
            STATES[x[0]] = " ".join(x[1:5])
    J = []
    for item in spec.split(";"):
        if item.strip():
            kind, jid, label = (item.split("=") + ["", ""])[:3]
            J.append((jid, label, kind))
    global TP, IDENT, NCU
    TP, IDENT, NCU = {}, {}, {}
    S = {}
    md = ["# gpu/bold overnight summary\n", "Written by `bench/gpu_bold/overnight_summary.py`. Rates are progress-weighted M pairs/s. Outputs of ceiling and toggle runs are not compared.\n"]
    for jid, label, kind in J:
        S[jid] = {"label": label, "kind": kind, "state": sacct(jid), "error": None}
        md.append(f"\n## {label} (job {jid}, {kind})\n\nSlurm: {S[jid]['state']}\n")
        try:
            if kind == "ident":
                IDENT[jid] = ident_job(jid); S[jid]["ident"] = IDENT[jid]; i = IDENT[jid]
                md.append(f"\nIdentity: switch `{i['switch']}`, commit {i['commit']}, sample {i['sample']}, **{i['verdict']}**" + (" (" + i["verdict_note"] + ")" if i["verdict_note"] else "") + f", prefill hold A and B {i['prefill_held_s']} s"
                          + (", arm B under MPS" if i["mps_arm"] else "") + "\n")
                if i["differs"]:
                    md.append("\nDifferences: " + "; ".join(i["differs"]) + "\n")
                if i["mps_errors"]:
                    md.append("\nMPS log errors: " + " | ".join(i["mps_errors"][:5]) + "\n")
            elif kind == "tp":
                TP[jid] = tp_job(jid); S[jid]["tp"] = TP[jid]; t = TP[jid]
                md.append("\n| point | cfg | k | T | slots | runs | failed | progress rate | core-s/M | prefill hold s | lane A s | lane M s | lane B s | H2D s | D2H s | host s | wall s |\n|" + "---|" * 17 + "\n")
                for n, p in t["points"].items():
                    md.append(f"| {n} | {p['cfg']} | {p['k']} | {p['T']} | {p['slots']} | {p['runs']} | {p['failed']} | **{f(p['progress_rate'], 3)}** | {p['core_s_per_M']} | {f(p['prefill_held_s'], 1)} | "
                              f"{f(p['laneA_kernels_s'], 1)} | {f(p['laneM_kernels_s'], 1)} | {f(p['laneB_kernels_s'], 1)} | {f(p['h2d_s'], 1)} | {f(p['d2h_s'], 1)} | {f(p['host_under_lock_s'], 1)} | {f(p['wall_s'], 0)} |\n")
                md.append(f"\nRuns with a CUDA or MPS error line: {t['run_errors']}. MPS log errors: {len(t['mps_errors'])}" + ((": " + " | ".join(t["mps_errors"][:5])) if t["mps_errors"] else "") + "\n")
            elif kind == "ncukern":
                NCU[jid] = ncukern_job(jid); S[jid]["ncukern"] = NCU[jid]; n = NCU[jid]
                md.append("\n| arm | kernel (lane B) | launches | device s | mean ms | regs | grid |\n|---|---|---|---|---|---|---|\n")
                for r in n["rows"]:
                    md.append(f"| {r['arm']} | {r['kernel']} | {r['launches']} | {r['device_s']:.3f} | {r['mean_ms']:.2f} | {r['regs']} | {r['grid']} |\n")
                md.append(f"\nEngine lane kernel seconds per arm (step timers, under ncu serialization): {json.dumps(n['engine_lane_kernels_s'])}\n")
                md.append("\nWork sizeof table (bytes): " + (n["work_sizes"] or "not found") + "\n")
        except Exception as e:   # one broken job must not stop the summary
            S[jid]["error"] = repr(e)
            md.append(f"\nSUMMARY ERROR for this job: {e!r}\n")
    R = rules(J)
    md.append("\n## Go rules\n\n" + ("\n".join("- " + r for r in R) if R else "No rule could be evaluated (missing results).") + "\n")
    S["_rules"] = R
    open(outd + "/summary.md", "w").write("".join(md))
    json.dump(S, open(outd + "/summary.json", "w"), indent=1, default=str)
    print("wrote", outd + "/summary.md", len("".join(md)), "bytes;", len(R), "rules")


if __name__ == "__main__":
    main()
