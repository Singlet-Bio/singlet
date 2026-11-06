#!/usr/bin/env python3
"""Table of throughput points for the gpu/bold stage ceiling runs.

Usage: analyze_points.py RESULTS_DIR [RESULTS_DIR ...]
Each RESULTS_DIR is a tp job directory (jobs/gpu-bold/results/tp-<job>) or a local copy of it that holds, per point,
point.json, results.tsv, meta.json and dmon.log.

Rate. The progress-weighted rate of bench/gpu-batch1 analysis (scratchpad trunc.py): every run adds pairs * overlap(run, window) / wall.
The window starts when the k-th run completes and ends at the last run start. Only equal run counts may be compared by span.
The windowed and span rates of tp_point.py are printed next to it.

Per-run measures are means over runs that start after the first start (the later runs).
Device util is the mean of nvidia-smi dmon sm% over the span. Copy load is the mean of dmon rxpci and txpci MB/s over the span.
"""
import glob
import json
import os
import statistics as st
import sys


def progress(runs, ws, we):
    tot = 0.0
    for r in runs:
        ov = min(r["end"], we) - max(r["start"], ws)
        if ov > 0 and r["end"] > r["start"]:
            tot += r["pairs"] / 1e6 * ov / (r["end"] - r["start"])
    return tot / (we - ws) if we > ws else float("nan")


def dmon_means(path, t0, t1):
    """mean sm%, mem%, rx MB/s, tx MB/s, power W between t0 and t1 (epoch s) from nvidia-smi dmon -o T output"""
    if not os.path.exists(path):
        return None
    hdr = None
    rows = []
    for l in open(path, errors="replace"):
        if l.startswith("#"):
            hdr = l.split()[1:] if "Time" in l or "gpu" in l else hdr
            continue
        f = l.split()
        if len(f) < 10:
            continue
        rows.append(f)
    if not rows:
        return None
    # the header line of dmon -o T starts with "# Date Time gpu pwr ..."; recover names from the first '#' line that has 'sm'
    names = None
    for l in open(path, errors="replace"):
        if l.startswith("#") and " sm " in l + " ":
            names = l.split()[1:]
            break
    if not names:
        return None
    out = {n: [] for n in ("sm", "mem", "rxpci", "txpci", "pwr")}
    import datetime
    for f in rows:
        try:
            d = dict(zip(names, f))
            ts = datetime.datetime.strptime(d["Date"] + " " + d["Time"], "%Y%m%d %H:%M:%S").timestamp()
        except Exception:
            continue
        if ts < t0 or ts > t1:
            continue
        for n in out:
            try:
                out[n].append(float(d[n]))
            except Exception:
                pass
    return {n: (st.mean(v) if v else float("nan")) for n, v in out.items()}


def main():
    rows = []
    for rd in sys.argv[1:]:
        for pj in sorted(glob.glob(os.path.join(rd, "*", "point.json"))):
            pd = os.path.dirname(pj)
            P = json.load(open(pj))
            meta = json.load(open(os.path.join(pd, "meta.json"))) if os.path.exists(os.path.join(pd, "meta.json")) else {}
            res = {}
            rt = os.path.join(pd, "results.tsv")
            if os.path.exists(rt):
                L = open(rt).read().splitlines()
                res = dict(zip(L[0].split("\t"), L[1].split("\t")))
            runs = [r for r in P["runs"] if r.get("pairs")]
            k = P["k"]
            ends = sorted(r["end"] for r in runs)
            starts = sorted(r["start"] for r in runs)
            prog = float("nan")
            if len(ends) >= k and len(runs) > 2 * k:
                prog = progress(runs, ends[k - 1], starts[-1])
            t0 = min(r["start"] for r in runs)
            later = [r for r in runs if r["start"] > t0 + 1] or runs
            m = lambda key: st.mean(r[key] for r in later if r.get(key) is not None)
            stages = {}
            for r in later:
                for n, v in (r.get("stages") or {}).items():
                    stages.setdefault(n, []).append(v)
            dm = dmon_means(os.path.join(pd, "dmon.log"), P["first_start"], P["last_end"])
            rows.append(dict(
                point=P["tag"], cfg=meta.get("cfg", "base"), k=k, T=P["T"], slots=P["slots"], n=len(runs),
                prog=prog, win=res.get("win_rate", "NA"), span=res.get("span_rate", "NA"), cs_M=res.get("core_s_per_M", "NA"),
                cores=res.get("cores_win", "NA"), held=res.get("lock_held_s", "NA"), wait=res.get("lock_wait_s", "NA"),
                util=res.get("gpu_util", "NA"), wall=m("wall"), core_s=m("core_s"), host=res.get("host_peak_gb", "NA"),
                dev=res.get("dev_peak_gb", "NA"), failed=res.get("failed_runs", "?"), fb=res.get("fallback_runs", "?"),
                stages={n: st.mean(v) for n, v in stages.items()}, dm=dm, job=res.get("job", os.path.basename(rd)), node=res.get("node", "?")))
    f = lambda v, p=2: ("NA" if v is None or v != v else (f"{v:.{p}f}" if isinstance(v, float) else str(v)))
    print("| job | node | point | cfg | k | T | slots | runs | progress M pairs/s | window | span | core-s/M | cores busy | lock held s | lock wait s | gpu util % | dmon sm % | per-sample wall s | core-s/run | host GB | dev GB | failed | fallback |")
    print("|" + "---|" * 23)
    for r in rows:
        dm = r["dm"] or {}
        print(f"| {r['job']} | {r['node']} | {r['point']} | {r['cfg']} | {r['k']} | {r['T']} | {r['slots']} | {r['n']} | **{f(r['prog'], 3)}** | {r['win']} | {r['span']} | {r['cs_M']} | "
              f"{r['cores']} | {r['held']} | {r['wait']} | {r['util']} | {f(dm.get('sm'), 0)} | {f(r['wall'], 1)} | {f(r['core_s'], 0)} | {r['host']} | {r['dev']} | {r['failed']} | {r['fb']} |")
    print()
    print("Stage wall s (mean of later runs):")
    names = ["start", "scan", "refs", "STAR", "stream", "handoff", "pileup", "pileupfin", "nonhost", "export", "demux"]
    print("| point | " + " | ".join(names) + " |")
    print("|" + "---|" * (len(names) + 1))
    for r in rows:
        print(f"| {r['point']} | " + " | ".join(f(r["stages"].get(n), 1) for n in names) + " |")
    print()
    print("Device copy load (dmon, mean MB/s over the span) and power:")
    for r in rows:
        dm = r["dm"] or {}
        print(f"  {r['point']}: rx {f(dm.get('rxpci'), 0)} tx {f(dm.get('txpci'), 0)} MB/s, sm {f(dm.get('sm'), 0)}%, mem {f(dm.get('mem'), 0)}%, {f(dm.get('pwr'), 0)} W")


if __name__ == "__main__":
    main()
