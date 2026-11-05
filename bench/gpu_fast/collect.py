#!/usr/bin/env python3
"""Collect the legs of a fast lane job into result.json (schema in abcore.py).

Usage: collect.py RES_DIR
RES_DIR holds params.json and arms.tsv (written by the job) and one directory per leg named leg<index><arm>, for example leg1A.
Each leg directory has the files of bench/gpu_batch1/tp_point.py: results.tsv, fallback.txt, point.json, and the file STOPPED when the
job ended the leg early. Safe to run more than once. The job runs it after the legs and again after the identity checks.
"""
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abcore  # noqa: E402


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def frac(s):
    m = re.fullmatch(r"(\d+)/(\d+)", s or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def collect(res):
    params = json.load(open(f"{res}/params.json")) if os.path.exists(f"{res}/params.json") else {}
    arms = {}
    for r in read_tsv(f"{res}/arms.tsv"):
        arms[r["arm"]] = r
    legs = []
    names = sorted((d for d in os.listdir(res) if re.fullmatch(r"leg\d+[A-Z]", d)), key=lambda d: int(re.findall(r"\d+", d)[0]))
    for d in names:
        idx, arm = int(re.findall(r"\d+", d)[0]), d[-1]
        D = f"{res}/{d}"
        a = arms.get(arm, {})
        leg = {"leg": idx, "arm": arm, "dir": d,
               "prov": {"bin": a.get("bin", "?"), "sha256": a.get("sha256", "?"), "commit": a.get("commit", "?"), "embedded": a.get("embedded", "?"),
                        "env": a.get("env", ""), "args": a.get("args", "")},
               "metrics": {}, "stage_core_s": {}, "ident": {"checked": 0, "identical": 0, "detail": f"{d}/"}, "fallback": {"runs": 0, "lines": []},
               "stopped": "", "completed": False}
        rows = read_tsv(f"{D}/results.tsv")
        if rows:
            r = rows[0]
            n = int(num(r.get("runs")) or 0)
            failed = int(num(r.get("failed_runs")) or 0)
            leg["completed"] = n > 0
            leg["metrics"] = {"wall_s": num(r.get("mean_wall_s")), "core_s_per_M": num(r.get("core_s_per_M")), "lock_held_s": num(r.get("lock_held_s")),
                              "lock_wait_s": num(r.get("lock_wait_s")), "dev_peak_gb": num(r.get("dev_peak_gb")), "host_peak_gb": num(r.get("host_peak_gb")),
                              "win_rate": num(r.get("win_rate")), "span_rate": num(r.get("span_rate")), "win_n": num(r.get("win_n")),
                              "win_s": num(r.get("win_s")), "runs": n, "failed_runs": failed, "cores_win": num(r.get("cores_win")),
                              "gpu_util": num(r.get("gpu_util")), "other_jobs_max": num(r.get("other_jobs_max"))}
            ok, tot = frac(r.get("ident_cpu"))
            leg["ident"].update(checked=tot, identical=ok)
            leg["fallback"]["runs"] = int(num(r.get("fallback_runs")) or 0)
            if r.get("node") and not params.get("node"):
                params["node"] = r["node"]
        fb = f"{D}/fallback.txt"
        if os.path.exists(fb):
            for l in open(fb, errors="replace"):
                f = l.rstrip("\n").split("\t")
                if len(f) >= 3 and f[1] == "FALLBACK":
                    leg["fallback"]["lines"].append(f"{f[0]}: {f[2]}")
        pj = f"{D}/point.json"
        if os.path.exists(pj):
            P = json.load(open(pj))
            acc = {}
            for r in P.get("runs", []):
                if r.get("rc") == 0:
                    for k, v in (r.get("stage_core_s") or {}).items():
                        acc.setdefault(k, []).append(v)
            leg["stage_core_s"] = {k: sum(v) / len(v) for k, v in acc.items()}
            k = len({r.get("worker") for r in P.get("runs", [])}) or 1
            leg["metrics"]["progress_rate"] = abcore.progress_rate(P.get("runs", []), k)
        sp = f"{D}/STOPPED"
        if os.path.exists(sp):
            leg["stopped"] = open(sp).read().strip()
        legs.append(leg)
    notes = []
    if os.path.exists(f"{res}/notes.txt"):
        notes = [l.strip() for l in open(f"{res}/notes.txt") if l.strip()]
    out = {"schema": 1, "tier": params.get("tier"), "name": params.get("name"), "job": params.get("job"), "node": params.get("node"),
           "scripts": params.get("scripts"), "ref": params.get("ref"), "params": params, "legs": legs, "notes": notes}
    json.dump(out, open(f"{res}/result.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    o = collect(sys.argv[1].rstrip("/"))
    print(f"collected {len(o['legs'])} legs into {sys.argv[1]}/result.json")
