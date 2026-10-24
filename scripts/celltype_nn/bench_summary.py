#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarise ``bench.py`` output: per method, depth 1-4 accuracy and the exact/coarse/finer/wrong fractions as means over
studies (each study pools its samples' cells), the same pooled over cells, CL-term consistency, calibration (ECE pooled over
cells) and a per-study table. Markdown to stdout, JSON with ``--json``.

  python scripts/celltype_nn/bench_summary.py bench.json [--json summary.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _pool(reps):
    """Pool per-sample reports (weights: testable cells, and per-depth n)."""
    out = dict(n=sum(r["n_testable"] for r in reps))
    w = np.array([r["n_testable"] for r in reps], float)
    for k in ("exact", "coarse", "finer", "wrong", "unassigned"):
        out[k] = float(np.average([r["fractions"][k] for r in reps], weights=w)) if w.sum() else np.nan
    out["consistent"] = out["exact"] + out["coarse"] + out["finer"]
    for d in (1, 2, 3, 4):
        xs = [(r["by_depth"][str(d)]["accuracy"], r["by_depth"][str(d)]["n"]) for r in reps if str(d) in r["by_depth"]]
        out[f"d{d}"] = float(np.average([x for x, _ in xs], weights=[n for _, n in xs])) if xs else np.nan
        out[f"n{d}"] = int(sum(n for _, n in xs))
    cl = defaultdict(float)
    for r, wi in zip(reps, w):
        for k, v in r.get("cl", {}).items():
            cl[k] += v * wi / max(w.sum(), 1)
    out["cl"] = dict(cl)
    bins = defaultdict(lambda: [0, 0.0, 0.0])
    for r in reps:
        for b in (r.get("calibration_subtree") or {}).get("bins", []):
            k = (b["lo"], b["hi"])
            bins[k][0] += b["n"]
            bins[k][1] += b["mean_prob"] * b["n"]
            bins[k][2] += b["accuracy"] * b["n"]
    if bins:
        tot = sum(v[0] for v in bins.values())
        rows = [dict(lo=k[0], hi=k[1], n=v[0], mean_prob=v[1] / v[0], accuracy=v[2] / v[0]) for k, v in sorted(bins.items())]
        out["calibration"] = dict(n=tot, ece=float(sum(r["n"] / tot * abs(r["mean_prob"] - r["accuracy"]) for r in rows)), bins=rows)
    secs = [r.get("s_per_million") for r in reps if r.get("s_per_million") is not None]
    if secs:
        out["s_per_million"] = float(np.average(secs, weights=[r["n_cells"] for r in reps if r.get("s_per_million") is not None]))
        wr = [(r["s_per_million_with_read"], r["n_cells"]) for r in reps if r.get("s_per_million_with_read") is not None]
        if wr:
            out["s_per_million_with_read"] = float(np.average([x for x, _ in wr], weights=[n for _, n in wr]))
    return out


def summarise(bench: dict) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    for rec in bench["results"]:
        for m, rep in rec["methods"].items():
            if "error" in rep:
                continue
            rep = dict(rep, n_cells=rec["n_cells"])
            rep["by_depth"] = {str(k): v for k, v in rep["by_depth"].items()}
            by[m][rec["gse"]].append(rep)
    out = {}
    for m, studies in by.items():
        per = {g: _pool(r) for g, r in studies.items()}
        mean = {}
        for k in ("d1", "d2", "d3", "d4", "exact", "coarse", "finer", "wrong", "unassigned", "consistent"):
            v = [p[k] for p in per.values() if not np.isnan(p[k])]
            mean[k] = float(np.mean(v)) if v else np.nan
            mean[f"n_studies_{k}"] = len(v)
        mean["cl_consistent"] = float(np.mean([p["cl"].get("consistent", np.nan) for p in per.values()]))
        mean["cl_wrong"] = float(np.mean([p["cl"].get("wrong", 0.0) for p in per.values()]))
        pooled = _pool([r for rs in studies.values() for r in rs])
        out[m] = dict(mean_over_studies=mean, pooled=pooled, per_study=per)
    return out


def markdown(s: dict, tissues: dict) -> str:
    L = ["| method | d1 | d2 | d3 | d4 | consistent | wrong | CL consistent | CL wrong | ECE |", "|---|---|---|---|---|---|---|---|---|---|"]
    for m, v in s.items():
        a = v["mean_over_studies"]
        ece = v["pooled"].get("calibration", {}).get("ece")
        L.append(f"| {m} | {a['d1']:.3f} | {a['d2']:.3f} | {a['d3']:.3f} | {a['d4']:.3f} | {a['consistent']:.3f} | {a['wrong']:.3f} | "
                 f"{a['cl_consistent']:.3f} | {a['cl_wrong']:.3f} | {'' if ece is None else f'{ece:.3f}'} |")
    L += ["", "Pooled over cells:", "", "| method | d1 | d2 | d3 | d4 | consistent | wrong |", "|---|---|---|---|---|---|---|"]
    for m, v in s.items():
        a = v["pooled"]
        L.append(f"| {m} | {a['d1']:.3f} | {a['d2']:.3f} | {a['d3']:.3f} | {a['d4']:.3f} | {a['consistent']:.3f} | {a['wrong']:.3f} |")
    ms = list(s)
    L += ["", "Per study (d1/d2/d3, wrong):", "", "| study | tissue | " + " | ".join(ms) + " |", "|---|---|" + "---|" * len(ms)]
    for g in sorted({g for v in s.values() for g in v["per_study"]}):
        cells = []
        for m in ms:
            p = s[m]["per_study"].get(g)
            cells.append("" if p is None else "/".join("-" if np.isnan(p[f"d{d}"]) else f"{p[f'd{d}']:.2f}" for d in (1, 2, 3)) + f", {p['wrong']:.2f}")
        L.append(f"| {g} | {tissues.get(g, '')} | " + " | ".join(cells) + " |")
    for m, v in s.items():
        c = v["pooled"].get("calibration")
        if c:
            L += ["", f"Reliability, {m} (rolled-up probability of the call; truth inside the call's subtree; n = {c['n']}, ECE {c['ece']:.3f}):", "",
                  "| bin | cells | mean probability | accuracy |", "|---|---|---|---|"]
            for b in c["bins"]:
                L.append(f"| {b['lo']:.1f}-{b['hi']:.1f} | {b['n']} | {b['mean_prob']:.3f} | {b['accuracy']:.3f} |")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bench", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    b = json.loads(a.bench.read_text())
    s = summarise(b)
    tissues = {r["gse"]: r.get("cxg_tissue", "") for r in b["results"]}
    if a.json:
        a.json.write_text(json.dumps(s, indent=1, default=float))
    print(markdown(s, tissues))
    return 0


if __name__ == "__main__":
    sys.exit(main())
