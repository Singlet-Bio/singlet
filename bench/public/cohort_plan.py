#!/usr/bin/env python3
"""Compute budget of the public benchmark cohort: task-hours per dataset and the wall clock at N concurrent tasks.

    python bench/public/cohort_plan.py [--concurrent 12] [--tier core] [--markdown]

Cost model (wall hours of one 16-thread Clipper task; R = read pairs used / 100M). The coefficients come from the
pipeline-benchmark runs (docs/pipeline-validation.md, `time -v` on 16 threads), rounded up:
  singlet 0.15 + 0.30 R | starsolo 0.10 + 0.30 R | alevinfry 0.05 + 0.06 R | cellranger family 0.20 + 0.80 R
  plate protocols (one run per cell; genome kept in shared memory) add 0.5 h per tool
  every other tool (SoupX, vireo, somalier, ...): 0.5 h per dataset; download: fastq_gb / 0.1 GB/s (a 4-CPU task)
Per-sample studies (cxg16, geo_sex21, mage150) are costed per sample below. Download tasks run alongside compute.
"""
from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path

HERE = Path(__file__).resolve().parent
LINEAR = {"singlet": (0.15, 0.30), "starsolo": (0.10, 0.30), "alevinfry": (0.05, 0.06), "cellranger": (0.20, 0.80),
          "cellranger-arc": (0.20, 0.80), "cellranger-vdj": (0.10, 0.10), "spaceranger": (0.20, 0.80),
          "star-featurecounts": (0.10, 0.30), "salmon": (0.05, 0.06), "dropseq-tools": (0.20, 0.40)}
PLATE = {"smartseq2", "celseq2"}
# studies made of many samples: (number of tasks, hours per task) for the whole row, all tools included
PER_SAMPLE = {"cxg16": (66, 0.9), "geo_sex21": (21, 0.4), "mage150": (150, 0.2)}


def tasks(row: dict) -> list[tuple[str, float]]:
    if row["id"] in PER_SAMPLE:
        n, h = PER_SAMPLE[row["id"]]
        return [(f"{row['id']}#{i}", h) for i in range(n)]
    R = float(row["read_pairs_used_m"] or 0) / 100
    out = []
    for t in row["tools"].split(","):
        a, b = LINEAR.get(t, (0.5, 0.0))
        h = a + b * R + (0.5 if row["chemistry"] in PLATE and t in LINEAR else 0)
        out.append((f"{row['id']}:{t}", h))
    return out


def schedule(durations: list[float], k: int) -> float:
    """Longest-processing-time-first list scheduling on k slots; returns the makespan."""
    slots = [0.0] * k
    heapq.heapify(slots)
    for d in sorted(durations, reverse=True):
        heapq.heappush(slots, heapq.heappop(slots) + d)
    return max(slots)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", type=Path, default=HERE / "cohort.tsv")
    ap.add_argument("--concurrent", type=int, default=12)
    ap.add_argument("--tier", choices=["core", "all"], default="core")
    ap.add_argument("--markdown", action="store_true")
    a = ap.parse_args(argv)
    rows = list(csv.DictReader((l for l in open(a.cohort, encoding="utf-8") if not l.startswith("#")), delimiter="\t"))
    rows = [r for r in rows if a.tier == "all" or r["tier"] == "core"]
    allt, dl = [], []
    print("| dataset | tier | read pairs used (M) | FASTQ GB | compute task-h | download h | longest task h |" if a.markdown
          else f"{'dataset':28s} {'tier':8s} {'pairsM':>7s} {'GB':>6s} {'task-h':>7s} {'dl-h':>5s} {'max-h':>6s}")
    if a.markdown:
        print("|---|---|---|---|---|---|---|")
    for r in rows:
        t = tasks(r)
        allt += [h for _, h in t]
        d = float(r["fastq_gb"] or 0) / 360
        dl.append(d)
        vals = (r["id"], r["tier"], r["read_pairs_used_m"] or "-", r["fastq_gb"] or "-", sum(h for _, h in t), d, max(h for _, h in t))
        print("| {} | {} | {} | {} | {:.1f} | {:.2f} | {:.1f} |".format(*vals) if a.markdown
              else "{:28s} {:8s} {:>7} {:>6} {:7.1f} {:5.2f} {:6.1f}".format(*vals))
    total = sum(allt)
    mk = schedule(allt, a.concurrent)
    gb = sum(float(r["fastq_gb"] or 0) for r in rows)
    print(f"\n{len(rows)} datasets, {gb:,.0f} GB of FASTQ, {total:,.0f} compute task-hours "
          f"(~{total * 16:,.0f} core-h at 16 threads); download {sum(dl):.1f} h at 0.1 GB/s.\n"
          f"Wall clock at {a.concurrent} concurrent tasks: {mk:.1f} h (list scheduling, longest first), "
          f"plus about {max(dl):.1f} h for the largest download before its first task.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
