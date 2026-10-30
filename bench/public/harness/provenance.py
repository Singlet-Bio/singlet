#!/usr/bin/env python3
"""Resources and provenance for the contract, from the harness's run directories.

    python bench/public/harness/provenance.py RUNS_DIR OUT.json

RUNS_DIR is $BP/runs/<dataset>/<tool>/. Each run directory has a time.txt from GNU time -v (Singlet also has
encode.time.txt, which is added to its run), plus node.txt. OUT.json holds one `resources` entry per (dataset, tool), in
the contract's shape: tool, dataset, wall_s, cpu_h, max_rss_gb, threads, node.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def parse_time(p: Path) -> dict:
    t = p.read_text(errors="replace")
    def f(pat, conv=float, default=0.0):
        m = re.search(pat, t)
        return conv(m.group(1)) if m else default
    wall = t and re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", t)
    secs = 0.0
    if wall:
        parts = [float(x) for x in wall.group(1).split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        secs = parts[0] * 3600 + parts[1] * 60 + parts[2]
    user = f(r"User time \(seconds\): ([\d.]+)")
    sys_ = f(r"System time \(seconds\): ([\d.]+)")
    rss_kb = f(r"Maximum resident set size \(kbytes\): (\d+)")
    return {"wall_s": secs, "cpu_s": user + sys_, "max_rss_kb": rss_kb}


def resources(runs: Path) -> list[dict]:
    out = []
    for tdir in sorted(runs.glob("*/*/")):
        ds, tool = tdir.parent.name, tdir.name
        tt = tdir / "time.txt"
        if not tt.exists():
            continue
        a = parse_time(tt)
        enc = tdir / "encode.time.txt"
        if enc.exists():
            b = parse_time(enc)
            a = {"wall_s": a["wall_s"] + b["wall_s"], "cpu_s": a["cpu_s"] + b["cpu_s"],
                 "max_rss_kb": max(a["max_rss_kb"], b["max_rss_kb"])}
        node = (tdir / "node.txt").read_text().strip() if (tdir / "node.txt").exists() else ""
        m = re.search(r"threads (\d+)", node)
        out.append({"tool": tool, "dataset": ds, "wall_s": round(a["wall_s"], 1), "cpu_h": round(a["cpu_s"] / 3600, 3),
                    "max_rss_gb": round(a["max_rss_kb"] / 1048576, 2), "threads": int(m.group(1)) if m else 16,
                    "node": node.split(";")[0].replace("cpu: ", "")})
    return out


if __name__ == "__main__":
    runs, out = Path(sys.argv[1]), Path(sys.argv[2])
    res = resources(runs)
    out.write_text(json.dumps(res, indent=1))
    print(f"{len(res)} runs -> {out}")
