"""Evidence table of the fq-read-source runs: core-s per M reads, wall, peak RSS, pass-1 scan, STAR input reads.

    python summarize.py $W/jobs/fq-read-source/runs
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def grab(text: str, pat: str, cast=float):
    m = re.search(pat, text, re.M)
    return cast(m.group(1)) if m else None


def wall_s(t: str | None):
    if not t:
        return None
    p = [float(x) for x in t.split(":")]
    return p[-1] + 60 * p[-2] + (3600 * p[-3] if len(p) > 2 else 0)


def main(root: Path) -> None:
    print("| sample | run | input reads (M) | core-s | core-s / M reads | wall s | peak RSS GB | cgroup peak GB | pass-1 s | node |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for d in sorted(p for p in root.glob("*/*") if (p / "mem.txt").exists()):
        mem = (d / "mem.txt").read_text()
        log = (d / "run.log").read_text(errors="replace") if (d / "run.log").exists() else ""
        fin = d / "out" / "star_Log.final.out"
        n = grab(fin.read_text(), r"Number of input reads \|\s+(\d+)", int) if fin.exists() else None
        cpu = grab(mem, r"cgroup_cpu_s=(\d+)", int)
        wall = wall_s(grab(mem, r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", str))
        rss = grab(mem, r"Maximum resident set size \(kbytes\): (\d+)", int)
        cg = grab(mem, r"memory_peak_mb=(\d+)", int)
        p1 = grab(log, r"\[1fq-pass1\] Scanned \d+ blocks \(([\d.]+)s\)")
        node = (d / "node.txt").read_text().split()[0].replace("node=", "") if (d / "node.txt").exists() else ""
        failed = "exit " in log[-300:]
        print(f"| {d.parent.name} | {d.name}{' (FAILED)' if failed else ''} | {n / 1e6 if n else float('nan'):.1f} | {cpu} | "
              f"{cpu / (n / 1e6) if cpu and n else float('nan'):.2f} | {wall or 0:.0f} | {rss / 1048576 if rss else 0:.1f} | "
              f"{cg / 1024 if cg else 0:.1f} | {p1 if p1 is not None else '-'} | {node} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
