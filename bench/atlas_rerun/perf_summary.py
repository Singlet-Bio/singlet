"""Summarise the perf matrix: wall, CPU, RSS and reserved core-hours per sample per config.

    python perf_summary.py <perf_dataset_dir> <reads_millions>

Reserved cores per sample: T for single runs (a campaign packs T-thread jobs), 16/4 for the x4
configs (4 samples on 16 reserved cores). The genome load into /dev/shm is charged once per
group of samples sharing it (4 for x4_*; for t4_shm it is reported separately because a campaign
job would amortise it over many samples).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

THREADS = {"t4": 4, "t8": 8, "t16": 16}


def gnu_time(p: Path) -> dict:
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text().splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock)"):
            t = ln.split(": ", 1)[1].split(":")
            d["wall"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
        elif ln.startswith("User time"):
            d["user"] = float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["sys"] = float(ln.split(": ")[1])
        elif ln.startswith("Maximum resident"):
            d["rss_gb"] = int(ln.split(": ")[1]) / 1024 ** 2
    return d


def main():
    base = Path(sys.argv[1]); reads_m = float(sys.argv[2])
    rows = []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not re.match(r".+_r\d+$", d.name):
            continue
        cfg = d.name.rsplit("_r", 1)[0]
        node = (d / "node.txt").read_text() if (d / "node.txt").exists() else ""
        m = re.search(r"wall_all=(\d+)s", node)
        cpu = re.search(r"cpu=([^\n]*?) 20\d\d-", node)
        load = gnu_time(d / "genome_load.time.txt").get("wall", 0.0)
        if cfg.startswith("x4"):
            parts = [gnu_time(c / "time.txt") for c in sorted(d.glob("c[0-9]"))]
            if not parts or not m:
                continue
            wall_all = float(m.group(1))
            core_h = 16 * (wall_all + load) / 3600 / 4
            wall = sum(p.get("wall", 0) for p in parts) / len(parts)
            cpu_s = sum(p.get("user", 0) + p.get("sys", 0) for p in parts) / len(parts)
            rss = max(p.get("rss_gb", 0) for p in parts)
        else:
            p = gnu_time(d / "time.txt")
            if not p:
                continue
            t = THREADS.get(cfg.split("_")[0], 4)
            wall, cpu_s, rss = p["wall"], p["user"] + p["sys"], p["rss_gb"]
            core_h = t * wall / 3600
        rows.append((cfg, d.name, wall, cpu_s, rss, core_h, load, cpu.group(1) if cpu else ""))
    print(f"{'config':12s} {'run':16s} {'wall_s':>7s} {'cpu_s':>7s} {'rss_GB':>6s} {'core_h':>6s} "
          f"{'core_s/Mread':>12s} {'genome_load_s':>13s}  cpu")
    for cfg, name, wall, cpu_s, rss, ch, load, c in sorted(rows):
        print(f"{cfg:12s} {name:16s} {wall:7.0f} {cpu_s:7.0f} {rss:6.1f} {ch:6.3f} {ch*3600/reads_m:12.1f} {load:13.0f}  {c}")


if __name__ == "__main__":
    main()
