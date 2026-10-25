"""Turn the raw timing directories of ap_run.sbatch and opt_run.sbatch into the small artifact tables.

    python collect.py BP_DIR OUT_DIR

BP_DIR   $BP (…/bench-public): reads perf/<dataset>/<tool>_r<rep>/ and perf_opt/<kind>/<name>_r<rep>/
OUT_DIR  $ART/alignment-performance/: writes runs.tsv (one row per tool x dataset x repetition) and opt_runs.tsv
Standard library only.
"""
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def gnu_time(p: Path) -> dict:
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text(errors="replace").splitlines():
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock)"):
            t = ln.split(": ", 1)[1].split(":")
            d["wall_s"] = sum(float(x) * 60 ** i for i, x in enumerate(reversed(t)))
        elif ln.startswith("User time"):
            d["user_s"] = float(ln.split(": ")[1])
        elif ln.startswith("System time"):
            d["sys_s"] = float(ln.split(": ")[1])
        elif ln.startswith("Maximum resident"):
            d["time_rss_gb"] = int(ln.split(": ")[1]) / 1024 ** 2
        elif ln.startswith("Exit status"):
            d["exit"] = int(ln.split(": ")[1])
    return d


def mem(p: Path) -> float | None:
    try:
        return json.loads(p.read_text())["peak_rss_gb"]
    except (OSError, ValueError, KeyError):
        return None


def node_info(d: Path) -> tuple[str, str]:
    t = (d / "node.txt").read_text(errors="replace") if (d / "node.txt").exists() else ""
    m = re.search(r"\] (.*?) (\S+) 20\d\d-", t)
    return (m.group(2), m.group(1)) if m else ("", "")


def log_stats(p: Path) -> dict:
    """Reads and unique-mapping rate from a STAR Log.final.out."""
    d = {}
    if not p.exists():
        return d
    for ln in p.read_text(errors="replace").splitlines():
        if "Number of input reads" in ln:
            d["reads_in"] = int(ln.split("|")[1])
        elif "Uniquely mapped reads %" in ln:
            d["uniq_pct"] = float(ln.split("|")[1].strip().rstrip("%"))
    return d


STAMP = re.compile(r"([A-Z][a-z]{2}) +(\d+) (\d\d:\d\d:\d\d)")


def stamp(line: str):
    m = STAMP.search(line)
    return datetime.strptime(f"2026 {m.group(1)} {m.group(2)} {m.group(3)}", "%Y %b %d %H:%M:%S") if m else None


def star_phases(p: Path) -> dict:
    """Seconds from 'Finished loading the genome' to 'finished mapping', to Solo counting, and the BAM sort, from star_Log.out."""
    d = {}
    if not p.exists():
        return d
    t_gen = t_map = t_solo = t_sort = None
    cache = ""
    for ln in p.read_text(errors="replace").splitlines():
        if ln.startswith("Finished loading the genome") and t_gen is None:
            t_gen = stamp(ln)
        elif "finished mapping" in ln and t_map is None:
            t_map = stamp(ln)
        elif "finished Solo counting" in ln and t_solo is None:
            t_solo = stamp(ln)
        elif "started sorting BAM" in ln and t_sort is None:
            t_sort = stamp(ln)
        elif "SEQCACHE" in ln:
            cache = ln.strip()
    if t_gen and t_map:
        d["mapping_s"] = (t_map - t_gen).total_seconds()
    if t_map and t_solo:
        d["solo_s"] = (t_solo - t_map).total_seconds()
    d["seqcache"] = cache
    m = re.search(r"hits?[ =:]+(\d+)\D+(?:lookups?|total|of)[ =:]+(\d+)", cache)
    if m and int(m.group(2)):
        d["cache_hit_rate"] = int(m.group(1)) / int(m.group(2))
    return d


def collect_runs(bp: Path, out: Path):
    rows = []
    for dsdir in sorted((bp / "perf").glob("*")):
        if not dsdir.is_dir():
            continue
        for d in sorted(dsdir.glob("*_r[0-9]*")):
            tool, rep = d.name.rsplit("_r", 1)
            host, cpu = node_info(d)
            run, enc = gnu_time(d / "run" / "time.txt"), gnu_time(d / "encode" / "time.txt")
            if not run:
                continue
            ex = (d / "exit.txt").read_text().split()[-1] if (d / "exit.txt").exists() else ""
            wall = run["wall_s"] + enc.get("wall_s", 0.0)
            cpu_s = run.get("user_s", 0) + run.get("sys_s", 0) + enc.get("user_s", 0) + enc.get("sys_s", 0)
            rss = max(filter(None, [mem(d / "run" / "mem.json"), mem(d / "encode" / "mem.json")]), default=None)
            st = log_stats(d / "Log.final.out")
            rows.append({"dataset": dsdir.name, "tool": tool, "rep": rep, "host": host, "cpu": cpu, "threads": 16,
                         "wall_s": round(wall, 2), "encode_s": round(enc.get("wall_s", 0.0), 2), "run_s": round(run["wall_s"], 2),
                         "cpu_s": round(cpu_s, 1), "max_rss_gb": round(rss, 3) if rss else "",
                         "time_rss_gb": round(max(run.get("time_rss_gb", 0), enc.get("time_rss_gb", 0)), 3),
                         "exit": ex, "reads_in": st.get("reads_in", ""), "uniq_pct": st.get("uniq_pct", "")})
    cols = ["dataset", "tool", "rep", "host", "cpu", "threads", "wall_s", "encode_s", "run_s", "cpu_s", "max_rss_gb", "time_rss_gb",
            "exit", "reads_in", "uniq_pct"]
    with open(out / "runs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"runs.tsv: {len(rows)} rows")


def collect_opt(bp: Path, out: Path):
    rows = []
    for kdir in sorted((bp / "perf_opt").glob("*")):
        if not kdir.is_dir():
            continue
        for d in sorted(kdir.glob("*_r[0-9]*")):
            if not d.is_dir() or d.name.endswith(".keep"):
                continue
            name, rep = d.name.rsplit("_r", 1)
            host, cpu = node_info(d)
            keep = kdir / (d.name + ".keep")
            if name.startswith("x4"):
                parts = [gnu_time(c / "run" / "time.txt") for c in sorted(d.glob("c[0-9]"))]
                m = re.search(r"wall_all=(\d+)", (d / "node.txt").read_text()) if (d / "node.txt").exists() else None
                if not parts or not m:
                    continue
                wall = float(m.group(1))
                cpu_s = sum(p.get("user_s", 0) + p.get("sys_s", 0) for p in parts)
                rss = max(filter(None, [mem(c / "run" / "mem.json") for c in sorted(d.glob("c[0-9]"))]), default=None)
                copies = 4
            else:
                t = gnu_time(d / "run" / "time.txt")
                if not t:
                    continue
                wall, cpu_s, rss, copies = t["wall_s"], t.get("user_s", 0) + t.get("sys_s", 0), mem(d / "run" / "mem.json"), 1
            ph = star_phases(keep / "star_Log.out")
            cs = keep / "checksums.txt"
            sig = hashlib.md5(cs.read_bytes()).hexdigest()[:12] if cs.exists() and cs.stat().st_size else ""
            rows.append({"outputs_sig": sig,"kind": kdir.name, "name": name, "rep": rep, "host": host, "cpu": cpu, "copies": copies,
                         "wall_s": round(wall, 2), "cpu_s": round(cpu_s, 1), "max_rss_gb": round(rss, 3) if rss else "",
                         "mapping_s": ph.get("mapping_s", ""), "solo_s": ph.get("solo_s", ""),
                         "cache_hit_rate": round(ph["cache_hit_rate"], 4) if "cache_hit_rate" in ph else "",
                         "seqcache_line": ph.get("seqcache", "")[:200]})
    cols = ["kind", "name", "rep", "host", "cpu", "copies", "wall_s", "cpu_s", "max_rss_gb", "mapping_s", "solo_s",
            "cache_hit_rate", "outputs_sig", "seqcache_line"]
    with open(out / "opt_runs.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    print(f"opt_runs.tsv: {len(rows)} rows")


if __name__ == "__main__":
    bp, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    collect_runs(bp, out)
    collect_opt(bp, out)
