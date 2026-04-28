#!/usr/bin/env python3
"""VAL1 Quality Report Generator — aggregates singlify validation results."""

import csv, re, sys
from pathlib import Path
from collections import defaultdict

SAMPLE_CSV = Path(__file__).parent / "val1_samples.csv"
VAL_BASE = Path("/mnt/projects/debruinz_project/singlify_validation/val1")
LOG_DIR = Path(__file__).parent.parent / "val1_logs"


def parse_star_log(log_path):
    metrics = {}
    if not log_path.exists():
        return metrics
    for line in open(log_path, errors="replace"):
        if "Number of input reads" in line:
            try:
                metrics["input_reads"] = int(line.split("|")[-1].strip())
            except ValueError:
                pass
        elif "Uniquely mapped reads %" in line:
            try:
                metrics["unique_map_pct"] = float(line.split("|")[-1].strip().rstrip("%"))
            except ValueError:
                pass
        elif "% of reads mapped to multiple loci" in line:
            try:
                metrics["multi_map_pct"] = float(line.split("|")[-1].strip().rstrip("%"))
            except ValueError:
                pass
    return metrics


def parse_proc_log(log_path):
    result = {"srr": None, "status": "not_started", "wall_sec": 0.0, "cells": 0, "error_hint": ""}
    if not log_path.exists() or log_path.stat().st_size == 0:
        return result
    try:
        text = open(log_path, errors="replace").read()
    except OSError:
        return result

    # Extract SRR from START line
    m = re.search(r"START:\s+(SRR\S+)", text)
    if m:
        result["srr"] = m.group(1)

    # Wall time
    m = re.search(r"wall=([\d.]+)", text)
    if m:
        result["wall_sec"] = float(m.group(1))

    # Cell count from cell_qc line: "[cell_qc] Wrote: ... (N cells)"
    m = re.search(r"\[cell_qc\] Wrote:.*\((\d+) cells\)", text)
    if m:
        result["cells"] = int(m.group(1))

    # Status determination (order matters: check success first)
    if re.search(r"DONE:\s+SRR\S+\s+exit=0", text):
        result["status"] = "success"
    elif re.search(r"SKIP:", text):
        result["status"] = "skipped"
        m2 = re.search(r"SKIP:\s+SRR\S+\s+(.*)", text)
        if m2:
            result["error_hint"] = m2.group(1).strip()
    elif "oom_kill" in text:
        result["status"] = "oom"
        result["error_hint"] = "SLURM OOM kill"
    elif re.search(r"footer magic invalid", text):
        result["status"] = "corrupt_1fq"
        result["error_hint"] = ".1fq truncated/corrupt"
    elif re.search(r"DONE:\s+SRR\S+\s+exit=(?!0)", text):
        result["status"] = "error"
        # Capture exit code
        m2 = re.search(r"exit=(\d+)", text)
        if m2:
            result["error_hint"] = f"exit={m2.group(1)}"
    elif re.search(r"\[singlify\] ERROR:", text):
        result["status"] = "error"
        m2 = re.search(r"\[singlify\] ERROR:\s*(.*)", text)
        if m2:
            result["error_hint"] = m2.group(1).strip()[:80]
    elif re.search(r"Command exited with non-zero status", text):
        result["status"] = "error"
        m2 = re.search(r"non-zero status (\d+)", text)
        if m2:
            result["error_hint"] = f"exit={m2.group(1)}"
    elif result["srr"]:
        result["status"] = "running"

    return result


def count_output_files(output_dir):
    if output_dir.exists():
        return sum(1 for _ in output_dir.iterdir())
    return 0


def median(vals):
    s = sorted(vals)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    # Read sample CSV
    samples = {}
    with open(SAMPLE_CSV) as f:
        for row in csv.DictReader(f):
            samples[row["srr_accession"]] = row

    # Parse all proc logs; key by SRR
    log_by_srr = {}
    for log_path in sorted(LOG_DIR.glob("proc_*.log"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0)):
        proc = parse_proc_log(log_path)
        if proc["srr"]:
            log_by_srr[proc["srr"]] = (log_path, proc)

    # Build result rows: all samples + any log-only entries
    all_srrs = sorted(set(list(samples.keys()) + list(log_by_srr.keys())))
    results = []
    for srr in all_srrs:
        meta = samples.get(srr, {})
        log_path, proc = log_by_srr.get(srr, (None, {"srr": srr, "status": "not_started", "wall_sec": 0.0, "cells": 0, "error_hint": ""}))

        output_dir = VAL_BASE / srr / "output"
        star = parse_star_log(output_dir / "star_Log.final.out")

        results.append({
            "srr": srr,
            "protocol": meta.get("protocol_family", ""),
            "species": meta.get("species", ""),
            "quality_tier": meta.get("quality_tier", ""),
            "status": proc["status"],
            "wall_sec": proc["wall_sec"],
            "cells": proc["cells"],
            "mapping_pct": star.get("unique_map_pct", 0.0),
            "multi_map_pct": star.get("multi_map_pct", 0.0),
            "input_reads": star.get("input_reads", 0),
            "n_outputs": count_output_files(output_dir),
            "error_hint": proc.get("error_hint", ""),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(results)
    status_counts = defaultdict(int)
    protocol_counts = defaultdict(lambda: {"success": 0, "total": 0})
    species_counts = defaultdict(lambda: {"success": 0, "total": 0})

    for r in results:
        status_counts[r["status"]] += 1
        proto = r["protocol"] or "unknown"
        sp = r["species"] or "unknown"
        protocol_counts[proto]["total"] += 1
        species_counts[sp]["total"] += 1
        if r["status"] == "success":
            protocol_counts[proto]["success"] += 1
            species_counts[sp]["success"] += 1

    print("=" * 62)
    print("VAL1 QUALITY REPORT")
    print("=" * 62)
    print(f"\nTotal samples tracked: {n}")

    print(f"\nStatus breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {status:20s}: {count:4d}  ({100*count/n:.1f}%)")

    print(f"\nProtocol success rates:")
    for proto, c in sorted(protocol_counts.items(), key=lambda x: -x[1]["total"]):
        rate = 100 * c["success"] / c["total"] if c["total"] else 0
        print(f"  {proto:25s}: {c['success']:3d}/{c['total']:3d}  ({rate:.0f}%)")

    print(f"\nSpecies success rates:")
    for sp, c in sorted(species_counts.items(), key=lambda x: -x[1]["total"]):
        rate = 100 * c["success"] / c["total"] if c["total"] else 0
        print(f"  {sp:30s}: {c['success']:3d}/{c['total']:3d}  ({rate:.0f}%)")

    successful = [r for r in results if r["status"] == "success"]
    if successful:
        print(f"\nSuccessful sample metrics (n={len(successful)}):")
        walls = [r["wall_sec"] for r in successful if r["wall_sec"] > 0]
        cells = [r["cells"] for r in successful if r["cells"] > 0]
        maps  = [r["mapping_pct"] for r in successful if r["mapping_pct"] > 0]
        if walls:
            print(f"  Wall time:    median={median(walls):.1f}s  range={min(walls):.1f}-{max(walls):.1f}s")
        if cells:
            print(f"  Cell count:   median={median(cells):,.0f}  range={min(cells):,}-{max(cells):,}")
        if maps:
            print(f"  Mapping rate: median={median(maps):.1f}%  range={min(maps):.1f}-{max(maps):.1f}%")

    failures = [r for r in results if r["status"] not in ("success", "skipped", "not_started", "running")]
    if failures:
        print(f"\nFailed samples ({len(failures)}):")
        for r in failures:
            print(f"  {r['srr']:14s}  {r['status']:15s}  {r['error_hint'][:55]}")

    # ── Detailed CSV ─────────────────────────────────────────────────────────
    detail_path = Path(__file__).parent / "val1_report.csv"
    fields = ["srr", "protocol", "species", "quality_tier", "status",
              "wall_sec", "cells", "mapping_pct", "multi_map_pct",
              "input_reads", "n_outputs", "error_hint"]
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nDetailed report written: {detail_path}")


if __name__ == "__main__":
    main()
