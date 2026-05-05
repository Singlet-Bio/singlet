#!/usr/bin/env python3
"""Collect pipeline metrics and push to singletai-website repo for the investor dashboard.

Runs hourly via cron on the HPC login node. Collects:
  - Corpus stats: GSE/GSM counts, completed quantifications, cell counts
  - Compute stats: SLURM job status, active nodes
  - Appends to a time-series history for processing-rate charts

Usage:
    python collect_dashboard_metrics.py [--dry-run]

Cron example (every hour at :05):
    5 * * * * cd /mnt/home/debruinz/Singlet-AI && python geo-reprocess/scripts/collect_dashboard_metrics.py
"""

import argparse
import json
import os
import random
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
PIPELINE_BASE = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")
CATALOG_PATH = Path("/mnt/projects/debruinz_project/cellarium/catalog/stage3_soft_parsed.parquet")
WEBSITE_REPO = Path("/mnt/home/debruinz/Singlet-AI/singletai-website")
OUTPUT_JSON = WEBSITE_REPO / "public" / "data" / "pipeline-metrics.json"
MAX_HISTORY_ENTRIES = 720  # 30 days at hourly intervals

# ── SPZ header reading ────────────────────────────────────────────────────────
MAGIC_SPRZ = 0x5A525053  # "SPRZ"
MAGIC_SPZ2 = 0x53505A32  # "SPZ2" legacy

# Known gene counts for reference genomes (single-layer and USA 3-layer)
KNOWN_GENE_COUNTS = {
    38606, 57180, 32520, 115818, 171540, 97560,
    36601, 33538, 36604, 32285, 55487, 54532,
}


def read_spz_cells(filepath: Path) -> int:
    """Read SPZ/1PZ header and return cell count, or 0 on failure.

    Determines orientation by checking if either dimension matches
    a known gene count. Falls back to: larger dim = genes.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read(24)
        if len(data) < 24:
            return 0
        magic = struct.unpack_from("<I", data, 0)[0]
        if magic not in (MAGIC_SPRZ, MAGIC_SPZ2):
            return 0
        m = struct.unpack_from("<I", data, 8)[0]
        n = struct.unpack_from("<I", data, 12)[0]
        # Determine which dimension is cells
        if m in KNOWN_GENE_COUNTS and n not in KNOWN_GENE_COUNTS:
            return n  # rows=genes, cols=cells
        if n in KNOWN_GENE_COUNTS and m not in KNOWN_GENE_COUNTS:
            return m  # cols=genes, rows=cells
        # USA heuristic: divisible by 3 and >30000
        if m % 3 == 0 and m > 30000 and not (n % 3 == 0 and n > 30000):
            return n
        if n % 3 == 0 and n > 30000 and not (m % 3 == 0 and m > 30000):
            return m
        # Fallback: larger dimension is genes
        return min(m, n)
    except (OSError, struct.error):
        return 0
MANIFEST_SAMPLE_SIZE = 500  # Number of manifests to sample for cell count estimation


def count_dirs(base: Path, depth: int) -> int:
    """Count directories at a specific depth under base."""
    if depth == 1:
        return sum(1 for p in base.iterdir() if p.is_dir())
    elif depth == 2:
        count = 0
        for gse in base.iterdir():
            if gse.is_dir():
                count += sum(1 for p in gse.iterdir() if p.is_dir())
        return count
    return 0


def count_files(base: Path, filename: str) -> int:
    """Count files with a given name at GSE/GSM/filename depth."""
    count = 0
    for gse in base.iterdir():
        if not gse.is_dir():
            continue
        for gsm in gse.iterdir():
            if not gsm.is_dir():
                continue
            if (gsm / filename).exists():
                count += 1
    return count


def collect_manifest_stats(base: Path, sample_size: int = MANIFEST_SAMPLE_SIZE) -> dict:
    """Read ALL manifests and SPZ headers for accurate cell counts.

    Uses SPZ/1PZ file headers to get true matrix dimensions, then
    determines cell count via gene-count heuristics. Falls back to
    manifest n_cells only for newer pipeline runs (which have
    n_cells_raw, indicating proper filtering was applied).

    Full scan takes ~30-60s on the HPC filesystem.
    """
    total_manifests = 0
    total_cells = 0
    cells_count = 0
    species_cells = {}
    species_samples = {}
    qc_pass = 0
    qc_total = 0

    for gse in base.iterdir():
        if not gse.is_dir():
            continue
        for gsm in gse.iterdir():
            mf = gsm / "sample_manifest.json"
            if not mf.exists():
                continue
            total_manifests += 1
            try:
                with open(mf) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            # Only count cells from successfully processed samples
            status = data.get("status", "")

            # Find count file
            count_file = None
            for fname in ("counts.1pz", "counts.spz"):
                p = gsm / fname
                if p.exists():
                    count_file = p
                    break

            if count_file is None or status != "success":
                organism = data.get("organism", "Unknown")
                species_samples[organism] = species_samples.get(organism, 0) + 1
                qc = data.get("qc_status", "")
                if qc:
                    qc_total += 1
                    if qc == "qc_pass":
                        qc_pass += 1
                continue

            # Determine cell count: prefer SPZ header (ground truth)
            header_cells = read_spz_cells(count_file)
            manifest_nc = data.get("n_cells", 0) or 0
            has_ncr = data.get("n_cells_raw") is not None

            if has_ncr and manifest_nc > 0:
                # New pipeline: manifest n_cells is properly filtered
                n_cells = manifest_nc
            elif header_cells > 0:
                # Old pipeline: use SPZ header dimensions
                n_cells = header_cells
            else:
                n_cells = manifest_nc

            if n_cells > 0:
                total_cells += n_cells
                cells_count += 1

            organism = data.get("organism", "Unknown")
            if n_cells > 0:
                species_cells[organism] = species_cells.get(organism, 0) + n_cells
            species_samples[organism] = species_samples.get(organism, 0) + 1

            qc = data.get("qc_status", "")
            if qc:
                qc_total += 1
                if qc == "qc_pass":
                    qc_pass += 1

    species = {}
    for org in species_cells:
        species[org] = {
            "samples": species_samples.get(org, 0),
            "cells": species_cells[org],
        }

    avg_cells_per_sample = total_cells / cells_count if cells_count > 0 else 0
    qc_rate = qc_pass / qc_total if qc_total > 0 else 0

    return {
        "total_manifests": total_manifests,
        "total_cells": total_cells,
        "samples_with_cells": cells_count,
        "avg_cells_per_sample": round(avg_cells_per_sample),
        "species": species,
        "qc_pass_rate": round(qc_rate, 4),
    }


def collect_slurm_stats() -> dict:
    """Collect current SLURM job information."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", "debruinz"),
             "-o", "%.15i %.30j %.8T %.12M %.6D %R"],
            capture_output=True, text=True, timeout=10
        )
        lines = result.stdout.strip().split("\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return {"active_jobs": 0, "running_tasks": 0, "pending_tasks": 0,
                "nodes_active": 0, "job_arrays": []}

    if len(lines) <= 1:
        return {"active_jobs": 0, "running_tasks": 0, "pending_tasks": 0,
                "nodes_active": 0, "job_arrays": []}

    running = 0
    pending = 0
    nodes = set()
    arrays = {}

    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 4:
            continue

        job_id = parts[0]
        name = parts[1]
        state = parts[2]
        node_info = parts[-1] if len(parts) >= 6 else ""

        if state == "RUNNING":
            running += 1
            if node_info and not node_info.startswith("("):
                nodes.add(node_info)
        elif state == "PENDING":
            pending += 1

        # Track job arrays
        base_name = name
        if base_name not in arrays:
            arrays[base_name] = {"name": base_name, "running": 0, "pending": 0}
        if state == "RUNNING":
            arrays[base_name]["running"] += 1
        elif state == "PENDING":
            arrays[base_name]["pending"] += 1

    return {
        "active_jobs": running + pending,
        "running_tasks": running,
        "pending_tasks": pending,
        "nodes_active": len(nodes),
        "nodes_list": sorted(nodes),
        "job_arrays": list(arrays.values()),
    }


def collect_corpus_stats(base: Path) -> dict:
    """Collect high-level corpus statistics."""
    total_series = count_dirs(base, 1)
    total_samples = count_dirs(base, 2)
    # Count unique samples with any count file (avoid double-counting 1pz+spz)
    samples_quantified = 0
    samples_with_metadata = 0
    for gse in base.iterdir():
        if not gse.is_dir():
            continue
        for gsm in gse.iterdir():
            if not gsm.is_dir():
                continue
            if (gsm / "counts.1pz").exists() or (gsm / "counts.spz").exists():
                samples_quantified += 1
            if (gsm / "cell_metadata.parquet").exists():
                samples_with_metadata += 1

    return {
        "total_series": total_series,
        "total_samples": total_samples,
        "samples_quantified": samples_quantified,
        "samples_with_metadata": samples_with_metadata,
    }


def load_existing(path: Path) -> dict:
    """Load existing metrics JSON if present."""
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout, don't write or push")
    args = parser.parse_args()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Collect all metrics
    print(f"[{now}] Collecting corpus stats...", file=sys.stderr)
    corpus = collect_corpus_stats(PIPELINE_BASE)

    print(f"[{now}] Reading all manifests for exact cell counts...", file=sys.stderr)
    manifest_stats = collect_manifest_stats(PIPELINE_BASE)

    print(f"[{now}] Collecting SLURM stats...", file=sys.stderr)
    slurm = collect_slurm_stats()

    # Merge manifest stats into corpus
    corpus.update({
        "total_cells": manifest_stats["total_cells"],
        "samples_with_cells": manifest_stats["samples_with_cells"],
        "avg_cells_per_sample": manifest_stats["avg_cells_per_sample"],
        "species": manifest_stats["species"],
        "qc_pass_rate": manifest_stats["qc_pass_rate"],
        "pipeline_version": "v3_pipelined_kraken2",
    })

    # Build history entry
    history_entry = {
        "timestamp": now,
        "samples_quantified": corpus["samples_quantified"],
        "total_cells": corpus["total_cells"],
        "running_tasks": slurm["running_tasks"],
    }

    # Load existing data and append to history
    existing = load_existing(OUTPUT_JSON)
    history = existing.get("history", [])
    history.append(history_entry)
    # Trim history to max entries
    if len(history) > MAX_HISTORY_ENTRIES:
        history = history[-MAX_HISTORY_ENTRIES:]

    # Assemble final output
    metrics = {
        "last_updated": now,
        "corpus": corpus,
        "compute": slurm,
        "history": history,
    }

    if args.dry_run:
        print(json.dumps(metrics, indent=2))
        return

    # Write JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{now}] Wrote {OUTPUT_JSON}", file=sys.stderr)

    # Git commit and push
    try:
        subprocess.run(
            ["git", "add", str(OUTPUT_JSON.relative_to(WEBSITE_REPO))],
            cwd=WEBSITE_REPO, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "commit", "-m", f"chore: update pipeline metrics {now}",
             "--allow-empty"],
            cwd=WEBSITE_REPO, check=True, capture_output=True, timeout=30,
        )
        subprocess.run(
            ["git", "push", "-u", "origin", "main"],
            cwd=WEBSITE_REPO, check=True, capture_output=True, timeout=60,
        )
        print(f"[{now}] Pushed to git", file=sys.stderr)
    except subprocess.CalledProcessError as e:
        print(f"[{now}] Git push failed: {e.stderr}", file=sys.stderr)
    except subprocess.TimeoutExpired:
        print(f"[{now}] Git push timed out", file=sys.stderr)


if __name__ == "__main__":
    main()
