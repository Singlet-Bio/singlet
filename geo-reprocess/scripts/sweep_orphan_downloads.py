#!/usr/bin/env python3
"""Sweep orphan FASTQ files from pipeline/downloads/.

Categorizes every download directory and deletes artifacts that are safe to remove:

  1. completed   — quant dir has counts.1pz → always safe to delete downloads
  2. manifested  — quant dir has sample_manifest.json (failed/skipped) but no .1pz
                   → safe to delete downloads (pipeline already recorded outcome)
  3. stale       — no manifest, no active SLURM job, download older than --max-age
                   → likely from a killed SLURM job, safe to delete after age check
  4. pending     — no manifest but a SLURM job might be running
                   → skipped unless --force-pending (dangerous!)

Usage:
    python sweep_orphan_downloads.py [--dry-run] [--gse GSE123456]
    python sweep_orphan_downloads.py --dry-run --max-age 12   # stale >12h
    python sweep_orphan_downloads.py                          # actually delete
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCGEO_BASE = Path(os.environ.get("SCGEO_BASE", "/mnt/projects/debruinz_project/cellarium"))
DOWNLOADS_DIR = SCGEO_BASE / "pipeline" / "downloads"
QUANT_DIR = SCGEO_BASE / "pipeline" / "quant"

FASTQ_PATTERNS = (
    "*.fastq.gz", "*.fq.gz",
    "*.fastq",    "*.fq",
    "*.sra",      "*.sra.cache",
    ".dl_segments_*",
)


def _get_active_slurm_gsms() -> set[str]:
    """Get set of GSM IDs currently claimed by active SLURM jobs.
    
    Reads the grab_batch ledger to find GSMs with active (non-abandoned,
    non-done, non-failed) claims.
    """
    ledger = SCGEO_BASE / "pipeline" / "claims" / "ledger.tsv"
    if not ledger.exists():
        return set()

    claims: dict[str, dict] = {}
    with open(ledger) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            cid, jid, gsms, ts, status = parts[:5]
            if status.startswith("update:"):
                _, target_cid, new_status = status.split(":", 2)
                if target_cid in claims:
                    claims[target_cid]["status"] = new_status
            else:
                claims[cid] = {
                    "gsms": [g for g in gsms.split(",") if g],
                    "status": status,
                }

    active_gsms = set()
    for claim in claims.values():
        if claim["status"] not in ("done", "failed", "abandoned"):
            active_gsms.update(claim["gsms"])
    return active_gsms


def _dir_size_and_age(d: Path) -> tuple[int, float]:
    """Return (total bytes, hours since newest modification) for a directory."""
    total = 0
    newest_mtime = 0.0
    for pattern in FASTQ_PATTERNS:
        for f in d.glob(pattern):
            try:
                st = f.stat()
                total += st.st_size
                if st.st_mtime > newest_mtime:
                    newest_mtime = st.st_mtime
            except OSError:
                pass
    age_hours = (time.time() - newest_mtime) / 3600 if newest_mtime > 0 else float("inf")
    return total, age_hours


def _delete_fastqs(gsm_dir: Path, dry_run: bool) -> tuple[int, int]:
    """Delete all FASTQ/SRA files from a GSM download directory.

    Returns (files_deleted, bytes_freed) — counts files even in dry-run.
    """
    freed = 0
    deleted = 0
    for pattern in FASTQ_PATTERNS:
        for f in sorted(gsm_dir.glob(pattern)):
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if dry_run:
                print(f"  [DRY] rm {f.name}  ({size / 1e9:.2f} GB)")
                freed += size
                deleted += 1
            else:
                try:
                    f.unlink()
                    freed += size
                    deleted += 1
                except Exception as e:
                    print(f"  ERROR deleting {f}: {e}", file=sys.stderr)
    return deleted, freed


def sweep(
    dry_run: bool = False,
    target_gse: str | None = None,
    max_age_hours: float = 6.0,
    force_pending: bool = False,
) -> None:
    """Sweep all orphan download artifacts."""
    active_gsms = _get_active_slurm_gsms()
    print(f"Active SLURM claims: {len(active_gsms)} GSMs")

    stats = {
        "completed": {"files": 0, "bytes": 0, "samples": 0},
        "manifested": {"files": 0, "bytes": 0, "samples": 0},
        "stale": {"files": 0, "bytes": 0, "samples": 0},
        "pending_skipped": {"files": 0, "bytes": 0, "samples": 0},
        "empty": {"dirs": 0},
    }

    gse_dirs = sorted(DOWNLOADS_DIR.iterdir()) if DOWNLOADS_DIR.exists() else []

    for gse_dir in gse_dirs:
        if not gse_dir.is_dir():
            continue
        gse_id = gse_dir.name
        if target_gse and gse_id != target_gse:
            continue

        for gsm_dir in sorted(gse_dir.iterdir()):
            if not gsm_dir.is_dir():
                continue
            gsm_id = gsm_dir.name

            # Check if dir is empty (harmless but messy)
            try:
                if not any(gsm_dir.iterdir()):
                    stats["empty"]["dirs"] += 1
                    if not dry_run:
                        gsm_dir.rmdir()
                    continue
            except OSError:
                continue

            quant_pz = QUANT_DIR / gse_id / gsm_id / "counts.1pz"
            manifest = QUANT_DIR / gse_id / gsm_id / "sample_manifest.json"

            # ── Category 1: Completed (has .1pz) ──
            if quant_pz.exists():
                category = "completed"
            # ── Category 2: Manifested (has manifest but no .1pz) ──
            elif manifest.exists():
                category = "manifested"
            # ── Category 3/4: No manifest — check if active or stale ──
            elif gsm_id in active_gsms and not force_pending:
                _sz, _age = _dir_size_and_age(gsm_dir)
                stats["pending_skipped"]["samples"] += 1
                stats["pending_skipped"]["bytes"] += _sz
                continue  # skip — SLURM job may be running
            else:
                _sz, age_h = _dir_size_and_age(gsm_dir)
                if age_h < max_age_hours and not force_pending:
                    stats["pending_skipped"]["samples"] += 1
                    stats["pending_skipped"]["bytes"] += _sz
                    continue  # too recent, might be in-progress
                category = "stale"

            # Delete
            prefix = f"[{category:>10}] {gsm_id}"
            _sz, _age = _dir_size_and_age(gsm_dir)
            print(f"{prefix}: {_sz / 1e9:.2f} GB, {_age:.1f}h old")
            nf, nb = _delete_fastqs(gsm_dir, dry_run)
            stats[category]["files"] += nf
            stats[category]["bytes"] += nb
            stats[category]["samples"] += 1

            # Remove sample dir if empty
            if not dry_run:
                try:
                    if not any(gsm_dir.iterdir()):
                        gsm_dir.rmdir()
                except Exception:
                    pass

        # Remove GSE dir if empty
        if not dry_run:
            try:
                if not any(gse_dir.iterdir()):
                    gse_dir.rmdir()
            except Exception:
                pass

    # Summary
    total_freed = sum(s.get("bytes", 0) for s in stats.values())
    total_files = sum(s.get("files", 0) for s in stats.values())
    total_samples = sum(s.get("samples", 0) for s in stats.values())

    print(f"\n{'═' * 60}")
    print(f"{'Category':<20} {'Samples':>8} {'Files':>8} {'Freed':>12}")
    print(f"{'─' * 60}")
    for cat in ("completed", "manifested", "stale", "pending_skipped"):
        s = stats[cat]
        freed_str = f"{s.get('bytes', 0) / 1e12:.3f} TB"
        print(f"{cat:<20} {s.get('samples', 0):>8} {s.get('files', 0):>8} {freed_str:>12}")
    print(f"{'─' * 60}")
    print(f"{'TOTAL':<20} {total_samples:>8} {total_files:>8} {total_freed / 1e12:.3f} TB")
    print(f"Empty dirs removed: {stats['empty']['dirs']}")
    if dry_run:
        print("\n⚠  DRY RUN — nothing was deleted. Re-run without --dry-run to delete.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep orphan FASTQ downloads")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be deleted without deleting")
    parser.add_argument("--gse", default=None,
                        help="Only sweep a specific GSE (for testing)")
    parser.add_argument("--max-age", type=float, default=6.0,
                        help="Hours since last modification before a pending download is considered stale (default: 6)")
    parser.add_argument("--force-pending", action="store_true",
                        help="Also delete pending downloads (dangerous — may affect running jobs)")
    args = parser.parse_args()
    sweep(
        dry_run=args.dry_run,
        target_gse=args.gse,
        max_age_hours=args.max_age,
        force_pending=args.force_pending,
    )
