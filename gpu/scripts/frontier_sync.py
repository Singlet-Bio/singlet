#!/usr/bin/env python3
"""
Sync singlet-gpu Pareto frontier to Supabase gpu_frontier table.

Parses state/pareto-frontier.md and upserts each frontier entry.
Also syncs benchmark-registry.md entries.

Required env vars:
    SUPABASE_URL         - https://vbswbitfyallghbgxkuw.supabase.co
    SUPABASE_SERVICE_KEY - Service role key (for writes)

Usage:
    python3 singlet-gpu/scripts/frontier_sync.py
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from supabase import create_client, Client
except ImportError:
    sys.exit("Install: pip install supabase")

# ─── Configuration ───────────────────────────────────────────────────────────

SINGLET_GPU_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = SINGLET_GPU_ROOT / "state"
FRONTIER_FILE = STATE_DIR / "pareto-frontier.md"
BENCHMARK_FILE = STATE_DIR / "benchmark-registry.md"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Warning: SUPABASE_URL and SUPABASE_SERVICE_KEY not set. Skipping sync.", file=sys.stderr)
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def parse_frontier() -> list[dict]:
    """Parse pareto-frontier.md into structured records matching gpu_frontier table schema."""
    if not FRONTIER_FILE.exists():
        print(f"Warning: {FRONTIER_FILE} not found", file=sys.stderr)
        return []

    content = FRONTIER_FILE.read_text()
    entries = []

    # Find each feature section
    feature_pattern = re.compile(
        r"###\s+(.+?)\s+\(feature\s+#(\d+)\)\s+—\s+promoted\s+([\d-]+),\s+commit\s+(\S+)"
    )
    # Find table rows within each section
    row_pattern = re.compile(
        r"\|\s*(\w+)\s*\|\s*([\d.]+|TBD|OOM)\s*\|\s*([\d.]+|TBD|OOM)\s*\|"
        r"\s*(.+?)\s*\|\s*([\d.]+|TBD|OOM)\s*\|\s*([\d.]+|TBD|OOM)\s*\|"
        r"\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|"
    )

    sections = content.split("###")[1:]  # Skip preamble
    for section in sections:
        header_match = feature_pattern.search("###" + section)
        if not header_match:
            continue

        feature_name = header_match.group(1).strip()
        feature_id = int(header_match.group(2))
        promoted_date = header_match.group(3)
        commit = header_match.group(4)

        for row_match in row_pattern.finditer(section):
            scale = row_match.group(1)
            our_wall = row_match.group(2)
            our_mem = row_match.group(3)
            our_accuracy = row_match.group(4).strip()
            sota_wall = row_match.group(5)
            sota_mem = row_match.group(6)
            sota_accuracy = row_match.group(7).strip()
            sota_lib = row_match.group(8).strip()
            dominates_on = row_match.group(9).strip()

            if our_wall == "TBD" and sota_wall == "TBD":
                continue  # Skip empty rows

            # Compute speedup ratio
            speedup = None
            if our_wall not in ("TBD", "OOM") and sota_wall not in ("TBD", "OOM"):
                try:
                    speedup = round(float(sota_wall) / float(our_wall), 1)
                except (ValueError, ZeroDivisionError):
                    pass

            # Parse correctness (look for numeric r value)
            correctness_r = None
            if our_accuracy and our_accuracy != "TBD":
                r_match = re.search(r"([\d.]+)", our_accuracy)
                if r_match:
                    try:
                        correctness_r = float(r_match.group(1))
                    except ValueError:
                        pass

            # Match Supabase gpu_frontier table schema exactly
            entry = {
                "feature": feature_name,
                "scale": scale,
                "wall_ms": float(our_wall) if our_wall not in ("TBD", "OOM") else None,
                "memory_mb": float(our_mem) if our_mem not in ("TBD", "OOM") else None,
                "sota_wall_ms": float(sota_wall) if sota_wall not in ("TBD", "OOM") else None,
                "sota_tool": sota_lib if sota_lib != "TBD" else None,
                "speedup": speedup,
                "correctness_r": correctness_r,
                "correctness_ref": sota_lib if sota_lib != "TBD" else None,
                "commit_hash": commit if commit != "no-git" else None,
                "measured_date": promoted_date,
                "cycle_number": None,  # Filled from cycle-log if available
            }
            entries.append(entry)

    return entries


def sync_to_supabase(client: Client, entries: list[dict]) -> int:
    """Sync frontier entries to gpu_frontier table.
    
    Uses delete-then-insert pattern since the table has no unique constraint
    on (feature, scale). Each sync is a full refresh of the frontier.
    """
    if not entries:
        print("No frontier entries to sync.")
        return 0

    # Delete all existing frontier entries and re-insert (full refresh)
    client.table("gpu_frontier").delete().neq("id", 0).execute()

    # Insert all current frontier entries
    result = client.table("gpu_frontier").insert(entries).execute()

    count = len(result.data) if result.data else 0
    print(f"Synced {count} frontier entries to Supabase (full refresh).")
    return count


def main():
    print(f"singlet-gpu frontier sync — {datetime.now(timezone.utc).isoformat()}")
    print(f"Frontier file: {FRONTIER_FILE}")

    client = get_supabase_client()
    if client is None:
        print("Supabase client unavailable. Writing local summary instead.")
        entries = parse_frontier()
        print(f"Parsed {len(entries)} frontier entries (not synced — no service key).")
        # Write local JSON for fallback
        local_out = STATE_DIR / "frontier_sync_cache.json"
        local_out.write_text(json.dumps(entries, indent=2, default=str))
        print(f"Cached to {local_out}")
        return

    entries = parse_frontier()
    if entries:
        synced = sync_to_supabase(client, entries)
        print(f"Done. {synced} entries in gpu_frontier table.")
    else:
        print("No entries parsed from frontier file.")


if __name__ == "__main__":
    main()
