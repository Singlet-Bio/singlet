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

# Supabase is only required for the actual sync. The parse-and-cache path
# (used when SUPABASE_SERVICE_KEY is unset) works without it.
try:
    from supabase import create_client, Client
    _HAS_SUPABASE = True
except ImportError:
    _HAS_SUPABASE = False
    Client = None  # type: ignore

# ─── Configuration ───────────────────────────────────────────────────────────

SINGLET_GPU_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = SINGLET_GPU_ROOT / "state"
FRONTIER_FILE = STATE_DIR / "pareto-frontier.md"
BENCHMARK_FILE = STATE_DIR / "benchmark-registry.md"

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")


def get_supabase_client():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Warning: SUPABASE_URL and SUPABASE_SERVICE_KEY not set. Skipping sync.", file=sys.stderr)
        return None
    if not _HAS_SUPABASE:
        print("Warning: supabase package not installed (pip install supabase). Skipping sync.", file=sys.stderr)
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)


# ─── Feature ID mapping ──────────────────────────────────────────────────────
# Short IDs in the `gpu_frontier.feature` column match the Benchmarks.tsx
# FEATURES table on singlet.bio (`pz_device_loader`, `lognorm`, `hvg`, ...).
# The long-form module path stays on the singlet-gpu side (cycle log, design
# docs, file paths). This dict is the single source of truth for mapping.

_LONG_TO_SHORT = {
    "io/pz_device_loader":            "pz_device_loader",
    "preprocess/lognorm":             "lognorm",
    "preprocess/deconv_size_factors": "lognorm",       # sub-variant of feature 2
    "preprocess/hvg":                 "hvg",
    "preprocess/scale":               "scale",
    "reduce/svd":                     "pca",
    "reduce/nmf":                     "nmf",
    "qc/metrics":                     "qc",
    "graph/knn":                      "knn",
    "graph/leiden":                   "leiden",
    "embed/umap":                     "umap",
    "de/wilcoxon":                    "de",
    "de/ttest":                       "de",
    "integrate/harmony":              "integration",
    "integrate/bbknn":                "integration",
}


def _short_feature_id(long_name: str) -> str:
    """Map the singlet-gpu long-form module path to the short ID the website expects.

    Variant suffixes like `preprocess/hvg::seurat_v3` are stripped to their base
    long form first, then mapped. Unknown long forms fall back to the original
    (so a new feature shows up in the DB instead of silently disappearing).
    """
    base = long_name.split("::", 1)[0]
    return _LONG_TO_SHORT.get(base, long_name)


_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.?\d*")


def _leading_num(cell: str):
    """Extract the first non-negative numeric value from a cell, or None for TBD/OOM/N/A/empty.

    Handles comma thousands separators (`383,134`). Negative values (e.g. `-1.0`) are
    treated as TBD-markers rather than real measurements — wall and memory cannot be negative.
    """
    if not cell:
        return None
    s = cell.strip()
    if s in ("TBD", "OOM", "N/A", "—", "-", ""):
        return None
    # Cells like "N/A (20.8k cells; ref N/A)" or "TBD — pending install" — the
    # marker is at the front, the rest is a comment, not a measurement.
    leading = s.split()[0].rstrip(",;:")
    if leading.upper() in ("TBD", "OOM", "N/A", "NA", "N/A.", "—", "-"):
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        v = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    if v < 0:
        return None
    return v


def parse_frontier() -> list[dict]:
    """Parse pareto-frontier.md into structured records matching gpu_frontier table schema.

    Each "###" section is one feature. Rows are split by "|" cells. We accept
    9-cell rows (scale, wall, mem, acc, sota_wall, sota_mem, sota_acc, sota_lib, dominates)
    and 10-cell rows (extra leading 'variant' column for HVG-style tables).
    """
    if not FRONTIER_FILE.exists():
        print(f"Warning: {FRONTIER_FILE} not found", file=sys.stderr)
        return []

    content = FRONTIER_FILE.read_text()
    entries = []

    # ### preprocess/hvg (feature #3) — promoted 2026-04-15, commit no-git
    feature_pattern = re.compile(
        r"###\s+(\S+?)\s+\(feature\s+#(\d+)(?:\s+sub-variant)?(?:\s+—\s+\*\*full frontier\*\*)?\)?(?:\s+—)?(?:\s+\S+)?\s*promoted\s+([\d-]+),\s+commit\s+(\S+)\)?"
    )

    sections = content.split("\n### ")
    for section in sections:
        header_match = feature_pattern.match("### " + section.lstrip("# "))
        if not header_match:
            continue

        feature_name = header_match.group(1).strip()
        if feature_name == "{feature}":  # Skip the schema example
            continue

        promoted_date = header_match.group(3)
        commit = header_match.group(4).rstrip(")")

        # Walk every line that looks like a data row of the frontier table.
        # Skip the header row and the alignment row.
        for line in section.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            # Header / separator detection
            if "---" in line:
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 9:
                continue
            # Header row: first cell is the literal "scale" or "variant"
            if cells[0].lower() in ("scale", "variant"):
                continue

            # Determine cell layout: 9 cols (scale-first) or 10 cols (variant-first).
            if len(cells) >= 10:
                variant = cells[0]
                scale = cells[1]
                offset = 2
            else:
                variant = None
                scale = cells[0]
                offset = 1

            our_wall = _leading_num(cells[offset])
            our_mem = _leading_num(cells[offset + 1])
            our_acc = cells[offset + 2]
            sota_wall = _leading_num(cells[offset + 3])
            sota_mem = _leading_num(cells[offset + 4])
            sota_acc = cells[offset + 5] if offset + 5 < len(cells) else ""
            sota_lib = cells[offset + 6] if offset + 6 < len(cells) else ""
            dominates_on = cells[offset + 7] if offset + 7 < len(cells) else ""

            # Skip rows that are entirely empty / TBD on both sides
            if our_wall is None and sota_wall is None:
                continue

            speedup = None
            if our_wall and sota_wall and our_wall > 0:
                speedup = round(sota_wall / our_wall, 1)

            correctness_r = None
            if our_acc and our_acc != "TBD":
                r_match = re.search(r"r\s*=\s*([\d.]+)|=\s*([\d.]+)|([01]\.\d{3,})", our_acc)
                if r_match:
                    val = next((g for g in r_match.groups() if g), None)
                    if val:
                        try:
                            correctness_r = float(val)
                        except ValueError:
                            pass

            feature_label = feature_name + (f"::{variant}" if variant else "")
            short_id = _short_feature_id(feature_label)
            # Encode the variant into the scale column so the page can show
            # multiple variants side-by-side (e.g. `seurat_v3 / small`,
            # `pearson_residuals / small` both under feature_id "hvg").
            scale_with_variant = f"{variant} / {scale}" if variant else scale

            entry = {
                "feature": short_id,
                "scale": scale_with_variant,
                "wall_ms": our_wall,
                "memory_mb": our_mem,
                "sota_wall_ms": sota_wall,
                "sota_tool": sota_lib if sota_lib and sota_lib != "TBD" else None,
                "speedup": speedup,
                "correctness_r": correctness_r,
                "correctness_ref": sota_lib if sota_lib and sota_lib != "TBD" else None,
                "commit_hash": commit if commit != "no-git" else None,
                "measured_date": promoted_date,
                "cycle_number": None,
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
