#!/usr/bin/env python3
"""Catalog curation engine.

Phase 1: Regex-based pre-classification from existing catalog metadata.
Phase 2: GEO SOFT metadata fetching for unresolved samples.
Phase 3: Integration point for LLM sub-agent classification results.

Usage:
    python curate_catalog.py regex-classify          # Regex pre-classification
    python curate_catalog.py fetch-metadata --batch N # Fetch GEO SOFT for batch N
    python curate_catalog.py apply-curation FILE     # Apply curation CSV to catalog
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CATALOG_PATH = Path("/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet")
CURATION_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog/curation")
METADATA_DIR = CURATION_DIR / "geo_soft"

# ─── Protocol regex patterns ───
# Applied to concatenated text of: summary, overall_design, sample_characteristics,
# sample_source, notes, series_title
PROTOCOL_PATTERNS = [
    # Specific 10x versions (order matters — most specific first)
    ("10xv3", r"(?:10x|chromium).*(?:3'|v3|version\s*3)|(?:3'|v3).*(?:10x|chromium)|GEX\s+3'|next\s*gem\s+3'", "medium"),
    ("10xv2", r"(?:10x|chromium).*(?:v2|version\s*2)|(?:v2).*(?:10x|chromium)", "medium"),
    ("10xv3_5prime", r"(?:10x|chromium).*5'|5'.*(?:10x|chromium)|GEX\s+5'", "medium"),

    # Droplet-based
    ("dropseq", r"drop[\-\s]?seq", "high"),
    ("indrop", r"\bindrop\b|in[\-\s]drop", "high"),
    ("bd_rhapsody", r"bd\s*rhapsody|rhapsody\s*(?:single|express)", "high"),
    ("ddseq", r"\bddseq\b|dolomite", "high"),
    ("dnbelab", r"\bdnbelab\b|dnbelab[\-\s]c4", "high"),
    ("seqwell", r"\bseq[\-\s]?well\b", "high"),
    ("microwell", r"\bmicrowell[\-\s]?seq\b", "high"),

    # Plate-based (skip candidates)
    ("smartseq2", r"smart[\-\s]?seq[\-\s]?2\b|smartseq2", "high"),
    ("smartseq3", r"smart[\-\s]?seq[\-\s]?3\b|smartseq3", "high"),
    ("smartseq", r"smart[\-\s]?seq\b(?![\-\s]?[23])|smartseq\b(?![23])", "medium"),
    ("celseq2", r"cel[\-\s]?seq[\-\s]?2", "high"),
    ("celseq", r"cel[\-\s]?seq\b(?![\-\s]?2)", "high"),
    ("marsseq", r"mars[\-\s]?seq", "high"),
    ("strt_seq", r"\bstrt[\-\s]?seq\b|STRT[\-\s]V[234]", "high"),
    ("plate_based", r"plate[\-\s]?based|384[\-\s]?well.*single|96[\-\s]?well.*single|facs[\-\s]?sort.*(?:single|individual)\s*cell", "medium"),

    # Combinatorial indexing
    ("scirna", r"\bsci[\-\s]?rna[\-\s]?seq\b|combinatorial\s+index(?:ing)?.*(?:rna|transcriptom)", "high"),
    ("parse", r"\bparse\s*biosciences\b|\bsplit[\-\s]?seq\b|\bevercode\b", "high"),

    # Non-RNA (skip candidates)
    ("scatac", r"\bsc[\-\s]?atac[\-\s]?seq\b|\bsn[\-\s]?atac[\-\s]?seq\b|single.{0,5}cell.{0,5}atac|chromatin\s+access.*single.{0,5}cell", "high"),
    ("spatial", r"\bvisium\b|\bslide[\-\s]?seq\b|\bmerfish\b|\bseqfish\b|spatial\s+transcriptom|\bstereo[\-\s]?seq\b", "high"),
    ("methylation", r"\bbs[\-\s]?seq\b|bisulfite|methyl.*single.{0,5}cell|\bsc[\-\s]?wgbs\b|methylome", "high"),
    ("chipseq", r"\bchip[\-\s]?seq\b|\bcut[\&\-\s]?run\b|\bcut[\&\-\s]?tag\b|sc[\-\s]?chip", "high"),
    ("vdj", r"\btcr[\-\s]?seq\b|\bbcr[\-\s]?seq\b|\bv\(?d\)?j\b|immune\s+repertoire|receptor\s+sequenc", "high"),

    # Multi-modal (process GEX component)
    ("citeseq", r"\bcite[\-\s]?seq\b|\btotalseq\b|antibody.derived.tag|\badt\b.*(?:protein|surface)", "high"),

    # Bulk (skip)
    ("bulk", r"\bbulk\s+rna[\-\s]?seq\b|population[\-\s]?level|pooled\s+(?:rna|sample)", "medium"),
]

# Protocols that should NOT be processed by our droplet pipeline
SKIP_PROTOCOLS = {
    "smartseq2": "skip_plate_based",
    "smartseq3": "skip_plate_based",
    "smartseq": "skip_likely_plate",
    "celseq": "skip_plate_based",
    "celseq2": "skip_plate_based",
    "marsseq": "skip_plate_based",
    "strt_seq": "skip_plate_based",
    "plate_based": "skip_likely_plate",
    "scatac": "skip_atac",
    "spatial": "skip_spatial",
    "methylation": "skip_non_rna",
    "chipseq": "skip_non_rna",
    "vdj": "skip_vdj",
    "bulk": "skip_bulk",
}

# Protocols processable by our pipeline
PROCESSABLE_PROTOCOLS = {
    "10xv3", "10xv2", "10xv3_5prime", "10xv4",
    "dropseq", "indrop", "bd_rhapsody", "ddseq",
    "dnbelab", "seqwell", "microwell",
    "scirna", "parse",
    "citeseq",  # Process GEX component
}


def _combine_text(row: pd.Series) -> str:
    """Combine all text fields for regex matching."""
    fields = ["summary", "overall_design", "sample_characteristics",
              "sample_source", "notes", "series_title", "gsm_title"]
    parts = []
    for f in fields:
        val = row.get(f, "")
        if pd.notna(val) and val:
            parts.append(str(val))
    return " ".join(parts)


def regex_classify(cat: pd.DataFrame) -> pd.DataFrame:
    """Apply regex patterns to catalog text fields.

    Returns DataFrame with columns:
        gsm_id, gse_id, old_protocol, old_confidence, old_status,
        new_protocol, new_confidence, new_status, match_pattern, skip_reason
    """
    # Target: pending samples with low/medium confidence or unknown/suspect protocol
    needs_curation = cat[
        (cat["processing_status"] == "pending") &
        (
            (cat["protocol_confidence"].isin(["low", "medium"])) |
            (cat["protocol_inferred"].isin(["unknown", "unknown_sc", "10x_suspect"]))
        )
    ].copy()

    print(f"Samples needing curation: {len(needs_curation):,}")

    results = []
    matched_count = 0

    for idx, row in needs_curation.iterrows():
        text = _combine_text(row)
        best_match = None

        for proto_name, pattern, confidence in PROTOCOL_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                best_match = (proto_name, confidence, pattern)
                break  # First match wins (patterns ordered by priority)

        if best_match:
            matched_count += 1
            proto, conf, pat = best_match
            skip = SKIP_PROTOCOLS.get(proto, "")
            new_status = skip if skip else "pending"

            results.append({
                "gsm_id": row["gsm_id"],
                "gse_id": row["gse_id"],
                "old_protocol": row["protocol_inferred"],
                "old_confidence": row["protocol_confidence"],
                "old_status": row["processing_status"],
                "new_protocol": proto,
                "new_confidence": conf,
                "new_status": new_status,
                "match_pattern": pat[:60],
                "skip_reason": skip,
            })

    result_df = pd.DataFrame(results)
    print(f"Regex-classified: {matched_count:,} / {len(needs_curation):,} ({matched_count/len(needs_curation)*100:.1f}%)")

    if len(result_df) > 0:
        print("\nClassification breakdown:")
        for (proto, conf), count in result_df.groupby(["new_protocol", "new_confidence"]).size().sort_values(ascending=False).items():
            skip = SKIP_PROTOCOLS.get(proto, "processable")
            print(f"  {proto}/{conf}: {count:,} → {skip}")

    return result_df


def fetch_geo_soft(gsm_id: str) -> dict:
    """Fetch and parse GEO SOFT format for a single GSM.

    Returns dict of parsed fields.
    """
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}&targ=self&form=text&view=full"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scgeo-catalog-curation/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except Exception as e:
        return {"error": str(e), "gsm_id": gsm_id}

    # Parse SOFT fields
    parsed = {"gsm_id": gsm_id, "raw_text": text}
    multi_fields = defaultdict(list)

    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("!Sample_"):
            key, _, val = line.partition(" = ")
            key = key[len("!Sample_"):]
            multi_fields[key].append(val)

    # Flatten single-value fields, keep lists for multi-value
    for key, vals in multi_fields.items():
        if len(vals) == 1:
            parsed[key] = vals[0]
        else:
            parsed[key] = " ;; ".join(vals)

    return parsed


def batch_fetch_metadata(gsm_ids: list[str], output_dir: Path,
                         rate_limit: float = 0.35) -> list[dict]:
    """Fetch GEO SOFT for a batch of GSMs with rate limiting.

    Args:
        gsm_ids: List of GSM IDs to fetch
        output_dir: Directory to store individual JSON files
        rate_limit: Seconds between requests (0.35 ≈ 3 req/sec)

    Returns list of parsed metadata dicts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    n_fetched = 0
    n_cached = 0
    n_errors = 0

    for i, gsm_id in enumerate(gsm_ids):
        # Check cache
        cache_path = output_dir / f"{gsm_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    data = json.load(f)
                if "error" not in data:
                    results.append(data)
                    n_cached += 1
                    continue
            except json.JSONDecodeError:
                pass

        # Fetch
        if i > 0 and n_fetched > 0:
            time.sleep(rate_limit)

        data = fetch_geo_soft(gsm_id)

        # Save to cache (without raw_text to save space)
        cache_data = {k: v for k, v in data.items() if k != "raw_text"}
        try:
            with open(cache_path, "w") as f:
                json.dump(cache_data, f, indent=1)
        except OSError:
            pass

        if "error" in data:
            n_errors += 1
            print(f"  Error fetching {gsm_id}: {data['error']}")
        else:
            results.append(data)
            n_fetched += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(gsm_ids)} (fetched={n_fetched}, cached={n_cached}, errors={n_errors})")

    print(f"Batch complete: {n_fetched} fetched, {n_cached} cached, {n_errors} errors")
    return results


def format_for_llm(gsm_metadata: dict, catalog_row: dict) -> str:
    """Format combined catalog + GEO SOFT metadata for LLM classification.

    Returns a compact text block suitable for sub-agent processing.
    """
    parts = [f"GSM: {gsm_metadata.get('gsm_id', catalog_row.get('gsm_id', '?'))}"]
    parts.append(f"GSE: {catalog_row.get('gse_id', '?')}")

    # Catalog fields
    parts.append(f"Series: {catalog_row.get('series_title', '')}")
    parts.append(f"Organism: {catalog_row.get('organism', '')}")
    parts.append(f"Source: {catalog_row.get('sample_source', '')}")
    parts.append(f"Catalog protocol: {catalog_row.get('protocol_inferred', '')}/{catalog_row.get('protocol_confidence', '')}")
    parts.append(f"Library: {catalog_row.get('library_strategy', '')}/{catalog_row.get('library_source', '')}/{catalog_row.get('library_selection', '')}")
    parts.append(f"Layout: {catalog_row.get('library_layout', '')}")
    parts.append(f"Reads: {catalog_row.get('read_count', '')}  AvgLen: {catalog_row.get('avg_read_length', '')}")

    # GEO SOFT fields (most informative)
    for key in ["title", "extract_protocol_ch1", "description",
                "data_processing", "characteristics_ch1",
                "molecule_ch1", "library_source", "library_strategy",
                "growth_protocol_ch1", "treatment_protocol_ch1",
                "instrument_model"]:
        val = gsm_metadata.get(key, "")
        if val:
            parts.append(f"GEO_{key}: {str(val)[:500]}")

    # Catalog text
    for key in ["summary", "overall_design", "sample_characteristics"]:
        val = catalog_row.get(key, "")
        if val and len(str(val)) > 5:
            parts.append(f"Catalog_{key}: {str(val)[:300]}")

    return "\n".join(parts)


def generate_llm_batch(cat: pd.DataFrame, gsm_ids: list[str],
                       metadata_dir: Path) -> str:
    """Generate a formatted batch of GSMs for LLM classification.

    Returns the complete prompt text to send to a sub-agent.
    """
    cat_indexed = cat.set_index("gsm_id")

    samples = []
    for gsm_id in gsm_ids:
        # Load GEO SOFT if available
        soft_path = metadata_dir / f"{gsm_id}.json"
        soft_data = {}
        if soft_path.exists():
            try:
                with open(soft_path) as f:
                    soft_data = json.load(f)
            except json.JSONDecodeError:
                pass

        # Get catalog row
        if gsm_id in cat_indexed.index:
            cat_row = cat_indexed.loc[gsm_id].to_dict() if isinstance(cat_indexed.loc[gsm_id], pd.Series) else cat_indexed.loc[gsm_id].iloc[0].to_dict()
        else:
            cat_row = {"gsm_id": gsm_id}

        samples.append(format_for_llm(soft_data, cat_row))

    batch_text = "\n---\n".join(samples)
    return batch_text


def apply_curation(curation_file: Path, dry_run: bool = True):
    """Apply curation CSV to the catalog (vectorized for speed).

    Curation CSV columns: gsm_id, new_protocol, new_confidence, new_status,
                          curator_status, curator_notes
    """
    print("Loading catalog...")
    cat = pd.read_parquet(CATALOG_PATH)
    print(f"  Catalog rows: {len(cat):,}")

    # Ensure curator columns exist
    if "curator_status" not in cat.columns:
        cat["curator_status"] = ""
    if "curator_notes" not in cat.columns:
        cat["curator_notes"] = ""

    print(f"Loading curation file: {curation_file}")
    curation = pd.read_csv(curation_file)
    print(f"  Curation rows: {len(curation):,}")

    # Vectorized merge approach — much faster than row-by-row
    # Accept both old and new column names
    avail_cols = [c for c in ["new_protocol", "new_confidence", "new_status",
                               "curator_status", "curator_notes"] if c in curation.columns]
    cur = curation.set_index("gsm_id")[avail_cols].copy()
    cur = cur[cur.index.isin(cat["gsm_id"])]
    print(f"  Matched GSMs in catalog: {len(cur):,}")

    # Create index mapping
    cat_idx = cat.set_index("gsm_id")

    changes = 0
    col_map = [
        ("protocol_inferred", "new_protocol"),
        ("protocol_confidence", "new_confidence"),
        ("processing_status", "new_status"),
        ("curator_status", "curator_status"),
        ("curator_notes", "curator_notes"),
    ]
    for col, new_col in col_map:
        if new_col not in cur.columns:
            continue
        valid = cur[new_col].dropna()
        valid = valid[valid != ""]
        if len(valid) > 0:
            old_vals = cat_idx.loc[valid.index, col]
            changed = old_vals != valid
            n_changed = changed.sum()
            cat_idx.loc[valid.index, col] = valid
            print(f"  {col}: {n_changed:,} changes")
            changes += n_changed

    cat = cat_idx.reset_index()
    print(f"Total changes: {changes:,}")

    if dry_run:
        print("DRY RUN — no changes written. Use --apply to write.")
    else:
        # Backup first
        backup_path = CATALOG_PATH.with_suffix(".pre_curation_backup.parquet")
        import shutil
        shutil.copy2(CATALOG_PATH, backup_path)
        print(f"  Backup saved to {backup_path}")

        cat.to_parquet(CATALOG_PATH, index=False)
        print(f"  Catalog updated: {CATALOG_PATH}")

    return changes


def main():
    parser = argparse.ArgumentParser(description="Catalog curation engine")
    sub = parser.add_subparsers(dest="command")

    # regex-classify
    p_regex = sub.add_parser("regex-classify", help="Regex pre-classification")
    p_regex.add_argument("--output", type=Path,
                         default=CURATION_DIR / "regex_curation.csv")

    # fetch-metadata
    p_fetch = sub.add_parser("fetch-metadata", help="Fetch GEO SOFT metadata")
    p_fetch.add_argument("--gsm-file", type=Path, help="File with GSM IDs (one per line)")
    p_fetch.add_argument("--limit", type=int, default=100, help="Max GSMs to fetch")
    p_fetch.add_argument("--offset", type=int, default=0, help="Start offset")

    # apply-curation
    p_apply = sub.add_parser("apply-curation", help="Apply curation CSV to catalog")
    p_apply.add_argument("file", type=Path, help="Curation CSV file")
    p_apply.add_argument("--apply", action="store_true", help="Actually write changes")

    # generate-llm-batch
    p_llm = sub.add_parser("generate-llm-batch", help="Generate LLM batch file")
    p_llm.add_argument("--gsm-file", type=Path, help="File with GSM IDs")
    p_llm.add_argument("--output", type=Path, help="Output text file")
    p_llm.add_argument("--limit", type=int, default=50)

    args = parser.parse_args()

    if args.command == "regex-classify":
        CURATION_DIR.mkdir(parents=True, exist_ok=True)
        cat = pd.read_parquet(CATALOG_PATH)
        result = regex_classify(cat)
        result.to_csv(args.output, index=False)
        print(f"\nWrote {len(result):,} classifications to {args.output}")

    elif args.command == "fetch-metadata":
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        if args.gsm_file:
            with open(args.gsm_file) as f:
                gsm_ids = [line.strip() for line in f if line.strip().startswith("GSM")]
        else:
            # Default: unresolved pending samples
            cat = pd.read_parquet(CATALOG_PATH)
            pending = cat[
                (cat["processing_status"] == "pending") &
                (cat["protocol_confidence"] == "low") &
                (cat["protocol_inferred"].isin(["unknown", "unknown_sc", "10x_suspect"]))
            ]
            gsm_ids = pending["gsm_id"].tolist()

        gsm_ids = gsm_ids[args.offset:args.offset + args.limit]
        print(f"Fetching GEO SOFT for {len(gsm_ids)} GSMs...")
        batch_fetch_metadata(gsm_ids, METADATA_DIR)

    elif args.command == "apply-curation":
        apply_curation(args.file, dry_run=not args.apply)

    elif args.command == "generate-llm-batch":
        cat = pd.read_parquet(CATALOG_PATH)
        if args.gsm_file:
            with open(args.gsm_file) as f:
                gsm_ids = [line.strip() for line in f if line.strip().startswith("GSM")]
        else:
            gsm_ids = []
        gsm_ids = gsm_ids[:args.limit]
        batch_text = generate_llm_batch(cat, gsm_ids, METADATA_DIR)
        if args.output:
            with open(args.output, "w") as f:
                f.write(batch_text)
            print(f"Wrote batch to {args.output}")
        else:
            print(batch_text)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
