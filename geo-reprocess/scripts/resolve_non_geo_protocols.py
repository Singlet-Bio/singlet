#!/usr/bin/env python3
"""
Resolve protocols for non-GEO 10x_suspect and unknown_sc samples.

Uses HTTP Range requests to peek at FASTQ R1 read lengths without full download.
Classifies samples as 10xv2, 10xv3, dropseq, smartseq2, or unresolvable.

Updates the catalog in-place with resolved protocols.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

# Add scgeo to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scgeo.pipeline.detect import peek_fastq_read_length

CATALOG = "/mnt/projects/debruinz_project/cellarium/catalog/geo_single_cell_catalog.parquet"
OUTDIR = "/mnt/projects/debruinz_project/cellarium/catalog"

# Protocol classification thresholds
BARCODE_MAX = 50  # R1 <= this = barcode read (droplet)


def classify_from_lengths(r1_len, r2_len):
    """Classify protocol from R1/R2 read lengths.
    
    Returns (protocol, confidence, mode) tuple.
    """
    if r1_len is None:
        return "peek_failed", "none", "unknown"
    
    # Both reads long → plate-based
    if r1_len > 50 and (r2_len is None or r2_len > 50):
        return "smartseq2", "high", "smartseq"
    
    # Short R1 + long R2 → standard droplet
    if r1_len <= BARCODE_MAX and r2_len is not None and r2_len > 50:
        if r1_len == 28:
            return "10xv3", "high", "droplet"
        elif r1_len in (26, 27):
            return "10xv2", "high", "droplet"
        elif r1_len == 24:
            return "10xv2", "medium", "droplet"  # older 10x v2
        elif 20 <= r1_len <= 25:
            return "dropseq", "medium", "droplet"
        elif r1_len < 20:
            return "unknown_short", "low", "unknown"
        else:
            # 29-50bp range — could be various protocols
            return "10x_other", "low", "droplet"
    
    # Short R2 + long R1 → swapped reads (common with ENA submissions)
    if r2_len is not None and r2_len <= BARCODE_MAX and r1_len > 50:
        if r2_len == 28:
            return "10xv3_swapped", "high", "droplet"
        elif r2_len in (26, 27):
            return "10xv2_swapped", "high", "droplet"
        elif 20 <= r2_len <= 25:
            return "dropseq_swapped", "medium", "droplet"
        else:
            return "swapped_unknown", "low", "droplet"
    
    # Single-end
    if r2_len is None or r2_len == 0:
        if r1_len > 50:
            return "smartseq2", "medium", "smartseq"
        else:
            return "unknown_se", "low", "unknown"
    
    return "ambiguous", "low", "unknown"


def resolve_samples(df, max_samples=0, rate_limit=0.1):
    """Resolve protocols by peeking at FASTQ R1 read lengths.
    
    Args:
        df: DataFrame of samples to resolve
        max_samples: Max samples to process (0=all)
        rate_limit: Seconds between requests
    
    Returns:
        List of (index, protocol, confidence, mode, r1_len, r2_len) tuples
    """
    results = []
    total = len(df) if max_samples == 0 else min(max_samples, len(df))
    
    for i, (idx, row) in enumerate(df.iterrows()):
        if max_samples > 0 and i >= max_samples:
            break
        
        r1_url = row.get("ena_fastq_r1", "")
        r2_url = row.get("ena_fastq_r2", "")
        
        if not r1_url or not isinstance(r1_url, str) or len(r1_url) < 10:
            # No ENA URL — try constructing from SRR
            srr = row.get("srr_accessions", "")
            if srr and isinstance(srr, str):
                srr_first = srr.split(";")[0].strip()
                if srr_first.startswith("SRR") or srr_first.startswith("ERR"):
                    prefix = srr_first[:6]
                    suffix = srr_first[-1] if len(srr_first) % 3 != 0 else ""
                    pad = f"00{suffix}" if suffix else ""
                    r1_url = f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/{pad}/{srr_first}/{srr_first}_1.fastq.gz"
                    r2_url = f"ftp://ftp.sra.ebi.ac.uk/vol1/fastq/{prefix}/{pad}/{srr_first}/{srr_first}_2.fastq.gz"
        
        if not r1_url:
            results.append((idx, "no_url", "none", "unknown", None, None))
            continue
        
        # Peek at R1
        r1_len = peek_fastq_read_length(r1_url, timeout=15)
        
        # Peek at R2 if available
        r2_len = None
        if r2_url and isinstance(r2_url, str) and len(r2_url) > 10:
            r2_len = peek_fastq_read_length(r2_url, timeout=15)
        
        protocol, confidence, mode = classify_from_lengths(r1_len, r2_len)
        results.append((idx, protocol, confidence, mode, r1_len, r2_len))
        
        if (i + 1) % 50 == 0:
            resolved = sum(1 for r in results if r[1] not in ("peek_failed", "no_url"))
            print(f"  [{i+1}/{total}] Resolved: {resolved}, Failed: {i+1-resolved}")
        
        time.sleep(rate_limit)
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to resolve (0=all)")
    parser.add_argument("--target", choices=["10x_suspect", "unknown_sc", "unknown", "all"],
                        default="all", help="Which samples to resolve")
    parser.add_argument("--dry-run", action="store_true",
                        help="Don't update catalog")
    args = parser.parse_args()
    
    print("Loading catalog...")
    cat = pd.read_parquet(CATALOG)
    non_geo = cat[cat["notes"].str.startswith("non_geo", na=False)]
    
    # Select target samples
    if args.target == "10x_suspect":
        targets = non_geo[non_geo["protocol_inferred"] == "10x_suspect"]
    elif args.target == "unknown_sc":
        targets = non_geo[non_geo["protocol_inferred"] == "unknown_sc"]
    elif args.target == "unknown":
        targets = non_geo[non_geo["protocol_inferred"] == "unknown"]
    else:
        targets = non_geo[non_geo["protocol_inferred"].isin(
            ["10x_suspect", "unknown_sc", "unknown"]
        )]
    
    print(f"Target samples: {len(targets):,}")
    print(f"  10x_suspect: {(targets['protocol_inferred'] == '10x_suspect').sum():,}")
    print(f"  unknown_sc: {(targets['protocol_inferred'] == 'unknown_sc').sum():,}")
    print(f"  unknown: {(targets['protocol_inferred'] == 'unknown').sum():,}")
    
    # Resolve
    print(f"\nResolving protocols via FASTQ peek...")
    results = resolve_samples(targets, max_samples=args.max_samples)
    
    # Summarize
    protocol_counts = Counter(r[1] for r in results)
    mode_counts = Counter(r[3] for r in results)
    
    print(f"\n{'='*60}")
    print("PROTOCOL RESOLUTION RESULTS")
    print(f"{'='*60}")
    print(f"Resolved: {len(results):,}")
    print(f"\nBy detected protocol:")
    for proto, cnt in protocol_counts.most_common():
        print(f"  {proto}: {cnt:,}")
    print(f"\nBy mode:")
    for mode, cnt in mode_counts.most_common():
        print(f"  {mode}: {cnt:,}")
    
    # Droplet protocols we can process
    droplet_protocols = {"10xv2", "10xv3", "10xv2_swapped", "10xv3_swapped",
                         "dropseq", "dropseq_swapped", "10x_other"}
    newly_processable = [(idx, proto, conf, mode, r1, r2) 
                         for idx, proto, conf, mode, r1, r2 in results
                         if proto in droplet_protocols]
    print(f"\nNewly processable droplet samples: {len(newly_processable):,}")
    
    # Update catalog
    if not args.dry_run:
        print("\nUpdating catalog...")
        n_updated = 0
        n_swapped = 0
        n_cleared = 0

        for idx, proto, conf, mode, r1, r2 in results:
            # Skip unresolved
            if proto in ("no_url",):
                continue

            # Clear fabricated ENA URLs for peek_failed samples
            if proto == "peek_failed":
                old_r1 = cat.at[idx, "ena_fastq_r1"]
                if old_r1 and isinstance(old_r1, str) and len(old_r1) > 10:
                    cat.at[idx, "ena_fastq_r1"] = ""
                    cat.at[idx, "ena_fastq_r2"] = ""
                    n_cleared += 1
                continue

            # Handle swapped reads — swap URLs and normalize protocol
            if "_swapped" in proto:
                base_proto = proto.replace("_swapped", "")
                old_r1 = cat.at[idx, "ena_fastq_r1"]
                old_r2 = cat.at[idx, "ena_fastq_r2"]
                cat.at[idx, "ena_fastq_r1"] = old_r2
                cat.at[idx, "ena_fastq_r2"] = old_r1
                cat.at[idx, "protocol_inferred"] = base_proto
                cat.at[idx, "protocol_confidence"] = conf
                notes = str(cat.at[idx, "notes"] or "")
                if "reads_swapped" not in notes:
                    cat.at[idx, "notes"] = notes + ";reads_swapped"
                n_swapped += 1
                n_updated += 1
            else:
                # Update all resolved protocols (not just droplet)
                cat.at[idx, "protocol_inferred"] = proto
                cat.at[idx, "protocol_confidence"] = conf
                n_updated += 1

        cat.to_parquet(CATALOG, index=False)
        print(f"Catalog updated: {n_updated} protocols resolved, {n_swapped} URLs swapped, {n_cleared} bad URLs cleared")
    elif args.dry_run:
        print("\nDRY RUN — catalog not updated")
    
    # Save detailed results
    result_rows = []
    for idx, proto, conf, mode, r1, r2 in results:
        row = targets.loc[idx]
        result_rows.append({
            "gsm_id": row["gsm_id"],
            "gse_id": row["gse_id"],
            "organism": row["organism"],
            "original_protocol": row["protocol_inferred"],
            "detected_protocol": proto,
            "confidence": conf,
            "mode": mode,
            "r1_len": r1,
            "r2_len": r2,
            "ena_fastq_r1": row.get("ena_fastq_r1", ""),
            "srr_accessions": row.get("srr_accessions", ""),
        })
    
    result_df = pd.DataFrame(result_rows)
    target_suffix = args.target if args.target != "all" else "all"
    result_path = Path(OUTDIR) / f"protocol_resolution_{target_suffix}.parquet"
    result_df.to_parquet(result_path, index=False)
    print(f"\nDetailed results: {result_path}")
    
    # Also write a CSV of newly-processable droplet samples in batch format
    if newly_processable:
        batch_rows = []
        for idx, proto, conf, mode, r1, r2 in newly_processable:
            row = cat.loc[idx] if not args.dry_run else targets.loc[idx]
            base_proto = proto.replace("_swapped", "")
            batch_rows.append({
                "gsm_id": row["gsm_id"],
                "gse_id": row["gse_id"],
                "organism": row["organism"],
                "protocol_inferred": base_proto,
                "ena_fastq_r1": row.get("ena_fastq_r1", ""),
                "ena_fastq_r2": row.get("ena_fastq_r2", ""),
                "srr_accessions": row.get("srr_accessions", ""),
                "read_count": row.get("read_count", 0),
            })
        
        batch_df = pd.DataFrame(batch_rows)
        batch_path = Path(OUTDIR) / "newly_resolved_droplet_batch.csv"
        batch_df.to_csv(batch_path, index=False)
        print(f"Newly-processable batch: {batch_path} ({len(batch_df)} samples)")


if __name__ == "__main__":
    main()
