#!/usr/bin/env python3
from __future__ import annotations
"""Build catalog index files from merged per-GSE directories.

Creates:
  catalog/catalog_v1.parquet  — one row per GSE
  catalog/sample_index.parquet — one row per GSM with col_offset/col_count

Run after merge_gse.py has processed all GSEs.

Usage:
  python build_catalog_index.py
  python build_catalog_index.py --output-dir /path/to/catalog
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

QUANT_DIR = Path("/mnt/projects/debruinz_project/cellarium/pipeline/quant")
DEFAULT_OUTPUT = Path("/mnt/projects/debruinz_project/cellarium/catalog")

CATALOG_VERSION = "1.0"


def scan_gse(gse_dir: Path) -> tuple[dict | None, list[dict]]:
    """Extract catalog and sample index rows from a merged GSE directory.

    Returns (catalog_row, sample_rows) or (None, []) if not merged.
    """
    # Check for merged output at top-level or species subdirs
    targets = []
    if (gse_dir / "provenance.json").exists():
        targets.append(("", gse_dir))
    else:
        for sub in sorted(gse_dir.iterdir()):
            if sub.is_dir() and not sub.name.startswith("GSM") and (sub / "provenance.json").exists():
                targets.append((sub.name, sub))

    if not targets:
        return None, []

    gse_id = gse_dir.name
    all_organisms = []
    total_cells = 0
    total_genes = 0
    total_samples = 0
    protocols = set()
    references = set()
    has_kraken2 = False
    has_author_meta = False
    sample_rows = []

    for species_label, target_dir in targets:
        # Read provenance
        with open(target_dir / "provenance.json") as f:
            prov = json.load(f)

        # Read study metadata if available
        study_meta = {}
        study_meta_path = target_dir / "study_metadata.json"
        if not study_meta_path.exists():
            # Multi-species: study_metadata.json might be at GSE root
            study_meta_path = gse_dir / "study_metadata.json"
        if study_meta_path.exists():
            with open(study_meta_path) as f:
                study_meta = json.load(f)

        source_gsms = prov.get("source_gsms", {})
        merged_shape = prov.get("merged_shape", [0, 0])
        n_genes = merged_shape[0]
        n_cells = merged_shape[1]

        # Accumulate at GSE level
        total_cells += n_cells
        total_genes = max(total_genes, n_genes)
        total_samples += len(source_gsms)

        for gsm_id, gsm_info in source_gsms.items():
            organism = gsm_info.get("organism", "unknown")
            if organism not in all_organisms:
                all_organisms.append(organism)
            protocols.add(gsm_info.get("protocol", "unknown"))
            sample_rows.append({
                "gsm_id": gsm_id,
                "gse_id": gse_id,
                "organism": organism,
                "n_cells": gsm_info.get("n_cells", 0),
                "pipeline_version": gsm_info.get("pipeline_version", "unknown"),
                "species_subdir": species_label,
            })

        ref = study_meta.get("reference", "")
        if ref:
            references.add(ref)
        if (target_dir / "kraken2.1pz").exists():
            has_kraken2 = True
        # Check for author metadata columns in metadata.parquet
        meta_path = target_dir / "metadata.parquet"
        if meta_path.exists():
            try:
                import pyarrow.parquet as pq
                schema = pq.read_schema(str(meta_path))
                author_cols = {"cell_type", "tissue", "disease", "sex", "age"}
                if any(f.name in author_cols for f in schema):
                    has_author_meta = True
            except Exception:
                pass

    # Compute col_offset for each sample
    offset = 0
    for row in sample_rows:
        row["col_offset"] = offset
        row["col_count"] = row["n_cells"]
        offset += row["n_cells"]

    # Build catalog row
    catalog_row = {
        "gse_id": gse_id,
        "organism": all_organisms,
        "n_samples": total_samples,
        "n_cells": total_cells,
        "n_genes": total_genes,
        "reference": "|".join(sorted(references)) if references else "",
        "protocol": "|".join(sorted(protocols)) if protocols else "",
        "has_kraken2": has_kraken2,
        "has_author_meta": has_author_meta,
        "license": "public_domain",
        "path": str(gse_dir.relative_to(QUANT_DIR.parent.parent)),
        "catalog_version": CATALOG_VERSION,
    }

    return catalog_row, sample_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning GSE directories...", file=sys.stderr)
    catalog_rows = []
    sample_rows = []

    gse_dirs = sorted(
        d for d in QUANT_DIR.iterdir()
        if d.is_dir() and d.name.startswith("GSE")
    )

    for i, gse_dir in enumerate(gse_dirs):
        try:
            cat_row, samp_rows = scan_gse(gse_dir)
        except Exception as e:
            print(f"  ERROR {gse_dir.name}: {e}", file=sys.stderr)
            continue
        if cat_row is not None:
            catalog_rows.append(cat_row)
            sample_rows.extend(samp_rows)

        if (i + 1) % 500 == 0:
            print(f"  Scanned {i+1}/{len(gse_dirs)} GSEs "
                  f"({len(catalog_rows)} merged)", file=sys.stderr)

    print(f"\nTotal merged GSEs: {len(catalog_rows)}", file=sys.stderr)
    print(f"Total samples: {len(sample_rows)}", file=sys.stderr)
    total_cells = sum(r["n_cells"] for r in catalog_rows)
    print(f"Total cells: {total_cells:,}", file=sys.stderr)

    # Write catalog_v1.parquet
    cat_df = pd.DataFrame(catalog_rows)
    # Convert organism list to string for parquet compatibility
    cat_df["organism"] = cat_df["organism"].apply(lambda x: "|".join(x) if isinstance(x, list) else x)
    cat_path = args.output_dir / "catalog_v1.parquet"
    cat_df.to_parquet(str(cat_path), index=False)
    print(f"Wrote {cat_path} ({len(cat_df)} rows)", file=sys.stderr)

    # Write sample_index.parquet
    samp_df = pd.DataFrame(sample_rows)
    samp_path = args.output_dir / "sample_index.parquet"
    samp_df.to_parquet(str(samp_path), index=False)
    print(f"Wrote {samp_path} ({len(samp_df)} rows)", file=sys.stderr)


if __name__ == "__main__":
    main()
