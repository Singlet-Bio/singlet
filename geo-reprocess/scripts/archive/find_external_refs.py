#!/usr/bin/env python3
"""Fetch GSE descriptions and look for external data references."""

import json
import re
import sys
from pathlib import Path

# Top GSEs by size that lack metadata — prioritize largest
TARGET_GSES = [
    "GSE263468", "GSE214283", "GSE192740", "GSE292971", "GSE266616",
    "GSE208418", "GSE304010", "GSE148829", "GSE81750", "GSE206391",
    "GSE186975", "GSE142595", "GSE157591", "GSE157344", "GSE139324",
    "GSE273373", "GSE297676", "GSE117784", "GSE150290", "GSE204702",
    "GSE288856", "GSE303581", "GSE292909", "GSE303256", "GSE288199",
    "GSE291080", "GSE305266", "GSE180578", "GSE216329", "GSE313894",
]

CATALOG_DIR = Path("/mnt/projects/debruinz_project/cellarium/catalog")

# Load descriptions if available
desc_path = CATALOG_DIR / "all_gse_descriptions.parquet"
import pandas as pd
desc = pd.read_parquet(str(desc_path)) if desc_path.exists() else pd.DataFrame()

# Also check uns sidecars
DATASET_DIR = Path("/mnt/projects/debruinz_project/cellarium/dataset")

# External data patterns
EXT_PATTERNS = re.compile(
    r"(figshare|zenodo|dryad|synapse|cellxgene|broad\.io|single-cell\.ca"
    r"|github\.com|gitlab\.com|bitbucket\.org|data\.mendeley"
    r"|arrayexpress|EGA[DS]\d+|phs\d+"
    r"|hub\.?[a-z]*\.com/[a-z]"
    r"|\.s3\.amazonaws\.com"
    r"|gs://|s3://"
    r"|10\.5281/zenodo|10\.6084/m9\.figshare"
    r"|doi\.org/10\.\d+"
    r"|supplementary|companion\s+website"
    r"|data\s+availab|data\s+access|deposited\s+(at|in|to)"
    r"|download\S*\s+from"
    r"|available\s+(at|from|through|via)"
    r"|hosted\s+(at|on))",
    re.IGNORECASE,
)

for gse_id in TARGET_GSES:
    print(f"\n{'='*60}")
    print(f"GSE: {gse_id}")
    print(f"{'='*60}")

    # Check description
    if not desc.empty and gse_id in desc['gse_id'].values:
        row = desc[desc['gse_id'] == gse_id].iloc[0]
        title = str(row.get('title', ''))
        summary = str(row.get('summary', ''))
        print(f"  Title: {title[:120]}")
        print(f"  Summary: {summary[:300]}")
        if summary:
            matches = EXT_PATTERNS.findall(summary)
            if matches:
                print(f"  ** EXTERNAL DATA HINTS: {matches}")
    else:
        print(f"  No description in catalog — check NCBI directly")

    # Check uns sidecar for first GSM
    gse_dir = DATASET_DIR / gse_id
    if gse_dir.exists():
        gsm_dirs = sorted([d for d in gse_dir.iterdir() if d.name.startswith("GSM")])
        if gsm_dirs:
            uns_path = gsm_dirs[0] / "author_metadata_uns.json"
            if uns_path.exists():
                with open(uns_path) as f:
                    uns = json.load(f)
                summary2 = uns.get("summary", "")
                if summary2 and summary2 != summary:
                    print(f"  UNS Summary: {summary2[:300]}")
                    matches2 = EXT_PATTERNS.findall(summary2)
                    if matches2:
                        print(f"  ** UNS EXTERNAL DATA HINTS: {matches2}")
                pubmed = uns.get("pubmed_ids", [])
                if pubmed:
                    print(f"  PubMed IDs: {pubmed}")
