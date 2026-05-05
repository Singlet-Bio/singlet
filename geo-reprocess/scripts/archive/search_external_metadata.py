#!/usr/bin/env python3
"""Search CellxGene and papers for external metadata for our target GSEs."""

import json
import re
import sys
import requests
import pandas as pd
from pathlib import Path

CATALOG = Path("/mnt/projects/debruinz_project/cellarium/catalog/metadata_catalog.parquet")

cat = pd.read_parquet(str(CATALOG))

# Priority GSEs — Category 3 (had extractable files but 0 match) top20 + Cat 1/2 top15
TARGET_GSES = [
    # Category 3: extract-failed (large) — these have h5ad/tabular but 0 match
    "GSE263468",  # 185 GSMs, h5ad:244
    "GSE192740",  # 105 GSMs, rds:132, tabular:132
    "GSE292971",  # 101 GSMs, h5ad:107
    "GSE148829",  # 65 GSMs, tabular:3490
    "GSE81750",   # 60 GSMs, tabular:120
    "GSE157591",  # 52 GSMs, tabular:52
    "GSE297676",  # 50 GSMs, rds:114
    "GSE288199",  # 45 GSMs, tabular:180
    "GSE305266",  # 45 GSMs, tabular:90
    "GSE273804",  # 35 GSMs, h5ad:280
    "GSE240016",  # 29 GSMs, h5ad:406
    # Category 2: skip-only (large)
    "GSE214283",  # 113 GSMs
    "GSE266616",  # 79 GSMs
    "GSE208418",  # 76 GSMs
    "GSE206391",  # 60 GSMs
    "GSE186975",  # 57 GSMs
    "GSE142595",  # 54 GSMs
    "GSE157344",  # 51 GSMs
    "GSE117784",  # 49 GSMs
    "GSE150290",  # 49 GSMs
    # Category 1: no files
    "GSE304010",  # 68 GSMs
    "GSE139324",  # 51 GSMs
    "GSE275071",  # 32 GSMs
    "GSE211191",  # 31 GSMs
]

# Step 1: Check CellxGene API
print("=" * 70)
print("STEP 1: CellxGene Discovery API search")
print("=" * 70)

cxg_base = "https://api.cellxgene.cziscience.com/curation/v1"
cxg_hits = {}

for gse in TARGET_GSES:
    try:
        r = requests.get(
            cxg_base + "/datasets",
            params={"term": gse},
            timeout=30,
            headers={"Accept": "application/json"},
        )
        if r.status_code == 200:
            data = r.json()
            if data:
                cxg_hits[gse] = data
                print("%s: FOUND %d datasets on CellxGene" % (gse, len(data)))
                for d in data[:3]:
                    title = d.get("title", "?")[:80]
                    cid = d.get("collection_id", "?")
                    print("  title: %s" % title)
                    print("  collection: %s" % cid)
            # else: not found, skip silently
        elif r.status_code == 404:
            pass  # not found
        else:
            print("%s: CellxGene API status %d" % (gse, r.status_code))
    except Exception as e:
        print("%s: CellxGene error: %s" % (gse, str(e)[:100]))

print()
print("CellxGene hits: %d / %d GSEs" % (len(cxg_hits), len(TARGET_GSES)))

# Step 2: Check PubMed for papers with data availability
print()
print("=" * 70)
print("STEP 2: Check PubMed/paper data availability")
print("=" * 70)

desc_path = Path("/mnt/projects/debruinz_project/cellarium/catalog/all_gse_descriptions.parquet")
desc = pd.read_parquet(str(desc_path)) if desc_path.exists() else pd.DataFrame()

# For studies with PubMed IDs, fetch the paper page for data availability
DATASET_DIR = Path("/mnt/projects/debruinz_project/cellarium/dataset")

for gse in TARGET_GSES:
    n_gsms = len(cat[cat["gse_id"] == gse])
    
    # Get pubmed IDs from uns
    gse_dir = DATASET_DIR / gse
    pubmed_ids = []
    if gse_dir.exists():
        gsm_dirs = sorted([d for d in gse_dir.iterdir() if d.name.startswith("GSM")])
        if gsm_dirs:
            uns_path = gsm_dirs[0] / "author_metadata_uns.json"
            if uns_path.exists():
                with open(uns_path) as f:
                    uns = json.load(f)
                pubmed_ids = uns.get("pubmed_ids", [])
    
    if pubmed_ids:
        # Get paper link 
        for pmid in pubmed_ids[:2]:
            try:
                # Use NCBI efetch to get the paper abstract
                r = requests.get(
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                    params={"db": "pubmed", "id": pmid, "retmode": "xml"},
                    timeout=30,
                )
                if r.status_code == 200:
                    text = r.text
                    # Look for data availability keywords
                    ext_refs = []
                    for pattern in ["figshare", "zenodo", "dryad", "synapse", 
                                    "cellxgene", "github.com", "mendeley",
                                    "doi.org/10.", "supplementary table",
                                    "data availab", "code availab"]:
                        if pattern.lower() in text.lower():
                            # Extract surrounding context
                            idx = text.lower().find(pattern.lower())
                            ctx = text[max(0,idx-100):idx+200]
                            ext_refs.append(ctx.strip())
                    
                    if ext_refs:
                        print("%s (%d GSMs, PMID:%s): EXTERNAL DATA REFS" % (gse, n_gsms, pmid))
                        for ref in ext_refs[:3]:
                            # Clean XML
                            ref_clean = re.sub(r'<[^>]+>', ' ', ref).strip()[:150]
                            print("  %s" % ref_clean)
            except Exception as e:
                pass
            
            import time
            time.sleep(0.5)  # rate limit

print()
print("=" * 70)
print("SUMMARY — High-value targets for manual metadata retrieval")
print("=" * 70)

# Count GSMs for CellxGene hits
total_cxg_gsms = 0
for gse in cxg_hits:
    n = len(cat[cat["gse_id"] == gse])
    total_cxg_gsms += n
    print("  CellxGene: %s (%d GSMs)" % (gse, n))

print("  Total CellxGene-discoverable GSMs: %d" % total_cxg_gsms)
