#!/usr/bin/env python3
"""Find large studies without cell-level metadata and categorize by failure mode."""

import pandas as pd
import json
import sys
from pathlib import Path

CATALOG = Path("/mnt/projects/debruinz_project/cellarium/catalog/metadata_catalog.parquet")

cat = pd.read_parquet(str(CATALOG))

# Group by GSE
gse_stats = cat.groupby('gse_id').agg(
    n_gsms=('gsm_id', 'count'),
    formats=('stage0_formats', 'first'),
    best_a_match=('stage2a_match_rate', 'max'),
    best_a_cols=('stage2a_n_cols', 'max'),
).reset_index()

no_files = []
skip_only = []
extract_failed = []

for _, row in gse_stats.iterrows():
    fmts = json.loads(row['formats']) if isinstance(row['formats'], str) and row['formats'] else {}
    n = int(row['n_gsms'])
    
    n_extract = sum(fmts.get(f, 0) for f in ('h5ad', 'tabular', 'loom', 'rds', 'tar', 'xlsx', 'h5seurat', 'hdf5', 'h5mu', 'rdata'))
    n_skip = fmts.get('skip', 0)
    
    best_match = float(row['best_a_match']) if row['best_a_match'] else 0
    best_cols = int(row['best_a_cols']) if row['best_a_cols'] else 0
    
    entry = (row['gse_id'], n, fmts, best_match, best_cols)
    
    if not fmts or sum(fmts.values()) == 0:
        no_files.append(entry)
    elif n_extract == 0 and n_skip > 0:
        skip_only.append(entry)
    elif n_extract > 0 and best_match == 0 and best_cols == 0:
        extract_failed.append(entry)

no_files.sort(key=lambda x: -x[1])
skip_only.sort(key=lambda x: -x[1])
extract_failed.sort(key=lambda x: -x[1])

print("=" * 70)
print("CATEGORY 1: No supplementary files at all (%d GSEs)" % len(no_files))
print("=" * 70)
for gse, n, fmts, mr, nc in no_files[:20]:
    print("  %12s  %4d GSMs" % (gse, n))

print()
print("=" * 70)
print("CATEGORY 2: Only skip-format files (%d GSEs)" % len(skip_only))
print("=" * 70)
for gse, n, fmts, mr, nc in skip_only[:30]:
    skip_n = fmts.get('skip', 0)
    print("  %12s  %4d GSMs  skip:%d" % (gse, n, skip_n))

print()
print("=" * 70)
print("CATEGORY 3: Had extractable files but 0 match on all GSMs (%d GSEs)" % len(extract_failed))
print("=" * 70)
for gse, n, fmts, mr, nc in extract_failed[:30]:
    fmt_str = ', '.join("%s:%d" % (k, v) for k, v in fmts.items() if v > 0 and k != 'skip')
    print("  %12s  %4d GSMs  [%s]" % (gse, n, fmt_str))

tot_no = sum(x[1] for x in no_files)
tot_skip = sum(x[1] for x in skip_only)
tot_fail = sum(x[1] for x in extract_failed)
print("\nTotal: %d no-file GSMs + %d skip-only GSMs + %d extract-failed GSMs" % (tot_no, tot_skip, tot_fail))
