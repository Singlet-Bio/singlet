#!/usr/bin/env python3
"""Refresh embedded catalog parquet files from Supabase.

Run this after ETL syncs to keep the singlet package's bundled data current.
Writes to singlet/data/sample_index.parquet and singlet/data/catalog_v1.parquet.

Usage:
    python scripts/refresh_catalog.py

Requires: SUPABASE_URL, SUPABASE_SERVICE_KEY environment variables.
"""
import os
import sys
import pandas as pd
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("pip install supabase-py")
    sys.exit(1)

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_KEY")
if not url or not key:
    print("Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
    sys.exit(1)

sb = create_client(url, key)

# Determine output directory
repo_root = Path(__file__).resolve().parent.parent
data_dir = repo_root / "singlet" / "data"
data_dir.mkdir(parents=True, exist_ok=True)

# Fetch all samples
print("Fetching samples from Supabase...")
all_rows = []
offset = 0
while True:
    r = sb.table("samples").select(
        "gsm_id,gse_id,organism,protocol,status,cells_called,mapping_rate,median_genes,title"
    ).range(offset, offset + 999).execute()
    if not r.data:
        break
    all_rows.extend(r.data)
    if len(r.data) < 1000:
        break
    offset += 1000

print(f"  Fetched {len(all_rows)} samples")

# Write sample_index
df = pd.DataFrame(all_rows)
df.to_parquet(data_dir / "sample_index.parquet", index=False)
print(f"  Wrote sample_index.parquet ({len(df)} rows)")

# Build catalog (series-level aggregation)
cat = df.groupby("gse_id").agg(
    n_samples=("gsm_id", "count"),
    organisms=("organism", lambda x: ",".join(sorted(set(str(v) for v in x if v)))),
    protocols=("protocol", lambda x: ",".join(sorted(set(str(v) for v in x if v)))),
    total_cells=("cells_called", "sum"),
    avg_mapping_rate=("mapping_rate", "mean"),
).reset_index()
cat.to_parquet(data_dir / "catalog_v1.parquet", index=False)
print(f"  Wrote catalog_v1.parquet ({len(cat)} series)")

# Also update the user cache
cache_dir = Path.home() / ".singlet" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
df.to_parquet(cache_dir / "sample_index.parquet", index=False)
cat.to_parquet(cache_dir / "catalog_v1.parquet", index=False)
print(f"  Updated ~/.singlet/cache/")

# Summary
success = df[df["status"] == "SUCCESS"]
species = set()
for org in cat["organisms"].dropna():
    for s in str(org).replace(";", ",").split(","):
        s = s.strip()
        if s and s != "unknown":
            species.add(s)

print(f"\nCatalog summary:")
print(f"  {len(df)} samples ({len(success)} SUCCESS)")
print(f"  {len(cat)} series")
print(f"  {len(species)} species")
print(f"  {success['cells_called'].sum():,.0f} total cells")
