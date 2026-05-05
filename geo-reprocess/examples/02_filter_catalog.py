"""
Catalog filtering example.

This script demonstrates how to filter catalogs by various criteria.
"""
import pandas as pd
from scgeo import filter_catalog

# Load catalog
print("Loading catalog...")
catalog = pd.read_parquet("human_catalog.parquet")
print(f"Total samples: {len(catalog):,}")

# Filter to 10x data only
print("\nFiltering to 10x protocols...")
catalog_10x = filter_catalog(
    catalog,
    protocols=["10x_v2", "10x_v3"],
)
print(f"10x samples: {len(catalog_10x):,}")

# Filter by read count
print("\nFiltering by read count (>10M reads)...")
catalog_high_reads = filter_catalog(
    catalog_10x,
    min_reads=10_000_000,
)
print(f"High-read samples: {len(catalog_high_reads):,}")

# Filter by sample count per series
print("\nFiltering series with 10-100 samples...")
catalog_medium = filter_catalog(
    catalog_high_reads,
    min_samples=10,
    max_samples=100,
)
print(f"Medium-sized studies: {len(catalog_medium):,}")

# Save filtered catalog
output_file = "human_10x_filtered.csv"
catalog_medium.to_csv(output_file, index=False)
print(f"\nFiltered catalog saved to: {output_file}")
