"""
Basic catalog building example.

This script demonstrates how to build a comprehensive GEO catalog
with metadata, SRA information, and protocol inference.
"""
from scgeo import build_catalog

# Build catalog for human single-cell data
print("Building catalog...")
catalog = build_catalog(
    query="single cell[Title] AND Homo sapiens[Organism]",
    output_file="human_catalog.parquet",
    include_soft=True,       # Include detailed metadata
    include_sra=True,        # Include SRA RunInfo
    include_protocols=True,  # Infer protocols
)

print(f"\nCatalog built:")
print(f"  Total samples: {len(catalog):,}")
print(f"  Total series: {catalog['gse_id'].nunique():,}")

# Show protocol distribution
print(f"\nProtocol distribution:")
print(catalog['protocol_inferred'].value_counts())

# Show organism distribution
print(f"\nOrganism distribution:")
print(catalog['organism'].value_counts())

print(f"\nOutput saved to: human_catalog.parquet")
