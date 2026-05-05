"""
Multi-species processing example.

This script demonstrates how to process data for multiple organisms.
"""
from scgeo import build_catalog, filter_catalog, build_indices, submit_batch
import pandas as pd

# Species to process
species = ["Homo sapiens", "Mus musculus", "Danio rerio"]

# Build indices for all species first (one-time setup)
print("Building indices for multiple species...")
print("This may take 1-2 hours total...\n")

index_map = {
    "Homo sapiens": "human",
    "Mus musculus": "mouse",
    "Danio rerio": "zebrafish",
}

for sp in species:
    common_name = index_map[sp]
    print(f"Building index for {sp} ({common_name})...")
    # build_indices([common_name])  # Uncomment to actually build

# Build comprehensive catalog
print("\nBuilding multi-species catalog...")
catalog = build_catalog(output_file="multi_species_catalog.parquet")

# Process each species separately
for sp in species:
    print(f"\n{'='*60}")
    print(f"Processing: {sp}")
    print('='*60)
    
    # Filter to this species
    sp_catalog = filter_catalog(catalog, organisms=[sp])
    
    if len(sp_catalog) < 10:
        print(f"  Skipping: only {len(sp_catalog)} samples")
        continue
    
    print(f"  Samples: {len(sp_catalog):,}")
    
    # Save species-specific catalog
    sp_file = f"{sp.replace(' ', '_')}_catalog.csv"
    sp_catalog.to_csv(sp_file, index=False)
    
    # Submit batch job
    job_name = f"{sp.split()[0].lower()}"
    
    job = submit_batch(
        catalog=sp_file,
        job_name=job_name,
        partition="cpu",
        samples_per_batch=50,
    )
    
    if job:
        print(f"  ✓ Job {job.job_id} submitted")
    else:
        print(f"  ✗ Job submission failed")

print("\n" + "="*60)
print("All jobs submitted!")
print("="*60)
print("\nMonitor jobs with:")
print("  sc-geo batch list")
