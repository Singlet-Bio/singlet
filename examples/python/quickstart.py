#!/usr/bin/env python3
"""Quick-start example: browse the atlas and load data.

This script demonstrates core singlet functionality without GPU.
It runs in ~2 seconds with no internet connection (uses bundled catalog).
"""

import singlet

# 1. Browse the atlas catalog
print("=== Atlas Overview ===")
print(singlet.summary())
print()

# 2. Show available species
species = singlet.species()
print("=== Species ===")
for s in species:
    print(f"  • {s}")
print()

# 3. Search for datasets
results = singlet.catalog("lung")
print(f"=== Lung datasets: {len(results)} series ===")
print(results[["gse_id", "organism", "n_cells", "protocol"]].head(5).to_string())
print()

# 4. Get info about a specific series
gse_id = results["gse_id"].iloc[0]
info = singlet.info(gse_id)
print(f"=== {gse_id} Info ===")
for key, val in info.items():
    print(f"  {key}: {val}")
print()

# 5. Filter by organism and cell count
big_human = singlet.datasets(organism="Homo sapiens", min_cells=50000)
print(f"=== Large human datasets: {len(big_human)} series ===")
print(big_human[["gse_id", "n_cells", "tissue"]].head(5).to_string())
print()

# 6. Browse samples
samples = singlet.samples(tissue="brain")
print(f"=== Brain samples: {len(samples)} ===")
if len(samples) > 0:
    print(samples[["gsm_id", "gse_id", "cells_called"]].head(5).to_string())
print()

print("Done! See docs/api/python.md for the full API reference.")
