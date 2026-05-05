#!/usr/bin/env python3
"""Quick-start example: browse the atlas and load a sample.

This script demonstrates core singlet functionality without GPU.
It runs in ~5 seconds on any machine with an internet connection.
"""

import singlet

# 1. Browse the atlas catalog
print("=== Atlas Overview ===")
print(singlet.summary())
print()

# 2. Show available species
species = singlet.species()
print("=== Species ===")
print(species.to_string())
print()

# 3. Search for datasets
results = singlet.catalog(search="lung")
print(f"=== Lung datasets: {len(results)} series ===")
print(results[["title", "organism", "n_samples"]].head(5).to_string())
print()

# 4. Get info about a specific series
info = singlet.info("GSE136831")
print("=== GSE136831 Info ===")
for key, val in list(info.items())[:6]:
    print(f"  {key}: {val}")
print()

# 5. Load a single sample (downloads ~5MB .1pz file)
print("=== Loading GSM4037629 ===")
adata = singlet.load("GSM4037629")
print(f"  Shape: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Obs columns: {list(adata.obs.columns[:5])}")
print(f"  Var columns: {list(adata.var.columns[:3])}")
print()
print("Done! See docs/api/python.md for the full API reference.")
