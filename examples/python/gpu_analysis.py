#!/usr/bin/env python3
"""GPU analysis example: full scanpy-equivalent pipeline.

Requires: pip install singlet[gpu]
Hardware: NVIDIA GPU with CUDA 12+, 8+ GB VRAM

This script replicates the standard scanpy analysis workflow
(normalize → HVG → PCA → neighbors → Leiden → UMAP) entirely on GPU.
Typical speedup: 10-50x vs scanpy on CPU for 100k+ cells.
"""

import singlet
from singlet import gpu

# Load sample
adata = singlet.load("GSM4037629")
print(f"Loaded: {adata.n_obs} cells × {adata.n_vars} genes")

# Transfer to GPU
adata = gpu.io.to_device(adata)

# Preprocessing (runs on GPU)
gpu.pp.normalize_total(adata)
gpu.pp.log1p(adata)
gpu.pp.highly_variable_genes(adata, n_top_genes=2000)

# Dimensionality reduction
adata_hvg = adata[:, adata.var["highly_variable"]].copy()
gpu.reduce.pca(adata_hvg, n_comps=50)

# Graph construction + clustering
gpu.pp.neighbors(adata_hvg, n_neighbors=15)
gpu.tools.leiden(adata_hvg, resolution=1.0)
gpu.tools.umap(adata_hvg)

# Transfer back to CPU for visualization/saving
adata_result = gpu.io.to_host(adata_hvg)

print("\nResults:")
print(f"  Clusters: {adata_result.obs['leiden'].nunique()}")
print(f"  UMAP shape: {adata_result.obsm['X_umap'].shape}")
print(f"  PCA shape: {adata_result.obsm['X_pca'].shape}")
