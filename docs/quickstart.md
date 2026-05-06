# Quick Start

## Browse the Atlas

The bundled catalog works offline — no internet needed:

```python
import singlet

# One-line summary
print(singlet.summary())
# singlet atlas: 4,697 samples (2,072 SUCCESS) • 1,711 series • ...

# Search datasets
singlet.catalog("lung")

# Filter samples
singlet.samples(organism="Homo sapiens", tissue="brain", min_cells=1000)
```

## Load a Pre-Processed Sample

Every public GEO single-cell sample processed by the singlet pipeline is available:

```python
# Load by GEO accession (downloads from Zenodo)
adata = singlet.load("GSM5238385")
print(adata)
# AnnData object with n_obs × n_vars = 8234 × 33538
```

## GPU-Accelerated Analysis

```python
from singlet import gpu
from singlet.gpu import preprocess as sgpp

# Full pipeline on GPU (10-50× faster than Scanpy for 100k+ cells)
sgpp.normalize_total(adata)
sgpp.log1p(adata)
sgpp.highly_variable_genes(adata, n_top_genes=2000)
gpu.reduce.pca(adata, n_comps=50)
gpu.pp.neighbors(adata, n_neighbors=15)
gpu.tools.leiden(adata, resolution=1.0)
gpu.tools.umap(adata)
```
