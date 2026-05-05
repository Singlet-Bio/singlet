# Quick Start

## Load a Pre-Processed Sample

Every public GEO single-cell sample is processed by the singlet pipeline and available instantly:

```python
import singlet

# Load by GEO accession
adata = singlet.load("GSM5238385")
print(adata)
# AnnData object with n_obs × n_vars = 8234 × 33538
```

## Browse Available Samples

```python
results = singlet.samples(organism="Homo sapiens", protocol="10xv3")
print(f"{len(results)} samples found")
```

## GPU-Accelerated Analysis

```python
import singlet.gpu as sg

# Full pipeline on GPU (100-500× faster than Scanpy)
sg.pp.normalize_total(adata)
sg.pp.log1p(adata)
sg.pp.highly_variable_genes(adata)
sg.pp.scale(adata)
sg.tl.pca(adata)
sg.tl.neighbors(adata)
sg.tl.leiden(adata)
sg.tl.umap(adata)
sg.tl.rank_genes_groups(adata)
```
