# Loading Data

singlet provides multiple ways to load single-cell data into AnnData objects.

## From the Catalog

The simplest way is to use a GEO accession:

```python
import singlet

# Downloads from Zenodo (free) and caches locally
adata = singlet.load("GSE136831")
```

Data is cached in `~/.singlet/data/` so subsequent loads are instant.

## Source Selection

```python
# Zenodo (free, cached, slightly slower first download)
adata = singlet.load("GSE136831", backend="zenodo")

# AWS (fast streaming, costs tokens)
singlet.login("sk-...")
adata = singlet.load("GSE136831", backend="aws")
```

## Gene Subsetting

Load only the genes you need — no need to download the full matrix:

```python
adata = singlet.load("GSE136831", genes=["TP53", "BRCA1", "KRAS", "MYC"])
```

## Cell Filtering

```python
adata = singlet.load("GSE136831", obs_filter={"tissue": "lung"})
```

## Local Files

singlet auto-detects file format from the extension:

```python
# .spz (SinglePress compressed)
adata = singlet.load("data/experiment.spz")

# .h5ad (HDF5)
adata = singlet.load("data/experiment.h5ad")

# .zarr
adata = singlet.load("data/experiment.zarr")
```

## Chunked Reading

For very large files, read a subset of cells:

```python
from singlet._io import read_spz

# Read only cells 0–999
adata = read_spz("large_dataset.spz", col_range=(0, 1000))
```

## Download Without Loading

```python
# Just download, don't load into memory
path = singlet.download("GSE136831")
print(path)  # ~/.singlet/data/GSE136831.spz
```
