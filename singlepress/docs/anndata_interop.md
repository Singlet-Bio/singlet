# AnnData & Ecosystem Interoperability

## Overview

`singlepress.interop` provides bidirectional conversion between .1pz and popular
single-cell data formats: AnnData, HDF5 (.h5ad), 10x CellRanger .h5, and standard
sparse/dense formats.

## AnnData

### Read .1pz → AnnData

```python
from singlepress.interop import to_anndata

adata = to_anndata("counts.1pz")
# adata.X       — sparse CSR matrix (cells × genes)
# adata.obs     — cell metadata DataFrame
# adata.var     — gene metadata DataFrame
# adata.uns     — study metadata dict
# adata.obs_names — cell barcodes
# adata.var_names — gene names
```

:::{note}
.1pz stores matrices as genes × cells (CSC). `to_anndata` automatically transposes
to the AnnData convention of cells × genes (CSR).
:::

### Write AnnData → .1pz

```python
from singlepress.interop import from_anndata

stats = from_anndata(adata, "counts.1pz",
                     store_transpose=True,  # enable gene-range reads
                     level=3)               # zstd compression level
```

All AnnData metadata is preserved:
- `adata.obs` → .1pz obs DataFrame
- `adata.var` → .1pz var DataFrame
- `adata.uns` → .1pz key-value pairs (string values only)
- `adata.obs_names` → .1pz colnames
- `adata.var_names` → .1pz rownames

## HDF5 / .h5ad

### Convert .1pz ↔ .h5ad

```python
from singlepress.interop import to_h5ad, from_h5ad

# .1pz → .h5ad
to_h5ad("counts.1pz", "counts.h5ad", compression="gzip")

# .h5ad → .1pz
from_h5ad("counts.h5ad", "counts.1pz")
```

### 10x CellRanger .h5

```python
from singlepress.interop import from_10x_h5

stats = from_10x_h5(
    "filtered_feature_bc_matrix.h5",
    "counts.1pz",
    store_transpose=True,
)
```

Reads the standard 10x HDF5 layout (`matrix/{barcodes, data, indices, indptr, features}`).

## Standard Sparse Formats

```python
from singlepress.interop import to_csr, to_coo, to_csc, to_dense
from singlepress.interop import from_csr, from_coo, from_dense

# Reading
csr = to_csr("counts.1pz")     # scipy CSR matrix
coo = to_coo("counts.1pz")     # scipy COO matrix
csc = to_csc("counts.1pz")     # scipy CSC matrix (native)
arr = to_dense("counts.1pz")   # numpy dense array

# Writing (any scipy sparse format → .1pz)
from_csr(csr_matrix, "counts.1pz", rownames=[...], colnames=[...])
from_coo(coo_matrix, "counts.1pz")
from_dense(numpy_array, "counts.1pz")
```

## DataFrame

```python
from singlepress.interop import to_dataframe, from_dataframe

# .1pz → pandas DataFrame (dense, uses rownames/colnames)
df = to_dataframe("counts.1pz")

# DataFrame → .1pz (uses index as rownames, columns as colnames)
from_dataframe(df, "counts.1pz")
```

:::{warning}
`to_dataframe()` and `to_dense()` materialize the full matrix in memory.
For large datasets (>100K cells), prefer sparse formats.
:::

## MatrixMarket (.mtx)

Read and write 10x-style MatrixMarket directories (the standard CellRanger output
before `.h5` became dominant).

```python
from singlepress.interop import from_mtx, to_mtx

# 10x MTX directory → .1pz
from_mtx("filtered_feature_bc_matrix/", "counts.1pz")

# .1pz → 10x MTX directory
to_mtx("counts.1pz", "mtx_output/")
# Creates: mtx_output/matrix.mtx.gz, features.tsv.gz, barcodes.tsv.gz
```

The `from_mtx()` function auto-detects:
- `matrix.mtx` or `matrix.mtx.gz`
- `features.tsv.gz`, `features.tsv`, `genes.tsv.gz`, or `genes.tsv`
- `barcodes.tsv.gz` or `barcodes.tsv`

## CSV / TSV

For small datasets or interchange with non-specialized tools:

```python
from singlepress.interop import from_csv, to_csv

# CSV → .1pz (genes as rows, cells as columns)
from_csv("expression.csv", "counts.1pz")

# TSV variant
from_csv("expression.tsv", "counts.1pz", sep="\t")

# .1pz → CSV
to_csv("counts.1pz", "expression.csv")

# .1pz → TSV
to_csv("counts.1pz", "expression.tsv", sep="\t")
```

Options for headerless or indexless files:

```python
from_csv("raw.csv", "counts.1pz",
         has_header=False,    # no column names
         has_index=False)     # no row names
```

:::{warning}
CSV/TSV conversion materializes the full dense matrix. Only suitable for
small datasets (< 10K cells).
:::

## Loom (.loom)

Read `.loom` files (an HDF5 variant used by some tools):

```python
from singlepress.interop import from_loom

from_loom("dataset.loom", "counts.1pz")

# Read a specific layer
from_loom("dataset.loom", "counts.1pz", layer="spliced")
```

Gene names are read from `ra/Gene` and barcodes from `ca/CellID` attributes.

## Pipeline Example

Convert a 10x dataset through the full ecosystem:

```python
from singlepress.interop import from_10x_h5, to_anndata, to_h5ad

# 10x .h5 → .1pz (13× smaller, fast random access)
from_10x_h5("filtered_feature_bc_matrix.h5", "counts.1pz")

# .1pz → AnnData for scanpy workflows
adata = to_anndata("counts.1pz")
import scanpy as sc
sc.pp.highly_variable_genes(adata)

# Back to .h5ad for sharing
to_h5ad("counts.1pz", "counts.h5ad")
```
