# Dataset Operations

## Overview

singlepress provides module-level functions for combining, subsetting, and sampling
`.1pz` files. These create new `.1pz` files on disk — they don't modify the originals.

## Horizontal Concatenation (cbind)

Concatenate cells (columns) from multiple `.1pz` files. All files must have the same
number of rows (features). Rownames are validated for consistency.

```python
import singlepress as sp

sp.cbind_1pz(
    "sample_A.1pz", "sample_B.1pz", "sample_C.1pz",
    output="combined.1pz",
    verify_rownames=True,   # raise if rownames differ
    num_threads=8,
)

pz = sp.open_1pz("combined.1pz")
print(pz.shape)  # features unchanged, cells summed
```

## Vertical Concatenation (rbind)

Concatenate features (rows) from multiple `.1pz` files. All files must have the same
number of columns (cells). Colnames are validated for consistency.

```python
sp.rbind_1pz(
    "genes_A.1pz", "genes_B.1pz",
    output="all_genes.1pz",
    verify_colnames=True,
    num_threads=8,
)
```

:::{note}
`rbind_1pz` requires full re-encoding since the chunk structure changes when
rows are concatenated. For large files, this can take a few seconds.
:::

## Subsetting

Write a filtered subset to a new `.1pz` file. Supports multiple selection methods
for both axes independently.

### By integer indices

```python
sp.subset_1pz(
    "counts.1pz", "subset.1pz",
    obs_indices=slice(0, 1000),       # first 1000 cells
    var_indices=[0, 5, 10, 20, 50],   # specific genes
)
```

### By names

Requires stored colnames/rownames in the file:

```python
sp.subset_1pz(
    "counts.1pz", "tcells.1pz",
    obs_names=["AACGTG-1", "TTGCAA-1", ...],  # specific barcodes
    var_names=["CD3D", "CD3E", "CD4"],         # specific genes
)
```

### By boolean mask

```python
import numpy as np

pz = sp.open_1pz("counts.1pz")
mask = pz.obs["cell_type"].values == "T-cell"

sp.subset_1pz(
    "counts.1pz", "tcells.1pz",
    obs_mask=mask,
)
```

### Combined axes

```python
sp.subset_1pz(
    "counts.1pz", "subset.1pz",
    obs_indices=slice(0, 500),
    var_names=["CD3D", "CD4", "CD8A"],
)
```

All metadata (obs, var, uns, rownames, colnames) is automatically subsetted to match.

## Random Sampling

Randomly downsample cells to a new `.1pz` file. Useful for quick exploration,
training set creation, or reducing dataset size.

### By count

```python
sp.sample_1pz(
    "counts.1pz", "sample_1k.1pz",
    n=1000,       # exactly 1000 cells
    seed=42,      # reproducible
)
```

### By fraction

```python
sp.sample_1pz(
    "counts.1pz", "sample_10pct.1pz",
    fraction=0.1,   # 10% of cells
    seed=42,
)
```

## Pipeline Example

Build a training set from multiple donors:

```python
import singlepress as sp

# Combine donors
sp.cbind_1pz(
    "donor_1.1pz", "donor_2.1pz", "donor_3.1pz",
    output="all_donors.1pz",
)

# Subset to genes of interest
marker_genes = ["CD3D", "CD3E", "CD4", "CD8A", "CD19", "MS4A1"]
sp.subset_1pz(
    "all_donors.1pz", "markers.1pz",
    var_names=marker_genes,
)

# Random training split
sp.sample_1pz("all_donors.1pz", "train.1pz", fraction=0.8, seed=42)
```
