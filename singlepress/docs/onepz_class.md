# The OnePZFile Class

## Overview

`OnePZFile` is a lazy file handle for `.1pz` files. It reads header metadata on
construction but defers matrix decompression until you explicitly request data.
The API is designed to feel familiar to AnnData users while respecting that `.1pz`
is an on-disk compressed format — not a mutable in-memory object.

```python
import singlepress as sp

pz = sp.open_1pz("counts.1pz")
print(pz)
```

```
OnePZFile: counts.1pz
  Shape:   30,000 features x 10,000 cells
  NNZ:     15,000,000 (5.0% dense)
  Size:    45.2 MB
  Format:  .1pz v3
  Stored:  names, colsums, obs/var
  Rows:    ENSG00000187634, ENSG00000188976, ENSG00000187961, ... (30,000 total)
  Cols:    AAACCTGCATCGAGGG-1, AAACCTGCATGCTAGT-1, ... (10,000 total)
  obs:     10,000 cells x 3 fields (cell_type, tissue, donor)
  var:     30,000 features x 2 fields (gene_name, highly_variable)
```

## AnnData-Style Properties

OnePZFile provides aliases that match AnnData naming conventions:

```python
pz.n_obs        # 10000 — number of cells (columns)
pz.n_vars       # 30000 — number of features (rows)
pz.obs_names    # cell barcodes (alias for .colnames)
pz.var_names    # gene names (alias for .rownames)
pz.shape        # (30000, 10000) — (features, cells)
```

And native `.1pz` accessors:

```python
pz.rownames     # gene names (list[str] or None)
pz.colnames     # cell barcodes (list[str] or None)
pz.colsums      # numpy array of column sums (or None)
pz.obs          # pandas DataFrame of cell metadata (or None)
pz.var          # pandas DataFrame of gene metadata (or None)
pz.uns          # dict of study metadata (or None)
```

## Indexing with `[]`

OnePZFile supports NumPy-style fancy indexing. All indexing returns a materialized
`scipy.sparse.csc_matrix` — this is a file handle, not a lazy view.

### Integer slices (fast path)

Contiguous column ranges are chunk-aligned and only decompress the required chunks:

```python
sub = pz[0:100, 0:500]     # first 100 genes × first 500 cells
sub = pz[:, 0:1024]        # all genes × first chunk
```

### Gene name indexing

Requires stored rownames in the file:

```python
sub = pz[["CD3D", "CD3E", "CD4"], :]       # 3 genes × all cells
sub = pz["CD3D"]                             # single gene → 1×n sparse
```

### Boolean mask on cells

```python
mask = pz.obs["cell_type"] == "T-cell"
sub = pz[:, mask.values]                    # all genes × T-cells only
```

### Integer list indexing

```python
sub = pz[[0, 5, 10, 20], [0, 1, 2, 3]]    # specific rows and columns
```

### Combining axes

```python
# Specific genes × boolean cell mask
genes = ["CD3D", "CD4", "CD8A", "CD8B"]
mask = pz.obs["tissue"] == "blood"
sub = pz[genes, mask.values]                # 4 genes × blood cells
```

## Summary Statistics

Compute summary statistics over the on-disk matrix:

```python
# Non-zeros per column (cell) — array of length n_obs
npc = pz.nnz_per_col()

# Non-zeros per row (feature) — array of length n_vars
npr = pz.nnz_per_row()

# Row sums — array of length n_vars
rs = pz.rowsums()

# Column sums — stored natively, no decompression needed
cs = pz.colsums
```

### Quick file summary

```python
desc = pz.describe()
# {'path': 'counts.1pz',
#  'shape': (30000, 10000),
#  'nnz': 15000000,
#  'density': 0.05,
#  'file_size': 47396044,
#  'version': 3,
#  'has_metadata': True,
#  'has_colsums': True,
#  'has_transpose': False,
#  'has_obs_var': True}
```

## Conversion Methods

OnePZFile provides direct conversion to all common formats:

```python
csc = pz.to_csc()         # scipy CSC (native)
csr = pz.to_csr()         # scipy CSR
coo = pz.to_coo()         # scipy COO
arr = pz.to_dense()       # numpy array (warning: large memory)
df  = pz.to_dataframe()   # pandas DataFrame
adata = pz.to_anndata()   # AnnData object (transposed to cells × genes)
```

NumPy interop:

```python
import numpy as np
arr = np.asarray(pz)      # uses __array__ protocol
```

## File Operations

### Copy

```python
pz2 = pz.copy("backup.1pz")   # binary copy → new OnePZFile handle
```

### Head / Tail

Peek at the first or last N cells:

```python
first = pz.head(10)   # 30000 × 10 sparse matrix
last  = pz.tail(10)   # 30000 × 10 sparse matrix
```

### Transpose

```python
t = pz.T   # returns 10000 × 30000 CSC (cells × genes)
```

## Full Read Methods

```python
# Full matrix (float64)
mat = pz.read()

# Partial column range
sub = pz.read(cols=(0, 1000))

# Partial row range (requires stored transpose)
sub = pz.read(rows=(0, 500))

# Log-normalized matrix
norm = pz.read_normalized(scale=10000)

# Integer matrix (saves memory)
mat_int = pz.read(dtype="int32")
```

## On-the-Fly Log-Normalization

The `.1pz` format stores precomputed column sums, enabling on-the-fly
Seurat-style LogNormalize during decompression — no separate normalization
step required. The formula is:

$$\text{normalized}_{ij} = \log\left(1 + \frac{x_{ij}}{\text{colsum}_j} \times \text{scale}\right)$$

### Open in normalized mode

```python
# All reads return log-normalized data automatically
pz = sp.open_1pz("counts.1pz", normalize=True)
pz[:, 0:100]       # log-normalized submatrix
pz.to_anndata()    # AnnData with log-normalized counts
pz.read()          # full log-normalized matrix
```

### Create a normalized view from an existing handle

```python
pz = sp.open_1pz("counts.1pz")
npz = pz.normalized()           # returns new handle with normalize=True
npz = pz.normalized(scale=1e6)  # CPM-style normalization
```

### Per-read normalization

```python
pz = sp.open_1pz("counts.1pz")
mat = pz.read(normalize=True)                 # normalize this read only
sub = pz.read(cols=(0, 500), normalize=True)   # partial read, normalized
```

### Module-level function

```python
mat = sp.read_1pz("counts.1pz", normalize=True, scale=10000)
```

### Manual normalization with `lognorm()`

For post-hoc normalization of already-loaded raw count matrices:

```python
mat = sp.read_1pz("counts.1pz")
cs = mat.colsums                   # stored column sums
norm = sp.lognorm(mat, cs, scale=10000)
```

Summary statistics (`.nnz_per_col()`, `.nnz_per_row()`, `.rowsums()`)
always operate on raw counts regardless of normalization mode.

## Threading

All reads use OpenMP parallelism. Set threads at construction:

```python
pz = sp.open_1pz("counts.1pz", num_threads=16)
```
