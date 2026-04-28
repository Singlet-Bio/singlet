# Quick Start

## Write a sparse matrix

```python
import singlepress as sp
import scipy.sparse as ss

# Create or load a CSC sparse matrix
mat = ss.random(10000, 5000, density=0.05, format="csc", dtype="float64")

# Write with gene names and cell barcodes
stats = sp.write_1pz(
    "counts.1pz", mat,
    rownames=[f"gene_{i}" for i in range(10000)],
    colnames=[f"cell_{i}" for i in range(5000)],
)
print(f"Compressed: {stats['compressed_bytes']} bytes, ratio: {stats.get('ratio', 'N/A')}")
```

## Write with AnnData-style metadata

```python
import pandas as pd

obs = pd.DataFrame({
    "cell_type": ["T-cell"] * 2500 + ["B-cell"] * 2500,
    "tissue": ["blood"] * 5000,
}, index=[f"cell_{i}" for i in range(5000)])

var = pd.DataFrame({
    "gene_name": [f"gene_{i}" for i in range(10000)],
    "highly_variable": [i < 2000 for i in range(10000)],
}, index=[f"gene_{i}" for i in range(10000)])

sp.write_1pz("annotated.1pz", mat,
              obs=obs, var=var,
              uns={"organism": "Homo sapiens", "tissue": "PBMC"})
```

## Read back

```python
mat2 = sp.read_1pz("annotated.1pz")
print(f"Shape: {mat2.shape}, NNZ: {mat2.nnz}")
print(f"Genes: {mat2.rownames[:3]}")
print(f"Cell types: {mat2.obs['cell_type'].value_counts()}")
```

## Lazy / partial access

```python
pz = sp.open_1pz("counts.1pz")
print(pz.shape)       # (10000, 5000)
print(pz.colsums[:5]) # Fast — reads only colsums section

# Read only columns 0–99 (100 cells)
sub = pz.read(cols=(0, 100))

# Read only rows 0–999 (1000 genes, requires stored transpose)
sub_genes = pz.read(rows=(0, 1000))

# Log-normalize using stored column sums
norm = pz.read_normalized(scale=10000)
```

## On-the-fly log-normalization

Read log-normalized data directly — no separate normalization step:

```python
# Method 1: module-level
mat = sp.read_1pz("counts.1pz", normalize=True)

# Method 2: open in normalized mode — all reads are normalized
pz = sp.open_1pz("counts.1pz", normalize=True)
pz[:, 0:100]       # returns log-normalized submatrix

# Method 3: create a normalized view from an existing handle
npz = pz.normalized(scale=10000)
```

The normalization formula is `log1p(x / colsum * scale)`, equivalent to
Seurat's `LogNormalize`. Default scale is 10000.

## Indexing (NumPy-style)

```python
pz = sp.open_1pz("counts.1pz")

# Slice by integers
sub = pz[0:100, 0:500]

# Slice by gene names
sub = pz[["gene_0", "gene_5"], :]

# Boolean mask on cells
mask = pz.obs["cell_type"].values == "T-cell"
sub = pz[:, mask]
```

See [onepz_class.md](onepz_class.md) for the full indexing guide.

## Subsetting and sampling

```python
# Write a filtered subset to a new .1pz
sp.subset_1pz("counts.1pz", "tcells.1pz",
               obs_mask=mask,
               var_names=["CD3D", "CD4", "CD8A"])

# Random downsample
sp.sample_1pz("counts.1pz", "sample_1k.1pz", n=1000, seed=42)
```

See [dataset_operations.md](dataset_operations.md) for concatenation, subsetting, and sampling.

## Integer reads (save memory)

```python
mat_int = sp.read_1pz_int("counts.1pz")
print(mat_int.dtype)  # int32
```

## Inspect without decompression

```python
info = sp.info_1pz("counts.1pz")
print(f"Shape: {info['m']}×{info['n']}, NNZ: {info['nnz']}")
print(f"Has metadata: {info['has_metadata']}")
print(f"Has obs/var: {info['has_obs_var']}")
```

## Validate file integrity

```python
result = sp.validate_1pz("counts.1pz")
print(f"Valid: {result['valid']}")
```

## Command-line interface

```bash
singlepress info counts.1pz
singlepress validate counts.1pz
singlepress colsums counts.1pz
singlepress inspect counts.1pz
```
