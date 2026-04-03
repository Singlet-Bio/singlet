# Format Conversions

singlet supports interoperability with all major single-cell data formats.

## HDF5 (.h5ad)

The standard AnnData format for scanpy ecosystems:

```python
import singlet

adata = singlet.load("GSE136831")

# Export
singlet.to_h5ad(adata, "output.h5ad")

# Import
adata = singlet.from_h5ad("input.h5ad")

# Backed mode (memory-mapped, read-only)
adata = singlet.from_h5ad("large.h5ad", backed="r")
```

## Zarr

Chunked, cloud-friendly storage:

```python
singlet.to_zarr(adata, "output.zarr")
adata = singlet.from_zarr("output.zarr")
```

## TileDB-SOMA

For integration with CELLxGENE and the Census:

```python
from singlet.convert import to_tiledb, from_tiledb

to_tiledb(adata, "soma://my_experiment")
adata = from_tiledb("soma://my_experiment")
```

Requires `pip install singlet[tiledb]`.

## Market Exchange (MTX)

10x Genomics Cell Ranger output format:

```python
from singlet.convert import to_mtx, from_mtx

# Export (creates matrix.mtx.gz, barcodes.tsv.gz, features.tsv.gz)
to_mtx(adata, "output_dir/")

# Import
adata = from_mtx("cellranger_output/filtered_feature_bc_matrix/")
```

## scipy CSC Sparse

```python
csc_matrix = singlet.to_csc(adata)
print(type(csc_matrix))  # scipy.sparse.csc_matrix
```

## Direct .1pz Conversion

```python
from singlet.convert import pz_to_h5ad, h5ad_to_pz

# .1pz → .h5ad
pz_to_h5ad("input.1pz", "output.h5ad")

# .h5ad → .1pz
h5ad_to_pz("input.h5ad", "output.1pz")

# Legacy .spz conversion still supported
from singlet.convert import spz_to_h5ad
spz_to_h5ad("old_input.spz", "output.h5ad")
```
