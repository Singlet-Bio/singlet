# Quick Start

## Browse the Catalog

The catalog is free and works offline after the first download:

```python
import singlet

# Full catalog as a DataFrame
df = singlet.catalog()
print(f"{len(df)} datasets available")

# Search by keyword
lung = singlet.catalog(search="lung")

# Filter by species and tissue
human_brain = singlet.datasets(species="human", tissue="brain")

# Get info on a specific dataset
info = singlet.info("GSE136831")
print(info["title"], f"— {info['cell_count']} cells")
```

## Load a Dataset

```python
# Download from Zenodo (free) → AnnData
adata = singlet.load("GSE136831")
print(adata)  # AnnData object with n_obs × n_vars

# With gene subset (memory-efficient)
adata = singlet.load("GSE136831", genes=["TP53", "BRCA1", "KRAS"])

# From AWS (faster, costs tokens)
adata = singlet.load("GSE136831", backend="aws")
```

## Read/Write .spz Files

```python
# Write AnnData to compressed .spz
singlet.write_spz(adata, "my_data.spz")

# Read it back
adata2 = singlet.read_spz("my_data.spz")

# Check file info without loading
info = singlet.spz_info("my_data.spz")
print(f"{info['rows']}×{info['cols']}, ratio={info['ratio']:.1f}×")
```

## Convert Formats

```python
# To HDF5
singlet.to_h5ad(adata, "output.h5ad")

# To Zarr
singlet.to_zarr(adata, "output.zarr")

# From HDF5
adata = singlet.from_h5ad("input.h5ad")
```

## PyTorch Integration

```python
from singlet.torch import to_sparse_csr, SpzDataset, DataLoader

# Zero-copy sparse CSR tensor (ideal for GPU)
tensor = to_sparse_csr("my_data.spz", device="cuda")
print(tensor.shape, tensor.device)

# DataLoader for training
loader = DataLoader("GSE136831", batch_size=512, device="cuda")
for batch in loader:
    # batch: (512, n_genes) dense tensor
    loss = model(batch)
```

## Cross-Atlas Queries (Token-Priced)

```python
singlet.login("sk-your-api-key")

# Structured query
result = singlet.query(
    species="human",
    tissue="lung",
    cell_type="macrophage",
    disease="Crohn's disease",
)

# Natural language search
result = singlet.search("exhausted T cells in pediatric leukemia")
```
