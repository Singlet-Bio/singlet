# Quick Start

## Browse the Catalog

The catalog is free and works offline after the first download:

```python
import singlet

# Full catalog as a DataFrame (3,309 GSE datasets, 354M cells)
df = singlet.catalog()
print(f"{len(df)} datasets, {df.n_cells.sum():,} total cells")

# Search by keyword
lung = singlet.catalog(search="lung")

# Filter by organism, protocol, cell count
large_human = singlet.datasets(organism="Homo sapiens", min_cells=100_000)

# Only datasets with microbiome data
k2 = singlet.datasets(has_kraken2=True)

# Get info on a specific dataset
meta = singlet.info("GSE136831")
print(meta["gse_id"], meta["organism"], f"— {meta['n_cells']:,} cells")

# Sample-level index with column offsets
samples = singlet.sample_index("GSE136831")
print(samples[["gsm_id", "n_cells", "col_offset"]])
```

## Load a Dataset

```python
# Load from local catalog (instant) or Zenodo (free download) → AnnData
adata = singlet.load("GSE136831")
print(adata)
# AnnData with embedded obs (barcode, gsm_id, organism, total_counts),
# var (gene_name, reference), and uns (title, protocol, pubmed_ids, etc.)

# With gene subset
adata = singlet.load("GSE136831", genes=["TP53", "BRCA1", "KRAS"])

# Filter cells at load time
adata = singlet.load("GSE136831", obs_filter={"organism": "Homo sapiens"})

# Load a single sample (column-range read, no full-matrix decompression)
sample = singlet.load_sample("GSM3308814")
```

### Local Catalog

For HPC environments with the full catalog on disk:

```python
singlet.set_catalog_dir("/path/to/cellarium/catalog")
# Or set SINGLET_CATALOG_DIR environment variable

# Now load() resolves accessions to local .1pz files (no download)
adata = singlet.load("GSE136831")
```

## Read/Write .1pz Files

```python
# Write AnnData to compressed .1pz
singlet.write_1pz(adata, "my_data.1pz")

# Read it back — all metadata preserved
adata2 = singlet.read_1pz("my_data.1pz")

# Check file info without loading
info = singlet.info_1pz("my_data.1pz")
print(f"{info['rows']}×{info['cols']}, ratio={info['ratio']:.1f}×")

# Legacy .spz files are also supported
adata3 = singlet.read_spz("old_data.spz")
```

## Kraken2 Microbiome Data

```python
# Load microbiome matrix alongside expression
adata_expr = singlet.read_1pz("/path/to/GSE117795/counts.1pz")
adata_k2 = singlet.read_kraken2("/path/to/GSE117795/")

# adata_k2 has cells × taxa with taxon metadata in var
print(f"{adata_k2.shape[1]} taxa detected across {adata_k2.shape[0]} cells")
```

## Convert Formats

```python
singlet.to_h5ad(adata, "output.h5ad")
singlet.to_zarr(adata, "output.zarr")
adata = singlet.from_h5ad("input.h5ad")
```

## PyTorch Integration

```python
from singlet.torch import OnePZDataset, DataLoader, to_sparse_csr

# Zero-copy sparse CSR tensor (ideal for GPU)
tensor = to_sparse_csr("my_data.1pz", device="cuda")

# Dataset with log-normalization
dataset = OnePZDataset("GSE136831", normalize=True, device="cuda")
print(f"{len(dataset)} cells, {dataset.n_genes} genes")

# DataLoader for training
loader = DataLoader("GSE136831", batch_size=512,
                    normalize=True, device="cuda")
for batch in loader:
    # batch: (512, n_genes) dense tensor
    loss = model(batch)
```

## Cross-Atlas Queries (Token-Priced)

```python
singlet.login("sk-your-api-key")

result = singlet.query(
    species="human",
    tissue="lung",
    cell_type="macrophage",
    disease="Crohn's disease",
)

result = singlet.search("exhausted T cells in pediatric leukemia")
```
