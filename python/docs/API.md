# API Reference

## Catalog Browsing

### `singlet.catalog(search=None)`
Return the full dataset catalog as a DataFrame. Optionally filter by keyword.

### `singlet.summary()`
Print a summary of the atlas: total samples, success count, species, cells.

### `singlet.samples(organism=None, status=None, protocol=None)`
Query individual samples (GSM-level). Returns DataFrame with per-sample metadata.

### `singlet.top_series(n=10)`
Return the top N series by total cell count.

### `singlet.info(accession)`
Return metadata dict for a single GSE accession.

### `singlet.species()`
Return sorted list of all species with processed data.

### `singlet.datasets(organism=None, protocol=None, min_cells=None, has_kraken2=None)`
Filter catalog by organism, protocol, minimum cell count, or kraken2 availability.

### `singlet.sample_index(gse_id=None)`
Return sample-level index with column offsets for partial reads. One row per GSM.

### `singlet.set_catalog_dir(path)`
Set local catalog directory containing `catalog_v1.parquet` and `sample_index.parquet`.

---

## Data Loading

### `singlet.load_dir(path)` → `AnnData`
Load a full singlify output directory as AnnData. Reads count matrix, cell metadata, gene features, doublet scores, cell cycle phases, ancestry, sex call, and pipeline summary.

**Returns:** AnnData with:
- `obs`: total_umis, total_genes, mt_pct, intronic_pct, doublet_score, is_doublet, phase, s_score, g2m_score
- `uns`: ancestry, sex_call, summary, singlify_dir
- `var`: gene_id, gene_name

### `singlet.load(source, *, genes=None, obs_filter=None, backend="zenodo")`
Primary entry point. Load a dataset as AnnData from:
- Local file path (`.1pz`, `.spz`, `.h5ad`, `.zarr`)
- GSE accession (resolved via local catalog, then Zenodo download)

**Parameters:**
- `source` — GEO accession or file path
- `genes` — Subset to these gene names
- `obs_filter` — Dict of `{column: value}` to filter cells
- `backend` — `"zenodo"` (free) or `"aws"` (token-priced)

**Returns:** `anndata.AnnData` with obs, var, and uns metadata.

### `singlet.load_sample(gsm_id, *, genes=None)`
Load a single GSM sample using column-range reads. Requires local catalog.

### `singlet.download(accession, output_dir=None, force=False, source="zenodo")`
Download a dataset file to local cache.

---

## Format I/O

### `singlet.read_1pz(path)` → `AnnData`
Read a `.1pz` file. Returns AnnData (cells × genes) with embedded obs, var, uns metadata.

### `singlet.write_1pz(adata, path, *, layer=None, store_transpose=False, include_obs=True, include_var=True, include_uns=True)` → `dict`
Write AnnData to `.1pz` format with VOCSC + zstd-3 compression.

### `singlet.info_1pz(path)` → `dict`
Read `.1pz` header without decompressing. Returns dimensions, nnz, compression ratio.

### `singlet.spz_info(path)` → `dict`
Read `.spz` header without decompressing. Returns dimensions, nnz, format version.

### `singlet.read_kraken2(gse_dir)` → `AnnData`
Read a `kraken2.1pz` microbiome matrix from a GSE directory. Returns cells × taxa.

### `singlet.read_matrix(path)` → `AnnData`
Auto-detect format (`.1pz` or `.spz`) and read.

### `singlet.read_spz(path, *, col_range=None)` → `AnnData`
Read a legacy `.spz` file.

### `singlet.write_spz(adata, path, *, layer=None, row_sort=False, precision="auto")` → `dict`
Write AnnData to legacy `.spz` format.

---

## Format Conversions

### `singlet.to_h5ad(adata, path)`
Write AnnData to HDF5 (`.h5ad`).

### `singlet.to_zarr(adata, path)`
Write AnnData to Zarr.

### `singlet.from_h5ad(path)` → `AnnData`
Read from HDF5.

### `singlet.from_zarr(path)` → `AnnData`
Read from Zarr.

### `singlet.to_csc(adata)` → `scipy.sparse.csc_matrix`
Extract sparse CSC matrix from AnnData.

---

## PyTorch Integration

```python
from singlet.torch import OnePZDataset, SpzDataset, DataLoader, to_sparse_csr, to_sparse_coo, from_anndata
```

### `to_sparse_csr(source, *, dtype="float32", device="cpu")`
Load a `.spz` file as a PyTorch sparse CSR tensor. Shape: (cells, genes).

### `to_sparse_coo(source, *, dtype="float32", device="cpu")`
Load a `.spz` file as a PyTorch sparse COO tensor.

### `from_anndata(adata, *, layer=None, dtype="float32", device="cpu")`
Convert AnnData to PyTorch sparse CSR tensor.

### `OnePZDataset(source, *, genes=None, normalize=False, device="cpu", sparse=False)`
PyTorch Dataset backed by a `.1pz` file or AnnData. Yields one cell per `__getitem__`.

- `normalize=True` applies `log1p(x * 10000 / total_counts)` normalization.
- `.n_genes` — number of genes
- `.gene_names` — gene name list

### `SpzDataset(source, *, layer=None, genes=None, device="cpu", sparse=False)`
Legacy PyTorch Dataset backed by a `.spz` file or AnnData.

### `DataLoader(source, *, batch_size=512, shuffle=True, num_workers=0, device="cpu", genes=None, normalize=False, sparse=False)`
Convenience wrapper around `torch.utils.data.DataLoader`. Uses `OnePZDataset` internally. Supports list of sources for multi-dataset training.

---

## Preprocessing (FASTQ → .1pz)

```python
from singlet.preprocessing import download_fastq, detect_protocol, quantify, run_qc, export_to_1pz
```

### `download_fastq(accession, output_dir, *, source="ena", max_parallel=4)`
Download FASTQ files for a GEO sample.

### `detect_protocol(fastq_dir)`
Detect sequencing protocol from FASTQ reads.

### `quantify(fastq_dir, output_dir, *, species="human", protocol=None)`
Run simpleaf/alevin-fry quantification.

### `run_qc(quant_dir)` → `dict`
Compute QC metrics (cells, genes detected, mapping rate).

### `export_to_1pz(quant_dir, output_path)`
Export quantification output to `.1pz` format.

---

## Authentication

### `singlet.login(key)`
Set API key for token-priced operations.

### `singlet.query(*, species=None, tissue=None, cell_type=None, disease=None, ...)`
Structured cross-atlas query. Returns AnnData.

### `singlet.search(text)`
Natural-language search across the atlas. Returns AnnData.
