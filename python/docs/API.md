# API Reference

## Catalog Browsing

### `singlet.summary()` → `str`
Return a one-line summary of the atlas.

```python
>>> singlet.summary()
'singlet atlas: 3,196 samples (1,368 SUCCESS) • 1,399 series • 8 species • 28 protocols • 36 tissues • 40 cell types • 4.0M cells'
```

### `singlet.samples(gse_id=None, organism=None, status=None, tissue=None, cell_type=None, protocol=None, min_cells=None, quality_tier=None, search=None)` → `DataFrame`
Query individual samples (GSM-level) with optional filters.

**Parameters:**
- `organism` — Filter by organism (substring match, e.g. "Homo sapiens")
- `status` — Filter by pipeline status ("SUCCESS", "SOFT_FAIL", "HARD_FAIL")
- `tissue` — Filter by tissue (e.g. "brain", "lung", "blood")
- `cell_type` — Filter by cell type (e.g. "PBMC", "T cells", "stem cells")
- `protocol` — Filter by protocol (e.g. "10xv3", "dropseq")
- `quality_tier` — "gold" (MR≥70%, genes≥500, cells≥500), "silver", or "bronze"
- `min_cells` — Minimum cell count
- `search` — Text search across title and text columns

**Returns:** DataFrame with columns: gsm_id, gse_id, organism, status, failure_category, protocol, mapping_rate, cells_called, median_genes, median_umis, mt_pct, doublet_rate, wall_time_s, title, source, tissue, cell_type.

### `singlet.catalog(search=None)` → `DataFrame`
Return the full series-level catalog. Optionally filter by keyword.

### `singlet.top_series(n=10, min_samples=3, organism=None)` → `DataFrame`
Return the top N series by total cell count. Includes tissue column.

### `singlet.info(accession)` → `dict`
Return metadata dict for a single GSE accession.

### `singlet.species()` → `list`
Return sorted list of all species with successful processed data (deduplicated from combo organisms).

### `singlet.tissues()` → `DataFrame`
Return tissue distribution across SUCCESS samples. Columns: tissue, count.

### `singlet.cell_types()` → `DataFrame`
Return cell type distribution across SUCCESS samples. 40 normalized categories. Columns: cell_type, count.

### `singlet.protocols()` → `DataFrame`
Return protocol distribution across SUCCESS samples. Columns: protocol, count.

### `singlet.quality_tiers()` → `DataFrame`
Return quality tier breakdown: gold/silver/bronze with average metrics.

### `singlet.failure_categories()` → `DataFrame`
Return failure category breakdown for non-SUCCESS samples (ground truth from pipeline). Categories: align_low_map, download_fail, cells_below_threshold, pipeline_crash, align_oom, data_incomplete, autodetect_species_fail.

### `singlet.datasets(organism=None, protocol=None, min_cells=None, has_kraken2=None)` → `DataFrame`
Filter catalog by organism, protocol, minimum cell count, or kraken2 availability.

### `singlet.sample_index(gse_id=None)` → `DataFrame`
Return sample-level index. One row per GSM.

### `singlet.set_catalog_dir(path)`
Set local catalog directory containing `catalog_v1.parquet` and `sample_index.parquet`.

### `singlet.refresh()`
Re-download the latest catalog and sample index from GitHub. Bypasses bundled data.

---

## Data Loading

### `singlet.load_dir(path)` → `AnnData`
Load a full singlify output directory as AnnData. Reads count matrix, cell metadata, gene features, doublet scores, cell cycle phases, ancestry, sex call, and pipeline summary.

**Returns:** AnnData with:
- `obs`: total_umis, total_genes, mt_pct, intronic_pct, doublet_score, is_doublet, phase, s_score, g2m_score
- `uns`: ancestry, sex_call, summary, singlify_dir
- `var`: gene_id, gene_name

### `singlet.load(source, *, genes=None, obs_filter=None, backend="zenodo")`
Primary entry point. Load a dataset as AnnData from local file path or GEO accession.

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

### `singlet.read_kraken2(gse_dir)` → `AnnData`
Read a `kraken2.1pz` microbiome matrix. Returns cells × taxa.

### `singlet.read_matrix(path)` → `AnnData`
Auto-detect format (`.1pz` or `.spz`) and read.

### `singlet.read_spz(path, *, col_range=None)` → `AnnData`
Read a legacy `.spz` file.

### `singlet.write_spz(adata, path, *, layer=None, row_sort=False, precision="auto")` → `dict`
Write AnnData to legacy `.spz` format.

### `singlet.spz_info(path)` → `dict`
Read `.spz` header without decompressing.

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
from singlet.torch import OnePZDataset, DataLoader, to_sparse_csr, from_anndata
```

### `OnePZDataset(source, *, genes=None, normalize=False, device="cpu", sparse=False)`
PyTorch Dataset backed by a `.1pz` file or AnnData. Yields one cell per `__getitem__`.
- `normalize=True` applies `log1p(x * 10000 / total_counts)`

### `DataLoader(source, *, batch_size=512, shuffle=True, num_workers=0, device="cpu", genes=None, normalize=False, sparse=False)`
Convenience wrapper around `torch.utils.data.DataLoader`. Supports multi-dataset training.

### `to_sparse_csr(source, *, dtype="float32", device="cpu")`
Load to PyTorch sparse CSR tensor.

### `from_anndata(adata, *, layer=None, dtype="float32", device="cpu")`
Convert AnnData to PyTorch sparse CSR tensor.

---

## MCP Server

```bash
python -m singlet.mcp.server
```

11 tools: `singlet_stats`, `singlet_search`, `singlet_qc`, `singlet_load`, `singlet_browse`, `singlet_protocols`, `singlet_quality`, `singlet_tissues`, `singlet_failures`, `singlet_cell_types`.

---

## Authentication

### `singlet.login(key)`
Set API key for token-priced operations.

### `singlet.query(*, species=None, tissue=None, cell_type=None, disease=None, ...)`
Structured cross-atlas query. Returns AnnData.

### `singlet.search(text)`
Natural-language search across the atlas. Returns AnnData.
