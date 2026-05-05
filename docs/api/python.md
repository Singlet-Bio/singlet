# Python API Reference

## Package: `singlet`

### Catalog & Browse

| Function | Description |
|----------|-------------|
| `singlet.catalog(search=None)` | Browse all datasets (DataFrame) |
| `singlet.samples(**filters)` | Query sample index with rich filters |
| `singlet.info(accession)` | Get metadata dict for GSE/GSM accession |
| `singlet.sample_index(gse_id=None)` | Full sample index DataFrame |
| `singlet.species()` | List of species in the atlas |
| `singlet.tissues()` | Tissue breakdown DataFrame |
| `singlet.protocols()` | Protocol breakdown DataFrame |
| `singlet.datasets(**filters)` | Filter series catalog |
| `singlet.top_series(n=10)` | Largest series by cell count |
| `singlet.quality_tiers()` | Gold/silver/bronze tier breakdown |
| `singlet.failure_categories()` | Pipeline failure analysis |
| `singlet.cell_types()` | Cell type distribution |
| `singlet.summary()` | One-line atlas overview string |
| `singlet.refresh()` | Re-download catalog from GitHub |

### Data Loading

| Function | Description |
|----------|-------------|
| `singlet.load(source, *, genes=None, obs_filter=None)` | Load GSE accession or file path → AnnData |
| `singlet.load_dir(path, *, layer="gene_counts")` | Load singlify output directory → AnnData |
| `singlet.load_sample(gsm_id, *, genes=None)` | Load single GSM via column-range read |
| `singlet.download(accession, *, source="zenodo")` | Download .1pz file from Zenodo/AWS |

### File I/O

| Function | Returns |
|----------|---------|
| `singlet.read_1pz(path)` | AnnData (preferred format) |
| `singlet.write_1pz(adata, path, **kwargs)` | dict (compression stats) |
| `singlet.read_spz(path)` | AnnData (legacy format) |
| `singlet.write_spz(adata, path, **kwargs)` | dict (compression stats) |
| `singlet.read_matrix(path)` | AnnData (auto-detect format) |
| `singlet.info_1pz(path)` | dict (header metadata) |
| `singlet.spz_info(path)` | dict (header metadata) |
| `singlet.read_kraken2(path)` | AnnData (microbiome matrix) |

### Format Conversion

| Function | Description |
|----------|-------------|
| `singlet.to_h5ad(adata, path)` | Write AnnData → .h5ad |
| `singlet.from_h5ad(path)` | Read .h5ad → AnnData |
| `singlet.to_zarr(adata, path)` | Write AnnData → Zarr |
| `singlet.from_zarr(path)` | Read Zarr → AnnData |
| `singlet.to_csc(adata)` | Extract scipy CSC matrix |

### Token-Priced Features

| Function | Description |
|----------|-------------|
| `singlet.login(key)` | Authenticate with API key |
| `singlet.query(**filters)` | Cross-atlas structured query |
| `singlet.search(text)` | Natural-language semantic search |

---

## Module: `singlet.torch`

PyTorch integration for sparse tensor loading.

```python
from singlet.torch import OnePZDataset, DataLoader, to_sparse_csr, from_anndata
```

| Class/Function | Description |
|----------------|-------------|
| `OnePZDataset(source, *, genes=None, normalize=False)` | Dataset yielding one cell per item |
| `DataLoader(source, *, batch_size=512, shuffle=True)` | Convenience DataLoader wrapper |
| `to_sparse_csr(path, *, dtype="float32", device="cpu")` | Load .spz as sparse CSR tensor |
| `to_sparse_coo(path, *, dtype="float32", device="cpu")` | Load .spz as sparse COO tensor |
| `from_anndata(adata, *, layer=None, device="cpu")` | Convert AnnData → sparse CSR tensor |

---

## Module: `singlet.gpu`

GPU-accelerated analysis (requires `pip install singlet[gpu]`).

```python
from singlet import gpu

gpu.pp.normalize(adata)
gpu.reduce.pca(adata, n_components=50)
gpu.pp.neighbors(adata, n_neighbors=15)
gpu.tools.leiden(adata)
gpu.tools.umap(adata)
gpu.de.pseudobulk(adata, groupby="cell_type", sample_col="donor_id")
```

### Submodules

| Submodule | Functions |
|-----------|-----------|
| `gpu.pp` | `normalize`, `log1p`, `highly_variable_genes`, `neighbors` |
| `gpu.reduce` | `pca`, `nmf`, `svd` |
| `gpu.tools` | `leiden`, `umap`, `diffmap` |
| `gpu.de` | `pseudobulk` (donor-level DESeq2-style) |
| `gpu.enrich` | `ssgsea` (single-sample GSEA) |
| `gpu.io` | `load_pz` (direct GPU load) |

---

## Module: `singlet.mcp`

MCP (Model Context Protocol) server for AI assistants.

```bash
python -m singlet.mcp.server  # Start stdio server
```

Exposes 11 tools: `singlet_stats`, `singlet_search`, `singlet_qc`,
`singlet_load`, `singlet_browse`, `singlet_protocols`, `singlet_quality`,
`singlet_tissues`, `singlet_failures`, `singlet_cell_types`, `singlet_species`.

---

## Configuration

| Function | Description |
|----------|-------------|
| `singlet.set_catalog_dir(path)` | Set local catalog directory |
| Environment: `SINGLET_CATALOG_DIR` | Auto-detected catalog path |
| Environment: `SUPABASE_ANON_KEY` | Required for MCP server |
