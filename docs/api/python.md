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
| `singlet.to_mtx(adata, directory)` | Write AnnData → 10x MTX directory |
| `singlet.from_mtx(directory)` | Read 10x MTX directory → AnnData |
| `singlet.to_tiledb(adata, uri)` | Write AnnData → TileDB-SOMA (requires `tiledbsoma`) |
| `singlet.from_tiledb(uri)` | Read TileDB-SOMA → AnnData (requires `tiledbsoma`) |
| `singlet.to_csc(adata)` | Extract scipy CSC matrix |

### Cell Type Annotation (free, local)

| Function | Description |
|----------|-------------|
| `singlet.gene_programs(organism)` | Download NMF gene programs (W matrix) |
| `singlet.project(adata)` | Project cells into gene program space (H matrix) |
| `singlet.annotate(adata)` | Annotate cells with predicted cell types |

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
from singlet.gpu import preprocess as sgpp

sgpp.normalize_total(adata)
sgpp.log1p(adata)
gpu.reduce.pca(adata, n_components=50)
gpu.pp.neighbors(adata, n_neighbors=15)
gpu.tools.leiden(adata)
gpu.tools.umap(adata)
gpu.de.pseudobulk_de(adata, groupby="cell_type", sample_col="donor_id")
```

### Submodules

| Submodule | Functions |
|-----------|-----------|
| `gpu.pp` | `neighbors` |
| `gpu.preprocess` | `normalize_total`, `log1p`, `highly_variable_genes`, `scale`, `regress_out` |
| `gpu.reduce` | `pca`, `nmf`, `nmf_chunked`, `nmf_graph_factorize`, `svd_*` |
| `gpu.tools` | `leiden`, `umap`, `rank_genes_groups`, `score_genes`, `celltypist_predict` |
| `gpu.de` | `pseudobulk_de` (donor-level NB GLM) |
| `gpu.enrich` | `run_ssgsea`, `run_progeny`, `run_score_genes` |
| `gpu.io` | `read_pz_to_anndata`, `write_anndata_to_pz`, `load_pz` |

---

## Module: `singlet.mcp`

MCP (Model Context Protocol) server for AI assistants.

```bash
python -m singlet.mcp  # Start stdio server
```

Exposes 11 tools: `singlet_stats`, `singlet_search`, `singlet_qc`,
`singlet_load`, `singlet_browse`, `singlet_protocols`, `singlet_quality`,
`singlet_tissues`, `singlet_failures`, `singlet_cell_types`, `singlet_species`.

---

## Module: `singlet.pp` / `singlet.preprocessing`

Pipeline-level preprocessing (power users / infrastructure operators).

```python
from singlet.pp import download_fastq, detect_protocol, quantify, run_qc
```

| Function | Description |
|----------|-------------|
| `download_fastq(gsm_id, *, ena_r1_url, output_dir)` | Download FASTQ from ENA/SRA |
| `detect_protocol(r1_path, r2_path)` | Auto-detect scRNA-seq protocol |
| `quantify(r1_paths, r2_paths, protocol, species, output_dir)` | Run STAR alignment + counting |
| `run_qc(quant_dir)` | Compute quality metrics |
| `export_to_1pz(quant_dir, output_path)` | Export quantification to .1pz |
| `list_supported_species()` | List available reference genomes |
| `get_species_info(species)` | Get reference paths for a species |

---

## Configuration & Diagnostics

| Function | Description |
|----------|-------------|
| `singlet.set_catalog_dir(path)` | Set local catalog directory |
| `singlet.show_versions()` | Print version info for bug reports |
| Environment: `SINGLET_CATALOG_DIR` | Auto-detected catalog path |
| Environment: `SUPABASE_ANON_KEY` | Required for MCP server |
