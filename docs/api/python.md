# Python API Reference

Install: `pip install "singlet @ git+https://github.com/Singlet-Bio/singlet"`
(needs a C++17 compiler and zstd; see [installation](../installation.md)).

## Package: `singlet`

### Find & load (live)

| Function | Description |
|----------|-------------|
| `singlet.find(query, *, level="gse", limit=50)` | Natural-language search → list of accessions (`"gse"` studies by default, `"gsm"` samples) |
| `singlet.find_load(query, *, level="gse", limit=3, **load_kwargs)` | `find` + `load` of the top matches (three studies by default) |
| `singlet.set_api_key(key)` | Optional key for a higher search rate limit (or `$SINGLET_API_KEY`) |
| `singlet.load(source, *, genes=None, obs_filter=None, force=False)` | Accession (`"GSE138867"`, `"GSM4120733"`), `.singlet`/`.h5ad`/`.zarr` path, or a list of them → one AnnData |
| `singlet.download(accession, output_dir=None, force=False)` | Download a study's `.singlet` bundle to the cache (or `output_dir`); returns its path |
| `singlet.open_bundle(source, *, force=False)` | Open a study's bundle (accession or path) as a `SingletBundle` for per-modality access |
| `singlet.load_dir(path, *, layer="gene_counts")` | Load a local pipeline output directory → AnnData |

Notes on `load`:

- `genes=` takes Ensembl ids (`var_names`) or gene symbols (`var["gene_name"]`,
  exact match first, then case-insensitive); unmatched names are warned about.
- Samples with an empty count matrix are skipped with a warning and listed in
  `adata.uns["skipped_samples"]` (a DataFrame with `gsm_id`, `reason`). A study
  with no usable sample raises `RuntimeError` naming its page,
  `https://singlet.bio/study/<GSE>`.
- `adata.uns["manifest"]["checksums"]` is stored as parallel `path` / `sha256`
  lists so that `write_h5ad` and `write_zarr` work.

### Catalog snapshot (offline)

These read a catalog snapshot bundled with the package. It works offline but
lags the live catalog at singlet.bio; `python -m singlet` prints live totals.

| Function | Description |
|----------|-------------|
| `singlet.catalog(search=None)` | Browse studies (DataFrame) |
| `singlet.samples(**filters)` | Query the sample index with filters |
| `singlet.info(accession, *, live=True)` | Metadata dict for a GSE/GSM; falls back to the live API (`/api/gse/<id>`, `/api/gsm/<id>`) when the snapshot lacks it |
| `singlet.sample_index(gse_id=None)` | Full sample index DataFrame |
| `singlet.species()` | Species in the snapshot |
| `singlet.tissues()` | Tissue breakdown DataFrame |
| `singlet.protocols()` | Protocol breakdown DataFrame |
| `singlet.datasets(**filters)` | Filter the study catalog |
| `singlet.top_series(n=10)` | Largest studies by cell count |
| `singlet.quality_tiers()` | Gold/silver/bronze tier breakdown |
| `singlet.failure_categories()` | Pipeline failure analysis |
| `singlet.cell_types()` | Cell type distribution |
| `singlet.summary()` | One-line snapshot overview |
| `singlet.refresh()` | Re-download the snapshot from GitHub |

### File I/O

> **Implementation**: `read_1pz`, `write_1pz`, and `info_1pz` use the
> in-tree pybind11 binding `singlet._pz` on
> `include/singlet/pileup/pz_{reader,writer}.h` — no external package required.

| Function | Returns |
|----------|---------|
| `singlet.read_1pz(path)` | AnnData |
| `singlet.write_1pz(adata, path, **kwargs)` | dict (path, shape, nnz) |
| `singlet.read_matrix(path)` | AnnData (auto-detect `.1pz` variant) |
| `singlet.info_1pz(path)` | dict (header metadata) |
| `singlet.read_kraken2(path)` | AnnData (microbiome matrix) |

### Format Conversion

| Function | Description |
|----------|-------------|
| `singlet.to_h5ad(adata, path)` | Write AnnData → .h5ad |
| `singlet.from_h5ad(path)` | Read .h5ad → AnnData |
| `singlet.to_zarr(adata, path)` | Write AnnData → Zarr (requires `zarr`) |
| `singlet.from_zarr(path)` | Read Zarr → AnnData |
| `singlet.to_mtx(adata, directory)` | Write AnnData → 10x MTX directory |
| `singlet.from_mtx(directory)` | Read 10x MTX directory → AnnData |
| `singlet.to_tiledb(adata, uri)` | Write AnnData → TileDB-SOMA (requires `tiledbsoma`) |
| `singlet.from_tiledb(uri)` | Read TileDB-SOMA → AnnData (requires `tiledbsoma`) |
| `singlet.to_csc(adata)` | Extract scipy CSC matrix |

Loom is not supported from Python.

### Cell type annotation (local)

| Function | Description |
|----------|-------------|
| `singlet.annotate_cell_types(adata, marker_dict, *, groupby="leiden")` | Score clusters against marker-gene sets |
| `singlet.predict_cell_type(adata, reference_adata, label_key)` | Classifier trained on a labelled reference |
| `singlet.transfer_labels(adata, ref_adata, label_key)` | kNN label transfer in the reference embedding |

### Retired

These targeted hosts that were never brought into service and now raise
`NotImplementedError` with a pointer to the replacement:

| Name | Former host | Use instead |
|------|-------------|-------------|
| `gene_programs`, `project`, `annotate` | models.singlet.bio | `annotate_cell_types`, `predict_cell_type` |
| `query`, `search`, `login` | api.singlet.bio/v1 | `find` + `load`; `set_api_key` |
| `fetch()` without `base_url` | data.singlet.bio/v1 | `load`, `download`, `open_bundle` (`fetch(base_url=...)` still works against a mirror) |
| `singlet.atlas.*` | r2.singlet.bio | `catalog`, `info`, `find` |

---

## Module: `singlet.torch`

PyTorch integration (`singlet[torch]`).

```python
from singlet.torch import SingletDataset, DataLoader, to_sparse_csr, from_anndata
```

| Class/Function | Description |
|----------------|-------------|
| `SingletDataset(source, *, genes=None, normalize=False)` | Map-style dataset over a `.singlet` file, accession or AnnData; one cell per item |
| `DataLoader(source, *, batch_size=512, shuffle=True)` | DataLoader wrapper; `source` may be a list of `.singlet` files/accessions (concatenated) |
| `to_sparse_csr(path, *, dtype="float32", device="cpu")` | Load a matrix file as a sparse CSR tensor |
| `to_sparse_coo(path, *, dtype="float32", device="cpu")` | Load a matrix file as a sparse COO tensor |
| `from_anndata(adata, *, layer=None, device="cpu")` | Convert AnnData → sparse CSR tensor |
| `OnePZDataset` | Deprecated alias of `SingletDataset` |

---

## Module: `singlet.gpu` (experimental)

`singlet[gpu]` installs CuPy only. The CUDA extension the module needs
(`singlet.gpu._core`) is not built by `pip install`; see
[the GPU install notes](../install.md). The submodules
(`singlet.gpu.preprocess`, `.reduce`, `.tools`, `.de`, ...) are importable for
tooling, but their functions require that extension.

---

## Module: `singlet.mcp`

The **local** MCP (Model Context Protocol) server, installed with
`singlet[mcp]`:

```bash
singlet-mcp            # stdio server (same as: python -m singlet.mcp)
```

It exposes 12 tools: `singlet_stats`, `singlet_nl_search`, `singlet_search`,
`singlet_qc`, `singlet_load`, `singlet_browse`, `singlet_protocols`,
`singlet_quality`, `singlet_tissues`, `singlet_failures`,
`singlet_cell_types`, `singlet_species`. The aggregate tools read the offline
catalog snapshot; search, QC, load and browse call the live REST API.

This is separate from the **hosted** MCP endpoint at
`https://singlet.bio/mcp`, which serves the live catalog and needs no install.
The old console-script name `singlet` still starts the local server but is
deprecated (it clashes with the pipeline binary's name).

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

| Setting | Description |
|----------|-------------|
| `singlet.show_versions()` | Print version info for bug reports |
| `python -m singlet` | Version plus live catalog totals (snapshot summary when offline) |
| `SINGLET_DATA_BASE` | Bundle host: files at `<base>/data/<GSE>/<GSE>.singlet` (default `https://data.singlet.bio`; a trailing `/data` is accepted) |
| `SINGLET_API_BASE` | REST API base (default `https://singlet.bio/api`) |
| `SINGLET_CACHE_DIR` | Download cache (default `~/.singlet/cache`) |
| `SINGLET_API_KEY` | Optional natural-language search key |
| `SINGLET_OFFLINE` | `1` disables the live lookups in `info()` and `python -m singlet` |
| `SINGLET_BINARY` | Path to the C++ pipeline binary |
| `singlet.set_catalog_dir(path)` / `SINGLET_CATALOG_DIR` | Local processing-tree catalog (cluster use) |
