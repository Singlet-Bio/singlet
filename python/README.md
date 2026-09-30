# singlet

**Find, load, and analyze re-processed public single-cell RNA-seq studies with one line of Python.**

Singlet re-processes public GEO single-cell RNA-seq studies through one
uniform pipeline and publishes each study as a `.singlet` file. You work with
two simple things: GEO accession strings (`GSE…` / `GSM…`) and `.singlet`
files. Everything loads as [AnnData](https://anndata.readthedocs.io/), ready
for scanpy or PyTorch. Browse the catalog at [singlet.bio](https://singlet.bio).

- **Data: CC0** (public domain) — **Code: MIT**.
- Downloads need no login and no API key.

## Install

Wheels are not on PyPI yet; install from GitHub:

```bash
pip install "singlet-bio @ git+https://github.com/Singlet-Bio/singlet"
```

The PyPI distribution is named **`singlet-bio`** (`singlet` on PyPI is an
unrelated project); the import name is still `singlet`. Once the first
release is published, the install is `pip install singlet-bio`.

This compiles the package's `.1pz` codec, so you need a **C++17 compiler**
and **zstd** (headers + library): `apt install build-essential libzstd-dev`
on Debian/Ubuntu, `brew install zstd` on macOS, or
`conda install -c conda-forge zstd`. Linux and macOS are supported; on
Windows use WSL.

Optional extras (same URL, extra in brackets):

```bash
pip install "singlet-bio[analysis] @ git+https://github.com/Singlet-Bio/singlet"  # matplotlib, scanpy, statsmodels, igraph, leidenalg
pip install "singlet-bio[torch] @ git+https://github.com/Singlet-Bio/singlet"     # PyTorch DataLoaders
pip install "singlet-bio[zarr] @ git+https://github.com/Singlet-Bio/singlet"      # Zarr export/import
pip install "singlet-bio[mcp] @ git+https://github.com/Singlet-Bio/singlet"       # local MCP server (singlet-mcp)
```

## Quick start

```python
import singlet

# Find studies in plain English → GSE accessions (live search)
studies = singlet.find("human PBMC 10x")

# Load a study → AnnData (free download, cached under ~/.singlet/cache)
adata = singlet.load("GSE138867")

# Load one sample: the parent study is downloaded, then filtered to it
pbmc1 = singlet.load("GSM4120733")

# Load and concatenate several studies into ONE AnnData
adata = singlet.load(["GSE138867", "GSE146974"])

# Load a local .singlet file
adata = singlet.load("path/to/GSE138867.singlet")

# Search and load the top three matching studies in one step
adata = singlet.find_load("human pancreas islet cells")
```

`load()` accepts a single accession, a single `.singlet` path, or a list/tuple
mixing both. Lists are concatenated into one AnnData; `obs["source"]` and
`obs["dataset"]` record where each cell came from.

What you get back:

- `X`: gene-level UMI counts (cells × genes, sparse), with `layers["spliced"]`
  and `layers["unspliced"]` when both are available.
- `var_names`: Ensembl gene ids; `var["gene_name"]`: gene symbols.
- `obs`: `gsm_id`, organism, protocol and other per-sample metadata.
- `uns["study_meta"]`, `uns["manifest"]`: study metadata and the file's
  manifest (stored so that `write_h5ad` / `write_zarr` work).
- `uns["skipped_samples"]`: present only when some samples had an empty count
  matrix; those are skipped with a warning. A study with no usable sample
  raises an error pointing to its page on singlet.bio.

Pick genes by symbol or by Ensembl id:

```python
adata = singlet.load("GSE138867", genes=["CD3E", "MS4A1", "ENSG00000105374"])
```

Symbols are matched exactly first, then case-insensitively; anything
unmatched is reported in a warning.

## Natural-language search

```python
import singlet

singlet.find("exhausted T cells in melanoma")            # study (GSE) accessions
singlet.find("mouse brain 10x", level="gsm", limit=20)   # sample (GSM) accessions

# find + load in one call (top 3 studies by default; raise limit deliberately)
adata = singlet.find_load("human kidney organoids", limit=5)
```

Search runs against the hosted endpoint (`https://singlet.bio/api`; override
with `$SINGLET_API_BASE`). It is rate-limited per client; for heavier use,
create a key at https://singlet.bio/account and call `singlet.set_api_key()`
(or set `$SINGLET_API_KEY`).

## Browse the catalog snapshot (offline)

```python
import singlet

singlet.summary()                                   # snapshot overview
singlet.samples(organism="Homo sapiens", status="SUCCESS")
singlet.tissues()                                   # tissue breakdown
singlet.info("GSE138867")                           # metadata
```

These functions read a catalog snapshot bundled with the package, so they
work offline but lag the live catalog. `singlet.info()` falls back to the
live API (`/api/gse/<GSE>`, `/api/gsm/<GSM>`) for accessions the snapshot
does not know, and `python -m singlet` prints live totals from
`https://singlet.bio/api/stats`. Set `SINGLET_OFFLINE=1` to disable those
live lookups.

## Export

```python
singlet.to_h5ad(adata, "study.h5ad")
singlet.to_zarr(adata, "study.zarr")    # needs singlet-bio[zarr]
singlet.to_mtx(adata, "study_mtx/")     # 10x-style matrix.mtx.gz + barcodes + features
```

Loom export is not supported from Python.

## Standard analysis (scanpy-compatible)

```python
import scanpy as sc

adata = singlet.load("GSE138867")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

The package also has built-in equivalents (`singlet.normalize`,
`singlet.pca`, `singlet.neighbors`, `singlet.leiden`, `singlet.umap`, ...)
that do not need scanpy.

## PyTorch training

```python
from singlet.torch import SingletDataset, DataLoader

# An array of .singlet files, concatenated into one training set
loader = DataLoader(
    ["a.singlet", "b.singlet", "c.singlet"],
    batch_size=512,
    device="cuda",
    normalize=True,
)

# A single dataset / accession / AnnData also works
ds = SingletDataset("GSE138867", normalize=True)
```

## MCP servers

- **Hosted:** `https://singlet.bio/mcp` serves the live catalog to MCP clients;
  nothing to install.
- **Local:** `singlet-mcp` (install `singlet-bio[mcp]`) runs 12 tools over the
  offline catalog snapshot plus live search, over stdio. The old command name
  `singlet` still works but is deprecated.

## Configuration

| Variable | Meaning |
|---|---|
| `SINGLET_DATA_BASE` | Bundle host: files live at `<base>/data/<GSE>/<GSE>.singlet` (default `https://data.singlet.bio`; a trailing `/data` is accepted, so the R client's value works too) |
| `SINGLET_SAMPLE_MIRROR` | Self-hosted mirror of per-sample directories for `singlet.fetch()` / `singlet.open()` (no public default) |
| `SINGLET_API_BASE` | REST API base (default `https://singlet.bio/api`) |
| `SINGLET_CACHE_DIR` | Download cache (default `~/.singlet/cache`) |
| `SINGLET_API_KEY` | Optional key for natural-language search |
| `SINGLET_OFFLINE` | `1` disables the live lookups in `info()` and `python -m singlet` |

## Links

- Homepage: https://singlet.bio
- Documentation: https://singlet.bio/docs
- Source: https://github.com/Singlet-Bio/singlet

## License

Code: MIT. Catalog data: CC0 (public domain).
