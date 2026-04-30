"""
singlet — Python client for the Singlet single-cell atlas.

Browse catalog (works offline):
    singlet.catalog()                      Browse all 3,309 datasets
    singlet.catalog("lung")                Search by keyword
    singlet.info("GSE136831")              Dataset metadata
    singlet.sample_index("GSE136831")      Per-sample column offsets
    singlet.datasets(organism="Homo sapiens", min_cells=100000)

Load data:
    singlet.load("GSE136831")              Load from local catalog or Zenodo → AnnData
    singlet.load("path/to/counts.1pz")     Load local .1pz file
    singlet.load_sample("GSM3308814")      Load single sample (column-range read)

Format I/O:
    singlet.read_1pz("file.1pz")           Read .1pz → AnnData (preferred)
    singlet.write_1pz(adata, "out.1pz")    Write AnnData → .1pz (preferred)
    singlet.read_kraken2("gse_dir/")       Read kraken2 microbiome matrix
    singlet.read_spz("file.spz")           Read .spz → AnnData (legacy)
    singlet.write_spz(adata, "out.spz")    Write AnnData → .spz (legacy)
    singlet.read_matrix("file")            Auto-detect .spz or .1pz

Configuration:
    singlet.set_catalog_dir("/path/to/catalog")  Set local catalog path

Token-priced (requires API key):
    singlet.login(key)                     Authenticate
    singlet.query(...)                     Cross-atlas query → AnnData
    singlet.search(text)                   Natural-language search → AnnData
"""

__version__ = "1.0.0"

from singlet._catalog import (
    catalog, info, species, datasets, sample_index, set_catalog_dir,
    summary, samples, top_series,
)
from singlet._loader import load, load_sample, download
from singlet._auth import login
from singlet._query import query, search
from singlet._io import (
    read_spz, write_spz, spz_info,
    read_1pz, write_1pz, info_1pz,
    read_matrix, read_kraken2,
)
from singlet.convert import to_h5ad, to_zarr, to_csc, from_h5ad, from_zarr

__all__ = [
    # Browse
    "catalog",
    "info",
    "species",
    "datasets",
    "sample_index",
    "set_catalog_dir",
    "summary",
    "samples",
    "top_series",
    # Load
    "load",
    "load_sample",
    "download",
    # Token-priced
    "login",
    "query",
    "search",
    # I/O
    "read_1pz",
    "write_1pz",
    "info_1pz",
    "read_kraken2",
    "read_matrix",
    "read_spz",
    "write_spz",
    "spz_info",
    # Conversions
    "to_h5ad",
    "to_zarr",
    "to_csc",
    "from_h5ad",
    "from_zarr",
]
