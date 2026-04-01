"""
singlet — Python client for the SingletDB atlas.

Free (works offline, Zenodo downloads):
    singlet.catalog()                      Browse all 19,790 datasets
    singlet.load("GSE136831")              Download from Zenodo → AnnData
    singlet.load("GSE136831", source="aws") Stream from AWS (costs tokens)

Token-priced (requires API key):
    singlet.login(key)                     Authenticate
    singlet.query(...)                     Cross-atlas query → AnnData
    singlet.search(text)                   Natural-language search → AnnData

Format I/O:
    singlet.read_spz("file.spz")           Read .spz → AnnData
    singlet.write_spz(adata, "out.spz")    Write AnnData → .spz

Conversions:
    singlet.to_h5ad(adata, "out.h5ad")     Convert to HDF5
    singlet.to_zarr(adata, "out.zarr")     Convert to Zarr
"""

__version__ = "0.2.0"

from singlet._catalog import catalog, info, species, tissues, datasets
from singlet._loader import load, download
from singlet._auth import login
from singlet._query import query, search
from singlet._io import read_spz, write_spz, spz_info
from singlet.convert import to_h5ad, to_zarr, to_csc, from_h5ad, from_zarr

__all__ = [
    # Browse
    "catalog",
    "info",
    "species",
    "tissues",
    "datasets",
    # Load (Free + AWS streaming)
    "load",
    "download",
    # Token-priced
    "login",
    "query",
    "search",
    # I/O
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
