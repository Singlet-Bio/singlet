"""
Interop adapters for :mod:`singlify.io` outputs.

Each submodule converts a :class:`singlify.io.PipelineDirectory` into a
bioinformatics-ecosystem-native object:

- :mod:`singlify.interop.anndata` — :class:`anndata.AnnData` with layers
- :mod:`singlify.interop.scanpy` — Scanpy convenience entry points
- :mod:`singlify.interop.seurat` — rpy2 bridge to R Seurat (optional)

Imports are lazy so that a user who only wants AnnData doesn't pay the
cost of touching rpy2/Seurat. Only the submodule actually used at call
time is imported.
"""

from __future__ import annotations

__all__ = ["anndata"]


def __getattr__(name: str):
    if name == "anndata":
        from . import anndata as _ad
        return _ad
    if name == "scanpy":
        from . import scanpy as _sc
        return _sc
    if name == "seurat":
        from . import seurat as _se
        return _se
    raise AttributeError(f"singlify.interop has no attribute {name!r}")
