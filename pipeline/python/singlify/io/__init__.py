"""
singlify.io — readers for singlify pipeline outputs (`.1pz` files).

The pipeline produces a directory of ``.1pz`` sparse-matrix files plus sidecar
TSV/JSON files. This module provides pure-Python readers that decode the
binary format directly into :class:`scipy.sparse.csc_matrix` objects with the
embedded GEO metadata attached.

The top-level convenience function is :func:`singlify.io.read_matrix` which
returns a sparse matrix for a single ``.1pz`` file. Full directory reads
(returning a multi-layer :class:`anndata.AnnData`) live in
:mod:`singlify.interop.anndata`.

See also
--------
singlify/docs/1PZ_FORMAT_SPEC.md
    Binary format specification.
"""

from __future__ import annotations

from ._format import (
    MAGIC,
    HEADER_SIZE,
    FOOTER_SIZE,
    VTCode,
    Flags,
    FeatureFlags,
    PZHeader,
    PZFooter,
)
from ._file import PZFile, PZError, open_pz
from ._metadata import PZMetadata, read_metadata
from ._reader import (
    PipelineDirectory,
    read_matrix,
    read_metadata_from_file,
    read_dir,
    read_cohort,
    aggregate_features_to_gene,
)

__all__ = [
    "MAGIC",
    "HEADER_SIZE",
    "FOOTER_SIZE",
    "VTCode",
    "Flags",
    "FeatureFlags",
    "PZHeader",
    "PZFooter",
    "PZFile",
    "PZError",
    "open_pz",
    "PZMetadata",
    "read_metadata",
    "PipelineDirectory",
    "read_matrix",
    "read_metadata_from_file",
    "read_dir",
    "read_cohort",
    "aggregate_features_to_gene",
]
