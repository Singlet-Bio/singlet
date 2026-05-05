"""Format conversions: .1pz/.spz <-> h5ad, zarr, TileDB-SOMA, MTX, CSC.

This module re-exports from singlet.io.convert to avoid code duplication.
"""

from singlet.io.convert import (
    from_h5ad,
    from_mtx,
    from_tiledb,
    from_zarr,
    h5ad_to_pz,
    h5ad_to_spz,
    pz_to_h5ad,
    spz_to_h5ad,
    to_csc,
    to_h5ad,
    to_mtx,
    to_tiledb,
    to_zarr,
)

__all__ = [
    "from_h5ad",
    "from_mtx",
    "from_tiledb",
    "from_zarr",
    "h5ad_to_pz",
    "h5ad_to_spz",
    "pz_to_h5ad",
    "spz_to_h5ad",
    "to_csc",
    "to_h5ad",
    "to_mtx",
    "to_tiledb",
    "to_zarr",
]
