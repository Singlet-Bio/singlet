"""singlet.io — Format I/O for single-cell data.

Provides read/write for .1pz (native), .spz (legacy), h5ad, zarr, and 10x formats.
"""
from singlet._io import (
    read_1pz, write_1pz, info_1pz,
    read_spz, write_spz, spz_info,
    read_matrix, read_kraken2,
)
from singlet.io.convert import to_h5ad, to_zarr, to_csc, from_h5ad, from_zarr

__all__ = [
    "read_1pz", "write_1pz", "info_1pz",
    "read_spz", "write_spz", "spz_info",
    "read_matrix", "read_kraken2",
    "to_h5ad", "to_zarr", "to_csc", "from_h5ad", "from_zarr",
]
