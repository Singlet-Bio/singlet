# SPDX-License-Identifier: MIT
"""singlet.io — Format I/O for single-cell data.

Provides read/write for .1pz (native), h5ad, zarr, and 10x formats.
"""

from singlet._io import (
    info_1pz,
    read_1pz,
    read_kraken2,
    read_matrix,
    write_1pz,
)
from singlet.io.convert import (
    from_h5ad,
    from_mtx,
    from_tiledb,
    from_zarr,
    to_csc,
    to_h5ad,
    to_mtx,
    to_tiledb,
    to_zarr,
)
from singlet.io.sample import (
    SingletCounts,
    SingletMt,
    SingletNonhost,
    SingletSample,
    SingletSnp,
)

__all__ = [
    "read_1pz",
    "write_1pz",
    "info_1pz",
    "read_matrix",
    "read_kraken2",
    "to_h5ad",
    "to_zarr",
    "to_mtx",
    "to_csc",
    "to_tiledb",
    "from_h5ad",
    "from_zarr",
    "from_mtx",
    "from_tiledb",
    # Canonical v2 readers (singlet.io.sample)
    "SingletSample",
    "SingletCounts",
    "SingletSnp",
    "SingletMt",
    "SingletNonhost",
]
