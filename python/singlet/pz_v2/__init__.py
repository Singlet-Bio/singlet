# SPDX-License-Identifier: MIT
"""singlet.pz_v2 — Canonical .1pz v2 multi-block sparse codec.

Implements the on-disk format from §4 of ``docs/CANONICAL_OUTPUT_FORMAT.md``:

* Magic ``1PZ02\\0\\0\\0`` (8 B)
* JSON header (length-prefixed, UTF-8)
* Multiple named row-blocks sharing one cell-barcode axis
* Two-data-layer CSC (one shared ``indptr``/``indices``, two ``data`` arrays)
  for AD/DP variant tracks

The format is intentionally Python-first so the v2 pipeline can produce
canonical outputs today while the C++ codec catches up.

Public API
----------

* :func:`write_pz_v2` — write a multi-block file.
* :func:`read_pz_v2` — open a file lazily; returns :class:`PzReader`.
* :class:`PzReader` — ``.block(name) -> Block``; supports two-layer CSC.

Examples
--------

>>> import numpy as np
>>> from scipy.sparse import csc_matrix
>>> from singlet.pz_v2 import write_pz_v2, read_pz_v2, BlockSpec
>>> X = csc_matrix(np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int32))
>>> write_pz_v2(
...     "/tmp/x.1pz",
...     cell_barcodes=["AAA", "CCC", "GGG"],
...     blocks=[BlockSpec("exon_body", X)],
... )
>>> with read_pz_v2("/tmp/x.1pz") as rd:
...     assert rd.cell_barcodes == ["AAA", "CCC", "GGG"]
...     assert (rd.block("exon_body").data().toarray() == X.toarray()).all()
"""

from singlet.pz_v2._codec import (
    MAGIC,
    VERSION,
    Block,
    BlockSpec,
    PzReader,
    PzV2Error,
    read_pz_v2,
    write_pz_v2,
)

__all__ = [
    "MAGIC",
    "VERSION",
    "Block",
    "BlockSpec",
    "PzReader",
    "PzV2Error",
    "read_pz_v2",
    "write_pz_v2",
]
