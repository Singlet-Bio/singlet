# SPDX-License-Identifier: MIT
"""The paired gene-expression sample: its column barcodes in ``col`` order, called flags and exon counts.

ADT columns must be the GEX sample's columns: the s1 ``cells`` table lists the columns of ``exon_counts.1pz`` (or
``intron_counts.1pz`` when exon is empty) in order, and ``called`` comes from ``cell_calls.tsv``
(``singlet.s1.migrate``). Legacy bundles are read by byte range from ``data.singlet.bio``; nothing is downloaded whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

BASE = "https://data.singlet.bio"


@dataclass
class GexSample:
    gse: str
    gsm: str
    barcodes: List[str]  # col order
    called: np.ndarray  # bool per col
    genes: List[str]  # exon matrix row names
    exon: "object"  # scipy.sparse.csc_matrix genes x cols (uint32), or None when not loaded
    source: str


def _csc(c):
    from scipy import sparse

    return sparse.csc_matrix((c.data, c.indices.astype(np.int64), c.indptr), shape=(c.m, c.n))


def read_legacy(
    gse: str, gsm: str, url: Optional[str] = None, with_counts: bool = True
) -> GexSample:
    from singlet.s1 import cells as cells_mod
    from singlet.s1.legacy import LegacyBundle, decode_1pz_bytes

    url = url or f"{BASE}/data/{gse}/{gse}.singlet"
    b = LegacyBundle.open(url)
    pre = f"samples/{gsm}/"
    ex = (
        decode_1pz_bytes(b.read(pre + "exon_counts.1pz"))
        if b.exists(pre + "exon_counts.1pz")
        else None
    )
    if ex is None or ex.n == 0:
        it = decode_1pz_bytes(b.read(pre + "intron_counts.1pz"))
        basis = it
    else:
        basis = ex
    called_list = None
    if b.exists(pre + "cell_calls.tsv"):
        called_list = cells_mod.parse_cell_calls(b.read(pre + "cell_calls.tsv"))
    bcs = list(basis.colnames)
    if called_list is None:
        called = np.ones(len(bcs), dtype=bool)
    else:
        cs = set(called_list)
        called = np.fromiter((x in cs for x in bcs), dtype=bool, count=len(bcs))
    return GexSample(
        gse,
        gsm,
        bcs,
        called,
        list(ex.rownames) if ex is not None else [],
        _csc(ex) if (with_counts and ex is not None) else None,
        url,
    )
