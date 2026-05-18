# SPDX-License-Identifier: MIT
"""Per-junction PSI (percent-spliced-in) — derived view.

PSI for each annotated alternative splice junction in ``features.fbin``,
computed on demand from the ``junctions`` row block of ``counts.1pz``.

For each junction ``j`` and cell ``c``::

    PSI[j, c] = inclusion(j, c) / (inclusion(j, c) + exclusion(j, c))

where ``exclusion(j, c)`` is the sum of read counts for competing
junctions sharing the same donor *or* acceptor site. Only positions where
the denominator is nonzero are kept in the output.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from singlet.refbundle import FeaturesBundle, load_features

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix

    from singlet.io.sample import SingletSample


__all__ = ["psi"]


def _resolve_features(features) -> FeaturesBundle:
    if isinstance(features, FeaturesBundle):
        return features
    return load_features(features)


def psi(
    sample: "SingletSample",
    features: Union[FeaturesBundle, str, Path],
) -> "csc_matrix":
    """Per-junction × cell PSI matrix.

    Returns
    -------
    csc_matrix
        Shape ``(n_junctions, n_cells)``, float32 in [0, 1]. Implicit-zero
        entries indicate undefined PSI (denominator == 0), NOT zero
        inclusion.
    """
    fb = _resolve_features(features)
    counts = sample.counts
    jct = counts.junctions().tocsr()  # n_jx × n_cells; we will operate per cell column
    n_jx = jct.shape[0]
    n_cells = jct.shape[1]

    # Build donor & acceptor groupings (donor_feat, acceptor_feat → list of jx_id).
    donor_groups = defaultdict(list)
    acceptor_groups = defaultdict(list)
    for jx_id in range(n_jx):
        _gid, df, af, _flags, _motif, _dp, _ap = fb.junction(jx_id)
        donor_groups[df].append(jx_id)
        acceptor_groups[af].append(jx_id)

    # Group-sum projector: for each junction, sum over its donor-group and
    # acceptor-group, then subtract one copy of itself (it was counted twice).
    rows, cols = [], []
    for group in donor_groups.values():
        for j in group:
            rows.extend([j] * len(group))
            cols.extend(group)
    for group in acceptor_groups.values():
        for j in group:
            rows.extend([j] * len(group))
            cols.extend(group)
    proj = csr_matrix(
        (
            np.ones(len(rows), dtype=np.int32),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(n_jx, n_jx),
    )
    # Subtract two self-counts: each junction is in its own donor group and own acceptor group.
    identity_diag = csr_matrix(
        (
            np.full(n_jx, 2, dtype=np.int32),
            (np.arange(n_jx, dtype=np.int64), np.arange(n_jx, dtype=np.int64)),
        ),
        shape=(n_jx, n_jx),
    )
    proj = proj - identity_diag

    # denom[j, c] = inclusion(j, c) + exclusion(j, c)
    #            = jct[j, c] + sum_{other ∈ groups(j)} jct[other, c]
    #            = (proj + I) @ jct   (proj already excludes self contribution after -2I,
    #                                   actually proj currently = donor_set + acceptor_set - 2I
    #                                   which equals: count of competing junctions WITH self
    #                                   once in each set, minus 2 self → exactly competing set)
    # So denom = inclusion + exclusion = jct + proj @ jct = (I + proj) @ jct.
    eye = csr_matrix(
        (
            np.ones(n_jx, dtype=np.int32),
            (np.arange(n_jx, dtype=np.int64), np.arange(n_jx, dtype=np.int64)),
        ),
        shape=(n_jx, n_jx),
    )
    denom = ((eye + proj) @ jct).tocsr()

    # Iterate over the inclusion sparsity pattern; PSI is only computed where
    # the junction itself has reads (otherwise denom may still be 0).
    inclusion = jct.tocoo()
    rows_o = inclusion.row
    cols_o = inclusion.col
    inc_data = inclusion.data.astype(np.float32)

    # Look up denom at matching positions.
    denom_vals = np.asarray(denom[rows_o, cols_o]).ravel().astype(np.float32)
    defined = denom_vals > 0
    psi_vals = np.zeros_like(inc_data)
    psi_vals[defined] = inc_data[defined] / denom_vals[defined]

    return coo_matrix(
        (psi_vals[defined], (rows_o[defined], cols_o[defined])),
        shape=(n_jx, n_cells),
    ).tocsc()
