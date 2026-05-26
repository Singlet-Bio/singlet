# SPDX-License-Identifier: MIT
"""Pseudobulk aggregation from canonical counts.1pz + cell metadata."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


def pseudobulk(
    counts,
    groupby: Union[str, Sequence[str]],
    *,
    layer: str = "gene_counts",
    method: str = "sum",
):
    """Aggregate per-cell counts into per-group pseudobulks.

    Parameters
    ----------
    counts : singlet.SingletCounts
        Canonical counts handle.
    groupby : str or sequence of str
        Cell-metadata column(s) from ``cell_meta.parquet`` to group on
        (e.g. ``"donor_id"``, ``["donor_id", "cell_type"]``).
    layer : str
        Which derived layer to aggregate. One of
        ``{"gene_counts", "usa", "psi", "gene_full"}``.
    method : str
        ``"sum"`` (default) or ``"mean"``.

    Returns
    -------
    pandas.DataFrame or scipy.sparse matrix
        Group × gene aggregated matrix plus a group-label index.

    Notes
    -----
    Skeleton — implementation pending BLOCKER #2 sprint.
    """
    raise NotImplementedError(
        "singlet.derive.pseudobulk is a BLOCKER #2 skeleton; "
        "implementation pending the canonical-layout writer landing"
    )
