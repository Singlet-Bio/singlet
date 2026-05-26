# SPDX-License-Identifier: MIT
"""Gene-level UMI count derivation from canonical counts.1pz."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


def gene_counts(
    counts,
    *,
    method: Literal["sum", "em"] = "sum",
    ambient_correction: bool = False,
) -> "csc_matrix":
    """Derive a (n_genes × n_cells) CSC matrix from a SingletCounts.

    Parameters
    ----------
    counts : singlet.SingletCounts
        Canonical counts handle backed by a `counts.1pz` file.
    method : {"sum", "em"}
        - "sum": exon_body + intron_body + junctions per gene per cell.
        - "em": expectation-maximization re-estimate using ambient correction.
    ambient_correction : bool
        When ``method == "sum"``, optionally subtract ambient profile.

    Returns
    -------
    scipy.sparse.csc_matrix
        Gene × cell UMI counts. Bit-identical to the legacy `gene_counts.1pz`
        when ``method == "sum"`` and ``ambient_correction is False``.

    Notes
    -----
    Skeleton — implementation pending BLOCKER #2 sprint.
    Performance target: <200 ms on 12K-cell sample.
    """
    raise NotImplementedError(
        "singlet.derive.gene_counts is a BLOCKER #2 skeleton; "
        "implementation pending the canonical-layout writer landing"
    )
