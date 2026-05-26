# SPDX-License-Identifier: MIT
"""Full-length gene quantification from canonical counts.1pz."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


def gene_full(counts) -> "csc_matrix":
    """Derive full-length gene quantification (exons + introns + junctions).

    For 5'/3' single-cell protocols this is approximately equal to
    ``gene_counts(counts, method="sum")``; for Smart-seq2 / full-length
    protocols it captures intronic + exonic coverage uniformly.

    Returns
    -------
    scipy.sparse.csc_matrix
        Gene × cell counts.

    Notes
    -----
    Skeleton — implementation pending BLOCKER #2 sprint.
    """
    raise NotImplementedError(
        "singlet.derive.gene_full is a BLOCKER #2 skeleton; "
        "implementation pending the canonical-layout writer landing"
    )
