# SPDX-License-Identifier: MIT
"""PSI (Percent Spliced In) derivation from canonical counts.1pz."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


def psi(counts) -> "csc_matrix":
    """Derive per-junction PSI matrix from a SingletCounts.

    PSI = inclusion_reads / (inclusion_reads + exclusion_reads), computed
    per junction per cell from the canonical ``junctions`` row block.

    Returns
    -------
    scipy.sparse.csc_matrix
        Junction × cell PSI values in [0, 1].

    Notes
    -----
    Skeleton — implementation pending BLOCKER #2 sprint.
    """
    raise NotImplementedError(
        "singlet.derive.psi is a BLOCKER #2 skeleton; "
        "implementation pending the canonical-layout writer landing"
    )
