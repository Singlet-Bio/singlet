# SPDX-License-Identifier: MIT
"""USA (Unspliced / Spliced / Ambiguous) derivation from canonical counts.1pz."""
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix


class USA(NamedTuple):
    spliced: "csc_matrix"
    unspliced: "csc_matrix"
    ambiguous: "csc_matrix"


def usa(counts) -> USA:
    """Derive USA three-matrix decomposition from a SingletCounts.

    The canonical layout stores UMIs partitioned across
    ``exon_body | intron_body | junctions``. USA is the projection:

      spliced   = exon_body + junctions[non-intronic]
      unspliced = intron_body
      ambiguous = junctions[intron-spanning]

    Returns
    -------
    USA
        NamedTuple of three (n_genes × n_cells) CSC matrices.

    Notes
    -----
    Skeleton — implementation pending BLOCKER #2 sprint.
    """
    raise NotImplementedError(
        "singlet.derive.usa is a BLOCKER #2 skeleton; "
        "implementation pending the canonical-layout writer landing"
    )
