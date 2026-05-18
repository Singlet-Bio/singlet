# SPDX-License-Identifier: MIT
"""Gene-level UMI counts: a derived projection over the three blocks of
``counts.1pz``.

For each cell ``c``:

.. code-block:: python

    g[gene_id, c] = sum(exon_body[exon_id, c]   for exon_id   in exons_of(gene_id))
                  + sum(intron_body[intron_id, c] for intron_id in introns_of(gene_id))
                  + sum(junctions[jct_id, c]    for jct_id    in junctions_of(gene_id))

The feature → gene mapping comes from ``features.fbin``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from scipy.sparse import csr_matrix

from singlet.refbundle import FeaturesBundle, load_features

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix

    from singlet.io.sample import SingletSample


__all__ = ["gene_counts"]


def _resolve_features(features) -> FeaturesBundle:
    if isinstance(features, FeaturesBundle):
        return features
    return load_features(features)


def _projector(n_rows: int, n_genes: int, ranges) -> csr_matrix:
    """Build a (n_genes, n_rows) sum-projector from (lo, hi) per gene."""
    rows = []
    cols = []
    for gid, (lo, hi) in enumerate(ranges):
        if hi > lo:
            cols.extend(range(lo, hi))
            rows.extend([gid] * (hi - lo))
    data = np.ones(len(rows), dtype=np.int32)
    return csr_matrix(
        (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_genes, n_rows),
    )


def gene_counts(
    sample: "SingletSample",
    features: Union[FeaturesBundle, str, Path],
) -> "csc_matrix":
    """Compute the gene × cell UMI count matrix on demand.

    Sum-projection of ``exon_body + intron_body + junctions`` onto the
    gene axis via ``features.fbin``.

    Parameters
    ----------
    sample
        :class:`singlet.io.SingletSample` instance.
    features
        :class:`singlet.refbundle.FeaturesBundle` or a path to
        ``features.fbin``.

    Returns
    -------
    csc_matrix
        Shape ``(n_genes, n_cells)``.
    """
    fb = _resolve_features(features)
    counts = sample.counts
    n_cells = counts.n_cells
    n_genes = fb.n_genes

    exon_ranges = [(g.exon_lo, g.exon_hi) for g in fb.iter_genes()]
    intron_ranges = [(g.intron_lo, g.intron_hi) for g in fb.iter_genes()]
    junction_ranges = [(g.junction_lo, g.junction_hi) for g in fb.iter_genes()]

    exon = counts.exon_body()
    intron = counts.intron_body()
    jct = counts.junctions()

    p_exon = _projector(exon.shape[0], n_genes, exon_ranges)
    p_intron = _projector(intron.shape[0], n_genes, intron_ranges)
    p_jct = _projector(jct.shape[0], n_genes, junction_ranges)

    out = (p_exon @ exon) + (p_intron @ intron) + (p_jct @ jct)
    return out.tocsc()
