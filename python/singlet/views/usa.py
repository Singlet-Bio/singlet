# SPDX-License-Identifier: MIT
"""USA decomposition (unspliced / spliced / ambiguous) — derived view.

Replaces the legacy ``spliced.1pz`` / ``unspliced.1pz`` / ``ambiguous.1pz``
trio. Computed on demand from ``counts.1pz`` row blocks plus the
``features.fbin`` junction classification.

Junction classification convention
----------------------------------
``JunctionRec.flags & 0x03`` encodes the type:

* ``0`` — EE (exon → exon, mature splice)
* ``1`` — EI (exon → intron)
* ``2`` — IE (intron → exon)
* ``3`` — II (intron → intron, only in dirty annotations)

Definitions
-----------
For each gene ``g`` and cell ``c``:

* ``spliced[g, c]``   = exon_body sum + EE junctions
* ``unspliced[g, c]`` = intron_body sum + (EI ∪ IE) junctions
* ``ambiguous[g, c]`` = II junctions only
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Union

import numpy as np
from scipy.sparse import csr_matrix

from singlet.refbundle import FeaturesBundle, load_features

if TYPE_CHECKING:
    from scipy.sparse import csc_matrix

    from singlet.io.sample import SingletSample


__all__ = ["usa", "UsaTriplet"]


class UsaTriplet(NamedTuple):
    """The three USA matrices, all shape ``(n_genes, n_cells)``."""

    spliced: "csc_matrix"
    unspliced: "csc_matrix"
    ambiguous: "csc_matrix"


def _resolve_features(features) -> FeaturesBundle:
    if isinstance(features, FeaturesBundle):
        return features
    return load_features(features)


def _gene_sum_projector(n_rows: int, n_genes: int, ranges) -> csr_matrix:
    rows, cols = [], []
    for gid, (lo, hi) in enumerate(ranges):
        if hi > lo:
            cols.extend(range(lo, hi))
            rows.extend([gid] * (hi - lo))
    data = np.ones(len(rows), dtype=np.int32)
    return csr_matrix(
        (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(n_genes, n_rows),
    )


def _junction_class_projector(fb: FeaturesBundle, want_type: int) -> csr_matrix:
    """Return a (n_genes, n_junctions) projector keeping only junctions
    of the requested class (``flags & 0x03 == want_type``)."""
    rows, cols = [], []
    for jx_id in range(fb.n_junctions):
        gene_id, _df, _af, flags, _motif, _dp, _ap = fb.junction(jx_id)
        if (flags & 0x03) == want_type:
            rows.append(gene_id)
            cols.append(jx_id)
    data = np.ones(len(rows), dtype=np.int32)
    return csr_matrix(
        (data, (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64))),
        shape=(fb.n_genes, fb.n_junctions),
    )


def usa(
    sample: "SingletSample",
    features: Union[FeaturesBundle, str, Path],
) -> UsaTriplet:
    """Compute the (spliced, unspliced, ambiguous) decomposition.

    Notes
    -----
    By construction, ``spliced + unspliced + ambiguous == gene_counts(sample, features)``.
    """
    fb = _resolve_features(features)
    counts = sample.counts
    n_genes = fb.n_genes

    exon = counts.exon_body()
    intron = counts.intron_body()
    jct = counts.junctions()

    p_exon = _gene_sum_projector(exon.shape[0], n_genes,
                                 [(g.exon_lo, g.exon_hi) for g in fb.iter_genes()])
    p_intron = _gene_sum_projector(intron.shape[0], n_genes,
                                   [(g.intron_lo, g.intron_hi) for g in fb.iter_genes()])

    p_ee = _junction_class_projector(fb, 0)
    p_ei = _junction_class_projector(fb, 1)
    p_ie = _junction_class_projector(fb, 2)
    p_ii = _junction_class_projector(fb, 3)

    spliced = (p_exon @ exon) + (p_ee @ jct)
    unspliced = (p_intron @ intron) + (p_ei @ jct) + (p_ie @ jct)
    ambiguous = p_ii @ jct

    return UsaTriplet(spliced.tocsc(), unspliced.tocsc(), ambiguous.tocsc())
