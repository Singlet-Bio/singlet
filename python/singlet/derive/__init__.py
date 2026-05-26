# SPDX-License-Identifier: MIT
"""
singlet.derive — fast client-side derivation of classical matrices from the
canonical `counts.1pz` ground-truth file plus the shared `features.fbin`
reference bundle.

Per MVP_ROADMAP.md BLOCKER #2 and CANONICAL_OUTPUT_FORMAT.md, the on-disk
layout stores only the partitioned UMI matrix (exon_body ‖ intron_body ‖
junctions). Any classical view — gene-level counts, USA, PSI, gene_full,
pseudobulk — is derived on demand. Target: <200 ms per derivation on a typical
12K-cell sample.

Public API (skeleton — implementations land in BLOCKER #2 sprint):

    from singlet.derive import gene_counts, usa, psi, gene_full, pseudobulk

    counts = singlet.SingletCounts("sample/counts.1pz")
    g  = gene_counts(counts)                  # CSC, shape (n_genes, n_cells)
    g2 = gene_counts(counts, method="em")     # ambient-corrected EM estimator
    u  = usa(counts)                           # (spliced, unspliced, ambiguous)
    p  = psi(counts)                           # per-junction PSI
    gf = gene_full(counts)                     # full-length gene quantification
    pb = pseudobulk(counts, groupby="donor")   # pseudobulk by metadata column

All routines read `features.fbin` from the reference bundle referenced by the
counts file's `reference_id` header field. Reference bundles are fetched (and
cached) by `singlet.fetch_reference()`.
"""
from __future__ import annotations

from .gene_counts import gene_counts
from .usa import usa
from .psi import psi
from .gene_full import gene_full
from .pseudobulk import pseudobulk

__all__ = [
    "gene_counts",
    "usa",
    "psi",
    "gene_full",
    "pseudobulk",
]
