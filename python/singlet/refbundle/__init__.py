# SPDX-License-Identifier: MIT
"""Reference bundle format (``features.fbin``, ``snp_sites.fbin``).

Implements the binary formats defined in §3 of
``docs/CANONICAL_OUTPUT_FORMAT.md``. The same Python module is used by
both the builder scripts and the Phase-6 view implementations that need
to resolve feature IDs at runtime.

Public surface
--------------

* :func:`load_features` / :class:`FeaturesBundle` — read ``features.fbin``.
* :func:`write_features` — write ``features.fbin``.
* :func:`load_snp_panel` / :class:`SnpPanel` — read ``snp_sites.fbin``.
* :func:`write_snp_panel` — write ``snp_sites.fbin``.

All reads are mmap-backed where possible; record decoding is lazy.
"""

from singlet.refbundle._features import (
    EXON_REC_SIZE,
    FEATURES_HEADER_SIZE,
    GENE_REC_SIZE,
    INTRON_REC_SIZE,
    JUNCTION_REC_SIZE,
    FeaturesBundle,
    GeneRecord,
    load_features,
    write_features,
)
from singlet.refbundle._snp import (
    SNP_HEADER_SIZE,
    SNP_REC_SIZE,
    SnpPanel,
    SnpSite,
    load_snp_panel,
    write_snp_panel,
)

__all__ = [
    "FeaturesBundle",
    "GeneRecord",
    "load_features",
    "write_features",
    "FEATURES_HEADER_SIZE",
    "GENE_REC_SIZE",
    "EXON_REC_SIZE",
    "INTRON_REC_SIZE",
    "JUNCTION_REC_SIZE",
    "SnpPanel",
    "SnpSite",
    "load_snp_panel",
    "write_snp_panel",
    "SNP_HEADER_SIZE",
    "SNP_REC_SIZE",
]
