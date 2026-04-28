# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.enrich — GPU-native pathway and gene-set enrichment.

NOTE: this package contains ssGSEA (per-cell KS enrichment) and PROGENy
(weighted-sum pathway activity scoring).  For ranked-list GSEA, see
``singlet_gpu.enrichment.run_gsea`` (fgsea.h, cycle 13).

Exposes:
    ssgsea.run_ssgsea    — per-cell ssGSEA (cycle 44)
    progeny.run_progeny  — PROGENy pathway activity (cycle 44)
"""

from .ssgsea import run_ssgsea
from .progeny import run_progeny

__all__ = ["run_ssgsea", "run_progeny"]
