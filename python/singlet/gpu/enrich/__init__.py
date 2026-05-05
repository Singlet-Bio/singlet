# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.enrich — GPU-native pathway and gene-set enrichment.

NOTE: this package contains ssGSEA (per-cell KS enrichment), PROGENy
(weighted-sum pathway activity scoring), and score_genes (Satija et al. 2015
/ Seurat AddModuleScore per-cell gene-set scoring).
For ranked-list GSEA, see ``singlet.gpu.enrichment.run_gsea`` (fgsea.h, cycle 13).

Exposes:
    ssgsea.run_ssgsea          — per-cell ssGSEA (cycle 44)
    progeny.run_progeny        — PROGENy pathway activity (cycle 44)
    score_genes.run_score_genes — per-cell gene-set scoring (cycle 129)
"""

from .progeny import run_progeny
from .score_genes import run_score_genes
from .ssgsea import run_ssgsea

__all__ = ["run_ssgsea", "run_progeny", "run_score_genes"]
