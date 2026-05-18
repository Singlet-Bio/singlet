# SPDX-License-Identifier: MIT
"""
singlet.gpu.enrichment — GPU-native gene-set and pathway enrichment analysis.

Public API
----------
run_gsea        — preranked GSEA via cycle-13 fgsea kernel (decoupleR-compatible).
run_aucell      — AUCell scoring via cycle-13 aucell kernel (decoupleR-compatible).
run_ssgsea      — per-cell single-sample GSEA enrichment (cycle 44).
run_progeny     — PROGENy weighted-sum pathway activity scoring (cycle 44).
run_score_genes — per-cell gene-set scoring, Seurat AddModuleScore parity (cycle 129).
"""

from .gsea import run_gsea
from .aucell import run_aucell
from .ssgsea import run_ssgsea
from .progeny import run_progeny
from .score_genes import run_score_genes

__all__ = [
    "run_gsea",
    "run_aucell",
    "run_ssgsea",
    "run_progeny",
    "run_score_genes",
]
