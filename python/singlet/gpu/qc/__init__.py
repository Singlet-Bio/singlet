# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.qc — GPU-native quality control.

Exposes:
    doublet_score.run_doublet_score  — Scrublet-style RNA-only (cycle 31)
    omnidoublet.run_omni_doublet     — multimodal CITE-seq (cycle 39)
    qc_metrics.calculate_qc_metrics  — per-cell/gene QC stats (cycle 103)
    qc_metrics.filter_cells          — cell filtering by QC thresholds (cycle 103)
    qc_metrics.filter_genes          — gene filtering by detection rate (cycle 103)
"""

from .doublet_score import run_doublet_score
from .omnidoublet import run_omni_doublet
from .qc_metrics import calculate_qc_metrics, filter_cells, filter_genes

__all__ = [
    "run_doublet_score",
    "run_omni_doublet",
    "calculate_qc_metrics",
    "filter_cells",
    "filter_genes",
]
