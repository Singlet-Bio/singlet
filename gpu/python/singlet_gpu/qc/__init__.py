# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.qc — GPU-native quality control.

Exposes:
    doublet_score.run_doublet_score  — Scrublet-style RNA-only (cycle 31)
    omnidoublet.run_omni_doublet     — multimodal CITE-seq (cycle 39)
"""

from .doublet_score import run_doublet_score
from .omnidoublet import run_omni_doublet

__all__ = ["run_doublet_score", "run_omni_doublet"]
