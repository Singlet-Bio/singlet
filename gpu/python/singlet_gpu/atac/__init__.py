# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.atac — GPU-native ATAC-seq analysis.

Exposes:
    chromvar.compute — chromVAR motif enrichment scoring (cycle 34).

Underlying C++ cycle: cycle 34 (atac/chromvar.h).
"""

from .chromvar import compute

__all__ = ["compute"]
