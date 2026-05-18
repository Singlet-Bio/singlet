# SPDX-License-Identifier: MIT
"""
singlet.gpu.atac — GPU-native ATAC-seq analysis.

Exposes:
    chromvar.compute — chromVAR motif enrichment scoring (cycle 34).

Underlying C++ cycle: cycle 34 (atac/chromvar.h).
"""

from .chromvar import compute

__all__ = ["compute"]
