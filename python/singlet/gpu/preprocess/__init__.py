# SPDX-License-Identifier: MIT
"""
singlet.gpu.preprocess — GPU-native preprocessing for single-cell AnnData.

Public API
----------
normalize_total       — total-count normalization    (cycle-3 lognorm kernel)
log1p                 — natural log1p transform       (cycle-3 lognorm kernel)
highly_variable_genes — HVG selection                (cycle-4 HVG kernel)
scale                 — Z-score scaling               (cycle-103 scale kernel)
regress_out           — OLS confounder regression     (cycle-103 regress_out kernel)
neighbors             — GPU kNN graph construction     (cycle-8 graph/knn kernel)

All functions are drop-in replacements for their scanpy.pp counterparts.
"""

from .lognorm import normalize_total, log1p
from .hvg import highly_variable_genes
from .scale import scale, regress_out
from .neighbors import neighbors

__all__ = [
    "normalize_total",
    "log1p",
    "highly_variable_genes",
    "scale",
    "regress_out",
    "neighbors",
]
