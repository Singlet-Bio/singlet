# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.preprocess — GPU-native preprocessing for single-cell AnnData.

Public API
----------
normalize_total       — total-count normalization    (cycle-3 lognorm kernel)
log1p                 — natural log1p transform       (cycle-3 lognorm kernel)
highly_variable_genes — HVG selection                (cycle-4 HVG kernel)
scale                 — Z-score scaling               (cycle-103 scale kernel)
regress_out           — OLS confounder regression     (cycle-103 regress_out kernel)

All functions are drop-in replacements for their scanpy.pp counterparts.
"""

from singlet.gpu.preprocess.hvg import highly_variable_genes
from singlet.gpu.preprocess.lognorm import log1p, normalize_total
from singlet.gpu.preprocess.scale import regress_out, scale

__all__ = [
    "normalize_total",
    "log1p",
    "highly_variable_genes",
    "scale",
    "regress_out",
]
