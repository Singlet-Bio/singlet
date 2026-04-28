# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.preprocess — GPU-native preprocessing for single-cell AnnData.

Public API
----------
normalize_total  — total-count normalization   (cycle-3 lognorm kernel)
log1p            — natural log1p transform      (cycle-3 lognorm kernel)
highly_variable_genes — HVG selection           (cycle-4 HVG kernel)

All functions are drop-in replacements for their scanpy.pp counterparts.
"""

from singlet_gpu.preprocess.lognorm import normalize_total, log1p
from singlet_gpu.preprocess.hvg import highly_variable_genes

__all__ = [
    "normalize_total",
    "log1p",
    "highly_variable_genes",
]
