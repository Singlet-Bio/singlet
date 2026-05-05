# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.disease — GPU-native disease relevance scoring.

Submodules
----------
scdrs : scDRS polygenic enrichment scoring (cycle 50).
"""

from singlet_gpu.disease.scdrs import run_from_csc as scdrs_run_from_csc
from singlet_gpu.disease.scdrs import run_from_anndata as scdrs
from singlet_gpu.disease import scdrs as scdrs_module

__all__ = [
    "scdrs",
    "scdrs_run_from_csc",
    "scdrs_module",
]
