# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.disease — GPU-native disease relevance scoring.

Submodules
----------
scdrs : scDRS polygenic enrichment scoring (cycle 50).
"""

from singlet.gpu.disease.scdrs import run_from_csc as scdrs_run_from_csc
from singlet.gpu.disease.scdrs import run_from_anndata as scdrs
from singlet.gpu.disease import scdrs as scdrs_module

__all__ = [
    "scdrs",
    "scdrs_run_from_csc",
    "scdrs_module",
]
