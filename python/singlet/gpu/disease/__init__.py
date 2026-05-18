# SPDX-License-Identifier: MIT
"""
singlet.gpu.disease — GPU-native disease relevance scoring.

Public API
----------
scdrs              — scDRS polygenic enrichment scoring (cycle 50).
scdrs_run_from_csc — scDRS from a raw DeviceCsc.
"""

from .scdrs import run_from_csc as scdrs_run_from_csc
from .scdrs import run_from_anndata as scdrs

__all__ = [
    "scdrs",
    "scdrs_run_from_csc",
]
