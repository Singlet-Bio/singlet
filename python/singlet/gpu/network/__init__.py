# SPDX-License-Identifier: MIT
"""
singlet.gpu.network — GPU-native co-expression network analysis.

Public API
----------
hdwgcna              — hdWGCNA WGCNA-style co-expression network inference (cycle 46).
hdwgcna_run_from_csc — hdWGCNA from a raw DeviceCsc.
"""

from .hdwgcna import run_from_csc as hdwgcna_run_from_csc
from .hdwgcna import run_from_anndata as hdwgcna

__all__ = [
    "hdwgcna",
    "hdwgcna_run_from_csc",
]
