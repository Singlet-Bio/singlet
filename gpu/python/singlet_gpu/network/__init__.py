# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.network — GPU-native co-expression network analysis.

Submodules
----------
hdwgcna : hdWGCNA WGCNA-style co-expression network inference (cycle 46).
"""

from singlet_gpu.network.hdwgcna import run_from_csc as hdwgcna_run_from_csc
from singlet_gpu.network.hdwgcna import run_from_anndata as hdwgcna
from singlet_gpu.network import hdwgcna as hdwgcna_module

__all__ = [
    "hdwgcna",
    "hdwgcna_run_from_csc",
    "hdwgcna_module",
]
