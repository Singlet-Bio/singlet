# SPDX-License-Identifier: MIT
"""
singlet.gpu.comm — GPU-native cell-cell communication analysis.

Public API
----------
cellchat              — CellChat ligand-receptor communication scoring (cycle 37).
cellchat_run_from_csc — CellChat from a raw DeviceCsc.
"""

from .cellchat import run_from_csc as cellchat_run_from_csc
from .cellchat import run_from_anndata as cellchat

__all__ = [
    "cellchat",
    "cellchat_run_from_csc",
]
