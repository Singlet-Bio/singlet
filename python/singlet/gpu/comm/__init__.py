# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.comm — GPU-native cell-cell communication analysis.

Submodules
----------
cellchat : CellChat ligand-receptor communication scoring (cycle 37).
"""

from singlet.gpu.comm import cellchat as cellchat_module
from singlet.gpu.comm.cellchat import run_from_anndata as cellchat
from singlet.gpu.comm.cellchat import run_from_csc as cellchat_run_from_csc

__all__ = [
    "cellchat",
    "cellchat_run_from_csc",
    "cellchat_module",
]
