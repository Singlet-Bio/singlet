# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.fate — GPU-native cell fate trajectory analysis.

Submodules
----------
cospar    : CoSpar cell fate transition mapping (cycle 41).
cellrank2 : CellRank 2 absorption probability solver (cycle 43).
palantir  : Palantir diffusion pseudotime + branch probabilities (cycle 45).
"""

from singlet_gpu.fate.cospar    import run_from_csc as cospar_run_from_csc
from singlet_gpu.fate.cospar    import run_from_anndata as cospar
from singlet_gpu.fate.cellrank2 import compute_absorption_probabilities
from singlet_gpu.fate.cellrank2 import run_from_anndata as cellrank2
from singlet_gpu.fate.palantir  import run_from_embedding as palantir_run_from_embedding
from singlet_gpu.fate.palantir  import run_from_anndata as palantir

from singlet_gpu.fate import cospar as cospar_module
from singlet_gpu.fate import cellrank2 as cellrank2_module
from singlet_gpu.fate import palantir as palantir_module

__all__ = [
    "cospar",
    "cospar_run_from_csc",
    "cellrank2",
    "compute_absorption_probabilities",
    "palantir",
    "palantir_run_from_embedding",
    "cospar_module",
    "cellrank2_module",
    "palantir_module",
]
