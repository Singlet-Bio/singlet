# SPDX-License-Identifier: MIT
"""
singlet.gpu.fate — GPU-native cell fate trajectory analysis.

Public API
----------
cospar                           — CoSpar cell fate transition mapping (cycle 41).
cospar_run_from_csc              — CoSpar from a raw DeviceCsc.
cellrank2                        — CellRank 2 fate analysis (cycle 43).
compute_absorption_probabilities — CellRank 2 absorption probability solver.
palantir                         — Palantir diffusion pseudotime (cycle 45).
palantir_run_from_embedding      — Palantir from a raw embedding.
"""

from .cospar import run_from_csc as cospar_run_from_csc
from .cospar import run_from_anndata as cospar
from .cellrank2 import compute_absorption_probabilities
from .cellrank2 import run_from_anndata as cellrank2
from .palantir import run_from_embedding as palantir_run_from_embedding
from .palantir import run_from_anndata as palantir

__all__ = [
    "cospar",
    "cospar_run_from_csc",
    "cellrank2",
    "compute_absorption_probabilities",
    "palantir",
    "palantir_run_from_embedding",
]
