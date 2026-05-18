# SPDX-License-Identifier: MIT
"""
singlet.gpu.abundance — GPU-native differential abundance testing.

Public API
----------
milo                     — Milo kNN-based differential abundance testing (cycle 47).
milo_run_from_embedding  — Milo from a raw embedding.
"""

from .milo import run_from_embedding as milo_run_from_embedding
from .milo import run_from_anndata as milo

__all__ = [
    "milo",
    "milo_run_from_embedding",
]
