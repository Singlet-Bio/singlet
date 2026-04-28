# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.abundance — GPU-native differential abundance testing.

Submodules
----------
milo : Milo kNN-based differential abundance testing (cycle 47).
"""

from singlet_gpu.abundance.milo import run_from_embedding as milo_run_from_embedding
from singlet_gpu.abundance.milo import run_from_anndata as milo
from singlet_gpu.abundance import milo as milo_module

__all__ = [
    "milo",
    "milo_run_from_embedding",
    "milo_module",
]
