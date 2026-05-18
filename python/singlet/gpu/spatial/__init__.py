# SPDX-License-Identifier: MIT
"""
singlet.gpu.spatial — GPU-native spatial analysis.

Public API
----------
run_flash_deconv — atlas-scale spatial deconvolution (cycle 33).
run_stagate      — spatial domain segmentation (cycle 29).
fit              — cell2fate velocity-module decomposition (cycle 27).
Cell2FateModel   — trained cell2fate model wrapper.
"""

from .flash_deconv import run_flash_deconv
from .stagate import run_stagate
from .cell2fate import fit, Cell2FateModel

__all__ = ["run_flash_deconv", "run_stagate", "fit", "Cell2FateModel"]
