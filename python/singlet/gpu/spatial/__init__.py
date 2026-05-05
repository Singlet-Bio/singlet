# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.spatial — GPU-native spatial analysis.

Exposes:
    flash_deconv.run_flash_deconv — atlas-scale spatial deconvolution (cycle 33)
    stagate.run_stagate           — spatial domain segmentation (cycle 29)
    cell2fate.fit                 — velocity-module decomposition (cycle 27)
    cell2fate.Cell2FateModel      — trained model wrapper
"""

from .flash_deconv import run_flash_deconv
from .stagate import run_stagate
from . import cell2fate

__all__ = ["run_flash_deconv", "run_stagate", "cell2fate"]
