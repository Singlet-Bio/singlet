# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.ase — GPU-native allele-specific expression analysis.

Exposes:
    daesc.run — DAESC beta-binomial ASE calling (cycle 40).

Underlying C++ cycle: cycle 40 (ase/daesc.h).
"""

from .daesc import run

__all__ = ["run"]
