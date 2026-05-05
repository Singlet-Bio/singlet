# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.eqtl — GPU-native single-cell eQTL mapping.

Exposes:
    nebula.run — NEBULA NB mixed-model eQTL mapping (cycle 38).

Underlying C++ cycle: cycle 38 (eqtl/nebula.h).
"""

from .nebula import run

__all__ = ["run"]
