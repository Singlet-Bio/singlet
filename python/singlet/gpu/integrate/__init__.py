# SPDX-License-Identifier: MIT
"""
singlet.gpu.integrate — GPU-native batch integration methods.

Exposes:
    harmony_integrate — GPU Harmony integration (drop-in for sc.external.pp.harmony_integrate).
    bbknn             — GPU BBKNN (drop-in for sc.external.pp.bbknn).

Underlying C++ cycle: cycle 14 (integrate/harmony.h, integrate/bbknn.h).

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.harmony`` and
``_core.bbknn`` are expected to have been added by the cycle-22 binding
extension.  Both wrappers raise ``AttributeError`` with this tag if the
bindings are missing.
"""

from .harmony import harmony_integrate
from .bbknn import bbknn

__all__ = ["harmony_integrate", "bbknn"]
