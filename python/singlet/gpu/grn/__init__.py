# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.grn — GPU-native gene regulatory network inference.

Exposes:
    granie.run_from_csc — GRaNIE multiome GRN inference (cycle 36).

Underlying C++ cycle: cycle 36 (grn/granie.h).
"""

from .granie import run_from_csc

__all__ = ["run_from_csc"]
