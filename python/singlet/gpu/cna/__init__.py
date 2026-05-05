# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.cna — GPU-native copy-number alteration detection.

Exposes:
    numbat.detect — Numbat CNA detection (cycle 35).

Underlying C++ cycle: cycle 35 (cna/numbat.h).
"""

from .numbat import detect

__all__ = ["detect"]
