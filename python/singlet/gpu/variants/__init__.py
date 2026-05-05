# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.variants — GPU-native somatic variant calling from scRNA pileup.

Exposes:
    monopogen.call — Monopogen germline + somatic SNV calling (cycle 42).

Underlying C++ cycle: cycle 42 (variants/monopogen.h).
"""

from .monopogen import call

__all__ = ["call"]
