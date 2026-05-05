# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.lineage — GPU-native mitochondrial lineage tracing.

Exposes:
    detect_clones — MT heteroplasmy clone calling (NEW, no scanpy/MQuad equivalent).

Underlying C++ cycle: cycle 16 (anno/mt_lineage.h —
``singlet::gpu::anno::mt_lineage_call_clones`` kernel).

This module exploits the ``mt_alleles.1pz`` singlify output — a (MT sites ×
cells) matrix of per-site heteroplasmy counts — that no other framework reads
natively.  Clone detection is performed entirely on device.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.mt_lineage_call_clones``
must be exposed by the cycle-22 pybind11 binding extension.  The wrapper
raises ``AttributeError`` with this tag if the binding is missing.
"""

from .mt import detect_clones

__all__ = ["detect_clones"]
