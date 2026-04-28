# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.reduce.nmf — GPU-native NMF methods.

Exposes:
    csi_gep.run_csi_gep  — consensus bootstrap NMF (cycle 28)

The core NMF entry points (nmf, nmf_chunked, nmf_graph_factorize)
live in ``singlet_gpu.reduce.nmf`` (from cycles 18-23 wrapper sprint).
This sub-package adds CSI-GEP.
"""

from .csi_gep import run_csi_gep

__all__ = ["run_csi_gep"]
