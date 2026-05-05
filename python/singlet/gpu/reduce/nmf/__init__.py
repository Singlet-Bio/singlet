# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.reduce.nmf — GPU-native NMF.

Public functions:
    nmf                  — in-memory NMF fit (Lee-Seung MU + Hsieh-Dhillon CD).
    nmf_chunked          — out-of-core streaming NMF.
    nmf_graph_factorize  — multi-modal joint NMF (gated until the FactorGraph
                            backend lands; raises if called).

CYCLE-110 (2026-04-29): the `nmf`/`nmf_chunked`/`nmf_graph_factorize` entry
points were originally in `singlet_gpu/reduce/nmf.py`, but the addition of
this `nmf/` subpackage (for CSI-GEP, cycle 28) silently shadowed it via
Python's package-vs-module precedence — `from singlet.gpu.reduce.nmf import nmf`
broke at import. Fixed by renaming the .py to `_nmf_core.py` and re-exporting
its public symbols from this `__init__.py`.

CSI-GEP (`run_csi_gep`) stays gated behind the deferred-indefinitely scope.
"""

# CYCLE-110: re-export the entry points from the renamed core module.
from singlet.gpu.reduce._nmf_core import (
    nmf,
    nmf_chunked,
    nmf_graph_factorize,
    NmfResult,
)

# CSI-GEP — deferred-indefinitely scope. Importable but gated; underlying
# binding (csi_gep) is only built when SINGLET_GPU_BUILD_DEFERRED is set.
try:
    from .csi_gep import run_csi_gep
except ImportError:
    run_csi_gep = None  # type: ignore[assignment]

__all__ = [
    "nmf",
    "nmf_chunked",
    "nmf_graph_factorize",
    "NmfResult",
    "run_csi_gep",
]
