# SPDX-License-Identifier: MIT
"""singlet.views — Derived projections over canonical v2 sample data.

These functions compute on demand from the canonical row blocks of
``counts.1pz`` (and the reference's ``features.fbin`` for the
feature → gene mapping). Nothing here is cached on disk; the v2 spec
deliberately stores only canonical inputs.

All views target sub-200 ms latency on a 12K-cell sample (single-pass
CSC scan with NumPy / SciPy).

Status: API stubs. Implementations land with Phase 6 of
docs/V2_IMPLEMENTATION_PLAN.md.
"""

from __future__ import annotations

from .gene_counts import gene_counts
from .psi import psi
from .usa import usa

__all__ = ["gene_counts", "usa", "psi"]
