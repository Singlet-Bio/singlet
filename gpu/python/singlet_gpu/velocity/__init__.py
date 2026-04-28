# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.velocity — GPU-native RNA velocity preprocessing and estimation.

Exposes:
    moments  — first/second-order moment smoothing (drop-in for scvelo.pp.moments).
    velocity — steady-state RNA velocity (drop-in for scvelo.tl.velocity, mode='steady_state').

Underlying C++ cycle: cycle 15 (preprocess/velocity_prep.h —
``singlet_gpu::preprocess::velocity_prep_compute`` kernel).

scvelo parity
-------------
Both functions mirror the scvelo 0.3 API for their respective modes.
Only ``mode='steady_state'`` is supported for velocity(); dynamical and
stochastic modes are deferred to a future cycle.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.velocity_moments`` and
``_core.velocity_prep_compute`` must be exposed by the cycle-22 binding
extension.  Both wrappers raise ``AttributeError`` with this tag if the
bindings are missing.
"""

from .moments import moments
from .velocity import velocity

__all__ = ["moments", "velocity"]
