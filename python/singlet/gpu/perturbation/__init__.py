# SPDX-License-Identifier: MIT
"""
singlet.gpu.perturbation — GPU-native perturbation modeling.

Public API
----------
fit               — train a CPA perturbation-graph model (cycle 32).
PerturbGraphModel — trained model with predict_perturbation().
"""

from .perturb_graph import fit, PerturbGraphModel

__all__ = ["fit", "PerturbGraphModel"]
