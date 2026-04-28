# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.perturbation — GPU-native perturbation modeling.

Exposes:
    perturb_graph.fit                  — train CPA (cycle 32)
    perturb_graph.PerturbGraphModel    — trained model with predict_perturbation()
"""

from . import perturb_graph

__all__ = ["perturb_graph"]
