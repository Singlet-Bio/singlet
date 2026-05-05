# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.generative — GPU-native generative models.

Exposes:
    discrete_diffusion.train   — train D3PM model (cycle 30)
    discrete_diffusion.sample  — sample from trained model
    discrete_diffusion.DiscreteDiffusionWrapper
"""

from . import discrete_diffusion

__all__ = ["discrete_diffusion"]
