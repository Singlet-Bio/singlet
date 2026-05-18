# SPDX-License-Identifier: MIT
"""
singlet.gpu.generative — GPU-native generative models.

Public API
----------
train                    — train a D3PM discrete-diffusion model (cycle 30).
sample                   — sample synthetic expression from a trained model.
DiscreteDiffusionWrapper — trained-model wrapper with save/load.
"""

from .discrete_diffusion import train, sample, DiscreteDiffusionWrapper

__all__ = ["train", "sample", "DiscreteDiffusionWrapper"]
