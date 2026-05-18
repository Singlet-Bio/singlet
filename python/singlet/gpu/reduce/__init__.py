# SPDX-License-Identifier: MIT
"""
singlet.gpu.reduce — GPU-native dimensionality reduction.

Public API
----------
pca                  — PCA via factornet 5-method SVD auto-select  (cycle 5)
svd_lanczos          — Lanczos SVD backend                          (cycle 5)
svd_irlba            — factornet IRLBA SVD backend                  (cycle 5)
svd_randomized       — randomized SVD backend                       (cycle 5)
svd_krylov           — Krylov-constrained SVD backend               (cycle 5)
svd_deflation        — deflation SVD backend                        (cycle 5)

nmf                  — GPU-native NMF via factornet                 (cycle 6)
nmf_chunked          — out-of-core streaming NMF                    (cycle 6)
nmf_graph_factorize  — multi-modal joint NMF via FactorGraph        (cycle 6)
"""

from .svd import (
    pca,
    svd_lanczos,
    svd_irlba,
    svd_randomized,
    svd_krylov,
    svd_deflation,
)
from .nmf import (
    nmf,
    nmf_chunked,
    nmf_graph_factorize,
)

__all__ = [
    "pca",
    "svd_lanczos",
    "svd_irlba",
    "svd_randomized",
    "svd_krylov",
    "svd_deflation",
    "nmf",
    "nmf_chunked",
    "nmf_graph_factorize",
]
