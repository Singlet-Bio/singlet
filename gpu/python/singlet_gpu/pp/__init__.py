# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.pp — GPU-native preprocessing functions.

Exposes:
    neighbors(adata, ...) — GPU kNN graph construction (drop-in for sc.pp.neighbors)

Underlying C++ cycles:
    - cycle 8 (graph/knn.h) → neighbors

Other preprocessing functions (normalize_total, log1p, hvg) live in
``singlet_gpu.preprocess`` for namespace parity with scanpy:

    import singlet_gpu.preprocess as sgpp   # normalize_total, log1p, hvg
    import singlet_gpu.pp as sgp            # neighbors
"""

from singlet_gpu.pp.neighbors import neighbors

__all__ = ["neighbors"]
