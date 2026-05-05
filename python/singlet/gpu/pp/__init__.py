# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.pp — GPU-native preprocessing functions.

Exposes:
    neighbors(adata, ...) — GPU kNN graph construction (drop-in for sc.pp.neighbors)

Underlying C++ cycles:
    - cycle 8 (graph/knn.h) → neighbors

Other preprocessing functions (normalize_total, log1p, hvg) live in
``singlet.gpu.preprocess`` for namespace parity with scanpy:

    import singlet.gpu.preprocess as sgpp   # normalize_total, log1p, hvg
    import singlet.gpu.pp as sgp            # neighbors
"""

from singlet.gpu.pp.neighbors import neighbors

__all__ = ["neighbors"]
