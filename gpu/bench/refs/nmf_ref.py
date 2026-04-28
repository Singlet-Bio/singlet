"""
singlet-gpu/bench/refs/nmf_ref.py
SPDX-License-Identifier: GPL-2.0-or-later

SOTA reference for reduce/nmf benchmark.
Primary: sklearn.decomposition.NMF (k components, coordinate descent).
  NMF operates on raw non-negative counts — do NOT log-normalize.

Note: sklearn NMF on 11k × 33k matrices is slow (~minutes). We use
max_iter=100 to match the C++ driver's cfg.maxit=100.

Usage: python3 nmf_ref.py <sample_dir> [k=10]
Emits: SOTA_WALL_SEC=... SOTA_MEM_MB=... SOTA_CELLS=...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import time_function, emit_result, find_sample_dir


def _run_nmf(sample_dir: Path, k: int):
    import scanpy as sc
    import numpy as np
    from sklearn.decomposition import NMF

    # Load raw counts — NMF requires non-negative input, do not normalize.
    adata = sc.read_10x_mtx(str(sample_dir), cache=False, gex_only=True)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Transpose to genes × cells (sklearn NMF: rows=samples, but we keep
    # cells × genes which is sklearn's expected convention).
    # sklearn NMF shape: (n_samples, n_features) = (cells, genes).
    nmf = NMF(n_components=k, max_iter=100, random_state=42,
              init="nndsvd", solver="cd")
    nmf.fit_transform(X)
    return adata.n_obs


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: nmf_ref.py <sample_dir> [k]", file=sys.stderr)
        sys.exit(1)

    sample_dir = find_sample_dir(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    if sample_dir is None:
        emit_result(-1.0, -1.0, 0)
        return

    try:
        wall_sec, mem_mb, n_cells = time_function(_run_nmf, sample_dir, k)
        emit_result(wall_sec, mem_mb, n_cells)
    except ImportError as e:
        print(f"[nmf_ref] missing dependency: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)
    except Exception as e:
        print(f"[nmf_ref] error: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)


if __name__ == "__main__":
    main()
