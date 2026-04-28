"""
singlet-gpu/bench/refs/svd_ref.py
SPDX-License-Identifier: GPL-2.0-or-later

SOTA reference for reduce/svd benchmark.
Primary: cuml.decomposition.TruncatedSVD (GPU, from rapids-singlecell env).
Fallback: sklearn.decomposition.TruncatedSVD (CPU).

The matrix is loaded via scanpy, normalized (TotalCount + log1p), then the
top-k components are computed with the chosen backend.

Usage: python3 svd_ref.py <sample_dir> [k=30]
Emits: SOTA_WALL_SEC=... SOTA_MEM_MB=... SOTA_CELLS=...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import time_function, emit_result, find_sample_dir


def _run_svd(sample_dir: Path, k: int):
    import scanpy as sc
    import numpy as np

    adata = sc.read_10x_mtx(str(sample_dir), cache=False, gex_only=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Convert to dense for cuml (cuml TruncatedSVD takes numpy array or cupy).
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Try cuml GPU first.
    try:
        import cuml
        svd = cuml.TruncatedSVD(n_components=k, algorithm="jacobi")
        svd.fit_transform(X)
        print("INFO backend=cuml_gpu", flush=True)
        return adata.n_obs
    except ImportError:
        pass

    # Fallback: sklearn CPU.
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=k, random_state=42)
    svd.fit_transform(X)
    print("INFO backend=sklearn_cpu", flush=True)
    return adata.n_obs


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: svd_ref.py <sample_dir> [k]", file=sys.stderr)
        sys.exit(1)

    sample_dir = find_sample_dir(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    if sample_dir is None:
        emit_result(-1.0, -1.0, 0)
        return

    try:
        wall_sec, mem_mb, n_cells = time_function(_run_svd, sample_dir, k)
        emit_result(wall_sec, mem_mb, n_cells)
    except ImportError as e:
        print(f"[svd_ref] missing dependency: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)
    except Exception as e:
        print(f"[svd_ref] error: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)


if __name__ == "__main__":
    main()
