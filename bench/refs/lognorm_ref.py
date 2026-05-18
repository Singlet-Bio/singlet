"""
singlet-gpu/bench/refs/lognorm_ref.py
SPDX-License-Identifier: MIT

SOTA reference for preprocess/lognorm benchmark.
Measures scanpy.pp.normalize_total(target_sum=1e4) + scanpy.pp.log1p().

Usage: python3 lognorm_ref.py <sample_dir>
Emits: SOTA_WALL_SEC=... SOTA_MEM_MB=... SOTA_CELLS=...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import time_function, emit_result, find_sample_dir


def _lognorm(sample_dir: Path):
    import scanpy as sc
    import numpy as np

    # Load a fresh copy each call (time_function includes warmup, so this
    # is intentional: we measure load + normalize together — but the
    # C++ driver measures just the normalize step. The ref thus includes
    # data loading overhead; this OVER-estimates SOTA wall time, making
    # ratio_wall conservative (lower bound on our speedup).
    adata = sc.read_10x_mtx(str(sample_dir), cache=False, gex_only=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: lognorm_ref.py <sample_dir>", file=sys.stderr)
        sys.exit(1)

    sample_dir = find_sample_dir(sys.argv[1])
    if sample_dir is None:
        emit_result(-1.0, -1.0, 0)
        return

    try:
        wall_sec, mem_mb, result = time_function(_lognorm, sample_dir)
        n_cells = result.n_obs if hasattr(result, "n_obs") else 0
        emit_result(wall_sec, mem_mb, n_cells)
    except ImportError as e:
        print(f"[lognorm_ref] missing dependency: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)
    except Exception as e:
        print(f"[lognorm_ref] error: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)


if __name__ == "__main__":
    main()
