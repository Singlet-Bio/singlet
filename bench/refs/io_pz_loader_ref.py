"""
singlet-gpu/bench/refs/io_pz_loader_ref.py
SPDX-License-Identifier: MIT

SOTA reference for io/pz_loader benchmark.
Measures scanpy.read_10x_mtx() on the standard 10x matrix output format.

The .1pz sample directory contains the count matrix data; we look for either:
  - matrix.mtx.gz  (Market Exchange format — standard 10x output)
  - barcodes.tsv.gz + features.tsv.gz alongside matrix.mtx.gz
  - As a fallback, scipy.io.mmread on any .mtx file in the directory.

Usage: python3 io_pz_loader_ref.py <sample_dir>
Emits: SOTA_WALL_SEC=... SOTA_MEM_MB=... SOTA_CELLS=...
"""

import sys
from pathlib import Path

# Allow import from sibling directory.
sys.path.insert(0, str(Path(__file__).parent))
from common import time_function, emit_result, find_sample_dir

def _load_10x(sample_dir: Path):
    """Attempt scanpy.read_10x_mtx; fallback to anndata.read_mtx."""
    mtx_path = sample_dir / "matrix.mtx.gz"
    if not mtx_path.exists():
        mtx_path = sample_dir / "matrix.mtx"
    if not mtx_path.exists():
        raise FileNotFoundError(f"No matrix.mtx(.gz) in {sample_dir}")

    try:
        import scanpy as sc
        adata = sc.read_10x_mtx(
            str(sample_dir),
            var_names="gene_ids",
            cache=False,
            gex_only=True,
        )
        return adata
    except Exception:
        # Fallback: scipy sparse read (covers any .mtx file).
        import scipy.io
        mat = scipy.io.mmread(str(mtx_path)).tocsc()
        return mat


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: io_pz_loader_ref.py <sample_dir>", file=sys.stderr)
        sys.exit(1)

    sample_dir = find_sample_dir(sys.argv[1])
    if sample_dir is None:
        print(f"Sample dir not found: {sys.argv[1]}", file=sys.stderr)
        # Emit placeholder — driver will treat as "not measured".
        emit_result(-1.0, -1.0, 0)
        return

    try:
        wall_sec, mem_mb, result = time_function(_load_10x, sample_dir)
        n_cells = result.shape[1] if hasattr(result, "shape") else 0
        if hasattr(result, "n_obs"):  # AnnData: cells are obs.
            n_cells = result.n_obs
        emit_result(wall_sec, mem_mb, n_cells)
    except ImportError as e:
        print(f"[io_pz_loader_ref] missing dependency: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)
    except Exception as e:
        print(f"[io_pz_loader_ref] error: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)


if __name__ == "__main__":
    main()
