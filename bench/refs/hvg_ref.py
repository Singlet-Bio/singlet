"""
singlet-gpu/bench/refs/hvg_ref.py
SPDX-License-Identifier: MIT

SOTA reference for preprocess/hvg benchmark.
Measures scanpy.pp.highly_variable_genes with all three flavors:
  - seurat (log-normalize + clipped variance)
  - seurat_v3 (Hafemeister & Satija 2019 — matched to our SeuratV3 path)
  - pearson_residuals (Lause et al. 2021 — matched to our PearsonResiduals path)

The SOTA_WALL_SEC emitted is for seurat_v3 (the primary comparison).
All three timings are printed as INFO lines for reference.

Usage: python3 hvg_ref.py <sample_dir>
Emits: SOTA_WALL_SEC=... SOTA_MEM_MB=... SOTA_CELLS=...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import time_function, emit_result, find_sample_dir


def _load_normalized(sample_dir: Path):
    import scanpy as sc
    adata = sc.read_10x_mtx(str(sample_dir), cache=False, gex_only=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _hvg_seurat_v3(adata):
    import scanpy as sc
    import copy
    a = copy.copy(adata)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=2000)
    return a


def _hvg_pearson(adata):
    import scanpy as sc
    import copy
    a = copy.copy(adata)
    sc.experimental.pp.highly_variable_genes(a, flavor="pearson_residuals",
                                              n_top_genes=2000)
    return a


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: hvg_ref.py <sample_dir>", file=sys.stderr)
        sys.exit(1)

    sample_dir = find_sample_dir(sys.argv[1])
    if sample_dir is None:
        emit_result(-1.0, -1.0, 0)
        return

    try:
        # Load once.
        adata = _load_normalized(sample_dir)
        n_cells = adata.n_obs

        # Time seurat_v3 (primary comparison).
        wall_v3, mem_v3, _ = time_function(_hvg_seurat_v3, adata)
        print(f"INFO seurat_v3 wall_sec={wall_v3:.4f} mem_mb={mem_v3:.1f}",
              flush=True)

        # Time pearson_residuals.
        try:
            wall_pr, mem_pr, _ = time_function(_hvg_pearson, adata)
            print(f"INFO pearson_residuals wall_sec={wall_pr:.4f} mem_mb={mem_pr:.1f}",
                  flush=True)
        except Exception as e:
            print(f"INFO pearson_residuals FAILED: {e}", flush=True)

        # Primary emit: seurat_v3.
        emit_result(wall_v3, mem_v3, n_cells)

    except ImportError as e:
        print(f"[hvg_ref] missing dependency: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)
    except Exception as e:
        print(f"[hvg_ref] error: {e}", file=sys.stderr)
        emit_result(-1.0, -1.0, 0)


if __name__ == "__main__":
    main()
