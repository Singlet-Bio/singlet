#!/usr/bin/env python3
"""
singlet-gpu/bench/refs/cycle57_convert.py
SPDX-License-Identifier: GPL-2.0-or-later

One-time format conversion for Cycle 57 baselines.
Reads a .1pz file and writes .h5 (10x HDF5) and .h5ad (AnnData) formats.
These conversions are NOT timed — they are setup work for the benchmark.

Usage:
    python3 cycle57_convert.py --pz=<path.1pz> --h5=<out.h5> --h5ad=<out.h5ad>
"""

import argparse
import sys
import time
from pathlib import Path

# Allow import from sibling + tests/refs directories.
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests" / "refs"))


def load_pz_as_scipy(pz_path: str):
    """Load a .1pz file as a scipy CSC matrix using pz_reference_reader."""
    try:
        from pz_reference_reader import read_pz_to_csc
        mat = read_pz_to_csc(pz_path)
        return mat
    except ImportError as e:
        print(f"[convert] pz_reference_reader not importable: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[convert] failed to load {pz_path}: {e}", file=sys.stderr)
        return None


def write_10x_h5(mat, gene_names, barcode_names, out_path: str) -> bool:
    """Write a scipy CSC as a 10x CellRanger HDF5 (minimal format for scanpy)."""
    try:
        import h5py
        import numpy as np

        n_genes, n_cells = mat.shape
        if gene_names is None or len(gene_names) != n_genes:
            gene_names = [f"GENE_{i}" for i in range(n_genes)]
        if barcode_names is None or len(barcode_names) != n_cells:
            barcode_names = [f"CELL_{i}-1" for i in range(n_cells)]

        # CellRanger v3 format expected by scanpy.read_10x_h5():
        #   /matrix/data, /matrix/indices, /matrix/indptr (CSC)
        #   /matrix/shape  (DATASET [n_genes, n_cells], NOT an attribute)
        #   /matrix/barcodes
        #   /matrix/features/id, name, feature_type
        with h5py.File(out_path, "w") as f:
            grp = f.create_group("matrix")
            grp.create_dataset("data",    data=mat.data.astype(np.float32))
            grp.create_dataset("indices", data=mat.indices.astype(np.int32))
            grp.create_dataset("indptr",  data=mat.indptr.astype(np.int32))
            # shape MUST be a dataset (int64), not an attribute — scanpy KeyError if missing
            grp.create_dataset("shape",   data=np.array([n_genes, n_cells], dtype=np.int64))
            feat = grp.create_group("features")
            feat.create_dataset("id",           data=np.array(gene_names,   dtype="S"))
            feat.create_dataset("name",         data=np.array(gene_names,   dtype="S"))
            feat.create_dataset("feature_type", data=np.array(
                [b"Gene Expression"] * n_genes))
            grp.create_dataset("barcodes", data=np.array(barcode_names, dtype="S"))
        print(f"[convert] wrote 10x h5: {out_path}")
        return True
    except ImportError as e:
        print(f"[convert] h5py not available: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[convert] h5 write failed: {e}", file=sys.stderr)
        return False


def write_h5ad(mat, gene_names, barcode_names, out_path: str) -> bool:
    """Write a scipy CSC as an AnnData .h5ad file."""
    try:
        import anndata as ad
        import numpy as np
        import scipy.sparse as sp

        n_genes, n_cells = mat.shape
        if gene_names is None or len(gene_names) != n_genes:
            gene_names = [f"GENE_{i}" for i in range(n_genes)]
        if barcode_names is None or len(barcode_names) != n_cells:
            barcode_names = [f"CELL_{i}-1" for i in range(n_cells)]

        # AnnData uses obs=cells, var=genes → transpose.
        mat_csr = mat.T.tocsr().astype(np.float32)
        adata = ad.AnnData(
            X=mat_csr,
            obs={"barcode": barcode_names},
            var={"gene_id": gene_names, "gene_name": gene_names},
        )
        adata.write_h5ad(out_path)
        print(f"[convert] wrote h5ad: {out_path}")
        return True
    except ImportError as e:
        print(f"[convert] anndata not available: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[convert] h5ad write failed: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert .1pz to .h5 and .h5ad")
    parser.add_argument("--pz",   required=True, help="Input .1pz path")
    parser.add_argument("--h5",   required=True, help="Output 10x .h5 path")
    parser.add_argument("--h5ad", required=True, help="Output .h5ad path")
    args = parser.parse_args()

    pz_path   = args.pz
    h5_path   = args.h5
    h5ad_path = args.h5ad

    # Skip if already converted.
    h5_exists   = Path(h5_path).exists()
    h5ad_exists = Path(h5ad_path).exists()
    if h5_exists and h5ad_exists:
        print(f"[convert] both output files exist — skipping conversion.")
        return

    print(f"[convert] loading {pz_path} ...", flush=True)
    t0 = time.time()
    mat = load_pz_as_scipy(pz_path)
    if mat is None:
        print("[convert] SKIP: cannot load .1pz", file=sys.stderr)
        sys.exit(1)
    print(f"[convert] loaded {mat.shape} nnz={mat.nnz} in {time.time()-t0:.1f}s")

    # Try to get gene/barcode names from pz_reference_reader.
    gene_names    = None
    barcode_names = None
    try:
        from pz_reference_reader import read_pz_metadata
        meta = read_pz_metadata(pz_path)
        gene_names    = meta.get("rownames")
        barcode_names = meta.get("colnames")
    except Exception:
        pass

    if not h5_exists:
        write_10x_h5(mat, gene_names, barcode_names, h5_path)
    if not h5ad_exists:
        write_h5ad(mat, gene_names, barcode_names, h5ad_path)


if __name__ == "__main__":
    main()
