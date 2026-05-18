#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/stagate_pyg_reference.py
#
# Reference subprocess for Stagate_TinySynthetic_VsPyG and
# Stagate_DLPFC_VsBenchmark tests.
#
# Loads a CSC count matrix (dump_csc.h binary format) and spatial coordinates
# (XYY\0 binary format written by spatial_stagate_correctness.cpp).
# Runs STAGATE_pyG train_STAGATE, then dumps:
#   spot_embedding  float32 [n_spots, d_embed]
#   spatial_domain  int32   [n_spots]
# to an .npz file at the path given by --out.
#
# Exit codes:
#   0  success
#   1  runtime error (data problem, STAGATE_pyG bug, etc.)
#   2  STAGATE_pyG (or a required dependency) is not installed — GTEST_SKIP
#
# Install (heavy): pip install STAGATE_pyG torch torch_geometric

import argparse
import struct
import sys

# ---------------------------------------------------------------------------
# Availability guard — exit 2 if packages are missing.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import anndata as ad
    import scanpy as sc
    import torch
except ImportError as e:
    print(f"[stagate_pyg_reference] Missing dependency: {e}. Skipping.", file=sys.stderr)
    sys.exit(2)

try:
    import STAGATE_pyG as STAGATE
except ImportError as e:
    print(f"[stagate_pyg_reference] STAGATE_pyG not installed: {e}. Skipping.", file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Binary readers (matching the C++ writers in spatial_stagate_correctness.cpp).
# ---------------------------------------------------------------------------

CSC_MAGIC   = 0x43535343  # "CSCC"
COORD_MAGIC = 0x58595900  # "XYY\0"


def read_csc_bin(path: str):
    """Read a dump_csc.h binary file.
    Format: [uint32 magic][uint32 m][uint32 n][uint64 nnz]
            [float32 * nnz values][int32 * (n+1) indptr][int32 * nnz indices]
    Returns (m, n, nnz, values, indptr, indices) as numpy arrays.
    """
    with open(path, "rb") as f:
        raw = f.read()
    offset = 0
    magic, = struct.unpack_from("<I", raw, offset); offset += 4
    if magic != CSC_MAGIC:
        raise ValueError(f"read_csc_bin: bad magic {magic:#010x} in {path}")
    m, n = struct.unpack_from("<II", raw, offset); offset += 8
    nnz, = struct.unpack_from("<Q", raw, offset); offset += 8
    values  = np.frombuffer(raw, dtype=np.float32, count=nnz,    offset=offset)
    offset += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,   count=n + 1,  offset=offset)
    offset += (n + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,   count=nnz,    offset=offset)
    return int(m), int(n), int(nnz), values.copy(), indptr.copy(), indices.copy()


def read_coords_bin(path: str):
    """Read the XYY\0 binary file.
    Format: [uint32 magic][uint32 n][float32 * n * 2 xy]
    Returns numpy array of shape (n, 2).
    """
    with open(path, "rb") as f:
        raw = f.read()
    magic, = struct.unpack_from("<I", raw, 0)
    if magic != COORD_MAGIC:
        raise ValueError(f"read_coords_bin: bad magic {magic:#010x} in {path}")
    n, = struct.unpack_from("<I", raw, 4)
    xy = np.frombuffer(raw, dtype=np.float32, count=n * 2, offset=8).reshape(n, 2)
    return xy.copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STAGATE_pyG reference subprocess for singlet-gpu tests")
    parser.add_argument("--counts",     required=True,
                        help="CSC counts binary (genes × spots)")
    parser.add_argument("--coords",     required=True,
                        help="XYY coords binary (spots × 2)")
    parser.add_argument("--out",        required=True,
                        help="Output .npz path")
    parser.add_argument("--n-domains",  type=int, default=4,
                        help="Number of spatial domains for Leiden resolution tuning")
    parser.add_argument("--k-neighbors", type=int, default=6,
                        help="Number of spatial nearest neighbours")
    parser.add_argument("--n-epochs",   type=int, default=500,
                        help="Number of STAGATE training epochs")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for torch and numpy")
    args = parser.parse_args()

    # Reproducibility.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data.
    m, n, nnz, values, indptr, indices = read_csc_bin(args.counts)
    xy = read_coords_bin(args.coords)

    # Build scipy CSC → dense (small test sizes; OK for reference).
    from scipy.sparse import csc_matrix
    X_csc = csc_matrix((values, indices, indptr), shape=(m, n))
    X_dense = np.array(X_csc.T.todense(), dtype=np.float32)  # cells × genes

    # Build AnnData.
    adata = ad.AnnData(X=X_dense)
    adata.obsm["spatial"] = xy.astype(np.float32)

    # Pre-processing (mimicking STAGATE_pyG tutorial).
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(m, 3000),
        subset=True,
        flavor="seurat_v3",
    )

    # Build spatial graph.
    STAGATE.Cal_Spatial_Net(adata, k_cutoff=args.k_neighbors, model="KNN")

    # Train STAGATE.
    adata = STAGATE.train_STAGATE(
        adata,
        n_epochs=args.n_epochs,
        random_seed=args.seed,
    )

    # Cluster latent embedding with Leiden to get spatial domains.
    sc.pp.neighbors(adata, use_rep="STAGATE")
    # Tune resolution to get approximately n_domains clusters.
    target = args.n_domains
    for resolution in [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        sc.tl.leiden(adata, key_added="STAGATE_Leiden",
                     resolution=resolution, random_state=args.seed)
        n_clust = adata.obs["STAGATE_Leiden"].nunique()
        if n_clust >= target:
            break

    spatial_domain = adata.obs["STAGATE_Leiden"].astype(int).values.astype(np.int32)
    embedding      = adata.obsm["STAGATE"].astype(np.float32)

    # Save results.
    np.savez(
        args.out,
        spot_embedding  = embedding,
        spatial_domain  = spatial_domain,
    )
    print(f"[stagate_pyg_reference] Written {args.out} "
          f"(n_spots={n}, d_embed={embedding.shape[1]}, "
          f"n_domains={len(set(spatial_domain.tolist()))})",
          file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[stagate_pyg_reference] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
