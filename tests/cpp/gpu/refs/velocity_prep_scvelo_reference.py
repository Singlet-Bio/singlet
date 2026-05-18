#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/velocity_prep_scvelo_reference.py
#
# Reference driver: load spliced + unspliced CSC .bin files (dump_csc.h format),
# run scvelo.pp.moments + scvelo.tl.velocity(mode='steady_state'), then write
# per-gene gamma, smoothed moments (S_smoothed, U_smoothed), and velocity matrix
# to a .npz file for comparison by the C++ correctness harness.
#
# Environment: pip install scvelo scanpy anndata numpy scipy
#
# Usage:
#   python velocity_prep_scvelo_reference.py \
#       --spliced   <spliced.bin>        \  # dump_csc.h binary: genes x cells float32
#       --unspliced <unspliced.bin>      \  # dump_csc.h binary: genes x cells float32
#       --out       <reference.npz>      \  # numpy .npz with keys below
#       [--n-neighbors <int>]            \  # default: 30 (scVelo default)
#       [--min-shared-counts <int>]      \  # default: 30 (scVelo default for moments)
#       [--n-pcs <int>]                  \  # default: 30 (for kNN in moments)
#
# Output .npz keys:
#   gamma          float32 (n_genes,)       — per-gene gamma (0 for filtered genes)
#   filter_mask    int32   (n_genes,)       — 1 = passes filter, 0 = filtered out
#   Ms             float32 (n_genes, n_cells) — smoothed spliced moments
#   Mu             float32 (n_genes, n_cells) — smoothed unspliced moments
#   velocity       float32 (n_genes, n_cells) — velocity matrix (NaN for filtered genes)
#   n_genes        int scalar               — total number of genes (rows)
#   n_cells        int scalar               — total number of cells (cols)
#   n_genes_passing int scalar              — number of genes passing filter
#
# The C++ test writes synthetic or real data via dump_csc.h format, calls this
# script as a subprocess, then loads the .npz and compares to our GPU output.

import argparse
import struct
import sys
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Binary reader (matches dump_csc.h format exactly)
# ---------------------------------------------------------------------------
def read_csc_bin(path: str):
    """
    Read the compact binary produced by tests/refs/dump_csc.h.
    Returns (scipy.sparse.csc_matrix, n_rows, n_cols).
    Format:
      uint32 magic=0x43535343, uint32 n_rows, uint32 n_cols,
      uint64 nnz, float32[nnz] values, int32[n_cols+1] indptr, int32[nnz] indices
    """
    with open(path, 'rb') as f:
        raw = f.read()
    offset = 0
    magic, n_rows, n_cols = struct.unpack_from('<III', raw, offset)
    offset += 12
    if magic != 0x43535343:
        raise ValueError(f"Bad magic: 0x{magic:08x} (expected 0x43535343)")
    (nnz,) = struct.unpack_from('<Q', raw, offset)
    offset += 8
    values  = np.frombuffer(raw, dtype=np.float32,
                            count=nnz, offset=offset).copy()
    offset += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,
                            count=n_cols + 1, offset=offset).copy()
    offset += (n_cols + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,
                            count=nnz, offset=offset).copy()
    mat = sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))
    return mat, n_rows, n_cols


# ---------------------------------------------------------------------------
# Main reference computation
# ---------------------------------------------------------------------------
def run_velocity_reference(
    spliced_csc: sp.csc_matrix,
    unspliced_csc: sp.csc_matrix,
    n_neighbors: int = 30,
    min_shared_counts: int = 30,
    n_pcs: int = 30,
):
    """
    Run scVelo steady-state velocity on spliced + unspliced matrices.

    Input matrices are (genes x cells) CSC in our convention.
    scVelo AnnData convention is (cells x genes), so we transpose.

    Returns a dict with arrays:
      gamma, filter_mask, Ms, Mu, velocity
    """
    try:
        import scvelo as scv
    except ImportError:
        print("[ref] ERROR: scvelo not installed. pip install scvelo", file=sys.stderr)
        raise
    try:
        import anndata as ad
        import scanpy as sc
    except ImportError:
        print("[ref] ERROR: anndata/scanpy not installed.", file=sys.stderr)
        raise

    n_genes, n_cells = spliced_csc.shape
    print(f"[ref] spliced: {n_genes} genes x {n_cells} cells  nnz={spliced_csc.nnz}",
          file=sys.stderr)
    print(f"[ref] unspliced: {unspliced_csc.shape}  nnz={unspliced_csc.nnz}",
          file=sys.stderr)

    # AnnData: rows = cells, cols = genes.
    # scVelo stores spliced as adata.X (or layers['spliced']), unspliced as layers['unspliced'].
    spliced_t   = spliced_csc.T.tocsr().astype(np.float32)    # (cells, genes)
    unspliced_t = unspliced_csc.T.tocsr().astype(np.float32)  # (cells, genes)

    adata = ad.AnnData(X=spliced_t)
    adata.layers['spliced']   = spliced_t
    adata.layers['unspliced'] = unspliced_t

    print(f"[ref] AnnData: {adata.n_obs} cells x {adata.n_vars} genes", file=sys.stderr)

    # scvelo.pp.filter_and_normalize handles gene filtering + normalization.
    # We use min_shared_counts to match what scVelo uses internally.
    # enforce_hvgs=False to keep all genes (let filter_mask reflect min_shared_counts only).
    scv.pp.filter_and_normalize(
        adata,
        min_shared_counts=min_shared_counts,
        n_top_genes=None,      # keep all genes that pass min_shared_counts
        enforce_hvgs=False,
        log=False,             # don't log-transform — velocity works on raw counts
    )
    print(f"[ref] After filter: {adata.n_obs} cells x {adata.n_vars} genes",
          file=sys.stderr)

    # Compute PCA + kNN for moments smoothing.
    sc.pp.pca(adata, n_comps=min(n_pcs, min(adata.n_obs, adata.n_vars) - 1),
              random_state=0)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_pca',
                    random_state=0)

    # Compute kNN-smoothed moments (Ms, Mu).
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)

    # Steady-state velocity: fits gamma per gene from top-quantile cells.
    scv.tl.velocity(adata, mode='steady_state')

    # Extract results — all in scVelo's (cells, genes) convention;
    # we transpose back to (genes, cells) to match our GPU kernel output.
    var_names = list(adata.var_names)
    n_genes_passing = len(var_names)
    print(f"[ref] Genes passing filter: {n_genes_passing}", file=sys.stderr)

    # gamma is stored in adata.var['velocity_gamma'] after scv.tl.velocity
    gamma_passing = adata.var['velocity_gamma'].values.astype(np.float32)  # (n_genes_passing,)

    # Ms, Mu are (cells, genes_passing) in AnnData convention.
    Ms_t = adata.layers['Ms'].astype(np.float32)   # (cells, genes_passing)
    Mu_t = adata.layers['Mu'].astype(np.float32)   # (cells, genes_passing)

    # Velocity is stored as (cells, genes_passing) sparse/dense.
    vel = adata.layers['velocity']
    if sp.issparse(vel):
        vel = vel.toarray()
    vel = vel.astype(np.float32)    # (cells, genes_passing)

    # Transpose to (genes, cells) to match GPU output convention.
    Ms  = Ms_t.T   # (genes_passing, cells)
    Mu  = Mu_t.T   # (genes_passing, cells)
    vel = vel.T    # (genes_passing, cells)

    print(f"[ref] gamma range: [{gamma_passing.min():.4f}, {gamma_passing.max():.4f}]",
          file=sys.stderr)
    print(f"[ref] Ms shape: {Ms.shape}  Mu shape: {Mu.shape}  vel shape: {vel.shape}",
          file=sys.stderr)

    # Build full-gene-count arrays (including filtered genes) for comparison.
    # Filtered genes get gamma=NaN, filter_mask=0.
    # We need to map var_names (strings) back to original gene indices.
    # scVelo uses integer index if no gene names given — var_names are '0','1','2',...
    passing_indices = np.array([int(v) for v in var_names], dtype=np.int32)
    filter_mask_full = np.zeros(n_genes, dtype=np.int32)
    filter_mask_full[passing_indices] = 1

    gamma_full = np.full(n_genes, np.nan, dtype=np.float32)
    gamma_full[passing_indices] = gamma_passing

    # Ms_full, Mu_full, vel_full: (n_genes, n_cells), NaN for filtered genes.
    Ms_full  = np.full((n_genes, n_cells), np.nan, dtype=np.float32)
    Mu_full  = np.full((n_genes, n_cells), np.nan, dtype=np.float32)
    vel_full = np.full((n_genes, n_cells), np.nan, dtype=np.float32)
    Ms_full [passing_indices, :]  = Ms
    Mu_full [passing_indices, :]  = Mu
    vel_full[passing_indices, :]  = vel

    return {
        'gamma':          gamma_full,
        'filter_mask':    filter_mask_full,
        'Ms':             Ms_full,
        'Mu':             Mu_full,
        'velocity':       vel_full,
        'n_genes':        np.int32(n_genes),
        'n_cells':        np.int32(n_cells),
        'n_genes_passing': np.int32(n_genes_passing),
    }


def main():
    parser = argparse.ArgumentParser(
        description="scVelo steady-state velocity reference for singlet-gpu cycle 15")
    parser.add_argument("--spliced",   required=True,
                        help="dump_csc.h binary for spliced (genes x cells) float32")
    parser.add_argument("--unspliced", required=True,
                        help="dump_csc.h binary for unspliced (genes x cells) float32")
    parser.add_argument("--out",       required=True,
                        help="Output .npz path for reference arrays")
    parser.add_argument("--n-neighbors",       type=int, default=30)
    parser.add_argument("--min-shared-counts", type=int, default=30)
    parser.add_argument("--n-pcs",             type=int, default=30)
    args = parser.parse_args()

    spliced_csc,   n_rows_s, n_cols_s = read_csc_bin(args.spliced)
    unspliced_csc, n_rows_u, n_cols_u = read_csc_bin(args.unspliced)

    if n_rows_s != n_rows_u or n_cols_s != n_cols_u:
        raise ValueError(
            f"Spliced shape ({n_rows_s},{n_cols_s}) != "
            f"unspliced shape ({n_rows_u},{n_cols_u})")

    result = run_velocity_reference(
        spliced_csc, unspliced_csc,
        n_neighbors=args.n_neighbors,
        min_shared_counts=args.min_shared_counts,
        n_pcs=args.n_pcs,
    )

    np.savez(args.out, **result)
    print(f"[ref] Saved reference output to {args.out}", file=sys.stderr)
    print(f"[ref] n_genes={result['n_genes']}  n_cells={result['n_cells']}  "
          f"n_genes_passing={result['n_genes_passing']}", file=sys.stderr)


if __name__ == "__main__":
    main()
