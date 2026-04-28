#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/deviance_hvg_numpy_reference.py
#
# Closed-form numpy/scipy reference for scry binomial/Poisson deviance HVG.
#
# Formula (Townes 2019, Eq. 4 — binomial):
#   pi_g  = sum_c(y_gc) / sum_c(n_c)         global gene proportion
#   n_c   = sum_g(y_gc)                       per-cell total count
#   d(y,n,pi) = 2 * [y*log(y/(n*pi)) + (n-y)*log((n-y)/(n*(1-pi)))]
#   D_g   = sum_c d(y_gc, n_c, pi_g)
#   with 0*log(0) = 0 convention.
#
# Formula (Poisson, for use_poisson=True):
#   lambda_g = sum_c(y_gc) / sum_c(n_c) * median(n_c)  (or mean? scry uses GLM)
#   For the approximation comparison in Test 4, use the simpler:
#   mu_gc = n_c * pi_g
#   d_poisson(y,mu) = 2 * [y*log(y/mu) - (y - mu)]
#   D_g = sum_c d_poisson(y_gc, mu_gc)
#   with 0*log(0) = 0 convention.
#
# Usage:
#   python3 deviance_hvg_numpy_reference.py \
#       --input  <binary_csc>  \
#       --output <output_csv>  \
#       [--use-poisson]        \
#       [--top-n 2000]
#
# Input binary format (same as dump_csc.h):
#   [0]  uint32  magic = 0x43535343 ("CSCC")
#   [4]  uint32  n_rows (genes)
#   [8]  uint32  n_cols (cells)
#   [12] uint64  nnz
#   [20] float32[nnz]    values
#   [20+4*nnz]  int32[n_cols+1]  indptr
#   [20+4*nnz+4*(n_cols+1)]  int32[nnz]  indices
#
# Output CSV format (0-indexed gene index, deviance value):
#   gene_idx,deviance
#   0,1234.56
#   ...
#
# If --top-n N is given, also appends a section:
#   ---top_n---
#   gene_idx,rank
#   (top-N genes sorted by descending deviance, 0-indexed rank)

import argparse
import struct
import sys

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Binary CSC reader (matches dump_csc.h / write_csc_bin in test harness)
# ---------------------------------------------------------------------------

def read_csc_bin(path: str) -> sp.csc_matrix:
    with open(path, "rb") as f:
        magic, n_rows, n_cols = struct.unpack("<III", f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic 0x{magic:08X}, expected 0x43535343")
        nnz = struct.unpack("<Q", f.read(8))[0]

        values  = np.frombuffer(f.read(nnz * 4),             dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4),    dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),             dtype=np.int32).copy()

    return sp.csc_matrix((values, indices, indptr), shape=(n_rows, n_cols))


# ---------------------------------------------------------------------------
# Deviance computation (vectorized, numerically stable)
# ---------------------------------------------------------------------------

def _xlogy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x * log(y) with 0 * log(0) = 0."""
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(x == 0, 0.0, x * np.log(y))
    return out


def binomial_deviance(csc: sp.csc_matrix) -> np.ndarray:
    """
    Compute per-gene binomial deviance (Townes 2019).

    Returns float64 array of shape (n_genes,).
    """
    csc = csc.astype(np.float64)
    n_genes, n_cells = csc.shape

    # n_c: per-cell total count  [n_cells]
    n_c = np.asarray(csc.sum(axis=0)).ravel()  # sum over genes

    # pi_g: per-gene global proportion  [n_genes]
    total_per_gene = np.asarray(csc.sum(axis=1)).ravel()  # sum over cells
    total_counts   = n_c.sum()
    pi_g = total_per_gene / total_counts  # shape [n_genes]

    deviance = np.zeros(n_genes, dtype=np.float64)

    # Process column-by-column (CSC: efficient column access)
    for j in range(n_cells):
        col_start = csc.indptr[j]
        col_end   = csc.indptr[j + 1]
        if col_start == col_end:
            continue  # empty cell

        row_idx = csc.indices[col_start:col_end]  # nonzero gene indices
        y       = csc.data[col_start:col_end]      # counts

        nc  = n_c[j]
        pi  = pi_g[row_idx]

        # d(y, n, pi) = 2 * [y*log(y / (n*pi)) + (n-y)*log((n-y) / (n*(1-pi)))]
        mu      = nc * pi                          # expected counts = n_c * pi_g
        y_comp  = nc - y                           # n - y
        mu_comp = nc * (1.0 - pi)                 # n * (1 - pi)

        contrib = 2.0 * (_xlogy(y, y / mu) + _xlogy(y_comp, y_comp / mu_comp))
        np.add.at(deviance, row_idx, contrib)

        # Cells where y==0 contribute only through the (n-y) term:
        # For zero-count genes in this cell, y_comp = n_c, mu_comp = n_c*(1-pi)
        # The loop above only touches nonzero rows; handle zero-count rows separately.
        # (All zero rows contribute: 2*(n_c)*log(1/(1-pi_g)) per cell)
        # This is the "background" deviance for absent genes.
        # We add it for all genes and subtract what was double-counted for nonzero rows.
        # Background for all genes in cell j:
        pi_all   = pi_g               # shape [n_genes]
        mu_comp_all = nc * (1.0 - pi_all)
        bg_all   = 2.0 * _xlogy(nc, nc / mu_comp_all)  # n_c * log(n_c / n_c*(1-pi))
        # This is the zero-count term for EVERY gene.
        deviance += bg_all

        # Subtract zero-count term for the nonzero genes (already computed differently):
        bg_nonzero = 2.0 * _xlogy(nc, nc / mu_comp)
        np.subtract.at(deviance, row_idx, bg_nonzero)

    return deviance


def poisson_deviance(csc: sp.csc_matrix) -> np.ndarray:
    """
    Compute per-gene Poisson deviance.

    mu_gc = n_c * pi_g  (same expected value as binomial)
    d_poisson(y, mu) = 2 * [y*log(y/mu) - (y - mu)]
    D_g = sum_c d_poisson(y_gc, mu_gc)

    Returns float64 array of shape (n_genes,).
    """
    csc = csc.astype(np.float64)
    n_genes, n_cells = csc.shape

    n_c            = np.asarray(csc.sum(axis=0)).ravel()
    total_per_gene = np.asarray(csc.sum(axis=1)).ravel()
    total_counts   = n_c.sum()
    pi_g           = total_per_gene / total_counts  # [n_genes]

    deviance = np.zeros(n_genes, dtype=np.float64)

    # For Poisson: the null contribution from zero-count cells is:
    # d_poisson(0, mu) = 2 * [0 - (0 - mu)] = 2 * mu = 2 * n_c * pi_g
    # Sum over all cells: 2 * sum(n_c) * pi_g = 2 * total_counts * pi_g
    # We first add this background for all genes, then correct for nonzero cells.

    bg_per_gene = 2.0 * total_counts * pi_g  # shape [n_genes]
    deviance   += bg_per_gene

    for j in range(n_cells):
        col_start = csc.indptr[j]
        col_end   = csc.indptr[j + 1]
        if col_start == col_end:
            continue

        row_idx = csc.indices[col_start:col_end]
        y       = csc.data[col_start:col_end]

        nc  = n_c[j]
        pi  = pi_g[row_idx]
        mu  = nc * pi

        # Actual d_poisson(y, mu):
        contrib_actual = 2.0 * (_xlogy(y, y / mu) - (y - mu))

        # Background (zero-count term) that was added for these nonzero genes:
        contrib_bg = 2.0 * mu  # 2*(n_c*pi_g) per gene per cell

        # Correct: subtract bg, add actual
        np.add.at(deviance, row_idx, contrib_actual - contrib_bg)

    return deviance


# ---------------------------------------------------------------------------
# Matrix Market writer for R scry bridge
# ---------------------------------------------------------------------------

def write_mtx(csc: sp.csc_matrix, path: str) -> None:
    """Write sparse matrix in Matrix Market format (for R scry_ref.R)."""
    coo = csc.tocoo()
    with open(path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"{csc.shape[0]} {csc.shape[1]} {coo.nnz}\n")
        for r, c, v in zip(coo.row + 1, coo.col + 1, coo.data):
            f.write(f"{r} {c} {v}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deviance HVG numpy reference")
    parser.add_argument("--input",       required=True, help="Binary CSC input")
    parser.add_argument("--output",      required=True, help="CSV output path")
    parser.add_argument("--use-poisson", action="store_true", default=False)
    parser.add_argument("--top-n",       type=int, default=0,
                        help="If >0, append top-N ranking section to CSV")
    parser.add_argument("--write-mtx",   default="",
                        help="If given, also write MTX for R scry reference")
    args = parser.parse_args()

    csc = read_csc_bin(args.input)
    n_genes, n_cells = csc.shape
    print(f"[numpy_ref] matrix: {n_genes} genes x {n_cells} cells, nnz={csc.nnz}")

    if args.use_poisson:
        dev = poisson_deviance(csc)
        fam = "poisson"
    else:
        dev = binomial_deviance(csc)
        fam = "binomial"

    print(f"[numpy_ref] fam={fam}, min={dev.min():.6f}, max={dev.max():.6f}, "
          f"mean={dev.mean():.6f}")

    # Write CSV
    with open(args.output, "w") as f:
        f.write("gene_idx,deviance\n")
        for i, d in enumerate(dev):
            f.write(f"{i},{d:.10f}\n")

    if args.top_n > 0:
        ranked = np.argsort(-dev)[:args.top_n]
        with open(args.output, "a") as f:
            f.write("---top_n---\n")
            f.write("gene_idx,rank\n")
            for rank, gi in enumerate(ranked):
                f.write(f"{int(gi)},{rank}\n")

    if args.write_mtx:
        write_mtx(csc, args.write_mtx)
        print(f"[numpy_ref] wrote MTX to: {args.write_mtx}")

    print(f"[numpy_ref] wrote {n_genes} deviance values to: {args.output}")


if __name__ == "__main__":
    main()
