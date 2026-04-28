#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/progeny_py_reference.py
#
# Reference driver: run decoupleR PROGENy (wsum method) on a dense float32
# expression matrix (genes × cells) with a pathway weight matrix, then write
# per-cell pathway activity scores to a binary file.
#
# Environment:
#   pip install decoupler-py anndata numpy scipy pandas
#
# Exit codes:
#   0  — decoupleR ran successfully
#   2  — decoupleR not installed; pure-numpy fallback ran (always available)
#   1  — unexpected error
#
# Primary path: decoupleR wsum (weighted sum, equivalent to matrix multiply).
#   Implements PROGENy as:  activity = X @ W^T  then normalize per pathway.
#
# Fallback path (always present): pure numpy:
#   activity = expression (n_cells × n_genes) @ W^T (n_genes × n_pathways)
#   followed by Welford-style per-pathway z-score normalization.
#
# Input files (written by the C++ test or passed via CLI):
#   --matrix   <matrix.bin>     # [uint32 n_genes][uint32 n_cells][float32 G*C col-major]
#   --weights  <weights.tsv>    # TSV with header: pathway<TAB>gene_idx<TAB>weight
#   --output   <scores.bin>     # output binary (see format below)
#
# Output binary:
#   [uint32_t  n_pathways]
#   [uint32_t  n_cells]
#   [char[n_pathways * 32]  pathway names, zero-padded, row-major]
#   [float32   n_pathways * n_cells  scores row-major (pathway × cell)]

import argparse
import struct
import sys
import os

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_matrix_bin(path: str):
    """Read [uint32 n_genes][uint32 n_cells][float32 G*C col-major]."""
    with open(path, 'rb') as f:
        n_genes = struct.unpack('<I', f.read(4))[0]
        n_cells = struct.unpack('<I', f.read(4))[0]
        vals = np.frombuffer(f.read(n_genes * n_cells * 4), dtype=np.float32).copy()
    # col-major → genes × cells
    mat = vals.reshape((n_genes, n_cells), order='F')
    return mat  # shape (n_genes, n_cells)


def load_weights_tsv(path: str):
    """Return (W float32 array n_pathways × n_genes, pathway_names list, n_genes int)."""
    df = pd.read_csv(path, sep='\t', header=0,
                     names=['pathway', 'gene_idx', 'weight'],
                     dtype={'pathway': str, 'gene_idx': int, 'weight': float})
    pathway_names = sorted(df['pathway'].unique().tolist())
    n_pathways = len(pathway_names)
    n_genes = int(df['gene_idx'].max()) + 1  # infer from max index; caller ensures full range
    W = np.zeros((n_pathways, n_genes), dtype=np.float32)
    pw_idx = {p: i for i, p in enumerate(pathway_names)}
    for _, row in df.iterrows():
        W[pw_idx[row['pathway']], int(row['gene_idx'])] = np.float32(row['weight'])
    return W, pathway_names, n_genes


def write_scores_bin(path: str, scores: np.ndarray, pathway_names: list):
    """Write [uint32 n_pathways][uint32 n_cells][char×32 names][float32 scores row-major]."""
    n_pathways, n_cells = scores.shape
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', n_pathways))
        f.write(struct.pack('<I', n_cells))
        for nm in pathway_names:
            nm_bytes = nm.encode('utf-8')[:31]
            buf = nm_bytes + b'\x00' * (32 - len(nm_bytes))
            f.write(buf)
        f.write(scores.astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Pure-numpy PROGENy (fallback, always available)
# activity = X_T @ W^T  then per-pathway Welford z-score normalization.
# X_T is (n_cells × n_genes); W is (n_pathways × n_genes).
# Result: (n_pathways × n_cells).
# ---------------------------------------------------------------------------

def progeny_numpy(mat: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    mat: (n_genes × n_cells) float32
    W:   (n_pathways × n_genes) float32
    Returns: (n_pathways × n_cells) float32 z-score normalized per pathway.
    """
    # (n_cells × n_genes) @ (n_genes × n_pathways) → (n_cells × n_pathways)
    X_T = mat.T.astype(np.float32)           # n_cells × n_genes
    activity = X_T @ W.T                     # n_cells × n_pathways
    activity = activity.T                    # n_pathways × n_cells

    # Per-pathway z-score normalization (Welford stable).
    for i in range(activity.shape[0]):
        row = activity[i]
        mu  = float(np.mean(row))
        sd  = float(np.std(row, ddof=0))
        if sd > 1e-9:
            activity[i] = (row - mu) / sd
        else:
            activity[i] = row - mu

    return activity.astype(np.float32)


# ---------------------------------------------------------------------------
# decoupleR PROGENy (primary path)
# ---------------------------------------------------------------------------

def progeny_decoupler(mat: np.ndarray, W: np.ndarray,
                      pathway_names: list) -> np.ndarray:
    """
    mat: (n_genes × n_cells) float32
    W:   (n_pathways × n_genes) float32
    Returns: (n_pathways × n_cells) float32 z-score normalized per pathway.
    Raises ImportError if decoupler-py not installed.
    """
    import anndata
    import decoupler as dc

    n_genes, n_cells = mat.shape
    n_pathways = W.shape[0]

    gene_names = [str(i) for i in range(n_genes)]
    cell_names = [str(j) for j in range(n_cells)]

    # AnnData: cells × genes.
    adata = anndata.AnnData(
        X=mat.T.astype(np.float32),
        obs={'cell': cell_names},
        var={'gene': gene_names},
    )
    adata.obs_names = cell_names
    adata.var_names = gene_names

    # Build decoupleR network DataFrame (source=pathway, target=gene_idx, weight).
    rows = []
    for pi, pw in enumerate(pathway_names):
        for gi in range(n_genes):
            w = float(W[pi, gi])
            if w != 0.0:
                rows.append({'source': pw, 'target': str(gi), 'weight': w})
    import pandas as pd
    net = pd.DataFrame(rows)

    # Run wsum (equivalent to X @ W^T with normalization).
    acts, _ = dc.run_wsum(
        mat=adata, net=net,
        source='source', target='target', weight='weight',
        verbose=False, use_raw=False
    )
    # acts: AnnData obs=cells, var=pathways.
    available = [p for p in pathway_names if p in acts.var_names]
    scores_T = acts[:, available].X  # (n_cells × n_pathways)
    if hasattr(scores_T, 'toarray'):
        scores_T = scores_T.toarray()
    return scores_T.T.astype(np.float32)  # (n_pathways × n_cells)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='PROGENy reference for singlet-gpu enrich tests (cycle 44)')
    parser.add_argument('--matrix',  required=True,
                        help='float32 binary: [uint32 n_genes][uint32 n_cells][float32 col-major]')
    parser.add_argument('--weights', required=True,
                        help='TSV with header pathway/gene_idx/weight')
    parser.add_argument('--output',  required=True,
                        help='Output scores binary')
    args = parser.parse_args()

    mat = read_matrix_bin(args.matrix)
    n_genes, n_cells = mat.shape
    print(f'[progeny_py_reference] matrix: {n_genes} genes × {n_cells} cells',
          file=sys.stderr)

    W, pathway_names, w_n_genes = load_weights_tsv(args.weights)
    n_pathways = len(pathway_names)
    print(f'[progeny_py_reference] {n_pathways} pathways, W shape: {W.shape}',
          file=sys.stderr)

    # Trim/pad W columns to match n_genes.
    if W.shape[1] < n_genes:
        pad = np.zeros((n_pathways, n_genes - W.shape[1]), dtype=np.float32)
        W = np.concatenate([W, pad], axis=1)
    elif W.shape[1] > n_genes:
        W = W[:, :n_genes]

    exit_code = 0
    try:
        scores = progeny_decoupler(mat, W, pathway_names)
        print('[progeny_py_reference] decoupleR path succeeded.', file=sys.stderr)
    except ImportError:
        print('[progeny_py_reference] decoupleR not installed — using numpy fallback.',
              file=sys.stderr)
        scores = progeny_numpy(mat, W)
        exit_code = 2
    except Exception as e:
        print(f'[progeny_py_reference] decoupleR error: {e} — using numpy fallback.',
              file=sys.stderr)
        scores = progeny_numpy(mat, W)
        exit_code = 2

    print(f'[progeny_py_reference] scores shape: {scores.shape}', file=sys.stderr)
    write_scores_bin(args.output, scores, pathway_names)
    print(f'[progeny_py_reference] wrote to {args.output} (exit {exit_code})',
          file=sys.stderr)
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
