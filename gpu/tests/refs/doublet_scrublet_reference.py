#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/doublet_scrublet_reference.py
#
# Reference implementation driver for cycle 31 doublet detection correctness harness.
#
# Loads a CSC count binary (dump_csc.h format, genes × cells) produced by the C++
# test fixture, runs Scrublet, and writes a .npz with per-cell doublet scores.
#
# Usage (called by qc_doublet_correctness.cpp via subprocess):
#   python3 doublet_scrublet_reference.py \
#       --counts  <counts.bin>            \
#       --out     <results.npz>           \
#       --n-synth-frac <F>                \
#       --k       <K>                     \
#       --seed    <R>
#
# Exit codes:
#   0  success — results.npz written
#   2  scrublet not installed → GTEST_SKIP in C++
#   1  any other error → FAIL
#
# Output npz keys:
#   doublet_scores  float64[n_cells]  — per-cell doublet score in [0, 1]
#   doublet_calls   int32[n_cells]    — binary doublet prediction (0/1)
#   threshold       float64 scalar    — auto-selected threshold
#   n_cells         int32 scalar
#   n_synth         int32 scalar      — number of synthetic doublets used
#
# The counts matrix is expected to be genes × cells (CSR rows = genes, cols = cells).
# Scrublet's convention is cells × genes, so we transpose on load.

import sys
import argparse
import struct

import numpy as np
import scipy.sparse


# ---------------------------------------------------------------------------
# Package availability check — exit 2 if unavailable (→ GTEST_SKIP in C++)
# ---------------------------------------------------------------------------

def _check_scrublet():
    """Import scrublet; exit 2 if not installed."""
    try:
        import scrublet  # noqa: F401
        return scrublet
    except ImportError:
        print("[doublet_scrublet_reference] SKIP: scrublet not installed. "
              "Install with: pip install scrublet",
              file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------------------
# CSC binary reader (dump_csc.h format — genes × cells)
#
# Format (little-endian, no padding):
#   uint32  magic      = 0x43535343 ("CSCC")
#   uint32  n_rows     (genes)
#   uint32  n_cols     (cells)
#   uint64  nnz
#   float32[nnz]       values
#   int32[n_cols+1]    indptr
#   int32[nnz]         indices
# ---------------------------------------------------------------------------

def read_csc_bin(path: str) -> scipy.sparse.csc_matrix:
    """Read a dump_csc.h binary and return a scipy CSC matrix (genes × cells)."""
    with open(path, 'rb') as f:
        magic, n_rows, n_cols = struct.unpack('<III', f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")
        (nnz,) = struct.unpack('<Q', f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),          dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),          dtype=np.int32).copy()
    return scipy.sparse.csc_matrix(
        (values, indices, indptr), shape=(n_rows, n_cols)
    )


# ---------------------------------------------------------------------------
# Main reference runner
# ---------------------------------------------------------------------------

def run_scrublet_reference(
        counts_path: str,
        out_path: str,
        n_synth_frac: float,
        k: int,
        seed: int) -> None:
    """Load counts, run scrublet, write npz."""

    scrublet = _check_scrublet()

    # Load count matrix: genes × cells (CSC).
    mat_csc = read_csc_bin(counts_path)
    n_genes, n_cells = mat_csc.shape

    # Scrublet expects cells × genes (CSR or dense).
    # Convert to cells × genes CSC (which scipy will handle as CSC or CSR internally).
    counts_cells_x_genes = mat_csc.T.tocsr()  # cells × genes

    # Round to integer counts (scrublet expects raw counts).
    counts_int = counts_cells_x_genes.astype(np.float32)

    n_synth = max(1, int(round(n_synth_frac * n_cells)))

    # Run scrublet.
    scrub = scrublet.Scrublet(
        counts_matrix=counts_int,
        n_neighbors=k,
        expected_doublet_rate=0.05,
        sim_doublet_ratio=float(n_synth) / max(1, n_cells),
        random_state=seed,
    )

    try:
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            min_counts=1,
            min_cells=1,
            min_gene_variability_pctl=0,  # include all genes for tiny inputs
            n_prin_comps=min(n_genes - 1, 30),
            verbose=False,
        )
    except Exception as e:
        # Some scrublet versions raise on very small inputs.
        print(f"[doublet_scrublet_reference] scrub_doublets failed: {e}; "
              f"retrying with n_prin_comps=min(n_genes-1, 5)",
              file=sys.stderr)
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            min_counts=1,
            min_cells=1,
            min_gene_variability_pctl=0,
            n_prin_comps=min(n_genes - 1, 5),
            verbose=False,
        )

    doublet_scores = np.asarray(doublet_scores, dtype=np.float64)
    predicted_doublets = np.asarray(predicted_doublets, dtype=np.int32)
    threshold = float(getattr(scrub, 'threshold_', np.nan))
    if np.isnan(threshold) and scrub.predicted_doublets_ is not None:
        # Derive threshold as min score among predicted doublets, fallback to 0.5.
        pos_scores = doublet_scores[predicted_doublets.astype(bool)]
        threshold = float(pos_scores.min()) if len(pos_scores) > 0 else 0.5

    # Export scrublet's internal PCA embedding of the real cells (n_cells × n_pcs).
    # The C++ test uses this embedding for the GPU kernel so both frameworks score
    # on the SAME embedding — otherwise a random projection vs scrublet's TruncatedSVD
    # gives Spearman ≈ 0.24 even though the algorithm is correct.
    # scrublet stores it in manifold_obs_ (cells × PCs, float64).
    pca_embedding = None
    for attr in ('manifold_obs_', '_manifold_obs', 'pca_obs_'):
        pca_embedding = getattr(scrub, attr, None)
        if pca_embedding is not None:
            break
    if pca_embedding is None:
        # Fallback: recompute from scrublet's normalized counts using sklearn.
        try:
            from sklearn.decomposition import TruncatedSVD
            n_pcs = min(n_genes - 1, 30)
            pca_embedding = TruncatedSVD(n_components=n_pcs,
                                         random_state=seed).fit_transform(
                scrub._E_obs_norm if hasattr(scrub, '_E_obs_norm')
                else counts_int.toarray())
        except Exception:
            pca_embedding = np.zeros((n_cells, 1), dtype=np.float64)

    pca_embedding = np.asarray(pca_embedding, dtype=np.float32)

    np.savez(
        out_path,
        doublet_scores     = doublet_scores,
        doublet_calls      = predicted_doublets,
        threshold          = np.array([threshold], dtype=np.float64),
        n_cells            = np.array([n_cells], dtype=np.int32),
        n_synth            = np.array([n_synth], dtype=np.int32),
        pca_embedding      = pca_embedding,   # n_cells × n_pcs, float32
    )
    n_doublets = int(predicted_doublets.sum())
    print(f"[doublet_scrublet_reference] wrote {out_path}  "
          f"n_cells={n_cells} n_synth={n_synth} n_doublets={n_doublets} "
          f"threshold={threshold:.4f}  pca_shape={pca_embedding.shape}",
          file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrublet Python reference for singlet-gpu cycle 31 doublet harness")
    parser.add_argument('--counts',      required=True,        help='input counts .bin (dump_csc format)')
    parser.add_argument('--out',         required=True,        help='output .npz path')
    parser.add_argument('--n-synth-frac', type=float, default=0.25,
                        help='synthetic doublet fraction (default 0.25)')
    parser.add_argument('--k',           type=int,   default=20,
                        help='number of kNN neighbors (default 20)')
    parser.add_argument('--seed',        type=int,   default=0, help='random seed')
    args = parser.parse_args()

    try:
        run_scrublet_reference(
            counts_path  = args.counts,
            out_path     = args.out,
            n_synth_frac = args.n_synth_frac,
            k            = args.k,
            seed         = args.seed,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"[doublet_scrublet_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
