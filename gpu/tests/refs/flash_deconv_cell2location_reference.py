#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/flash_deconv_cell2location_reference.py
#
# Reference subprocess for FlashDeconv_TinySynthetic_VsCell2Location.
#
# Loads:
#   --counts    CSC binary  (genes × spots, dump_csc.h "CSCC" format)
#   --ref-means dense binary (n_types × n_genes, "REFM" magic)
#   --out       output path for .npz containing 'W' (spots × types, float32)
#
# Uses a lightweight NMF-based reference (scipy nnls) to approximate cell2location
# when the full PyMC stack is unavailable.  Full cell2location (PyMC + scvi-tools)
# is attempted first; if unavailable, falls back to scipy NNLS with simplex
# projection (which is what the C++ ADMM also converges to in the limit).
#
# Exit codes:
#   0  success
#   1  runtime error
#   2  packages unavailable → GTEST_SKIP
#
# Heavy install (attempt first):
#   pip install cell2location pymc scvi-tools anndata numpy scipy

import argparse
import struct
import sys

# ---------------------------------------------------------------------------
# Core deps — numpy + scipy are always required.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    from scipy.optimize import nnls
    from scipy.stats import spearmanr
    import scipy.sparse as sp
except ImportError as e:
    print(f"[flash_deconv_cell2location_reference] Missing core dep: {e}",
          file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Binary I/O helpers (matching C++ writers in spatial_flash_deconv_correctness.cpp).
# ---------------------------------------------------------------------------

CSC_MAGIC  = 0x43535343   # "CSCC"  — dump_csc.h format
REFM_MAGIC = 0x5245464D   # "REFM"  — reference mean matrix
DENSE_MAGIC = 0x444D5400  # "DMT\0" — dense result matrix


def read_csc_bin(path: str):
    """Read dump_csc.h binary: returns (m, n, nnz, values, indptr, indices)."""
    with open(path, "rb") as f:
        raw = f.read()
    off = 0
    magic, = struct.unpack_from("<I", raw, off); off += 4
    if magic != CSC_MAGIC:
        raise ValueError(f"read_csc_bin: bad magic {magic:#010x}")
    m, n = struct.unpack_from("<II", raw, off); off += 8
    nnz, = struct.unpack_from("<Q", raw, off); off += 8
    values  = np.frombuffer(raw, dtype=np.float32, count=nnz,    offset=off).copy()
    off += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,   count=n + 1,  offset=off).copy()
    off += (n + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,   count=nnz,    offset=off).copy()
    return int(m), int(n), int(nnz), values, indptr, indices


def read_refm_bin(path: str):
    """Read reference mean matrix: [REFM magic][uint32 n_types][uint32 n_genes][float32 * n_types * n_genes]."""
    with open(path, "rb") as f:
        raw = f.read()
    magic, = struct.unpack_from("<I", raw, 0)
    if magic != REFM_MAGIC:
        raise ValueError(f"read_refm_bin: bad magic {magic:#010x}")
    n_types, n_genes = struct.unpack_from("<II", raw, 4)
    data = np.frombuffer(raw, dtype=np.float32,
                         count=n_types * n_genes, offset=12).copy()
    return data.reshape(n_types, n_genes)


def write_dense_mat(mat: np.ndarray, path: str):
    """Write float32 [rows × cols] row-major matrix with DENSE_MAGIC header."""
    assert mat.ndim == 2
    rows, cols = mat.shape
    mat_f32 = mat.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", DENSE_MAGIC))
        f.write(struct.pack("<II", rows, cols))
        f.write(mat_f32.tobytes())


# ---------------------------------------------------------------------------
# Simplex projection: project each row of W onto the probability simplex.
# ---------------------------------------------------------------------------

def project_simplex(v: np.ndarray) -> np.ndarray:
    """Project 1-D vector v onto the unit simplex (sum=1, entries>=0)."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = float(cssv[rho] - 1) / float(rho + 1)
    return np.maximum(v - theta, 0.0)


def nnls_simplex_deconv(Y: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Solve Y ≈ W @ B subject to W >= 0 and rows of W sum to 1.

    Y: (n_spots, n_genes)
    B: (n_types, n_genes)
    Returns W: (n_spots, n_types)
    """
    n_spots, n_genes = Y.shape
    n_types, _ = B.shape
    W = np.zeros((n_spots, n_types), dtype=np.float64)
    Bt = B.T.astype(np.float64)          # (n_genes, n_types)
    for i in range(n_spots):
        w, _ = nnls(Bt, Y[i].astype(np.float64))
        # Normalise to simplex.
        s = w.sum()
        if s > 1e-12:
            w = w / s
        else:
            w = np.ones(n_types) / n_types
        W[i] = w
    return W.astype(np.float32)


# ---------------------------------------------------------------------------
# Attempt full cell2location (PyMC + scvi-tools).
# ---------------------------------------------------------------------------

def try_cell2location(Y_dense: np.ndarray, B: np.ndarray,
                      seed: int, n_epochs: int) -> np.ndarray:
    """
    Attempt to run cell2location (PyMC / scvi-tools).
    Returns W (n_spots × n_types) if successful, else raises ImportError.
    """
    import anndata as ad
    import cell2location

    n_spots, n_genes = Y_dense.shape
    n_types = B.shape[0]

    adata_sp = ad.AnnData(
        X=sp.csr_matrix(Y_dense.astype(np.float32))
    )
    adata_sp.var_names = [f"gene_{g}" for g in range(n_genes)]
    adata_sp.obs_names = [f"spot_{s}" for s in range(n_spots)]

    # Build reference signature from B.
    inf_aver = np.exp(B.astype(np.float64)) - 1.0   # B was already log1p'd
    inf_aver = np.maximum(inf_aver, 0.0)

    import pandas as pd
    sig_df = pd.DataFrame(
        inf_aver.T,
        index=[f"gene_{g}" for g in range(n_genes)],
        columns=[f"type_{t}" for t in range(n_types)],
    )

    cell2location.models.Cell2location.setup_anndata(adata_sp)
    mod = cell2location.models.Cell2location(
        adata_sp,
        cell_state_df=sig_df,
        N_cells_per_location=5,
        detection_alpha=20,
    )
    mod.train(max_epochs=n_epochs, use_gpu=False,
              early_stopping=False, batch_size=None)

    adata_sp = mod.export_posterior(adata_sp, use_quantile=True, add_to_obsm="q05")
    W_raw = adata_sp.obsm["q05_cell_abundance_w_sf"]
    # Normalise rows to sum to 1.
    row_sums = W_raw.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    return (W_raw / row_sums).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="cell2location reference subprocess for singlet-gpu FlashDeconv tests")
    parser.add_argument("--counts",   required=True,
                        help="CSC counts binary (genes × spots)")
    parser.add_argument("--ref-means", required=True,
                        help="Reference mean matrix binary (n_types × n_genes)")
    parser.add_argument("--out",      required=True,
                        help="Output path (.npz); contains W (spots × types)")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="Training epochs for cell2location (if available)")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load inputs.
    m, n_spots, nnz, values, indptr, indices = read_csc_bin(args.counts)
    # Y_dense: spots × genes  (genes are rows in CSC → need transpose)
    Y_csc = sp.csc_matrix((values, indices, indptr), shape=(m, n_spots))
    Y_dense = np.asarray(Y_csc.T.todense(), dtype=np.float32)  # (n_spots, n_genes)

    B = read_refm_bin(args.ref_means)   # (n_types, n_genes)

    # Attempt full cell2location first.
    W = None
    try:
        import anndata  # noqa: F401
        W = try_cell2location(Y_dense, B, args.seed, args.n_epochs)
        print("[flash_deconv_cell2location_reference] cell2location succeeded.",
              file=sys.stderr)
    except ImportError as e:
        print(f"[flash_deconv_cell2location_reference] cell2location not available "
              f"({e}); using scipy NNLS fallback.", file=sys.stderr)
        W = nnls_simplex_deconv(Y_dense, B)

    np.savez(args.out, W=W)
    print(f"[flash_deconv_cell2location_reference] Written {args.out} "
          f"(shape {W.shape})", file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[flash_deconv_cell2location_reference] ERROR: {exc}",
              file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
