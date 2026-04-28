#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# singlet-gpu/tests/refs/cell2fate_python_reference.py
#
# Reference implementation driver for cycle 27 Cell2fate correctness harness.
#
# Loads spliced + unspliced CSC binary files (dump_csc.h format) produced by
# the C++ test fixture, constructs an AnnData object, runs Cell2fate Python,
# and writes results to a numpy .npz for the C++ test to compare against.
#
# Usage (called by embed_cell2fate_correctness.cpp via subprocess):
#   python3 cell2fate_python_reference.py \
#       --spliced   <spliced.bin>           \
#       --unspliced <unspliced.bin>         \
#       --out       <results.npz>           \
#       --n-modules <K>                     \
#       --n-iter    <N>                     \
#       --seed      <S>
#
# Exit codes:
#   0  success — results.npz written
#   2  cell2fate not installed (test harness treats this as SKIP)
#   1  any other error (FAIL)
#
# Output npz keys:
#   gamma          float32[n_genes]        — posterior mean degradation rate
#   alpha          float32[n_genes]        — posterior mean transcription rate
#   beta           float32[n_genes]        — posterior mean splicing rate
#   cell_activities  float32[n_cells, K]   — per-cell module activities
#   module_loadings  float32[K, n_genes]   — per-module gene loadings
#   elbo_history   float32[n_iter]         — ELBO per iteration
#   n_modules      int scalar              — K actually used
#
# Dependencies (heavy install):
#   pip install cell2fate scvi-tools anndata pyro-ppl torch numpy scipy
#
# Note: cell2fate uses Pyro / SVI on CPU by default; GPU optional.
# For the synthetic tiny test this is fast (~5s).  For GSM4037629 it may take
# several minutes.  The C++ test is responsible for subprocess timeout handling.

import sys
import argparse
import struct
import numpy as np
import scipy.sparse

# ---- check cell2fate availability before importing anything else ------------

def _check_cell2fate():
    try:
        import cell2fate          # noqa: F401
        import scvi               # noqa: F401
        import anndata            # noqa: F401
        import pyro               # noqa: F401
        return True
    except ImportError as e:
        print(f"[cell2fate_python_reference] SKIP: {e}", file=sys.stderr)
        sys.exit(2)


# ---- CSC binary reader (dump_csc.h format) ----------------------------------
#
# Format (little-endian, no padding):
#   uint32  magic      = 0x43535343 ("CSCC")
#   uint32  n_rows
#   uint32  n_cols
#   uint64  nnz
#   float32[nnz]      values
#   int32[n_cols+1]   indptr
#   int32[nnz]        indices

def read_csc_bin(path: str) -> scipy.sparse.csc_matrix:
    with open(path, 'rb') as f:
        magic, n_rows, n_cols = struct.unpack('<III', f.read(12))
        if magic != 0x43535343:
            raise ValueError(f"Bad magic in {path}: 0x{magic:08X}")
        (nnz,) = struct.unpack('<Q', f.read(8))
        values  = np.frombuffer(f.read(nnz * 4),         dtype=np.float32).copy()
        indptr  = np.frombuffer(f.read((n_cols + 1) * 4), dtype=np.int32).copy()
        indices = np.frombuffer(f.read(nnz * 4),         dtype=np.int32).copy()
    return scipy.sparse.csc_matrix(
        (values, indices, indptr), shape=(n_rows, n_cols)
    )


# ---- Cell2fate driver -------------------------------------------------------

def run_cell2fate(spliced_path: str,
                  unspliced_path: str,
                  out_path: str,
                  n_modules: int,
                  n_iter: int,
                  seed: int) -> None:
    import anndata as ad
    import cell2fate

    # Load CSC matrices (genes × cells); transpose to cells × genes for AnnData.
    S = read_csc_bin(spliced_path).T.tocsr()    # cells × genes
    U = read_csc_bin(unspliced_path).T.tocsr()  # cells × genes

    n_cells, n_genes = S.shape
    assert U.shape == (n_cells, n_genes), (
        f"Shape mismatch: spliced {S.shape} vs unspliced {U.shape}")

    # Build AnnData.
    adata = ad.AnnData(X=S)
    adata.layers['spliced']   = S
    adata.layers['unspliced'] = U

    # Minimal gene/cell names.
    adata.var_names  = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names  = [f"cell_{j}" for j in range(n_cells)]

    # --- run Cell2fate --------------------------------------------------------
    # cell2fate.train_module is the primary entry point from the Nature Methods
    # 2025 paper (Lederer & Trynka).  The API changed between early preprints
    # and the published version; we guard for both interfaces.
    #
    # Published API (v0.1+):
    #   cell2fate.train_module(
    #       adata,
    #       n_modules=K,
    #       max_iter=N,
    #       seed=S,
    #       accelerator='cpu',     # we run CPU reference always
    #       use_gpu=False,
    #       verbose=False,
    #   )
    # Older API (preprint):
    #   cell2fate.model.Cell2fate(adata, n_modules=K)
    #   model.train(max_epochs=N)
    #
    # We try the published API first; fall back to the older interface.

    try:
        # Published interface.
        cell2fate.train_module(
            adata,
            n_modules=n_modules,
            max_iter=n_iter,
            seed=seed,
            accelerator='cpu',
            use_gpu=False,
            verbose=False,
        )
        # Results live in adata.var columns and adata.obsm.
        gamma            = adata.var.get('gamma',   adata.var.get('gamma_mean', None))
        alpha            = adata.var.get('alpha',   adata.var.get('alpha_mean', None))
        beta             = adata.var.get('beta',    adata.var.get('beta_mean', None))
        cell_activities  = adata.obsm.get('cell_activities',
                           adata.obsm.get('X_cell2fate', None))
        module_loadings  = (adata.varm.get('module_loadings',
                            adata.varm.get('cell2fate_loadings', None)))
        elbo_history     = adata.uns.get('cell2fate_elbo', np.array([], dtype=np.float32))

    except (AttributeError, TypeError):
        # Older preprint interface.
        model = cell2fate.model.Cell2fate(adata, n_modules=n_modules)
        torch_seed = seed
        try:
            import torch
            torch.manual_seed(torch_seed)
        except ImportError:
            pass
        model.train(max_epochs=n_iter)
        model.export_posterior(adata)
        gamma            = adata.var.get('gamma_mean', None)
        alpha            = adata.var.get('alpha_mean', None)
        beta             = adata.var.get('beta_mean', None)
        cell_activities  = adata.obsm.get('cell_activities', None)
        module_loadings  = adata.varm.get('module_loadings', None)
        elbo_history     = np.array(
            getattr(model, 'history', {}).get('elbo_train', []), dtype=np.float32)

    # ---- coerce and validate ------------------------------------------------
    def _to_f32(x, name: str, fallback_shape) -> np.ndarray:
        if x is None:
            print(f"[cell2fate_python_reference] WARNING: {name} not found in output; "
                  f"writing zeros {fallback_shape}", file=sys.stderr)
            return np.zeros(fallback_shape, dtype=np.float32)
        arr = np.asarray(x, dtype=np.float32)
        return arr

    gamma_arr   = _to_f32(gamma,           'gamma',           (n_genes,))
    alpha_arr   = _to_f32(alpha,           'alpha',           (n_genes,))
    beta_arr    = _to_f32(beta,            'beta',            (n_genes,))

    if cell_activities is not None:
        ca = np.asarray(cell_activities, dtype=np.float32)
        # Ensure shape (n_cells, K_actual).
        if ca.ndim == 1:
            ca = ca.reshape(n_cells, -1)
    else:
        ca = np.zeros((n_cells, n_modules), dtype=np.float32)

    if module_loadings is not None:
        ml = np.asarray(module_loadings, dtype=np.float32)
        # cell2fate stores loadings as (n_genes, K) in varm; transpose to (K, n_genes).
        if ml.shape == (n_genes, ca.shape[1]):
            ml = ml.T
        elif ml.ndim != 2 or ml.shape[0] != ca.shape[1]:
            ml = np.zeros((ca.shape[1], n_genes), dtype=np.float32)
    else:
        ml = np.zeros((ca.shape[1], n_genes), dtype=np.float32)

    k_actual = ca.shape[1]

    np.savez(
        out_path,
        gamma            = gamma_arr,
        alpha            = alpha_arr,
        beta             = beta_arr,
        cell_activities  = ca,
        module_loadings  = ml,
        elbo_history     = np.asarray(elbo_history, dtype=np.float32),
        n_modules        = np.array([k_actual], dtype=np.int32),
    )
    print(f"[cell2fate_python_reference] wrote {out_path}  "
          f"n_cells={n_cells} n_genes={n_genes} K={k_actual}", file=sys.stderr)


# ---- CLI --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cell2fate Python reference for singlet-gpu cycle 27 harness")
    parser.add_argument('--spliced',   required=True,  help='spliced .bin (dump_csc format)')
    parser.add_argument('--unspliced', required=True,  help='unspliced .bin (dump_csc format)')
    parser.add_argument('--out',       required=True,  help='output .npz path')
    parser.add_argument('--n-modules', type=int, default=3,    help='number of modules K')
    parser.add_argument('--n-iter',    type=int, default=2000, help='SVI iterations')
    parser.add_argument('--seed',      type=int, default=0,    help='random seed')
    args = parser.parse_args()

    _check_cell2fate()

    try:
        run_cell2fate(
            spliced_path  = args.spliced,
            unspliced_path= args.unspliced,
            out_path      = args.out,
            n_modules     = args.n_modules,
            n_iter        = args.n_iter,
            seed          = args.seed,
        )
    except Exception as e:
        print(f"[cell2fate_python_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
