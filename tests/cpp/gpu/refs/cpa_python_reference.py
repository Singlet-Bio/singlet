#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/cpa_python_reference.py
#
# Reference subprocess for PerturbGraph_TinySynthetic_VsCPA.
#
# Loads a CSC count matrix (dump_csc.h binary format) + perturbation labels +
# dose values.  Trains CPA Python (cpa-tools) on cells with labels ≠ held_out.
# Predicts expression for the held-out perturbation.
# Dumps:
#   pred_held_out   float32 [n_cells × n_genes]  (cells from held-out pert)
# to an .npz file at the path given by --out.
#
# Exit codes:
#   0  success
#   1  runtime error (data problem, cpa-tools bug, etc.)
#   2  cpa-tools (or a required dependency) is not installed — GTEST_SKIP
#
# Install (heavy): pip install cpa-tools scvi-tools anndata numpy scipy

import argparse
import struct
import sys

# ---------------------------------------------------------------------------
# Availability guard — exit 2 if packages are missing.
# ---------------------------------------------------------------------------
try:
    import numpy as np
    import scipy.sparse as sp
    import anndata as ad
except ImportError as e:
    print(f"[cpa_python_reference] Missing core dependency: {e}. Skipping.",
          file=sys.stderr)
    sys.exit(2)

try:
    import cpa
except ImportError as e:
    print(f"[cpa_python_reference] cpa-tools not installed: {e}. Skipping.",
          file=sys.stderr)
    sys.exit(2)

try:
    import scvi
except ImportError as e:
    print(f"[cpa_python_reference] scvi-tools not installed: {e}. Skipping.",
          file=sys.stderr)
    sys.exit(2)

# ---------------------------------------------------------------------------
# Binary readers (matching the C++ writers in perturbation_perturb_graph_correctness.cpp).
# ---------------------------------------------------------------------------

CSC_MAGIC        = 0x43535343   # "CSCC"
PERT_LABEL_MAGIC = 0x504C424C   # "PLBL"
DOSE_MAGIC       = 0x444F5345   # "DOSE"


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
    values  = np.frombuffer(raw, dtype=np.float32, count=nnz,    offset=offset).copy()
    offset += nnz * 4
    indptr  = np.frombuffer(raw, dtype=np.int32,   count=n + 1,  offset=offset).copy()
    offset += (n + 1) * 4
    indices = np.frombuffer(raw, dtype=np.int32,   count=nnz,    offset=offset).copy()
    return int(m), int(n), int(nnz), values, indptr, indices


def read_pert_labels(path: str):
    """Read perturbation label binary file.
    Format: [uint32 magic][uint32 n_cells][uint32 n_perts][int32 * n_cells]
    Returns (n_cells, n_perts, labels_array).
    """
    with open(path, "rb") as f:
        raw = f.read()
    magic, = struct.unpack_from("<I", raw, 0)
    if magic != PERT_LABEL_MAGIC:
        raise ValueError(f"read_pert_labels: bad magic {magic:#010x}")
    n_cells, n_perts = struct.unpack_from("<II", raw, 4)
    labels = np.frombuffer(raw, dtype=np.int32, count=n_cells, offset=12).copy()
    return int(n_cells), int(n_perts), labels


def read_doses(path: str):
    """Read dose binary file.
    Format: [uint32 magic][uint32 n_cells][float32 * n_cells]
    """
    with open(path, "rb") as f:
        raw = f.read()
    magic, = struct.unpack_from("<I", raw, 0)
    if magic != DOSE_MAGIC:
        raise ValueError(f"read_doses: bad magic {magic:#010x}")
    n_cells, = struct.unpack_from("<I", raw, 4)
    doses = np.frombuffer(raw, dtype=np.float32, count=n_cells, offset=8).copy()
    return doses


# ---------------------------------------------------------------------------
# Binary writer for dense float32 matrix (matching the C++ reader).
# Format: [uint32 magic][uint32 n_cells][uint32 n_genes][float32 * n_cells * n_genes]
# ---------------------------------------------------------------------------

DENSE_MAT_MAGIC = 0x444D5400   # "DMT\0"


def write_dense_mat(mat: np.ndarray, path: str):
    """Write float32 [n_cells × n_genes] row-major matrix."""
    assert mat.ndim == 2, "Expected 2-D matrix"
    n_cells, n_genes = mat.shape
    mat_f32 = mat.astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", DENSE_MAT_MAGIC))
        f.write(struct.pack("<II", n_cells, n_genes))
        f.write(mat_f32.tobytes())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CPA Python reference subprocess for singlet-gpu tests")
    parser.add_argument("--counts",    required=True,
                        help="CSC counts binary (genes × cells)")
    parser.add_argument("--labels",    required=True,
                        help="Perturbation label binary (cells × 1, int32)")
    parser.add_argument("--doses",     required=True,
                        help="Dose binary (cells, float32)")
    parser.add_argument("--held-out",  type=int, required=True,
                        help="Index of the held-out perturbation (0-indexed)")
    parser.add_argument("--n-perts",   type=int, required=True,
                        help="Total number of perturbations")
    parser.add_argument("--n-epochs",  type=int, default=300,
                        help="Number of CPA training epochs")
    parser.add_argument("--seed",      type=int, default=0,
                        help="Random seed")
    parser.add_argument("--out",       required=True,
                        help="Output .npz path (contains pred_held_out)")
    args = parser.parse_args()

    # Reproducibility.
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)

    # Load data.
    m, n, nnz, values, indptr, indices = read_csc_bin(args.counts)
    n_cells_chk, n_perts, labels = read_pert_labels(args.labels)
    doses = read_doses(args.doses)

    if n_cells_chk != n:
        raise ValueError(f"counts n={n} but labels n={n_cells_chk}")
    if n_perts != args.n_perts:
        raise ValueError(f"n_perts mismatch: file={n_perts} arg={args.n_perts}")

    # Build scipy CSC (genes × cells).
    X_csc = sp.csc_matrix((values, indices, indptr), shape=(m, n))
    # CPA expects cells × genes.
    X_cells_genes = X_csc.T.tocsr()

    # Build AnnData.
    adata = ad.AnnData(X=X_cells_genes.astype(np.float32))
    adata.obs["perturbation"] = [f"pert_{l}" for l in labels.tolist()]
    adata.obs["dose"]         = doses.astype(np.float32)

    # Mark control cells (perturbation index == 0 treated as control).
    # CPA requires a control_group key.
    adata.obs["control"] = (labels == 0).astype(int)

    # Gene names.
    adata.var_names = [f"gene_{i}" for i in range(m)]

    # Normalise (CPA tutorial preprocessing).
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Split: train on everything except the held-out perturbation.
    held_out_label = f"pert_{args.held_out}"
    train_mask = adata.obs["perturbation"] != held_out_label
    adata_train = adata[train_mask].copy()
    adata_held  = adata[~train_mask].copy()

    if len(adata_held) == 0:
        raise ValueError(f"No cells found for held-out pert {held_out_label}")

    # Set up CPA model.
    # cpa.CPA expects 'perturbation_key' and optionally 'dose_key'.
    cpa.CPA.setup_anndata(
        adata_train,
        perturbation_key="perturbation",
        dose_key="dose",
        control_group=f"pert_0",
    )

    model = cpa.CPA(
        adata=adata_train,
        n_latent=64,
        loss_ae="gauss",   # Gaussian reconstruction (approx to NB; simpler for small data)
        seed=args.seed,
    )

    model.train(
        max_epochs=args.n_epochs,
        use_gpu=False,  # CPU reference — the GPU kernel is what we're testing
    )

    # Predict held-out perturbation.
    # CPA predict() API: pass an AnnData with obs["perturbation"] and obs["dose"].
    # We re-use the held-out cells' covariates but feed the trained decoder.
    cpa.CPA.setup_anndata(
        adata_held,
        perturbation_key="perturbation",
        dose_key="dose",
        control_group="pert_0",
    )

    pred = model.predict(adata_held)
    # pred is an AnnData; pred.X contains the reconstructed expression (cells × genes).
    pred_matrix = np.array(pred.X, dtype=np.float32)
    if pred_matrix.ndim != 2 or pred_matrix.shape[1] != m:
        raise ValueError(
            f"CPA predict output shape unexpected: {pred_matrix.shape}, "
            f"expected (*, {m})")

    # Save results as .npz (numpy compressed archive).
    np.savez(
        args.out,
        pred_held_out=pred_matrix,
    )
    print(f"[cpa_python_reference] Written {args.out} "
          f"(n_held_out_cells={pred_matrix.shape[0]}, n_genes={pred_matrix.shape[1]})",
          file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[cpa_python_reference] ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
