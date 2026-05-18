#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/tests/refs/palantir_py_reference.py
#
# Reference implementation for fate/palantir correctness harness (cycle 45).
#
# Dispatch: called by fate_palantir_correctness.cpp via subprocess.
#
# Exit codes:
#   0  — Palantir Python ran successfully; reference written to --out_dir
#   2  — Palantir Python not installed; pure-scipy diffusion map + Markov
#        absorption fallback ran successfully; reference written to --out_dir
#   1  — unexpected error
#
# Primary path: Palantir Python package (Setty et al. Nature Biotech 2019).
#   pip install palantir anndata scanpy
#   Computes pseudotime and branch probabilities via Palantir's own API.
#
# Fallback path (always present): pure-scipy diffusion map + Markov absorption.
#   Steps match the algorithm spec in state/designs/45-palantir.md:
#   1. kNN graph from expression PCA.
#   2. Diffusion kernel: w[i,j] = exp(-d^2 / (2*sigma_i^2)).
#   3. Row-normalize → stochastic P.
#   4. Anisotropic correction: P' = D^{-1/2} P D^{-1/2}.
#   5. Top-k eigendecomposition of P' (scipy.sparse.linalg.eigsh).
#   6. Pseudotime: sqrt( sum_k lambda_k^2 * (psi_k[c] - psi_k[start])^2 ).
#   7. Terminal detection: max-pseudotime cell per Leiden/KMeans cluster.
#   8. Branch probs: absorbing Markov chain (I-Q)^-1 R via GMRES.
#
# Input files (from --data_dir, written by C++ test):
#   expression.bin     — [uint32_t n_cells][uint32_t n_genes]
#                        [float × n_cells × n_genes]  (row-major, dense)
#   start_cell.bin     — [uint32_t start_cell_index]
#   n_terminals.bin    — [uint32_t n_terminals]  (desired number of terminal states)
#   terminal_ids.bin   — [uint32_t n][int32_t × n]  (provided terminals, may be empty)
#
# Output files (to --out_dir):
#   py_pseudotime.bin       — [uint32_t n_cells][float × n_cells]
#   py_branch_probs.bin     — [uint32_t n_cells][uint32_t n_terminals]
#                             [float × n_cells × n_terminals]  (row-major)
#   py_terminal_ids.bin     — [uint32_t n][int32_t × n]  (detected or provided)
#
# Usage:
#   python3 palantir_py_reference.py --task tiny_synthetic \
#       --data_dir /tmp/... --out_dir /tmp/...

import argparse
import struct
import sys
import os

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_expression_bin(path):
    """Read dense float32 matrix: [uint32_t n_cells][uint32_t n_genes][float×n_cells×n_genes]."""
    with open(path, 'rb') as f:
        n_cells = struct.unpack('<I', f.read(4))[0]
        n_genes = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n_cells * n_genes * 4), dtype=np.float32).copy()
    return data.reshape(n_cells, n_genes), n_cells, n_genes


def read_u32(path):
    """Read a single uint32 from a binary file."""
    with open(path, 'rb') as f:
        return struct.unpack('<I', f.read(4))[0]


def read_i32_vec(path):
    """Read [uint32_t n][int32_t × n]; returns empty array if n=0."""
    with open(path, 'rb') as f:
        n = struct.unpack('<I', f.read(4))[0]
        if n == 0:
            return np.array([], dtype=np.int32)
        return np.frombuffer(f.read(n * 4), dtype=np.int32).copy()


def write_f32_vec(path, arr):
    """Write [uint32_t n][float32 × n]."""
    arr32 = np.asarray(arr, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(arr32)))
        f.write(arr32.tobytes())


def write_branch_probs(path, prob):
    """Write [uint32_t n_cells][uint32_t n_terms][float32 × n_cells × n_terms]."""
    n_cells, n_terms = prob.shape
    prob32 = np.asarray(prob, dtype=np.float32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', n_cells))
        f.write(struct.pack('<I', n_terms))
        f.write(prob32.tobytes())


def write_i32_vec(path, arr):
    """Write [uint32_t n][int32_t × n]."""
    arr32 = np.asarray(arr, dtype=np.int32)
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', len(arr32)))
        f.write(arr32.tobytes())


# ---------------------------------------------------------------------------
# Pure-scipy diffusion map + Markov absorption fallback
# ---------------------------------------------------------------------------

def _pca_embed(X, n_components=50):
    """Simple truncated SVD for PCA embedding (no sklearn needed)."""
    from scipy.sparse.linalg import svds
    # Center
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD
    n_comp = min(n_components, min(Xc.shape) - 1)
    try:
        U, s, Vt = svds(Xc.astype(np.float64), k=n_comp)
        # svds returns ascending order; flip.
        idx = np.argsort(-s)
        U = U[:, idx]
        s = s[idx]
    except Exception:
        # Fallback: dense SVD for very small matrices.
        U, s, Vt = np.linalg.svd(Xc.astype(np.float64), full_matrices=False)
        U = U[:, :n_comp]
        s = s[:n_comp]
    return (U * s).astype(np.float32)


def _knn_graph(embed, k=30):
    """Brute-force kNN. Returns (indices [n×k], distances [n×k])."""
    n = embed.shape[0]
    k = min(k, n - 1)
    # squared distances
    sq = np.sum(embed ** 2, axis=1)
    # pairwise dist^2
    dist2 = sq[:, None] + sq[None, :] - 2.0 * (embed @ embed.T)
    np.fill_diagonal(dist2, np.inf)
    dist2 = np.maximum(dist2, 0.0)
    # k nearest
    knn_idx = np.argsort(dist2, axis=1)[:, :k]
    knn_d2  = np.take_along_axis(dist2, knn_idx, axis=1)
    return knn_idx, knn_d2


def scipy_diffusion_palantir(X, start_cell, n_terminals_desired,
                              terminal_ids_provided, k_knn=30, n_eigs=20):
    """
    Pure-scipy implementation of the Palantir diffusion pseudotime + fate algorithm.
    Returns (pseudotime, branch_probs, terminal_ids).
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    from scipy.sparse.linalg import gmres
    from scipy.sparse import eye as speye

    n_cells = X.shape[0]
    k_knn = min(k_knn, n_cells - 1)
    n_eigs = min(n_eigs, n_cells - 2)

    # --- Step 1: PCA embed + kNN ---
    n_pca = min(50, min(X.shape) - 1)
    embed = _pca_embed(X, n_components=n_pca)
    knn_idx, knn_d2 = _knn_graph(embed, k=k_knn)

    # --- Step 2: Diffusion kernel weights ---
    # sigma_i^2 = distance to k'-th neighbor (use median distance).
    # Palantir uses adaptive bandwidth: sigma_i = distance to k-th neighbor.
    sigma2 = knn_d2[:, -1].astype(np.float64)  # shape (n_cells,)
    sigma2 = np.maximum(sigma2, 1e-10)

    # Build sparse weight matrix W (n_cells × n_cells).
    rows, cols, wvals = [], [], []
    for i in range(n_cells):
        si2 = sigma2[i]
        for jj, d2 in zip(knn_idx[i], knn_d2[i]):
            j = int(jj)
            if j == i:
                continue
            sj2 = sigma2[j]
            # Symmetric kernel: use geometric mean bandwidth.
            sigma_ij2 = np.sqrt(si2 * sj2)
            w = np.exp(-float(d2) / (2.0 * sigma_ij2))
            rows.append(i); cols.append(j); wvals.append(w)
            rows.append(j); cols.append(i); wvals.append(w)

    W = csr_matrix((wvals, (rows, cols)), shape=(n_cells, n_cells), dtype=np.float64)
    # Deduplicate (symmetric entries may appear twice).
    W = W.tocsr()
    W.sum_duplicates()

    # --- Step 3: Row-normalize → P ---
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    row_sum = np.maximum(row_sum, 1e-10)
    D_inv = csr_matrix((1.0 / row_sum, (np.arange(n_cells), np.arange(n_cells))),
                        shape=(n_cells, n_cells))
    P = D_inv @ W  # stochastic: rows sum to 1

    # --- Step 4: Anisotropic correction P' = D^{-1/2} P D^{-1/2} ---
    p_row_sum = np.asarray(P.sum(axis=1)).ravel()
    p_row_sum = np.maximum(p_row_sum, 1e-10)
    D_invsqrt = csr_matrix(
        (1.0 / np.sqrt(p_row_sum), (np.arange(n_cells), np.arange(n_cells))),
        shape=(n_cells, n_cells))
    P_sym = D_invsqrt @ P @ D_invsqrt

    # --- Step 5: Top-k eigendecomposition of symmetric P' ---
    n_eigs = min(n_eigs, n_cells - 2)
    try:
        eigvals, eigvecs = eigsh(P_sym, k=n_eigs, which='LM')
        # Sort descending by eigenvalue.
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
    except Exception as e:
        print(f"[palantir_py_reference] eigsh failed ({e}); using dense fallback",
              file=sys.stderr)
        eigvals, eigvecs = np.linalg.eigh(P_sym.toarray())
        idx = np.argsort(-eigvals)
        eigvals = eigvals[idx][:n_eigs]
        eigvecs = eigvecs[:, idx][:, :n_eigs]

    # --- Step 6: Pseudotime ---
    # pseudotime[c] = sqrt( sum_k lambda_k^2 * (psi_k[c] - psi_k[start])^2 )
    psi_start = eigvecs[start_cell, :]  # (n_eigs,)
    diff = eigvecs - psi_start[None, :]  # (n_cells, n_eigs)
    lam2 = eigvals ** 2  # (n_eigs,)
    pseudotime = np.sqrt(np.sum(lam2[None, :] * diff ** 2, axis=1))  # (n_cells,)
    # Normalize to [0, 1].
    pt_max = pseudotime.max()
    if pt_max > 0:
        pseudotime /= pt_max

    # --- Step 7: Terminal state detection or use provided ---
    if terminal_ids_provided is not None and len(terminal_ids_provided) > 0:
        terminal_ids = np.asarray(terminal_ids_provided, dtype=np.int32)
    else:
        # Simple detection: pick cells with locally maximal pseudotime.
        # Cluster on eigenvectors using k-means.
        n_terms = max(2, n_terminals_desired)
        n_terms = min(n_terms, n_cells)
        from scipy.cluster.vq import kmeans2
        np.random.seed(42)
        features = eigvecs[:, :min(10, n_eigs)]
        try:
            centroids, labels = kmeans2(features.astype(np.float64), n_terms,
                                         minit='points', iter=50)
        except Exception:
            labels = np.arange(n_cells) % n_terms
        # Per cluster: cell with maximum pseudotime.
        terminal_ids = []
        for k in range(n_terms):
            mask = (labels == k)
            if not mask.any():
                continue
            idx_k = np.where(mask)[0]
            term = idx_k[np.argmax(pseudotime[idx_k])]
            terminal_ids.append(int(term))
        terminal_ids = np.array(sorted(set(terminal_ids)), dtype=np.int32)

    # --- Step 8: Branch probabilities via absorbing Markov chain ---
    terminal_set = set(int(t) for t in terminal_ids)
    transient_idx = np.array([i for i in range(n_cells) if i not in terminal_set],
                               dtype=np.int32)
    absorbing_idx = np.array(sorted(terminal_set), dtype=np.int32)
    n_terms_actual = len(absorbing_idx)
    n_tr = len(transient_idx)

    absorption_prob = np.zeros((n_cells, n_terms_actual), dtype=np.float64)
    # Identity rows for terminals.
    for c, t in enumerate(absorbing_idx):
        absorption_prob[int(t), c] = 1.0

    if n_tr > 0:
        Q = P[transient_idx, :][:, transient_idx]
        R = P[transient_idx, :][:, absorbing_idx]
        A = speye(n_tr, format='csr') - Q

        for k in range(n_terms_actual):
            rhs = np.asarray(R[:, k].todense(), dtype=np.float64).ravel()
            x0  = np.zeros(n_tr, dtype=np.float64)
            sol, info = gmres(A, rhs, x0=x0, rtol=1e-6, maxiter=300, restart=30)
            if info != 0:
                print(f"[palantir_py_reference] GMRES warning: terminal {k}, "
                      f"info={info}", file=sys.stderr)
            absorption_prob[transient_idx, k] = np.maximum(sol, 0.0)

    # Normalize rows so they sum to 1.
    row_sums = absorption_prob.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    absorption_prob /= row_sums

    return (pseudotime.astype(np.float32),
            absorption_prob.astype(np.float32),
            terminal_ids)


# ---------------------------------------------------------------------------
# Palantir Python primary path
# ---------------------------------------------------------------------------

def run_palantir_python(X, start_cell, n_terminals_desired, terminal_ids_provided):
    """
    Primary path: use palantir Python package.
    Returns (pseudotime, branch_probs, terminal_ids, used_palantir=True).
    Raises ImportError if palantir is not installed.
    """
    import palantir
    import anndata as ad
    import scanpy as sc

    n_cells, n_genes = X.shape
    adata = ad.AnnData(X=X.copy())
    adata.obs_names = [str(i) for i in range(n_cells)]

    # Preprocessing: PCA.
    sc.pp.pca(adata, n_comps=min(50, min(n_cells, n_genes) - 1))

    # Build diffusion maps via Palantir.
    palantir.utils.run_diffusion_maps(adata, n_components=20)
    palantir.utils.determine_multiscale_space(adata)

    start_name = str(start_cell)
    terminal_names = (
        [str(int(t)) for t in terminal_ids_provided]
        if (terminal_ids_provided is not None and len(terminal_ids_provided) > 0)
        else None
    )

    pr_res = palantir.core.run_palantir(
        adata,
        early_cell=start_name,
        terminal_states=terminal_names,
        num_waypoints=min(500, n_cells),
        n_jobs=1,
    )

    pseudotime = np.array([pr_res.pseudotime[str(i)] for i in range(n_cells)],
                           dtype=np.float32)
    pt_max = pseudotime.max()
    if pt_max > 0:
        pseudotime /= pt_max

    # Branch probabilities.
    bp_df = pr_res.branch_probs
    terminal_names_detected = list(bp_df.columns)
    n_terms = len(terminal_names_detected)
    branch_probs = bp_df.values.astype(np.float32)  # (n_cells, n_terms)

    # Map terminal names back to cell indices.
    term_ids = np.array([int(t) for t in terminal_names_detected], dtype=np.int32)

    return pseudotime, branch_probs, term_ids


# ---------------------------------------------------------------------------
# Main task runner
# ---------------------------------------------------------------------------

def run_task(args):
    data_dir = args.data_dir
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Read expression matrix and metadata.
    X, n_cells, n_genes = read_expression_bin(
        os.path.join(data_dir, 'expression.bin'))
    start_cell       = read_u32(os.path.join(data_dir, 'start_cell.bin'))
    n_terminals_des  = read_u32(os.path.join(data_dir, 'n_terminals.bin'))
    terminal_ids_prov = read_i32_vec(os.path.join(data_dir, 'terminal_ids.bin'))
    if len(terminal_ids_prov) == 0:
        terminal_ids_prov = None

    print(f"[palantir_py_reference] n_cells={n_cells}, n_genes={n_genes}, "
          f"start_cell={start_cell}, n_terminals_desired={n_terminals_des}")

    # --- Try Palantir Python first ---
    use_palantir = False
    pseudotime = branch_probs = terminal_ids = None
    try:
        pseudotime, branch_probs, terminal_ids = run_palantir_python(
            X, start_cell, n_terminals_des, terminal_ids_prov)
        use_palantir = True
        print(f"[palantir_py_reference] Palantir Python path: "
              f"n_terminals={len(terminal_ids)}, "
              f"pt range=[{pseudotime.min():.4f}, {pseudotime.max():.4f}], "
              f"branch_probs row_sum range=["
              f"{branch_probs.sum(axis=1).min():.4f}, "
              f"{branch_probs.sum(axis=1).max():.4f}]")
    except (ImportError, ModuleNotFoundError):
        print("[palantir_py_reference] Palantir not installed; "
              "using scipy diffusion map fallback", file=sys.stderr)
    except Exception as e:
        print(f"[palantir_py_reference] Palantir failed ({e}); "
              f"falling back to scipy", file=sys.stderr)

    if not use_palantir:
        pseudotime, branch_probs, terminal_ids = scipy_diffusion_palantir(
            X, start_cell, n_terminals_des, terminal_ids_prov)
        rs = branch_probs.sum(axis=1)
        print(f"[palantir_py_reference] scipy fallback: "
              f"n_terminals={len(terminal_ids)}, "
              f"pt range=[{pseudotime.min():.4f}, {pseudotime.max():.4f}], "
              f"branch_probs row_sum range=[{rs.min():.4f}, {rs.max():.4f}]")

    # Write outputs.
    write_f32_vec(os.path.join(out_dir, 'py_pseudotime.bin'), pseudotime)
    write_branch_probs(os.path.join(out_dir, 'py_branch_probs.bin'), branch_probs)
    write_i32_vec(os.path.join(out_dir, 'py_terminal_ids.bin'), terminal_ids)

    rc = 0 if use_palantir else 2
    return rc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Palantir Python reference for singlet-gpu correctness harness')
    parser.add_argument('--task',     required=True,
                        choices=['tiny_synthetic', 'gsm_real'],
                        help='Which reference task to run')
    parser.add_argument('--data_dir', required=True,
                        help='Directory containing input binary files')
    parser.add_argument('--out_dir',  required=True,
                        help='Directory to write output binary files')
    args = parser.parse_args()

    try:
        rc = run_task(args)
        sys.exit(rc)
    except Exception as e:
        print(f"[palantir_py_reference] ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
