#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# singlet-gpu/bench/refs/magic_ref.py
#
# CYCLE-174 Phase E — manual scipy SpMM CPU baseline for preprocess/magic.
#
# GPU kernel (preprocess::magic_impute):
#   (a) row_sum_graph_kernel     — per-row Markov weight sums (warp-shuffle).
#   (b) normalize_weights_kernel — D⁻¹W Markov transition matrix.
#   (c) csc_transpose_to_dense_kernel — scatter CSC X (m×n) → dense Y₀ (n×m).
#   (d) cusparseSpMM × t=3      — Y_s = M · Y_{s-1} (ping-pong dense).
# Reference: van Dijk et al. (2018) "Recovering Gene Interactions from
#            Single-Cell Data Using Data Diffusion." Cell 174:716-729.
#
# CPU implementation (scipy/numpy — avoids magic-impute pkg dependency):
#   1. Build synthetic CSC expression matrix (scipy.sparse.csc_matrix).
#   2. Build synthetic PCA embedding (numpy float32).
#   3. Build kNN graph: sklearn.neighbors.NearestNeighbors (L2, k=10, brute).
#   4. Build SNN adjacency as symmetric binary kNN adjacency (row-normalize to Markov).
#   5. Iterate t=3 times: X = M @ X  (scipy.sparse SpMM, col-major dense X).
#   Step 5 only is timed (steps 1-4 are untimed — matches GPU bench protocol).
#
# Protocol: 2 warmup + 5 timed iterations per scale; median wall time.
#   Only the t=3 SpMM diffusion loop is timed (not graph build or matrix init).
#
# Scales:
#   10k: n_cells=10000, n_genes=5000, density=5%, k=10, t=3
#   30k: n_cells=30000, n_genes=5000, density=5%, k=10, t=3
#
# Output (stdout): CSV header + rows:
#   scale,n_cells,n_genes,density,k,t,wall_ms
#
# Usage: python3 magic_ref.py

import sys
import time

import numpy as np
import scipy.sparse as sp

WARMUP_ITERS = 2
TIMED_ITERS  = 5

SCALES = [
    ("10k", 10000, 5000, 0.05, 10, 3),
    ("30k", 30000, 5000, 0.05, 10, 3),
]


def make_sparse_csc(n_genes: int, n_cells: int,
                    density: float, seed: int = 42) -> sp.csc_matrix:
    """Synthetic CSC expression matrix (n_genes × n_cells) at target density.

    Uses numpy default_rng for reproducibility; values are integer counts 0–10
    to match the C++ LCG synthetic data intent.
    """
    rng = np.random.default_rng(seed=seed)
    nnz_per_col = max(1, round(n_genes * density))
    total_nnz   = nnz_per_col * n_cells

    # Pre-allocate COO arrays then convert to CSC.
    rows = np.empty(total_nnz, dtype=np.int32)
    cols = np.empty(total_nnz, dtype=np.int32)
    vals = np.empty(total_nnz, dtype=np.float32)

    for c in range(n_cells):
        r = rng.choice(n_genes, size=nnz_per_col, replace=False)
        r.sort()
        rows[c * nnz_per_col: (c + 1) * nnz_per_col] = r
        cols[c * nnz_per_col: (c + 1) * nnz_per_col] = c
        vals[c * nnz_per_col: (c + 1) * nnz_per_col] = rng.integers(
            0, 11, size=nnz_per_col).astype(np.float32)

    return sp.csc_matrix((vals, (rows, cols)), shape=(n_genes, n_cells))


def make_pca_embedding(n_cells: int, n_dims: int = 50,
                       seed: int = 137) -> np.ndarray:
    """Synthetic PCA embedding (n_cells × n_dims, float32)."""
    rng = np.random.default_rng(seed=seed)
    return rng.standard_normal((n_cells, n_dims)).astype(np.float32)


def build_markov_matrix(emb: np.ndarray, k: int) -> sp.csr_matrix:
    """Build row-stochastic Markov transition matrix from kNN adjacency.

    Steps:
      1. sklearn NearestNeighbors (brute L2, k=k) → binary kNN adjacency (CSR).
      2. Symmetrize: W = max(A, A^T) (any edge in either direction kept).
      3. Row-normalize: M[i,j] = W[i,j] / sum_j W[i,j].
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        # Fallback: pure numpy brute-force for small n.
        print("[ref] sklearn not found; using numpy brute-force kNN", file=sys.stderr)
        return _build_markov_numpy(emb, k)

    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean',
                          n_jobs=-1)
    nn.fit(emb)
    dist, idx = nn.kneighbors(emb, n_neighbors=k, return_distance=True)

    # Build binary CSR adjacency from kNN indices.
    row_idx = np.repeat(np.arange(n, dtype=np.int32), k)
    col_idx = idx.ravel().astype(np.int32)
    ones    = np.ones(n * k, dtype=np.float32)
    A = sp.csr_matrix((ones, (row_idx, col_idx)), shape=(n, n))

    # Symmetrize.
    W = A.maximum(A.T).tocsr().astype(np.float32)

    # Row-normalize.
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero (isolated nodes)
    D_inv = sp.diags(1.0 / row_sums, format='csr')
    M = D_inv @ W
    return M.tocsr().astype(np.float32)


def _build_markov_numpy(emb: np.ndarray, k: int) -> sp.csr_matrix:
    """Numpy brute-force fallback when sklearn is not installed."""
    n = emb.shape[0]
    # Compute pairwise squared L2.
    sq = (emb ** 2).sum(axis=1, keepdims=True)
    D2 = sq + sq.T - 2.0 * (emb @ emb.T)
    np.fill_diagonal(D2, np.inf)
    idx = np.argpartition(D2, k, axis=1)[:, :k].astype(np.int32)
    row_idx = np.repeat(np.arange(n, dtype=np.int32), k)
    col_idx = idx.ravel()
    ones    = np.ones(n * k, dtype=np.float32)
    A = sp.csr_matrix((ones, (row_idx, col_idx)), shape=(n, n))
    W = A.maximum(A.T).tocsr().astype(np.float32)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    M = sp.diags(1.0 / row_sums, format='csr') @ W
    return M.tocsr().astype(np.float32)


def run_magic_diffusion(M: sp.csr_matrix, X: sp.csc_matrix, t: int) -> np.ndarray:
    """Iterate t steps of Y = M @ Y (scipy SpMM), starting from Y = X^T (dense).

    X is (n_genes × n_cells); Y₀ = X.T.toarray() is (n_cells × n_genes).
    Each SpMM step: Y = M @ Y  (M is n×n CSR, Y is n×m dense).
    Returns final dense Y (n_cells × n_genes).
    """
    # Materialise initial dense n×m matrix (transposed X).
    Y = np.asfortranarray(X.T.toarray(), dtype=np.float32)  # col-major, shape (n_cells, n_genes)

    for _ in range(t):
        # M @ Y: M is (n×n) sparse, Y is (n×m) dense.
        # scipy sparse @ dense returns row-major np.ndarray; keep as float32.
        Y = np.asfortranarray((M @ Y).astype(np.float32))

    return Y


def time_magic(M: sp.csr_matrix, X: sp.csc_matrix, t: int) -> float:
    """2 warmup + 5 timed iterations of the t-step SpMM diffusion loop.

    Returns median wall time in ms.
    """
    for _ in range(WARMUP_ITERS):
        try:
            run_magic_diffusion(M, X, t)
        except Exception:
            pass

    wall_times = []
    for _ in range(TIMED_ITERS):
        t0 = time.perf_counter()
        run_magic_diffusion(M, X, t)
        t1 = time.perf_counter()
        wall_times.append((t1 - t0) * 1000.0)

    wall_times.sort()
    n_t = len(wall_times)
    return (wall_times[n_t // 2] if n_t % 2 == 1
            else 0.5 * (wall_times[n_t // 2 - 1] + wall_times[n_t // 2]))


def main():
    print("scale,n_cells,n_genes,density,k,t,wall_ms", flush=True)

    for scale_name, n_cells, n_genes, density, k, t in SCALES:
        print(f"[ref] Building synthetic {scale_name}: {n_genes} genes × "
              f"{n_cells} cells, density={density:.0%}, k={k}, t={t}...",
              file=sys.stderr, flush=True)

        X   = make_sparse_csc(n_genes, n_cells, density, seed=42)
        emb = make_pca_embedding(n_cells, n_dims=50, seed=137)

        print(f"[ref] {scale_name}: X.nnz={X.nnz} "
              f"(actual density={X.nnz / (n_genes * n_cells):.2%}). "
              f"Building Markov matrix (untimed)...",
              file=sys.stderr, flush=True)

        try:
            M = build_markov_matrix(emb, k=k)
            print(f"[ref] {scale_name}: M.nnz={M.nnz}. "
                  f"Timing t={t} SpMM diffusion...",
                  file=sys.stderr, flush=True)

            wall_ms = time_magic(M, X, t)
        except Exception as exc:
            print(f"[ref] ERROR {scale_name}: {exc}", file=sys.stderr, flush=True)
            wall_ms = -1.0

        print(f"{scale_name},{n_cells},{n_genes},{density:.2f},{k},{t},{wall_ms:.3f}",
              flush=True)


if __name__ == "__main__":
    main()
