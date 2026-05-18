# SPDX-License-Identifier: MIT
"""Harmony batch correction for AnnData objects.

Implements the Harmony algorithm (Korsunsky et al., Nature Methods 2019)
for integrating single-cell data across batches. Operates on PCA embeddings
and produces corrected embeddings suitable for downstream clustering/UMAP.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


def harmony(
    adata,
    key: str,
    *,
    n_comps: Optional[int] = None,
    max_iter: int = 20,
    max_iter_cluster: int = 200,
    n_clusters: Optional[int] = None,
    sigma: float = 0.1,
    theta: float = 2.0,
    block_size: float = 0.05,
    tol: float = 1e-4,
    random_state: int = 0,
    inplace: bool = True,
) -> Optional["np.ndarray"]:
    """Integrate data across batches using Harmony.

    Corrects PCA embeddings (adata.obsm['X_pca']) by iteratively adjusting
    for batch effects while preserving biological variation.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have 'X_pca' in obsm.
    key : str
        Column in adata.obs identifying batch membership.
    n_comps : int or None, default None
        Number of PCA components to use. If None, uses all available.
    max_iter : int, default 20
        Maximum number of Harmony iterations.
    max_iter_cluster : int, default 200
        Maximum iterations for k-means clustering step.
    n_clusters : int or None, default None
        Number of clusters for soft clustering. If None, uses
        min(n_cells / 30, 100), clamped to [2, 200].
    sigma : float, default 0.1
        Bandwidth of soft clustering kernel.
    theta : float, default 2.0
        Diversity penalty parameter. Higher values = more aggressive
        batch correction. Set to 0 to disable diversity penalty.
    block_size : float, default 0.05
        Proportion of cells to update per iteration for online learning.
    tol : float, default 1e-4
        Convergence tolerance (objective function change).
    random_state : int, default 0
        Random seed for reproducibility.
    inplace : bool, default True
        If True, stores corrected embedding in adata.obsm['X_pca_harmony'].
        If False, returns the corrected embedding array.

    Returns
    -------
    numpy.ndarray or None
        Corrected PCA embeddings (n_cells × n_comps) if inplace=False.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.pca(adata)
    >>> singlet.harmony(adata, key="batch")
    >>> 'X_pca_harmony' in adata.obsm
    True
    """
    import numpy as np

    if not hasattr(adata, "obsm"):
        raise TypeError(f"harmony() requires an AnnData object, got {type(adata).__name__}")

    if "X_pca" not in adata.obsm:
        raise KeyError("adata.obsm['X_pca'] not found. Run singlet.pca() first.")

    if key not in adata.obs.columns:
        raise KeyError(f"'{key}' not found in adata.obs.columns")

    Z = np.array(adata.obsm["X_pca"], dtype=np.float64)
    n_cells, n_pcs = Z.shape

    if n_comps is not None:
        n_comps = min(n_comps, n_pcs)
        Z = Z[:, :n_comps]
    else:
        n_comps = n_pcs

    # Encode batch membership as one-hot matrix
    batch_labels = adata.obs[key].values
    unique_batches = np.unique(batch_labels)
    n_batches = len(unique_batches)

    if n_batches < 2:
        # Nothing to correct
        if inplace:
            adata.obsm["X_pca_harmony"] = Z.copy()
            return None
        return Z.copy()

    # One-hot: Phi (n_batches × n_cells)
    batch_to_idx = {b: i for i, b in enumerate(unique_batches)}
    Phi = np.zeros((n_batches, n_cells), dtype=np.float64)
    for i, label in enumerate(batch_labels):
        Phi[batch_to_idx[label], i] = 1.0

    # Batch proportions
    N_b = Phi.sum(axis=1)  # cells per batch
    Pr_b = N_b / n_cells

    # Determine number of clusters
    if n_clusters is None:
        n_clusters = int(np.clip(n_cells / 30, 2, 200))

    Z_corrected = _run_harmony(
        Z,
        Phi,
        Pr_b,
        n_clusters,
        sigma,
        theta,
        block_size,
        max_iter,
        max_iter_cluster,
        tol,
        random_state,
    )

    if inplace:
        adata.obsm["X_pca_harmony"] = Z_corrected
        return None
    return Z_corrected


def _run_harmony(
    Z_orig, Phi, Pr_b, K, sigma, theta, block_size, max_iter, max_iter_cluster, tol, random_state
):
    """Core Harmony algorithm.

    Parameters
    ----------
    Z_orig : ndarray (n_cells × n_comps)
        Original PCA embeddings.
    Phi : ndarray (n_batches × n_cells)
        One-hot batch encoding.
    Pr_b : ndarray (n_batches,)
        Batch proportions.
    K : int
        Number of clusters.
    sigma : float
        Soft-clustering bandwidth.
    theta : float
        Diversity penalty.
    block_size : float
        Proportion of cells updated per mini-iteration.
    max_iter : int
        Max outer iterations.
    max_iter_cluster : int
        Max clustering iterations.
    tol : float
        Convergence tolerance.
    random_state : int
        RNG seed.

    Returns
    -------
    ndarray (n_cells × n_comps)
        Corrected embeddings.
    """
    import numpy as np

    rng = np.random.default_rng(random_state)
    n_cells, n_comps = Z_orig.shape

    # L2-normalize rows of Z for cosine-like distances
    Z_cos = Z_orig / (np.linalg.norm(Z_orig, axis=1, keepdims=True) + 1e-10)

    # Initialize cluster centroids via kmeans++
    centroids = _kmeans_plusplus(Z_cos, K, rng)

    # Soft cluster assignment: R (K × n_cells)
    R = _compute_R(Z_cos, centroids, sigma, Phi, Pr_b, theta)

    # Keep corrected embeddings
    Z_corr = Z_orig.copy()

    obj_prev = _objective(Z_cos, centroids, R, sigma, Phi, Pr_b, theta)

    for iteration in range(max_iter):
        # Step 1: Update centroids and cluster assignment
        for _ in range(max_iter_cluster):
            # Update centroids: weighted mean of assigned cells
            # centroids[k] = sum(R[k,i] * Z_cos[i]) / sum(R[k,i])
            R_sum = R.sum(axis=1, keepdims=True) + 1e-10  # (K, 1)
            centroids = (R @ Z_cos) / R_sum  # (K × n_comps)
            # Re-normalize centroids
            centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10

            # Update R
            R_new = _compute_R(Z_cos, centroids, sigma, Phi, Pr_b, theta)

            # Check cluster convergence
            if np.max(np.abs(R_new - R)) < tol:
                R = R_new
                break
            R = R_new

        # Step 2: Correction — regress out batch within each cluster
        Z_corr = _correct(Z_orig, R, Phi, theta)

        # Update Z_cos from corrected
        Z_cos = Z_corr / (np.linalg.norm(Z_corr, axis=1, keepdims=True) + 1e-10)

        # Check outer convergence
        obj = _objective(Z_cos, centroids, R, sigma, Phi, Pr_b, theta)
        if abs(obj - obj_prev) < tol * abs(obj_prev + 1e-10):
            break
        obj_prev = obj

    return Z_corr


def _kmeans_plusplus(X, K, rng):
    """Initialize K centroids using kmeans++ on rows of X."""
    import numpy as np

    n = X.shape[0]
    centroids = np.empty((K, X.shape[1]), dtype=np.float64)

    # First centroid: random
    idx = rng.integers(n)
    centroids[0] = X[idx]

    for k in range(1, K):
        # Distance to nearest existing centroid
        dists = 1.0 - X @ centroids[:k].T  # cosine distance
        min_dists = dists.min(axis=1)
        min_dists = np.clip(min_dists, 0, None)

        # Probability proportional to distance squared
        probs = min_dists**2
        total = probs.sum()
        if total < 1e-15:
            idx = rng.integers(n)
        else:
            probs /= total
            idx = rng.choice(n, p=probs)
        centroids[k] = X[idx]

    return centroids


def _compute_R(Z_cos, centroids, sigma, Phi, Pr_b, theta):
    """Compute soft cluster assignment matrix R (K × n_cells).

    Includes diversity penalty to encourage mixing batches within clusters.
    """
    import numpy as np

    # Similarity: dot product (cosine similarity since both normalized)
    # (K × n_cells)
    sim = centroids @ Z_cos.T  # cosine similarity

    # Kernel
    R = np.exp(sim / sigma)

    # Diversity penalty: penalize clusters that are batch-enriched
    if theta > 0:
        # Expected batch proportions per cluster vs actual
        observed = R @ Phi.T  # (K × n_batches)
        # Expected: E_kb = R_k_total * Pr_b
        R_totals = R.sum(axis=1, keepdims=True) + 1e-10  # (K, 1)
        expected = R_totals * Pr_b[np.newaxis, :]  # (K × n_batches)

        # Penalty: (O/E)^theta -> penalize over-representation
        ratio = (observed + 1e-10) / (expected + 1e-10)
        penalty = ratio**theta  # (K × n_batches)

        # Apply penalty per cell based on its batch
        # For cell i with batch b: multiply R[k,i] by 1/penalty[k,b]
        penalty_per_cell = (1.0 / penalty) @ Phi  # (K × n_cells)
        R *= penalty_per_cell

    # Normalize columns to sum to 1
    R /= R.sum(axis=0, keepdims=True) + 1e-10

    return R


def _correct(Z_orig, R, Phi, theta):
    """Correct embeddings by regressing out batch effect within clusters.

    For each cluster k, estimates batch effect via ridge regression
    and subtracts it.
    """
    import numpy as np

    n_cells, n_comps = Z_orig.shape
    K = R.shape[0]

    Z_corr = Z_orig.copy()

    for k in range(K):
        # Weights for this cluster
        w = R[k]  # (n_cells,)

        # Weighted batch design: Phi_w (n_batches × n_cells) * w
        Phi_w = Phi * w[np.newaxis, :]  # weight by cluster membership

        # Weighted batch means for each PC
        # batch_sums = Phi_w @ Z_orig -> (n_batches × n_comps)
        batch_sums = Phi_w @ Z_orig  # (n_batches × n_comps)
        batch_weights = Phi_w.sum(axis=1, keepdims=True) + 1e-10  # (n_batches, 1)
        batch_means = batch_sums / batch_weights

        # Overall weighted mean for this cluster
        total_weight = w.sum() + 1e-10
        overall_mean = (w @ Z_orig) / total_weight  # (n_comps,)

        # Batch effect = batch_mean - overall_mean
        batch_effect = batch_means - overall_mean[np.newaxis, :]  # (n_batches × n_comps)

        # Subtract weighted batch effect from each cell
        # correction for cell i = w[i] * batch_effect[batch_of_i]
        correction = (Phi.T @ batch_effect) * w[:, np.newaxis]  # (n_cells × n_comps)
        Z_corr -= correction

    return Z_corr


def _objective(Z_cos, centroids, R, sigma, Phi, Pr_b, theta):
    """Compute Harmony objective (negative log-likelihood)."""
    import numpy as np

    sim = centroids @ Z_cos.T  # (K × n_cells)
    # Weighted cosine distance
    obj = -np.sum(R * sim) / sigma
    return obj
