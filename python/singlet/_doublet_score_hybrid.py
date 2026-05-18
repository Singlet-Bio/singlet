# SPDX-License-Identifier: MIT
"""Hybrid doublet detection combining multiple signals.

Provides singlet.doublet_score_hybrid() — combines simulated doublet kNN
fraction, library size outlier score, and gene complexity outlier score
for robust doublet detection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from anndata import AnnData


def doublet_score_hybrid(
    adata: "AnnData",
    methods: Optional[list] = None,
    *,
    n_neighbors: int = 30,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> "AnnData":
    """Hybrid doublet detection combining multiple signals.

    Computes a weighted average of three normalized doublet scores:
    1. Simulated doublet kNN fraction (Scrublet-style)
    2. Library size outlier score
    3. Gene complexity outlier score

    Parameters
    ----------
    adata
        Annotated data matrix (raw counts recommended).
    methods
        List of methods to combine. Options: 'simulation', 'library_size',
        'complexity'. Default None uses all three.
    n_neighbors
        Number of neighbors for kNN graph. Default 30.
    expected_doublet_rate
        Expected fraction of doublets. Used to set threshold. Default 0.06.
    random_state
        Random seed for reproducibility. Default 0.

    Returns
    -------
    AnnData
        Returns adata with:
        - adata.obs['doublet_score_hybrid']: combined score in [0, 1]
        - adata.obs['predicted_doublet_hybrid']: boolean prediction

    Raises
    ------
    TypeError
        If adata is not an AnnData object.
    ValueError
        If methods contains invalid entries or n_neighbors < 1.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.doublet_score_hybrid(adata)
    >>> adata.obs['predicted_doublet_hybrid'].sum()  # number of predicted doublets
    """
    import numpy as np
    import scipy.sparse as sp

    # Validate input
    if not hasattr(adata, "obs"):
        msg = "doublet_score_hybrid requires an AnnData object, got " + type(adata).__name__
        raise TypeError(msg)

    valid_methods = ("simulation", "library_size", "complexity")
    if methods is None:
        methods = list(valid_methods)
    else:
        for meth in methods:
            if meth not in valid_methods:
                msg = f"Invalid method '{meth}'. Valid methods: {valid_methods}"
                raise ValueError(msg)

    if n_neighbors < 1:
        msg = f"n_neighbors must be >= 1, got {n_neighbors}"
        raise ValueError(msg)

    if expected_doublet_rate <= 0 or expected_doublet_rate >= 1:
        msg = f"expected_doublet_rate must be in (0, 1), got {expected_doublet_rate}"
        raise ValueError(msg)

    rng = np.random.default_rng(random_state)
    n_cells = adata.n_obs

    # Get expression matrix
    expr = adata.X
    if sp.issparse(expr):
        expr_dense = expr.toarray()
    else:
        expr_dense = np.asarray(expr, dtype=np.float64)

    scores = {}
    weights = {}

    if "simulation" in methods:
        scores["simulation"] = _simulation_score(expr_dense, n_neighbors=n_neighbors, rng=rng)
        weights["simulation"] = 0.5

    if "library_size" in methods:
        scores["library_size"] = _library_size_score(expr_dense)
        weights["library_size"] = 0.25

    if "complexity" in methods:
        scores["complexity"] = _complexity_score(expr_dense)
        weights["complexity"] = 0.25

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight

    # Weighted average of scores
    combined = np.zeros(n_cells)
    for key in scores:
        combined += weights[key] * scores[key]

    # Threshold: use quantile based on expected doublet rate
    threshold = np.quantile(combined, 1 - expected_doublet_rate)

    adata.obs["doublet_score_hybrid"] = combined
    adata.obs["predicted_doublet_hybrid"] = combined >= threshold

    return adata


def _simulation_score(
    expr,
    n_neighbors: int,
    rng,
):
    """Compute doublet score by simulating doublets and measuring kNN overlap.

    Creates synthetic doublets by averaging random pairs, builds a kNN graph
    on the combined real+synthetic data, then scores each real cell by
    the fraction of its neighbors that are synthetic doublets.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    n_cells, n_genes = expr.shape

    # Number of simulated doublets
    n_sim = min(n_cells, max(100, int(n_cells * 0.5)))

    # Simulate doublets by summing random pairs
    idx1 = rng.integers(0, n_cells, size=n_sim)
    idx2 = rng.integers(0, n_cells, size=n_sim)
    simulated = expr[idx1] + expr[idx2]

    # Simple log-normalization for distance computation
    combined = np.vstack([expr, simulated])
    lib_sizes = combined.sum(axis=1, keepdims=True)
    lib_sizes = np.where(lib_sizes > 0, lib_sizes, 1.0)
    normalized = np.log1p(combined / lib_sizes * 10000)

    # Reduce dimensions with truncated SVD for speed
    n_components = min(30, n_genes - 1, n_cells - 1)
    if n_components < 2:
        n_components = 2

    # Center
    mean_vec = normalized.mean(axis=0)
    centered = normalized - mean_vec

    # SVD via randomized approach (simple power iteration)
    u_mat, s_vec, _ = np.linalg.svd(centered, full_matrices=False)
    reduced = u_mat[:, :n_components] * s_vec[:n_components]

    # Build kNN on combined space
    k_use = min(n_neighbors, len(reduced) - 1)
    tree = cKDTree(reduced)
    _, indices = tree.query(reduced[:n_cells], k=k_use + 1)

    # Score: fraction of neighbors that are simulated doublets
    neighbor_indices = indices[:, 1:]  # exclude self
    is_simulated = neighbor_indices >= n_cells
    sim_scores = is_simulated.mean(axis=1)

    # Normalize to [0, 1]
    score_min = sim_scores.min()
    score_max = sim_scores.max()
    if score_max > score_min:
        sim_scores = (sim_scores - score_min) / (score_max - score_min)
    else:
        sim_scores = np.zeros_like(sim_scores)

    return sim_scores


def _library_size_score(expr):
    """Score cells by library size deviation (doublets tend to have 2x reads)."""
    import numpy as np

    lib_sizes = expr.sum(axis=1).astype(np.float64)

    if lib_sizes.std() == 0:
        return np.zeros(len(lib_sizes))

    # Z-score of log library size
    log_lib = np.log1p(lib_sizes)
    z_scores = (log_lib - log_lib.mean()) / (log_lib.std() + 1e-10)

    # Convert to [0, 1] score — higher library size = more likely doublet
    # Use sigmoid-like transform on positive z-scores
    scores = np.clip(z_scores / 4.0, 0, 1)  # z=4 maps to 1.0

    return scores


def _complexity_score(expr):
    """Score cells by gene complexity (doublets express more genes)."""
    import numpy as np

    # Number of detected genes per cell
    n_genes_per_cell = (expr > 0).sum(axis=1).astype(np.float64)

    if n_genes_per_cell.std() == 0:
        return np.zeros(len(n_genes_per_cell))

    # Z-score
    z_scores = (n_genes_per_cell - n_genes_per_cell.mean()) / (n_genes_per_cell.std() + 1e-10)

    # Convert to [0, 1] — higher complexity = more likely doublet
    scores = np.clip(z_scores / 4.0, 0, 1)

    return scores
