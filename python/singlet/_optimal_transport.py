# SPDX-License-Identifier: MIT
"""Optimal transport for trajectory inference (Waddington-OT style).

Provides singlet.optimal_transport() — compute OT coupling matrix between
cells at two time points using the Sinkhorn algorithm with entropic
regularization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from anndata import AnnData


def optimal_transport(
    adata: AnnData,
    *,
    time_key: str,
    source_time: str | int | float,
    target_time: str | int | float,
    use_rep: str = "X_pca",
    epsilon: float = 0.05,
    max_iter: int = 100,
) -> np.ndarray:
    """Compute optimal transport coupling between cells at two time points.

    Uses the Sinkhorn algorithm with entropic regularization to find
    the transport plan between source and target cell populations,
    enabling trajectory inference in the style of Waddington-OT.

    Parameters
    ----------
    adata
        Annotated data matrix with a representation in ``.obsm[use_rep]``.
    time_key
        Column in ``adata.obs`` containing time point labels.
    source_time
        Value in ``adata.obs[time_key]`` identifying source cells.
    target_time
        Value in ``adata.obs[time_key]`` identifying target cells.
    use_rep
        Key in ``adata.obsm`` for the cell representation.
    epsilon
        Entropic regularization parameter. Smaller values give sparser
        transport plans but slower convergence. Typical range: 0.01–0.1.
    max_iter
        Maximum number of Sinkhorn iterations.

    Returns
    -------
    numpy.ndarray
        Coupling matrix of shape (n_source, n_target) where entry (i, j)
        represents the transport mass from source cell i to target cell j.
        Rows sum to 1/n_source (uniform source marginal).
        Also stored in ``adata.uns['ot_coupling_{source_time}_{target_time}']``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> coupling = singlet.optimal_transport(
    ...     adata, time_key="day", source_time=0, target_time=3
    ... )
    >>> coupling.shape  # (n_source_cells, n_target_cells)
    """
    import numpy as np

    # Validate inputs
    if time_key not in adata.obs.columns:
        msg = f"Key {time_key!r} not found in adata.obs"
        raise KeyError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm"
        raise KeyError(msg)

    if epsilon <= 0:
        msg = f"epsilon must be positive, got {epsilon}"
        raise ValueError(msg)

    if max_iter < 1:
        msg = f"max_iter must be >= 1, got {max_iter}"
        raise ValueError(msg)

    # Get cell indices for source and target
    time_values = adata.obs[time_key]
    source_mask = time_values == source_time
    target_mask = time_values == target_time

    n_source = int(source_mask.sum())
    n_target = int(target_mask.sum())

    if n_source == 0:
        msg = f"No cells found with {time_key}={source_time!r}"
        raise ValueError(msg)
    if n_target == 0:
        msg = f"No cells found with {time_key}={target_time!r}"
        raise ValueError(msg)

    # Extract representations
    rep = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    source_rep = rep[np.asarray(source_mask)]
    target_rep = rep[np.asarray(target_mask)]

    # Compute cost matrix (squared Euclidean distance)
    # C[i,j] = ||x_i - y_j||^2
    source_sq = np.sum(source_rep**2, axis=1, keepdims=True)
    target_sq = np.sum(target_rep**2, axis=1, keepdims=True)
    cost = source_sq + target_sq.T - 2.0 * source_rep @ target_rep.T

    # Ensure non-negative (numerical precision)
    np.maximum(cost, 0.0, out=cost)

    # Normalize cost to [0, 1] range for numerical stability
    cost_max = cost.max()
    if cost_max > 0:
        cost = cost / cost_max

    # Sinkhorn algorithm
    # K = exp(-C / epsilon)
    kernel = np.exp(-cost / epsilon)

    # Uniform marginals
    mu = np.ones(n_source, dtype=np.float64) / n_source
    nu = np.ones(n_target, dtype=np.float64) / n_target

    # Initialize scaling vectors
    u_vec = np.ones(n_source, dtype=np.float64)
    v_vec = np.ones(n_target, dtype=np.float64)

    # Sinkhorn iterations
    for _iteration in range(max_iter):
        u_prev = u_vec.copy()

        # Update v: v = nu / (K^T u)
        ktu = kernel.T @ u_vec
        ktu = np.maximum(ktu, 1e-300)  # avoid division by zero
        v_vec = nu / ktu

        # Update u: u = mu / (K v)
        kv = kernel @ v_vec
        kv = np.maximum(kv, 1e-300)
        u_vec = mu / kv

        # Check convergence
        change = np.max(np.abs(u_vec - u_prev))
        if change < 1e-9:
            break

    # Compute coupling: T = diag(u) @ K @ diag(v)
    coupling = u_vec[:, None] * kernel * v_vec[None, :]

    # Store in adata.uns
    key = f"ot_coupling_{source_time}_{target_time}"
    adata.uns[key] = coupling

    return coupling
