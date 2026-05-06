"""Cell fate probability estimation via absorption probabilities.

Provides singlet.fate_probabilities() — computes the probability of each
cell reaching each terminal fate using a directed kNN graph (Markov chain
absorption probabilities).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def fate_probabilities(
    adata: ad.AnnData,
    terminal_states: dict | str,
    *,
    pseudotime_key: str = "dpt_pseudotime",
    n_neighbors: int = 15,
) -> ad.AnnData:
    """Compute cell fate probabilities via Markov chain absorption.

    For each cell, estimates the probability of reaching each terminal state
    by computing absorption probabilities on a pseudotime-directed kNN graph.
    Cells flow from low pseudotime to high pseudotime.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Must have pseudotime in ``adata.obs[pseudotime_key]``
        and a PCA embedding in ``adata.obsm['X_pca']``.
    terminal_states : dict or str
        If dict: mapping of fate name → list of cell indices defining each
        terminal state (e.g., ``{'Monocyte': [10, 20, 30], 'T cell': [100, 110]}``).
        If str: key in ``adata.obs`` containing terminal state labels (NaN for
        non-terminal cells).
    pseudotime_key : str, default 'dpt_pseudotime'
        Key in ``adata.obs`` for pseudotime values used to direct edges.
    n_neighbors : int, default 15
        Number of nearest neighbors for kNN graph construction.

    Returns
    -------
    anndata.AnnData
        Returns the same adata with:
        - ``adata.obsm['fate_probabilities']``: DataFrame (n_cells × n_fates)
        - ``adata.uns['fate_probabilities']``: Dict with parameters and fate names

    Notes
    -----
    The algorithm:
    1. Build kNN graph in PCA space
    2. Direct edges using pseudotime (keep only edges going forward in time)
    3. Normalize rows to create transition probability matrix
    4. Compute absorption probabilities for terminal states via linear system

    Examples
    --------
    >>> import singlet
    >>> singlet.fate_probabilities(
    ...     adata,
    ...     terminal_states={'Monocyte': [10, 20], 'T cell': [100, 110]},
    ... )
    >>> adata.obsm['fate_probabilities']  # DataFrame with fate columns
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from sklearn.neighbors import NearestNeighbors

    if not hasattr(adata, "X") or not hasattr(adata, "obs"):
        raise TypeError(
            f"fate_probabilities() requires an AnnData object, got {type(adata).__name__}"
        )

    if pseudotime_key not in adata.obs.columns:
        raise KeyError(
            f"Pseudotime key '{pseudotime_key}' not found in adata.obs. "
            f"Run singlet.dpt() or provide pseudotime first."
        )

    # Get pseudotime
    pseudotime = adata.obs[pseudotime_key].values.astype(np.float64)

    # Parse terminal states
    if isinstance(terminal_states, str):
        # obs key containing labels
        if terminal_states not in adata.obs.columns:
            raise KeyError(f"Terminal states key '{terminal_states}' not found in adata.obs.")
        labels = adata.obs[terminal_states]
        fate_names = sorted(labels.dropna().unique().tolist())
        terminal_dict = {}
        for fate in fate_names:
            terminal_dict[fate] = np.where(labels == fate)[0].tolist()
    elif isinstance(terminal_states, dict):
        terminal_dict = terminal_states
        fate_names = sorted(terminal_dict.keys())
    else:
        raise TypeError(
            f"terminal_states must be a dict or str obs key, got {type(terminal_states).__name__}"
        )

    if len(fate_names) < 1:
        raise ValueError("At least one terminal state is required.")

    n_cells = adata.n_obs

    # Get embedding for kNN
    if "X_pca" in adata.obsm:
        X_embed = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    elif "X_umap" in adata.obsm:
        X_embed = np.asarray(adata.obsm["X_umap"], dtype=np.float64)
    else:
        # Fall back to expression data
        X_raw = adata.X
        if sp.issparse(X_raw):
            X_embed = X_raw.toarray().astype(np.float64)
        else:
            X_embed = np.asarray(X_raw, dtype=np.float64)

    # Build kNN graph
    nn = NearestNeighbors(n_neighbors=min(n_neighbors, n_cells - 1), metric="euclidean")
    nn.fit(X_embed)
    distances, indices = nn.kneighbors(X_embed)

    # Build directed transition matrix (edges go forward in pseudotime)
    rows = []
    cols = []
    vals = []

    for cell_idx in range(n_cells):
        pt_cell = pseudotime[cell_idx]
        if np.isnan(pt_cell):
            continue

        neighbor_indices = indices[cell_idx]
        neighbor_distances = distances[cell_idx]

        # Keep only neighbors with higher pseudotime (forward edges)
        forward_mask = pseudotime[neighbor_indices] >= pt_cell
        fwd_neighbors = neighbor_indices[forward_mask]
        fwd_distances = neighbor_distances[forward_mask]

        if len(fwd_neighbors) == 0:
            # Terminal-like cell: self-loop
            rows.append(cell_idx)
            cols.append(cell_idx)
            vals.append(1.0)
        else:
            # Weight by inverse distance (closer neighbors get higher weight)
            weights = 1.0 / (fwd_distances + 1e-10)
            weights /= weights.sum()

            for neighbor, weight in zip(fwd_neighbors, weights):
                rows.append(cell_idx)
                cols.append(neighbor)
                vals.append(weight)

    # Transition matrix T
    transition = sp.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))

    # Identify terminal (absorbing) cell indices
    absorbing_set = set()
    for fate in fate_names:
        for idx in terminal_dict[fate]:
            absorbing_set.add(idx)
    absorbing_indices = sorted(absorbing_set)

    # Transient cells
    all_indices = np.arange(n_cells)
    transient_mask = np.ones(n_cells, dtype=bool)
    transient_mask[absorbing_indices] = False
    transient_indices = all_indices[transient_mask]

    n_transient = len(transient_indices)
    n_absorbing = len(absorbing_indices)

    # Build index mappings
    absorbing_map = {idx: pos for pos, idx in enumerate(absorbing_indices)}

    # Extract Q (transient→transient) and R (transient→absorbing)
    if n_transient == 0:
        # All cells are terminal
        fate_probs = np.zeros((n_cells, len(fate_names)), dtype=np.float64)
        for fi, fate in enumerate(fate_names):
            for idx in terminal_dict[fate]:
                fate_probs[idx, fi] = 1.0
    else:
        # Q submatrix: transitions among transient states
        q_matrix = transition[np.ix_(transient_indices, transient_indices)]
        # R submatrix: transitions from transient to absorbing states
        r_matrix = transition[np.ix_(transient_indices, absorbing_indices)]

        # Absorption probabilities: solve (I - Q) * B = R
        # B[i, j] = probability that transient cell i is absorbed by absorbing cell j
        identity_q = sp.eye(n_transient, format="csr")
        lhs = identity_q - q_matrix

        # Solve the linear system
        from scipy.sparse.linalg import spsolve

        # Solve column by column for numerical stability
        absorption = np.zeros((n_transient, n_absorbing), dtype=np.float64)
        lhs_csc = lhs.tocsc()

        for col_idx in range(n_absorbing):
            rhs_col = np.asarray(r_matrix[:, col_idx].todense()).ravel()
            if rhs_col.sum() > 0:
                try:
                    sol = spsolve(lhs_csc, rhs_col)
                    # Handle NaN from singular matrix
                    sol = np.nan_to_num(sol, nan=0.0)
                    absorption[:, col_idx] = sol
                except Exception:
                    # Fallback: least squares
                    from scipy.sparse.linalg import lsqr

                    result = lsqr(lhs_csc, rhs_col)
                    absorption[:, col_idx] = np.nan_to_num(result[0], nan=0.0)

        # Aggregate absorption probs by fate
        fate_probs = np.zeros((n_cells, len(fate_names)), dtype=np.float64)

        for fi, fate in enumerate(fate_names):
            fate_cell_indices = terminal_dict[fate]
            for cell_idx in fate_cell_indices:
                if cell_idx in absorbing_map:
                    abs_pos = absorbing_map[cell_idx]
                    fate_probs[transient_indices, fi] += absorption[:, abs_pos]
            # Terminal cells of this fate get probability 1.0
            for idx in fate_cell_indices:
                fate_probs[idx, fi] = 1.0

    # Normalize rows to sum to 1 (handle numerical issues)
    row_sums = fate_probs.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    fate_probs = fate_probs / row_sums

    # Clip to [0, 1]
    fate_probs = np.clip(fate_probs, 0.0, 1.0)

    # Store as DataFrame in obsm
    fate_df = pd.DataFrame(
        fate_probs.astype(np.float32),
        index=adata.obs_names,
        columns=fate_names,
    )
    adata.obsm["fate_probabilities"] = fate_df

    adata.uns["fate_probabilities"] = {
        "params": {
            "pseudotime_key": pseudotime_key,
            "n_neighbors": n_neighbors,
        },
        "fate_names": fate_names,
    }

    return adata
