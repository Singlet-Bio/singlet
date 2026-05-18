# SPDX-License-Identifier: MIT
"""Cell-level differential expression.

Provides singlet.cell_level_de() — compute local differential expression
for each cell by comparing its neighborhood across conditions, enabling
analysis of heterogeneous treatment responses.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def cell_level_de(
    adata: AnnData,
    *,
    condition_key: str,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
    n_top_genes: int = 50,
) -> AnnData:
    """Compute cell-level differential expression across conditions.

    For each cell, identifies its local neighborhood and computes
    log fold-changes between cells in condition A vs condition B within
    that neighborhood. This reveals heterogeneous treatment responses
    at single-cell resolution.

    Parameters
    ----------
    adata
        Annotated data matrix with a representation in ``.obsm[use_rep]``.
        Must have exactly two conditions in ``adata.obs[condition_key]``.
    condition_key
        Column in ``adata.obs`` containing condition labels.
        Must have exactly two unique values.
    use_rep
        Key in ``adata.obsm`` for computing the neighborhood graph.
    n_neighbors
        Number of neighbors for the local neighborhood.
    n_top_genes
        Number of top variable genes to compute cell-level DE for.

    Returns
    -------
    AnnData
        The input ``adata`` with added:
        - ``adata.layers['cell_level_lfc']``: sparse matrix of per-cell
          log2 fold-changes for top genes (n_cells × n_vars, nonzero only
          for the n_top_genes columns).
        - ``adata.uns['cell_level_de_genes']``: list of gene names analyzed.
        - ``adata.uns['cell_level_de_params']``: dict of parameters used.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.cell_level_de(adata, condition_key="treatment")
    >>> # Per-cell fold-changes for top genes
    >>> adata.layers["cell_level_lfc"]
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial import cKDTree

    # Validate inputs
    if condition_key not in adata.obs.columns:
        msg = f"Key {condition_key!r} not found in adata.obs"
        raise KeyError(msg)

    if use_rep not in adata.obsm:
        msg = f"Representation {use_rep!r} not found in adata.obsm"
        raise KeyError(msg)

    conditions = adata.obs[condition_key].unique()
    if len(conditions) != 2:
        msg = (
            f"condition_key must have exactly 2 unique values, "
            f"got {len(conditions)}: {list(conditions)}"
        )
        raise ValueError(msg)

    if n_neighbors < 1:
        msg = f"n_neighbors must be >= 1, got {n_neighbors}"
        raise ValueError(msg)

    if n_top_genes < 1:
        msg = f"n_top_genes must be >= 1, got {n_top_genes}"
        raise ValueError(msg)

    n_cells = adata.n_obs
    n_vars = adata.n_vars
    cond_a, cond_b = conditions[0], conditions[1]
    cond_labels = np.asarray(adata.obs[condition_key])
    mask_a = cond_labels == cond_a
    mask_b = cond_labels == cond_b

    # Select top variable genes
    n_top = min(n_top_genes, n_vars)
    expr_full = adata.X
    if sp.issparse(expr_full):
        # Compute variance for gene selection
        mean = np.asarray(expr_full.mean(axis=0)).ravel()
        mean_sq = np.asarray(expr_full.power(2).mean(axis=0)).ravel()
        variances = mean_sq - mean**2
    else:
        expr_arr = np.asarray(expr_full, dtype=np.float64)
        variances = np.var(expr_arr, axis=0)

    top_gene_idx = np.argsort(variances)[::-1][:n_top]
    top_gene_idx = np.sort(top_gene_idx)  # Keep original order
    top_gene_names = adata.var_names[top_gene_idx].tolist()

    # Extract expression for selected genes
    expr_selected = expr_full[:, top_gene_idx]
    if sp.issparse(expr_selected):
        expr_selected = np.asarray(expr_selected.todense())
    else:
        expr_selected = np.asarray(expr_selected, dtype=np.float64)

    # Build kNN graph
    rep = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    k = min(n_neighbors, n_cells - 1)
    tree = cKDTree(rep)
    _, nn_indices = tree.query(rep, k=k + 1)
    # Include self in neighborhood
    nn_indices = nn_indices[:, : k + 1]

    # Compute cell-level log fold-changes
    lfc_dense = np.zeros((n_cells, n_top), dtype=np.float32)
    pseudocount = 1e-2

    for cell_idx in range(n_cells):
        neighbors = nn_indices[cell_idx]

        # Split neighbors by condition
        nb_mask_a = mask_a[neighbors]
        nb_mask_b = mask_b[neighbors]

        n_a = nb_mask_a.sum()
        n_b = nb_mask_b.sum()

        if n_a == 0 or n_b == 0:
            # Cannot compute DE without cells from both conditions
            continue

        # Mean expression in each condition within neighborhood
        mean_a = expr_selected[neighbors[nb_mask_a]].mean(axis=0)
        mean_b = expr_selected[neighbors[nb_mask_b]].mean(axis=0)

        # Log2 fold-change (B vs A)
        lfc = np.log2((mean_b + pseudocount) / (mean_a + pseudocount))
        lfc_dense[cell_idx] = lfc.astype(np.float32)

    # Store as sparse matrix (full n_vars width)
    lfc_full = sp.lil_matrix((n_cells, n_vars), dtype=np.float32)
    for local_j, global_j in enumerate(top_gene_idx):
        col_data = lfc_dense[:, local_j]
        nonzero = np.nonzero(col_data)[0]
        for row in nonzero:
            lfc_full[row, global_j] = col_data[row]
    lfc_full = lfc_full.tocsr()

    # Store results
    adata.layers["cell_level_lfc"] = lfc_full
    adata.uns["cell_level_de_genes"] = top_gene_names
    adata.uns["cell_level_de_params"] = {
        "condition_key": condition_key,
        "conditions": [str(cond_a), str(cond_b)],
        "use_rep": use_rep,
        "n_neighbors": n_neighbors,
        "n_top_genes": n_top,
    }

    return adata
