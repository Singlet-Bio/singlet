"""Partition-based graph abstraction (PAGA)."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def paga(
    adata: AnnData,
    *,
    groups: str = "leiden",
    threshold: float = 0.01,
    copy: bool = False,
) -> AnnData | None:
    """Compute PAGA connectivity between groups.

    Measures inter-group connectivity relative to expected random connectivity.
    Requires neighbors to be computed.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groups
        Key in .obs containing group assignments.
    threshold
        Minimum connectivity to retain (edges below this are set to 0).
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores results in .uns['paga']:
        - 'connectivities': connectivity matrix between groups
        - 'connectivities_tree': minimum spanning tree of connectivities
        - 'groups': the groupby key used
    """
    from scipy.sparse import csr_matrix, issparse
    from scipy.sparse.csgraph import minimum_spanning_tree

    adata = adata.copy() if copy else adata

    if "connectivities" not in adata.obsp:
        raise KeyError("'connectivities' not found in .obsp. Run singlet.neighbors() first.")

    if groups not in adata.obs.columns:
        raise KeyError(f"'{groups}' not found in .obs columns.")

    conn = adata.obsp["connectivities"]
    group_labels = adata.obs[groups]

    # Get unique groups
    if hasattr(group_labels, "cat"):
        unique_groups = list(group_labels.cat.categories)
    else:
        unique_groups = sorted(group_labels.unique(), key=str)

    n_groups = len(unique_groups)
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}

    # Assign each cell to a group index
    cell_groups = np.array([group_to_idx[g] for g in group_labels])

    # Count cells per group
    n_cells_per_group = np.zeros(n_groups)
    for i in range(n_groups):
        n_cells_per_group[i] = np.sum(cell_groups == i)

    # Compute inter-group and intra-group connectivity
    inter_conn = np.zeros((n_groups, n_groups))

    if issparse(conn):
        conn_coo = conn.tocoo()
        for i, j, v in zip(conn_coo.row, conn_coo.col, conn_coo.data):
            gi = cell_groups[i]
            gj = cell_groups[j]
            inter_conn[gi, gj] += v
    else:
        for i in range(conn.shape[0]):
            for j in range(conn.shape[1]):
                if conn[i, j] > 0:
                    gi = cell_groups[i]
                    gj = cell_groups[j]
                    inter_conn[gi, gj] += conn[i, j]

    # Normalize: divide by expected connectivity
    # Expected = n_i * n_j for inter-group, n_i * (n_i - 1) for intra
    connectivity = np.zeros((n_groups, n_groups))
    for i in range(n_groups):
        for j in range(i, n_groups):
            if i == j:
                expected = n_cells_per_group[i] * (n_cells_per_group[i] - 1)
            else:
                expected = 2 * n_cells_per_group[i] * n_cells_per_group[j]

            if expected > 0:
                raw = inter_conn[i, j] + inter_conn[j, i] if i != j else inter_conn[i, j]
                connectivity[i, j] = raw / expected
                connectivity[j, i] = connectivity[i, j]

    # Apply threshold
    connectivity[connectivity < threshold] = 0

    # Compute minimum spanning tree (on negative weights for max spanning tree)
    conn_sparse = csr_matrix(connectivity)
    if conn_sparse.nnz > 0:
        mst = minimum_spanning_tree(-conn_sparse)
        mst = -mst
        mst = mst.toarray()
        # Symmetrize
        tree = mst + mst.T
    else:
        tree = np.zeros((n_groups, n_groups))

    # Store results
    adata.uns["paga"] = {
        "connectivities": csr_matrix(connectivity),
        "connectivities_tree": csr_matrix(tree),
        "groups": groups,
    }

    return adata if copy else None
