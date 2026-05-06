"""Inter-cluster connectivity scoring for AnnData objects.

Provides singlet.connectivity_score() — compute pairwise connectivity
between cell clusters based on shared k-nearest-neighbor (kNN) edges.
Useful for understanding cluster relationships and graph topology.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def connectivity_score(
    adata: AnnData,
    groupby: str,
    *,
    method: str = "jaccard",
) -> pd.DataFrame:
    """Compute pairwise connectivity between clusters via shared kNN edges.

    For each pair of clusters (A, B), measures how connected they are
    in the kNN graph by counting shared inter-cluster edges.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix with a precomputed kNN graph in
        ``adata.obsp['connectivities']`` (e.g., via ``singlet.neighbors()``).
    groupby : str
        Key in ``adata.obs`` containing cluster/group labels.
    method : str, default 'jaccard'
        Scoring method:
        - 'jaccard': |edges(A,B)| / |edges(A,*) ∪ edges(*,B)|
        - 'overlap': |edges(A,B)| / min(|edges(A,*)|, |edges(*,B)|)
        - 'cosine': edges(A,B) / sqrt(|edges(A,*)|·|edges(*,B)|)

    Returns
    -------
    pandas.DataFrame
        Square DataFrame of shape (n_groups, n_groups) with connectivity
        scores. Diagonal is 1.0. Also stored in
        ``adata.uns[f'{groupby}_connectivity']``.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> singlet.leiden(adata)
    >>> conn = singlet.connectivity_score(adata, "leiden")
    >>> conn.shape[0] == adata.obs['leiden'].nunique()
    True
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    if method not in ("jaccard", "overlap", "cosine"):
        msg = f"method must be 'jaccard', 'overlap', or 'cosine', got {method!r}"
        raise ValueError(msg)

    if groupby not in adata.obs.columns:
        msg = f"Column {groupby!r} not found in adata.obs"
        raise KeyError(msg)

    if "connectivities" not in adata.obsp:
        msg = (
            "No kNN graph found in adata.obsp['connectivities']. "
            "Run singlet.neighbors(adata) first."
        )
        raise KeyError(msg)

    # Get adjacency and labels
    adj = adata.obsp["connectivities"]
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    else:
        adj = adj.tocsr()

    labels = adata.obs[groupby].astype(str).values
    unique_labels = np.sort(np.unique(labels))
    n_groups = len(unique_labels)
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}

    # Build group membership indicator matrix (n_cells x n_groups)
    cell_indices = np.array([label_to_idx[lab] for lab in labels])
    indicator = sp.csc_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), cell_indices)),
        shape=(len(labels), n_groups),
    )

    # Compute inter-group edge counts: G[i,j] = number of edges between group i and j
    # G = indicator.T @ adj @ indicator
    edge_counts = (indicator.T @ adj @ indicator).toarray().astype(np.float64)

    # Total edges incident to each group (sum of row in edge_counts)
    group_totals = edge_counts.sum(axis=1)

    # Compute pairwise scores
    scores = np.zeros((n_groups, n_groups), dtype=np.float64)

    for row_idx in range(n_groups):
        for col_idx in range(n_groups):
            if row_idx == col_idx:
                scores[row_idx, col_idx] = 1.0
                continue

            shared = edge_counts[row_idx, col_idx]

            if method == "jaccard":
                union = group_totals[row_idx] + group_totals[col_idx] - shared
                scores[row_idx, col_idx] = shared / union if union > 0 else 0.0
            elif method == "overlap":
                min_total = min(group_totals[row_idx], group_totals[col_idx])
                scores[row_idx, col_idx] = shared / min_total if min_total > 0 else 0.0
            elif method == "cosine":
                denom = np.sqrt(group_totals[row_idx] * group_totals[col_idx])
                scores[row_idx, col_idx] = shared / denom if denom > 0 else 0.0

    result = pd.DataFrame(scores, index=unique_labels, columns=unique_labels)
    result.index.name = groupby
    result.columns.name = groupby

    # Store in adata.uns
    adata.uns[f"{groupby}_connectivity"] = result

    return result
