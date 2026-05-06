"""Differential abundance testing for AnnData objects.

Tests whether cell populations are differentially abundant between
conditions. Supports neighborhood-level testing (Milo-style) and
simple cluster-level contingency tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def differential_abundance(
    adata,
    groupby: str,
    condition_key: str,
    *,
    method: str = "milo",
    n_neighbors: int = 30,
    prop: float = 0.1,
    random_state: int = 0,
) -> "pd.DataFrame":
    """Test for differential abundance of cell populations between conditions.

    Identifies cell populations that are over- or under-represented in one
    condition relative to another, using either neighborhood-level (Milo-style)
    or cluster-level statistical tests.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str
        Column in adata.obs defining cell groups (e.g., clusters or cell types).
    condition_key : str
        Column in adata.obs defining the condition to compare (e.g., 'treatment').
        Must have at least 2 unique values.
    method : str, default 'milo'
        Testing method. One of:

        - ``'milo'``: Neighborhood-level testing. Samples index cells, defines
          neighborhoods via k-nearest neighbors, and tests differential
          abundance per neighborhood.
        - ``'simple'``: Cluster-level testing. Tests differential abundance
          for each group defined by ``groupby`` using contingency tables.
    n_neighbors : int, default 30
        Number of neighbors for neighborhood construction (only used
        when method='milo').
    prop : float, default 0.1
        Proportion of cells to sample as neighborhood index cells (only
        used when method='milo'). Stratified by ``groupby``.
    random_state : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Results table with columns depending on ``method``:

        - ``'milo'``: ['neighborhood_index', 'n_cells', 'logFC', 'pvalue',
          'padj', 'group']
        - ``'simple'``: ['group', 'n_cells', 'logFC', 'pvalue', 'padj']

        Results are also stored in ``adata.uns['differential_abundance']``
        as a dict with keys 'params' and 'results'.

    Raises
    ------
    TypeError
        If ``adata`` is not an AnnData object.
    KeyError
        If ``groupby`` or ``condition_key`` is not in ``adata.obs``.
    ValueError
        If ``method`` is not one of ('milo', 'simple').
    ValueError
        If ``condition_key`` has fewer than 2 unique values.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.pca(adata)
    >>> singlet.neighbors(adata)
    >>> result = singlet.differential_abundance(adata, groupby="leiden", condition_key="batch")
    >>> result.columns.tolist()
    ['neighborhood_index', 'n_cells', 'logFC', 'pvalue', 'padj', 'group']
    """
    # --- Input validation ---
    if not hasattr(adata, "obs"):
        raise TypeError(
            f"differential_abundance() requires an AnnData object, got {type(adata).__name__}"
        )

    if groupby not in adata.obs.columns:
        raise KeyError(f"'{groupby}' not found in adata.obs.columns")

    if condition_key not in adata.obs.columns:
        raise KeyError(f"'{condition_key}' not found in adata.obs.columns")

    if method not in ("milo", "simple"):
        raise ValueError(f"method must be one of ('milo', 'simple'), got '{method}'")

    conditions = adata.obs[condition_key].unique()
    if len(conditions) < 2:
        raise ValueError(
            f"condition_key '{condition_key}' must have at least 2 unique values, "
            f"got {len(conditions)}"
        )

    # --- Dispatch ---
    if method == "milo":
        result_df = _milo(
            adata,
            groupby=groupby,
            condition_key=condition_key,
            n_neighbors=n_neighbors,
            prop=prop,
            random_state=random_state,
        )
    else:
        result_df = _simple(
            adata,
            groupby=groupby,
            condition_key=condition_key,
        )

    # Store results in adata.uns
    adata.uns["differential_abundance"] = {
        "params": {
            "groupby": groupby,
            "condition_key": condition_key,
            "method": method,
            "n_neighbors": n_neighbors,
            "prop": prop,
            "random_state": random_state,
        },
        "results": result_df,
    }

    return result_df


def _milo(adata, *, groupby, condition_key, n_neighbors, prop, random_state):
    """Neighborhood-level differential abundance testing (Milo-style).

    Samples index cells stratified by group, constructs neighborhoods from
    k-nearest neighbors, and tests each neighborhood for differential
    abundance between conditions.
    """
    import numpy as np
    import pandas as pd
    from scipy import sparse, stats

    rng = np.random.default_rng(random_state)
    n_cells = adata.n_obs

    # Step 1: Sample index cells (stratified by groupby)
    groups = adata.obs[groupby].values
    unique_groups = np.unique(groups)
    index_cells = []

    for grp in unique_groups:
        grp_indices = np.where(groups == grp)[0]
        n_sample = max(1, int(len(grp_indices) * prop))
        sampled = rng.choice(grp_indices, size=n_sample, replace=False)
        index_cells.append(sampled)

    index_cells = np.concatenate(index_cells)

    # Step 2: Get or compute neighbor graph
    dist_matrix = _get_neighbor_graph(adata, n_neighbors)

    # Step 3: For each index cell, get its neighborhood
    condition_values = adata.obs[condition_key].values
    unique_conditions = np.unique(condition_values)

    # Overall condition proportions
    total_counts = np.array([np.sum(condition_values == cond) for cond in unique_conditions])
    overall_props = total_counts / n_cells

    results = []
    for idx_cell in index_cells:
        # Get neighbors of this cell
        if sparse.issparse(dist_matrix):
            row = dist_matrix[idx_cell]
            if hasattr(row, "toarray"):
                row = row.toarray().ravel()
            else:
                row = np.asarray(row).ravel()
        else:
            row = np.asarray(dist_matrix[idx_cell]).ravel()

        # Non-zero entries are neighbors (in sparse kNN graphs)
        neighbor_mask = row > 0
        # Always include the index cell itself
        neighbor_indices = np.where(neighbor_mask)[0]
        neighbor_indices = np.union1d(neighbor_indices, [idx_cell])

        # Limit to n_neighbors if needed
        if len(neighbor_indices) > n_neighbors + 1:
            # Sort by distance and take closest
            dists = row[neighbor_indices]
            sorted_order = np.argsort(dists)
            # Keep non-zero distances first, but cap at n_neighbors+1
            neighbor_indices = neighbor_indices[sorted_order[: n_neighbors + 1]]

        n_hood = len(neighbor_indices)
        hood_conditions = condition_values[neighbor_indices]

        # Count per condition in neighborhood
        hood_counts = np.array([np.sum(hood_conditions == cond) for cond in unique_conditions])

        # Assign majority group label
        hood_groups = groups[neighbor_indices]
        grp_values, grp_counts = np.unique(hood_groups, return_counts=True)
        majority_group = grp_values[np.argmax(grp_counts)]

        # Step 4: Statistical test
        if len(unique_conditions) == 2:
            # Fisher's exact test: 2x2 table
            # [neighborhood_cond1, neighborhood_cond2]
            # [rest_cond1, rest_cond2]
            rest_counts = total_counts - hood_counts
            table = np.array([hood_counts, rest_counts])
            _, pvalue = stats.fisher_exact(table)

            # Log fold change
            hood_prop = (hood_counts[0] + 1e-10) / (n_hood + 2e-10)
            log_fc = np.log2(hood_prop / (overall_props[0] + 1e-10))
        else:
            # Chi-squared test for >2 conditions
            rest_counts = total_counts - hood_counts
            table = np.array([hood_counts, rest_counts])
            chi2, pvalue, _, _ = stats.chi2_contingency(table)

            # Log fold change relative to first condition
            hood_prop = (hood_counts[0] + 1e-10) / (n_hood + 2e-10)
            log_fc = np.log2(hood_prop / (overall_props[0] + 1e-10))

        results.append(
            {
                "neighborhood_index": idx_cell,
                "n_cells": n_hood,
                "logFC": log_fc,
                "pvalue": pvalue,
                "group": majority_group,
            }
        )

    result_df = pd.DataFrame(results)

    # Step 5: BH correction
    result_df["padj"] = _benjamini_hochberg(result_df["pvalue"].values)

    # Reorder columns
    result_df = result_df[["neighborhood_index", "n_cells", "logFC", "pvalue", "padj", "group"]]

    return result_df


def _simple(adata, *, groupby, condition_key):
    """Cluster-level differential abundance testing.

    For each group, builds a contingency table of group membership vs
    condition and tests using Fisher's exact or chi-squared test.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    groups = adata.obs[groupby].values
    condition_values = adata.obs[condition_key].values
    unique_groups = np.unique(groups)
    unique_conditions = np.unique(condition_values)
    n_cells = adata.n_obs

    # Overall condition counts
    total_counts = np.array([np.sum(condition_values == cond) for cond in unique_conditions])
    overall_props = total_counts / n_cells

    results = []
    for grp in unique_groups:
        in_group = groups == grp
        n_in_group = np.sum(in_group)
        grp_conditions = condition_values[in_group]

        # Counts per condition in group
        grp_counts = np.array([np.sum(grp_conditions == cond) for cond in unique_conditions])
        rest_counts = total_counts - grp_counts

        if len(unique_conditions) == 2:
            # Fisher's exact: 2x2 table
            table = np.array([grp_counts, rest_counts])
            _, pvalue = stats.fisher_exact(table)
        else:
            # Chi-squared for >2 conditions
            table = np.array([grp_counts, rest_counts])
            _, pvalue, _, _ = stats.chi2_contingency(table)

        # Log fold change: proportion of condition[0] in group vs overall
        grp_prop = (grp_counts[0] + 1e-10) / (n_in_group + 2e-10)
        log_fc = np.log2(grp_prop / (overall_props[0] + 1e-10))

        results.append(
            {
                "group": grp,
                "n_cells": n_in_group,
                "logFC": log_fc,
                "pvalue": pvalue,
            }
        )

    result_df = pd.DataFrame(results)

    # BH correction
    result_df["padj"] = _benjamini_hochberg(result_df["pvalue"].values)

    # Reorder columns
    result_df = result_df[["group", "n_cells", "logFC", "pvalue", "padj"]]

    return result_df


def _get_neighbor_graph(adata, n_neighbors):
    """Retrieve or compute the neighbor connectivity/distance matrix.

    Checks adata.obsp for precomputed neighbors. Falls back to computing
    neighbors from X_pca or X.
    """
    import numpy as np
    from scipy import sparse
    from scipy.spatial import KDTree

    # Try precomputed graphs
    if "distances" in adata.obsp:
        return adata.obsp["distances"]
    if "connectivities" in adata.obsp:
        return adata.obsp["connectivities"]

    # Compute from embeddings
    if "X_pca" in adata.obsm:
        coords = np.asarray(adata.obsm["X_pca"])
    else:
        # Fall back to raw X
        if sparse.issparse(adata.X):
            coords = np.asarray(adata.X.toarray())
        else:
            coords = np.asarray(adata.X)

    # Build kNN graph using KDTree
    tree = KDTree(coords)
    distances, indices = tree.query(coords, k=n_neighbors + 1)

    # Build sparse distance matrix (exclude self which is index 0)
    n_cells = coords.shape[0]
    rows = np.repeat(np.arange(n_cells), n_neighbors)
    cols = indices[:, 1:].ravel()
    vals = distances[:, 1:].ravel()

    dist_matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(n_cells, n_cells))

    return dist_matrix


def _benjamini_hochberg(pvalues):
    """Apply Benjamini-Hochberg FDR correction to an array of p-values.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values (FDR).
    """
    import numpy as np

    pvalues = np.asarray(pvalues, dtype=np.float64)
    n_tests = len(pvalues)

    if n_tests == 0:
        return pvalues.copy()

    # Sort p-values
    sorted_idx = np.argsort(pvalues)
    sorted_pvals = pvalues[sorted_idx]

    # BH adjustment: p_adj[i] = p[i] * n / rank
    ranks = np.arange(1, n_tests + 1)
    adjusted = sorted_pvals * n_tests / ranks

    # Enforce monotonicity (cumulative minimum from the end)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    # Restore original order
    result = np.empty(n_tests, dtype=np.float64)
    result[sorted_idx] = adjusted

    return result
