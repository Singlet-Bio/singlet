"""Gene activity/pathway scoring for AnnData objects.

Provides singlet.gene_activity_score() — compute pathway activity scores
per cell from gene set expression using various scoring methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def gene_activity_score(
    adata: "AnnData",
    gene_sets: dict[str, list[str]],
    *,
    method: str = "mean",
    scale: bool = True,
    layer: str | None = None,
) -> "pd.DataFrame":
    """Compute pathway activity scores per cell from gene set expression.

    For each gene set (pathway), computes an activity score per cell using
    the specified aggregation method. Scores are stored in adata.obs and
    returned as a DataFrame.

    Parameters
    ----------
    adata
        Annotated data matrix. Should be log-normalized for best results.
    gene_sets
        Dictionary mapping pathway names to lists of gene names.
        Genes not found in adata.var_names are silently skipped.
    method
        Scoring method:
        - 'mean': Average expression of genes in the set.
        - 'weighted': Variance-weighted average (genes with higher variance
          contribute more).
        - 'z-score': Z-score each gene across cells, then average.
    scale
        If True, standardize each pathway score to zero mean and unit
        variance across cells.
    layer
        Layer to use for expression values. If None, uses adata.X.

    Returns
    -------
    pd.DataFrame
        DataFrame with shape (n_cells, n_pathways) containing activity scores.
        Columns are named 'activity_{pathway_name}'.

    Raises
    ------
    ValueError
        If method is not one of 'mean', 'weighted', 'z-score'.
        If gene_sets is empty.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> gene_sets = {
    ...     "cell_cycle": ["MCM5", "PCNA", "TYMS"],
    ...     "apoptosis": ["BAX", "BCL2", "CASP3"],
    ... }
    >>> scores = singlet.gene_activity_score(adata, gene_sets)
    >>> scores.head()
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    valid_methods = ("mean", "weighted", "z-score")
    if method not in valid_methods:
        msg = f"method must be one of {valid_methods}, got '{method}'"
        raise ValueError(msg)

    if not gene_sets:
        msg = "gene_sets must not be empty"
        raise ValueError(msg)

    # Get expression matrix
    mat = adata.layers[layer] if layer is not None else adata.X

    var_names = list(adata.var_names)
    var_idx = {name: idx for idx, name in enumerate(var_names)}

    results = {}

    for pathway_name, genes in gene_sets.items():
        # Find valid gene indices
        indices = [var_idx[g] for g in genes if g in var_idx]

        col_name = f"activity_{pathway_name}"

        if not indices:
            # No genes found — score is zero
            scores = np.zeros(adata.n_obs, dtype=np.float64)
        else:
            # Extract submatrix for this gene set
            sub = mat[:, indices]
            if sp.issparse(sub):
                sub = sub.toarray()
            sub = np.asarray(sub, dtype=np.float64)

            if method == "mean":
                scores = np.mean(sub, axis=1)

            elif method == "weighted":
                # Variance-weighted average
                variances = np.var(sub, axis=0)
                total_var = variances.sum()
                if total_var == 0:
                    scores = np.mean(sub, axis=1)
                else:
                    weights = variances / total_var
                    scores = sub @ weights

            elif method == "z-score":
                # Z-score each gene, then average
                means = np.mean(sub, axis=0)
                stds = np.std(sub, axis=0)
                # Avoid division by zero
                stds[stds == 0] = 1.0
                z_scored = (sub - means) / stds
                scores = np.mean(z_scored, axis=1)

        if scale:
            std = np.std(scores)
            if std > 0:
                scores = (scores - np.mean(scores)) / std

        results[col_name] = scores
        adata.obs[col_name] = scores

    return pd.DataFrame(results, index=adata.obs_names)
