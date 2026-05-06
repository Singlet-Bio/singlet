"""Cell neighborhood entropy scoring.

Provides singlet.entropy_score() — compute Shannon entropy of cluster labels
in each cell's kNN neighborhood. High entropy indicates cells in mixed-identity
regions (transition zones, integration boundaries).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def entropy_score(
    adata: AnnData,
    *,
    groupby: str | None = None,
    use_rep: str = "X_pca",
    n_neighbors: int = 30,
) -> AnnData:
    """Compute neighborhood entropy for each cell.

    For each cell, finds its k nearest neighbors and computes the Shannon
    entropy of the group label distribution in that neighborhood.

    - High entropy → cell is in a mixed-identity region (transition zone)
    - Low entropy → cell is surrounded by same-type neighbors (cluster core)

    If groupby is None, computes per-cell expression entropy (information
    content of the expression vector).

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    groupby : str or None, default None
        Key in adata.obs for group labels. If None, computes expression
        entropy per cell instead of neighborhood label entropy.
    use_rep : str, default 'X_pca'
        Representation in adata.obsm to use for kNN computation.
        Only used when groupby is not None.
    n_neighbors : int, default 30
        Number of nearest neighbors to consider.

    Returns
    -------
    anndata.AnnData
        The input adata with adata.obs['neighborhood_entropy'] added
        (or adata.obs['expression_entropy'] if groupby is None).

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.pca(adata)
    >>> singlet.entropy_score(adata, groupby='leiden')
    >>> adata.obs['neighborhood_entropy']  # per-cell entropy values
    """
    import numpy as np
    import scipy.sparse as sp

    if groupby is None:
        # Expression entropy: Shannon entropy of normalized expression per cell
        mat = adata.X
        if sp.issparse(mat):
            mat = np.asarray(mat.todense())
        else:
            mat = np.asarray(mat, dtype=np.float64)

        # Ensure non-negative (entropy needs probabilities)
        mat = np.maximum(mat, 0)

        # Normalize each row to sum to 1 (probability distribution)
        row_sums = mat.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs = mat / row_sums

        # Shannon entropy: -sum(p * log(p))
        with np.errstate(divide="ignore", invalid="ignore"):
            log_probs = np.log2(probs)
            log_probs = np.where(np.isfinite(log_probs), log_probs, 0)

        entropy_vals = -np.sum(probs * log_probs, axis=1)
        adata.obs["expression_entropy"] = np.asarray(entropy_vals).ravel()
    else:
        # Neighborhood entropy: entropy of group labels in kNN
        if groupby not in adata.obs.columns:
            msg = f"groupby key {groupby!r} not found in adata.obs"
            raise KeyError(msg)

        if use_rep not in adata.obsm:
            msg = (
                f"Representation {use_rep!r} not found in adata.obsm. Run singlet.pca(adata) first."
            )
            raise KeyError(msg)

        from sklearn.neighbors import NearestNeighbors

        coords = np.asarray(adata.obsm[use_rep])
        n_cells = coords.shape[0]

        # Cap n_neighbors to n_cells - 1
        k_actual = min(n_neighbors, n_cells - 1)
        if k_actual < 1:
            adata.obs["neighborhood_entropy"] = 0.0
            return adata

        nn = NearestNeighbors(n_neighbors=k_actual, algorithm="auto")
        nn.fit(coords)
        indices = nn.kneighbors(coords, return_distance=False)

        # Get labels as integer codes for fast counting
        labels = adata.obs[groupby]
        if hasattr(labels, "cat"):
            codes = labels.cat.codes.values
            n_categories = len(labels.cat.categories)
        else:
            from pandas import Categorical

            cat = Categorical(labels)
            codes = cat.codes
            n_categories = len(cat.categories)

        # Compute entropy for each cell's neighborhood
        entropy_vals = np.zeros(n_cells, dtype=np.float64)
        for idx in range(n_cells):
            neighbor_labels = codes[indices[idx]]
            # Count occurrences of each category
            counts = np.bincount(neighbor_labels[neighbor_labels >= 0], minlength=n_categories)
            total = counts.sum()
            if total == 0:
                continue
            probs = counts / total
            probs = probs[probs > 0]
            entropy_vals[idx] = -np.sum(probs * np.log2(probs))

        adata.obs["neighborhood_entropy"] = entropy_vals

    return adata
