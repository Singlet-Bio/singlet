# SPDX-License-Identifier: MIT
"""Canonical Correlation Analysis for AnnData objects.

Implements CCA for integrating single-cell datasets by finding shared
correlation structure between batches or paired datasets. Produces a
joint low-dimensional embedding stored in adata.obsm['X_cca'].
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata


def cca(
    adata,
    adata2=None,
    *,
    batch_key: str | None = None,
    n_components: int = 30,
    n_features: int = 2000,
    random_state: int = 0,
) -> "anndata.AnnData":
    """Integrate datasets using Canonical Correlation Analysis.

    Finds a shared low-dimensional embedding that maximizes the correlation
    between two or more datasets. Can operate on two AnnData objects directly
    or split a single AnnData by a batch key.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix.
    adata2 : anndata.AnnData or None, default None
        Optional second dataset for pairwise CCA. If provided, CCA is
        computed between adata and adata2, and a merged AnnData is returned.
    batch_key : str or None, default None
        Column in adata.obs identifying batch membership. Used to split
        adata into batches for CCA integration. Ignored if adata2 is given.
    n_components : int, default 30
        Number of canonical correlation components to compute.
    n_features : int, default 2000
        Number of top variable genes (by variance) to select from the
        shared gene space between datasets.
    random_state : int, default 0
        Random seed for reproducibility.

    Returns
    -------
    anndata.AnnData
        If adata2 is provided, returns a merged AnnData with the CCA
        embedding in obsm['X_cca']. If batch_key is used, returns the
        original adata with obsm['X_cca'] added.

    Raises
    ------
    TypeError
        If inputs are not AnnData objects.
    ValueError
        If neither adata2 nor batch_key is provided, if batch_key is
        not in adata.obs, or if there are fewer shared genes than
        n_components.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.cca(adata, batch_key="batch")
    >>> 'X_cca' in adata.obsm
    True
    """
    if not hasattr(adata, "obsm"):
        raise TypeError(f"cca() requires an AnnData object, got {type(adata).__name__}")

    if adata2 is not None:
        if not hasattr(adata2, "obsm"):
            raise TypeError(
                f"cca() requires an AnnData object for adata2, got {type(adata2).__name__}"
            )
        return _cca_two_datasets(adata, adata2, n_components, n_features, random_state)

    if batch_key is not None:
        if batch_key not in adata.obs.columns:
            raise ValueError(f"'{batch_key}' not found in adata.obs.columns")
        return _cca_batch_key(adata, batch_key, n_components, n_features, random_state)

    raise ValueError("Either 'adata2' or 'batch_key' must be provided for CCA integration.")


def _cca_two_datasets(adata, adata2, n_components, n_features, random_state):
    """Run CCA between two AnnData objects and return merged result."""
    import anndata as ad
    import numpy as np

    shared_genes = _get_shared_genes(adata, adata2)

    if len(shared_genes) < n_components:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes found, but n_components={n_components} "
            f"requires at least that many shared genes."
        )

    # Subset to shared genes
    adata_sub = adata[:, shared_genes]
    adata2_sub = adata2[:, shared_genes]

    # Select top variable features from shared genes
    selected_genes = _select_variable_genes(adata_sub, adata2_sub, n_features)

    if len(selected_genes) < n_components:
        raise ValueError(
            f"Only {len(selected_genes)} variable shared genes found, but "
            f"n_components={n_components} requires at least that many."
        )

    # Extract dense, centered, and scaled matrices
    mat1 = _get_scaled_matrix(adata_sub[:, selected_genes])
    mat2 = _get_scaled_matrix(adata2_sub[:, selected_genes])

    # Compute CCA embeddings
    cca1, cca2 = _compute_cca(mat1, mat2, n_components)

    # Merge datasets
    embedding = np.vstack([cca1, cca2])
    merged = ad.concat([adata, adata2], join="outer")
    merged.obsm["X_cca"] = embedding
    merged.uns["cca"] = {
        "params": {
            "n_components": n_components,
            "n_features": n_features,
            "random_state": random_state,
            "n_shared_genes": len(shared_genes),
            "n_selected_genes": len(selected_genes),
        }
    }

    return merged


def _cca_batch_key(adata, batch_key, n_components, n_features, random_state):
    """Run CCA using batch_key to split datasets, store result in adata."""
    import numpy as np

    batch_labels = adata.obs[batch_key].values
    unique_batches = np.unique(batch_labels)

    if len(unique_batches) < 2:
        raise ValueError(
            f"batch_key '{batch_key}' has fewer than 2 unique values; "
            f"CCA requires at least 2 batches."
        )

    # Use first batch as reference
    ref_mask = batch_labels == unique_batches[0]
    ref_adata = adata[ref_mask]

    n_cells = adata.shape[0]
    embeddings = np.zeros((n_cells, n_components), dtype=np.float64)

    # Store reference embedding (will be averaged if >2 batches)
    ref_embeddings = []
    query_results = []

    for batch_idx in range(1, len(unique_batches)):
        query_mask = batch_labels == unique_batches[batch_idx]
        query_adata = adata[query_mask]

        shared_genes = _get_shared_genes(ref_adata, query_adata)

        if len(shared_genes) < n_components:
            raise ValueError(
                f"Only {len(shared_genes)} shared genes between batch "
                f"'{unique_batches[0]}' and '{unique_batches[batch_idx]}', "
                f"but n_components={n_components} requires at least that many."
            )

        ref_sub = ref_adata[:, shared_genes]
        query_sub = query_adata[:, shared_genes]

        selected_genes = _select_variable_genes(ref_sub, query_sub, n_features)

        if len(selected_genes) < n_components:
            raise ValueError(
                f"Only {len(selected_genes)} variable shared genes between batch "
                f"'{unique_batches[0]}' and '{unique_batches[batch_idx]}', "
                f"but n_components={n_components} requires at least that many."
            )

        mat_ref = _get_scaled_matrix(ref_sub[:, selected_genes])
        mat_query = _get_scaled_matrix(query_sub[:, selected_genes])

        cca_ref, cca_query = _compute_cca(mat_ref, mat_query, n_components)

        ref_embeddings.append(cca_ref)
        query_results.append((query_mask, cca_query))

    # Average reference embeddings across pairwise comparisons
    ref_embedding_avg = np.mean(ref_embeddings, axis=0)
    embeddings[ref_mask] = ref_embedding_avg

    # Assign query embeddings
    for query_mask, cca_query in query_results:
        embeddings[query_mask] = cca_query

    adata.obsm["X_cca"] = embeddings
    adata.uns["cca"] = {
        "params": {
            "batch_key": batch_key,
            "n_components": n_components,
            "n_features": n_features,
            "random_state": random_state,
            "n_batches": len(unique_batches),
        }
    }

    return adata


def _get_shared_genes(adata1, adata2):
    """Return sorted intersection of var_names between two datasets."""
    import numpy as np

    genes1 = set(adata1.var_names)
    genes2 = set(adata2.var_names)
    shared = sorted(genes1 & genes2)
    return np.array(shared)


def _select_variable_genes(adata1, adata2, n_features):
    """Select top variable genes by combined variance across two datasets."""
    import numpy as np

    var1 = _compute_variance(adata1.X)
    var2 = _compute_variance(adata2.X)

    combined_var = var1 + var2
    n_select = min(n_features, len(combined_var))
    top_indices = np.argsort(combined_var)[::-1][:n_select]
    top_indices = np.sort(top_indices)

    return adata1.var_names[top_indices].values


def _compute_variance(mat):
    """Compute per-column variance, handling sparse matrices."""
    import numpy as np
    import scipy.sparse as sp

    if sp.issparse(mat):
        mean = np.asarray(mat.mean(axis=0)).ravel()
        mean_sq = np.asarray(mat.multiply(mat).mean(axis=0)).ravel()
        variance = mean_sq - mean**2
    else:
        mat = np.asarray(mat)
        variance = np.var(mat, axis=0)

    return np.asarray(variance).ravel()


def _get_scaled_matrix(adata):
    """Extract dense matrix, center, and scale columns (genes)."""
    import numpy as np
    import scipy.sparse as sp

    mat = adata.X
    if sp.issparse(mat):
        mat = np.asarray(mat.todense())
    else:
        mat = np.asarray(mat, dtype=np.float64).copy()

    # Center each gene
    means = mat.mean(axis=0)
    mat -= means

    # Scale each gene
    stds = mat.std(axis=0)
    stds[stds < 1e-10] = 1.0
    mat /= stds

    return mat


def _compute_cca(mat1, mat2, n_components):
    """Compute CCA embeddings for two centered/scaled matrices.

    Parameters
    ----------
    mat1 : ndarray (n1 × genes)
        Centered and scaled expression matrix for dataset 1.
    mat2 : ndarray (n2 × genes)
        Centered and scaled expression matrix for dataset 2.
    n_components : int
        Number of canonical components.

    Returns
    -------
    cca1 : ndarray (n1 × n_components)
        CCA embedding for dataset 1.
    cca2 : ndarray (n2 × n_components)
        CCA embedding for dataset 2.
    """
    import numpy as np

    n1 = mat1.shape[0]
    n2 = mat2.shape[0]

    # Cross-covariance matrix
    cross_cov = mat1.T @ mat2 / (n1 * n2)

    # SVD of cross-covariance
    u_mat, _, vt_mat = np.linalg.svd(cross_cov, full_matrices=False)

    # Truncate to n_components
    u_mat = u_mat[:, :n_components]
    v_mat = vt_mat[:n_components, :].T

    # Project each dataset
    cca1 = mat1 @ u_mat
    cca2 = mat2 @ v_mat

    return cca1, cca2
