"""Label transfer from a reference dataset to query cells.

Provides singlet.transfer_labels() — projects query cells into reference
PCA space, finds kNN, and assigns labels via distance-weighted majority vote.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def transfer_labels(
    adata,
    ref_adata,
    label_key: str,
    *,
    use_rep: str = "X_pca",
    n_neighbors: int = 10,
):
    """Transfer labels from a reference to query dataset via kNN.

    Projects query cells into the reference embedding space, finds k
    nearest neighbors in the reference, and transfers labels using
    distance-weighted majority voting.

    Parameters
    ----------
    adata : anndata.AnnData
        Query AnnData to annotate. Modified in-place.
    ref_adata : anndata.AnnData
        Reference AnnData with labels in .obs[label_key].
    label_key : str
        Column name in ref_adata.obs containing labels to transfer.
    use_rep : str, default 'X_pca'
        Representation to use for kNN search. Must exist in ref_adata.obsm.
        If not present in query, will project query using reference PCA loadings.
    n_neighbors : int, default 10
        Number of nearest neighbors to consider for voting.

    Returns
    -------
    anndata.AnnData
        The input adata with transferred labels in
        adata.obs[f'{label_key}_transferred'] and confidence scores in
        adata.obs[f'{label_key}_confidence'].

    Raises
    ------
    KeyError
        If label_key not in ref_adata.obs or use_rep not in ref_adata.obsm.
    ValueError
        If gene spaces don't overlap sufficiently.

    Examples
    --------
    >>> import singlet
    >>> singlet.transfer_labels(query, ref, "celltype")
    >>> query.obs["celltype_transferred"]
    """
    import numpy as np
    import pandas as pd
    from scipy.spatial import cKDTree

    # Validate inputs
    if label_key not in ref_adata.obs.columns:
        raise KeyError(
            f"'{label_key}' not found in ref_adata.obs. Available: {list(ref_adata.obs.columns)}"
        )
    if use_rep not in ref_adata.obsm:
        raise KeyError(
            f"'{use_rep}' not found in ref_adata.obsm. "
            f"Run PCA or other embedding on reference first."
        )

    # Get reference embedding
    ref_embedding = np.asarray(ref_adata.obsm[use_rep], dtype=np.float64)

    # Get or compute query embedding
    if use_rep in adata.obsm:
        query_embedding = np.asarray(adata.obsm[use_rep], dtype=np.float64)
    elif use_rep == "X_pca":
        # Project query into reference PCA space
        query_embedding = _project_to_ref_pca(adata, ref_adata)
    else:
        raise KeyError(
            f"'{use_rep}' not found in query adata.obsm and cannot auto-project. "
            f"Compute the embedding on query first, or use use_rep='X_pca'."
        )

    # Match dimensions (use minimum of both)
    n_dims = min(ref_embedding.shape[1], query_embedding.shape[1])
    ref_embedding = ref_embedding[:, :n_dims]
    query_embedding = query_embedding[:, :n_dims]

    # Build KD-tree on reference
    tree = cKDTree(ref_embedding)
    distances, indices = tree.query(query_embedding, k=n_neighbors)

    # Get reference labels
    ref_labels = ref_adata.obs[label_key].values

    # Distance-weighted majority vote
    transferred_labels = []
    confidences = []

    for idx in range(len(adata)):
        neighbor_idx = indices[idx]
        neighbor_dist = distances[idx]

        # Convert distances to weights (inverse distance, with epsilon)
        epsilon = 1e-10
        weights = 1.0 / (neighbor_dist + epsilon)

        # Weighted vote
        neighbor_labels = ref_labels[neighbor_idx]
        label_weights: dict[str, float] = {}
        for label, weight in zip(neighbor_labels, weights):
            label_str = str(label)
            label_weights[label_str] = label_weights.get(label_str, 0.0) + weight

        # Winner
        best_label = max(label_weights, key=lambda k: label_weights[k])
        total_weight = sum(label_weights.values())
        confidence = label_weights[best_label] / total_weight

        transferred_labels.append(best_label)
        confidences.append(confidence)

    # Store results
    transferred_col = f"{label_key}_transferred"
    confidence_col = f"{label_key}_confidence"
    adata.obs[transferred_col] = pd.Categorical(transferred_labels)
    adata.obs[confidence_col] = np.array(confidences, dtype=np.float32)

    return adata


def _project_to_ref_pca(adata, ref_adata):
    """Project query data into reference PCA space."""
    import numpy as np
    import scipy.sparse as sp

    # Check if reference has PCA loadings
    if "PCs" in ref_adata.varm:
        loadings = np.asarray(ref_adata.varm["PCs"], dtype=np.float64)
        ref_genes = list(ref_adata.var_names)
    elif "pca" in ref_adata.uns and "components" in ref_adata.uns.get("pca", {}):
        loadings = ref_adata.uns["pca"]["components"].T
        ref_genes = list(ref_adata.var_names)
    else:
        # Fallback: compute joint PCA
        return _joint_pca(adata, ref_adata)

    # Find shared genes
    query_genes = list(adata.var_names)
    shared = [g for g in ref_genes if g in set(query_genes)]

    if len(shared) < 10:
        raise ValueError(
            f"Only {len(shared)} genes shared between query and reference. "
            f"Need at least 10 for meaningful projection."
        )

    # Subset to shared genes
    ref_gene_idx = [ref_genes.index(g) for g in shared]
    query_gene_idx = [query_genes.index(g) for g in shared]

    # Get query data for shared genes
    query_x = adata.X[:, query_gene_idx]
    if sp.issparse(query_x):
        query_x = query_x.toarray()
    query_x = np.asarray(query_x, dtype=np.float64)

    # Center using reference mean if available
    if "pca" in ref_adata.uns and "mean" in ref_adata.uns.get("pca", {}):
        mean = ref_adata.uns["pca"]["mean"][ref_gene_idx]
        query_x = query_x - mean
    else:
        query_x = query_x - query_x.mean(axis=0)

    # Project: X @ loadings (subset to shared genes)
    shared_loadings = loadings[ref_gene_idx, :]
    projected = query_x @ shared_loadings

    return projected


def _joint_pca(adata, ref_adata, n_comps: int = 50):
    """Compute joint PCA when reference lacks loadings."""
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds

    # Find shared genes
    shared_genes = sorted(set(adata.var_names) & set(ref_adata.var_names))
    if len(shared_genes) < 10:
        raise ValueError(f"Only {len(shared_genes)} shared genes. Need at least 10.")

    # Get reference data for computing PCA
    ref_idx = [list(ref_adata.var_names).index(g) for g in shared_genes]
    ref_x = ref_adata.X[:, ref_idx]
    if sp.issparse(ref_x):
        ref_x = ref_x.toarray()
    ref_x = np.asarray(ref_x, dtype=np.float64)

    # Compute mean from reference
    mean = ref_x.mean(axis=0)
    ref_centered = ref_x - mean

    # Compute PCA on reference
    n_comps = min(n_comps, ref_centered.shape[0] - 1, ref_centered.shape[1] - 1)
    ref_sparse = sp.csr_matrix(ref_centered)
    _, _, vt = svds(ref_sparse, k=n_comps)
    vt = vt[::-1, :]  # Descending order
    loadings = vt.T  # (n_genes, n_comps)

    # Project query
    query_idx = [list(adata.var_names).index(g) for g in shared_genes]
    query_x = adata.X[:, query_idx]
    if sp.issparse(query_x):
        query_x = query_x.toarray()
    query_x = np.asarray(query_x, dtype=np.float64) - mean

    return query_x @ loadings
