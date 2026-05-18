# SPDX-License-Identifier: MIT
"""Label transfer via ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def ingest(
    adata: AnnData,
    adata_ref: AnnData,
    *,
    obs_to_transfer: list[str] | str = "leiden",
    embedding_method: str = "umap",
    n_pcs: int | None = None,
    n_neighbors: int = 10,
    copy: bool = False,
) -> AnnData | None:
    """Transfer labels and embeddings from a reference to query dataset.

    Projects query cells into reference PCA space, finds nearest neighbors
    in the reference, and transfers labels by majority vote.

    Parameters
    ----------
    adata
        Query AnnData to annotate.
    adata_ref
        Reference AnnData with labels and embeddings.
    obs_to_transfer
        Column(s) in adata_ref.obs to transfer to adata.
    embedding_method
        Embedding to transfer. Options: 'umap', 'tsne'. Will look for
        'X_{method}' in adata_ref.obsm.
    n_pcs
        Number of PCs to use. If None, uses all available.
    n_neighbors
        Number of nearest neighbors for label transfer.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Transfers labels to .obs and embedding
    to .obsm['X_{embedding_method}'].
    """
    from scipy.sparse import issparse
    from scipy.spatial import cKDTree

    adata = adata.copy() if copy else adata

    if isinstance(obs_to_transfer, str):
        obs_to_transfer = [obs_to_transfer]

    # Validate reference has PCA
    if "X_pca" not in adata_ref.obsm:
        raise KeyError("Reference must have 'X_pca' in .obsm. Run singlet.pca() on reference.")

    # Get PCA loadings from reference
    ref_pca = adata_ref.obsm["X_pca"]
    if n_pcs is not None:
        ref_pca = ref_pca[:, :n_pcs]
    n_pcs_use = ref_pca.shape[1]

    # Project query into reference PCA space
    if "PCs" in adata_ref.varm:
        loadings = adata_ref.varm["PCs"][:, :n_pcs_use]
    elif "pca" in adata_ref.uns and "components" in adata_ref.uns["pca"]:
        loadings = adata_ref.uns["pca"]["components"][:n_pcs_use].T
    else:
        raise KeyError(
            "Cannot find PCA loadings in reference. "
            "Need 'PCs' in .varm or 'components' in .uns['pca']."
        )

    # Find shared genes
    shared_genes = list(set(adata.var_names) & set(adata_ref.var_names))
    if len(shared_genes) < 10:
        raise ValueError(
            f"Only {len(shared_genes)} shared genes between query and reference. Need at least 10."
        )

    # Get query expression for shared genes
    ref_gene_idx = [list(adata_ref.var_names).index(g) for g in shared_genes]
    query_gene_idx = [list(adata.var_names).index(g) for g in shared_genes]

    if issparse(adata.X):
        X_query = np.asarray(adata.X[:, query_gene_idx].todense())
    else:
        X_query = np.asarray(adata.X[:, query_gene_idx])

    # Get reference mean for centering
    if issparse(adata_ref.X):
        X_ref_shared = np.asarray(adata_ref.X[:, ref_gene_idx].todense())
    else:
        X_ref_shared = np.asarray(adata_ref.X[:, ref_gene_idx])

    ref_mean = X_ref_shared.mean(axis=0)

    # Center query data using reference mean
    X_query_centered = X_query - ref_mean[None, :]

    # Project onto reference PCA loadings (only shared genes)
    loadings_shared = loadings[ref_gene_idx, :]
    query_pca = X_query_centered @ loadings_shared

    # Build KD-tree on reference PCA
    tree = cKDTree(ref_pca)

    # Find nearest neighbors
    distances, indices = tree.query(query_pca, k=n_neighbors)

    # Transfer labels by majority vote
    for obs_key in obs_to_transfer:
        if obs_key not in adata_ref.obs.columns:
            continue

        ref_labels = adata_ref.obs[obs_key].values
        transferred = []

        for nn_idx in indices:
            nn_labels = ref_labels[nn_idx]
            unique, counts = np.unique(nn_labels, return_counts=True)
            winner = unique[np.argmax(counts)]
            transferred.append(winner)

        adata.obs[obs_key] = pd.Categorical(transferred)

    # Transfer embedding by weighted average
    emb_key = f"X_{embedding_method}"
    if emb_key in adata_ref.obsm:
        ref_emb = adata_ref.obsm[emb_key]

        # Weighted by inverse distance
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        query_emb = np.zeros((adata.n_obs, ref_emb.shape[1]), dtype=np.float32)
        for i in range(adata.n_obs):
            query_emb[i] = np.average(ref_emb[indices[i]], axis=0, weights=weights[i])

        adata.obsm[emb_key] = query_emb

    return adata if copy else None
