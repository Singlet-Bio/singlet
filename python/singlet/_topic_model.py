# SPDX-License-Identifier: MIT
"""Topic modeling for single-cell data.

Provides singlet.topic_model() — Latent Dirichlet Allocation or NMF
to decompose cells into gene expression "topics".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anndata import AnnData


def topic_model(
    adata: "AnnData",
    n_topics: int = 20,
    *,
    method: str = "lda",
    random_state: int = 0,
    max_iter: int = 50,
) -> "AnnData":
    """Decompose cells into gene expression topics.

    Apply Latent Dirichlet Allocation (LDA) or Non-negative Matrix
    Factorization (NMF) to discover latent topics in gene expression
    count data.

    Parameters
    ----------
    adata
        Annotated data matrix (cells × genes). For LDA, values should be
        non-negative counts. For NMF, values should be non-negative.
    n_topics
        Number of topics to extract.
    method
        Decomposition method: 'lda' for Latent Dirichlet Allocation
        or 'nmf' for Non-negative Matrix Factorization.
    random_state
        Random seed for reproducibility.
    max_iter
        Maximum number of iterations for the solver.

    Returns
    -------
    AnnData
        Input ``adata`` with:
        - ``adata.obsm['topics']``: topic proportions (cells × topics).
        - ``adata.uns['topic_gene_weights']``: gene weights per topic
          (topics × genes ndarray).
    """
    import numpy as np
    import scipy.sparse as sp

    if method not in ("lda", "nmf"):
        msg = f"method must be 'lda' or 'nmf', got {method!r}"
        raise ValueError(msg)

    # Get expression matrix as dense non-negative values
    X = adata.X
    if sp.issparse(X):
        X = np.asarray(X.toarray())
    else:
        X = np.asarray(X)

    # Ensure non-negative
    X = np.clip(X, 0, None)

    if method == "lda":
        from sklearn.decomposition import LatentDirichletAllocation

        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=max_iter,
        )
        # LDA expects non-negative integer-like values
        topic_proportions = model.fit_transform(X)
        # Normalize to sum to 1 per cell
        row_sums = topic_proportions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        topic_proportions = topic_proportions / row_sums
        # Gene weights: components_ is (n_topics, n_genes)
        gene_weights = model.components_

    else:  # nmf
        from sklearn.decomposition import NMF

        model = NMF(
            n_components=n_topics,
            random_state=random_state,
            max_iter=max_iter,
        )
        topic_proportions = model.fit_transform(X)
        # Normalize to sum to 1 per cell
        row_sums = topic_proportions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        topic_proportions = topic_proportions / row_sums
        gene_weights = model.components_

    adata.obsm["topics"] = topic_proportions.astype(np.float32)
    adata.uns["topic_gene_weights"] = gene_weights.astype(np.float32)

    return adata
