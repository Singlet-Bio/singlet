# SPDX-License-Identifier: MIT
"""Gene-space dimensionality reduction.

Provides singlet.gene_space_embedding() — embed genes (not cells) in 2D
based on their expression patterns across cells.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from anndata import AnnData


def gene_space_embedding(
    adata: "AnnData",
    *,
    n_components: int = 2,
    n_top_genes: int = 2000,
    method: str = "umap",
    random_state: int = 0,
) -> "pd.DataFrame":
    """Embed genes in low-dimensional space based on expression patterns.

    Transposes the expression matrix so genes become observations and
    cells become features, selects top highly-variable genes, reduces
    dimensionality via PCA, then embeds with UMAP, t-SNE, or PCA.

    Parameters
    ----------
    adata
        Annotated data matrix with log-normalized counts in .X.
    n_components
        Number of embedding dimensions (typically 2).
    n_top_genes
        Number of top variable genes to embed. If more genes are available,
        selects by variance. If fewer genes exist, uses all.
    method
        Embedding method: 'umap', 'tsne', or 'pca'.
    random_state
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: gene, dim_0, dim_1, ... (one row per gene).
        Also stores the embedding in adata.varm['gene_embedding'].

    Raises
    ------
    ValueError
        If method is not 'umap', 'tsne', or 'pca'.
        If adata has fewer than 3 genes.

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> gene_df = singlet.gene_space_embedding(adata, n_top_genes=500)
    >>> gene_df.head()
    """
    import numpy as np
    import pandas as pd

    if method not in ("umap", "tsne", "pca"):
        msg = f"method must be 'umap', 'tsne', or 'pca', got '{method}'"
        raise ValueError(msg)

    n_genes = adata.n_vars
    if n_genes < 3:
        msg = f"Need at least 3 genes for embedding, got {n_genes}"
        raise ValueError(msg)

    # Get expression matrix (genes × cells after transpose)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Select top variable genes by variance across cells
    gene_variances = X.var(axis=0)
    n_select = min(n_top_genes, n_genes)
    top_gene_idx = np.argsort(-gene_variances)[:n_select]

    # Transpose: now shape is (n_selected_genes, n_cells)
    X_genes = X[:, top_gene_idx].T

    # Standardize gene expression vectors (z-score across cells)
    gene_means = X_genes.mean(axis=1, keepdims=True)
    gene_stds = X_genes.std(axis=1, keepdims=True)
    gene_stds[gene_stds < 1e-12] = 1.0  # avoid division by zero
    X_genes_scaled = (X_genes - gene_means) / gene_stds

    # PCA for initial dimensionality reduction before UMAP/tSNE
    n_pca = min(50, X_genes_scaled.shape[0] - 1, X_genes_scaled.shape[1])
    if n_pca < n_components:
        n_pca = n_components

    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_genes_scaled)

    # Final embedding
    if method == "pca":
        embedding = X_pca[:, :n_components]
    elif method == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(
                n_components=n_components,
                random_state=random_state,
                n_neighbors=min(15, n_select - 1),
            )
            embedding = reducer.fit_transform(X_pca)
        except ImportError:
            # Fallback to PCA if UMAP not available
            embedding = X_pca[:, :n_components]
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, (n_select - 1) / 3.0)
        perplexity = max(perplexity, 2.0)
        tsne_model = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=perplexity,
        )
        embedding = tsne_model.fit_transform(X_pca)

    # Build result DataFrame
    gene_names = np.asarray(adata.var_names)[top_gene_idx]
    columns = {"gene": gene_names}
    for dim in range(n_components):
        columns[f"dim_{dim}"] = embedding[:, dim]
    result_df = pd.DataFrame(columns)

    # Store in adata.varm — full matrix with NaN for non-selected genes
    full_embedding = np.full((n_genes, n_components), np.nan, dtype=np.float64)
    full_embedding[top_gene_idx] = embedding
    adata.varm["gene_embedding"] = full_embedding

    return result_df
