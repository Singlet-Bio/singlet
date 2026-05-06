"""t-SNE dimensionality reduction."""

from __future__ import annotations

from anndata import AnnData


def tsne(
    adata: AnnData,
    *,
    n_pcs: int | None = None,
    perplexity: float = 30.0,
    early_exaggeration: float = 12.0,
    learning_rate: float | str = "auto",
    random_state: int = 0,
    use_rep: str | None = None,
    n_jobs: int = 1,
    copy: bool = False,
) -> AnnData | None:
    """Compute t-SNE embedding.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_pcs
        Number of principal components to use. If None, uses all available.
    perplexity
        Related to number of nearest neighbors. Larger datasets usually
        require larger perplexity (5-50).
    early_exaggeration
        Controls tightness of clusters. Default 12.0.
    learning_rate
        Learning rate for t-SNE. "auto" sets it to max(N/early_exaggeration/4, 50).
    random_state
        Random seed.
    use_rep
        Key in `.obsm` to use as input. Default: 'X_pca'.
    n_jobs
        Number of threads for Barnes-Hut approximation.
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    None or AnnData if copy=True. Stores embedding in `.obsm['X_tsne']`.
    """
    from sklearn.manifold import TSNE

    adata = adata.copy() if copy else adata

    # Determine representation
    if use_rep is None:
        use_rep = "X_pca"

    if use_rep not in adata.obsm:
        raise KeyError(f"'{use_rep}' not found in .obsm. Run singlet.pca() first.")

    X = adata.obsm[use_rep]
    if n_pcs is not None:
        X = X[:, :n_pcs]

    # Compute learning rate
    if learning_rate == "auto":
        lr = max(X.shape[0] / early_exaggeration / 4, 50)
    else:
        lr = float(learning_rate)

    tsne_obj = TSNE(
        n_components=2,
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=lr,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    adata.obsm["X_tsne"] = tsne_obj.fit_transform(X)

    return adata if copy else None
