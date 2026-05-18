# SPDX-License-Identifier: MIT
"""Independent Component Analysis for AnnData objects.

Provides singlet.ica() — FastICA for separating independent sources
in single-cell expression data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import anndata as ad


def ica(
    adata: ad.AnnData,
    *,
    n_components: int = 50,
    random_state: int = 0,
    max_iter: int = 200,
    whiten: str = "unit-variance",
    use_highly_variable: bool = True,
) -> ad.AnnData:
    """Compute Independent Component Analysis (ICA) using FastICA.

    Separates the expression matrix into statistically independent components,
    useful for identifying independent biological signals (e.g., cell cycle,
    stress response) that may be mixed in PCA components.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated data matrix. Should be normalized and log-transformed.
    n_components : int, default 50
        Number of independent components to extract.
    random_state : int, default 0
        Random seed for reproducibility.
    max_iter : int, default 200
        Maximum number of iterations for FastICA convergence.
    whiten : str, default 'unit-variance'
        Whitening strategy. Options: 'unit-variance', 'arbitrary-variance', False.
    use_highly_variable : bool, default True
        If True and adata.var['highly_variable'] exists, only use HVGs.

    Returns
    -------
    anndata.AnnData
        Returns the same adata with:
        - ``adata.obsm['X_ica']``: ICA embedding (n_cells × n_components)
        - ``adata.varm['ica_components']``: Mixing matrix (n_genes × n_components)
        - ``adata.uns['ica']``: Dict with parameters

    Examples
    --------
    >>> import singlet
    >>> adata = singlet.load("GSE264667")
    >>> singlet.normalize(adata)
    >>> singlet.highly_variable_genes(adata)
    >>> singlet.ica(adata, n_components=30)
    >>> adata.obsm['X_ica'].shape  # (n_cells, 30)
    """
    import numpy as np
    import scipy.sparse as sp
    from sklearn.decomposition import FastICA

    if not hasattr(adata, "X") or not hasattr(adata, "var_names"):
        raise TypeError(f"ica() requires an AnnData object, got {type(adata).__name__}")

    # Select genes
    if use_highly_variable and "highly_variable" in adata.var.columns:
        X = adata[:, adata.var["highly_variable"]].X
        var_mask = adata.var["highly_variable"].values
    else:
        X = adata.X
        var_mask = None

    # Convert sparse to dense
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    n_cells, n_genes = X.shape
    n_components = min(n_components, n_cells - 1, n_genes - 1)

    if n_components < 1:
        raise ValueError(
            f"Cannot compute ICA with n_components={n_components}. "
            "Need at least 2 cells and 2 genes."
        )

    # Run FastICA
    fica = FastICA(
        n_components=n_components,
        random_state=random_state,
        max_iter=max_iter,
        whiten=whiten,
    )
    X_ica = fica.fit_transform(X)

    # Store results
    adata.obsm["X_ica"] = X_ica.astype(np.float32)

    # Mixing matrix: maps components back to gene space
    # fica.mixing_ has shape (n_genes, n_components)
    mixing = fica.mixing_.astype(np.float32)

    if var_mask is None:
        adata.varm["ica_components"] = mixing
    else:
        # Store full-size mixing matrix (zeros for non-HVG genes)
        full_mixing = np.zeros((adata.n_vars, n_components), dtype=np.float32)
        full_mixing[var_mask] = mixing
        adata.varm["ica_components"] = full_mixing

    adata.uns["ica"] = {
        "params": {
            "n_components": n_components,
            "random_state": random_state,
            "max_iter": max_iter,
            "whiten": whiten,
        }
    }

    return adata
