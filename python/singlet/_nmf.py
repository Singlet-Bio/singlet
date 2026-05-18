# SPDX-License-Identifier: MIT
"""Non-negative matrix factorization."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def nmf(
    adata: AnnData,
    *,
    n_components: int = 20,
    layer: str | None = None,
    random_state: int = 0,
    max_iter: int = 200,
    init: str = "nndsvda",
    copy: bool = False,
) -> AnnData | None:
    """Compute NMF decomposition of the expression matrix.

    Decomposes X ≈ W @ H where W is cells×components (usage) and
    H is components×genes (gene loadings/programs).

    Parameters
    ----------
    adata
        Annotated data matrix. Values should be non-negative.
    n_components
        Number of NMF components (programs).
    layer
        Layer to use. None uses .X.
    random_state
        Random seed.
    max_iter
        Maximum iterations for NMF solver.
    init
        Initialization method: 'nndsvda', 'nndsvd', 'random'.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores:
        - `.obsm['X_nmf']`: cell×component usage matrix (W)
        - `.varm['nmf_loadings']`: gene×component loadings (H.T)
        - `.uns['nmf']`: {'components': H, 'reconstruction_err': float}
    """
    from scipy.sparse import issparse
    from sklearn.decomposition import NMF as SklearnNMF

    adata = adata.copy() if copy else adata

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    # Ensure non-negative
    if issparse(X):
        if X.data.min() < 0:
            raise ValueError(
                "NMF requires non-negative input. Consider using a non-log-transformed layer."
            )
        X_input = X
    else:
        X_arr = np.asarray(X)
        if X_arr.min() < 0:
            raise ValueError(
                "NMF requires non-negative input. Consider using a non-log-transformed layer."
            )
        X_input = X_arr

    n_components_use = min(n_components, min(X_input.shape) - 1)

    model = SklearnNMF(
        n_components=n_components_use,
        init=init,
        random_state=random_state,
        max_iter=max_iter,
    )

    W = model.fit_transform(X_input)
    H = model.components_

    adata.obsm["X_nmf"] = W.astype(np.float32)
    adata.varm["nmf_loadings"] = H.T.astype(np.float32)
    adata.uns["nmf"] = {
        "components": H.astype(np.float32),
        "reconstruction_err": float(model.reconstruction_err_),
        "n_iter": int(model.n_iter_),
    }

    return adata if copy else None
