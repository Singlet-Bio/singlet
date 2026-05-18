# SPDX-License-Identifier: MIT
"""Diffusion maps for trajectory inference."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def diffmap(
    adata: AnnData,
    *,
    n_comps: int = 15,
    copy: bool = False,
) -> AnnData | None:
    """Compute diffusion map embedding.

    Requires neighbors to be computed first (singlet.neighbors()).
    Uses the transition matrix to compute diffusion components.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_comps
        Number of diffusion components to compute.
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    None or AnnData if copy=True. Stores embedding in `.obsm['X_diffmap']`
    and eigenvalues in `.uns['diffmap_evals']`.
    """
    from scipy.sparse import issparse
    from scipy.sparse.linalg import eigsh

    adata = adata.copy() if copy else adata

    if "connectivities" not in adata.obsp:
        raise KeyError("'connectivities' not found in .obsp. Run singlet.neighbors() first.")

    # Get connectivities and build transition matrix
    W = adata.obsp["connectivities"].copy()
    if issparse(W):
        W = W.tocsr()

    # Symmetrize
    W = (W + W.T) / 2

    # Compute degree matrix
    if issparse(W):
        degrees = np.asarray(W.sum(axis=1)).flatten()
    else:
        degrees = np.sum(W, axis=1)

    # Avoid division by zero
    degrees[degrees == 0] = 1.0

    # Normalized graph Laplacian approach: D^{-1/2} W D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(degrees)

    if issparse(W):
        from scipy.sparse import diags

        D_inv_sqrt = diags(d_inv_sqrt)
        T_norm = D_inv_sqrt @ W @ D_inv_sqrt
    else:
        T_norm = d_inv_sqrt[:, None] * W * d_inv_sqrt[None, :]

    # Number of components to compute (plus one for the trivial eigenvector)
    n_comps_compute = min(n_comps + 1, W.shape[0] - 1)

    # Compute eigendecomposition
    evals, evecs = eigsh(T_norm, k=n_comps_compute, which="LM")

    # Sort by decreasing eigenvalue
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Transform back: multiply by D^{-1/2}
    evecs = d_inv_sqrt[:, None] * evecs

    # Skip the first trivial component (constant eigenvector)
    adata.obsm["X_diffmap"] = evecs[:, 1:n_comps_compute].astype(np.float32)
    adata.uns["diffmap_evals"] = evals[1:n_comps_compute].astype(np.float32)

    return adata if copy else None
