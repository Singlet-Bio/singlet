# SPDX-License-Identifier: MIT
"""Diffusion pseudotime."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def dpt(
    adata: AnnData,
    *,
    n_dcs: int = 10,
    root_cell: int | None = None,
    n_branchings: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Compute diffusion pseudotime (DPT).

    Requires diffusion map to be computed first (singlet.diffmap()).

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_dcs
        Number of diffusion components to use for distance computation.
    root_cell
        Index of the root cell. If None, uses the cell with the smallest
        first diffusion component value (most extreme position).
    n_branchings
        Not used in this implementation (kept for API compatibility).
    copy
        Return a copy instead of modifying in place.

    Returns
    -------
    None or AnnData if copy=True. Stores pseudotime in `.obs['dpt_pseudotime']`.
    """
    adata = adata.copy() if copy else adata

    if "X_diffmap" not in adata.obsm:
        raise KeyError("'X_diffmap' not found in .obsm. Run singlet.diffmap() first.")

    X_diffmap = adata.obsm["X_diffmap"]

    # Use up to n_dcs components
    n_available = X_diffmap.shape[1]
    n_use = min(n_dcs, n_available)
    X_dc = X_diffmap[:, :n_use]

    # Weight by eigenvalues if available
    if "diffmap_evals" in adata.uns:
        evals = adata.uns["diffmap_evals"][:n_use]
        weights = evals / (1 - evals + 1e-10)
        X_weighted = X_dc * weights[None, :]
    else:
        X_weighted = X_dc

    # Determine root cell
    if root_cell is None:
        root_cell = int(np.argmin(X_dc[:, 0]))

    # Compute distances from root cell in weighted diffusion space
    root_vec = X_weighted[root_cell]
    diffs = X_weighted - root_vec[None, :]
    distances = np.sqrt(np.sum(diffs**2, axis=1))

    # Normalize to [0, 1]
    max_dist = distances.max()
    if max_dist > 0:
        pseudotime = distances / max_dist
    else:
        pseudotime = distances

    adata.obs["dpt_pseudotime"] = pseudotime.astype(np.float32)
    adata.uns["iroot"] = root_cell

    return adata if copy else None
