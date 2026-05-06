"""Embedding density estimation."""

from __future__ import annotations

import numpy as np
from anndata import AnnData


def embedding_density(
    adata: AnnData,
    *,
    basis: str = "umap",
    groupby: str | None = None,
    group: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Compute cell density on a 2D embedding using KDE.

    Useful for highlighting regions with high/low cell density or
    comparing density between conditions.

    Parameters
    ----------
    adata
        Annotated data matrix.
    basis
        Embedding to use. Looks for 'X_{basis}' in .obsm.
    groupby
        Key in .obs for per-group density computation.
    group
        Specific group to compute density for. If None and groupby is set,
        computes for all groups.
    key_added
        Key for storing result in .obs. Default: '{basis}_density' or
        '{basis}_density_{group}'.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Stores density values in .obs.
    """
    from scipy.stats import gaussian_kde

    adata = adata.copy() if copy else adata

    obsm_key = f"X_{basis}"
    if obsm_key not in adata.obsm:
        raise KeyError(f"'{obsm_key}' not found in .obsm.")

    coords = adata.obsm[obsm_key][:, :2]

    if groupby is not None and group is not None:
        # Compute density for specific group
        if groupby not in adata.obs.columns:
            raise KeyError(f"'{groupby}' not found in .obs.")

        mask = (adata.obs[groupby] == group).values
        if mask.sum() == 0:
            raise ValueError(f"No cells found in group '{group}'.")

        group_coords = coords[mask]

        # KDE on group cells, evaluate on all cells
        kde = gaussian_kde(group_coords.T)
        density = kde(coords.T)

        obs_key = key_added or f"{basis}_density_{group}"
        adata.obs[obs_key] = density.astype(np.float32)

    elif groupby is not None:
        # Compute density for all groups
        if groupby not in adata.obs.columns:
            raise KeyError(f"'{groupby}' not found in .obs.")

        groups = adata.obs[groupby].unique()
        for grp in groups:
            mask = (adata.obs[groupby] == grp).values
            if mask.sum() < 2:
                continue
            group_coords = coords[mask]
            kde = gaussian_kde(group_coords.T)
            density = kde(coords.T)
            obs_key = f"{basis}_density_{grp}"
            adata.obs[obs_key] = density.astype(np.float32)
    else:
        # Global density
        kde = gaussian_kde(coords.T)
        density = kde(coords.T)
        obs_key = key_added or f"{basis}_density"
        adata.obs[obs_key] = density.astype(np.float32)

    return adata if copy else None
