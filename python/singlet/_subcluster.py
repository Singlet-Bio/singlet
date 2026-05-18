# SPDX-License-Identifier: MIT
"""Subclustering within existing clusters."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData


def leiden_subclustering(
    adata: AnnData,
    *,
    restrict_to: tuple[str, list[str]],
    resolution: float = 1.0,
    key_added: str | None = None,
    random_state: int = 0,
    copy: bool = False,
) -> AnnData | None:
    """Perform Leiden clustering within a subset of existing clusters.

    Re-runs neighbors + Leiden on cells within specified clusters only,
    producing hierarchical sub-cluster labels.

    Parameters
    ----------
    adata
        Annotated data matrix with PCA computed.
    restrict_to
        Tuple of (obs_key, list_of_categories). Subclustering is done
        within cells belonging to these categories.
    resolution
        Resolution for subclustering.
    key_added
        Key for storing results. Default: '{obs_key}_sub'.
    random_state
        Random seed.
    copy
        Return a copy.

    Returns
    -------
    None or AnnData if copy=True. Adds subclustering labels to .obs.
    """
    import singlet

    adata = adata.copy() if copy else adata

    obs_key, categories = restrict_to

    if obs_key not in adata.obs.columns:
        raise KeyError(f"'{obs_key}' not found in .obs.")

    if key_added is None:
        key_added = f"{obs_key}_sub"

    # Start with existing labels
    labels = adata.obs[obs_key].astype(str).copy()

    for cat in categories:
        mask = (adata.obs[obs_key] == cat).values
        if mask.sum() < 5:
            continue

        # Subset
        sub = adata[mask].copy()

        # Run PCA + neighbors + leiden on subset
        if "X_pca" in sub.obsm:
            singlet.neighbors(sub)
            singlet.leiden(sub, resolution=resolution, random_state=random_state)

            # Assign sub-labels
            sub_labels = sub.obs["leiden"].values
            cell_indices = np.where(mask)[0]
            for i, idx in enumerate(cell_indices):
                labels.iloc[idx] = f"{cat},{sub_labels[i]}"

    adata.obs[key_added] = pd.Categorical(labels)

    return adata if copy else None
