# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.spatial.stagate — GPU-native spatial domain segmentation.

Underlying C++: cycle 29, ``spatial/stagate.h``.
Algorithm: GAT autoencoder + Leiden clustering (Dong & Zhang, Nat Commun 2022).

API
---
``run_stagate``   — fit + embed + cluster.  Writes embedding to ``adata.obsm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


def run_stagate(
    adata: "anndata.AnnData",
    *,
    spatial_key: str = "spatial",
    n_neighbors: int = 6,
    d_hidden: int = 256,
    d_embed: int = 64,
    n_epochs: int = 500,
    learning_rate: float = 1e-3,
    sparse_mask_recon: bool = True,
    spot_tile: int = 10_000,
    run_post_leiden: bool = True,
    leiden_resolution: float = 0.5,
    embedding_key: str = "X_stagate",
    cluster_key: str = "stagate_domain",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Spatial domain segmentation via STAGATE (cycle 29).

    Parameters
    ----------
    adata : AnnData
        Spatial transcriptomics data.  ``adata.X`` is genes × spots (or spots × genes,
        auto-detected from shape).  ``adata.obsm[spatial_key]`` must contain XY coordinates.
    spatial_key : str
        Key in ``adata.obsm`` for (n_spots, 2) XY coordinates.
    d_hidden : int
        First GAT encoder hidden dimension.
    d_embed : int
        Bottleneck embedding dimension.
    run_post_leiden : bool
        Run Leiden clustering on the embedding after training.
    embedding_key : str
        Key written to ``adata.obsm`` for the embedding.
    cluster_key : str
        Key written to ``adata.obs`` for Leiden domain labels.
    stream : int or None
    seed : int

    Returns
    -------
    None or AnnData
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "stagate"):
        raise AttributeError(
            "_core.stagate is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    working = adata.copy() if copy else adata

    if spatial_key not in working.obsm:
        raise KeyError(
            f"adata.obsm['{spatial_key}'] not found.  "
            "Provide spot coordinates via the spatial_key parameter."
        )

    try:
        import cupy as cp
        import cupy.sparse as csp

        X = working.X
        if X.shape[0] == working.n_obs:
            X = X.T   # ensure genes × spots

        import scipy.sparse as sp
        if sp.issparse(X):
            mat = csp.csc_matrix(X)
        else:
            mat = csp.csc_matrix(cp.array(X))

        coords = cp.asarray(working.obsm[spatial_key].astype(np.float32))

    except ImportError as e:
        raise ImportError(
            "singlet_gpu.spatial.run_stagate requires cupy.  "
            f"Original error: {e}"
        )

    result = _core.stagate(
        mat, coords,
        n_neighbors=n_neighbors,
        d_hidden=d_hidden,
        d_embed=d_embed,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        sparse_mask_recon=sparse_mask_recon,
        spot_tile=spot_tile,
        run_post_leiden=run_post_leiden,
        leiden_resolution=leiden_resolution,
        stream=stream,
        seed=seed,
    )

    n_spots = result.n_spots
    d_emb   = result.d_embed

    emb_arr = cp.asarray(result.embedding_view).reshape(n_spots, d_emb).get()
    working.obsm[embedding_key] = emb_arr

    if run_post_leiden and result.spatial_domain_view is not None:
        domain_arr = cp.asarray(result.spatial_domain_view).get()
        working.obs[cluster_key] = domain_arr

    working.uns["stagate_params"] = {
        "n_neighbors": n_neighbors,
        "d_hidden": d_hidden,
        "d_embed": d_embed,
        "n_epochs_used": result.n_epochs_used,
        "leiden_resolution": leiden_resolution if run_post_leiden else None,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_stagate"]
