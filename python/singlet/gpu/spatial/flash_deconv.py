# SPDX-License-Identifier: MIT
"""
singlet.gpu.spatial.flash_deconv — GPU-native spatial deconvolution.

Underlying C++: cycle 33, ``spatial/flash_deconv.h``.
Algorithm: leverage-score sketching + ADMM (FlashDeconv, bioRxiv 2026).

Scanpy / AnnData compatibility
-------------------------------
``run_flash_deconv`` accepts two AnnData objects (spatial and reference) or
two DeviceCsc objects directly and returns a ``FlashDeconvResult``.
When AnnData is provided, results are written into ``adata_spatial.obsm``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from singlet.gpu._coreutil import require_core

if TYPE_CHECKING:
    import anndata


def run_flash_deconv(
    adata_spatial: "anndata.AnnData",
    adata_reference: "anndata.AnnData",
    cell_type_key: str = "cell_type",
    *,
    sketch_size: int = 500,
    max_admm_iter: int = 50,
    admm_tol: float = 1e-4,
    l1_weight: float = 0.01,
    spatial_lambda: float = 0.0,
    n_bootstrap: int = 20,
    spot_batch: int = 100_000,
    obsm_key: str = "deconv_abundance",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Spatial deconvolution of a Visium / Xenium spatial dataset.

    Parameters
    ----------
    adata_spatial : AnnData
        Spatial transcriptomics data.  ``adata_spatial.X`` is the
        genes × spots count matrix (CSC or CSR).
    adata_reference : AnnData
        scRNA-seq reference.  ``adata_reference.X`` is genes × cells.
    cell_type_key : str
        Key in ``adata_reference.obs`` with integer or string cell-type labels.
        Integers are used as-is; strings are label-encoded.
    sketch_size : int
        Gene sketch size K for leverage-score sketching.
    max_admm_iter : int
        Maximum ADMM iterations per OOC batch.
    admm_tol : float
        Relative primal residual convergence threshold.
    l1_weight : float
        L1 regularization weight.
    spatial_lambda : float
        Spatial regularization weight (0 = disabled).
    n_bootstrap : int
        Bootstrap draws for uncertainty estimation.
    spot_batch : int
        Spots per ADMM OOC batch.
    obsm_key : str
        Key written to ``adata_spatial.obsm``.  Shape: n_spots × n_types.
    stream : int or None
        CUDA stream handle.
    seed : int
        Philox4x32 base seed.
    copy : bool
        Return a modified copy; otherwise modify in-place and return None.

    Returns
    -------
    None or AnnData
    """
    _core = require_core("flash_deconv")

    working = adata_spatial.copy() if copy else adata_spatial

    # ---- Build device matrices -----------------------------------------------
    try:
        import cupy as cp
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback

        def _to_device_csc(X):
            if hasattr(X, "get"):
                return X.tocsc()
            import scipy.sparse as sp
            if sp.issparse(X):
                return csp.csc_matrix(X)
            return csp.csc_matrix(cp.array(X))

        sc = _to_device_csc(working.X.T)        # genes × spots
        rc = _to_device_csc(adata_reference.X.T) # genes × cells
    except ImportError as e:
        raise ImportError(
            "singlet.gpu.spatial.run_flash_deconv requires cupy.  "
            f"Original error: {e}"
        )

    # ---- Cell-type labels → int32 device array --------------------------------
    labels = adata_reference.obs[cell_type_key].values
    if labels.dtype.kind not in ("i", "u"):
        # String labels → integer encode.
        cats = list(dict.fromkeys(labels))
        label_map = {c: i for i, c in enumerate(cats)}
        labels = np.array([label_map[l] for l in labels], dtype=np.int32)
    else:
        labels = np.asarray(labels, dtype=np.int32)
    n_types = int(labels.max()) + 1
    d_labels = cp.asarray(labels)

    # ---- Spatial coords (optional) -------------------------------------------
    d_coords = None
    if spatial_lambda > 0.0 and hasattr(working, "obsm") and "spatial" in working.obsm:
        d_coords = cp.asarray(working.obsm["spatial"].astype(np.float32))

    result = _core.flash_deconv(
        sc, rc, d_labels, n_types, d_coords,
        sketch_size=sketch_size,
        max_admm_iter=max_admm_iter,
        admm_tol=admm_tol,
        l1_weight=l1_weight,
        spatial_lambda=spatial_lambda,
        n_bootstrap=n_bootstrap,
        spot_batch=spot_batch,
        stream=stream,
        seed=seed,
    )

    # Write results to obsm.
    abundance_view = result.abundance_view
    uncert_view    = result.uncertainty_view
    working.obsm[obsm_key]              = cp.asarray(abundance_view).get()
    working.obsm[obsm_key + "_stddev"]  = cp.asarray(uncert_view).get()
    working.uns["flash_deconv_params"] = {
        "sketch_size": sketch_size,
        "n_bootstrap": n_bootstrap,
        "seed": seed,
        "n_types": n_types,
        "n_admm_iters": result.n_admm_iters_used,
    }

    return working if copy else None


__all__ = ["run_flash_deconv"]
