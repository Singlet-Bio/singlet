# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.integrate.bbknn — GPU-native Batch-Balanced KNN graph.

Underlying C++ cycle: cycle 14 (integrate/bbknn.h —
``singlet_gpu::integrate::bbknn`` kernel).

Drop-in for ``sc.external.pp.bbknn``.  All parameter names and defaults
match the BBKNN / scanpy 1.10 external API.

BBKNN constructs a kNN graph by, for each cell, selecting *k* nearest
neighbours from *each batch* independently, then merging those per-batch
neighbour lists.  This preserves batch-specific topology while equalising
cross-batch representation.

Result location (matches bbknn / scanpy external)
--------------------------------------------------
``adata.obsp['distances']``         — (n_cells × n_cells) sparse distances.
``adata.obsp['connectivities']``    — (n_cells × n_cells) sparse UMAP weights.
``adata.uns['neighbors']``          — metadata dict (bbknn-style).

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.bbknn`` must be
exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embedding(
    adata: "anndata.AnnData",
    use_rep: str,
    n_pcs: Optional[int],
) -> "np.ndarray":
    """
    Extract the cell embedding to use for kNN.

    Parameters
    ----------
    adata : AnnData
    use_rep : str
        Key in ``adata.obsm``.
    n_pcs : int or None
        If set, truncate the embedding to the first *n_pcs* dimensions.

    Returns
    -------
    np.ndarray of shape (n_cells, n_dims), float32.
    """
    if use_rep not in adata.obsm:
        raise KeyError(
            f"Embedding key '{use_rep}' not found in adata.obsm.  "
            f"Available keys: {list(adata.obsm.keys())}"
        )
    emb = adata.obsm[use_rep]
    try:
        import cupy as cp
        if isinstance(emb, cp.ndarray):
            emb = emb.get()
    except ImportError:
        pass
    emb = np.asarray(emb, dtype=np.float32)
    if n_pcs is not None:
        emb = emb[:, :n_pcs]
    return emb


def _resolve_batch_codes(
    adata: "anndata.AnnData",
    batch_key: str,
) -> "np.ndarray":
    """Encode batch column as int32 codes."""
    if batch_key not in adata.obs.columns:
        raise KeyError(
            f"batch_key='{batch_key}' not found in adata.obs.  "
            f"Available columns: {list(adata.obs.columns)}"
        )
    series = adata.obs[batch_key].astype("category")
    return series.cat.codes.to_numpy(dtype=np.int32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bbknn(
    adata: "anndata.AnnData",
    *,
    batch_key: str = "batch",
    use_rep: str = "X_pca",
    neighbors_within_batch: int = 3,
    n_pcs: Optional[int] = None,
    metric: str = "euclidean",
    set_op_mix_ratio: float = 1.0,
    local_connectivity: int = 1,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native Batch-Balanced KNN graph construction (cycle-14 kernel).

    Mirrors ``sc.external.pp.bbknn`` — parameter names and defaults are
    identical so this function is a drop-in replacement.

    For each cell, *neighbors_within_batch* nearest neighbours are found
    from every other batch independently, and merged into a single kNN
    graph.  Connectivities are computed via the UMAP fuzzy-union metric
    (controlled by *set_op_mix_ratio* and *local_connectivity*).

    Parameters
    ----------
    adata : AnnData
        Must contain a cell embedding in ``adata.obsm[use_rep]``.
    batch_key : str, default ``"batch"``
        Column in ``adata.obs`` that holds the batch label.
    use_rep : str, default ``"X_pca"``
        Key in ``adata.obsm`` for the input embedding.
    neighbors_within_batch : int, default 3
        Number of neighbours to select from each batch.  The total
        number of neighbours per cell is
        ``neighbors_within_batch × (n_batches − 1)`` (excluding own batch).
    n_pcs : int or None, default None
        If set, only the first *n_pcs* dimensions of *use_rep* are used.
        ``None`` → use all dimensions.
    metric : str, default ``"euclidean"``
        Distance metric for kNN search.  Passed directly to the C++
        kNN kernel.  Common values: ``"euclidean"``, ``"cosine"``,
        ``"manhattan"``.
    set_op_mix_ratio : float, default 1.0
        UMAP fuzzy-union / intersection mix.  1.0 = pure union (bbknn
        default); 0.0 = pure intersection.  Matches UMAP ``set_op_mix_ratio``.
    local_connectivity : int, default 1
        Expected number of locally connected non-overlapping nearest
        neighbours per cell.  Matches UMAP ``local_connectivity``.
    copy : bool, default False
        Return a modified copy of *adata*.  When ``False`` (default),
        *adata* is modified in-place and ``None`` is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.bbknn`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If *use_rep* is not in ``adata.obsm`` or *batch_key* is not in
        ``adata.obs``.
    ValueError
        If *neighbors_within_batch* < 1.

    Notes
    -----
    **Result location** (matches scanpy external / bbknn):

    - ``adata.obsp['distances']``      — float32 sparse (cells × cells).
    - ``adata.obsp['connectivities']`` — float32 sparse (cells × cells).
    - ``adata.uns['neighbors']``       — dict with keys ``'params'``,
      ``'distances_key'``, ``'connectivities_key'``.

    **No host copies in the hot path.**  The embedding is transferred to
    device once; the per-batch kNN search, merge, and UMAP weight
    computation all run on device.  Only the sparse output matrices are
    transferred to host for ``obsp`` storage (they are too large to keep
    on device).

    Examples
    --------
    Default BBKNN::

        import singlet_gpu.integrate as sgi
        sgi.bbknn(adata, batch_key="batch")
        print(adata.obsp['distances'].shape)

    Cosine metric, 5 neighbours per batch::

        sgi.bbknn(adata, batch_key="donor", neighbors_within_batch=5,
                  metric="cosine")
    """
    if neighbors_within_batch < 1:
        raise ValueError(
            f"neighbors_within_batch must be ≥1, got {neighbors_within_batch!r}."
        )

    import singlet_gpu._core as _core

    if not hasattr(_core, "bbknn"):
        raise AttributeError(
            "_core.bbknn is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose bbknn before "
            "singlet_gpu.integrate.bbknn() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    emb = _get_embedding(working, use_rep, n_pcs)            # (n_cells, n_dims) f32
    batch_codes = _resolve_batch_codes(working, batch_key)    # (n_cells,) int32
    n_batches = int(batch_codes.max()) + 1

    # C++ kernel: bbknn(embedding, batch_codes, n_batches,
    #                    neighbors_within_batch, metric,
    #                    set_op_mix_ratio, local_connectivity)
    # Returns (distances_sparse, connectivities_sparse) — scipy/cupy sparse.
    result = _core.bbknn(
        emb,
        batch_codes,
        n_batches,
        int(neighbors_within_batch),
        str(metric),
        float(set_op_mix_ratio),
        int(local_connectivity),
    )

    # result.distances and result.connectivities are sparse (cells × cells).
    # Transfer to scipy for obsp storage.
    try:
        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp         # cupy < 14 fallback
        if hasattr(result.distances, "get"):
            distances = result.distances.get()
            connectivities = result.connectivities.get()
        else:
            distances = result.distances
            connectivities = result.connectivities
    except ImportError:
        distances = result.distances
        connectivities = result.connectivities

    working.obsp["distances"] = distances
    working.obsp["connectivities"] = connectivities
    working.uns["neighbors"] = {
        "params": {
            "method": "bbknn",
            "metric": metric,
            "n_neighbors": neighbors_within_batch * (n_batches - 1),
            "neighbors_within_batch": neighbors_within_batch,
            "n_pcs": n_pcs,
            "batch_key": batch_key,
            "use_rep": use_rep,
        },
        "distances_key": "distances",
        "connectivities_key": "connectivities",
    }

    return working if copy else None


__all__ = ["bbknn"]
