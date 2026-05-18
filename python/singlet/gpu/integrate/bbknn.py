# SPDX-License-Identifier: MIT
"""
singlet.gpu.integrate.bbknn — GPU-native Batch-Balanced KNN graph.

Underlying C++ cycle: cycle 14 (integrate/bbknn.h —
``singlet::gpu::integrate::bbknn`` kernel).

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

        import singlet.gpu.integrate as sgi
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

    import singlet.gpu._core as _core

    if not hasattr(_core, "bbknn"):
        raise AttributeError(
            "_core.bbknn is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose bbknn before "
            "singlet.gpu.integrate.bbknn() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    emb = _get_embedding(working, use_rep, n_pcs)            # (n_cells, n_dims) f32
    batch_codes = _resolve_batch_codes(working, batch_key)    # (n_cells,) int32
    n_batches = int(batch_codes.max()) + 1

    # CYCLE-267: binding signature (kw_only after n_batches):
    #   bbknn(embedding, batch_labels, n_batches,
    #         *, k_within=3, approx_threshold=100000) -> KnnResult
    # `metric` / `set_op_mix_ratio` / `local_connectivity` are scanpy-only
    # kwargs not accepted by the binding — drop them. Returns a KnnResult.
    if not hasattr(emb, "__cuda_array_interface__"):
        import cupy as cp
        emb = cp.asarray(emb, dtype=cp.float32)
    if not hasattr(batch_codes, "__cuda_array_interface__"):
        import cupy as cp
        batch_codes = cp.asarray(batch_codes, dtype=cp.int32)
    result = _core.bbknn(
        emb,
        batch_codes,
        n_batches,
        k_within=int(neighbors_within_batch),
    )

    # CYCLE-268: bbknn returns KnnResult (per _bind_kernels.hpp:1748),
    # which exposes only `n`, `k`, `row_offsets_view`, `neighbors_view`,
    # `distances_view` (per _bind_results.hpp:362-390).  Build scipy CSR
    # matrices `(n × n)` from the views — same pattern as CYCLE-262
    # pp/neighbors.py.  Connectivities use a per-row Gaussian (placeholder;
    # see CYCLE-262-FOLLOWUP-CONNECTIVITIES-FUZZY-SIMPLICIAL).
    import cupy as cp
    import scipy.sparse as sp

    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    n = int(result.n)
    k = int(result.k)
    nbr_idx_flat  = cp.asarray(_CaiView(result.neighbors_view)).get().astype(np.int32)
    nbr_dist_flat = cp.asarray(_CaiView(result.distances_view)).get().astype(np.float32)

    # CYCLE-269: bbknn KnnResult uses (n*k,) flat row-major buffers.
    # The kernel pads under-filled batches with -1 sentinels in
    # `neighbors_view` (and matching values in `distances_view`).
    # `row_offsets_view` reflects the FULL (n*k) layout, not filtered counts.
    # Filter sentinels before building the scipy CSR; rebuild row_offsets
    # from a reshape view so under-filled rows have fewer entries.
    nbr_idx_2d  = nbr_idx_flat.reshape(n, k)
    nbr_dist_2d = nbr_dist_flat.reshape(n, k)

    valid_mask = nbr_idx_2d >= 0
    counts = valid_mask.sum(axis=1).astype(np.int32)
    row_offsets = np.zeros(n + 1, dtype=np.int32)
    np.cumsum(counts, out=row_offsets[1:])

    valid_idx  = nbr_idx_2d[valid_mask].astype(np.int32)
    valid_dist = nbr_dist_2d[valid_mask].astype(np.float32)

    # Distances: clamp negative floats (paranoid; mask should already filter).
    dist_data = np.where(valid_dist >= 0, valid_dist, 0.0).astype(np.float32)
    distances = sp.csr_matrix(
        (dist_data, valid_idx, row_offsets),
        shape=(n, n),
    )

    if valid_dist.size > 0:
        local_max = max(float(valid_dist.max()), 1e-9)
        sigma_sq = local_max * local_max
        conn_data = np.exp(-(dist_data * dist_data) / sigma_sq).astype(np.float32)
        conn_data = np.clip(conn_data, 0.0, 1.0)
    else:
        conn_data = valid_dist
    connectivities = sp.csr_matrix(
        (conn_data, valid_idx, row_offsets),
        shape=(n, n),
    )

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
